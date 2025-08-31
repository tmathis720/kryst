use std::sync::Arc;

use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::format::FormatHint;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::{PcSide, Preconditioner};

mod pivot;
mod symbolic;
mod tri_solve;

pub use pivot::PivotStrategy;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IluKind {
    Ilu0,
    Iluk { k: usize },
    Ilut { drop_tol: f64, max_per_row: usize },
}

#[derive(Clone, Debug)]
pub struct IluCsrConfig {
    pub kind: IluKind,
    pub pivot: PivotStrategy,
    pub pivot_threshold: f64,
    pub diag_perturb_factor: f64,
    pub level_sched: bool,
    pub numeric_update_fixed: bool,
    pub logging: usize,
}

impl Default for IluCsrConfig {
    fn default() -> Self {
        Self {
            kind: IluKind::Ilu0,
            pivot: PivotStrategy::DiagonalPerturbation,
            pivot_threshold: 1e-12,
            diag_perturb_factor: 1e-10,
            level_sched: cfg!(feature = "rayon"),
            numeric_update_fixed: true,
            logging: 0,
        }
    }
}

pub struct IluCsr {
    pub(crate) cfg: IluCsrConfig,

    // reuse policy: last operator IDs
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,

    // factors (CSR by rows)
    n: usize,
    // L strictly lower (unit diagonal implied)
    l_row: Vec<usize>,
    l_col: Vec<usize>,
    l_val: Vec<f64>,
    // U upper including diagonal
    u_row: Vec<usize>,
    u_col: Vec<usize>,
    u_val: Vec<f64>,
    u_diag_ix: Vec<usize>,
    // Optional per-entry levels for ILUK
    l_lev: Vec<usize>,
    u_lev: Vec<usize>,

    // optional level scheduling
    levels_fwd: Vec<usize>,
    levels_bwd: Vec<usize>,
    buckets_fwd: Vec<Vec<usize>>,
    buckets_bwd: Vec<Vec<usize>>,

    // scratch for apply
    tmp: Vec<f64>,
}

impl IluCsr {
    pub(crate) fn empty() -> Self {
        Self {
            cfg: IluCsrConfig::default(),
            last_sid: None,
            last_vid: None,
            n: 0,
            l_row: Vec::new(),
            l_col: Vec::new(),
            l_val: Vec::new(),
            u_row: Vec::new(),
            u_col: Vec::new(),
            u_val: Vec::new(),
            u_diag_ix: Vec::new(),
            l_lev: Vec::new(),
            u_lev: Vec::new(),
            levels_fwd: Vec::new(),
            levels_bwd: Vec::new(),
            buckets_fwd: Vec::new(),
            buckets_bwd: Vec::new(),
            tmp: Vec::new(),
        }
    }

    pub fn new_with_config(cfg: IluCsrConfig) -> Self {
        let mut me = Self::empty();
        me.cfg = cfg;
        me
    }

    fn clear_levels(&mut self) {
        self.levels_fwd.clear();
        self.levels_bwd.clear();
        self.buckets_fwd.clear();
        self.buckets_bwd.clear();
    }

    fn build_levels_if_enabled(&mut self) {
        if !self.cfg.level_sched {
            self.clear_levels();
            return;
        }
        // Forward levels from L dependency graph (i <- j if L(i,j) != 0)
        let n = self.n;
        self.levels_fwd.resize(n, 0);
        for i in 0..n {
            let mut lv = 0usize;
            let rs = self.l_row[i];
            let re = self.l_row[i + 1];
            for p in rs..re {
                let j = self.l_col[p];
                lv = lv.max(self.levels_fwd[j] + 1);
            }
            self.levels_fwd[i] = lv;
        }
        let max_lv_fwd = self.levels_fwd.iter().copied().max().unwrap_or(0);
        self.buckets_fwd.clear();
        self.buckets_fwd.resize(max_lv_fwd + 1, Vec::new());
        for i in 0..n {
            let lv = self.levels_fwd[i];
            self.buckets_fwd[lv].push(i);
        }

        // Backward levels from U dependency graph (i <- j if U(i,j) != 0 and j>i)
        self.levels_bwd.resize(n, 0);
        for i in (0..n).rev() {
            let mut lv = 0usize;
            let rs = self.u_row[i];
            let re = self.u_row[i + 1];
            for p in rs..re {
                let j = self.u_col[p];
                if j > i {
                    lv = lv.max(self.levels_bwd[j] + 1);
                }
            }
            self.levels_bwd[i] = lv;
        }
        let max_lv_bwd = self.levels_bwd.iter().copied().max().unwrap_or(0);
        self.buckets_bwd.clear();
        self.buckets_bwd.resize(max_lv_bwd + 1, Vec::new());
        // For backward we want to visit decreasing rows within each bucket for numerical dependencies.
        for i in (0..n).rev() {
            let lv = self.levels_bwd[i];
            self.buckets_bwd[lv].push(i);
        }
    }

    fn factor_symbolic_and_numeric(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        match self.cfg.kind {
            IluKind::Ilu0 => self.factor_ilu0(a),
            IluKind::Iluk { k } => self.factor_iluk(a, k),
            IluKind::Ilut { drop_tol, max_per_row } => self.factor_ilut(a, drop_tol, max_per_row),
        }
    }

    fn factor_numeric_only(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        match self.cfg.kind {
            IluKind::Ilu0 => self.factor_ilu0_numeric_only(a),
            IluKind::Iluk { k } => self.factor_iluk_numeric_only(a, k),
            IluKind::Ilut { .. } => self.factor_ilut_numeric_only(a),
        }
    }

    // === ILU(0) implementation over CSR ===
    fn factor_ilu0(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(KError::InvalidInput("ILU requires square matrix".into()));
        }
        self.n = n;

        // Build L/U symbolic pattern by splitting A and ensuring a diagonal slot exists in U.
        self.l_row.clear();
        self.l_col.clear();
        self.u_row.clear();
        self.u_col.clear();
        self.u_diag_ix.clear();
        self.l_row.resize(n + 1, 0);
        self.u_row.resize(n + 1, 0);
        self.u_diag_ix.resize(n, 0);

        let rp = a.row_ptr();
        let cj = a.col_idx();

        // First pass: collect columns per row, split at diagonal, and sort.
        let mut lcols_row: Vec<usize> = Vec::new();
        let mut ucols_row: Vec<usize> = Vec::new();
        for i in 0..n {
            lcols_row.clear();
            ucols_row.clear();
            let mut have_diag = false;
            for p in rp[i]..rp[i + 1] {
                let j = cj[p];
                if j < i {
                    lcols_row.push(j);
                } else if j == i {
                    ucols_row.push(j);
                    have_diag = true;
                } else {
                    ucols_row.push(j);
                }
            }
            if !have_diag {
                ucols_row.push(i);
            }
            lcols_row.sort_unstable();
            ucols_row.sort_unstable();

            // Append to global structures and record diag index
            self.l_row[i + 1] = self.l_row[i] + lcols_row.len();
            self.u_row[i + 1] = self.u_row[i] + ucols_row.len();
            self.l_col.extend_from_slice(&lcols_row);
            let u_start = self.u_col.len();
            self.u_col.extend_from_slice(&ucols_row);
            // Find diag position in this appended segment
            let d_rel = ucols_row
                .iter()
                .position(|&c| c == i)
                .expect("diagonal present");
            self.u_diag_ix[i] = u_start + d_rel;
        }

        // Allocate values
        self.l_val.clear();
        self.u_val.clear();
        self.l_lev.clear();
        self.u_lev.clear();
        self.l_val.resize(self.l_col.len(), 0.0);
        self.u_val.resize(self.u_col.len(), 0.0);
        // not used in ILU0
        self.l_lev.resize(self.l_col.len(), 0);
        self.u_lev.resize(self.u_col.len(), 0);

        // Numeric factorization using work row over A's pattern only (no fill added).
        self.ilu0_numeric(a)
    }

    fn factor_ilu0_numeric_only(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        if self.n == 0 {
            return self.factor_ilu0(a);
        }
        if self.n != a.nrows() || a.nrows() != a.ncols() {
            return Err(KError::InvalidInput("ILU0 numeric update: size/shape mismatch".into()));
        }
        // Keep pattern intact; just recompute numeric values.
        self.ilu0_numeric(a)
    }

    fn ilu0_numeric(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        use symbolic::RowWork;
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();

        let mut w = RowWork { mark: Vec::new(), idx: Vec::new(), val: Vec::new() };
        symbolic::ensure_rowwork(&mut w, n);

        // Precompute max |A_ii| for pivot handling
        let mut max_diag_abs = 0.0f64;
        for i in 0..n {
            let mut di = 0.0;
            for p in rp[i]..rp[i + 1] {
                if cj[p] == i { di = vv[p]; break; }
            }
            max_diag_abs = max_diag_abs.max(di.abs());
        }

        for i in 0..n {
            // Load row i of A into work row
            symbolic::ensure_rowwork(&mut w, n);
            for p in rp[i]..rp[i + 1] {
                let j = cj[p];
                let pos = symbolic::find_or_insert(&mut w, j);
                w.val[pos] = vv[p];
            }

            // Eliminate against previously computed rows, restricted to L pattern of row i
            let ls = self.l_row[i];
            let le = self.l_row[i + 1];
            for pos in ls..le {
                let j = self.l_col[pos];
                // work value at column j (0 if not present)
                let lij_num = if w.mark[j] >= 0 { w.val[w.mark[j] as usize] } else { 0.0 };
                if lij_num == 0.0 {
                    self.l_val[pos] = 0.0;
                    continue;
                }
                // divide by U(j,j)
                let djj = self.u_val[self.u_diag_ix[j]];
                if djj == 0.0 {
                    // row j diagonal not yet set means j<i and we should have a pivot; if 0.0, treat as error
                    return Err(KError::FactorError(format!("zero U(j,j) encountered at row {}", j)));
                }
                let lij = lij_num / djj;
                self.l_val[pos] = lij;

                // AXPY into k>j strictly following ILU0 (only update if k exists in current row pattern)
                let urs = self.u_row[j];
                let ure = self.u_row[j + 1];
                for q in urs..ure {
                    let kcol = self.u_col[q];
                    if kcol <= j { continue; }
                    let mk = w.mark.get(kcol).copied().unwrap_or(-1);
                    if mk >= 0 {
                        let idx = mk as usize;
                        w.val[idx] -= lij * self.u_val[q];
                    }
                }
            }

            // Finalize U row: copy from work for pattern k >= i (zeros if absent)
            let us = self.u_row[i];
            let ue = self.u_row[i + 1];
            let mut diag_val = 0.0;
            for q in us..ue {
                let k = self.u_col[q];
                let v = if w.mark.get(k).copied().unwrap_or(-1) >= 0 {
                    w.val[w.mark[k] as usize]
                } else { 0.0 };
                if k == i { diag_val = v; }
                self.u_val[q] = v;
            }

            // Handle pivot on U(i,i)
            let fixed = pivot::handle_pivot(
                diag_val,
                self.cfg.pivot,
                self.cfg.pivot_threshold,
                self.cfg.diag_perturb_factor,
                max_diag_abs,
            ).map_err(|_| KError::ZeroPivot(i))?;

            let dix = self.u_diag_ix[i];
            self.u_val[dix] = fixed;

            // Update row pointers already set during symbolic; clear work row
            symbolic::clear_rowwork(&mut w);
        }

        Ok(())
    }

    // === ILUK(k) implementation ===
    fn factor_iluk(&mut self, a: &CsrMatrix<f64>, k_limit: usize) -> Result<(), KError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(KError::InvalidInput("ILUK requires square matrix".into()));
        }
        self.n = n;

        // Initialize CSR row pointers to 0; we’ll build per-row then append.
        self.l_row.clear(); self.u_row.clear();
        self.l_col.clear(); self.u_col.clear();
        self.l_val.clear(); self.u_val.clear();
        self.l_lev.clear(); self.u_lev.clear();
        self.u_diag_ix.clear();
        self.l_row.resize(n + 1, 0);
        self.u_row.resize(n + 1, 0);
        self.u_diag_ix.resize(n, 0);

        use symbolic::RowWork;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut w = RowWork { mark: Vec::new(), idx: Vec::new(), val: Vec::new() };
        let mut wlev: Vec<usize> = Vec::new();
        symbolic::ensure_rowwork(&mut w, n);

        // Precompute max |A_ii| for pivot handling
        let mut max_diag_abs = 0.0f64;
        for i in 0..n {
            let mut di = 0.0;
            for p in rp[i]..rp[i + 1] { if cj[p] == i { di = vv[p]; break; } }
            max_diag_abs = max_diag_abs.max(di.abs());
        }

        for i in 0..n {
            // Load A row with level 0
            symbolic::ensure_rowwork(&mut w, n);
            wlev.clear();
            for p in rp[i]..rp[i + 1] {
                let j = cj[p];
                let pos = symbolic::find_or_insert(&mut w, j);
                if pos == wlev.len() { wlev.push(0); } else { wlev[pos] = 0; }
                w.val[pos] = vv[p];
            }

            // Create sorted list of lower columns present
            let mut lowers: Vec<(usize, usize)> = w.idx.iter().enumerate()
                .filter_map(|(pos, &col)| if col < i { Some((col, pos)) } else { None })
                .collect();
            lowers.sort_by_key(|x| x.0);

            // Eliminate against j < i that are kept (level <= k)
            for &(j, pos) in &lowers {
                let lij_level = wlev[pos];
                // If level exceeds k, skip elimination for this j
                if lij_level > k_limit { continue; }
                let wij = w.val[pos];
                if wij == 0.0 { continue; }
                let djj = {
                    let dix = self.u_diag_ix.get(j).copied().unwrap_or(0);
                    if j < i && self.u_val.get(dix).copied().unwrap_or(0.0) == 0.0 {
                        // Not yet built; for row 0 there is none — but we will handle when j<i holds
                    }
                    if j < i { self.u_val[self.u_diag_ix[j]] } else { 1.0 }
                };
                let lij = wij / djj;

                // AXPY to k > j using U(j,*)
                let urs = self.u_row.get(j).copied().unwrap_or(0);
                let ure = self.u_row.get(j + 1).copied().unwrap_or(0);
                for q in urs..ure {
                    let kcol = self.u_col[q];
                    if kcol <= j { continue; }
                    let new_level = lij_level + self.u_lev[q] + 1;
                    if new_level > k_limit { continue; }
                    let kpos = symbolic::find_or_insert(&mut w, kcol);
                    if kpos == wlev.len() { wlev.push(new_level); } else {
                        if new_level < wlev[kpos] { wlev[kpos] = new_level; }
                    }
                    w.val[kpos] -= lij * self.u_val[q];
                }
                // store L(i,j) entry (value+level) later when we finalize L row
            }

            // Finalize L and U rows from work row with level <= k
            // Gather L (j<i)
            let mut l_pairs: Vec<(usize, f64, usize)> = w.idx.iter().enumerate()
                .filter_map(|(pos, &col)| if col < i && wlev[pos] <= k_limit {
                    Some((col, w.val[pos], wlev[pos]))
                } else { None }).collect();
            l_pairs.sort_by_key(|x| x.0);

            // Gather U (k>=i); ensure diagonal exists with some level (0)
            let mut u_pairs: Vec<(usize, f64, usize)> = w.idx.iter().enumerate()
                .filter_map(|(pos, &col)| if col >= i && wlev[pos] <= k_limit {
                    Some((col, w.val[pos], wlev[pos]))
                } else { None }).collect();
            if !u_pairs.iter().any(|(c, _, _)| *c == i) {
                u_pairs.push((i, 0.0, 0));
            }
            u_pairs.sort_by_key(|x| x.0);

            // Write L row
            self.l_row[i + 1] = self.l_row[i] + l_pairs.len();
            for (c, v, lev) in l_pairs { self.l_col.push(c); self.l_val.push(v); self.l_lev.push(lev); }

            // Write U row and remember diag ix; pivot later after elimination loop
            let u_start = self.u_col.len();
            self.u_row[i + 1] = self.u_row[i] + u_pairs.len();
            for (c, v, lev) in &u_pairs { self.u_col.push(*c); self.u_val.push(*v); self.u_lev.push(*lev); }
            let d_rel = u_pairs.iter().position(|(c, _, _)| *c == i).unwrap();
            self.u_diag_ix[i] = u_start + d_rel;

            // Clear work row
            symbolic::clear_rowwork(&mut w);
        }

        // Numeric refinement: run numeric-only to enforce pivot strategy and compute final values.
        self.iluk_numeric_only(a, k_limit, max_diag_abs)
    }

    fn iluk_numeric_only(&mut self, a: &CsrMatrix<f64>, _k_limit: usize, max_diag_abs: f64) -> Result<(), KError> {
        use symbolic::RowWork;
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut w = RowWork { mark: Vec::new(), idx: Vec::new(), val: Vec::new() };
        symbolic::ensure_rowwork(&mut w, n);

        for i in 0..n {
            // load A row into work
            symbolic::ensure_rowwork(&mut w, n);
            for p in rp[i]..rp[i + 1] {
                let j = cj[p];
                let pos = symbolic::find_or_insert(&mut w, j);
                w.val[pos] = vv[p];
            }

            // eliminate for j in L pattern (already filtered by <=k)
            let ls = self.l_row[i];
            let le = self.l_row[i + 1];
            for pos in ls..le {
                let j = self.l_col[pos];
                let wij = if w.mark[j] >= 0 { w.val[w.mark[j] as usize] } else { 0.0 };
                let djj = self.u_val[self.u_diag_ix[j]];
                let lij = if djj != 0.0 { wij / djj } else { 0.0 };
                self.l_val[pos] = lij;
                // AXPY into k>j but only if k exists in this row's U pattern
                let urs = self.u_row[j];
                let ure = self.u_row[j + 1];
                for q in urs..ure {
                    let kcol = self.u_col[q]; if kcol <= j { continue; }
                    let mk = w.mark.get(kcol).copied().unwrap_or(-1);
                    if mk >= 0 { w.val[mk as usize] -= lij * self.u_val[q]; }
                }
            }

            // finalize U row values from work restricted to U pattern
            let us = self.u_row[i];
            let ue = self.u_row[i + 1];
            let mut diag = 0.0;
            for q in us..ue {
                let k = self.u_col[q];
                let v = if w.mark.get(k).copied().unwrap_or(-1) >= 0 { w.val[w.mark[k] as usize] } else { 0.0 };
                if k == i { diag = v; }
                self.u_val[q] = v;
            }
            // pivot
            let fixed = pivot::handle_pivot(
                diag,
                self.cfg.pivot,
                self.cfg.pivot_threshold,
                self.cfg.diag_perturb_factor,
                max_diag_abs,
            ).map_err(|_| KError::ZeroPivot(i))?;
            let dix = self.u_diag_ix[i];
            self.u_val[dix] = fixed;

            symbolic::clear_rowwork(&mut w);
        }
        Ok(())
    }

    fn factor_iluk_numeric_only(&mut self, a: &CsrMatrix<f64>, k_limit: usize) -> Result<(), KError> {
        // Recompute max diag
        let mut max_diag_abs = 0.0f64;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        for i in 0..self.n {
            let mut di = 0.0;
            for p in rp[i]..rp[i + 1] { if cj[p] == i { di = vv[p]; break; } }
            max_diag_abs = max_diag_abs.max(di.abs());
        }
        self.iluk_numeric_only(a, k_limit, max_diag_abs)
    }

    // === ILUT(drop_tol, max_per_row) implementation ===
    fn factor_ilut(&mut self, a: &CsrMatrix<f64>, drop_tol: f64, max_per_row: usize) -> Result<(), KError> {
        let n = a.nrows();
        if n != a.ncols() { return Err(KError::InvalidInput("ILUT requires square matrix".into())); }
        self.n = n;

        self.l_row.clear(); self.l_col.clear(); self.l_val.clear();
        self.u_row.clear(); self.u_col.clear(); self.u_val.clear();
        self.u_diag_ix.clear();
        self.l_lev.clear(); self.u_lev.clear(); // not used by ILUT
        self.l_row.resize(n + 1, 0);
        self.u_row.resize(n + 1, 0);
        self.u_diag_ix.resize(n, 0);

        use symbolic::RowWork;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut w = RowWork { mark: Vec::new(), idx: Vec::new(), val: Vec::new() };
        symbolic::ensure_rowwork(&mut w, n);

        // Precompute max |A_ii| for pivot handling
        let mut max_diag_abs = 0.0f64;
        for i in 0..n {
            let mut di = 0.0; for p in rp[i]..rp[i + 1] { if cj[p] == i { di = vv[p]; break; } }
            max_diag_abs = max_diag_abs.max(di.abs());
        }

        for i in 0..n {
            symbolic::ensure_rowwork(&mut w, n);
            for p in rp[i]..rp[i + 1] { let j = cj[p]; let pos = symbolic::find_or_insert(&mut w, j); w.val[pos] = vv[p]; }

            // lower candidates sorted by column
            let mut lowers: Vec<usize> = w.idx.iter().copied().filter(|&c| c < i).collect();
            lowers.sort_unstable();
            for &j in &lowers {
                let pos = w.mark[j] as usize;
                let wij = w.val[pos];
                if wij.abs() < drop_tol { continue; }
                let djj = self.u_val.get(self.u_diag_ix.get(j).copied().unwrap_or(0)).copied().unwrap_or(1.0);
                let lij = wij / djj;
                // update work for k>j, but skip tiny values
                let rs = self.u_row.get(j).copied().unwrap_or(0);
                let re = self.u_row.get(j + 1).copied().unwrap_or(0);
                for q in rs..re {
                    let kcol = self.u_col[q]; if kcol <= j { continue; }
                    let kpos = symbolic::find_or_insert(&mut w, kcol);
                    let newv = w.val[kpos] - lij * self.u_val[q];
                    // Drop small ones early to control workspace
                    if newv.abs() < drop_tol { w.val[kpos] = 0.0; } else { w.val[kpos] = newv; }
                }
            }

            // Partition and apply per-row cap on off-diagonals
            // L part (j<i, off-diagonals only) with threshold
            let mut l_keep: Vec<(usize, f64)> = w.idx.iter().filter_map(|&c| if c < i {
                let v = w.val[w.mark[c] as usize]; if v.abs() >= drop_tol { Some((c, v)) } else { None }
            } else { None }).collect();
            if max_per_row > 0 && l_keep.len() > max_per_row {
                let m = max_per_row; l_keep.select_nth_unstable_by(m, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap()); l_keep.truncate(m);
            }
            l_keep.sort_by_key(|x| x.0);

            // U part (k>=i); ensure diagonal always present
            let mut u_keep: Vec<(usize, f64)> = w.idx.iter().filter_map(|&c| if c >= i {
                let v = w.val[w.mark[c] as usize]; if c == i || v.abs() >= drop_tol { Some((c, v)) } else { None }
            } else { None }).collect();
            if !u_keep.iter().any(|(c, _)| *c == i) { u_keep.push((i, 0.0)); }
            // Apply cap to off-diagonals only
            if max_per_row > 0 {
                let mut offs: Vec<(usize, f64)> = u_keep.iter().cloned().filter(|(c, _)| *c > i).collect();
                if offs.len() > max_per_row {
                    let m = max_per_row; offs.select_nth_unstable_by(m, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap()); offs.truncate(m);
                }
                offs.sort_by_key(|x| x.0);
                // rebuild u_keep with diag + offs
                let mut new_u: Vec<(usize, f64)> = Vec::with_capacity(1 + offs.len());
                let diagv = u_keep.iter().find(|(c, _)| *c == i).map(|(_, v)| *v).unwrap_or(0.0);
                new_u.push((i, diagv));
                new_u.extend(offs.into_iter());
                u_keep = new_u;
            }
            u_keep.sort_by_key(|x| x.0);

            // Store L row
            self.l_row[i + 1] = self.l_row[i] + l_keep.len();
            for (c, v) in l_keep { self.l_col.push(c); self.l_val.push(v); }
            // Store U row
            let u_start = self.u_col.len();
            self.u_row[i + 1] = self.u_row[i] + u_keep.len();
            for (c, v) in &u_keep { self.u_col.push(*c); self.u_val.push(*v); }
            let d_rel = u_keep.iter().position(|(c, _)| *c == i).unwrap();
            self.u_diag_ix[i] = u_start + d_rel;

            // Clear work row
            symbolic::clear_rowwork(&mut w);
        }

        // Numeric refine on fixed pattern
        self.ilut_numeric_only(a, max_diag_abs)
    }

    fn ilut_numeric_only(&mut self, a: &CsrMatrix<f64>, max_diag_abs: f64) -> Result<(), KError> {
        // Re-run elimination using fixed L/U patterns (no drop/cap)
        use symbolic::RowWork;
        let n = self.n;
        let rp = a.row_ptr(); let cj = a.col_idx(); let vv = a.values();
        let mut w = RowWork { mark: Vec::new(), idx: Vec::new(), val: Vec::new() };
        symbolic::ensure_rowwork(&mut w, n);
        for i in 0..n {
            symbolic::ensure_rowwork(&mut w, n);
            for p in rp[i]..rp[i + 1] { let j = cj[p]; let pos = symbolic::find_or_insert(&mut w, j); w.val[pos] = vv[p]; }
            // eliminate across L pattern
            let ls = self.l_row[i]; let le = self.l_row[i + 1];
            for pos in ls..le {
                let j = self.l_col[pos];
                let wij = if w.mark[j] >= 0 { w.val[w.mark[j] as usize] } else { 0.0 };
                let djj = self.u_val[self.u_diag_ix[j]];
                let lij = if djj != 0.0 { wij / djj } else { 0.0 };
                self.l_val[pos] = lij;
                let urs = self.u_row[j]; let ure = self.u_row[j + 1];
                for q in urs..ure { let kcol = self.u_col[q]; if kcol <= j { continue; }
                    let mk = w.mark.get(kcol).copied().unwrap_or(-1); if mk >= 0 { w.val[mk as usize] -= lij * self.u_val[q]; }
                }
            }
            // finalize U row
            let us = self.u_row[i]; let ue = self.u_row[i + 1]; let mut diag = 0.0;
            for q in us..ue { let k = self.u_col[q]; let v = if w.mark.get(k).copied().unwrap_or(-1) >= 0 { w.val[w.mark[k] as usize] } else { 0.0 };
                if k == i { diag = v; } self.u_val[q] = v; }
            let fixed = pivot::handle_pivot(diag, self.cfg.pivot, self.cfg.pivot_threshold, self.cfg.diag_perturb_factor, max_diag_abs).map_err(|_| KError::ZeroPivot(i))?;
            self.u_val[self.u_diag_ix[i]] = fixed;
            symbolic::clear_rowwork(&mut w);
        }
        Ok(())
    }

    fn factor_ilut_numeric_only(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        // recompute max_diag_abs
        let rp = a.row_ptr(); let cj = a.col_idx(); let vv = a.values();
        let mut max_diag_abs = 0.0f64; for i in 0..self.n { let mut di=0.0; for p in rp[i]..rp[i+1] { if cj[p]==i { di = vv[p]; break; } } max_diag_abs = max_diag_abs.max(di.abs()); }
        self.ilut_numeric_only(a, max_diag_abs)
    }
}

impl Preconditioner for IluCsr {
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let drop = 0.0; // use full numerical content by default
        let a: Arc<CsrMatrix<f64>> = csr_from_linop(op, drop)?;
        let sid = op.structure_id();
        let vid = op.values_id();

        let structure_changed = self.last_sid != Some(sid);
        let values_changed = self.last_vid != Some(vid);

        if structure_changed || !self.cfg.numeric_update_fixed {
            self.factor_symbolic_and_numeric(&a)?;
            self.build_levels_if_enabled();
            self.last_sid = Some(sid);
            self.last_vid = Some(vid);
            // scratch
            self.tmp.resize(a.nrows(), 0.0);
            Ok(())
        } else if values_changed {
            self.factor_numeric_only(&a)?;
            self.last_vid = Some(vid);
            Ok(())
        } else {
            Ok(())
        }
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "IluCsr::apply dimension mismatch: n={}, x.len()={}, y.len()={}",
                self.n, x.len(), y.len()
            )));
        }
        if self.cfg.level_sched {
            tri_solve::tri_solve_level_scheduled(self, x, y)
        } else {
            tri_solve::tri_solve_serial(self, x, y)
        }
    }

    fn apply_mut(&mut self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        self.apply(side, x, y)
    }

    fn supports_numeric_update(&self) -> bool {
        self.cfg.numeric_update_fixed
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        if !self.cfg.numeric_update_fixed {
            return Err(KError::Unsupported("numeric update requires fixed pattern"));
        }
        if Some(op.structure_id()) != self.last_sid {
            return Err(KError::Unsupported("pattern changed; call update_symbolic"));
        }
        let a = csr_from_linop(op, 0.0)?;
        self.factor_numeric_only(&a)?;
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let a = csr_from_linop(op, 0.0)?;
        self.factor_symbolic_and_numeric(&a)?;
        self.build_levels_if_enabled();
        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn required_format(&self) -> FormatHint {
        FormatHint::Csr
    }
}

// === Simple accessors for internal solves ===
impl IluCsr {
    #[inline]
    pub(crate) fn n(&self) -> usize { self.n }
    #[inline]
    pub(crate) fn l_row(&self) -> &[usize] { &self.l_row }
    #[inline]
    pub(crate) fn l_col(&self) -> &[usize] { &self.l_col }
    #[inline]
    pub(crate) fn l_val(&self) -> &[f64] { &self.l_val }
    #[inline]
    pub(crate) fn u_row(&self) -> &[usize] { &self.u_row }
    #[inline]
    pub(crate) fn u_col(&self) -> &[usize] { &self.u_col }
    #[inline]
    pub(crate) fn u_val(&self) -> &[f64] { &self.u_val }
    #[inline]
    pub(crate) fn u_diag_ix(&self) -> &[usize] { &self.u_diag_ix }
    #[inline]
    pub(crate) fn tmp(&self) -> &[f64] { &self.tmp }
    #[inline]
    pub(crate) fn tmp_mut(&mut self) -> &mut [f64] { &mut self.tmp }

    #[inline]
    pub(crate) fn buckets_fwd(&self) -> &[Vec<usize>] { &self.buckets_fwd }
    #[inline]
    pub(crate) fn buckets_bwd(&self) -> &[Vec<usize>] { &self.buckets_bwd }
}
