use std::sync::Arc;

use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::format::FormatHint;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::{Op, PcCaps, PcSide, Preconditioner};
use once_cell::sync::OnceCell;

mod pivot;
mod tri_solve;
mod ilut_params;
mod csr_builder;
mod row_work;

pub use pivot::PivotStrategy;
pub use ilut_params::{IlutParams, PivotPolicy, Pivoting};

use csr_builder::CsrBuilder;
use row_work::RowWork;

// Workspace for fast lookups of U(i, j) positions within a row during
// numeric factorization. Uses the marker/epoch trick to provide O(1)
// amortized access without clearing the entire array each iteration.
#[derive(Clone, Debug)]
struct URowMap {
    epoch: usize,
    mark: Vec<usize>,
    pos: Vec<usize>,
}

impl URowMap {
    fn new() -> Self {
        Self {
            epoch: 0,
            mark: Vec::new(),
            pos: Vec::new(),
        }
    }

    fn ensure_size(&mut self, n: usize) {
        if self.mark.len() < n {
            self.mark.resize(n, 0);
            self.pos.resize(n, 0);
        }
    }

    fn prime(&mut self, u_row: &[usize], u_col: &[usize], i: usize) {
        self.epoch = self.epoch.wrapping_add(1);
        let rs = u_row[i];
        let re = u_row[i + 1];
        for (offset, &col) in u_col[rs..re].iter().enumerate() {
            self.mark[col] = self.epoch;
            self.pos[col] = rs + offset;
        }
    }

    #[inline]
    fn get(&self, j: usize) -> Option<usize> {
        if self.mark.get(j).copied().unwrap_or(0) == self.epoch {
            Some(self.pos[j])
        } else {
            None
        }
    }
}

mod symbolic;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IluKind {
    Ilu0,
    Milu0,
    Iluk { k: usize },
    Ilut { params: IlutParams },
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

    // cached transposes, built lazily
    lt: OnceCell<(Vec<usize>, Vec<usize>, Vec<f64>)>,
    ut: OnceCell<(Vec<usize>, Vec<usize>, Vec<f64>)>,

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
            lt: OnceCell::new(),
            ut: OnceCell::new(),
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
            IluKind::Ilu0 | IluKind::Milu0 => self.factor_ilu0(a),
            IluKind::Iluk { k } => self.factor_iluk(a, k),
            IluKind::Ilut { params } => self.factor_ilut(a, &params),
        }
    }

    fn factor_numeric_only(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        match self.cfg.kind {
            IluKind::Ilu0 | IluKind::Milu0 => self.factor_ilu0_numeric_only(a),
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
        let milu = matches!(self.cfg.kind, IluKind::Milu0);
        self.ilu0_numeric(a, milu)
    }

    fn factor_ilu0_numeric_only(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        if self.n == 0 {
            return self.factor_ilu0(a);
        }
        if self.n != a.nrows() || a.nrows() != a.ncols() {
            return Err(KError::InvalidInput(
                "ILU0 numeric update: size/shape mismatch".into(),
            ));
        }
        // Keep pattern intact; just recompute numeric values.
        let milu = matches!(self.cfg.kind, IluKind::Milu0);
        self.ilu0_numeric(a, milu)
    }

    fn ilu0_numeric(&mut self, a: &CsrMatrix<f64>, milu: bool) -> Result<(), KError> {
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();

        // Workspace for locating U(i, j) quickly
        let mut map = URowMap::new();
        map.ensure_size(n);

        // Precompute max |A_ii| for pivot handling
        let mut max_diag_abs = 0.0f64;
        for i in 0..n {
            let mut di = 0.0;
            for p in rp[i]..rp[i + 1] {
                if cj[p] == i {
                    di = vv[p];
                    break;
                }
            }
            max_diag_abs = max_diag_abs.max(di.abs());
        }

        for i in 0..n {
            map.prime(&self.u_row, &self.u_col, i);

            // Initialize L and U values from A for this row
            let mut p = rp[i];
            while p < rp[i + 1] {
                let j = cj[p];
                let val = vv[p];
                if j < i {
                    // L part
                    let ls = self.l_row[i];
                    if let Ok(off) = self.l_col[ls..self.l_row[i + 1]].binary_search(&j) {
                        self.l_val[ls + off] = val;
                    }
                } else {
                    // U part (including diagonal)
                    if let Some(pos) = map.get(j) {
                        self.u_val[pos] = val;
                    }
                }
                p += 1;
            }

            // Eliminate using previous rows
            let ls = self.l_row[i];
            let le = self.l_row[i + 1];
            for pos in ls..le {
                let k = self.l_col[pos];
                let ukk = self.u_val[self.u_diag_ix[k]];
                if ukk == 0.0 {
                    return Err(KError::FactorError(format!(
                        "zero U(j,j) encountered at row {}",
                        k
                    )));
                }
                let mult = self.l_val[pos] / ukk;
                self.l_val[pos] = mult;

                // Update U(i, j)
                let urs = self.u_row[k];
                let ure = self.u_row[k + 1];
                for q in urs..ure {
                    let j = self.u_col[q];
                    if j <= k {
                        continue;
                    }
                    if let Some(pos_ij) = map.get(j) {
                        self.u_val[pos_ij] -= mult * self.u_val[q];
                    } else if milu {
                        let di_pos = self.u_diag_ix[i];
                        self.u_val[di_pos] -= mult * self.u_val[q];
                    }
                }
            }

            // Handle pivot on U(i,i)
            let di_pos = self.u_diag_ix[i];
            let fixed = pivot::handle_pivot(
                self.u_val[di_pos],
                self.cfg.pivot,
                self.cfg.pivot_threshold,
                self.cfg.diag_perturb_factor,
                max_diag_abs,
            )
            .map_err(|_| KError::ZeroPivot(i))?;
            self.u_val[di_pos] = fixed;
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
        self.l_row.clear();
        self.u_row.clear();
        self.l_col.clear();
        self.u_col.clear();
        self.l_val.clear();
        self.u_val.clear();
        self.l_lev.clear();
        self.u_lev.clear();
        self.u_diag_ix.clear();
        self.l_row.resize(n + 1, 0);
        self.u_row.resize(n + 1, 0);
        self.u_diag_ix.resize(n, 0);

        use symbolic::RowWork;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut w = RowWork {
            mark: Vec::new(),
            idx: Vec::new(),
            val: Vec::new(),
        };
        let mut wlev: Vec<usize> = Vec::new();
        symbolic::ensure_rowwork(&mut w, n);

        // Precompute max |A_ii| for pivot handling
        let mut max_diag_abs = 0.0f64;
        for i in 0..n {
            let mut di = 0.0;
            for p in rp[i]..rp[i + 1] {
                if cj[p] == i {
                    di = vv[p];
                    break;
                }
            }
            max_diag_abs = max_diag_abs.max(di.abs());
        }

        for i in 0..n {
            // Load A row with level 0
            symbolic::ensure_rowwork(&mut w, n);
            wlev.clear();
            for p in rp[i]..rp[i + 1] {
                let j = cj[p];
                let pos = symbolic::find_or_insert(&mut w, j);
                if pos == wlev.len() {
                    wlev.push(0);
                } else {
                    wlev[pos] = 0;
                }
                w.val[pos] = vv[p];
            }

            // Create sorted list of lower columns present
            let mut lowers: Vec<(usize, usize)> = w
                .idx
                .iter()
                .enumerate()
                .filter_map(|(pos, &col)| if col < i { Some((col, pos)) } else { None })
                .collect();
            lowers.sort_by_key(|x| x.0);

            // Eliminate against j < i that are kept (level <= k)
            for &(j, pos) in &lowers {
                let lij_level = wlev[pos];
                // If level exceeds k, skip elimination for this j
                if lij_level > k_limit {
                    continue;
                }
                let wij = w.val[pos];
                if wij == 0.0 {
                    continue;
                }
                let djj = {
                    let dix = self.u_diag_ix.get(j).copied().unwrap_or(0);
                    if j < i && self.u_val.get(dix).copied().unwrap_or(0.0) == 0.0 {
                        // Not yet built; for row 0 there is none — but we will handle when j<i holds
                    }
                    if j < i {
                        self.u_val[self.u_diag_ix[j]]
                    } else {
                        1.0
                    }
                };
                let lij = wij / djj;

                // AXPY to k > j using U(j,*)
                let urs = self.u_row.get(j).copied().unwrap_or(0);
                let ure = self.u_row.get(j + 1).copied().unwrap_or(0);
                for q in urs..ure {
                    let kcol = self.u_col[q];
                    if kcol <= j {
                        continue;
                    }
                    let new_level = lij_level + self.u_lev[q] + 1;
                    if new_level > k_limit {
                        continue;
                    }
                    let kpos = symbolic::find_or_insert(&mut w, kcol);
                    if kpos == wlev.len() {
                        wlev.push(new_level);
                    } else {
                        if new_level < wlev[kpos] {
                            wlev[kpos] = new_level;
                        }
                    }
                    w.val[kpos] -= lij * self.u_val[q];
                }
                // store L(i,j) entry (value+level) later when we finalize L row
            }

            // Finalize L and U rows from work row with level <= k
            // Gather L (j<i)
            let mut l_pairs: Vec<(usize, f64, usize)> = w
                .idx
                .iter()
                .enumerate()
                .filter_map(|(pos, &col)| {
                    if col < i && wlev[pos] <= k_limit {
                        Some((col, w.val[pos], wlev[pos]))
                    } else {
                        None
                    }
                })
                .collect();
            l_pairs.sort_by_key(|x| x.0);

            // Gather U (k>=i); ensure diagonal exists with some level (0)
            let mut u_pairs: Vec<(usize, f64, usize)> = w
                .idx
                .iter()
                .enumerate()
                .filter_map(|(pos, &col)| {
                    if col >= i && wlev[pos] <= k_limit {
                        Some((col, w.val[pos], wlev[pos]))
                    } else {
                        None
                    }
                })
                .collect();
            if !u_pairs.iter().any(|(c, _, _)| *c == i) {
                u_pairs.push((i, 0.0, 0));
            }
            u_pairs.sort_by_key(|x| x.0);

            // Write L row
            self.l_row[i + 1] = self.l_row[i] + l_pairs.len();
            for (c, v, lev) in l_pairs {
                self.l_col.push(c);
                self.l_val.push(v);
                self.l_lev.push(lev);
            }

            // Write U row and remember diag ix; pivot later after elimination loop
            let u_start = self.u_col.len();
            self.u_row[i + 1] = self.u_row[i] + u_pairs.len();
            for (c, v, lev) in &u_pairs {
                self.u_col.push(*c);
                self.u_val.push(*v);
                self.u_lev.push(*lev);
            }
            let d_rel = u_pairs.iter().position(|(c, _, _)| *c == i).unwrap();
            self.u_diag_ix[i] = u_start + d_rel;

            // Clear work row
            symbolic::clear_rowwork(&mut w);
        }

        // Numeric refinement: run numeric-only to enforce pivot strategy and compute final values.
        self.iluk_numeric_only(a, k_limit, max_diag_abs)
    }

    fn iluk_numeric_only(
        &mut self,
        a: &CsrMatrix<f64>,
        _k_limit: usize,
        max_diag_abs: f64,
    ) -> Result<(), KError> {
        use symbolic::RowWork;
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut w = RowWork {
            mark: Vec::new(),
            idx: Vec::new(),
            val: Vec::new(),
        };
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
                let wij = if w.mark[j] >= 0 {
                    w.val[w.mark[j] as usize]
                } else {
                    0.0
                };
                let djj = self.u_val[self.u_diag_ix[j]];
                let lij = if djj != 0.0 { wij / djj } else { 0.0 };
                self.l_val[pos] = lij;
                // AXPY into k>j but only if k exists in this row's U pattern
                let urs = self.u_row[j];
                let ure = self.u_row[j + 1];
                for q in urs..ure {
                    let kcol = self.u_col[q];
                    if kcol <= j {
                        continue;
                    }
                    let mk = w.mark.get(kcol).copied().unwrap_or(-1);
                    if mk >= 0 {
                        w.val[mk as usize] -= lij * self.u_val[q];
                    }
                }
            }

            // finalize U row values from work restricted to U pattern
            let us = self.u_row[i];
            let ue = self.u_row[i + 1];
            let mut diag = 0.0;
            for q in us..ue {
                let k = self.u_col[q];
                let v = if w.mark.get(k).copied().unwrap_or(-1) >= 0 {
                    w.val[w.mark[k] as usize]
                } else {
                    0.0
                };
                if k == i {
                    diag = v;
                }
                self.u_val[q] = v;
            }
            // pivot
            let fixed = pivot::handle_pivot(
                diag,
                self.cfg.pivot,
                self.cfg.pivot_threshold,
                self.cfg.diag_perturb_factor,
                max_diag_abs,
            )
            .map_err(|_| KError::ZeroPivot(i))?;
            let dix = self.u_diag_ix[i];
            self.u_val[dix] = fixed;

            symbolic::clear_rowwork(&mut w);
        }
        Ok(())
    }

    fn factor_iluk_numeric_only(
        &mut self,
        a: &CsrMatrix<f64>,
        k_limit: usize,
    ) -> Result<(), KError> {
        // Recompute max diag
        let mut max_diag_abs = 0.0f64;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        for i in 0..self.n {
            let mut di = 0.0;
            for p in rp[i]..rp[i + 1] {
                if cj[p] == i {
                    di = vv[p];
                    break;
                }
            }
            max_diag_abs = max_diag_abs.max(di.abs());
        }
        self.iluk_numeric_only(a, k_limit, max_diag_abs)
    }

    // === ILUT(p, tau) implementation with separate L/U caps ===
    fn factor_ilut(&mut self, a: &CsrMatrix<f64>, params: &IlutParams) -> Result<(), KError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(KError::InvalidInput("ILUT requires square matrix".into()));
        }
        self.n = n;

        // Builders for L and U
        let mut l_build = CsrBuilder::new(n);
        let mut u_build = CsrBuilder::new(n);
        let mut inv_diag_u = vec![0.0f64; n];

        // Row workspace
        let mut w = RowWork::<f64>::new();
        w.ensure_size(n);
        let mut l_tmp: Vec<(usize, f64)> = Vec::new();
        let mut u_tmp: Vec<(usize, f64)> = Vec::new();

        let mut max_diag_abs = 0.0f64;

        for i in 0..n {
            w.clear_row();
            l_tmp.clear();
            u_tmp.clear();

            // Seed w from row i of A
            let (a_cols, a_vals) = a.row(i);
            let mut row_inf: f64 = 0.0;
            for (&j, &v) in a_cols.iter().zip(a_vals.iter()) {
                if v != 0.0 {
                    w.set(j, v);
                    row_inf = row_inf.max(v.abs());
                }
            }
            let tau = params.droptol_abs + params.droptol_rel * row_inf;

            // Eliminate lower entries
            let mut lowers: Vec<usize> = w.iter().filter(|&(j, _)| j < i).map(|(j, _)| j).collect();
            lowers.sort_unstable();
            for &k in &lowers {
                let wk = w.get(k);
                if wk == 0.0 {
                    continue;
                }
                let lik = wk * inv_diag_u[k];
                if params.early_drop && lik.abs() < tau {
                    w.set(k, 0.0);
                    continue;
                }
                l_tmp.push((k, lik));
                w.set(k, 0.0);

                let (u_cols_k, u_vals_k) = u_build.row(k);
                for (&j, &ukj) in u_cols_k.iter().zip(u_vals_k.iter()) {
                    if j <= k {
                        continue;
                    }
                    let newv: f64 = w.get(j) - lik * ukj;
                    if params.early_drop && newv.abs() < tau {
                        w.set(j, 0.0);
                    } else {
                        w.set(j, newv);
                    }
                }
            }

            // Split remaining w into U-part
            for (j, v) in w.iter() {
                if j >= i && (j == i || v.abs() >= tau) {
                    u_tmp.push((j, v));
                }
            }
            if !u_tmp.iter().any(|(j, _)| *j == i) {
                u_tmp.push((i, 0.0));
            }

            // Cap L entries
            if params.p_l > 0 && l_tmp.len() > params.p_l {
                l_tmp.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                l_tmp.truncate(params.p_l);
            }

            // Cap U entries (excluding diagonal)
            let mut diag = 0.0;
            if let Some(pos) = u_tmp.iter().position(|(j, _)| *j == i) {
                diag = u_tmp[pos].1;
                u_tmp.remove(pos);
            }
            if params.p_u > 0 && u_tmp.len() > params.p_u {
                u_tmp.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                u_tmp.truncate(params.p_u);
            }
            u_tmp.push((i, diag));

            // Sort by column for determinism
            if params.reproducible_order {
                l_tmp.sort_by(|a, b| a.0.cmp(&b.0));
                u_tmp.sort_by(|a, b| a.0.cmp(&b.0));
            } else {
                l_tmp.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                u_tmp.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            }

            // Pivot handling
            let diag_pos = u_tmp.iter().position(|(j, _)| *j == i).unwrap();
            let mut uii = u_tmp[diag_pos].1;
            max_diag_abs = max_diag_abs.max(uii.abs());
            match params.pivot {
                PivotPolicy::Strict => {
                    if uii.abs() < params.pivot_tau {
                        return Err(KError::ZeroPivot(i));
                    }
                }
                PivotPolicy::Threshold => {
                    if uii.abs() < params.pivot_tau {
                        uii = uii.signum() * params.pivot_tau;
                    }
                }
                PivotPolicy::DiagonalPerturbation => {
                    if uii.abs() < params.pivot_tau {
                        uii = uii + params.pivot_tau;
                    }
                }
            }
            u_tmp[diag_pos].1 = uii;
            inv_diag_u[i] = 1.0 / uii;

            // Store rows into builders
            for &(k, v) in &l_tmp {
                l_build.push(i, k, v);
            }
            l_build.push(i, i, 1.0);
            for &(j, v) in &u_tmp {
                u_build.push(i, j, v);
            }
        }

        // Finalize builders into CSR arrays
        let (l_row, l_col, l_val) = l_build.finalize_sorted_unique(params.reproducible_order);
        let (u_row, u_col, u_val) = u_build.finalize_sorted_unique(params.reproducible_order);

        self.l_row = l_row;
        self.l_col = l_col;
        self.l_val = l_val;
        self.u_row = u_row;
        self.u_col = u_col;
        self.u_val = u_val;

        self.u_diag_ix.clear();
        self.u_diag_ix.resize(n, 0);
        for i in 0..n {
            let rs = self.u_row[i];
            let re = self.u_row[i + 1];
            if let Some(pos) = self.u_col[rs..re].iter().position(|&c| c == i) {
                self.u_diag_ix[i] = rs + pos;
            } else {
                return Err(KError::InvalidInput("missing diagonal".into()));
            }
        }

        self.tmp.resize(n, 0.0);

        // Optional numeric refine
        self.ilut_numeric_only(a, max_diag_abs)
    }

    fn ilut_numeric_only(&mut self, a: &CsrMatrix<f64>, max_diag_abs: f64) -> Result<(), KError> {
        // Re-run elimination using fixed L/U patterns (no drop/cap)
        use symbolic::RowWork;
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut w = RowWork {
            mark: Vec::new(),
            idx: Vec::new(),
            val: Vec::new(),
        };
        symbolic::ensure_rowwork(&mut w, n);
        for i in 0..n {
            symbolic::ensure_rowwork(&mut w, n);
            for p in rp[i]..rp[i + 1] {
                let j = cj[p];
                let pos = symbolic::find_or_insert(&mut w, j);
                w.val[pos] = vv[p];
            }
            // eliminate across L pattern
            let ls = self.l_row[i];
            let le = self.l_row[i + 1];
            for pos in ls..le {
                let j = self.l_col[pos];
                let wij = if w.mark[j] >= 0 {
                    w.val[w.mark[j] as usize]
                } else {
                    0.0
                };
                let djj = self.u_val[self.u_diag_ix[j]];
                let lij = if djj != 0.0 { wij / djj } else { 0.0 };
                self.l_val[pos] = lij;
                let urs = self.u_row[j];
                let ure = self.u_row[j + 1];
                for q in urs..ure {
                    let kcol = self.u_col[q];
                    if kcol <= j {
                        continue;
                    }
                    let mk = w.mark.get(kcol).copied().unwrap_or(-1);
                    if mk >= 0 {
                        w.val[mk as usize] -= lij * self.u_val[q];
                    }
                }
            }
            // finalize U row
            let us = self.u_row[i];
            let ue = self.u_row[i + 1];
            let mut diag = 0.0;
            for q in us..ue {
                let k = self.u_col[q];
                let v = if w.mark.get(k).copied().unwrap_or(-1) >= 0 {
                    w.val[w.mark[k] as usize]
                } else {
                    0.0
                };
                if k == i {
                    diag = v;
                }
                self.u_val[q] = v;
            }
            let fixed = pivot::handle_pivot(
                diag,
                self.cfg.pivot,
                self.cfg.pivot_threshold,
                self.cfg.diag_perturb_factor,
                max_diag_abs,
            )
            .map_err(|_| KError::ZeroPivot(i))?;
            self.u_val[self.u_diag_ix[i]] = fixed;
            symbolic::clear_rowwork(&mut w);
        }
        Ok(())
    }

    fn factor_ilut_numeric_only(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        // recompute max_diag_abs
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut max_diag_abs = 0.0f64;
        for i in 0..self.n {
            let mut di = 0.0;
            for p in rp[i]..rp[i + 1] {
                if cj[p] == i {
                    di = vv[p];
                    break;
                }
            }
            max_diag_abs = max_diag_abs.max(di.abs());
        }
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
        self.apply_op(Op::NoTrans, x, y)
    }

    fn apply_op(&self, op: Op, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "IluCsr::apply dimension mismatch: n={}, x.len()={}, y.len()={}",
                self.n,
                x.len(),
                y.len()
            )));
        }
        match op {
            Op::NoTrans => {
                if self.cfg.level_sched {
                    tri_solve::tri_solve_level_scheduled(self, x, y)
                } else {
                    tri_solve::tri_solve_serial(self, x, y)
                }
            }
            Op::Trans | Op::ConjTrans => {
                let ut = self.ut.get_or_init(|| transpose_csr(self.n, &self.u_row, &self.u_col, &self.u_val));
                let lt = self.lt.get_or_init(|| transpose_csr(self.n, &self.l_row, &self.l_col, &self.l_val));
                tri_solve::tri_solve_transpose_serial(
                    self,
                    &ut.0,
                    &ut.1,
                    &ut.2,
                    &lt.0,
                    &lt.1,
                    &lt.2,
                    x,
                    y,
                )
            }
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

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: true,
            supports_conj_trans: false,
            is_spd: false,
            side_restriction: None,
        }
    }
}

// === Simple accessors for internal solves ===
impl IluCsr {
    #[inline]
    pub(crate) fn n(&self) -> usize {
        self.n
    }
    #[inline]
    pub(crate) fn l_row(&self) -> &[usize] {
        &self.l_row
    }
    #[inline]
    pub(crate) fn l_col(&self) -> &[usize] {
        &self.l_col
    }
    #[inline]
    pub(crate) fn l_val(&self) -> &[f64] {
        &self.l_val
    }
    #[inline]
    pub(crate) fn u_row(&self) -> &[usize] {
        &self.u_row
    }
    #[inline]
    pub(crate) fn u_col(&self) -> &[usize] {
        &self.u_col
    }
    #[inline]
    pub(crate) fn u_val(&self) -> &[f64] {
        &self.u_val
    }
    #[inline]
    pub(crate) fn u_diag_ix(&self) -> &[usize] {
        &self.u_diag_ix
    }
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn tmp(&self) -> &[f64] {
        &self.tmp
    }
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn tmp_mut(&mut self) -> &mut [f64] {
        &mut self.tmp
    }

    #[inline]
    pub(crate) fn buckets_fwd(&self) -> &[Vec<usize>] {
        &self.buckets_fwd
    }
    #[inline]
    pub(crate) fn buckets_bwd(&self) -> &[Vec<usize>] {
        &self.buckets_bwd
    }
}

fn transpose_csr(
    n: usize,
    row: &[usize],
    col: &[usize],
    val: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = col.len();
    let mut t_row = vec![0usize; n + 1];
    for &j in col {
        t_row[j + 1] += 1;
    }
    for i in 0..n {
        t_row[i + 1] += t_row[i];
    }
    let mut t_col = vec![0usize; nnz];
    let mut t_val = vec![0f64; nnz];
    let mut offset = t_row.clone();
    for i in 0..n {
        for p in row[i]..row[i + 1] {
            let j = col[p];
            let dest = offset[j];
            t_col[dest] = i;
            t_val[dest] = val[p];
            offset[j] += 1;
        }
    }
    (t_row, t_col, t_val)
}
