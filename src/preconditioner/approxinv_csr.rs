//! CSR-based Sparse Approximate Inverse (SPAI) and FSAI preconditioners.
//!
//! These implementations operate directly on the CSR graph of the system
//! matrix and build either a full sparse approximate inverse (SPAI) or a
//! factored sparse approximate inverse for SPD matrices (FSAI), targeting
//! linear-ish build time in nnz(A) with small per-column dense solves.

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
#[cfg(feature = "complex")]
use crate::preconditioner::bridge::{
    apply_pc_mut_s as bridge_apply_pc_mut_s, apply_pc_s as bridge_apply_pc_s,
};
use crate::preconditioner::{PcSide, Preconditioner};

use faer::Mat;
use faer::prelude::SolveLstsq;

/// Which approximate inverse to build.
#[derive(Clone, Copy, Debug)]
pub enum ApproxInvKind {
    /// Factored sparse approximate inverse (SPD only): M^{-1} ≈ G G^T
    FSAI,
    /// Sparse approximate inverse (general): M^{-1} ≈ M
    SPAI,
}

/// Builder parameters for SPAI/FSAI.
#[derive(Clone, Copy, Debug)]
pub struct ApproxInvParams {
    pub kind: ApproxInvKind,
    pub levels: usize,      // graph expansion depth (k-ring)
    pub max_per_col: usize, // hard cap on nnz per column
    pub drop_tol: f64,      // prune small entries after solve
    pub reg: f64,           // diagonal regularization for small dense solves
    pub max_cond: f64,      // reserved (not enforced in first cut)
    pub parallel: bool,     // reserved (single-threaded build in first cut)
}

impl Default for ApproxInvParams {
    fn default() -> Self {
        Self {
            kind: ApproxInvKind::FSAI,
            levels: 1,
            max_per_col: 20,
            drop_tol: 1e-3,
            reg: 1e-12,
            max_cond: 1e12,
            parallel: cfg!(feature = "rayon"),
        }
    }
}

/// Fluent builder for SPAI/FSAI preconditioners.
pub struct ApproxInvBuilder {
    p: ApproxInvParams,
}

impl ApproxInvBuilder {
    pub fn new(kind: ApproxInvKind) -> Self {
        let mut p = ApproxInvParams::default();
        p.kind = kind;
        Self { p }
    }
    pub fn levels(mut self, l: usize) -> Self {
        self.p.levels = l;
        self
    }
    pub fn max_per_col(mut self, s: usize) -> Self {
        self.p.max_per_col = s.max(1);
        self
    }
    pub fn drop_tol(mut self, t: f64) -> Self {
        self.p.drop_tol = t.max(S::zero().real());
        self
    }
    pub fn reg(mut self, r: f64) -> Self {
        self.p.reg = if r >= S::zero().real() {
            r
        } else {
            S::zero().real()
        };
        self
    }
    pub fn max_cond(mut self, c: f64) -> Self {
        self.p.max_cond = if c > S::zero().real() { c } else { 1e12 };
        self
    }
    pub fn parallel(mut self, on: bool) -> Self {
        self.p.parallel = on;
        self
    }

    pub fn build_fsai(self, a: &CsrMatrix<f64>) -> Result<FsaiCsr, KError> {
        FsaiCsr::build_from_csr(a.clone(), self.p)
    }
    pub fn build_spai(self, a: &CsrMatrix<f64>) -> Result<SpaiCsr, KError> {
        SpaiCsr::build_from_csr(a.clone(), self.p)
    }
}

/// FSAI: lower-triangular factor G in CSR and per-column patterns.
pub struct FsaiCsr {
    pub(crate) g: CsrMatrix<f64>,    // lower triangular
    pub(crate) pat: Vec<Vec<usize>>, // per-column patterns (global indices), restricted to rows >= col
    pub(crate) params: ApproxInvParams,
    // book-keeping for update reuse
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,
}

/// SPAI: full sparse approximate inverse M in CSR and per-column patterns.
pub struct SpaiCsr {
    pub(crate) m: CsrMatrix<f64>,
    pub(crate) pat: Vec<Vec<usize>>, // per-column patterns
    pub(crate) params: ApproxInvParams,
    // book-keeping for update reuse
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,
}

// ----------------------------- Utilities ---------------------------------

#[inline]
fn csr_find(a: &CsrMatrix<f64>, row: usize, col: usize) -> f64 {
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vv = a.values();
    let (rs, re) = (rp[row], rp[row + 1]);
    let cols = &ci[rs..re];
    match cols.binary_search(&col) {
        Ok(k) => vv[rs + k],
        Err(_) => S::zero().real(),
    }
}

#[inline]
fn spmv_csr(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), a.ncols());
    assert_eq!(y.len(), a.nrows());
    // clear y
    for yi in y.iter_mut() {
        *yi = S::zero().real();
    }
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vv = a.values();
    for i in 0..a.nrows() {
        let (rs, re) = (rp[i], rp[i + 1]);
        let mut sum = S::zero().real();
        for p in rs..re {
            sum += vv[p] * x[ci[p]];
        }
        y[i] = sum;
    }
}

#[inline]
fn spmv_csr_transpose(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), a.nrows());
    assert_eq!(y.len(), a.ncols());
    for yi in y.iter_mut() {
        *yi = S::zero().real();
    }
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vv = a.values();
    for i in 0..a.nrows() {
        let (rs, re) = (rp[i], rp[i + 1]);
        let xi = x[i];
        for p in rs..re {
            y[ci[p]] += vv[p] * xi;
        }
    }
}

/// Grow a sparsity pattern around column `i` using the row-adjacency graph.
fn grow_pattern_row_graph(a: &CsrMatrix<f64>, i: usize, levels: usize, cap: usize) -> Vec<usize> {
    // Start with {i}
    let n = a.nrows();
    assert!(i < n);
    let mut cur: Vec<usize> = vec![i];
    let mut acc: Vec<usize> = vec![i];

    // BFS up to `levels` on row-adjacency:
    for _ in 0..levels {
        let mut next: Vec<usize> = Vec::new();
        for &u in &cur {
            // neighbors are columns present in row u
            let rp = a.row_ptr();
            let ci = a.col_idx();
            let (rs, re) = (rp[u], rp[u + 1]);
            for p in rs..re {
                let v = ci[p];
                next.push(v);
            }
        }
        // Merge: append to acc, then sort+dedup
        acc.extend(next);
        acc.sort_unstable();
        acc.dedup();
        if acc.len() > cap {
            acc.truncate(cap);
        }
        // frontier becomes the just-added nodes (approximate) — simple heuristic
        cur = acc.clone();
        if cur.len() >= cap {
            break;
        }
    }

    // Ensure i present and sorted unique
    if !acc.contains(&i) {
        acc.push(i);
    }
    acc.sort_unstable();
    acc.dedup();
    if acc.len() > cap {
        acc.truncate(cap);
    }
    acc
}

// -------------------------- FSAI: build/apply ----------------------------

impl FsaiCsr {
    fn build_from_csr(a: CsrMatrix<f64>, cfg: ApproxInvParams) -> Result<Self, KError> {
        let n = a.nrows().min(a.ncols());

        // 1) Build per-column patterns, restricted to lower triangle rows (r >= i)
        let mut pat: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut s = grow_pattern_row_graph(&a, i, cfg.levels, cfg.max_per_col);
            // restrict to lower triangle for column i
            s.retain(|&r| r >= i);
            if !s.contains(&i) {
                s.insert(0, i);
            }
            s.sort_unstable();
            s.dedup();
            if s.len() > cfg.max_per_col {
                s.truncate(cfg.max_per_col);
            }
            pat.push(s);
        }

        // 2) For each column, solve (A_SS + reg I) g = e_i|_S
        let mut trips: Vec<(usize, usize, R)> = Vec::new();
        let mut a_ss = Mat::<R>::from_fn(1, 1, |_, _| R::default()); // resized per column
        let mut b = vec![R::default(); 1];

        for i in 0..n {
            let s = &pat[i];
            let m = s.len();
            if m == 0 {
                continue;
            }
            if a_ss.nrows() != m || a_ss.ncols() != m {
                a_ss = Mat::<R>::from_fn(m, m, |_, _| R::default());
                b.resize(m, R::default());
            }
            // A_SS
            for p in 0..m {
                for q in 0..=p {
                    let v = csr_find(&a, s[p], s[q]);
                    a_ss[(p, q)] = v;
                    a_ss[(q, p)] = v;
                }
            }
            // regularize
            for d in 0..m {
                a_ss[(d, d)] += cfg.reg;
            }
            // b = e_i restricted to S
            for k in 0..m {
                b[k] = R::default();
            }
            if let Ok(pos) = s.binary_search(&i) {
                b[pos] = S::one().real();
            } else {
                // enforce presence
                // should not happen as we inserted i
                continue;
            }

            // Solve using QR (robust for SPD + reg)
            let rhs = Mat::<R>::from_fn(m, 1, |r, _| b[r]);
            let sol = faer::linalg::solvers::Qr::new(a_ss.as_ref()).solve_lstsq(rhs);
            let mut norm2: R = R::default();
            for r in 0..m {
                norm2 += sol[(r, 0)] * sol[(r, 0)];
            }
            norm2 = norm2.sqrt();
            let thr = cfg.drop_tol * norm2.max(1e-32);

            // track kept rows for this column to pin structure for updates
            let mut kept: Vec<usize> = Vec::with_capacity(m);
            for (k, &row) in s.iter().enumerate() {
                let val = sol[(k, 0)];
                if val.abs() >= thr {
                    trips.push((row, i, val)); // lower rows only ensured by pattern
                    kept.push(row);
                }
            }
            // Overwrite pattern with kept rows to preserve numeric update structure
            // (still guaranteed sorted and lower-triangular)
            // Safety: `pat[i]` is small.
            let mut kept_sorted = kept;
            kept_sorted.sort_unstable();
            pat[i] = kept_sorted;
        }

        // 3) Assemble CSR from triplets
        let g = assemble_csr(n, n, &mut trips);
        Ok(Self {
            g,
            pat,
            params: cfg,
            last_sid: None,
            last_vid: None,
        })
    }
}

impl Preconditioner for FsaiCsr {
    fn setup(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        // Always require CSR view
        let csr = csr_from_linop(a, S::zero().real())?; // no drop on A
        let sid = a.structure_id();
        let vid = a.values_id();

        // Recompute numeric with fixed pattern if possible, else rebuild.
        if let (Some(ls), Some(lv)) = (self.last_sid, self.last_vid)
            && ls == sid
            && lv != vid
        {
            // values changed only: refresh numeric with fixed pattern
            self.update_numeric(a)?;
            self.last_vid = Some(vid);
            return Ok(());
        }

        // Full rebuild using current params against CSR
        let rebuilt = FsaiCsr::build_from_csr((*csr).clone(), self.params)?;
        *self = rebuilt;
        self.last_sid = Some(sid);
        self.last_vid = Some(vid);
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        // y = G (G^T x)
        if x.len() != self.g.nrows() || y.len() != self.g.nrows() {
            return Err(KError::InvalidInput(format!(
                "FsaiCsr::apply dimension mismatch: n={}, x.len()={}, y.len()={}",
                self.g.nrows(),
                x.len(),
                y.len()
            )));
        }
        let n = x.len();
        let mut t = vec![R::default(); n];
        spmv_csr_transpose(&self.g, x, &mut t);
        spmv_csr(&self.g, &t, y);
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        // Re-solve per-column with fixed pattern and write values back into existing G
        let csr = csr_from_linop(a, S::zero().real())?;
        let n = self.g.nrows().min(self.g.ncols());
        let mut a_ss = Mat::<R>::from_fn(1, 1, |_, _| R::default());
        let mut b = vec![R::default(); 1];

        // We'll rebuild triplets but write into a fresh CSR to keep code simple and pattern stable
        let mut trips: Vec<(usize, usize, R)> = Vec::new();
        for i in 0..n {
            let s = &self.pat[i];
            let m = s.len();
            if m == 0 {
                continue;
            }
            if a_ss.nrows() != m || a_ss.ncols() != m {
                a_ss = Mat::<R>::from_fn(m, m, |_, _| R::default());
                b.resize(m, R::default());
            }
            for p in 0..m {
                for q in 0..=p {
                    let v = csr_find(&csr, s[p], s[q]);
                    a_ss[(p, q)] = v;
                    a_ss[(q, p)] = v;
                }
            }
            for d in 0..m {
                a_ss[(d, d)] += self.params.reg;
            }
            for k in 0..m {
                b[k] = R::default();
            }
            if let Ok(pos) = s.binary_search(&i) {
                b[pos] = S::one().real();
            } else {
                continue;
            }
            let rhs = Mat::<R>::from_fn(m, 1, |r, _| b[r]);
            let sol = faer::linalg::solvers::Qr::new(a_ss.as_ref()).solve_lstsq(rhs);
            // No pruning during update to preserve structure
            for (k, &row) in s.iter().enumerate() {
                let val = sol[(k, 0)];
                trips.push((row, i, val));
            }
        }
        self.g = assemble_csr(n, n, &mut trips);
        Ok(())
    }

    fn required_format(&self) -> crate::matrix::format::FormatHint {
        crate::matrix::format::FormatHint::Csr
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for FsaiCsr {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        (self.g.nrows(), self.g.ncols())
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_mut_s(self, side, x, y, scratch)
    }
}

// -------------------------- SPAI: build/apply ----------------------------

impl SpaiCsr {
    fn build_from_csr(a: CsrMatrix<f64>, cfg: ApproxInvParams) -> Result<Self, KError> {
        let n = a.nrows().min(a.ncols());

        // 1) Build per-column patterns
        let mut pat: Vec<Vec<usize>> = Vec::with_capacity(n);
        for j in 0..n {
            let mut s = grow_pattern_row_graph(&a, j, cfg.levels, cfg.max_per_col);
            if !s.contains(&j) {
                s.push(j);
            }
            s.sort_unstable();
            s.dedup();
            if s.len() > cfg.max_per_col {
                s.truncate(cfg.max_per_col);
            }
            pat.push(s);
        }

        // 2) Per-column normal equations: (A_S^T A_S + reg I) m = A_S^T e_j
        let mut trips: Vec<(usize, usize, R)> = Vec::new();
        let rp = a.row_ptr();
        let ci = a.col_idx();
        let vv = a.values();

        // Thread-local scratch (single-threaded here)
        let mut idx_in_s: Vec<i32> = vec![-1; n]; // map global col -> local pos or -1
        for j in 0..n {
            let s = &pat[j];
            let m = s.len();
            if m == 0 {
                continue;
            }
            // map
            for (pos, &g) in s.iter().enumerate() {
                idx_in_s[g] = pos as i32;
            }
            let mut nmat = Mat::<R>::from_fn(m, m, |_, _| R::default());
            let mut cvec = Mat::<R>::from_fn(m, 1, |_, _| R::default());

            // c = row j of A restricted to S
            let (rj, rj2) = (rp[j], rp[j + 1]);
            for pidx in rj..rj2 {
                let col = ci[pidx];
                let val = vv[pidx];
                let pos = idx_in_s[col];
                if pos >= 0 {
                    cvec[(pos as usize, 0)] = val;
                }
            }

            // N = A_S^T A_S: iterate rows, intersect with S by membership check via idx_in_s
            for i in 0..n {
                let (rs, re) = (rp[i], rp[i + 1]);
                // collect positions and values in S for this row
                // small temporary buffer
                let mut pos_tmp: smallvec::SmallVec<[(usize, R); 32]> = smallvec::SmallVec::new();
                for p in rs..re {
                    let col = ci[p];
                    let pos = idx_in_s[col];
                    if pos >= 0 {
                        pos_tmp.push((pos as usize, vv[p]));
                    }
                }
                // accumulate into lower triangle
                for ix in 0..pos_tmp.len() {
                    let (px, vx) = pos_tmp[ix];
                    for iy in 0..=ix {
                        let (py, vy) = pos_tmp[iy];
                        let v = vx * vy;
                        nmat[(px, py)] += v;
                        if px != py {
                            nmat[(py, px)] += v;
                        }
                    }
                }
            }

            // regularize
            for d in 0..m {
                nmat[(d, d)] += cfg.reg;
            }
            // Solve SPD system via QR (robust for small m)
            let sol = faer::linalg::solvers::Qr::new(nmat.as_ref()).solve_lstsq(cvec);
            // prune
            let mut norm2: R = R::default();
            for r in 0..m {
                norm2 += sol[(r, 0)] * sol[(r, 0)];
            }
            norm2 = norm2.sqrt();
            let thr = cfg.drop_tol * norm2.max(1e-32);
            let mut kept: Vec<usize> = Vec::with_capacity(m);
            for (k, &row) in s.iter().enumerate() {
                let val = sol[(k, 0)];
                if val.abs() >= thr {
                    trips.push((row, j, val));
                    kept.push(row);
                }
            }
            // clear map for all original entries in s
            for &g in s.iter() {
                idx_in_s[g] = -1;
            }
            // Pin pattern to kept rows to preserve update structure
            kept.sort_unstable();
            pat[j] = kept;
        }

        let m = assemble_csr(n, n, &mut trips);
        Ok(Self {
            m,
            pat,
            params: cfg,
            last_sid: None,
            last_vid: None,
        })
    }
}

impl Preconditioner for SpaiCsr {
    fn setup(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let csr = csr_from_linop(a, S::zero().real())?;
        let sid = a.structure_id();
        let vid = a.values_id();

        if let (Some(ls), Some(lv)) = (self.last_sid, self.last_vid)
            && ls == sid
            && lv != vid
        {
            self.update_numeric(a)?;
            self.last_vid = Some(vid);
            return Ok(());
        }

        let rebuilt = SpaiCsr::build_from_csr((*csr).clone(), self.params)?;
        *self = rebuilt;
        self.last_sid = Some(sid);
        self.last_vid = Some(vid);
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if x.len() != self.m.ncols() || y.len() != self.m.nrows() {
            return Err(KError::InvalidInput(format!(
                "SpaiCsr::apply dimension mismatch: A={}x{}, x.len()={}, y.len()={}",
                self.m.nrows(),
                self.m.ncols(),
                x.len(),
                y.len()
            )));
        }
        spmv_csr(&self.m, x, y);
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let csr = csr_from_linop(a, S::zero().real())?;
        let n = self.m.nrows().min(self.m.ncols());
        let rp = csr.row_ptr();
        let ci = csr.col_idx();
        let vv = csr.values();
        let mut idx_in_s: Vec<i32> = vec![-1; n];
        let mut trips: Vec<(usize, usize, R)> = Vec::new();

        for j in 0..n {
            let s = &self.pat[j];
            let m = s.len();
            if m == 0 {
                continue;
            }
            for (pos, &g) in s.iter().enumerate() {
                idx_in_s[g] = pos as i32;
            }
            let mut nmat = Mat::<R>::from_fn(m, m, |_, _| R::default());
            let mut cvec = Mat::<R>::from_fn(m, 1, |_, _| R::default());
            // cvec
            let (rj, rj2) = (rp[j], rp[j + 1]);
            for pidx in rj..rj2 {
                let col = ci[pidx];
                let val = vv[pidx];
                let pos = idx_in_s[col];
                if pos >= 0 {
                    cvec[(pos as usize, 0)] = val;
                }
            }
            // N
            for i in 0..n {
                let (rs, re) = (rp[i], rp[i + 1]);
                let mut pos_tmp: smallvec::SmallVec<[(usize, f64); 32]> = smallvec::SmallVec::new();
                for p in rs..re {
                    let col = ci[p];
                    let pos = idx_in_s[col];
                    if pos >= 0 {
                        pos_tmp.push((pos as usize, vv[p]));
                    }
                }
                for ix in 0..pos_tmp.len() {
                    let (px, vx) = pos_tmp[ix];
                    for iy in 0..=ix {
                        let (py, vy) = pos_tmp[iy];
                        let v = vx * vy;
                        nmat[(px, py)] += v;
                        if px != py {
                            nmat[(py, px)] += v;
                        }
                    }
                }
            }
            for d in 0..m {
                nmat[(d, d)] += self.params.reg;
            }
            let sol = faer::linalg::solvers::Qr::new(nmat.as_ref()).solve_lstsq(cvec);
            // no pruning during update to preserve structure
            for (k, &row) in s.iter().enumerate() {
                trips.push((row, j, sol[(k, 0)]));
            }
            for &g in s.iter() {
                idx_in_s[g] = -1;
            }
        }
        self.m = assemble_csr(n, n, &mut trips);
        Ok(())
    }

    fn required_format(&self) -> crate::matrix::format::FormatHint {
        crate::matrix::format::FormatHint::Csr
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for SpaiCsr {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        (self.m.nrows(), self.m.ncols())
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_mut_s(self, side, x, y, scratch)
    }
}

// ---------------------------- Assembly helper ----------------------------

fn assemble_csr(nrows: usize, ncols: usize, trips: &mut Vec<(usize, usize, R)>) -> CsrMatrix<f64> {
    // Sort by (row, col)
    trips.sort_unstable_by(|a, b| match a.0.cmp(&b.0) {
        std::cmp::Ordering::Equal => a.1.cmp(&b.1),
        o => o,
    });
    // Build CSR arrays
    let mut row_ptr = vec![0usize; nrows + 1];
    let mut col_idx: Vec<usize> = Vec::with_capacity(trips.len());
    let mut vals: Vec<R> = Vec::with_capacity(trips.len());

    let mut cur_row = 0usize;
    let mut acc = 0usize;
    let mut k = 0usize;
    while k < trips.len() {
        let (r, c, mut v) = trips[k];
        // advance rows
        while cur_row < r {
            row_ptr[cur_row + 1] = acc;
            cur_row += 1;
        }
        // combine duplicates
        k += 1;
        while k < trips.len() && trips[k].0 == r && trips[k].1 == c {
            v += trips[k].2;
            k += 1;
        }
        col_idx.push(c);
        vals.push(v);
        acc += 1;
    }
    while cur_row < nrows {
        row_ptr[cur_row + 1] = acc;
        cur_row += 1;
    }

    // Ensure row_ptr nondecreasing and col_idx sorted within rows (implied by sort)
    CsrMatrix::from_csr(nrows, ncols, row_ptr, col_idx, vals)
}

// ---------------------------- Public entrypoints -------------------------

impl FsaiCsr {
    /// Create a new, empty FsaiCsr with parameters (primarily for builder use).
    pub fn new_with_params(params: ApproxInvParams) -> Self {
        Self {
            g: CsrMatrix::from_csr(0, 0, vec![0], vec![], vec![]),
            pat: Vec::new(),
            params,
            last_sid: None,
            last_vid: None,
        }
    }
}

impl SpaiCsr {
    /// Create a new, empty SpaiCsr with parameters (primarily for builder use).
    pub fn new_with_params(params: ApproxInvParams) -> Self {
        Self {
            m: CsrMatrix::from_csr(0, 0, vec![0], vec![], vec![]),
            pat: Vec::new(),
            params,
            last_sid: None,
            last_vid: None,
        }
    }
}

#[cfg(all(test, feature = "complex"))]
mod tests {
    use super::*;
    use crate::algebra::bridge::BridgeScratch;
    use crate::ops::kpc::KPreconditioner;

    fn poisson_1d_matrix() -> CsrMatrix<f64> {
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        CsrMatrix::from_csr(3, 3, row_ptr, col_idx, values)
    }

    #[test]
    fn fsai_apply_s_matches_real_path() {
        let a = poisson_1d_matrix();
        let pc = ApproxInvBuilder::new(ApproxInvKind::FSAI)
            .levels(1)
            .build_fsai(&a)
            .expect("fsai build");

        let rhs_real = vec![1.0, 2.0, 3.0];
        let mut out_real = vec![S::zero().real(); rhs_real.len()];
        pc.apply(PcSide::Left, &rhs_real, &mut out_real)
            .expect("fsai apply real");

        let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
        let mut out_s = vec![S::zero(); rhs_s.len()];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
            .expect("fsai apply_s");

        for (ys, &yr) in out_s.iter().zip(out_real.iter()) {
            assert!((ys.real() - yr).abs() < 1e-10);
        }
    }

    #[test]
    fn spai_apply_s_matches_real_path() {
        let a = poisson_1d_matrix();
        let pc = ApproxInvBuilder::new(ApproxInvKind::SPAI)
            .levels(1)
            .build_spai(&a)
            .expect("spai build");

        let rhs_real = vec![1.5, -0.5, 0.25];
        let mut out_real = vec![S::zero().real(); rhs_real.len()];
        pc.apply(PcSide::Left, &rhs_real, &mut out_real)
            .expect("spai apply real");

        let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
        let mut out_s = vec![S::zero(); rhs_s.len()];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
            .expect("spai apply_s");

        for (ys, &yr) in out_s.iter().zip(out_real.iter()) {
            assert!((ys.real() - yr).abs() < 1e-10);
        }
    }
}
