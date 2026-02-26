use std::sync::Arc;

use crate::algebra::scalar::KrystScalar;
use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::format::OpFormat;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::{
    LocalPreconditioner, Op, PcCaps, PcDistributedSupport, PcSide, Preconditioner,
};
use crate::utils::conditioning::{ConditioningOptions, apply_csr_transforms};
use crate::utils::permutation::{Permutation, amd_csr, permute_csr_symmetric, rcm_csr};

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::algebra::scalar::S;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;

use once_cell::sync::OnceCell;

// ILU_CSR is restricted to real scalars for now.
type Real = f64;

mod csr_builder;
mod ilut_params;
mod pivot;
mod pos_map;
mod row_work;
mod tri_solve;

pub use ilut_params::{IlutParams, PivotPolicy, Pivoting};
pub use pivot::PivotStrategy;

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
    pub reordering: ReorderingOptions,
    pub conditioning: ConditioningOptions,
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
            reordering: ReorderingOptions::default(),
            conditioning: ConditioningOptions::default(),
        }
    }
}

#[cfg(feature = "complex")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IluComplexKernelMode {
    Native,
    DegradedRealProjection,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReorderingKind {
    None,
    Rcm,
    Amd,
}

#[derive(Clone, Debug)]
pub struct ReorderingOptions {
    pub kind: ReorderingKind,
    pub symmetric: bool,
    pub deterministic: bool,
}

impl Default for ReorderingOptions {
    fn default() -> Self {
        Self {
            kind: ReorderingKind::None,
            symmetric: true,
            deterministic: true,
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
    l_val: Vec<Real>,
    // U upper including diagonal
    u_row: Vec<usize>,
    u_col: Vec<usize>,
    u_val: Vec<Real>,
    u_diag_ix: Vec<usize>,
    // Optional per-entry levels for ILUK
    l_lev: Vec<usize>,
    u_lev: Vec<usize>,

    // cached transposes, built lazily
    lt: OnceCell<(Vec<usize>, Vec<usize>, Vec<Real>)>,
    ut: OnceCell<(Vec<usize>, Vec<usize>, Vec<Real>)>,

    // optional level scheduling
    levels_fwd: Vec<usize>,
    levels_bwd: Vec<usize>,
    buckets_fwd: Vec<Vec<usize>>,
    buckets_bwd: Vec<Vec<usize>>,

    // scratch for apply
    tmp: Vec<Real>,
    perm: Permutation,
    #[cfg(feature = "complex")]
    c_l_val: Vec<S>,
    #[cfg(feature = "complex")]
    c_u_val: Vec<S>,
    #[cfg(feature = "complex")]
    c_tmp: Vec<S>,
    #[cfg(feature = "complex")]
    native_complex_active: bool,
    #[cfg(feature = "complex")]
    complex_kernel_mode: IluComplexKernelMode,
    #[cfg(feature = "complex")]
    complex_force_degraded: bool,
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
            lt: OnceCell::new(),
            ut: OnceCell::new(),
            levels_fwd: Vec::new(),
            levels_bwd: Vec::new(),
            buckets_fwd: Vec::new(),
            buckets_bwd: Vec::new(),
            tmp: Vec::new(),
            perm: Permutation::identity(0),
            #[cfg(feature = "complex")]
            c_l_val: Vec::new(),
            #[cfg(feature = "complex")]
            c_u_val: Vec::new(),
            #[cfg(feature = "complex")]
            c_tmp: Vec::new(),
            #[cfg(feature = "complex")]
            native_complex_active: false,
            #[cfg(feature = "complex")]
            complex_kernel_mode: IluComplexKernelMode::DegradedRealProjection,
            #[cfg(feature = "complex")]
            complex_force_degraded: false,
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

    #[cfg(feature = "complex")]
    fn factor_ilu0_complex(&mut self, a: &CsrMatrix<S>) -> Result<(), KError> {
        let a_zero = CsrMatrix::from_csr(
            a.nrows(),
            a.ncols(),
            a.row_ptr().to_vec(),
            a.col_idx().to_vec(),
            vec![0.0; a.values().len()],
        );
        self.factor_ilu0(&a_zero)?;
        self.c_l_val.resize(self.l_val.len(), S::zero());
        self.c_u_val.resize(self.u_val.len(), S::zero());

        let n = a.nrows();
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut map = URowMap::new();
        map.ensure_size(n);
        let milu = matches!(self.cfg.kind, IluKind::Milu0);

        for i in 0..n {
            map.prime(&self.u_row, &self.u_col, i);
            for p in rp[i]..rp[i + 1] {
                let j = cj[p];
                let val = vv[p];
                if j < i {
                    let ls = self.l_row[i];
                    if let Ok(off) = self.l_col[ls..self.l_row[i + 1]].binary_search(&j) {
                        self.c_l_val[ls + off] = val;
                    }
                } else if let Some(pos) = map.get(j) {
                    self.c_u_val[pos] = val;
                }
            }

            for p in self.l_row[i]..self.l_row[i + 1] {
                let j = self.l_col[p];
                let wij = self.c_l_val[p];
                if wij == S::zero() {
                    continue;
                }
                let djj = self.c_u_val[self.u_diag_ix[j]];
                let mult = wij / djj;
                self.c_l_val[p] = mult;
                for q in self.u_row[j]..self.u_row[j + 1] {
                    let k = self.u_col[q];
                    if k <= j {
                        continue;
                    }
                    let ujk = self.c_u_val[q];
                    if let Some(pos_ik) = map.get(k) {
                        self.c_u_val[pos_ik] -= mult * ujk;
                    } else if milu {
                        self.c_u_val[self.u_diag_ix[i]] -= mult * ujk;
                    }
                }
            }

            let d = self.c_u_val[self.u_diag_ix[i]];
            if d.abs() <= self.cfg.pivot_threshold {
                return Err(KError::ZeroPivot(i));
            }
        }

        self.native_complex_active = true;
        self.complex_kernel_mode = IluComplexKernelMode::Native;
        Ok(())
    }

    #[cfg(feature = "complex")]
    fn factor_iluk_complex(&mut self, a: &CsrMatrix<S>, k: usize) -> Result<(), KError> {
        let a_zero = CsrMatrix::from_csr(
            a.nrows(),
            a.ncols(),
            a.row_ptr().to_vec(),
            a.col_idx().to_vec(),
            vec![0.0; a.values().len()],
        );
        self.factor_iluk(&a_zero, k)?;
        self.c_l_val.resize(self.l_val.len(), S::zero());
        self.c_u_val.resize(self.u_val.len(), S::zero());

        let n = a.nrows();
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let mut map = URowMap::new();
        map.ensure_size(n);

        for i in 0..n {
            map.prime(&self.u_row, &self.u_col, i);
            for p in rp[i]..rp[i + 1] {
                let j = cj[p];
                let val = vv[p];
                if j < i {
                    let ls = self.l_row[i];
                    if let Ok(off) = self.l_col[ls..self.l_row[i + 1]].binary_search(&j) {
                        self.c_l_val[ls + off] = val;
                    }
                } else if let Some(pos) = map.get(j) {
                    self.c_u_val[pos] = val;
                }
            }

            for p in self.l_row[i]..self.l_row[i + 1] {
                let j = self.l_col[p];
                let wij = self.c_l_val[p];
                if wij == S::zero() {
                    continue;
                }
                let djj = self.c_u_val[self.u_diag_ix[j]];
                let mult = wij / djj;
                self.c_l_val[p] = mult;
                for q in self.u_row[j]..self.u_row[j + 1] {
                    let kcol = self.u_col[q];
                    if kcol <= j {
                        continue;
                    }
                    let ujk = self.c_u_val[q];
                    if let Some(pos_ik) = map.get(kcol) {
                        self.c_u_val[pos_ik] -= mult * ujk;
                    }
                }
            }

            if self.c_u_val[self.u_diag_ix[i]].abs() <= self.cfg.pivot_threshold {
                return Err(KError::ZeroPivot(i));
            }
        }

        self.native_complex_active = true;
        self.complex_kernel_mode = IluComplexKernelMode::Native;
        Ok(())
    }

    #[cfg(feature = "complex")]
    fn factor_ilut_complex(&mut self, a: &CsrMatrix<S>, params: &IlutParams) -> Result<(), KError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(KError::InvalidInput("ILUT requires square matrix".into()));
        }
        self.n = n;

        let mut l_cols_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut l_vals_rows: Vec<Vec<S>> = vec![Vec::new(); n];
        let mut u_cols_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut u_vals_rows: Vec<Vec<S>> = vec![Vec::new(); n];
        let mut inv_diag_u = vec![S::zero(); n];

        let mut w = RowWork::new();
        w.ensure_size(n);
        let mut l_tmp: Vec<(usize, S)> = Vec::new();
        let mut u_tmp: Vec<(usize, S)> = Vec::new();

        for i in 0..n {
            w.clear_row();
            l_tmp.clear();
            u_tmp.clear();

            let (a_cols, a_vals) = a.row(i);
            let mut row_inf = 0.0f64;
            for (&j, &v) in a_cols.iter().zip(a_vals.iter()) {
                if v != S::zero() {
                    w.set(j, v);
                    row_inf = row_inf.max(v.abs());
                }
            }
            let tau = params.droptol_abs + params.droptol_rel * row_inf;

            let mut lowers: Vec<usize> = w.iter().filter(|&(j, _)| j < i).map(|(j, _)| j).collect();
            lowers.sort_unstable();
            for &k in &lowers {
                let wk = w.get(k);
                if wk == S::zero() {
                    continue;
                }
                let lik = wk * inv_diag_u[k];
                if params.early_drop && lik.abs() < tau {
                    w.set(k, S::zero());
                    continue;
                }
                l_tmp.push((k, lik));
                w.set(k, S::zero());

                for (&j, &ukj) in u_cols_rows[k].iter().zip(u_vals_rows[k].iter()) {
                    if j <= k {
                        continue;
                    }
                    let newv = w.get(j) - lik * ukj;
                    if params.early_drop && newv.abs() < tau {
                        w.set(j, S::zero());
                    } else {
                        w.set(j, newv);
                    }
                }
            }

            for (j, v) in w.iter() {
                if j >= i && (j == i || v.abs() >= tau) {
                    u_tmp.push((j, v));
                }
            }
            if !u_tmp.iter().any(|(j, _)| *j == i) {
                u_tmp.push((i, S::zero()));
            }

            if params.p_l > 0 && l_tmp.len() > params.p_l {
                l_tmp.sort_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                l_tmp.truncate(params.p_l);
            }

            let diag_pos = u_tmp
                .iter()
                .position(|(j, _)| *j == i)
                .ok_or_else(|| KError::InvalidInput("missing diagonal".into()))?;
            let mut diag = u_tmp[diag_pos].1;
            let mut u_offdiag: Vec<(usize, S)> = u_tmp
                .iter()
                .enumerate()
                .filter_map(|(idx, &(j, v))| if idx == diag_pos { None } else { Some((j, v)) })
                .collect();
            if params.p_u > 0 && u_offdiag.len() > params.p_u {
                u_offdiag.sort_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                u_offdiag.truncate(params.p_u);
            }

            if params.reproducible_order {
                l_tmp.sort_by(|a, b| a.0.cmp(&b.0));
                u_offdiag.sort_by(|a, b| a.0.cmp(&b.0));
            } else {
                l_tmp.sort_unstable_by_key(|(c, _)| *c);
                u_offdiag.sort_unstable_by_key(|(c, _)| *c);
            }

            let tau_pivot = params.pivot_tau.max(0.0);
            match params.pivot {
                PivotPolicy::Strict => {
                    if diag.abs() < tau_pivot {
                        return Err(KError::ZeroPivot(i));
                    }
                }
                PivotPolicy::Threshold => {
                    if diag.abs() < tau_pivot {
                        diag = if diag == S::zero() {
                            S::from_real(tau_pivot)
                        } else {
                            diag * S::from_real(tau_pivot / diag.abs())
                        };
                    }
                }
                PivotPolicy::DiagonalPerturbation => {
                    if diag.abs() < tau_pivot {
                        let direction = if diag == S::zero() {
                            S::one()
                        } else {
                            diag / S::from_real(diag.abs())
                        };
                        diag += direction * S::from_real(tau_pivot);
                    }
                }
            }

            inv_diag_u[i] = diag.inv();
            l_cols_rows[i].extend(l_tmp.iter().map(|(c, _)| *c));
            l_vals_rows[i].extend(l_tmp.iter().map(|(_, v)| *v));

            u_cols_rows[i].extend(u_offdiag.iter().map(|(c, _)| *c));
            u_vals_rows[i].extend(u_offdiag.iter().map(|(_, v)| *v));
            u_cols_rows[i].push(i);
            u_vals_rows[i].push(diag);
            if params.reproducible_order {
                let mut pairs: Vec<(usize, S)> = u_cols_rows[i]
                    .iter()
                    .copied()
                    .zip(u_vals_rows[i].iter().copied())
                    .collect();
                pairs.sort_by_key(|(c, _)| *c);
                u_cols_rows[i].clear();
                u_vals_rows[i].clear();
                for (c, v) in pairs {
                    u_cols_rows[i].push(c);
                    u_vals_rows[i].push(v);
                }
            }
        }

        self.l_row.clear();
        self.l_col.clear();
        self.l_val.clear();
        self.u_row.clear();
        self.u_col.clear();
        self.u_val.clear();
        self.c_l_val.clear();
        self.c_u_val.clear();
        self.l_row.push(0);
        self.u_row.push(0);
        for i in 0..n {
            self.l_col.extend_from_slice(&l_cols_rows[i]);
            self.c_l_val.extend_from_slice(&l_vals_rows[i]);
            self.l_val.resize(self.l_col.len(), 0.0);
            self.l_row.push(self.l_col.len());

            self.u_col.extend_from_slice(&u_cols_rows[i]);
            self.c_u_val.extend_from_slice(&u_vals_rows[i]);
            self.u_val.resize(self.u_col.len(), 0.0);
            self.u_row.push(self.u_col.len());
        }
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
        self.native_complex_active = true;
        self.complex_kernel_mode = IluComplexKernelMode::Native;
        Ok(())
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
        self.l_val.resize(self.l_col.len(), Real::zero());
        self.u_val.resize(self.u_col.len(), Real::zero());
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
                        self.l_val[ls + off] = Real::from_real(val);
                    }
                } else {
                    // U part (including diagonal)
                    if let Some(pos) = map.get(j) {
                        self.u_val[pos] = Real::from_real(val);
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
                if ukk == Real::zero() {
                    return Err(KError::FactorError(format!(
                        "zero U(j,j) encountered at row {k}"
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
                    let val_q = self.u_val[q];
                    if let Some(pos_ij) = map.get(j) {
                        self.u_val[pos_ij] -= mult * val_q;
                    } else if milu {
                        let di_pos = self.u_diag_ix[i];
                        self.u_val[di_pos] -= mult * val_q;
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
                w.val[pos] = Real::from_real(vv[p]);
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
                if wij == Real::zero() {
                    continue;
                }
                let djj = {
                    let dix = self.u_diag_ix.get(j).copied().unwrap_or(0);
                    if j < i && self.u_val.get(dix).copied().unwrap_or(Real::zero()) == Real::zero()
                    {
                        // Not yet built; for row 0 there is none — but we will handle when j<i holds
                    }
                    if j < i {
                        self.u_val[self.u_diag_ix[j]]
                    } else {
                        Real::one()
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
                    } else if new_level < wlev[kpos] {
                        wlev[kpos] = new_level;
                    }
                    w.val[kpos] -= lij * self.u_val[q];
                }
                // store L(i,j) entry (value+level) later when we finalize L row
            }

            // Finalize L and U rows from work row with level <= k
            // Gather L (j<i)
            let mut l_pairs: Vec<(usize, Real, usize)> = w
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
            let mut u_pairs: Vec<(usize, Real, usize)> = w
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
                u_pairs.push((i, Real::zero(), 0));
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
                w.val[pos] = Real::from_real(vv[p]);
            }

            // eliminate for j in L pattern (already filtered by <=k)
            let ls = self.l_row[i];
            let le = self.l_row[i + 1];
            for pos in ls..le {
                let j = self.l_col[pos];
                let wij = if w.mark[j] >= 0 {
                    w.val[w.mark[j] as usize]
                } else {
                    Real::zero()
                };
                let djj = self.u_val[self.u_diag_ix[j]];
                let lij = if djj == Real::zero() {
                    Real::zero()
                } else {
                    wij / djj
                };
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
            let mut diag = Real::zero();
            for q in us..ue {
                let k = self.u_col[q];
                let v = if w.mark.get(k).copied().unwrap_or(-1) >= 0 {
                    w.val[w.mark[k] as usize]
                } else {
                    Real::zero()
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
        let mut inv_diag_u = vec![Real::zero(); n];

        // Row workspace
        let mut w = RowWork::new();
        w.ensure_size(n);
        let mut l_tmp: Vec<(usize, Real)> = Vec::new();
        let mut u_tmp: Vec<(usize, Real)> = Vec::new();

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
                    w.set(j, Real::from_real(v));
                    row_inf = row_inf.max(v.abs());
                }
            }
            let tau = params.droptol_abs + params.droptol_rel * row_inf;

            // Eliminate lower entries
            let mut lowers: Vec<usize> = w.iter().filter(|&(j, _)| j < i).map(|(j, _)| j).collect();
            lowers.sort_unstable();
            for &k in &lowers {
                let wk = w.get(k);
                if wk == Real::zero() {
                    continue;
                }
                let lik = wk * inv_diag_u[k];
                if params.early_drop && lik.abs() < tau {
                    w.set(k, Real::zero());
                    continue;
                }
                l_tmp.push((k, lik));
                w.set(k, Real::zero());

                let (u_cols_k, u_vals_k) = u_build.row(k);
                for (&j, &ukj) in u_cols_k.iter().zip(u_vals_k.iter()) {
                    if j <= k {
                        continue;
                    }
                    let newv: Real = w.get(j) - lik * ukj;
                    if params.early_drop && newv.abs() < tau {
                        w.set(j, Real::zero());
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
                u_tmp.push((i, Real::zero()));
            }

            // Cap L entries
            if params.p_l > 0 && l_tmp.len() > params.p_l {
                l_tmp.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                l_tmp.truncate(params.p_l);
            }

            // Cap U entries (excluding diagonal)
            let mut diag = Real::zero();
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
            let tau = params.pivot_tau;
            match params.pivot {
                PivotPolicy::Strict => {
                    if uii.abs() < tau {
                        return Err(KError::ZeroPivot(i));
                    }
                }
                PivotPolicy::Threshold => {
                    if uii.abs() < tau {
                        if uii == Real::zero() {
                            uii = Real::from_real(tau);
                        } else {
                            uii = uii * Real::from_real(tau / uii.abs());
                        }
                    }
                }
                PivotPolicy::DiagonalPerturbation => {
                    if uii.abs() < tau {
                        let direction = if uii == Real::zero() {
                            Real::one()
                        } else {
                            uii / Real::from_real(uii.abs())
                        };
                        uii += direction * Real::from_real(tau);
                    }
                }
            }
            u_tmp[diag_pos].1 = uii;
            inv_diag_u[i] = uii.inv();

            // Store rows into builders
            for &(k, v) in &l_tmp {
                l_build.push(i, k, v);
            }
            l_build.push(i, i, Real::one());
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

        self.tmp.resize(n, Real::zero());

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
                w.val[pos] = Real::from_real(vv[p]);
            }
            // eliminate across L pattern
            let ls = self.l_row[i];
            let le = self.l_row[i + 1];
            for pos in ls..le {
                let j = self.l_col[pos];
                let wij = if w.mark[j] >= 0 {
                    w.val[w.mark[j] as usize]
                } else {
                    Real::zero()
                };
                let djj = self.u_val[self.u_diag_ix[j]];
                let lij = if djj == Real::zero() {
                    Real::zero()
                } else {
                    wij / djj
                };
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
            let mut diag = Real::zero();
            for q in us..ue {
                let k = self.u_col[q];
                let v = if w.mark.get(k).copied().unwrap_or(-1) >= 0 {
                    w.val[w.mark[k] as usize]
                } else {
                    Real::zero()
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

#[cfg(not(feature = "complex"))]
impl Preconditioner for IluCsr {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let drop = 0.0; // use full numerical content by default
        let a: Arc<CsrMatrix<f64>> = csr_from_linop(op, drop)?;
        let mut conditioned = None;
        let a = if self.cfg.conditioning.is_active() {
            let mut local = (*a).clone();
            apply_csr_transforms("ILU/ILUT", &mut local, &self.cfg.conditioning)?;
            conditioned = Some(local);
            conditioned.as_ref().unwrap()
        } else {
            a.as_ref()
        };
        let sid = op.structure_id();
        let vid = op.values_id();

        let structure_changed = self.last_sid != Some(sid);
        let values_changed = self.last_vid != Some(vid);

        if structure_changed || !self.cfg.numeric_update_fixed {
            // compute permutation
            let perm = match self.cfg.reordering.kind {
                ReorderingKind::None => Permutation::identity(a.nrows()),
                ReorderingKind::Rcm => rcm_csr(&a),
                ReorderingKind::Amd => amd_csr(&a),
            };
            let a_perm = if self.cfg.reordering.symmetric {
                permute_csr_symmetric(&a, &perm)
            } else {
                // nonsymmetric not yet supported
                permute_csr_symmetric(&a, &perm)
            };
            self.perm = perm;
            self.factor_symbolic_and_numeric(&a_perm)?;
            self.build_levels_if_enabled();
            self.last_sid = Some(sid);
            self.last_vid = Some(vid);
            self.tmp.resize(a_perm.nrows(), Real::zero());
            Ok(())
        } else if values_changed {
            let a_perm = permute_csr_symmetric(&a, &self.perm);
            self.factor_numeric_only(&a_perm)?;
            self.last_vid = Some(vid);
            Ok(())
        } else {
            Ok(())
        }
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        self.apply_op_scalar(Op::NoTrans, x, y)
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::LocalOnly
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
        self.apply_op_scalar(op, x, y)
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

    fn required_format(&self) -> OpFormat {
        OpFormat::Csr
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: true,
            supports_conj_trans: false,
            is_spd: false,
            side_restriction: Some(PcSide::Left),
        }
    }
}

#[cfg(feature = "complex")]
impl Preconditioner for IluCsr {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        let csr = op
            .as_any()
            .downcast_ref::<CsrMatrix<S>>()
            .ok_or_else(|| {
                KError::Unsupported(
                    "IluCsr complex setup currently requires a CSR operator; non-CSR LinOp paths have no lossless complex ILU fallback".into(),
                )
            })?;

        let sid = op.structure_id();
        let vid = op.values_id();
        let structure_changed = self.last_sid != Some(sid);
        let values_changed = self.last_vid != Some(vid);

        if structure_changed || !self.cfg.numeric_update_fixed {
            let perm = match self.cfg.reordering.kind {
                ReorderingKind::None => Permutation::identity(csr.nrows()),
                ReorderingKind::Rcm => {
                    let a_real = CsrMatrix::from_csr(
                        csr.nrows(),
                        csr.ncols(),
                        csr.row_ptr().to_vec(),
                        csr.col_idx().to_vec(),
                        csr.values().iter().map(|v| v.real()).collect(),
                    );
                    rcm_csr(&a_real)
                }
                ReorderingKind::Amd => {
                    let a_real = CsrMatrix::from_csr(
                        csr.nrows(),
                        csr.ncols(),
                        csr.row_ptr().to_vec(),
                        csr.col_idx().to_vec(),
                        csr.values().iter().map(|v| v.real()).collect(),
                    );
                    amd_csr(&a_real)
                }
            };
            let a_perm = if self.cfg.reordering.symmetric {
                permute_csr_symmetric(csr, &perm)
            } else {
                permute_csr_symmetric(csr, &perm)
            };
            self.perm = perm;

            match self.cfg.kind {
                IluKind::Ilu0 | IluKind::Milu0 => self.factor_ilu0_complex(&a_perm)?,
                IluKind::Iluk { k } => {
                    if self.complex_force_degraded {
                        let a_real = CsrMatrix::from_csr(
                            a_perm.nrows(),
                            a_perm.ncols(),
                            a_perm.row_ptr().to_vec(),
                            a_perm.col_idx().to_vec(),
                            a_perm.values().iter().map(|v| v.real()).collect(),
                        );
                        self.native_complex_active = false;
                        self.complex_kernel_mode = IluComplexKernelMode::DegradedRealProjection;
                        self.factor_symbolic_and_numeric(&a_real)?;
                    } else {
                        self.factor_iluk_complex(&a_perm, k)?;
                    }
                }
                IluKind::Ilut { params } => {
                    if self.complex_force_degraded {
                        let a_real = CsrMatrix::from_csr(
                            a_perm.nrows(),
                            a_perm.ncols(),
                            a_perm.row_ptr().to_vec(),
                            a_perm.col_idx().to_vec(),
                            a_perm.values().iter().map(|v| v.real()).collect(),
                        );
                        self.native_complex_active = false;
                        self.complex_kernel_mode = IluComplexKernelMode::DegradedRealProjection;
                        self.factor_symbolic_and_numeric(&a_real)?;
                    } else {
                        self.factor_ilut_complex(&a_perm, &params)?;
                    }
                }
            }

            self.build_levels_if_enabled();
            self.last_sid = Some(sid);
            self.last_vid = Some(vid);
            self.tmp.resize(csr.nrows(), Real::zero());
            self.c_tmp.resize(csr.nrows(), S::zero());
            Ok(())
        } else if values_changed {
            self.update_numeric(op)
        } else {
            Ok(())
        }
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let n = self.n;
        if x.len() != n || y.len() != n {
            return Err(KError::InvalidInput(format!(
                "IluCsr::apply dimension mismatch: n={}, x.len()={}, y.len()={}",
                self.n,
                x.len(),
                y.len()
            )));
        }

        if self.native_complex_active {
            let mut w = x.to_vec();
            for i in 0..n {
                let mut s = w[i];
                for p in self.l_row[i]..self.l_row[i + 1] {
                    s -= self.c_l_val[p] * w[self.l_col[p]];
                }
                w[i] = s;
            }
            for i in (0..n).rev() {
                let mut s = w[i];
                for p in self.u_row[i]..self.u_row[i + 1] {
                    let j = self.u_col[p];
                    if j > i {
                        s -= self.c_u_val[p] * w[j];
                    }
                }
                w[i] = s / self.c_u_val[self.u_diag_ix[i]];
            }
            y.copy_from_slice(&w);
            Ok(())
        } else {
            let mut xr = vec![0.0; n];
            let mut xi = vec![0.0; n];
            let mut yr = vec![0.0; n];
            let mut yi = vec![0.0; n];
            for i in 0..n {
                xr[i] = x[i].real();
                xi[i] = x[i].imag();
            }
            self.apply_op_scalar(Op::NoTrans, &xr, &mut yr)?;
            self.apply_op_scalar(Op::NoTrans, &xi, &mut yi)?;
            for i in 0..n {
                y[i] = S::from_parts(yr[i], yi[i]);
            }
            Ok(())
        }
    }

    fn supports_numeric_update(&self) -> bool {
        self.cfg.numeric_update_fixed
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        if !self.cfg.numeric_update_fixed {
            return Err(KError::Unsupported("numeric update requires fixed pattern"));
        }
        if Some(op.structure_id()) != self.last_sid {
            return Err(KError::Unsupported("pattern changed; call update_symbolic"));
        }

        let csr = op.as_any().downcast_ref::<CsrMatrix<S>>().ok_or_else(|| {
            KError::Unsupported("IluCsr complex numeric update requires CSR".into())
        })?;
        let a_perm = permute_csr_symmetric(csr, &self.perm);
        match self.cfg.kind {
            IluKind::Ilu0 | IluKind::Milu0 => self.factor_ilu0_complex(&a_perm)?,
            IluKind::Iluk { k } => {
                if self.complex_force_degraded {
                    let a_real = CsrMatrix::from_csr(
                        a_perm.nrows(),
                        a_perm.ncols(),
                        a_perm.row_ptr().to_vec(),
                        a_perm.col_idx().to_vec(),
                        a_perm.values().iter().map(|v| v.real()).collect(),
                    );
                    self.native_complex_active = false;
                    self.complex_kernel_mode = IluComplexKernelMode::DegradedRealProjection;
                    self.factor_numeric_only(&a_real)?;
                } else {
                    self.factor_iluk_complex(&a_perm, k)?;
                }
            }
            IluKind::Ilut { params } => {
                if self.complex_force_degraded {
                    let a_real = CsrMatrix::from_csr(
                        a_perm.nrows(),
                        a_perm.ncols(),
                        a_perm.row_ptr().to_vec(),
                        a_perm.col_idx().to_vec(),
                        a_perm.values().iter().map(|v| v.real()).collect(),
                    );
                    self.native_complex_active = false;
                    self.complex_kernel_mode = IluComplexKernelMode::DegradedRealProjection;
                    self.factor_numeric_only(&a_real)?;
                } else {
                    self.factor_ilut_complex(&a_perm, &params)?;
                }
            }
        }
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.last_sid = None;
        self.last_vid = None;
        self.setup(op)
    }

    fn required_format(&self) -> OpFormat {
        OpFormat::Csr
    }

    fn capabilities(&self) -> PcCaps {
        PcCaps {
            supports_transpose: true,
            supports_conj_trans: false,
            is_spd: false,
            side_restriction: Some(PcSide::Left),
        }
    }
}

impl LocalPreconditioner<f64> for IluCsr {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply_local(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "IluCsr::apply_local dimension mismatch: n={}, x.len()={}, y.len()={}",
                self.n,
                x.len(),
                y.len()
            )));
        }

        self.apply_op_scalar(Op::NoTrans, x, y)
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for IluCsr {
    // Use the *complex* scalar type from the algebra prelude, not the local f64 alias.
    type Scalar = crate::algebra::prelude::S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        // IluCsr already implements the real Preconditioner
        crate::preconditioner::Preconditioner::dims(self)
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[Self::Scalar],
        y: &mut [Self::Scalar],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        let _ = scratch;
        self.apply(side, x, y)
    }
}

impl IluCsr {
    #[cfg(feature = "complex")]
    pub fn complex_kernel_mode(&self) -> IluComplexKernelMode {
        self.complex_kernel_mode
    }

    #[cfg(feature = "complex")]
    pub fn set_complex_force_degraded(&mut self, on: bool) {
        self.complex_force_degraded = on;
    }

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
    pub(crate) fn l_val(&self) -> &[Real] {
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
    pub(crate) fn u_val(&self) -> &[Real] {
        &self.u_val
    }
    #[inline]
    pub(crate) fn u_diag_ix(&self) -> &[usize] {
        &self.u_diag_ix
    }
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn tmp(&self) -> &[Real] {
        &self.tmp
    }
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn tmp_mut(&mut self) -> &mut [Real] {
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

    fn apply_op_scalar(&self, op: Op, x: &[Real], y: &mut [Real]) -> Result<(), KError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "IluCsr::apply dimension mismatch: n={}, x.len()={}, y.len()={}",
                self.n,
                x.len(),
                y.len()
            )));
        }
        let mut x_perm = vec![Real::zero(); self.n];
        let mut y_perm = vec![Real::zero(); self.n];
        self.perm.apply_vec(x, &mut x_perm);
        match op {
            Op::NoTrans => {
                if self.cfg.level_sched {
                    tri_solve::tri_solve_level_scheduled(self, &x_perm, &mut y_perm)
                } else {
                    tri_solve::tri_solve_serial(self, &x_perm, &mut y_perm)
                }
            }
            Op::Trans | Op::ConjTrans => {
                let ut = self
                    .ut
                    .get_or_init(|| transpose_csr(self.n, &self.u_row, &self.u_col, &self.u_val));
                let lt = self
                    .lt
                    .get_or_init(|| transpose_csr(self.n, &self.l_row, &self.l_col, &self.l_val));
                tri_solve::tri_solve_transpose_serial(
                    self,
                    &ut.0,
                    &ut.1,
                    &ut.2,
                    &lt.0,
                    &lt.1,
                    &lt.2,
                    &x_perm,
                    &mut y_perm,
                )
            }
        }?;
        self.perm.apply_vec_t(&y_perm, y);
        Ok(())
    }
}

fn transpose_csr(
    n: usize,
    row: &[usize],
    col: &[usize],
    val: &[Real],
) -> (Vec<usize>, Vec<usize>, Vec<Real>) {
    let nnz = col.len();
    let mut t_row = vec![0usize; n + 1];
    for &j in col {
        t_row[j + 1] += 1;
    }
    for i in 0..n {
        t_row[i + 1] += t_row[i];
    }
    let mut t_col = vec![0usize; nnz];
    let mut t_val = vec![Real::zero(); nnz];
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
