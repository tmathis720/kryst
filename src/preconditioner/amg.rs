#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::KError;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::matrix::{convert::csr_from_linop, sparse::CsrMatrix};
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// New sparse SA/RS submodules
mod strength;
mod coarsen;
mod prolong;
mod rap_ops;
mod coarse_solver;
mod row_filter;

use strength::Strength;
use coarsen::{AggAlgo, build_aggregates};
use prolong::{TentativeP, Pcsr, smooth_tentative_sa, smooth_sa_values_only};
use rap_ops::{CsrPattern, rap_symbolic, rap_numeric};
use coarse_solver::{CoarseSolve, CoarseSolver, CoarseDenseLu};
use row_filter::{RowFilter, apply_filter_to_csr_values_in_place};

// ===== Public enums (kept compatible with your old file) =====================

/// Coarsening strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarsenType {
    RS,
    HMIS,
    PMIS,
    Falgout,
}

/// Interpolation strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpType {
    Classical,
    Direct,
    Multipass,
    Extended,
    Standard,
}

/// Relaxation/smoothing choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelaxType {
    Jacobi,
    GaussSeidel,
    GaussSeidelBackward,
    SymmetricGaussSeidel,
    HybridGaussSeidel,
    L1Jacobi,
    Chebyshev,
}

/// Per-phase relaxation controls mirroring Boomer semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelaxPhase {
    Fine = 0,
    Down = 1,
    Up = 2,
    Coarsest = 3,
}

enum RelaxWhere {
    Pre,
    Post,
}

impl RelaxPhase {
    #[inline]
    pub fn ix(self) -> usize {
        self as usize
    }
    pub const ALL: [RelaxPhase; 4] = [
        RelaxPhase::Fine,
        RelaxPhase::Down,
        RelaxPhase::Up,
        RelaxPhase::Coarsest,
    ];
}

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
pub static RELAX_CALL_COUNTS: [AtomicUsize; 4] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
#[cfg(test)]
pub fn reset_relax_counts() {
    for c in &RELAX_CALL_COUNTS {
        c.store(0, Ordering::SeqCst);
    }
}
#[cfg(test)]
pub fn get_relax_counts() -> [usize; 4] {
    let mut out = [0; 4];
    for (i, c) in RELAX_CALL_COUNTS.iter().enumerate() {
        out[i] = c.load(Ordering::SeqCst);
    }
    out
}

// ===== Config + Builder ======================================================

#[derive(Clone, Debug)]
pub struct AMGConfig {
    pub max_levels: usize,            // HYPRE default: 25
    pub strong_threshold: f64,        // HYPRE default: 0.25
    pub coarse_threshold: usize,      // HYPRE default: 9
    pub max_coarse_size: usize,       // HYPRE default: 9
    pub min_coarse_size: usize,       // HYPRE minimum: 1
    pub truncation_factor: f64,       // 0 => no truncation
    pub max_elements_per_row: usize,  // 0 => unlimited
    pub interpolation_truncation: f64,
    pub rap_truncation_abs: f64,
    pub rap_max_elements_per_row: usize,
    pub keep_pivot_in_rap: bool,
    pub grid_relax_type: [RelaxType; 4],   // [Fine, Down, Up, Coarsest]
    pub num_grid_sweeps: [usize; 4],      // [Fine, Down, Up, Coarsest]
    // legacy shims
    pub pre_sweeps: usize,            // HYPRE default: 1
    pub post_sweeps: usize,           // HYPRE default: 1
    pub coarsen_type: CoarsenType,    // HYPRE default: HMIS
    pub interp_type: InterpType,      // robust: Extended/Standard
    pub relax_type: RelaxType,        // HYPRE default: Gauss-Seidel (we implement Jacobi)
    pub logging_level: usize,
    pub print_level: usize,
    pub tolerance: f64,               // for coarse direct solve (CG)
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub ieee_checks: bool,
    pub optimize_workspace: bool,
    pub jacobi_omega: f64,
    pub chebyshev_degree: usize,
    pub drop_tol: f64,                // NEW: used for dense->CSR conversion
    // New SA/RS controls
    pub normalize_strength: bool,
    pub coarse_solve: CoarseSolve,
    pub ilu_drop_tol: f64,
    pub ilu_fill_per_row: usize,
    pub max_operator_complexity: Option<f64>,
}

impl Default for AMGConfig {
    fn default() -> Self {
        let mut cfg = Self {
            max_levels: 25,
            strong_threshold: 0.25,
            coarse_threshold: 9,
            max_coarse_size: 9,
            min_coarse_size: 1,
            truncation_factor: 0.0,
            max_elements_per_row: 0,
            interpolation_truncation: 0.0,
            rap_truncation_abs: 0.0,
            rap_max_elements_per_row: 0,
            keep_pivot_in_rap: true,
            grid_relax_type: [RelaxType::GaussSeidel; 4],
            num_grid_sweeps: [1; 4],
            pre_sweeps: 1,
            post_sweeps: 1,
            coarsen_type: CoarsenType::HMIS,
            interp_type: InterpType::Extended,
            relax_type: RelaxType::GaussSeidel,
            logging_level: 0,
            print_level: 0,
            tolerance: 1e-6,
            max_iterations: 20,
            min_iterations: 0,
            ieee_checks: true,
            optimize_workspace: true,
            jacobi_omega: 2.0 / 3.0,
            chebyshev_degree: 0,
            drop_tol: 1e-12,
            normalize_strength: true,
            coarse_solve: CoarseSolve::CG,
            ilu_drop_tol: 1e-2,
            ilu_fill_per_row: 0,
            max_operator_complexity: None,
        };
        cfg.grid_relax_type = [
            cfg.relax_type,
            cfg.relax_type,
            cfg.relax_type,
            RelaxType::GaussSeidel,
        ];
        cfg.num_grid_sweeps = [cfg.pre_sweeps, cfg.pre_sweeps, cfg.post_sweeps, 1];
        cfg
    }
}

/// Builder for `AMG` (preserves old chaining API).
pub struct AMGBuilder {
    cfg: AMGConfig,
}

impl AMGBuilder {
    pub fn new() -> Self {
        Self { cfg: AMGConfig::default() }
    }
    pub fn max_levels(mut self, v: usize) -> Self { self.cfg.max_levels = v; self }
    pub fn strong_threshold(mut self, v: f64) -> Self { self.cfg.strong_threshold = v; self }
    pub fn coarse_threshold(mut self, v: usize) -> Self { self.cfg.coarse_threshold = v; self }
    pub fn max_coarse_size(mut self, v: usize) -> Self { self.cfg.max_coarse_size = v; self }
    pub fn min_coarse_size(mut self, v: usize) -> Self { self.cfg.min_coarse_size = v; self }
    pub fn truncation_factor(mut self, v: f64) -> Self { self.cfg.truncation_factor = v; self }
    pub fn interpolation_drop_abs(mut self, v: f64) -> Self { self.cfg.interpolation_truncation = v; self }
    pub fn interpolation_cap(mut self, k: usize) -> Self { self.cfg.max_elements_per_row = k; self }
    pub fn rap_drop_abs(mut self, v: f64) -> Self { self.cfg.rap_truncation_abs = v; self }
    pub fn rap_cap(mut self, k: usize) -> Self { self.cfg.rap_max_elements_per_row = k; self }
    pub fn keep_pivot_in_rap(mut self, yes: bool) -> Self { self.cfg.keep_pivot_in_rap = yes; self }
    pub fn interpolation_truncation(self, v: f64) -> Self { self.interpolation_drop_abs(v) }
    pub fn smoothing_sweeps(mut self, pre: usize, post: usize) -> Self {
        self.cfg.pre_sweeps = pre;
        self.cfg.post_sweeps = post;
        self.cfg.num_grid_sweeps[RelaxPhase::Fine.ix()] = pre;
        self.cfg.num_grid_sweeps[RelaxPhase::Down.ix()] = pre;
        self.cfg.num_grid_sweeps[RelaxPhase::Up.ix()] = post;
        // leave Coarsest as-is
        self
    }
    pub fn coarsening_type(mut self, v: CoarsenType) -> Self { self.cfg.coarsen_type = v; self }
    pub fn interpolation_type(mut self, v: InterpType) -> Self { self.cfg.interp_type = v; self }
    pub fn relaxation_type(mut self, v: RelaxType) -> Self {
        self.cfg.relax_type = v;
        for ph in RelaxPhase::ALL {
            self.cfg.grid_relax_type[ph.ix()] = v;
        }
        self.cfg.grid_relax_type[RelaxPhase::Coarsest.ix()] = RelaxType::GaussSeidel;
        self
    }
    pub fn grid_relax_type(mut self, phase: RelaxPhase, t: RelaxType) -> Self {
        self.cfg.grid_relax_type[phase.ix()] = t;
        self
    }
    pub fn num_grid_sweeps(mut self, phase: RelaxPhase, k: usize) -> Self {
        self.cfg.num_grid_sweeps[phase.ix()] = k;
        self
    }
    pub fn grid_relax_type_all(mut self, t: RelaxType) -> Self {
        for ph in RelaxPhase::ALL {
            self.cfg.grid_relax_type[ph.ix()] = t;
        }
        self
    }
    pub fn num_grid_sweeps_all(mut self, k: usize) -> Self {
        for ph in RelaxPhase::ALL {
            self.cfg.num_grid_sweeps[ph.ix()] = k;
        }
        self
    }
    pub fn enable_logging(mut self) -> Self { self.cfg.logging_level = 1; self }
    pub fn logging_level(mut self, lvl: usize) -> Self { self.cfg.logging_level = lvl; self }
    pub fn enable_printing(mut self) -> Self { self.cfg.print_level = 1; self }
    pub fn print_level(mut self, lvl: usize) -> Self { self.cfg.print_level = lvl; self }
    pub fn jacobi_omega(mut self, w: f64) -> Self { self.cfg.jacobi_omega = w; self }
    pub fn chebyshev_degree(mut self, k: usize) -> Self { self.cfg.chebyshev_degree = k; self }
    pub fn drop_tolerance(mut self, t: f64) -> Self { self.cfg.drop_tol = t; self }

    pub fn build(self, _matrix: &Mat<f64>) -> Result<AMG, KError> {
        Ok(AMG::with_config(self.cfg))
    }
}

impl Default for AMGBuilder { fn default() -> Self { Self::new() } }

// ===== Workspace, levels & hierarchy ========================================

fn validate_relax_policy(cfg: &AMGConfig, coarse_solver: CoarseSolve) -> Result<(), KError> {
    if !matches!(coarse_solver, CoarseSolve::CG | CoarseSolve::ILU) {
        if cfg.num_grid_sweeps[RelaxPhase::Coarsest.ix()] != 0 {
            return Err(KError::InvalidInput(
                "num_grid_sweeps[Coarsest] must be 0 when coarse_solve is DirectDense".into(),
            ));
        }
    }

    for (i, &rt) in cfg.grid_relax_type.iter().enumerate() {
        match rt {
            RelaxType::Jacobi => {}
            _ => {
                return Err(KError::InvalidInput(format!(
                    "RelaxType {:?} not yet supported (phase index {}); choose Jacobi for now",
                    rt, i
                )));
            }
        }
    }

    for (i, &k) in cfg.num_grid_sweeps.iter().enumerate() {
        if i != RelaxPhase::Coarsest.ix() && k == 0 {
            return Err(KError::InvalidInput(format!(
                "num_grid_sweeps for phase {} must be >= 1",
                i
            )));
        }
    }
    Ok(())
}

fn validate_truncation_and_caps(cfg: &AMGConfig) -> Result<(), KError> {
    if !(0.0..1.0).contains(&cfg.truncation_factor) {
        return Err(KError::InvalidInput("truncation_factor must satisfy 0 ≤ τ_rel < 1".into()));
    }
    if cfg.interpolation_truncation < 0.0 || cfg.rap_truncation_abs < 0.0 {
        return Err(KError::InvalidInput("absolute drop tolerances must be ≥ 0".into()));
    }
    Ok(())
}

#[derive(Debug)]
struct AMGWorkspace {
    temp: Vec<f64>,
    work: Vec<f64>,
    residual: Vec<f64>,
    coarse_rhs: Vec<f64>,
    fine_corr: Vec<f64>,
}

impl AMGWorkspace {
    fn new(cap: usize) -> Self {
        Self {
            temp: vec![0.0; cap],
            work: vec![0.0; cap],
            residual: vec![0.0; cap],
            coarse_rhs: vec![0.0; cap],
            fine_corr: vec![0.0; cap],
        }
    }
    fn ensure(&mut self, n: usize) {
        let grow = |v: &mut Vec<f64>, n: usize| if v.len() < n { v.resize(n, 0.0) };
        grow(&mut self.temp, n);
        grow(&mut self.work, n);
        grow(&mut self.residual, n);
        grow(&mut self.coarse_rhs, n);
        grow(&mut self.fine_corr, n);
    }
}

#[derive(Clone)]
struct AMGLevel {
    /// A_l (coarse operator at this level, l = 0 is finest)
    a: CsrMatrix<f64>,
    /// P_l (interpolation to next coarser level)
    p: CsrMatrix<f64>,
    /// R_l (restriction to next coarser level)
    r: CsrMatrix<f64>,
    /// diag(A_l)^{-1}
    diag_inv: Vec<f64>,
    /// fine->coarse aggregate id used to rebuild P values (SA numeric refresh)
    agg_of: Vec<usize>,
    /// Mapping from P entry index -> index in R (transpose) values array
    p2r_pos: Vec<usize>,
    /// Symbolic pattern for A_{l+1}
    a_next_pat: Option<CsrPattern>,
}

#[derive(Clone)]
struct RelaxPolicy {
    kind: [RelaxType; 4],
    sweeps: [usize; 4],
    omega: f64,
}

#[derive(Clone)]
struct AmgHierarchy {
    levels: Vec<AMGLevel>, // 0..L ; L is coarsest
    policy: RelaxPolicy,
    coarse_solve: CoarseSolve,
}

impl AmgHierarchy {
    fn finest(&self) -> &AMGLevel { &self.levels[0] }
    fn coarsest_ix(&self) -> usize { self.levels.len() - 1 }
}

// ===== Main AMG object =======================================================

pub struct AMG {
    csr: Option<Arc<CsrMatrix<f64>>>,
    state: Option<AmgHierarchy>,
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,
    cfg: AMGConfig,
    stats: Option<AmgStats>,
}

impl Default for AMG {
    fn default() -> Self {
        Self {
            csr: None,
            state: None,
            last_sid: None,
            last_vid: None,
            cfg: AMGConfig::default(),
            stats: None,
        }
    }
}

impl AMG {
    pub fn new(_matrix: &Mat<f64>, _max_levels: usize, _coarsening_threshold: f64) -> Self {
        AMG::default()
    }
    pub fn builder() -> AMGBuilder { AMGBuilder::new() }
    pub fn with_config(cfg: AMGConfig) -> Self {
        Self { cfg, ..Default::default() }
    }

    // ---- Setup paths --------------------------------------------------------

    fn build_symbolic(&mut self, fine: &CsrMatrix<f64>) -> Result<(), KError> {
        // Build the full hierarchy from scratch (symbolic + numeric)
        let hier = build_hierarchy(fine, &self.cfg)?;
        self.state = Some(hier);
        Ok(())
    }

    fn refresh_numeric(&mut self, fine: &CsrMatrix<f64>) -> Result<(), KError> {
        // Numeric refresh: keep structure; recompute diag, P values, R values and A_l via RAP.
        if self.state.is_none() {
            return self.build_symbolic(fine);
        }
        let mut h = self.state.clone().unwrap();
        if h.levels.is_empty() {
            return self.build_symbolic(fine);
        }

        // Update finest A_0 and diag(A_0)^{-1}
        h.levels[0].a = fine.clone();
        h.levels[0].diag_inv = diag_inv_from_csr(&h.levels[0].a)?;

        // Recompute P_l values, R_l values, and A_{l+1} values using fixed patterns
        for l in 0..h.coarsest_ix() {
            // Recompute P_l values in-place using SA smoother with fixed pattern
            let tp = TentativeP { agg_of: h.levels[l].agg_of.clone(), n_coarse: h.levels[l+1].a.nrows() };
            let d_inv = &h.levels[l].diag_inv;
            // Recompute P values under fixed pattern without borrowing P mutably during computation
            let pr = h.levels[l].p.row_ptr().to_vec();
            let pc = h.levels[l].p.col_idx().to_vec();
            let mut p_new_vals = vec![0.0f64; pc.len()];
            smooth_sa_values_only(
                &h.levels[l].a,
                d_inv,
                &tp,
                self.cfg.jacobi_omega,
                &pr,
                &pc,
                &mut p_new_vals,
            )?;
            h.levels[l].p.values_mut().copy_from_slice(&p_new_vals);
            // Update R values from P via precomputed transpose mapping
            {
                let pvals = h.levels[l].p.values().to_vec();
                let p2r = h.levels[l].p2r_pos.clone();
                let rvalsm = h.levels[l].r.values_mut();
                for (pi, &ri) in p2r.iter().enumerate() {
                    rvalsm[ri] = pvals[pi];
                }
            }
            // Recompute A_{l+1} values by RAP numeric using fixed pattern
            if let Some(ref pat) = h.levels[l].a_next_pat {
                let nnz = pat.col_idx.len();
                let mut vals = vec![0.0; nnz];
                rap_numeric(
                    pat,
                    &h.levels[l].r,
                    &h.levels[l].a,
                    &h.levels[l].p,
                    &mut vals,
                );
                {
                    let mut rf = |row: usize| RowFilter {
                        tau_abs: self.cfg.rap_truncation_abs,
                        tau_rel: self.cfg.truncation_factor,
                        k_max: self.cfg.rap_max_elements_per_row,
                        must_keep: if self.cfg.keep_pivot_in_rap { Some(row) } else { None },
                    };
                    apply_filter_to_csr_values_in_place(pat.nrows, &pat.row_ptr, &pat.col_idx, &mut vals, &mut rf);
                }
                h.levels[l + 1].a = CsrMatrix::from_csr(
                    h.levels[l + 1].a.nrows(),
                    h.levels[l + 1].a.ncols(),
                    pat.row_ptr.clone(),
                    pat.col_idx.clone(),
                    vals,
                );
                h.levels[l + 1].diag_inv = diag_inv_from_csr(&h.levels[l + 1].a)?;
            } else {
                // Safety fallback: full RAP (structure + values)
                let a_coarse = rap(&h.levels[l].r, &h.levels[l].a, &h.levels[l].p)?;
                h.levels[l + 1].diag_inv = diag_inv_from_csr(&a_coarse)?;
                h.levels[l + 1].a = a_coarse;
            }
        }

        self.state = Some(h);
        Ok(())
    }

    // ---- Smoother -----------------------------------------------------------

    fn jacobi_smooth_sparse(
        omega: f64,
        a: &CsrMatrix<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        iters: usize,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if iters == 0 { return Ok(()); }
        let n = a.nrows();
        if diag_inv.len() != n || r.len() != n || z.len() != n {
            return Err(KError::InvalidInput("Jacobi: dimension mismatch".into()));
        }
        ws.ensure(n);
        ws.temp[..n].copy_from_slice(z);

        for _ in 0..iters {
            // work = A * temp
            a.spmv_scaled(1.0, &ws.temp[..n], 0.0, &mut ws.work[..n])?;
            // temp += omega * D^{-1} * (r - work)
            #[cfg(feature = "rayon")]
            ws.temp[..n].par_iter_mut().enumerate().for_each(|(i, zi)| {
                *zi += omega * diag_inv[i] * (r[i] - ws.work[i]);
            });
            #[cfg(not(feature = "rayon"))]
            for i in 0..n {
                ws.temp[i] += omega * diag_inv[i] * (r[i] - ws.work[i]);
            }
        }
        z.copy_from_slice(&ws.temp[..n]);
        Ok(())
    }

    // single dispatch point for all relaxation strategies
    fn apply_relax(
        pol: &RelaxPolicy,
        phase: RelaxPhase,
        _where: RelaxWhere,
        a: &CsrMatrix<f64>,
        diag_inv: &[f64],
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        let k = pol.sweeps[phase.ix()];
        if k == 0 {
            return Ok(());
        }
        #[cfg(test)]
        {
            RELAX_CALL_COUNTS[phase.ix()].fetch_add(1, Ordering::SeqCst);
        }
        match pol.kind[phase.ix()] {
            RelaxType::Jacobi => Self::jacobi_smooth_sparse(pol.omega, a, diag_inv, rhs, sol, k, ws),
            other => Err(KError::InvalidInput(format!("RelaxType {:?} not yet supported", other))),
        }
    }

    // ---- V-cycle ------------------------------------------------------------

    fn v_cycle(
        &self,
        level: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        let h = self.state.as_ref().ok_or_else(|| KError::InvalidInput("AMG not set up".into()))?;
        let lc = h.coarsest_ix();

        let a = &h.levels[level].a;
        let d = &h.levels[level].diag_inv;
        let pol = &h.policy;

        if level == lc {
            // Coarsest: choose solver, with heuristic to use dense when tiny
            let use_dense = matches!(h.coarse_solve, CoarseSolve::DirectDense)
                || a.nrows() <= self.cfg.max_coarse_size;
            if use_dense {
                let mut solver = CoarseDenseLu::new();
                solver.setup(a)?;
                solver.solve(rhs, sol)?;
            } else {
                match h.coarse_solve {
                    CoarseSolve::CG | CoarseSolve::ILU => {
                        // Fallback to CG; ILU path can be added later
                        cg_sparse(a, rhs, sol, self.cfg.tolerance, a.nrows().min(50))?;
                    }
                    CoarseSolve::DirectDense => unreachable!(),
                }
            }
            return Ok(());
        }

        let n = a.nrows();
        ws.ensure(n);

        // Pre-smooth
        let phase_pre = if level == 0 { RelaxPhase::Fine } else { RelaxPhase::Down };
        Self::apply_relax(pol, phase_pre, RelaxWhere::Pre, a, d, rhs, sol, ws)?;

        // residual = rhs - A * sol
        a.spmv_scaled(1.0, sol, 0.0, &mut ws.work[..n])?;
        #[cfg(feature = "rayon")]
        ws.residual[..n].par_iter_mut().enumerate().for_each(|(i, ri)| {
            *ri = rhs[i] - ws.work[i];
        });
        #[cfg(not(feature = "rayon"))]
        for i in 0..n { ws.residual[i] = rhs[i] - ws.work[i]; }

        // r_c = R * residual
        let r = &h.levels[level].r;
        let p = &h.levels[level].p;
        let nc = h.levels[level + 1].a.nrows();

        // Take ownership of the workspace coarse buffer to avoid cloning and
        // to prevent simultaneous immutable and mutable borrows of `ws`.
        let mut local_coarse = std::mem::take(&mut ws.coarse_rhs);
        local_coarse.resize(nc, 0.0);
        // Fill local buffer with R * residual
        r.spmv_scaled(1.0, &ws.residual[..n], 0.0, &mut local_coarse[..nc])?;

        // recurse
        let mut zc = vec![0.0; nc];
        self.v_cycle(level + 1, &local_coarse[..nc], &mut zc, ws)?;
        // Restore workspace buffer for reuse
        ws.coarse_rhs = local_coarse;

        // fine_corr = P * zc
        ws.fine_corr[..n].fill(0.0);
        p.spmv_scaled(1.0, &zc, 0.0, &mut ws.fine_corr[..n])?;

        // sol += fine_corr
        #[cfg(feature = "rayon")]
        sol.par_iter_mut().enumerate().for_each(|(i, zi)| { *zi += ws.fine_corr[i]; });
        #[cfg(not(feature = "rayon"))]
        for i in 0..n { sol[i] += ws.fine_corr[i]; }

        // Post-smooth
        let phase_post = if level == 0 { RelaxPhase::Fine } else { RelaxPhase::Up };
        Self::apply_relax(pol, phase_post, RelaxWhere::Post, a, d, rhs, sol, ws)?;
        Ok(())
    }

    // Convenience to avoid trait ambiguity in examples
    pub fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        Preconditioner::apply(self, side, x, y)
    }
    pub fn stats(&self) -> Option<AmgStats> { self.stats.clone() }
}

// ===== Preconditioner trait (new API) =======================================

impl Preconditioner for AMG {
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        validate_relax_policy(&self.cfg, self.cfg.coarse_solve)?;
        validate_truncation_and_caps(&self.cfg)?;
        // Convert to CSR via the new matrix layer entry point.
        let csr = csr_from_linop(op, self.cfg.drop_tol)?;
        let sid = op.structure_id();
        let vid = op.values_id();

        match (self.last_sid, self.last_vid) {
            (None, _) => self.build_symbolic(&csr)?,
            (Some(old_sid), _) if old_sid != sid => self.build_symbolic(&csr)?,
            (Some(_), Some(old_vid)) if old_vid != vid => self.refresh_numeric(&csr)?,
            _ => {}
        }

        self.csr = Some(csr);
        self.last_sid = Some(sid);
        self.last_vid = Some(vid);
        // compute stats if available
        if let Some(h) = &self.state {
            self.stats = Some(AmgStats::from_hierarchy(h));
        }
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        if r.len() != z.len() {
            return Err(KError::InvalidInput(format!(
                "AMG.apply: r/z size mismatch: {} vs {}", r.len(), z.len()
            )));
        }
        let h = self.state.as_ref().ok_or_else(|| KError::InvalidInput("AMG not set up".into()))?;
        if h.levels.is_empty() {
            // Fallback Jacobi with diagonal of input matrix if hierarchy missing
            let a = self.csr.as_ref().ok_or_else(|| KError::InvalidInput("AMG not set up".into()))?;
            let d = diag_inv_from_csr(a)?;
            let mut ws = AMGWorkspace::new(r.len());
            Self::jacobi_smooth_sparse(self.cfg.jacobi_omega, a, &d, r, z, 10, &mut ws)
        } else {
            let mut ws = AMGWorkspace::new(h.finest().a.nrows());
            z.fill(0.0);
            self.v_cycle(0, r, z, &mut ws)
        }
    }

    fn supports_numeric_update(&self) -> bool { true }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        validate_truncation_and_caps(&self.cfg)?;
        let csr = csr_from_linop(op, self.cfg.drop_tol)?;
        self.refresh_numeric(&csr)?;
        self.csr = Some(csr);
        self.last_vid = Some(op.values_id());
        if let Some(h) = &self.state { self.stats = Some(AmgStats::from_hierarchy(h)); }
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        validate_relax_policy(&self.cfg, self.cfg.coarse_solve)?;
        validate_truncation_and_caps(&self.cfg)?;
        let csr = csr_from_linop(op, self.cfg.drop_tol)?;
        self.build_symbolic(&csr)?;
        self.csr = Some(csr);
        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());
        if let Some(h) = &self.state { self.stats = Some(AmgStats::from_hierarchy(h)); }
        Ok(())
    }
}

// ===== Legacy adapter (unchanged external signature) ========================

impl crate::preconditioner::legacy::Preconditioner<Mat<f64>, Vec<f64>> for AMG {
    fn setup(&mut self, a: &Mat<f64>) -> Result<(), KError> {
        Preconditioner::setup(self, a)
    }
    fn apply(&self, side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        Preconditioner::apply(self, side, r.as_slice(), z.as_mut_slice())
    }
}

// ===== Hierarchy construction (symbolic + numeric) ==========================

fn build_hierarchy(fine: &CsrMatrix<f64>, cfg: &AMGConfig) -> Result<AmgHierarchy, KError> {
    let mut levels: Vec<AMGLevel> = Vec::with_capacity(cfg.max_levels);
    let mut a_cur = fine.clone();

    // Level 0 (finest)
    let l0 = AMGLevel {
        diag_inv: diag_inv_from_csr(&a_cur)?,
        a: a_cur.clone(),
        p: CsrMatrix::identity(a_cur.nrows()), // placeholder; updated after coarsening step
        r: CsrMatrix::identity(a_cur.nrows()),
        agg_of: (0..a_cur.nrows()).collect(), // identity initially
        p2r_pos: Vec::new(),
        a_next_pat: None,
    };
    levels.push(l0);

    // Drive coarsening: build levels 0..L (inclusive L is coarsest)
    for _level in 0..cfg.max_levels {
        let n = a_cur.nrows();
        if n <= cfg.coarse_threshold || n <= cfg.min_coarse_size { break; }

        // 1) Strength of connection (sparse)
        let s = Strength::from_csr(&a_cur, cfg.strong_threshold, cfg.normalize_strength);
        // 2) Aggregates
        let agg = build_aggregates(&s, match cfg.coarsen_type { CoarsenType::RS => AggAlgo::RSGreedy, CoarsenType::HMIS => AggAlgo::HMIS, CoarsenType::PMIS => AggAlgo::PMIS, CoarsenType::Falgout => AggAlgo::HMIS });
        let tp = TentativeP { n_coarse: 1 + agg.iter().copied().max().unwrap_or(0), agg_of: agg.clone() };
        // 3) Smoothed aggregation P (sparse-only)
        let p_csr: Pcsr = smooth_tentative_sa(
            &a_cur,
            &diag_inv_from_csr(&a_cur)?,
            &tp,
            cfg.jacobi_omega,
            cfg.interpolation_truncation,
            cfg.max_elements_per_row,
            cfg.truncation_factor,
        );
        // Build owned CSR for P and R
        let p = CsrMatrix::from_csr(p_csr.m, p_csr.n, p_csr.row_ptr.clone(), p_csr.col_idx.clone(), p_csr.vals.clone());
        // R = P^T pattern and values
        let (r_row_ptr, r_col_idx, r_vals, p2r_pos) = transpose_csr_with_pos(&p_csr);
        let r = CsrMatrix::from_csr(p_csr.n, p_csr.m, r_row_ptr.clone(), r_col_idx.clone(), r_vals.clone());

        // 4) Coarse operator A_c symbolic and numeric
        let pat = rap_symbolic(&r, &a_cur, &p);
        let mut a_coarse_vals = vec![0.0; pat.col_idx.len()];
        rap_numeric(&pat, &r, &a_cur, &p, &mut a_coarse_vals);
        {
            let mut rf = |row: usize| RowFilter {
                tau_abs: cfg.rap_truncation_abs,
                tau_rel: cfg.truncation_factor,
                k_max: cfg.rap_max_elements_per_row,
                must_keep: if cfg.keep_pivot_in_rap { Some(row) } else { None },
            };
            apply_filter_to_csr_values_in_place(pat.nrows, &pat.row_ptr, &pat.col_idx, &mut a_coarse_vals, &mut rf);
        }
        let a_coarse = CsrMatrix::from_csr(pat.nrows, pat.ncols, pat.row_ptr.clone(), pat.col_idx.clone(), a_coarse_vals);
        let diag_inv_coarse = diag_inv_from_csr(&a_coarse)?;

        // Replace previous temporary P/R by actual inter-level transfers and agg mapping
        if let Some(prev) = levels.last_mut() {
            prev.p = p.clone();
            prev.r = r.clone();
            prev.agg_of = tp.agg_of.clone();
            prev.p2r_pos = p2r_pos;
            prev.a_next_pat = Some(pat.clone());
        }

        // Next level (coarser)
        a_cur = a_coarse.clone();
        levels.push(AMGLevel {
            a: a_coarse,
            p: CsrMatrix::identity(a_cur.nrows()),
            r: CsrMatrix::identity(a_cur.nrows()),
            diag_inv: diag_inv_coarse,
            agg_of: (0..a_cur.nrows()).collect(),
            p2r_pos: Vec::new(),
            a_next_pat: None,
        });

        if a_cur.nrows() >= n { break; } // stalled
        if a_cur.nrows() <= cfg.max_coarse_size { break; }
        if let Some(limit) = cfg.max_operator_complexity {
            let oc = operator_complexity_estimate(&levels);
            if oc > limit { break; }
        }
    }

    Ok(AmgHierarchy {
        policy: RelaxPolicy {
            kind: cfg.grid_relax_type,
            sweeps: cfg.num_grid_sweeps,
            omega: cfg.jacobi_omega,
        },
        coarse_solve: cfg.coarse_solve,
        levels,
    })
}

// ===== Sparse utilities (local; avoid dense on hot path) ====================

fn diag_inv_from_csr(a: &CsrMatrix<f64>) -> Result<Vec<f64>, KError> {
    let n = a.nrows();
    let mut d = vec![0.0; n];
    for i in 0..n {
        let rs = a.row_ptr()[i];
        let re = a.row_ptr()[i + 1];
        let mut aii = 0.0;
        for p in rs..re {
            if a.col_idx()[p] == i {
                aii = a.values()[p];
                break;
            }
        }
        if aii.abs() < 1e-14 {
            return Err(KError::SolveError(format!("near-zero diagonal at row {}", i)));
        }
        d[i] = 1.0 / aii;
    }
    Ok(d)
}

/// CSR * CSR using per-row growing maps (Gustavson-style, simple).
fn csr_mul(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> Result<CsrMatrix<f64>, KError> {
    if a.ncols() != b.nrows() {
        return Err(KError::InvalidInput("csr_mul: dimension mismatch".into()));
    }
    let m = a.nrows();
    let n = b.ncols();
    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);

    for i in 0..m {
        let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
        for ap in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let k = a.col_idx()[ap];
            let a_ik = a.values()[ap];
            let rs = b.row_ptr()[k];
            let re = b.row_ptr()[k + 1];
            for bp in rs..re {
                let j = b.col_idx()[bp];
                let v = a_ik * b.values()[bp];
                *acc.entry(j).or_insert(0.0) += v;
            }
        }
        // append in column order
        for (j, v) in acc.into_iter() {
            if v.abs() > 0.0 {
                col_idx.push(j);
                vals.push(v);
            }
        }
        row_ptr.push(col_idx.len());
    }

    Ok(CsrMatrix::from_csr(m, n, row_ptr, col_idx, vals))
}

/// RAP = R * A * P
fn rap(r: &CsrMatrix<f64>, a: &CsrMatrix<f64>, p: &CsrMatrix<f64>) -> Result<CsrMatrix<f64>, KError> {
    let ap = csr_mul(a, p)?;
    csr_mul(r, &ap)
}

// ===== Coarsening & interpolation (dense helpers, same as old) ==============

fn compute_anisotropy(a: &Mat<f64>) -> Vec<f64> {
    let n = a.nrows();
    #[cfg(feature = "rayon")]
    return (0..n).into_par_iter().map(|i| {
        let diag = a[(i, i)];
        let mut max_off: f64 = 0.0;
        for j in 0..n { if i != j { max_off = max_off.max(a[(i, j)].abs()); } }
        if diag.abs() > 1e-14 { max_off / diag.abs() } else { 0.0 }
    }).collect();
    #[cfg(not(feature = "rayon"))]
    {
        let mut out = vec![0.0; n];
        for i in 0..n {
            let diag = a[(i, i)];
            let mut max_off: f64 = 0.0;
            for j in 0..n { if i != j { max_off = max_off.max(a[(i, j)].abs()); } }
            out[i] = if diag.abs() > 1e-14 { max_off / diag.abs() } else { 0.0 };
        }
        out
    }
}

fn compute_adaptive_threshold(a: &Mat<f64>, base_threshold: f64) -> f64 {
    let anis = compute_anisotropy(a);
    let avg = if anis.is_empty() { 1.0 } else { anis.iter().sum::<f64>() / anis.len() as f64 };
    base_threshold * (1.0 + avg.max(0.5))
}

/// S(i,j) = |A_ij| / sqrt(|A_ii| |A_jj|) if above threshold.
fn compute_strength_matrix(a: &Mat<f64>, thr: f64) -> Mat<f64> {
    let n = a.nrows();
    let mut s = Mat::<f64>::zeros(n, n);
    let mut diag = vec![0.0; n];
    for i in 0..n { diag[i] = a[(i, i)].abs(); }
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            let denom = (diag[i] * diag[j]).sqrt();
            if denom > 1e-14 {
                let st = a[(i, j)].abs() / denom;
                if st > thr { s[(i, j)] = st; }
            }
        }
    }
    s
}

fn pairwise_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let n = s.nrows();
    let mut agg = vec![usize::MAX; n];
    let mut vis = vec![false; n];
    let mut id = 0usize;
    for i in 0..n {
        if vis[i] { continue; }
        let mut best = None;
        let mut bestv = 0.0;
        for j in 0..n {
            if i == j || vis[j] { continue; }
            let v = s[(i, j)];
            if v > bestv { bestv = v; best = Some(j); }
        }
        if let Some(j) = best {
            agg[i] = id; agg[j] = id; vis[i] = true; vis[j] = true; id += 1;
        } else {
            agg[i] = id; vis[i] = true; id += 1;
        }
    }
    agg
}

fn build_coarse_graph(s: &Mat<f64>, agg: &[usize]) -> Mat<f64> {
    let max_id = *agg.iter().max().unwrap_or(&0);
    let cn = max_id + 1;
    let mut cg = Mat::<f64>::zeros(cn, cn);
    let n = s.nrows();
    for i in 0..n {
        for j in 0..n {
            let ai = agg[i];
            let aj = agg[j];
            let v = s[(i, j)];
            if v != 0.0 { cg[(ai, aj)] += v; }
        }
    }
    cg
}

fn remap_aggregates(first: &[usize], second: &[usize]) -> Vec<usize> {
    #[cfg(feature = "rayon")]
    return first.par_iter().map(|&c| second[c]).collect();
    #[cfg(not(feature = "rayon"))]
    first.iter().map(|&c| second[c]).collect()
}

fn double_pairwise_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let pass1 = pairwise_aggregation(s);
    let coarse = build_coarse_graph(s, &pass1);
    let pass2 = pairwise_aggregation(&coarse);
    remap_aggregates(&pass1, &pass2)
}

/// Greedy aggregation (balanced, small aggregates).
fn greedy_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let n = s.nrows();
    let mut agg = vec![usize::MAX; n];
    let mut next = 0usize;
    let max_sz = 4usize;

    // Order by total strength descending
    let mut order: Vec<(f64, usize)> = (0..n)
        .map(|i| ((0..n).map(|j| s[(i, j)]).sum::<f64>(), i))
        .collect();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    for &(_, seed) in &order {
        if agg[seed] != usize::MAX { continue; }
        agg[seed] = next;
        // pick strongest distinct neighbors
        let mut neigh: Vec<(f64, usize)> =
            (0..n).filter(|&j| j != seed && agg[j] == usize::MAX).map(|j| (s[(seed, j)], j)).collect();
        neigh.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(_, j) in neigh.iter() {
            if (0..n).filter(|&i| agg[i] == next).count() >= max_sz { break; }
            if s[(seed, j)] > 0.1 && agg[j] == usize::MAX {
                agg[j] = next;
            }
        }
        next += 1;
    }
    agg
}

/// Piecewise-constant prolongation from aggregates (dense).
fn construct_prolongation(_a: &Mat<f64>, aggregates: &[usize]) -> Mat<f64> {
    let n = aggregates.len();
    let max_id = *aggregates.iter().max().unwrap_or(&0);
    let nc = max_id + 1;
    let mut p = Mat::<f64>::zeros(n, nc);
    for (i, &g) in aggregates.iter().enumerate() { p[(i, g)] = 1.0; }
    p
}

/// Simple smoothing of P with weight* A (Jacobi-like).
fn smooth_interpolation(p: &mut Mat<f64>, a: &Mat<f64>, weight: f64) {
    let r = p.nrows().min(a.nrows());
    let c = p.ncols();
    for i in 0..r {
        for j in 0..c {
            p[(i, j)] -= weight * a[(i, j.min(a.ncols() - 1))];
        }
    }
}

/// Row 2-norm normalization.
fn minimize_energy(p: &mut Mat<f64>, _a: &Mat<f64>) {
    let (m, n) = (p.nrows(), p.ncols());
    for i in 0..m {
        let mut norm2 = 0.0;
        for j in 0..n { norm2 += p[(i, j)] * p[(i, j)]; }
        let s = if norm2 > 1e-14 { norm2.sqrt() } else { 1.0 };
        for j in 0..n { p[(i, j)] /= s; }
    }
}

// ===== Small sparse CG for coarsest level ===================================

fn cg_sparse(a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], tol: f64, maxit: usize) -> Result<(), KError> {
    let n = a.nrows();
    if n == 0 { return Ok(()); }
    x.fill(0.0);

    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut ap = vec![0.0; n];

    let mut rsold = dot(&r, &r);
    let atol = tol.max(1e-12) * rsold.sqrt().max(1e-30);

    for _ in 0..maxit {
        a.spmv_scaled(1.0, &p, 0.0, &mut ap)?;
        let denom = dot(&p, &ap);
        if denom.abs() < 1e-30 { break; }
        let alpha = rsold / denom;

        #[cfg(feature = "rayon")]
        { x.par_iter_mut().zip(p.par_iter()).for_each(|(xi, &pi)| *xi += alpha * pi); }
        #[cfg(not(feature = "rayon"))]
        for i in 0..n { x[i] += alpha * p[i]; }

        #[cfg(feature = "rayon")]
        { r.par_iter_mut().zip(ap.par_iter()).for_each(|(ri, &api)| *ri -= alpha * api); }
        #[cfg(not(feature = "rayon"))]
        for i in 0..n { r[i] -= alpha * ap[i]; }

        let rsnew = dot(&r, &r);
        if rsnew.sqrt() < atol { break; }
        let beta = rsnew / rsold;

        #[cfg(feature = "rayon")]
        { p.par_iter_mut().zip(r.par_iter()).for_each(|(pi, &ri)| *pi = ri + beta * *pi); }
        #[cfg(not(feature = "rayon"))]
        for i in 0..n { p[i] = r[i] + beta * p[i]; }

        rsold = rsnew;
    }
    Ok(())
}

#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(feature = "rayon")] { x.par_iter().zip(y.par_iter()).map(|(a,b)| a*b).sum() }
    #[cfg(not(feature = "rayon"))] { x.iter().zip(y.iter()).map(|(a,b)| a*b).sum() }
}

// ===== Helpers for transpose mapping and stats ==============================

#[derive(Clone, Debug)]
pub struct AmgStats {
    pub grid_complexity: f64,
    pub operator_complexity: f64,
    pub num_levels: usize,
}

impl AmgStats {
    fn from_hierarchy(h: &AmgHierarchy) -> Self {
        let n0 = h.levels.first().map(|l| l.a.nrows() as f64).unwrap_or(1.0);
        let nnz0 = h.levels.first().map(|l| l.a.nnz() as f64).unwrap_or(1.0);
        let mut ng_sum = 0.0;
        let mut nnz_sum = 0.0;
        for l in &h.levels {
            ng_sum += l.a.nrows() as f64;
            nnz_sum += l.a.nnz() as f64;
        }
        Self {
            grid_complexity: ng_sum / n0,
            operator_complexity: nnz_sum / nnz0,
            num_levels: h.levels.len(),
        }
    }
}

fn operator_complexity_estimate(levels: &[AMGLevel]) -> f64 {
    if levels.is_empty() { return 0.0; }
    let nnz0 = levels[0].a.nnz() as f64;
    let nnz_sum: f64 = levels.iter().map(|l| l.a.nnz() as f64).sum();
    nnz_sum / nnz0
}

fn transpose_csr_with_pos(p: &Pcsr) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<usize>) {
    // Compute transpose pattern and values, and mapping from P entry index -> R entry index
    let (m, n) = (p.m, p.n);
    let nnz = p.col_idx.len();
    let mut r_row_counts = vec![0usize; n + 1];
    for &cj in &p.col_idx { r_row_counts[cj + 1] += 1; }
    for i in 0..n { r_row_counts[i + 1] += r_row_counts[i]; }
    let mut r_col_idx = vec![0usize; nnz];
    let mut r_vals = vec![0.0f64; nnz];
    let mut r_row_next = r_row_counts.clone();
    let mut p2r_pos = vec![0usize; nnz];
    for i in 0..m {
        let rs = p.row_ptr[i]; let re = p.row_ptr[i+1];
        for pi in rs..re {
            let cj = p.col_idx[pi];
            let dest = r_row_next[cj];
            r_col_idx[dest] = i;
            r_vals[dest] = p.vals[pi];
            p2r_pos[pi] = dest;
            r_row_next[cj] += 1;
        }
    }
    (r_row_counts, r_col_idx, r_vals, p2r_pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn identity_level() -> AMGLevel {
        AMGLevel {
            a: CsrMatrix::identity(1),
            p: CsrMatrix::identity(1),
            r: CsrMatrix::identity(1),
            diag_inv: vec![1.0],
            agg_of: vec![0],
            p2r_pos: vec![],
            a_next_pat: None,
        }
    }

    #[test]
    fn phase_selection_logic() {
        reset_relax_counts();
        let levels = vec![identity_level(), identity_level(), identity_level()];
        let policy = RelaxPolicy { kind: [RelaxType::Jacobi; 4], sweeps: [1, 1, 1, 0], omega: 1.0 };
        let hier = AmgHierarchy { levels, policy, coarse_solve: CoarseSolve::DirectDense };
        let amg = AMG { state: Some(hier), ..Default::default() };
        let rhs = [1.0];
        let mut sol = [0.0];
        amg.apply(PcSide::Left, &rhs, &mut sol).unwrap();
        let counts = get_relax_counts();
        assert_eq!(counts[RelaxPhase::Fine.ix()], 2);
        assert_eq!(counts[RelaxPhase::Down.ix()], 1);
        assert_eq!(counts[RelaxPhase::Up.ix()], 1);
        assert_eq!(counts[RelaxPhase::Coarsest.ix()], 0);
    }

    #[test]
    fn validation_failures() {
        let mut cfg = AMGConfig::default();
        cfg.grid_relax_type = [RelaxType::GaussSeidel; 4];
        let err = validate_relax_policy(&cfg, cfg.coarse_solve).unwrap_err();
        assert!(matches!(err, KError::InvalidInput(_)));

        let mut cfg = AMGConfig::default();
        cfg.coarse_solve = CoarseSolve::DirectDense;
        cfg.num_grid_sweeps[RelaxPhase::Coarsest.ix()] = 1;
        let err = validate_relax_policy(&cfg, cfg.coarse_solve).unwrap_err();
        assert!(matches!(err, KError::InvalidInput(_)));

        let mut cfg = AMGConfig::default();
        cfg.truncation_factor = 1.2;
        assert!(validate_truncation_and_caps(&cfg).is_err());
        cfg.truncation_factor = -0.1;
        assert!(validate_truncation_and_caps(&cfg).is_err());
        cfg.truncation_factor = 0.0;
        cfg.interpolation_truncation = -1.0;
        assert!(validate_truncation_and_caps(&cfg).is_err());
        cfg.interpolation_truncation = 0.0;
        cfg.rap_truncation_abs = -1.0;
        assert!(validate_truncation_and_caps(&cfg).is_err());
    }

    #[test]
    fn legacy_shim_populates_arrays() {
        let amg = AMG::builder().smoothing_sweeps(2, 3).build(&Mat::<f64>::zeros(0, 0)).unwrap();
        assert_eq!(amg.cfg.num_grid_sweeps, [2, 2, 3, 1]);
    }
}
