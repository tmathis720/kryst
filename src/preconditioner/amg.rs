#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::KError;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::matrix::{convert::csr_from_linop, sparse::CsrMatrix};
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// New sparse SA/RS submodules
mod coarse_solver;
pub mod coarsen;
pub(crate) mod prolong;
mod rap_ops;
mod row_filter;
pub mod strength;

use coarse_solver::{CoarseDenseLu, CoarseSolve, CoarseSolver};
use coarsen::{build_aggregates, AggAlgo, AggOpts};
use prolong::{
    smooth_sa_values_only,
    smooth_tentative_sa,
    Pcsr,
    TentativeP,
    CFInfo,
    classical_pattern,
    classical_values_only,
    ClassicalParams,
    ClassicalVariant,
};
use rap_ops::{rap_numeric, rap_symbolic, CsrPattern};
use row_filter::{apply_filter_to_csr_values_in_place, RowFilter};
use strength::Strength;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleType {
    V,
    W { gamma: usize },
}

impl Default for CycleType {
    fn default() -> Self {
        CycleType::V
    }
}

#[derive(Clone, Debug)]
pub struct AMGConfig {
    pub max_levels: usize,           // HYPRE default: 25
    pub strong_threshold: f64,       // HYPRE default: 0.25
    pub coarse_threshold: usize,     // HYPRE default: 9
    pub max_coarse_size: usize,      // HYPRE default: 9
    pub min_coarse_size: usize,      // HYPRE minimum: 1
    pub truncation_factor: f64,      // 0 => no truncation
    pub max_elements_per_row: usize, // 0 => unlimited
    pub interpolation_truncation: f64,
    pub rap_truncation_abs: f64,
    pub rap_max_elements_per_row: usize,
    pub keep_pivot_in_rap: bool,
    pub grid_relax_type: [RelaxType; 4], // [Fine, Down, Up, Coarsest]
    pub num_grid_sweeps: [usize; 4],     // [Fine, Down, Up, Coarsest]
    // legacy shims
    pub pre_sweeps: usize,         // HYPRE default: 1
    pub post_sweeps: usize,        // HYPRE default: 1
    pub coarsen_type: CoarsenType, // HYPRE default: HMIS
    pub interp_type: InterpType,   // robust: Extended/Standard
    pub relax_type: RelaxType,     // HYPRE default: Gauss-Seidel (we implement Jacobi)
    pub logging_level: usize,
    pub print_level: usize,
    pub tolerance: f64, // for coarse direct solve (CG)
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub ieee_checks: bool,
    pub optimize_workspace: bool,
    pub jacobi_omega: f64,
    pub chebyshev_degree: usize,
    pub chebyshev_min_ratio: f64,
    pub chebyshev_recompute: bool,
    pub chebyshev_power_iters: usize,
    pub use_level_scheduling: bool,
    pub drop_tol: f64,  // NEW: used for dense->CSR conversion
    pub stats_eps: f64, // threshold for effective nnz reporting
    // New SA/RS controls
    pub normalize_strength: bool,
    pub coarse_solve: CoarseSolve,
    pub ilu_drop_tol: f64,
    pub ilu_fill_per_row: usize,
    pub max_operator_complexity: Option<f64>,
    pub agg_num_levels: usize,
    pub aggressive_mis_k: usize,
    pub max_strong_per_row: Option<usize>,
    pub cycle_type: CycleType,
    pub fmg_nu_pre: usize,
    pub fmg_nu_post: usize,
    pub fmg_gamma: usize,
    pub fmg_levels_use: Option<usize>,
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
            chebyshev_min_ratio: 0.3,
            chebyshev_recompute: true,
            chebyshev_power_iters: 10,
            use_level_scheduling: false,
            drop_tol: 1e-12,
            stats_eps: 1e-12,
            normalize_strength: true,
            coarse_solve: CoarseSolve::CG,
            ilu_drop_tol: 1e-2,
            ilu_fill_per_row: 0,
            max_operator_complexity: None,
            agg_num_levels: 1,
            aggressive_mis_k: 2,
            max_strong_per_row: None,
            cycle_type: CycleType::V,
            fmg_nu_pre: 1,
            fmg_nu_post: 1,
            fmg_gamma: 1,
            fmg_levels_use: None,
        };
        cfg.grid_relax_type = [
            cfg.relax_type,
            cfg.relax_type,
            cfg.relax_type,
            RelaxType::GaussSeidel,
        ];
        cfg.num_grid_sweeps = [cfg.pre_sweeps, cfg.pre_sweeps, cfg.post_sweeps, 1];
        cfg.stats_eps = cfg.drop_tol;
        cfg
    }
}

/// Builder for `AMG` (preserves old chaining API).
pub struct AMGBuilder {
    cfg: AMGConfig,
}

impl AMGBuilder {
    pub fn new() -> Self {
        Self {
            cfg: AMGConfig::default(),
        }
    }
    pub fn cycle_v(mut self) -> Self {
        self.cfg.cycle_type = CycleType::V;
        self
    }
    pub fn cycle_w(mut self, gamma: usize) -> Self {
        let g = gamma.max(2);
        self.cfg.cycle_type = CycleType::W { gamma: g };
        self
    }
    pub fn max_levels(mut self, v: usize) -> Self {
        self.cfg.max_levels = v;
        self
    }
    pub fn strong_threshold(mut self, v: f64) -> Self {
        self.cfg.strong_threshold = v;
        self
    }
    pub fn coarse_threshold(mut self, v: usize) -> Self {
        self.cfg.coarse_threshold = v;
        self
    }
    pub fn max_coarse_size(mut self, v: usize) -> Self {
        self.cfg.max_coarse_size = v;
        self
    }
    pub fn min_coarse_size(mut self, v: usize) -> Self {
        self.cfg.min_coarse_size = v;
        self
    }
    pub fn truncation_factor(mut self, v: f64) -> Self {
        self.cfg.truncation_factor = v;
        self
    }
    pub fn interpolation_drop_abs(mut self, v: f64) -> Self {
        self.cfg.interpolation_truncation = v;
        self
    }
    pub fn interpolation_cap(mut self, k: usize) -> Self {
        self.cfg.max_elements_per_row = k;
        self
    }
    pub fn rap_drop_abs(mut self, v: f64) -> Self {
        self.cfg.rap_truncation_abs = v;
        self
    }
    pub fn rap_cap(mut self, k: usize) -> Self {
        self.cfg.rap_max_elements_per_row = k;
        self
    }
    pub fn keep_pivot_in_rap(mut self, yes: bool) -> Self {
        self.cfg.keep_pivot_in_rap = yes;
        self
    }
    pub fn interpolation_truncation(self, v: f64) -> Self {
        self.interpolation_drop_abs(v)
    }
    pub fn smoothing_sweeps(mut self, pre: usize, post: usize) -> Self {
        self.cfg.pre_sweeps = pre;
        self.cfg.post_sweeps = post;
        self.cfg.num_grid_sweeps[RelaxPhase::Fine.ix()] = pre;
        self.cfg.num_grid_sweeps[RelaxPhase::Down.ix()] = pre;
        self.cfg.num_grid_sweeps[RelaxPhase::Up.ix()] = post;
        // leave Coarsest as-is
        self
    }
    pub fn coarsening_type(mut self, v: CoarsenType) -> Self {
        self.cfg.coarsen_type = v;
        self
    }
    pub fn agg_num_levels(mut self, v: usize) -> Self {
        self.cfg.agg_num_levels = v;
        self
    }
    pub fn aggressive_mis_k(mut self, v: usize) -> Self {
        self.cfg.aggressive_mis_k = v;
        self
    }
    pub fn max_strong_per_row(mut self, k: usize) -> Self {
        self.cfg.max_strong_per_row = Some(k);
        self
    }
    pub fn interpolation_type(mut self, v: InterpType) -> Self {
        self.cfg.interp_type = v;
        self
    }
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
    pub fn enable_logging(mut self) -> Self {
        self.cfg.logging_level = 1;
        self
    }
    pub fn logging_level(mut self, lvl: usize) -> Self {
        self.cfg.logging_level = lvl;
        self
    }
    pub fn enable_printing(mut self) -> Self {
        self.cfg.print_level = 1;
        self
    }
    pub fn print_level(mut self, lvl: usize) -> Self {
        self.cfg.print_level = lvl;
        self
    }
    pub fn jacobi_omega(mut self, w: f64) -> Self {
        self.cfg.jacobi_omega = w;
        self
    }
    pub fn chebyshev_degree(mut self, k: usize) -> Self {
        self.cfg.chebyshev_degree = k;
        self
    }
    pub fn chebyshev_min_ratio(mut self, v: f64) -> Self {
        self.cfg.chebyshev_min_ratio = v;
        self
    }
    pub fn chebyshev_recompute(mut self, v: bool) -> Self {
        self.cfg.chebyshev_recompute = v;
        self
    }
    pub fn chebyshev_power_iters(mut self, iters: usize) -> Self {
        self.cfg.chebyshev_power_iters = iters;
        self
    }
    pub fn use_level_scheduling(mut self, v: bool) -> Self {
        self.cfg.use_level_scheduling = v;
        self
    }
    pub fn drop_tolerance(mut self, t: f64) -> Self {
        self.cfg.drop_tol = t;
        self.cfg.stats_eps = t;
        self
    }
    pub fn stats_eps(mut self, t: f64) -> Self {
        self.cfg.stats_eps = t;
        self
    }

    pub fn build(self, _matrix: &Mat<f64>) -> Result<AMG, KError> {
        Ok(AMG::with_config(self.cfg))
    }
}

impl Default for AMGBuilder {
    fn default() -> Self {
        Self::new()
    }
}

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
            RelaxType::Jacobi
            | RelaxType::GaussSeidel
            | RelaxType::GaussSeidelBackward
            | RelaxType::SymmetricGaussSeidel
            | RelaxType::L1Jacobi
            | RelaxType::Chebyshev => {}
            _ => {
                return Err(KError::InvalidInput(format!(
                    "RelaxType {:?} not yet supported (phase index {})",
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
        return Err(KError::InvalidInput(
            "truncation_factor must satisfy 0 ≤ τ_rel < 1".into(),
        ));
    }
    if cfg.interpolation_truncation < 0.0 || cfg.rap_truncation_abs < 0.0 {
        return Err(KError::InvalidInput(
            "absolute drop tolerances must be ≥ 0".into(),
        ));
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
        let grow = |v: &mut Vec<f64>, n: usize| {
            if v.len() < n {
                v.resize(n, 0.0)
            }
        };
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
    /// 1 / (\sum_j |a_ij|) for L1-Jacobi
    l1_inv: Option<Vec<f64>>,
    /// Cached spectral bounds for Chebyshev smoother
    cheb: Option<ChebCache>,
    /// fine->coarse aggregate id used to rebuild P values (SA numeric refresh)
    agg_of: Vec<usize>,
    /// coarse/fine flags for classical interpolation
    is_c: Vec<bool>,
    /// CF metadata for classical interpolation
    cf: Option<CFInfo>,
    /// Mapping from P entry index -> index in R (transpose) values array
    p2r_pos: Vec<usize>,
    /// Symbolic pattern for A_{l+1}
    a_next_pat: Option<CsrPattern>,
}

#[derive(Clone)]
struct ChebCache {
    lambda_max: f64,
    lambda_min: f64,
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
    fn finest(&self) -> &AMGLevel {
        &self.levels[0]
    }
    fn coarsest_ix(&self) -> usize {
        self.levels.len() - 1
    }
}

// ===== Main AMG object =======================================================

pub struct AMG {
    csr: Option<Arc<CsrMatrix<f64>>>,
    state: Option<AmgHierarchy>,
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,
    cfg: AMGConfig,
    stats: Option<AmgStats>,
    runtime: Mutex<AmgRuntime>,
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
            runtime: Mutex::new(AmgRuntime::default()),
        }
    }
}

impl AMG {
    pub fn new(_matrix: &Mat<f64>, _max_levels: usize, _coarsening_threshold: f64) -> Self {
        AMG::default()
    }
    pub fn builder() -> AMGBuilder {
        AMGBuilder::new()
    }
    pub fn with_config(cfg: AMGConfig) -> Self {
        Self {
            cfg,
            ..Default::default()
        }
    }

    // ---- Setup paths --------------------------------------------------------

    fn build_symbolic(&mut self, fine: &CsrMatrix<f64>) -> Result<(), KError> {
        // Build the full hierarchy from scratch (symbolic + numeric)
        let (hier, stats) = build_hierarchy(fine, &self.cfg)?;
        self.state = Some(hier);
        self.stats = stats;
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

        let need_l1 = self
            .cfg
            .grid_relax_type
            .iter()
            .any(|&t| t == RelaxType::L1Jacobi);
        let need_cheb = self
            .cfg
            .grid_relax_type
            .iter()
            .any(|&t| t == RelaxType::Chebyshev);

        // Recompute P_l values, R_l values, and A_{l+1} values using fixed patterns
        for l in 0..h.coarsest_ix() {
            // Recompute P_l values in-place using SA smoother with fixed pattern
            let pr = h.levels[l].p.row_ptr().to_vec();
            let pc = h.levels[l].p.col_idx().to_vec();
            let mut p_new_vals = vec![0.0f64; pc.len()];
            if let Some(ref cf) = h.levels[l].cf {
                let s = Strength::from_csr(
                    &h.levels[l].a,
                    self.cfg.strong_threshold,
                    self.cfg.normalize_strength,
                );
                let s_sym = s.symmetrize();
                let params = ClassicalParams {
                    variant: match self.cfg.interp_type {
                        InterpType::Direct => ClassicalVariant::Direct,
                        InterpType::Standard | InterpType::Classical | InterpType::Extended => ClassicalVariant::Standard,
                        _ => ClassicalVariant::Standard,
                    },
                    extended: matches!(self.cfg.interp_type, InterpType::Extended),
                    drop_abs: self.cfg.interpolation_truncation,
                    trunc_rel: self.cfg.truncation_factor,
                    cap_row: self.cfg.max_elements_per_row,
                    keep_at_least_one: true,
                };
                classical_values_only(
                    &h.levels[l].a,
                    &s_sym,
                    cf,
                    &params,
                    &pr,
                    &pc,
                    &mut p_new_vals,
                )?;
            } else {
                let tp = TentativeP {
                    agg_of: h.levels[l].agg_of.clone(),
                    n_coarse: h.levels[l + 1].a.nrows(),
                };
                smooth_sa_values_only(
                    &h.levels[l].a,
                    &h.levels[l].diag_inv,
                    &tp,
                    self.cfg.jacobi_omega,
                    &pr,
                    &pc,
                    &mut p_new_vals,
                )?;
            }
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
                        must_keep: if self.cfg.keep_pivot_in_rap {
                            Some(row)
                        } else {
                            None
                        },
                    };
                    apply_filter_to_csr_values_in_place(
                        pat.nrows,
                        &pat.row_ptr,
                        &pat.col_idx,
                        &mut vals,
                        &mut rf,
                    );
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

        for lvl in 0..=h.coarsest_ix() {
            if need_l1 {
                h.levels[lvl].l1_inv = Some(l1_diag_inv(&h.levels[lvl].a));
            } else {
                h.levels[lvl].l1_inv = None;
            }
            if need_cheb {
                if self.cfg.chebyshev_recompute || h.levels[lvl].cheb.is_none() {
                    let lmax = estimate_lambda_max(
                        &h.levels[lvl].a,
                        &h.levels[lvl].diag_inv,
                        self.cfg.chebyshev_power_iters,
                    );
                    let lmin = (self.cfg.chebyshev_min_ratio * lmax).max(1e-12);
                    h.levels[lvl].cheb = Some(ChebCache { lambda_max: lmax, lambda_min: lmin });
                }
            } else {
                h.levels[lvl].cheb = None;
            }
        }

        if self.cfg.logging_level > 0 {
            let mut st = AmgStats::from_hierarchy(&h);
            st.levels = collect_level_stats(&h, &self.cfg);
            self.stats = Some(st);
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
        if iters == 0 {
            return Ok(());
        }
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

    fn l1_jacobi(
        omega: f64,
        a: &CsrMatrix<f64>,
        l1_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        iters: usize,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if iters == 0 {
            return Ok(());
        }
        let n = a.nrows();
        if l1_inv.len() != n || r.len() != n || z.len() != n {
            return Err(KError::InvalidInput("L1-Jacobi: dimension mismatch".into()));
        }
        ws.ensure(n);
        ws.temp[..n].copy_from_slice(z);
        for _ in 0..iters {
            a.spmv_scaled(1.0, &ws.temp[..n], 0.0, &mut ws.work[..n])?;
            for i in 0..n {
                ws.temp[i] += omega * l1_inv[i] * (r[i] - ws.work[i]);
            }
        }
        z.copy_from_slice(&ws.temp[..n]);
        Ok(())
    }

    fn gs_forward(
        omega: f64,
        a: &CsrMatrix<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        sweeps: usize,
    ) -> Result<(), KError> {
        let n = a.nrows();
        if diag_inv.len() != n || r.len() != n || z.len() != n {
            return Err(KError::InvalidInput("GS: dimension mismatch".into()));
        }
        for _ in 0..sweeps {
            for i in 0..n {
                let mut s = 0.0;
                let rs = a.row_ptr()[i];
                let re = a.row_ptr()[i + 1];
                for p in rs..re {
                    s += a.values()[p] * z[a.col_idx()[p]];
                }
                z[i] += omega * diag_inv[i] * (r[i] - s);
            }
        }
        Ok(())
    }

    fn gs_backward(
        omega: f64,
        a: &CsrMatrix<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        sweeps: usize,
    ) -> Result<(), KError> {
        let n = a.nrows();
        if diag_inv.len() != n || r.len() != n || z.len() != n {
            return Err(KError::InvalidInput("GS: dimension mismatch".into()));
        }
        for _ in 0..sweeps {
            for i in (0..n).rev() {
                let mut s = 0.0;
                let rs = a.row_ptr()[i];
                let re = a.row_ptr()[i + 1];
                for p in rs..re {
                    s += a.values()[p] * z[a.col_idx()[p]];
                }
                z[i] += omega * diag_inv[i] * (r[i] - s);
            }
        }
        Ok(())
    }

    fn sym_gs(
        omega: f64,
        a: &CsrMatrix<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        sweeps: usize,
    ) -> Result<(), KError> {
        for _ in 0..sweeps {
            Self::gs_forward(omega, a, diag_inv, r, z, 1)?;
            Self::gs_backward(omega, a, diag_inv, r, z, 1)?;
        }
        Ok(())
    }

    fn chebyshev_smooth(
        a: &CsrMatrix<f64>,
        d_inv: &[f64],
        rhs: &[f64],
        z: &mut [f64],
        k: usize,
        lambda_min: f64,
        lambda_max: f64,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if k == 0 {
            return Ok(());
        }
        let n = a.nrows();
        ws.ensure(n);

        a.spmv_scaled(1.0, z, 0.0, &mut ws.work[..n])?;
        for i in 0..n {
            ws.residual[i] = rhs[i] - ws.work[i];
        }

        let c = 0.5 * (lambda_max - lambda_min);
        let d = 0.5 * (lambda_max + lambda_min);
        let mut alpha = 0.0;
        let mut beta = 0.0;
        let p = &mut ws.coarse_rhs[..n];
        p[..n].fill(0.0);

        for t in 0..k {
            if t == 0 {
                alpha = 1.0 / d;
                p[..n].copy_from_slice(&ws.residual[..n]);
            } else {
                let tmp = 0.5 * c * alpha;
                beta = tmp * tmp;
                alpha = 1.0 / (d - beta);
                for i in 0..n {
                    p[i] = ws.residual[i] + beta * p[i];
                }
            }

            for i in 0..n {
                ws.temp[i] = d_inv[i] * p[i];
            }
            for i in 0..n {
                z[i] += alpha * ws.temp[i];
            }
            a.spmv_scaled(alpha, &ws.temp[..n], 0.0, &mut ws.work[..n])?;
            for i in 0..n {
                ws.residual[i] -= ws.work[i];
            }
        }
        Ok(())
    }

    // single dispatch point for all relaxation strategies
    fn apply_relax(
        pol: &RelaxPolicy,
        phase: RelaxPhase,
        where_: RelaxWhere,
        lvl: &AMGLevel,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
        cfg: &AMGConfig,
    ) -> Result<(), KError> {
        let k = pol.sweeps[phase.ix()];
        if k == 0 {
            return Ok(());
        }
        #[cfg(test)]
        {
            RELAX_CALL_COUNTS[phase.ix()].fetch_add(1, Ordering::SeqCst);
        }
        let a = &lvl.a;
        match pol.kind[phase.ix()] {
            RelaxType::Jacobi => {
                Self::jacobi_smooth_sparse(pol.omega, a, &lvl.diag_inv, rhs, sol, k, ws)
            }
            RelaxType::GaussSeidel => {
                if matches!(where_, RelaxWhere::Pre) {
                    Self::gs_forward(1.0, a, &lvl.diag_inv, rhs, sol, k)
                } else {
                    Self::gs_backward(1.0, a, &lvl.diag_inv, rhs, sol, k)
                }
            }
            RelaxType::GaussSeidelBackward => {
                Self::gs_backward(1.0, a, &lvl.diag_inv, rhs, sol, k)
            }
            RelaxType::SymmetricGaussSeidel => {
                Self::sym_gs(1.0, a, &lvl.diag_inv, rhs, sol, k)
            }
            RelaxType::L1Jacobi => {
                if let Some(ref l1) = lvl.l1_inv {
                    Self::l1_jacobi(pol.omega, a, l1, rhs, sol, k, ws)
                } else {
                    Err(KError::InvalidInput("L1Jacobi cache missing".into()))
                }
            }
            RelaxType::Chebyshev => {
                if let Some(ref cheb) = lvl.cheb {
                    let mut lmax = cheb.lambda_max.max(1e-12);
                    let mut lmin = cheb.lambda_min.min(0.99 * lmax).max(1e-16);
                    if lmin >= lmax {
                        Self::jacobi_smooth_sparse(pol.omega, a, &lvl.diag_inv, rhs, sol, k.min(2).max(1), ws)
                    } else {
                        let deg = if cfg.chebyshev_degree > 0 {
                            cfg.chebyshev_degree
                        } else {
                            k
                        };
                        Self::chebyshev_smooth(a, &lvl.diag_inv, rhs, sol, deg, lmin, lmax, ws)
                    }
                } else {
                    Err(KError::InvalidInput("Chebyshev cache missing".into()))
                }
            }
            other => Err(KError::InvalidInput(format!(
                "RelaxType {:?} not yet supported",
                other
            ))),
        }
    }

    // ---- Multigrid cycle ----------------------------------------------------

    fn cycle_profiled(
        &self,
        level: usize,
        gamma: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
        mut cyc: Option<&mut CycleTimings>,
    ) -> Result<(), KError> {
        let h = self
            .state
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("AMG not set up".into()))?;
        let lc = h.coarsest_ix();

        let a = &h.levels[level].a;
        let pol = &h.policy;
        let mut lv = CycleLevelTiming {
            level,
            ..Default::default()
        };
        let prof = cyc.is_some();

        if level == lc {
            with_timing(prof, &mut lv.coarse_solve, || {
                let use_dense = matches!(h.coarse_solve, CoarseSolve::DirectDense)
                    || a.nrows() <= self.cfg.max_coarse_size;
                if use_dense {
                    let mut solver = CoarseDenseLu::new();
                    solver.setup(a)?;
                    solver.solve(rhs, sol)
                } else {
                    match h.coarse_solve {
                        CoarseSolve::CG | CoarseSolve::ILU => {
                            cg_sparse(a, rhs, sol, self.cfg.tolerance, a.nrows().min(50))
                        }
                        CoarseSolve::DirectDense => unreachable!(),
                    }
                }
            })?;
            if let Some(c) = cyc {
                c.per_level.push(lv);
            }
            return Ok(());
        }

        let n = a.nrows();
        ws.ensure(n);

        // Pre-smooth
        let phase_pre = if level == 0 {
            RelaxPhase::Fine
        } else {
            RelaxPhase::Down
        };
        with_timing(prof, &mut lv.pre_smooth, || {
            Self::apply_relax(
                pol,
                phase_pre,
                RelaxWhere::Pre,
                &h.levels[level],
                rhs,
                sol,
                ws,
                &self.cfg,
            )
        })?;

        // residual = rhs - A * sol
        with_timing(prof, &mut lv.matvec, || {
            a.spmv_scaled(1.0, sol, 0.0, &mut ws.work[..n])
        })?;
        with_timing(prof, &mut lv.residual_axpy, || {
            #[cfg(feature = "rayon")]
            ws.residual[..n]
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, ri)| {
                    *ri = rhs[i] - ws.work[i];
                });
            #[cfg(not(feature = "rayon"))]
            for i in 0..n {
                ws.residual[i] = rhs[i] - ws.work[i];
            }
        });

        // r_c = R * residual
        let r = &h.levels[level].r;
        let p = &h.levels[level].p;
        let nc = h.levels[level + 1].a.nrows();

        let mut local_coarse = std::mem::take(&mut ws.coarse_rhs);
        local_coarse.resize(nc, 0.0);
        with_timing(prof, &mut lv.restrict, || {
            r.spmv_scaled(1.0, &ws.residual[..n], 0.0, &mut local_coarse[..nc])
        })?;

        let gamma = gamma.max(1);
        for t in 0..gamma {
            let mut zc = vec![0.0; nc];
            if level + 1 == lc {
                with_timing(prof, &mut lv.coarse_solve, || {
                    let mut solver = CoarseDenseLu::new();
                    solver.setup(&h.levels[level + 1].a)?;
                    solver.solve(&local_coarse[..nc], &mut zc)
                })?;
            } else {
                self.cycle_profiled(
                    level + 1,
                    gamma,
                    &local_coarse[..nc],
                    &mut zc,
                    ws,
                    cyc.as_deref_mut(),
                )?;
            }
            with_timing(prof, &mut lv.prolong, || {
                ws.fine_corr[..n].fill(0.0);
                p.spmv_scaled(1.0, &zc, 0.0, &mut ws.fine_corr[..n])
            })?;
            for i in 0..n {
                sol[i] += ws.fine_corr[i];
            }

            if t + 1 < gamma {
                with_timing(prof, &mut lv.matvec, || {
                    a.spmv_scaled(1.0, sol, 0.0, &mut ws.work[..n])
                })?;
                with_timing(prof, &mut lv.residual_axpy, || {
                    #[cfg(feature = "rayon")]
                    ws.residual[..n]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, ri)| {
                            *ri = rhs[i] - ws.work[i];
                        });
                    #[cfg(not(feature = "rayon"))]
                    for i in 0..n {
                        ws.residual[i] = rhs[i] - ws.work[i];
                    }
                });
                with_timing(prof, &mut lv.restrict, || {
                    r.spmv_scaled(1.0, &ws.residual[..n], 0.0, &mut local_coarse[..nc])
                })?;
            }
        }
        ws.coarse_rhs = local_coarse;

        // Post-smooth
        let phase_post = if level == 0 {
            RelaxPhase::Fine
        } else {
            RelaxPhase::Up
        };
        with_timing(prof, &mut lv.post_smooth, || {
            Self::apply_relax(
                pol,
                phase_post,
                RelaxWhere::Post,
                &h.levels[level],
                rhs,
                sol,
                ws,
                &self.cfg,
            )
        })?;

        if let Some(c) = cyc {
            c.per_level.push(lv);
        }
        Ok(())
    }

    fn cycle(
        &self,
        level: usize,
        gamma: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        self.cycle_profiled(level, gamma, rhs, sol, ws, None)
    }

    #[inline]
    fn v_cycle(
        &self,
        level: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        self.cycle(level, 1, rhs, sol, ws)
    }

    pub fn fmg_solve(&self, _b: &[f64], _x: &mut [f64]) -> Result<(), KError> {
        Err(KError::NotImplemented("FMG solve not yet implemented".into()))
    }

    pub fn cascade_solve(&self, _b: &[f64], _x: &mut [f64]) -> Result<(), KError> {
        Err(KError::NotImplemented("Cascade solve not yet implemented".into()))
    }

    // Convenience to avoid trait ambiguity in examples
    pub fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        Preconditioner::apply(self, side, x, y)
    }
    pub fn stats(&self) -> Option<AmgStats> {
        let mut out = self.stats.clone();
        if let (Some(s), Ok(rt)) = (out.as_mut(), self.runtime.lock()) {
            s.last_cycle = rt.last_cycle.clone();
        }
        out
    }
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
        if self.cfg.logging_level >= 2 && self.cfg.print_level >= 1 {
            if let Some(s) = self.stats.as_ref() {
                print_setup_tables(s);
            }
        }
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        if r.len() != z.len() {
            return Err(KError::InvalidInput(format!(
                "AMG.apply: r/z size mismatch: {} vs {}",
                r.len(),
                z.len()
            )));
        }
        let h = self
            .state
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("AMG not set up".into()))?;
        if h.levels.is_empty() {
            // Fallback Jacobi with diagonal of input matrix if hierarchy missing
            let a = self
                .csr
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("AMG not set up".into()))?;
            let d = diag_inv_from_csr(a)?;
            let mut ws = AMGWorkspace::new(r.len());
            Self::jacobi_smooth_sparse(self.cfg.jacobi_omega, a, &d, r, z, 10, &mut ws)
        } else {
            let mut ws = AMGWorkspace::new(h.finest().a.nrows());
            let do_prof = self.cfg.logging_level >= 2;
            let gamma = match self.cfg.cycle_type {
                CycleType::V => 1,
                CycleType::W { gamma } => gamma.max(2),
            };
            if do_prof {
                let mut cyc = CycleTimings::default();
                let t_all = tic();
                z.fill(0.0);
                self.cycle_profiled(0, gamma, r, z, &mut ws, Some(&mut cyc))?;
                cyc.total_cycle = toc(t_all);
                cyc.cycle_type = self.cfg.cycle_type;
                if let Ok(mut rt) = self.runtime.lock() {
                    rt.last_cycle = Some(cyc.clone());
                }
                if self.cfg.print_level >= 2 {
                    print_cycle_table(&cyc);
                }
            } else {
                z.fill(0.0);
                self.cycle(0, gamma, r, z, &mut ws)?;
            }
            Ok(())
        }
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        validate_truncation_and_caps(&self.cfg)?;
        let csr = csr_from_linop(op, self.cfg.drop_tol)?;
        self.refresh_numeric(&csr)?;
        self.csr = Some(csr);
        self.last_vid = Some(op.values_id());
        if self.cfg.logging_level >= 2 && self.cfg.print_level >= 1 {
            if let Some(s) = self.stats.as_ref() {
                print_setup_tables(s);
            }
        }
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
        if self.cfg.logging_level >= 2 && self.cfg.print_level >= 1 {
            if let Some(s) = self.stats.as_ref() {
                print_setup_tables(s);
            }
        }
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

fn build_hierarchy(
    fine: &CsrMatrix<f64>,
    cfg: &AMGConfig,
) -> Result<(AmgHierarchy, Option<AmgStats>), KError> {
    let mut levels: Vec<AMGLevel> = Vec::with_capacity(cfg.max_levels);
    let mut a_cur = fine.clone();
    let do_stats = cfg.logging_level > 0;
    let mut level_stats: Vec<LevelStats> = Vec::new();
    let mut timings: Vec<LevelSetupTiming> = Vec::new();
    let t_setup_all = if do_stats { Some(tic()) } else { None };

    let need_l1 = cfg
        .grid_relax_type
        .iter()
        .any(|&t| t == RelaxType::L1Jacobi);
    let need_cheb = cfg
        .grid_relax_type
        .iter()
        .any(|&t| t == RelaxType::Chebyshev);

    // Level 0 (finest)
    let mut lt0 = LevelSetupTiming::default();
    let t = tic();
    let diag0 = diag_inv_from_csr(&a_cur)?;
    if do_stats {
        lt0.diag = toc(t);
        lt0.total = lt0.diag;
    }
    let l1_inv0 = if need_l1 {
        Some(l1_diag_inv(&a_cur))
    } else {
        None
    };
    let cheb0 = if need_cheb {
        let lmax = estimate_lambda_max(&a_cur, &diag0, cfg.chebyshev_power_iters);
        let lmin = (cfg.chebyshev_min_ratio * lmax).max(1e-12);
        Some(ChebCache { lambda_max: lmax, lambda_min: lmin })
    } else {
        None
    };
    let l0 = AMGLevel {
        a: a_cur.clone(),
        p: CsrMatrix::identity(a_cur.nrows()),
        r: CsrMatrix::identity(a_cur.nrows()),
        diag_inv: diag0,
        l1_inv: l1_inv0,
        cheb: cheb0,
        agg_of: (0..a_cur.nrows()).collect(),
        is_c: Vec::new(),
        cf: None,
        p2r_pos: Vec::new(),
        a_next_pat: None,
    };
    levels.push(l0);
    if do_stats {
        level_stats.push(LevelStats {
            level: 0,
            n: a_cur.nrows(),
            nnz_a: a_cur.nnz(),
            nnz_p: 0,
            nnz_r: 0,
            max_row_sum_a: max_row_sum_abs(&a_cur),
            eff_nnz_a: Some(eff_nnz(&a_cur, cfg.stats_eps)),
        });
        timings.push(lt0);
    }

    // Drive coarsening: build levels 0..L (inclusive L is coarsest)
    for level in 0..cfg.max_levels {
        let n = a_cur.nrows();
        if n <= cfg.coarse_threshold || n <= cfg.min_coarse_size {
            break;
        }

        let mut lt = LevelSetupTiming::default();

        // 1) Strength of connection (sparse)
        let s = with_timing(do_stats, &mut lt.strength, || {
            Strength::from_csr(&a_cur, cfg.strong_threshold, cfg.normalize_strength)
        });
        // 2) Aggregates
        let mis_k = if level < cfg.agg_num_levels {
            cfg.aggressive_mis_k.max(2)
        } else {
            1
        };
        let (agg, is_c) = with_timing(do_stats, &mut lt.aggregate, || {
            build_aggregates(
                &s,
                match cfg.coarsen_type {
                    CoarsenType::RS => AggAlgo::RSGreedy,
                    CoarsenType::HMIS => AggAlgo::HMIS,
                    CoarsenType::PMIS => AggAlgo::PMIS,
                    CoarsenType::Falgout => AggAlgo::Falgout,
                },
                &AggOpts { mis_k, cap_per_row: cfg.max_strong_per_row },
            )
        });
        let tp = TentativeP {
            n_coarse: 1 + agg.iter().copied().max().unwrap_or(0),
            agg_of: agg.clone(),
        };
        let s_sym = s.symmetrize();
        let (p_csr, cf_opt): (Pcsr, Option<CFInfo>) = with_timing(do_stats, &mut lt.prolong, || {
            if matches!(cfg.interp_type, InterpType::Direct | InterpType::Standard | InterpType::Extended | InterpType::Classical) {
                let extended = matches!(cfg.interp_type, InterpType::Extended);
                let (pat, cf) = classical_pattern(&a_cur, &s_sym, &is_c, extended);
                let mut vals = vec![0.0; pat.col_idx.len()];
                let params = ClassicalParams {
                    variant: match cfg.interp_type {
                        InterpType::Direct => ClassicalVariant::Direct,
                        InterpType::Standard | InterpType::Classical | InterpType::Extended => ClassicalVariant::Standard,
                        _ => ClassicalVariant::Standard,
                    },
                    extended,
                    drop_abs: cfg.interpolation_truncation,
                    trunc_rel: cfg.truncation_factor,
                    cap_row: cfg.max_elements_per_row,
                    keep_at_least_one: true,
                };
                classical_values_only(
                    &a_cur,
                    &s_sym,
                    &cf,
                    &params,
                    &pat.row_ptr,
                    &pat.col_idx,
                    &mut vals,
                )?;
                let mut p = pat.clone();
                p.vals = vals;
                Ok((p, Some(cf)))
            } else {
                let d = diag_inv_from_csr(&a_cur)?;
                Ok((
                    smooth_tentative_sa(
                        &a_cur,
                        &d,
                        &tp,
                        cfg.jacobi_omega,
                        cfg.interpolation_truncation,
                        cfg.max_elements_per_row,
                        cfg.truncation_factor,
                    ),
                    None,
                ))
            }
        })?;
        let p = CsrMatrix::from_csr(
            p_csr.m,
            p_csr.n,
            p_csr.row_ptr.clone(),
            p_csr.col_idx.clone(),
            p_csr.vals.clone(),
        );
        // R = P^T pattern and values
        let (r_row_ptr, r_col_idx, r_vals, p2r_pos) =
            with_timing(do_stats, &mut lt.restrict, || {
                transpose_csr_with_pos(&p_csr)
            });
        let r = CsrMatrix::from_csr(
            p_csr.n,
            p_csr.m,
            r_row_ptr.clone(),
            r_col_idx.clone(),
            r_vals.clone(),
        );

        // 4) Coarse operator A_c symbolic and numeric
        let pat = with_timing(do_stats, &mut lt.rap_symbolic, || {
            rap_symbolic(&r, &a_cur, &p)
        });
        let mut a_coarse_vals = vec![0.0; pat.col_idx.len()];
        with_timing(do_stats, &mut lt.rap_numeric, || {
            rap_numeric(&pat, &r, &a_cur, &p, &mut a_coarse_vals);
        });
        {
            let mut rf = |row: usize| RowFilter {
                tau_abs: cfg.rap_truncation_abs,
                tau_rel: cfg.truncation_factor,
                k_max: cfg.rap_max_elements_per_row,
                must_keep: if cfg.keep_pivot_in_rap {
                    Some(row)
                } else {
                    None
                },
            };
            apply_filter_to_csr_values_in_place(
                pat.nrows,
                &pat.row_ptr,
                &pat.col_idx,
                &mut a_coarse_vals,
                &mut rf,
            );
        }
        let a_coarse = CsrMatrix::from_csr(
            pat.nrows,
            pat.ncols,
            pat.row_ptr.clone(),
            pat.col_idx.clone(),
            a_coarse_vals,
        );
        let diag_inv_coarse = with_timing(do_stats, &mut lt.diag, || diag_inv_from_csr(&a_coarse))?;
        lt.total = lt.strength
            + lt.aggregate
            + lt.prolong
            + lt.restrict
            + lt.rap_symbolic
            + lt.rap_numeric
            + lt.diag;
        if do_stats {
            timings.push(lt);
        }

        // Replace previous temporary P/R by actual inter-level transfers and agg mapping
        if let Some(prev) = levels.last_mut() {
            prev.p = p.clone();
            prev.r = r.clone();
            prev.agg_of = tp.agg_of.clone();
            prev.is_c = is_c.clone();
            prev.cf = cf_opt.clone();
            prev.p2r_pos = p2r_pos;
            prev.a_next_pat = Some(pat.clone());
        }

        // Next level (coarser)
        a_cur = a_coarse.clone();
        let l1_inv_coarse = if need_l1 {
            Some(l1_diag_inv(&a_cur))
        } else {
            None
        };
        let cheb_coarse = if need_cheb {
            let lmax = estimate_lambda_max(&a_cur, &diag_inv_coarse, cfg.chebyshev_power_iters);
            let lmin = (cfg.chebyshev_min_ratio * lmax).max(1e-12);
            Some(ChebCache { lambda_max: lmax, lambda_min: lmin })
        } else {
            None
        };
        levels.push(AMGLevel {
            a: a_coarse,
            p: CsrMatrix::identity(a_cur.nrows()),
            r: CsrMatrix::identity(a_cur.nrows()),
            diag_inv: diag_inv_coarse,
            l1_inv: l1_inv_coarse,
            cheb: cheb_coarse,
            agg_of: (0..a_cur.nrows()).collect(),
            is_c: Vec::new(),
            cf: None,
            p2r_pos: Vec::new(),
            a_next_pat: None,
        });

        if do_stats {
            level_stats.push(LevelStats {
                level: levels.len() - 1,
                n: a_cur.nrows(),
                nnz_a: a_cur.nnz(),
                nnz_p: 0,
                nnz_r: 0,
                max_row_sum_a: max_row_sum_abs(&a_cur),
                eff_nnz_a: Some(eff_nnz(&a_cur, cfg.stats_eps)),
            });
            let ls_len = level_stats.len();
            if ls_len >= 2 {
                if let Some(prev) = level_stats.get_mut(ls_len - 2) {
                    prev.nnz_p = p.nnz();
                    prev.nnz_r = r.nnz();
                }
            }
        }

        if a_cur.nrows() >= n {
            break;
        } // stalled
        if a_cur.nrows() <= cfg.max_coarse_size {
            break;
        }
        if let Some(limit) = cfg.max_operator_complexity {
            let oc = operator_complexity_estimate(&levels);
            if oc > limit {
                break;
            }
        }
    }

    let hier = AmgHierarchy {
        policy: RelaxPolicy {
            kind: cfg.grid_relax_type,
            sweeps: cfg.num_grid_sweeps,
            omega: cfg.jacobi_omega,
        },
        coarse_solve: cfg.coarse_solve,
        levels,
    };

    let stats_opt = if do_stats {
        let mut stats = AmgStats::from_hierarchy(&hier);
        stats.levels = level_stats;
        let mut setup = SetupTimings::default();
        setup.per_level = timings;
        if let Some(t0) = t_setup_all {
            setup.total_setup = toc(t0);
        }
        for lt in &setup.per_level {
            setup.total_symbolic +=
                lt.strength + lt.aggregate + lt.prolong + lt.restrict + lt.rap_symbolic;
            setup.total_numeric += lt.rap_numeric + lt.diag;
        }
        stats.setup = setup;
        Some(stats)
    } else {
        None
    };

    Ok((hier, stats_opt))
}

// ===== Sparse utilities (local; avoid dense on hot path) ====================

fn l1_diag_inv(a: &CsrMatrix<f64>) -> Vec<f64> {
    let n = a.nrows();
    let mut inv = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for p in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            s += a.values()[p].abs();
        }
        inv[i] = 1.0 / s.max(1e-30);
    }
    inv
}

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
            return Err(KError::SolveError(format!(
                "near-zero diagonal at row {}",
                i
            )));
        }
        d[i] = 1.0 / aii;
    }
    Ok(d)
}

fn estimate_lambda_max(a: &CsrMatrix<f64>, d_inv: &[f64], iters: usize) -> f64 {
    let n = a.nrows();
    let mut v = vec![0.0; n];
    for i in 0..n {
        v[i] = 1.0 + (i % 7) as f64 * 0.01;
    }
    let mut w = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut lam = 0.0;
    let nit = iters.max(3);
    for _ in 0..nit {
        a.spmv_scaled(1.0, &v, 0.0, &mut t).ok();
        for i in 0..n {
            w[i] = d_inv[i] * t[i];
        }
        let num = dot(&v, &w);
        let den = dot(&v, &v).max(1e-30);
        lam = num / den;
        let nw = dot(&w, &w).sqrt().max(1e-30);
        for i in 0..n {
            v[i] = w[i] / nw;
        }
    }
    lam.max(1e-12)
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

    let mut tmp_cols: Vec<usize> = Vec::new();
    let mut tmp_vals: Vec<f64> = Vec::new();
    let mut order: Vec<usize> = Vec::new();

    for i in 0..m {
        tmp_cols.clear();
        tmp_vals.clear();

        let ars = a.row_ptr()[i];
        let are = a.row_ptr()[i + 1];
        for ap in ars..are {
            let k = a.col_idx()[ap];
            let aik = a.values()[ap];
            let brs = b.row_ptr()[k];
            let bre = b.row_ptr()[k + 1];
            for bp in brs..bre {
                let j = b.col_idx()[bp];
                tmp_cols.push(j);
                tmp_vals.push(aik * b.values()[bp]);
            }
        }

        if tmp_cols.is_empty() {
            row_ptr.push(col_idx.len());
            continue;
        }

        order.clear();
        order.extend(0..tmp_cols.len());
        order.sort_unstable_by(|&u, &v| match tmp_cols[u].cmp(&tmp_cols[v]) {
            std::cmp::Ordering::Equal => u.cmp(&v),
            o => o,
        });

        let mut run_col = tmp_cols[order[0]];
        let mut acc = 0.0f64;
        for &idx in &order {
            let c = tmp_cols[idx];
            if c == run_col {
                acc += tmp_vals[idx];
            } else {
                if acc != 0.0 {
                    col_idx.push(run_col);
                    vals.push(acc);
                }
                run_col = c;
                acc = tmp_vals[idx];
            }
        }
        if acc != 0.0 {
            col_idx.push(run_col);
            vals.push(acc);
        }

        row_ptr.push(col_idx.len());
    }

    Ok(CsrMatrix::from_csr(m, n, row_ptr, col_idx, vals))
}

/// RAP = R * A * P
fn rap(
    r: &CsrMatrix<f64>,
    a: &CsrMatrix<f64>,
    p: &CsrMatrix<f64>,
) -> Result<CsrMatrix<f64>, KError> {
    let ap = csr_mul(a, p)?;
    csr_mul(r, &ap)
}

// ===== Coarsening & interpolation (dense helpers, same as old) ==============

fn compute_anisotropy(a: &Mat<f64>) -> Vec<f64> {
    let n = a.nrows();
    #[cfg(feature = "rayon")]
    return (0..n)
        .into_par_iter()
        .map(|i| {
            let diag = a[(i, i)];
            let mut max_off: f64 = 0.0;
            for j in 0..n {
                if i != j {
                    max_off = max_off.max(a[(i, j)].abs());
                }
            }
            if diag.abs() > 1e-14 {
                max_off / diag.abs()
            } else {
                0.0
            }
        })
        .collect();
    #[cfg(not(feature = "rayon"))]
    {
        let mut out = vec![0.0; n];
        for i in 0..n {
            let diag = a[(i, i)];
            let mut max_off: f64 = 0.0;
            for j in 0..n {
                if i != j {
                    max_off = max_off.max(a[(i, j)].abs());
                }
            }
            out[i] = if diag.abs() > 1e-14 {
                max_off / diag.abs()
            } else {
                0.0
            };
        }
        out
    }
}

fn compute_adaptive_threshold(a: &Mat<f64>, base_threshold: f64) -> f64 {
    let anis = compute_anisotropy(a);
    let avg = if anis.is_empty() {
        1.0
    } else {
        anis.iter().sum::<f64>() / anis.len() as f64
    };
    base_threshold * (1.0 + avg.max(0.5))
}

/// S(i,j) = |A_ij| / sqrt(|A_ii| |A_jj|) if above threshold.
fn compute_strength_matrix(a: &Mat<f64>, thr: f64) -> Mat<f64> {
    let n = a.nrows();
    let mut s = Mat::<f64>::zeros(n, n);
    let mut diag = vec![0.0; n];
    for i in 0..n {
        diag[i] = a[(i, i)].abs();
    }
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let denom = (diag[i] * diag[j]).sqrt();
            if denom > 1e-14 {
                let st = a[(i, j)].abs() / denom;
                if st > thr {
                    s[(i, j)] = st;
                }
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
        if vis[i] {
            continue;
        }
        let mut best = None;
        let mut bestv = 0.0;
        for j in 0..n {
            if i == j || vis[j] {
                continue;
            }
            let v = s[(i, j)];
            if v > bestv {
                bestv = v;
                best = Some(j);
            }
        }
        if let Some(j) = best {
            agg[i] = id;
            agg[j] = id;
            vis[i] = true;
            vis[j] = true;
            id += 1;
        } else {
            agg[i] = id;
            vis[i] = true;
            id += 1;
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
            if v != 0.0 {
                cg[(ai, aj)] += v;
            }
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
    order.sort_by(|a, b| match b.0.total_cmp(&a.0) {
        std::cmp::Ordering::Equal => a.1.cmp(&b.1),
        o => o,
    });

    for &(_, seed) in &order {
        if agg[seed] != usize::MAX {
            continue;
        }
        agg[seed] = next;
        // pick strongest distinct neighbors
        let mut neigh: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != seed && agg[j] == usize::MAX)
            .map(|j| (s[(seed, j)], j))
            .collect();
        neigh.sort_by(|a, b| match b.0.total_cmp(&a.0) {
            std::cmp::Ordering::Equal => a.1.cmp(&b.1),
            o => o,
        });
        for &(_, j) in neigh.iter() {
            if (0..n).filter(|&i| agg[i] == next).count() >= max_sz {
                break;
            }
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
    for (i, &g) in aggregates.iter().enumerate() {
        p[(i, g)] = 1.0;
    }
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
        for j in 0..n {
            norm2 += p[(i, j)] * p[(i, j)];
        }
        let s = if norm2 > 1e-14 { norm2.sqrt() } else { 1.0 };
        for j in 0..n {
            p[(i, j)] /= s;
        }
    }
}

// ===== Small sparse CG for coarsest level ===================================

fn cg_sparse(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    maxit: usize,
) -> Result<(), KError> {
    let n = a.nrows();
    if n == 0 {
        return Ok(());
    }
    x.fill(0.0);

    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut ap = vec![0.0; n];

    let mut rsold = dot(&r, &r);
    let atol = tol.max(1e-12) * rsold.sqrt().max(1e-30);

    for _ in 0..maxit {
        a.spmv_scaled(1.0, &p, 0.0, &mut ap)?;
        let denom = dot(&p, &ap);
        if denom.abs() < 1e-30 {
            break;
        }
        let alpha = rsold / denom;

        #[cfg(feature = "rayon")]
        {
            x.par_iter_mut()
                .zip(p.par_iter())
                .for_each(|(xi, &pi)| *xi += alpha * pi);
        }
        #[cfg(not(feature = "rayon"))]
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        #[cfg(feature = "rayon")]
        {
            r.par_iter_mut()
                .zip(ap.par_iter())
                .for_each(|(ri, &api)| *ri -= alpha * api);
        }
        #[cfg(not(feature = "rayon"))]
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        let rsnew = dot(&r, &r);
        if rsnew.sqrt() < atol {
            break;
        }
        let beta = rsnew / rsold;

        #[cfg(feature = "rayon")]
        {
            p.par_iter_mut()
                .zip(r.par_iter())
                .for_each(|(pi, &ri)| *pi = ri + beta * *pi);
        }
        #[cfg(not(feature = "rayon"))]
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }
    Ok(())
}

#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() {
        s += x[i] * y[i];
    }
    s
}

// ===== Helpers for transpose mapping and stats ==============================

#[derive(Clone, Debug)]
pub struct LevelStats {
    pub level: usize,
    pub n: usize,
    pub nnz_a: usize,
    pub nnz_p: usize,
    pub nnz_r: usize,
    pub max_row_sum_a: f64,
    pub eff_nnz_a: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct LevelSetupTiming {
    pub strength: Duration,
    pub aggregate: Duration,
    pub prolong: Duration,
    pub restrict: Duration,
    pub rap_symbolic: Duration,
    pub rap_numeric: Duration,
    pub diag: Duration,
    pub total: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct SetupTimings {
    pub per_level: Vec<LevelSetupTiming>,
    pub total_setup: Duration,
    pub total_symbolic: Duration,
    pub total_numeric: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct CycleLevelTiming {
    pub level: usize,
    pub pre_smooth: Duration,
    pub matvec: Duration,
    pub residual_axpy: Duration,
    pub restrict: Duration,
    pub coarse_solve: Duration,
    pub prolong: Duration,
    pub post_smooth: Duration,
}

#[derive(Clone, Debug)]
pub struct CycleTimings {
    pub per_level: Vec<CycleLevelTiming>,
    pub total_cycle: Duration,
    pub cycle_type: CycleType,
}

impl Default for CycleTimings {
    fn default() -> Self {
        Self {
            per_level: Vec::new(),
            total_cycle: Duration::default(),
            cycle_type: CycleType::V,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AmgStats {
    pub grid_complexity: f64,
    pub operator_complexity: f64,
    pub num_levels: usize,
    pub levels: Vec<LevelStats>,
    pub setup: SetupTimings,
    pub last_cycle: Option<CycleTimings>,
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
            levels: Vec::new(),
            setup: SetupTimings::default(),
            last_cycle: None,
        }
    }
}

#[derive(Default)]
struct AmgRuntime {
    last_cycle: Option<CycleTimings>,
}

fn operator_complexity_estimate(levels: &[AMGLevel]) -> f64 {
    if levels.is_empty() {
        return 0.0;
    }
    let nnz0 = levels[0].a.nnz() as f64;
    let nnz_sum: f64 = levels.iter().map(|l| l.a.nnz() as f64).sum();
    nnz_sum / nnz0
}

fn collect_level_stats(h: &AmgHierarchy, cfg: &AMGConfig) -> Vec<LevelStats> {
    let mut out = Vec::with_capacity(h.levels.len());
    for (i, lvl) in h.levels.iter().enumerate() {
        out.push(LevelStats {
            level: i,
            n: lvl.a.nrows(),
            nnz_a: lvl.a.nnz(),
            nnz_p: if i < h.coarsest_ix() { lvl.p.nnz() } else { 0 },
            nnz_r: if i < h.coarsest_ix() { lvl.r.nnz() } else { 0 },
            max_row_sum_a: max_row_sum_abs(&lvl.a),
            eff_nnz_a: Some(eff_nnz(&lvl.a, cfg.stats_eps)),
        });
    }
    out
}

fn print_setup_tables(stats: &AmgStats) {
    if stats.levels.is_empty() {
        return;
    }
    println!(
        "AMG hierarchy: {} levels\nGrid complexity: {:.3}, Operator complexity: {:.3}",
        stats.num_levels, stats.grid_complexity, stats.operator_complexity
    );
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "lev", "n", "nnz(A)", "nnz(P)", "nnz(R)", "max_row_sum"
    );
    for ls in &stats.levels {
        println!(
            "{:>5} {:>10} {:>10} {:>10} {:>10} {:>12.4e}",
            ls.level, ls.n, ls.nnz_a, ls.nnz_p, ls.nnz_r, ls.max_row_sum_a
        );
    }
    if !stats.setup.per_level.is_empty() {
        println!("Setup timings (ms): level | strength agg prolon restr symRAP numRAP diag total");
        let ms = |d: Duration| (d.as_secs_f64() * 1e3).round() as u64;
        for (i, lt) in stats.setup.per_level.iter().enumerate() {
            println!(
                "{:>5} {:>9} {:>3} {:>6} {:>5} {:>7} {:>8} {:>4} {:>6}",
                i,
                ms(lt.strength),
                ms(lt.aggregate),
                ms(lt.prolong),
                ms(lt.restrict),
                ms(lt.rap_symbolic),
                ms(lt.rap_numeric),
                ms(lt.diag),
                ms(lt.total)
            );
        }
        println!(
            "Total setup: {} ms (symbolic {} ms, numeric {} ms)",
            ms(stats.setup.total_setup),
            ms(stats.setup.total_symbolic),
            ms(stats.setup.total_numeric)
        );
    }
}

fn print_cycle_table(c: &CycleTimings) {
    let desc = match c.cycle_type {
        CycleType::V => "V-cycle".to_string(),
        CycleType::W { gamma } => format!("W-cycle(gamma={})", gamma),
    };
    println!("{} timings (ms): level | pre mv axpy R coarse P post", desc);
    let ms = |d: Duration| (d.as_secs_f64() * 1e3).round() as u64;
    for lv in &c.per_level {
        println!(
            "{:>5} {:>4} {:>2} {:>4} {:>1} {:>6} {:>1} {:>4}",
            lv.level,
            ms(lv.pre_smooth),
            ms(lv.matvec),
            ms(lv.residual_axpy),
            ms(lv.restrict),
            ms(lv.coarse_solve),
            ms(lv.prolong),
            ms(lv.post_smooth)
        );
    }
    println!("Total cycle: {} ms", ms(c.total_cycle));
}

#[inline]
fn tic() -> Instant {
    Instant::now()
}
#[inline]
fn toc(t0: Instant) -> Duration {
    t0.elapsed()
}

fn max_row_sum_abs(a: &CsrMatrix<f64>) -> f64 {
    let n = a.nrows();
    let rp = a.row_ptr();
    let vv = a.values();
    #[cfg(feature = "rayon")]
    {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut s = 0.0;
                for p in rp[i]..rp[i + 1] {
                    s += vv[p].abs();
                }
                s
            })
            .reduce(|| 0.0, |x, y| x.max(y))
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut m = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for p in rp[i]..rp[i + 1] {
                s += vv[p].abs();
            }
            if s > m {
                m = s;
            }
        }
        m
    }
}

fn eff_nnz(a: &CsrMatrix<f64>, eps: f64) -> usize {
    if eps <= 0.0 {
        return a.nnz();
    }
    a.values().iter().filter(|&&v| v.abs() >= eps).count()
}

#[inline]
fn with_timing<F, R>(enabled: bool, acc: &mut Duration, f: F) -> R
where
    F: FnOnce() -> R,
{
    if enabled {
        let t = tic();
        let out = f();
        *acc += toc(t);
        out
    } else {
        f()
    }
}

fn transpose_csr_with_pos(p: &Pcsr) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<usize>) {
    // Compute transpose pattern and values, and mapping from P entry index -> R entry index
    let (m, n) = (p.m, p.n);
    let nnz = p.col_idx.len();
    let mut r_row_counts = vec![0usize; n + 1];
    for &cj in &p.col_idx {
        r_row_counts[cj + 1] += 1;
    }
    for i in 0..n {
        r_row_counts[i + 1] += r_row_counts[i];
    }
    let mut r_col_idx = vec![0usize; nnz];
    let mut r_vals = vec![0.0f64; nnz];
    let mut r_row_next = r_row_counts.clone();
    let mut p2r_pos = vec![0usize; nnz];
    for i in 0..m {
        let rs = p.row_ptr[i];
        let re = p.row_ptr[i + 1];
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
    use std::cmp::Ordering;

    #[inline]
    fn feq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
        let diff = (a - b).abs();
        diff <= atol.max(rtol * a.abs()).max(rtol * b.abs())
    }

    fn assert_dense_eq(a: &Mat<f64>, b: &Mat<f64>, atol: f64, rtol: f64) {
        assert_eq!(a.nrows(), b.nrows());
        assert_eq!(a.ncols(), b.ncols());
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    feq(a[(i, j)], b[(i, j)], atol, rtol),
                    "dense mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    fn csr_from_triples(m: usize, n: usize, mut trip: Vec<(usize, usize, f64)>) -> CsrMatrix<f64> {
        trip.sort_by(|a, b| match a.0.cmp(&b.0) {
            Ordering::Equal => a.1.cmp(&b.1),
            o => o,
        });
        let mut row_ptr = vec![0usize; m + 1];
        let mut col_idx = Vec::<usize>::new();
        let mut vals = Vec::<f64>::new();
        let mut i_cur = 0usize;
        let mut j_prev = usize::MAX;
        let mut acc = 0.0;

        let mut push_acc = |row: usize,
                            col: usize,
                            v: f64,
                            row_ptr: &mut [usize],
                            col_idx: &mut Vec<usize>,
                            vals: &mut Vec<f64>| {
            if v != 0.0 {
                col_idx.push(col);
                vals.push(v);
            }
            row_ptr[row + 1] = col_idx.len();
        };

        for (r, c, v) in trip {
            while i_cur < r {
                if j_prev != usize::MAX {
                    push_acc(i_cur, j_prev, acc, &mut row_ptr, &mut col_idx, &mut vals);
                    j_prev = usize::MAX;
                    acc = 0.0;
                }
                i_cur += 1;
                row_ptr[i_cur] = col_idx.len();
            }
            if j_prev == c {
                acc += v;
            } else {
                if j_prev != usize::MAX {
                    push_acc(i_cur, j_prev, acc, &mut row_ptr, &mut col_idx, &mut vals);
                }
                j_prev = c;
                acc = v;
            }
        }
        while i_cur < m {
            if j_prev != usize::MAX {
                push_acc(i_cur, j_prev, acc, &mut row_ptr, &mut col_idx, &mut vals);
                j_prev = usize::MAX;
                acc = 0.0;
            }
            i_cur += 1;
            row_ptr[i_cur] = col_idx.len();
        }

        CsrMatrix::from_csr(m, n, row_ptr, col_idx, vals)
    }

    fn identity_level() -> AMGLevel {
        AMGLevel {
            a: CsrMatrix::identity(1),
            p: CsrMatrix::identity(1),
            r: CsrMatrix::identity(1),
            diag_inv: vec![1.0],
            l1_inv: None,
            cheb: None,
            agg_of: vec![0],
            is_c: Vec::new(),
            cf: None,
            p2r_pos: vec![],
            a_next_pat: None,
        }
    }

    #[test]
    fn phase_selection_logic() {
        reset_relax_counts();
        let levels = vec![identity_level(), identity_level(), identity_level()];
        let policy = RelaxPolicy {
            kind: [RelaxType::Jacobi; 4],
            sweeps: [1, 1, 1, 0],
            omega: 1.0,
        };
        let hier = AmgHierarchy {
            levels,
            policy,
            coarse_solve: CoarseSolve::DirectDense,
        };
        let amg = AMG {
            state: Some(hier),
            ..Default::default()
        };
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
        cfg.grid_relax_type = [RelaxType::HybridGaussSeidel; 4];
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
        let amg = AMG::builder()
            .smoothing_sweeps(2, 3)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        assert_eq!(amg.cfg.num_grid_sweeps, [2, 2, 3, 1]);
    }

    #[test]
    fn rap_numeric_matches_dense_small() {
        let a = csr_from_triples(
            3,
            3,
            vec![
                (0, 0, 4.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 4.0),
            ],
        );
        let p = csr_from_triples(3, 2, vec![(0, 0, 1.0), (1, 0, 1.0), (2, 1, 1.0)]);
        let r = csr_from_triples(2, 3, vec![(0, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0)]);

        let pat = rap_ops::rap_symbolic(&r, &a, &p);
        let mut vals = vec![0.0; pat.col_idx.len()];
        rap_ops::rap_numeric(&pat, &r, &a, &p, &mut vals);

        let ad = a.to_dense();
        let pd = p.to_dense();
        let rd = r.to_dense();
        let cd = &rd * &ad * &pd;

        let mut cpat = Mat::<f64>::zeros(pat.nrows, pat.ncols);
        for i in 0..pat.nrows {
            for k in pat.row_ptr[i]..pat.row_ptr[i + 1] {
                let j = pat.col_idx[k];
                cpat[(i, j)] = vals[k];
            }
        }
        assert_dense_eq(&cpat, &cd, 1e-12, 1e-12);
    }

    #[test]
    fn transpose_bijection_and_values_small() {
        let m = 3;
        let n = 4;
        let p = prolong::Pcsr {
            m,
            n,
            row_ptr: vec![0, 2, 3, 5],
            col_idx: vec![0, 2, 1, 1, 3],
            vals: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let (rr, rc, rv, p2r) = super::transpose_csr_with_pos(&p);
        assert_eq!(rr.len(), n + 1);
        assert_eq!(rc.len(), p.col_idx.len());
        assert_eq!(rv.len(), p.vals.len());

        let nnz = p.vals.len();
        let mut seen = vec![false; nnz];
        for &q in &p2r {
            assert!(q < nnz);
            assert!(!seen[q]);
            seen[q] = true;
        }
        assert!(seen.into_iter().all(|b| b));

        for (pi, &ri) in p2r.iter().enumerate() {
            assert!(feq(p.vals[pi], rv[ri], 0.0, 0.0));
        }

        let mut p_dense = Mat::<f64>::zeros(m, n);
        for i in 0..m {
            for k in p.row_ptr[i]..p.row_ptr[i + 1] {
                p_dense[(i, p.col_idx[k])] = p.vals[k];
            }
        }
        let r_dense = p_dense.transpose().to_owned();
        let mut r_pat = Mat::<f64>::zeros(n, m);
        for i in 0..n {
            for k in rr[i]..rr[i + 1] {
                r_pat[(i, rc[k])] = rv[k];
            }
        }
        assert_dense_eq(&r_pat, &r_dense, 0.0, 0.0);
    }

    fn poisson1d(n: usize) -> CsrMatrix<f64> {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        row_ptr.push(0);
        for i in 0..n {
            if i > 0 {
                col_idx.push(i - 1);
                vals.push(-1.0);
            }
            col_idx.push(i);
            vals.push(2.0);
            if i + 1 < n {
                col_idx.push(i + 1);
                vals.push(-1.0);
            }
            row_ptr.push(col_idx.len());
        }
        CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
    }

    #[test]
    fn gs_symgs_residual() {
        let a = poisson1d(3);
        let d = diag_inv_from_csr(&a).unwrap();
        let rhs = vec![1.0; 3];
        let mut zf = vec![0.0; 3];
        AMG::gs_forward(1.0, &a, &d, &rhs, &mut zf, 1).unwrap();
        let mut work = vec![0.0; 3];
        a.spmv_scaled(1.0, &zf, 0.0, &mut work).unwrap();
        let mut res_f = vec![0.0; 3];
        for i in 0..3 {
            res_f[i] = rhs[i] - work[i];
        }
        let norm_f = dot(&res_f, &res_f);

        let mut zs = vec![0.0; 3];
        AMG::sym_gs(1.0, &a, &d, &rhs, &mut zs, 1).unwrap();
        a.spmv_scaled(1.0, &zs, 0.0, &mut work).unwrap();
        let mut res_s = vec![0.0; 3];
        for i in 0..3 {
            res_s[i] = rhs[i] - work[i];
        }
        let norm_s = dot(&res_s, &res_s);
        assert!(norm_s < norm_f);
    }

    #[test]
    fn l1_jacobi_no_worse_than_jacobi() {
        let a = poisson1d(3);
        let d = diag_inv_from_csr(&a).unwrap();
        let l1 = l1_diag_inv(&a);
        let rhs = vec![1.0; 3];
        let mut z_j = vec![0.0; 3];
        let mut z_l1 = vec![0.0; 3];
        let mut ws_j = AMGWorkspace::new(3);
        let mut ws_l1 = AMGWorkspace::new(3);
        AMG::jacobi_smooth_sparse(1.0, &a, &d, &rhs, &mut z_j, 1, &mut ws_j).unwrap();
        AMG::l1_jacobi(1.0, &a, &l1, &rhs, &mut z_l1, 1, &mut ws_l1).unwrap();
        a.spmv_scaled(1.0, &z_j, 0.0, &mut ws_j.work[..3]).unwrap();
        let mut rj = 0.0;
        for i in 0..3 {
            let ri = rhs[i] - ws_j.work[i];
            rj += ri * ri;
        }
        a.spmv_scaled(1.0, &z_l1, 0.0, &mut ws_l1.work[..3]).unwrap();
        let mut rl1 = 0.0;
        for i in 0..3 {
            let ri = rhs[i] - ws_l1.work[i];
            rl1 += ri * ri;
        }
        let r0 = 3.0; // initial residual norm squared for rhs=[1,1,1]
        assert!(rj < r0);
        assert!(rl1 < r0);
    }

    #[test]
    fn chebyshev_degree_improves() {
        let n = 10;
        let a = poisson1d(n);
        let d = diag_inv_from_csr(&a).unwrap();
        let lmax = estimate_lambda_max(&a, &d, 10);
        assert!(lmax > 1.8 && lmax < 2.1);
        let lmin = 0.3 * lmax;
        let rhs = vec![1.0; n];
        let mut ws = AMGWorkspace::new(n);
        let mut z2 = vec![0.0; n];
        AMG::chebyshev_smooth(&a, &d, &rhs, &mut z2, 2, lmin, lmax, &mut ws).unwrap();
        a.spmv_scaled(1.0, &z2, 0.0, &mut ws.work[..n]).unwrap();
        let mut r2 = 0.0;
        for i in 0..n {
            let ri = rhs[i] - ws.work[i];
            r2 += ri * ri;
        }
        let mut z4 = vec![0.0; n];
        AMG::chebyshev_smooth(&a, &d, &rhs, &mut z4, 4, lmin, lmax, &mut ws).unwrap();
        a.spmv_scaled(1.0, &z4, 0.0, &mut ws.work[..n]).unwrap();
        let mut r4 = 0.0;
        for i in 0..n {
            let ri = rhs[i] - ws.work[i];
            r4 += ri * ri;
        }
        let mut z8 = vec![0.0; n];
        AMG::chebyshev_smooth(&a, &d, &rhs, &mut z8, 8, lmin, lmax, &mut ws).unwrap();
        a.spmv_scaled(1.0, &z8, 0.0, &mut ws.work[..n]).unwrap();
        let mut r8 = 0.0;
        for i in 0..n {
            let ri = rhs[i] - ws.work[i];
            r8 += ri * ri;
        }
        assert!(r4 < r2);
        assert!(r8 < r4);
    }

    #[test]
    fn refresh_updates_caches() {
        let a = poisson1d(4);
        let mut amg_l1 = AMGBuilder::new()
            .grid_relax_type_all(RelaxType::L1Jacobi)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        amg_l1.setup(&a).unwrap();
        let old = amg_l1.state.as_ref().unwrap().levels[0]
            .l1_inv
            .as_ref()
            .unwrap()[0];
        let mut a2 = a.clone();
        let rp = a2.row_ptr();
        for p in rp[0]..rp[1] {
            a2.values_mut()[p] *= 2.0;
        }
        amg_l1.update_numeric(&a2).unwrap();
        let new = amg_l1.state.as_ref().unwrap().levels[0]
            .l1_inv
            .as_ref()
            .unwrap()[0];
        assert!((new - old).abs() > 1e-12);

        let mut amg_ch = AMGBuilder::new()
            .grid_relax_type_all(RelaxType::Chebyshev)
            .chebyshev_recompute(true)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        amg_ch.setup(&a).unwrap();
        let old_l = amg_ch.state.as_ref().unwrap().levels[0]
            .cheb
            .as_ref()
            .unwrap()
            .lambda_max;
        let mut a3 = a.clone();
        let rp3 = a3.row_ptr();
        for p in rp3[0]..rp3[1] {
            if a3.col_idx()[p] == 0 {
                a3.values_mut()[p] *= 1.5;
            }
        }
        amg_ch.update_numeric(&a3).unwrap();
        let new_l = amg_ch.state.as_ref().unwrap().levels[0]
            .cheb
            .as_ref()
            .unwrap()
            .lambda_max;
        assert!((new_l - old_l).abs() > 1e-6);
    }
}
