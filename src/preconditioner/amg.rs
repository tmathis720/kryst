#![allow(dead_code)]

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::config::kinds::{AmgCoarsenKind, AmgInterpKind, AmgRelaxKind};
use crate::config::options::PcOptions;
use crate::error::KError;
use crate::matrix::op::{LinOp, StructureId, ValuesId};
use crate::matrix::{
    convert::csr_from_linop,
    dense_api::DenseMatRef,
    sparse::CsrMatrix,
    spmv::{csr_spmm_dense, spmv_scaled_f32_on_pattern},
};
#[cfg(not(feature = "complex"))]
use crate::matrix::DistCsrOp;
#[cfg(not(feature = "complex"))]
use crate::matrix::dist::halo::HaloPlan;
#[cfg(feature = "complex")]
use crate::matrix::parcsr::HaloPlan;
#[cfg(feature = "simd")]
use crate::matrix::{spmv::SpmvTuning, utils};
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::dist::DistCoarseStrategy;
use crate::preconditioner::asm::{Asm, AsmCombine, AsmConfig, AsmLocalSolver};
#[cfg(feature = "complex")]
use crate::preconditioner::bridge::{
    apply_pc_mut_s as bridge_apply_pc_mut_s, apply_pc_s as bridge_apply_pc_s,
};
use crate::preconditioner::chebyshev::{self, ChebBounds};
use crate::preconditioner::deflation::{AmgCoarseSpace, DeflationOptions, ZSource};
use crate::preconditioner::ilu_csr::{
    IluCsr, IluCsrConfig, IluKind, PivotStrategy, ReorderingOptions,
};
use crate::preconditioner::{PcCaps, PcSide, Preconditioner};
use crate::utils::conditioning::{ConditioningOptions, apply_csr_transforms};
use faer::Mat;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// New sparse SA/RS submodules
mod coarse_solver;
pub use coarse_solver::CoarseSolve;
pub mod coarsen;
mod non_galerkin;
pub(crate) mod prolong;
pub use prolong::AdaptiveWeight;
mod rap_ops;
mod row_filter;
pub mod strength;
pub(crate) mod strength_nodal;
pub(crate) mod util;

use coarse_solver::{CoarseDenseLu, CoarseIlu, CoarseSolver};
use coarsen::{
    AggAlgo, AggOpts, build_aggregates, build_aggregates_nodal, lift_node_aggregates_to_dofs,
};
use non_galerkin::{NgRowFilter, non_galerkin_filter_coarse};
use prolong::{
    CFInfo, ClassicalParams, ClassicalVariant, Pcsr, TentativeNodal, TentativeP,
    adaptive_fit_values_only, classical_pattern, classical_values_only, restrict_samples_to_coarse,
    sample_low_modes, smooth_sa_values_only, smooth_sa_values_only_mf, smooth_sa_values_only_multi,
    smooth_tentative_sa_mf, smooth_tentative_sa_multi,
};
use rap_ops::{CsrPattern, rap_numeric, rap_symbolic};
use row_filter::{
    RowFilter, apply_filter_to_csr_values_in_place, compensate_nodal_diag, compensate_scalar_rows,
    restrict_trials,
};
use strength::Strength;
use strength_nodal::strength_nodal_from_csr;
use util::DofLayout;

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
    HE,
}

/// Response when rank diagnostics flag interpolation issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankFallback {
    RetryLooserInterp,
    SwitchInterpKind,
    Reaggregate,
    Abort,
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
    ChebyshevSafe,
    SafeguardedGaussSeidel,
    Ilu0,
    Ras,
    Fsai,
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
use std::cell::Cell;
#[cfg(test)]
thread_local! {
    static RELAX_CALL_COUNTS: Cell<[usize; 4]> = Cell::new([0; 4]);
}
#[cfg(test)]
thread_local! {
    static BUILD_SYMBOLIC_COUNT: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

#[cfg(test)]
pub fn reset_relax_counts() {
    RELAX_CALL_COUNTS.with(|counts| counts.set([0; 4]));
}
#[cfg(test)]
pub fn get_relax_counts() -> [usize; 4] {
    RELAX_CALL_COUNTS.with(|counts| counts.get())
}

// ===== Config + Builder ======================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CycleType {
    #[default]
    V,
    W {
        gamma: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KrylovAlgo {
    FCG,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KCycle {
    /// Levels (0 = finest) where the Krylov smoother is enabled.
    pub levels: Vec<usize>,
    /// Number of Krylov iterations when enabled.
    pub iters: usize,
    pub algo: KrylovAlgo,
    /// Apply the Krylov smoother after the standard post-smoothing step.
    pub place_post: bool,
    /// Apply the Krylov smoother before the standard post-smoothing step.
    pub place_pre: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedPrecision {
    pub smooth: bool,
    pub residual: bool,
}

impl MixedPrecision {
    #[inline]
    fn smoothers_enabled(self) -> bool {
        self.smooth
    }

    #[inline]
    fn residual_enabled(self) -> bool {
        self.residual
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedStorage {
    Cached,
    Transient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NgSymmetry {
    None,
    Symmetric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodalMode {
    Off,
    Nodal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowScaleMode {
    /// For scalar or when near-nullspace is provided: enforce per-row reproduction
    /// target by scaling the subset of columns for each function α.
    ToNearNullspace,
    /// Fallback for scalar r=1 without NNS: enforce ∑ P[i,*] = 1.
    SumToOne,
    /// Make each row’s α-slice unit L2 norm (robust when T is noisy).
    L2Unit,
    /// Diagonal-weighted norm: sqrt(p^T D p) = 1 using D = diag(A).
    DUnit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PostInterpType {
    None,
    /// Pure row-scaling family (stable, cheap, deterministic)
    RowScaling(RowScaleMode),
    /// Optional orthonormalization of per-aggregate blocks
    LocalQR,
    /// Extra SA-like smoothing passes on P values (fixed pattern).
    EnergyPolish {
        sweeps: usize,
        omega: f64,
    },
}

#[derive(Clone, Debug)]
pub struct NearNullspace {
    pub basis: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct NonGalerkin {
    pub enabled: bool,
    pub start_level: usize,
    pub drop_abs: f64,
    pub drop_rel: f64,
    pub cap_row: usize,
    pub symmetry: NgSymmetry,
    pub lump_diagonal: bool,
    pub oc_target: Option<f64>,
    pub oc_max_iter: usize,
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
    pub keep_transpose: bool,
    pub keep_pivot_in_rap: bool,
    pub require_spd: bool,
    pub spd_diag_floor: f64,
    pub forbid_non_galerkin_in_spd: bool,
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
    pub adaptive_interp: bool,
    pub adaptive_samples: usize,
    pub adaptive_smooth_steps: usize,
    pub adaptive_smooth_omega: f64,
    pub adaptive_lambda: f64,
    pub adaptive_enforce_sum1: bool,
    pub adaptive_weight_mode: AdaptiveWeight,
    pub chebyshev_recompute_esteig: bool,
    pub chebyshev_safety: f64,
    pub chebyshev_lower_ratio: f64,
    pub chebyshev_power_steps: usize,
    pub chebyshev_degree: usize,
    pub use_level_scheduling: bool,
    pub drop_tol: f64,  // NEW: used for dense->CSR conversion
    pub stats_eps: f64, // threshold for effective nnz reporting
    // P rank diagnostics
    pub verify_p_rank: bool,
    pub rank_sketch_cols: usize,
    pub rank_cond_threshold: f64,
    pub rank_min_col_norm: f64,
    // Galerkin sampling controls
    pub verify_galerkin: bool,
    pub galerkin_samples: usize,
    pub galerkin_rel_tol: f64,
    // Rank fallback policy
    pub on_rank_failure: RankFallback,
    // FSAI smoother controls
    pub fsai_dist: usize,
    pub fsai_use_strength: bool,
    pub fsai_max_per_row: usize,
    pub fsai_lambda: f64,
    pub fsai_adaptive_passes: usize,
    pub fsai_drop_tol: f64,
    pub fsai_damping: f64,
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
    pub kcycle: Option<KCycle>,
    pub fmg_nu_pre: usize,
    pub fmg_nu_post: usize,
    pub fmg_gamma: usize,
    pub fmg_levels_use: Option<usize>,
    pub non_galerkin: NonGalerkin,
    pub nodal: NodalMode,
    pub block_size: usize,
    pub num_functions: usize,
    pub near_nullspace: Option<NearNullspace>,
    pub filter_trial_vectors: Option<Vec<Vec<f64>>>,
    pub filter_omega: f64,
    pub filter_after_non_galerkin: bool,
    pub post_interp: PostInterpType,
    pub flexible_level: Option<usize>,
    pub flexible_iters: usize,
    pub flexible_rtol: f64,
    pub flexible_pc_sweeps: usize,
    pub mixed_precision: Option<MixedPrecision>,
    pub mixed_storage: MixedStorage,
    pub conditioning: ConditioningOptions,
    pub dist_coarse_strategy: DistCoarseStrategy,
    pub dist_apply_instrumentation: bool,
    pub dist_coarse_ghost_scale: f64,
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
            keep_transpose: true,
            keep_pivot_in_rap: true,
            require_spd: true,
            spd_diag_floor: 0.0,
            forbid_non_galerkin_in_spd: true,
            grid_relax_type: [RelaxType::GaussSeidel; 4],
            num_grid_sweeps: [1; 4],
            pre_sweeps: 1,
            post_sweeps: 1,
            coarsen_type: CoarsenType::HMIS,
            interp_type: InterpType::Extended,
            relax_type: RelaxType::SymmetricGaussSeidel,
            logging_level: 0,
            print_level: 0,
            tolerance: 1e-6,
            max_iterations: 20,
            min_iterations: 0,
            ieee_checks: true,
            optimize_workspace: true,
            jacobi_omega: 2.0 / 3.0,
            adaptive_interp: false,
            adaptive_samples: 4,
            adaptive_smooth_steps: 3,
            adaptive_smooth_omega: 2.0 / 3.0,
            adaptive_lambda: 1e-10,
            adaptive_enforce_sum1: true,
            adaptive_weight_mode: AdaptiveWeight::Diag,
            chebyshev_recompute_esteig: true,
            chebyshev_safety: 1.10,
            chebyshev_lower_ratio: 0.10,
            chebyshev_power_steps: 4,
            chebyshev_degree: 2,
            use_level_scheduling: false,
            drop_tol: 1e-12,
            stats_eps: 1e-12,
            verify_p_rank: true,
            rank_sketch_cols: 8,
            rank_cond_threshold: 1e8,
            rank_min_col_norm: 1e-12,
            verify_galerkin: true,
            galerkin_samples: 2,
            galerkin_rel_tol: 1e-10,
            on_rank_failure: RankFallback::RetryLooserInterp,
            fsai_dist: 1,
            fsai_use_strength: true,
            fsai_max_per_row: 12,
            fsai_lambda: 1e-12,
            fsai_adaptive_passes: 0,
            fsai_drop_tol: 1e-12,
            fsai_damping: 0.3,
            normalize_strength: true,
            coarse_solve: CoarseSolve::CG,
            ilu_drop_tol: 1e-2,
            ilu_fill_per_row: 0,
            max_operator_complexity: None,
            agg_num_levels: 1,
            aggressive_mis_k: 2,
            max_strong_per_row: None,
            cycle_type: CycleType::V,
            kcycle: None,
            fmg_nu_pre: 1,
            fmg_nu_post: 1,
            fmg_gamma: 1,
            fmg_levels_use: None,
            non_galerkin: NonGalerkin {
                enabled: false,
                start_level: 1,
                drop_abs: 0.0,
                drop_rel: 0.0,
                cap_row: 0,
                symmetry: NgSymmetry::Symmetric,
                lump_diagonal: true,
                oc_target: None,
                oc_max_iter: 4,
            },
            nodal: NodalMode::Off,
            block_size: 1,
            num_functions: 1,
            near_nullspace: None,
            filter_trial_vectors: None,
            filter_omega: 1.0,
            filter_after_non_galerkin: true,
            post_interp: PostInterpType::None,
            flexible_level: None,
            flexible_iters: 0,
            flexible_rtol: 0.0,
            flexible_pc_sweeps: 1,
            mixed_precision: None,
            mixed_storage: MixedStorage::Cached,
            conditioning: ConditioningOptions::default(),
            dist_coarse_strategy: DistCoarseStrategy::RootGather,
            dist_apply_instrumentation: false,
            dist_coarse_ghost_scale: 0.0,
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

impl AMGConfig {
    fn validate(&self) -> Result<(), KError> {
        validate_relax_policy(self, self.coarse_solve)?;
        validate_truncation_and_caps(self)?;
        if self.max_levels == 0 {
            return Err(KError::InvalidInput("max_levels must be at least 1".into()));
        }
        if self.min_coarse_size == 0 {
            return Err(KError::InvalidInput(
                "min_coarse_size must be at least 1".into(),
            ));
        }
        if self.max_coarse_size > 0 && self.max_coarse_size < self.min_coarse_size {
            return Err(KError::InvalidInput(
                "max_coarse_size must be ≥ min_coarse_size".into(),
            ));
        }
        if self.max_iterations < self.min_iterations {
            return Err(KError::InvalidInput(
                "max_iterations must be ≥ min_iterations".into(),
            ));
        }
        if self.jacobi_omega <= 0.0 {
            return Err(KError::InvalidInput("jacobi_omega must be positive".into()));
        }
        if self.chebyshev_power_steps == 0 {
            return Err(KError::InvalidInput(
                "chebyshev_power_steps must be ≥ 1".into(),
            ));
        }
        if !(0.0 < self.chebyshev_lower_ratio && self.chebyshev_lower_ratio < 1.0) {
            return Err(KError::InvalidInput(
                "chebyshev_lower_ratio must be in (0, 1)".into(),
            ));
        }
        if self.chebyshev_safety <= 0.0 {
            return Err(KError::InvalidInput(
                "chebyshev_safety must be positive".into(),
            ));
        }
        if self.drop_tol < 0.0 {
            return Err(KError::InvalidInput("drop_tol must be ≥ 0".into()));
        }
        if !self.dist_coarse_ghost_scale.is_finite() || self.dist_coarse_ghost_scale < 0.0 {
            return Err(KError::InvalidInput(
                "dist_coarse_ghost_scale must be finite and ≥ 0".into(),
            ));
        }
        if self.non_galerkin.enabled && self.non_galerkin.start_level >= self.max_levels {
            return Err(KError::InvalidInput(
                "non_galerkin.start_level must be less than max_levels".into(),
            ));
        }
        if self.verify_galerkin && self.galerkin_samples == 0 {
            return Err(KError::InvalidInput(
                "galerkin_samples must be > 0 when verify_galerkin is enabled".into(),
            ));
        }
        Ok(())
    }

    fn set_smoothing_sweeps(&mut self, pre: usize, post: usize) {
        self.pre_sweeps = pre;
        self.post_sweeps = post;
        self.num_grid_sweeps[RelaxPhase::Fine.ix()] = pre;
        self.num_grid_sweeps[RelaxPhase::Down.ix()] = pre;
        self.num_grid_sweeps[RelaxPhase::Up.ix()] = post;
    }

    fn apply_relax_type(&mut self, value: &str) -> Result<(), KError> {
        let kind = AmgRelaxKind::from_str(value)?;
        let relax = match kind {
            AmgRelaxKind::Jacobi => RelaxType::Jacobi,
            AmgRelaxKind::Gs => RelaxType::GaussSeidel,
            AmgRelaxKind::Gsr => RelaxType::GaussSeidelBackward,
            AmgRelaxKind::Sgs => RelaxType::SymmetricGaussSeidel,
            AmgRelaxKind::Hgs => RelaxType::HybridGaussSeidel,
            AmgRelaxKind::L1Jacobi => RelaxType::L1Jacobi,
            AmgRelaxKind::Chebyshev => RelaxType::Chebyshev,
            AmgRelaxKind::ChebyshevSafe => RelaxType::ChebyshevSafe,
            AmgRelaxKind::SafeguardedGs => RelaxType::SafeguardedGaussSeidel,
            AmgRelaxKind::Ilu0 => RelaxType::Ilu0,
            AmgRelaxKind::Ras => RelaxType::Ras,
        };
        self.relax_type = relax;
        for phase in RelaxPhase::ALL {
            self.grid_relax_type[phase.ix()] = relax;
        }
        self.grid_relax_type[RelaxPhase::Coarsest.ix()] = RelaxType::GaussSeidel;
        Ok(())
    }

    pub fn try_from_opts(opts: &PcOptions) -> Result<Self, KError> {
        let mut cfg = Self::default();

        if let Some(levels) = opts.amg_levels {
            cfg.max_levels = levels;
        }
        if let Some(threshold) = opts.amg_strength_threshold {
            let threshold = ensure_finite("amg_strength_threshold", threshold)?;
            if !(threshold > 0.0 && threshold <= 1.0) {
                return Err(KError::InvalidInput(
                    "amg_strength_threshold must be in (0, 1]".into(),
                ));
            }
            cfg.strong_threshold = threshold;
        }
        if let Some(pre) = opts.amg_nu_pre {
            cfg.set_smoothing_sweeps(pre, cfg.post_sweeps);
        }
        if let Some(post) = opts.amg_nu_post {
            cfg.set_smoothing_sweeps(cfg.pre_sweeps, post);
        }
        if let Some(threshold) = opts.amg_coarse_threshold {
            cfg.coarse_threshold = threshold;
        }
        if let Some(max) = opts.amg_max_coarse_size {
            cfg.max_coarse_size = max;
        }
        if let Some(min) = opts.amg_min_coarse_size {
            cfg.min_coarse_size = min;
        }
        if let Some(trunc) = opts.amg_truncation_factor {
            cfg.truncation_factor = trunc;
        }
        if let Some(cap) = opts.amg_max_elements_per_row {
            cfg.max_elements_per_row = cap;
        }
        if let Some(interop) = opts.amg_interpolation_truncation {
            cfg.interpolation_truncation = interop;
        }
        if let Some(abs) = opts.amg_rap_truncation_abs {
            cfg.rap_truncation_abs = abs;
        }
        if let Some(cap) = opts.amg_rap_max_elements_per_row {
            cfg.rap_max_elements_per_row = cap;
        }
        if let Some(ref coarsen) = opts.amg_coarsen_type {
            cfg.coarsen_type = map_coarsen(AmgCoarsenKind::from_str(coarsen)?);
        }
        if let Some(ref interp) = opts.amg_interp_type {
            cfg.interp_type = map_interp(AmgInterpKind::from_str(interp)?);
        }
        if let Some(ref smoother) = opts.amg_smoother {
            cfg.apply_relax_type(smoother)?;
        } else if let Some(ref relax) = opts.amg_relax_type {
            cfg.apply_relax_type(relax)?;
        }
        if let Some(steps) = opts.amg_smoother_steps {
            cfg.set_smoothing_sweeps(steps, steps);
        }
        if let Some(omega) = opts.amg_smoother_omega {
            let omega = ensure_finite("amg_smoother_omega", omega)?;
            if omega <= 0.0 {
                return Err(KError::InvalidInput(
                    "amg_smoother_omega must be > 0".into(),
                ));
            }
            cfg.jacobi_omega = omega;
            cfg.adaptive_smooth_omega = omega;
        }
        if let Some(val) = opts.amg_logging_level {
            cfg.logging_level = val;
        }
        if let Some(val) = opts.amg_print_level {
            cfg.print_level = val;
        }
        if let Some(val) = opts.amg_tolerance {
            cfg.tolerance = val;
        }
        if let Some(val) = opts.amg_max_iterations {
            cfg.max_iterations = val;
        }
        if let Some(val) = opts.amg_min_iterations {
            cfg.min_iterations = val;
        }
        if let Some(flag) = opts.amg_ieee_checks {
            cfg.ieee_checks = flag;
        }
        if let Some(flag) = opts.amg_optimize_workspace {
            cfg.optimize_workspace = flag;
        }
        if let Some(flag) = opts.amg_keep_transpose {
            cfg.keep_transpose = flag;
        }
        if let Some(flag) = opts.amg_keep_pivot_in_rap {
            cfg.keep_pivot_in_rap = flag;
        }
        if let Some(flag) = opts.amg_require_spd {
            cfg.require_spd = flag;
        }
        if cfg.require_spd && !cfg.keep_transpose {
            return Err(KError::InvalidInput(
                "SPD mode requires pc_amg_keep_transpose true".into(),
            ));
        }
        if let Some(flag) = opts.amg_print_setup {
            if flag {
                cfg.print_level = cfg.print_level.max(1);
                cfg.logging_level = cfg.logging_level.max(2);
            }
        }
        if let Some(ref mode) = opts.amg_dist_apply_mode {
            cfg.dist_coarse_strategy = parse_dist_apply_mode(mode)?;
        }
        if let Some(flag) = opts.amg_dist_instrumentation {
            cfg.dist_apply_instrumentation = flag;
        }
        if let Some(scale) = opts.amg_dist_coarse_ghost_scale {
            cfg.dist_coarse_ghost_scale =
                ensure_finite("amg_dist_coarse_ghost_scale", scale)?;
        }
        cfg.conditioning = opts.conditioning_options()?;
        cfg.validate()?;
        Ok(cfg)
    }
}

fn ensure_finite(name: &str, value: f64) -> Result<f64, KError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(KError::InvalidInput(format!("{name} must be finite")))
    }
}

fn map_coarsen(kind: AmgCoarsenKind) -> CoarsenType {
    match kind {
        AmgCoarsenKind::Rs => CoarsenType::RS,
        AmgCoarsenKind::Hmis => CoarsenType::HMIS,
        AmgCoarsenKind::Pmis => CoarsenType::PMIS,
        AmgCoarsenKind::Falgout => CoarsenType::Falgout,
    }
}

fn parse_dist_apply_mode(value: &str) -> Result<DistCoarseStrategy, KError> {
    DistCoarseStrategy::from_str(value).map_err(|_| {
        KError::InvalidInput(format!("invalid amg_dist_apply_mode: {value}"))
    })
}

fn map_interp(kind: AmgInterpKind) -> InterpType {
    match kind {
        AmgInterpKind::Classical => InterpType::Classical,
        AmgInterpKind::Direct => InterpType::Direct,
        AmgInterpKind::Multipass => InterpType::Multipass,
        AmgInterpKind::Extended => InterpType::Extended,
        AmgInterpKind::Standard => InterpType::Standard,
        AmgInterpKind::He => InterpType::HE,
    }
}

#[cfg(test)]
mod config_mapping_tests {
    use super::*;

    fn opts_from(args: &[&str]) -> PcOptions {
        PcOptions::from_args(args).expect("valid AMG args")
    }

    #[test]
    fn amg_config_applies_cli_overrides() {
        let opts = opts_from(&[
            "-pc_type",
            "amg",
            "-pc_amg_levels",
            "6",
            "-pc_amg_strength_threshold",
            "0.25",
            "-pc_amg_coarsen",
            "hmis",
            "-pc_amg_interp",
            "extended",
            "-pc_amg_smoother",
            "chebyshev",
            "-pc_amg_smoother_steps",
            "2",
            "-pc_amg_smoother_omega",
            "0.8",
            "-pc_amg_truncation_factor",
            "0.2",
            "-pc_amg_interp_maxnnz",
            "8",
            "-pc_amg_rap_truncation_abs",
            "0.0",
            "-pc_amg_rap_maxnnz",
            "16",
            "-pc_amg_keep_transpose",
            "true",
            "-pc_amg_keep_pivot_in_rap",
            "true",
            "-pc_amg_require_spd",
            "true",
            "-pc_amg_print_setup",
            "true",
        ]);
        let cfg = AMGConfig::try_from_opts(&opts).unwrap();
        assert_eq!(cfg.max_levels, 6);
        assert!((cfg.strong_threshold - 0.25).abs() < 1e-12);
        assert_eq!(cfg.coarsen_type, CoarsenType::HMIS);
        assert_eq!(cfg.interp_type, InterpType::Extended);
        assert_eq!(cfg.relax_type, RelaxType::Chebyshev);
        assert_eq!(cfg.pre_sweeps, 2);
        assert_eq!(cfg.post_sweeps, 2);
        assert!((cfg.jacobi_omega - 0.8).abs() < 1e-12);
        assert!((cfg.adaptive_smooth_omega - 0.8).abs() < 1e-12);
        assert_eq!(cfg.truncation_factor, 0.2);
        assert_eq!(cfg.max_elements_per_row, 8);
        assert_eq!(cfg.rap_truncation_abs, 0.0);
        assert_eq!(cfg.rap_max_elements_per_row, 16);
        assert!(cfg.keep_transpose);
        assert!(cfg.keep_pivot_in_rap);
        assert!(cfg.require_spd);
        assert!(cfg.logging_level >= 2);
        assert!(cfg.print_level >= 1);
    }

    #[test]
    fn amg_config_denies_keep_transpose_when_spd() {
        let opts = opts_from(&["-pc_type", "amg", "-pc_amg_keep_transpose", "false"]);
        assert!(AMGConfig::try_from_opts(&opts).is_err());
    }

    #[test]
    fn amg_config_allows_keep_transpose_when_spd_off() {
        let opts = opts_from(&[
            "-pc_type",
            "amg",
            "-pc_amg_require_spd",
            "false",
            "-pc_amg_keep_transpose",
            "false",
        ]);
        let cfg = AMGConfig::try_from_opts(&opts).unwrap();
        assert!(!cfg.require_spd);
        assert!(!cfg.keep_transpose);
    }

    #[test]
    fn amg_levels_zero_is_invalid() {
        let opts = opts_from(&["-pc_type", "amg", "-pc_amg_levels", "0"]);
        assert!(AMGConfig::try_from_opts(&opts).is_err());
    }

    #[test]
    fn amg_strength_threshold_negative_is_invalid() {
        let opts = opts_from(&["-pc_type", "amg", "-pc_amg_strength_threshold", "-0.1"]);
        assert!(AMGConfig::try_from_opts(&opts).is_err());
    }

    #[test]
    fn pc_amg_alias_sets_pc_type() {
        let opts = opts_from(&["-pc_amg"]);
        assert_eq!(opts.pc_type.as_deref(), Some("amg"));
        let cfg = AMGConfig::try_from_opts(&opts).unwrap();
        assert_eq!(cfg.max_levels, AMGConfig::default().max_levels);
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
    pub fn cycle(mut self, c: CycleType) -> Self {
        self.cfg.cycle_type = c;
        self
    }
    pub fn kcycle(mut self, kc: KCycle) -> Self {
        self.cfg.kcycle = Some(kc);
        self
    }
    pub fn disable_kcycle(mut self) -> Self {
        self.cfg.kcycle = None;
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
    pub fn keep_transpose(mut self, on: bool) -> Self {
        self.cfg.keep_transpose = on;
        self
    }
    pub fn keep_pivot_in_rap(mut self, yes: bool) -> Self {
        self.cfg.keep_pivot_in_rap = yes;
        self
    }
    pub fn require_spd(mut self, on: bool) -> Self {
        self.cfg.require_spd = on;
        self
    }
    pub fn verify_p_rank(mut self, on: bool) -> Self {
        self.cfg.verify_p_rank = on;
        self
    }
    pub fn rank_cond_threshold(mut self, v: f64) -> Self {
        self.cfg.rank_cond_threshold = v;
        self
    }
    pub fn rank_min_col_norm(mut self, v: f64) -> Self {
        self.cfg.rank_min_col_norm = v;
        self
    }
    pub fn verify_galerkin(mut self, on: bool) -> Self {
        self.cfg.verify_galerkin = on;
        self
    }
    pub fn galerkin_rel_tol(mut self, v: f64) -> Self {
        self.cfg.galerkin_rel_tol = v;
        self
    }
    pub fn spd_diag_floor(mut self, eps: f64) -> Self {
        self.cfg.spd_diag_floor = eps.max(0.0);
        self
    }
    pub fn forbid_non_galerkin_in_spd(mut self, on: bool) -> Self {
        self.cfg.forbid_non_galerkin_in_spd = on;
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
        self.cfg.adaptive_smooth_omega = w;
        self
    }
    pub fn adaptive_interp(mut self, on: bool) -> Self {
        self.cfg.adaptive_interp = on;
        self
    }
    pub fn adaptive_samples(mut self, r: usize) -> Self {
        self.cfg.adaptive_samples = r;
        self
    }
    pub fn adaptive_smooth_steps(mut self, nu: usize) -> Self {
        self.cfg.adaptive_smooth_steps = nu;
        self
    }
    pub fn adaptive_smooth_omega(mut self, w: f64) -> Self {
        self.cfg.adaptive_smooth_omega = w;
        self
    }
    pub fn adaptive_lambda(mut self, lam: f64) -> Self {
        self.cfg.adaptive_lambda = lam;
        self
    }
    pub fn adaptive_enforce_sum1(mut self, on: bool) -> Self {
        self.cfg.adaptive_enforce_sum1 = on;
        self
    }
    pub fn adaptive_weight_mode(mut self, mode: AdaptiveWeight) -> Self {
        self.cfg.adaptive_weight_mode = mode;
        self
    }
    pub fn chebyshev_recompute_esteig(mut self, on: bool) -> Self {
        self.cfg.chebyshev_recompute_esteig = on;
        self
    }
    pub fn chebyshev_safety(mut self, s: f64) -> Self {
        self.cfg.chebyshev_safety = s;
        self
    }
    pub fn chebyshev_lower_ratio(mut self, r: f64) -> Self {
        self.cfg.chebyshev_lower_ratio = r;
        self
    }
    pub fn chebyshev_power_steps(mut self, steps: usize) -> Self {
        self.cfg.chebyshev_power_steps = steps;
        self
    }
    pub fn chebyshev_degree(mut self, d: usize) -> Self {
        self.cfg.chebyshev_degree = d;
        self
    }
    pub fn filter_trials(mut self, trials: Vec<Vec<f64>>) -> Self {
        self.cfg.filter_trial_vectors = Some(trials);
        self
    }
    pub fn filter_omega(mut self, omega: f64) -> Self {
        self.cfg.filter_omega = omega;
        self
    }
    pub fn filter_after_non_galerkin(mut self, on: bool) -> Self {
        self.cfg.filter_after_non_galerkin = on;
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
    pub fn fsai_dist(mut self, dist: usize) -> Self {
        self.cfg.fsai_dist = dist.max(1);
        self
    }
    pub fn fsai_use_strength(mut self, use_strength: bool) -> Self {
        self.cfg.fsai_use_strength = use_strength;
        self
    }
    pub fn fsai_max_per_row(mut self, cap: usize) -> Self {
        self.cfg.fsai_max_per_row = cap.max(1);
        self
    }
    pub fn fsai_lambda(mut self, lambda: f64) -> Self {
        self.cfg.fsai_lambda = if lambda >= 0.0 { lambda } else { 0.0 };
        self
    }
    pub fn fsai_adaptive_passes(mut self, passes: usize) -> Self {
        self.cfg.fsai_adaptive_passes = passes;
        self
    }
    pub fn fsai_drop_tol(mut self, drop: f64) -> Self {
        self.cfg.fsai_drop_tol = if drop >= 0.0 { drop } else { 0.0 };
        self
    }
    pub fn fsai_damping(mut self, tau: f64) -> Self {
        self.cfg.fsai_damping = tau;
        self
    }

    pub fn coarse_solve(mut self, v: CoarseSolve) -> Self {
        self.cfg.coarse_solve = v;
        self
    }

    pub fn ilu_params(mut self, drop_tol: f64, fill_per_row: usize) -> Self {
        self.cfg.ilu_drop_tol = drop_tol;
        self.cfg.ilu_fill_per_row = fill_per_row;
        self
    }

    pub fn non_galerkin(
        mut self,
        enabled: bool,
        start_level: usize,
        drop_abs: f64,
        drop_rel: f64,
        cap_row: usize,
    ) -> Self {
        self.cfg.non_galerkin.enabled = enabled;
        self.cfg.non_galerkin.start_level = start_level;
        self.cfg.non_galerkin.drop_abs = drop_abs;
        self.cfg.non_galerkin.drop_rel = drop_rel;
        self.cfg.non_galerkin.cap_row = cap_row;
        self
    }

    pub fn non_galerkin_symmetry(mut self, sym: NgSymmetry, lump_diag: bool) -> Self {
        self.cfg.non_galerkin.symmetry = sym;
        self.cfg.non_galerkin.lump_diagonal = lump_diag;
        self
    }

    pub fn non_galerkin_oc_target(mut self, target: Option<f64>, iters: usize) -> Self {
        self.cfg.non_galerkin.oc_target = target;
        self.cfg.non_galerkin.oc_max_iter = iters;
        self
    }

    pub fn nodal(mut self, on: bool, block_size: usize) -> Self {
        self.cfg.nodal = if on { NodalMode::Nodal } else { NodalMode::Off };
        self.cfg.block_size = block_size.max(1);
        self
    }

    pub fn num_functions(mut self, r: usize) -> Self {
        self.cfg.num_functions = r.max(1);
        self
    }

    pub fn near_nullspace(mut self, basis: Vec<Vec<f64>>) -> Self {
        let count = basis.len().max(1);
        self.cfg.near_nullspace = Some(NearNullspace { basis });
        self.cfg.num_functions = count;
        self
    }

    pub fn post_interp(mut self, t: PostInterpType) -> Self {
        self.cfg.post_interp = t;
        self
    }

    pub fn flexible_level(mut self, level: usize) -> Self {
        self.cfg.flexible_level = Some(level);
        self
    }

    pub fn flexible_iters(mut self, iters: usize) -> Self {
        self.cfg.flexible_iters = iters;
        self
    }

    pub fn flexible_rtol(mut self, tol: f64) -> Self {
        self.cfg.flexible_rtol = tol;
        self
    }

    pub fn flexible_pc_sweeps(mut self, sweeps: usize) -> Self {
        self.cfg.flexible_pc_sweeps = sweeps;
        self
    }

    pub fn mixed_precision(mut self, mp: Option<MixedPrecision>) -> Self {
        self.cfg.mixed_precision = mp;
        self
    }

    pub fn mixed_storage(mut self, storage: MixedStorage) -> Self {
        self.cfg.mixed_storage = storage;
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
    if matches!(coarse_solver, CoarseSolve::DirectDense)
        && cfg.num_grid_sweeps[RelaxPhase::Coarsest.ix()] != 0
    {
        return Err(KError::InvalidInput(
            "num_grid_sweeps[Coarsest] must be 0 when coarse_solve is DirectDense".into(),
        ));
    }

    for (i, &rt) in cfg.grid_relax_type.iter().enumerate() {
        match rt {
            RelaxType::Jacobi
            | RelaxType::GaussSeidel
            | RelaxType::GaussSeidelBackward
            | RelaxType::SymmetricGaussSeidel
            | RelaxType::L1Jacobi
            | RelaxType::Chebyshev
            | RelaxType::ChebyshevSafe
            | RelaxType::SafeguardedGaussSeidel
            | RelaxType::Ilu0
            | RelaxType::Ras
            | RelaxType::Fsai => {}
            _ => {
                return Err(KError::InvalidInput(format!(
                    "RelaxType {rt:?} not yet supported (phase index {i})"
                )));
            }
        }
    }

    if let Some(mp) = cfg.mixed_precision
        && mp.smoothers_enabled()
    {
        for (i, &rt) in cfg.grid_relax_type.iter().enumerate() {
            if i == RelaxPhase::Coarsest.ix() {
                continue;
            }
            match rt {
                RelaxType::Jacobi
                | RelaxType::L1Jacobi
                | RelaxType::Chebyshev
                | RelaxType::Fsai => {}
                other => {
                    return Err(KError::InvalidInput(format!(
                        "Mixed-precision smoothing only supports Jacobi, L1Jacobi, Chebyshev, or Fsai; got {other:?}"
                    )));
                }
            }
        }
    }

    for (i, &k) in cfg.num_grid_sweeps.iter().enumerate() {
        if i != RelaxPhase::Coarsest.ix() && k == 0 {
            return Err(KError::InvalidInput(format!(
                "num_grid_sweeps for phase {i} must be >= 1"
            )));
        }
    }

    if cfg.flexible_iters > 0 {
        let level = cfg.flexible_level.ok_or_else(|| {
            KError::InvalidInput("flexible_iters > 0 requires flexible_level to be set".into())
        })?;
        if level >= cfg.max_levels {
            return Err(KError::InvalidInput(
                "flexible_level must be less than max_levels".into(),
            ));
        }
        if cfg.flexible_rtol < 0.0 {
            return Err(KError::InvalidInput(
                "flexible_rtol must be non-negative".into(),
            ));
        }
    }

    if cfg.require_spd {
        if matches!(coarse_solver, CoarseSolve::ILU) {
            return Err(KError::InvalidInput(
                "SPD mode requires DirectDense or CG as the coarse solver; ILU is not SPD-safe"
                    .into(),
            ));
        }

        if cfg.non_galerkin.enabled && cfg.forbid_non_galerkin_in_spd {
            return Err(KError::InvalidInput(
                concat!(
                    "Non-Galerkin filtering is disabled when require_spd is true ",
                    "(set forbid_non_galerkin_in_spd = false to override)."
                )
                .into(),
            ));
        }

        if !matches!(
            cfg.relax_type,
            RelaxType::Chebyshev
                | RelaxType::Jacobi
                | RelaxType::L1Jacobi
                | RelaxType::SymmetricGaussSeidel
                | RelaxType::Fsai
        ) {
            return Err(KError::InvalidInput(
                "SPD mode requires symmetric or Jacobi-type smoothers".into(),
            ));
        }

        let down_sweeps = cfg.num_grid_sweeps[RelaxPhase::Down.ix()];
        let up_sweeps = cfg.num_grid_sweeps[RelaxPhase::Up.ix()];
        if down_sweeps != up_sweeps {
            return Err(KError::InvalidInput(
                "SPD mode requires symmetric pre/post smoothing counts".into(),
            ));
        }
        if cfg.pre_sweeps != cfg.post_sweeps {
            return Err(KError::InvalidInput(
                "SPD mode requires pre_sweeps == post_sweeps".into(),
            ));
        }

        let down_type = cfg.grid_relax_type[RelaxPhase::Down.ix()];
        let up_type = cfg.grid_relax_type[RelaxPhase::Up.ix()];
        if down_type != up_type {
            return Err(KError::InvalidInput(
                "SPD mode requires matching relax types for Down and Up phases".into(),
            ));
        }

        for phase in [RelaxPhase::Fine, RelaxPhase::Down, RelaxPhase::Up] {
            match cfg.grid_relax_type[phase.ix()] {
                RelaxType::GaussSeidelBackward
                | RelaxType::HybridGaussSeidel
                | RelaxType::SafeguardedGaussSeidel
                | RelaxType::Ilu0
                | RelaxType::Ras
                | RelaxType::ChebyshevSafe => {
                    return Err(KError::InvalidInput(
                        "SPD mode does not support asymmetric Gauss-Seidel variants".into(),
                    ));
                }
                RelaxType::Jacobi
                | RelaxType::GaussSeidel
                | RelaxType::SymmetricGaussSeidel
                | RelaxType::L1Jacobi
                | RelaxType::Chebyshev
                | RelaxType::Fsai => {}
            }
        }

        if cfg.flexible_iters > 0 {
            let level = cfg.flexible_level.expect("validated above");
            let phase = if level == 0 {
                RelaxPhase::Fine
            } else {
                RelaxPhase::Down
            };
            match cfg.grid_relax_type[phase.ix()] {
                RelaxType::Jacobi | RelaxType::L1Jacobi | RelaxType::SymmetricGaussSeidel => {}
                other => {
                    return Err(KError::InvalidInput(format!(
                        "SPD mode requires a symmetric positive definite smoother for flexible presmoothing; got {other:?}"
                    )));
                }
            }
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
    temp: Vec<R>,
    work: Vec<R>,
    residual: Vec<R>,
    coarse_rhs: Vec<R>,
    fine_corr: Vec<R>,
    k_zeta: Vec<R>,
    k_p: Vec<R>,
    k_ap: Vec<R>,
    k_temp: Vec<R>,
    k_work: Vec<R>,
    k_residual: Vec<R>,
    mp: Option<MixedWs>,
}

impl AMGWorkspace {
    fn new(cap: usize) -> Self {
        Self {
            temp: vec![R::zero(); cap],
            work: vec![R::zero(); cap],
            residual: vec![R::zero(); cap],
            coarse_rhs: vec![R::zero(); cap],
            fine_corr: vec![R::zero(); cap],
            k_zeta: vec![R::zero(); cap],
            k_p: vec![R::zero(); cap],
            k_ap: vec![R::zero(); cap],
            k_temp: vec![R::zero(); cap],
            k_work: vec![R::zero(); cap],
            k_residual: vec![R::zero(); cap],
            mp: None,
        }
    }
    fn ensure(&mut self, n: usize) {
        let grow = |v: &mut Vec<R>, n: usize| {
            if v.len() < n {
                v.resize(n, R::zero())
            }
        };
        grow(&mut self.temp, n);
        grow(&mut self.work, n);
        grow(&mut self.residual, n);
        grow(&mut self.coarse_rhs, n);
        grow(&mut self.fine_corr, n);
        grow(&mut self.k_zeta, n);
        grow(&mut self.k_p, n);
        grow(&mut self.k_ap, n);
        grow(&mut self.k_temp, n);
        grow(&mut self.k_work, n);
        grow(&mut self.k_residual, n);
    }

    fn ensure_mixed(&mut self, n: usize) {
        if self.mp.is_none() {
            self.mp = Some(MixedWs::with_capacity(n));
        }
        if let Some(ref mut mp) = self.mp {
            mp.ensure_vectors(n);
        }
    }
}

#[derive(Debug)]
struct MixedWs {
    temp32: Vec<f32>,
    work32: Vec<f32>,
    residual32: Vec<f32>,
    coarse_rhs32: Vec<f32>,
    fine_corr32: Vec<f32>,
    a_vals32: Vec<f32>,
    diag_inv32: Vec<f32>,
    l1_inv32: Vec<f32>,
    d_sqrt_inv32: Vec<f32>,
    fsai_g_vals32: Vec<f32>,
    fsai_gt_vals32: Vec<f32>,
}

impl MixedWs {
    fn with_capacity(n: usize) -> Self {
        Self {
            temp32: vec![0.0; n],
            work32: vec![0.0; n],
            residual32: vec![0.0; n],
            coarse_rhs32: vec![0.0; n],
            fine_corr32: vec![0.0; n],
            a_vals32: Vec::new(),
            diag_inv32: Vec::new(),
            l1_inv32: Vec::new(),
            d_sqrt_inv32: Vec::new(),
            fsai_g_vals32: Vec::new(),
            fsai_gt_vals32: Vec::new(),
        }
    }

    fn ensure_vectors(&mut self, n: usize) {
        let grow = |v: &mut Vec<f32>, n: usize| {
            if v.len() < n {
                v.resize(n, 0.0);
            }
        };
        grow(&mut self.temp32, n);
        grow(&mut self.work32, n);
        grow(&mut self.residual32, n);
        grow(&mut self.coarse_rhs32, n);
        grow(&mut self.fine_corr32, n);
    }

    fn ensure_vals(&mut self, nnz: usize) -> &mut Vec<f32> {
        if self.a_vals32.len() < nnz {
            self.a_vals32.resize(nnz, 0.0);
        }
        &mut self.a_vals32
    }

    fn ensure_diag(&mut self, n: usize) -> &mut Vec<f32> {
        if self.diag_inv32.len() < n {
            self.diag_inv32.resize(n, 0.0);
        }
        &mut self.diag_inv32
    }

    fn ensure_l1(&mut self, n: usize) -> &mut Vec<f32> {
        if self.l1_inv32.len() < n {
            self.l1_inv32.resize(n, 0.0);
        }
        &mut self.l1_inv32
    }

    fn ensure_d_sqrt(&mut self, n: usize) -> &mut Vec<f32> {
        if self.d_sqrt_inv32.len() < n {
            self.d_sqrt_inv32.resize(n, 0.0);
        }
        &mut self.d_sqrt_inv32
    }

    fn ensure_fsai_g(&mut self, nnz: usize) -> &mut Vec<f32> {
        if self.fsai_g_vals32.len() < nnz {
            self.fsai_g_vals32.resize(nnz, 0.0);
        }
        &mut self.fsai_g_vals32
    }

    fn ensure_fsai_gt(&mut self, nnz: usize) -> &mut Vec<f32> {
        if self.fsai_gt_vals32.len() < nnz {
            self.fsai_gt_vals32.resize(nnz, 0.0);
        }
        &mut self.fsai_gt_vals32
    }
}

#[derive(Clone)]
struct FsaiData {
    g: CsrMatrix<f64>,
    gt: CsrMatrix<f64>,
    g2gt_pos: Vec<usize>,
}

struct AMGLevel {
    /// A_l (coarse operator at this level, l = 0 is finest)
    a: CsrMatrix<f64>,
    /// P_l (interpolation to next coarser level)
    p: CsrMatrix<f64>,
    /// R_l (restriction to next coarser level)
    r: CsrMatrix<f64>,
    /// diag(A_l)^{-1}
    diag_inv: Vec<f64>,
    /// diag(A_l)^{-1/2} (optional)
    d_sqrt_inv: Option<Vec<f64>>,
    /// 1 / (\sum_j |a_ij|) for L1-Jacobi
    l1_inv: Option<Vec<f64>>,
    /// diag(A_l)^{-1} with safeguards for poor diagonals (optional)
    diag_inv_safe: Option<Vec<f64>>,
    /// diag(A_l)^{-1/2} derived from safeguarded diagonal (optional)
    d_sqrt_inv_safe: Option<Vec<f64>>,
    /// Cached spectral bounds for Chebyshev smoother
    cheb: Option<ChebData>,
    /// Cached spectral bounds for safeguarded Chebyshev smoother
    cheb_safe: Option<ChebData>,
    /// fine->coarse aggregate id used to rebuild P values (SA numeric refresh)
    agg_of: Vec<usize>,
    /// coarse/fine flags for classical interpolation
    is_c: Vec<bool>,
    /// CF metadata for classical interpolation
    cf: Option<CFInfo>,
    /// Mapping from P entry index -> index in R (transpose) values array
    p2r_pos: Vec<usize>,
    /// number of coarse basis functions per aggregate
    num_functions: usize,
    /// Row-local orthonormalized basis coefficients for nodal SA, length = nrows * num_functions.
    row_basis: Option<Vec<f64>>,
    /// DOF layout for nodal aggregation
    layout: Option<DofLayout>,
    /// stored near-nullspace basis (optional)
    nns: Option<Vec<Vec<f64>>>,
    /// Symbolic pattern for A_{l+1}
    a_next_pat: Option<CsrPattern>,
    /// Filtered NG pattern for A_{l+1}
    a_next_pat_ng: Option<CsrPattern>,
    /// Mapping from full RAP nnz index -> NG nnz index
    rap_full2ng_pos: Option<Vec<Option<usize>>>,
    /// Optional transpose storage when keep_transpose is disabled
    r_row_ptr: Option<Vec<usize>>,
    r_col_idx: Option<Vec<usize>>,
    r_vals_scratch: Option<Vec<f64>>,
    #[allow(clippy::redundant_allocation)]
    coarse_solver: Option<Mutex<Box<dyn CoarseSolver + Send>>>,
    ilu0: Option<Mutex<IluCsr>>,
    ras: Option<Mutex<Asm>>,
    fsai: Option<FsaiData>,
    a_vals_f32: Option<Vec<f32>>,
    diag_inv_f32: Option<Vec<f32>>,
    d_sqrt_inv_f32: Option<Vec<f32>>,
    l1_inv_f32: Option<Vec<f32>>,
    fsai_g_vals_f32: Option<Vec<f32>>,
    fsai_gt_vals_f32: Option<Vec<f32>>,
}

#[cfg(feature = "simd")]
fn build_level_spmv_plans(level: &mut AMGLevel, tuning: &SpmvTuning) {
    level.a.build_spmv_plan(tuning);
    level.p.build_spmv_plan(tuning);
    level.r.build_spmv_plan(tuning);
}

#[derive(Clone)]
struct ChebData {
    lambda_max: f64,
    lambda_min: f64,
}

#[derive(Clone)]
struct RelaxPolicy {
    kind: [RelaxType; 4],
    sweeps: [usize; 4],
    omega: f64,
}

trait CyclePolicy: Send + Sync {
    fn gamma_visits(&self, level: usize) -> usize;
    fn k_presmooth(&self, _level: usize) -> Option<(KrylovAlgo, usize)> {
        None
    }
    fn k_postsmooth(&self, _level: usize) -> Option<(KrylovAlgo, usize)> {
        None
    }
}

struct VPolicy;

impl CyclePolicy for VPolicy {
    fn gamma_visits(&self, _level: usize) -> usize {
        1
    }
}

struct WPolicy {
    gamma: usize,
}

impl CyclePolicy for WPolicy {
    fn gamma_visits(&self, _level: usize) -> usize {
        self.gamma.max(1)
    }
}

struct KPolicy {
    base: CycleType,
    cfg: KCycle,
}

impl KPolicy {
    fn new(base: CycleType, mut cfg: KCycle) -> Self {
        cfg.levels.sort_unstable();
        cfg.levels.dedup();
        cfg.iters = cfg.iters.max(1);
        Self { base, cfg }
    }

    fn contains(&self, level: usize) -> bool {
        self.cfg.levels.binary_search(&level).is_ok()
    }
}

impl CyclePolicy for KPolicy {
    fn gamma_visits(&self, _level: usize) -> usize {
        match self.base {
            CycleType::V => 1,
            CycleType::W { gamma } => gamma.max(2),
        }
        .max(1)
    }

    fn k_presmooth(&self, level: usize) -> Option<(KrylovAlgo, usize)> {
        if self.cfg.place_pre && self.contains(level) {
            Some((self.cfg.algo, self.cfg.iters))
        } else {
            None
        }
    }

    fn k_postsmooth(&self, level: usize) -> Option<(KrylovAlgo, usize)> {
        if self.cfg.place_post && self.contains(level) {
            Some((self.cfg.algo, self.cfg.iters))
        } else {
            None
        }
    }
}

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

fn build_r_from_p(lvl: &mut AMGLevel) -> CsrMatrix<f64> {
    let rr = lvl.r_row_ptr.as_ref().expect("missing R pattern");
    let rc = lvl.r_col_idx.as_ref().expect("missing R pattern");
    let p2r = &lvl.p2r_pos;
    let pvals = lvl.p.values();
    let rvals = lvl.r_vals_scratch.as_mut().expect("missing R scratch");
    for (pi, &ri) in p2r.iter().enumerate() {
        rvals[ri] = pvals[pi];
    }
    CsrMatrix::from_csr(
        lvl.p.ncols(),
        lvl.p.nrows(),
        rr.clone(),
        rc.clone(),
        rvals.clone(),
    )
}

fn rap_numeric_with_pt(lvl: &mut AMGLevel, out_vals: &mut [f64]) -> Result<(), KError> {
    let r_tmp = build_r_from_p(lvl);
    let pat = lvl.a_next_pat.as_ref().expect("missing pattern").clone();
    rap_numeric(&pat, &r_tmp, &lvl.a, &lvl.p, out_vals);
    Ok(())
}

fn make_trial_matrix(cfg: &AMGConfig, n: usize) -> Result<Option<Mat<f64>>, KError> {
    if cfg.filter_omega <= 0.0 {
        return Ok(None);
    }
    if cfg.require_spd && cfg.filter_trial_vectors.is_none() {
        return Ok(None);
    }
    let mat = if let Some(basis) = cfg.filter_trial_vectors.as_ref() {
        if basis.is_empty() {
            return Ok(None);
        }
        let r = basis.len();
        let mut m = Mat::<f64>::zeros(n, r);
        for (col, vec) in basis.iter().enumerate() {
            if vec.len() != n {
                return Err(KError::InvalidInput(
                    "filter_trial_vectors entry has mismatched length".into(),
                ));
            }
            for i in 0..n {
                m[(i, col)] = vec[i];
            }
        }
        m
    } else {
        let mut m = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            m[(i, 0)] = 1.0;
        }
        m
    };
    Ok(Some(mat))
}

fn apply_trial_compensation(
    cfg: &AMGConfig,
    a: &mut CsrMatrix<f64>,
    trials: Option<&Mat<f64>>,
    block_size: usize,
) -> Result<(), KError> {
    if cfg.filter_omega <= 0.0 {
        return Ok(());
    }
    let Some(trials_mat) = trials else {
        return Ok(());
    };
    let min_diag = if cfg.require_spd {
        Some(cfg.spd_diag_floor.max(1e-12))
    } else {
        None
    };
    if block_size > 1 && matches!(cfg.nodal, NodalMode::Nodal) && a.nrows() % block_size == 0 {
        compensate_nodal_diag(
            a,
            trials_mat.as_ref(),
            block_size,
            cfg.filter_omega,
            min_diag,
        )
    } else {
        compensate_scalar_rows(a, trials_mat.as_ref(), cfg.filter_omega, min_diag)
    }
}

struct LevelPostContext<'a> {
    r: usize,
    agg_of: &'a [usize],
    nns: Option<Vec<&'a [f64]>>,
    a: Option<&'a CsrMatrix<f64>>,
    d_inv: Option<&'a [f64]>,
}

pub(crate) fn row_scaling(
    mode: RowScaleMode,
    r: usize,
    nns: Option<&[&[f64]]>,
    agg_of: &[usize],
    d_inv: Option<&[f64]>,
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    p_vals: &mut [f64],
) -> Result<(), KError> {
    let n = p_row_ptr.len() - 1;
    let eps = 1e-30;
    for i in 0..n {
        let rs = p_row_ptr[i];
        let re = p_row_ptr[i + 1];
        match mode {
            RowScaleMode::SumToOne => {
                let sum: f64 = p_vals[rs..re].iter().copied().sum();
                if sum.abs() > eps {
                    let s = 1.0 / sum;
                    for k in rs..re {
                        p_vals[k] *= s;
                    }
                }
            }
            RowScaleMode::L2Unit => {
                for alpha in 0..r {
                    let mut n2 = 0.0;
                    for k in rs..re {
                        if p_col_idx[k] % r == alpha {
                            n2 += p_vals[k] * p_vals[k];
                        }
                    }
                    if n2 > eps {
                        let s = 1.0 / n2.sqrt();
                        for k in rs..re {
                            if p_col_idx[k] % r == alpha {
                                p_vals[k] *= s;
                            }
                        }
                    }
                }
            }
            RowScaleMode::DUnit => {
                let d = d_inv.expect("DUnit requires diag_inv");
                for alpha in 0..r {
                    let mut n2 = 0.0;
                    for k in rs..re {
                        if p_col_idx[k] % r == alpha {
                            n2 += p_vals[k] * p_vals[k];
                        }
                    }
                    let w = d[i].abs().recip().sqrt().max(1e-15);
                    if n2 > eps {
                        let s = 1.0 / (w * n2.sqrt());
                        for k in rs..re {
                            if p_col_idx[k] % r == alpha {
                                p_vals[k] *= s;
                            }
                        }
                    }
                }
            }
            RowScaleMode::ToNearNullspace => {
                let t = nns.expect("ToNearNullspace requires NNS basis");
                for alpha in 0..r {
                    let target = t[alpha][i];
                    let mut sum = 0.0;
                    for k in rs..re {
                        if p_col_idx[k] % r == alpha {
                            sum += p_vals[k];
                        }
                    }
                    if sum.abs() > eps {
                        let s = target / sum;
                        for k in rs..re {
                            if p_col_idx[k] % r == alpha {
                                p_vals[k] *= s;
                            }
                        }
                    } else {
                        let own_c = agg_of[i] * r + alpha;
                        for k in rs..re {
                            if p_col_idx[k] == own_c {
                                p_vals[k] = target;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn local_qr(
    r: usize,
    agg_of: &[usize],
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    p_vals: &mut [f64],
) -> Result<(), KError> {
    let n = agg_of.len();
    let n_aggs = 1 + agg_of.iter().copied().max().unwrap_or(0);
    let mut rows_in_agg: Vec<Vec<usize>> = vec![Vec::new(); n_aggs];
    for i in 0..n {
        rows_in_agg[agg_of[i]].push(i);
    }
    let mut pos_alpha: Vec<Vec<usize>> = vec![vec![usize::MAX; r]; n];
    for i in 0..n {
        let g = agg_of[i];
        let rs = p_row_ptr[i];
        let re = p_row_ptr[i + 1];
        for k in rs..re {
            let col = p_col_idx[k];
            if col / r == g {
                pos_alpha[i][col % r] = k;
            }
        }
    }
    for g in 0..n_aggs {
        let rows = &rows_in_agg[g];
        if rows.is_empty() {
            continue;
        }
        let m = rows.len();
        let mut q = vec![vec![R::default(); r]; m];
        for (ii, &i) in rows.iter().enumerate() {
            for alpha in 0..r {
                let k = pos_alpha[i][alpha];
                if k != usize::MAX {
                    q[ii][alpha] = p_vals[k];
                }
            }
        }
        for alpha in 0..r {
            for beta in 0..alpha {
                let mut dot = R::default();
                for ii in 0..m {
                    dot += q[ii][alpha] * q[ii][beta];
                }
                for ii in 0..m {
                    q[ii][alpha] -= dot * q[ii][beta];
                }
            }
            let mut n2 = R::default();
            for ii in 0..m {
                n2 += q[ii][alpha] * q[ii][alpha];
            }
            if n2 > 1e-30 {
                let inv = 1.0 / n2.sqrt();
                for ii in 0..m {
                    q[ii][alpha] *= inv;
                }
            }
        }
        for (ii, &i) in rows.iter().enumerate() {
            for alpha in 0..r {
                let k = pos_alpha[i][alpha];
                if k != usize::MAX {
                    p_vals[k] = q[ii][alpha];
                }
            }
        }
    }
    Ok(())
}

fn energy_polish(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    p_vals: &mut [f64],
    sweeps: usize,
    omega: f64,
) -> Result<(), KError> {
    let n = a.nrows();
    for _ in 0..sweeps {
        let old = p_vals.to_vec();
        for i in 0..n {
            let di = d_inv[i];
            let rs = p_row_ptr[i];
            let re = p_row_ptr[i + 1];
            for k in rs..re {
                let c = p_col_idx[k];
                let mut sum = 0.0;
                let ars = a.row_ptr()[i];
                let are = a.row_ptr()[i + 1];
                for ap in ars..are {
                    let j = a.col_idx()[ap];
                    let aij = a.values()[ap];
                    let prs = p_row_ptr[j];
                    let pre = p_row_ptr[j + 1];
                    for pk in prs..pre {
                        if p_col_idx[pk] == c {
                            sum += aij * old[pk];
                            break;
                        }
                    }
                }
                p_vals[k] = old[k] - omega * di * sum;
            }
        }
    }
    Ok(())
}

fn apply_post_interp(
    cfg: &AMGConfig,
    ctx: &LevelPostContext,
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    p_vals: &mut [f64],
) -> Result<(), KError> {
    match cfg.post_interp {
        PostInterpType::None => Ok(()),
        PostInterpType::RowScaling(mode) => {
            if matches!(mode, RowScaleMode::SumToOne) && ctx.r > 1 && ctx.nns.is_none() {
                return Ok(());
            }
            row_scaling(
                mode,
                ctx.r,
                ctx.nns.as_deref(),
                ctx.agg_of,
                ctx.d_inv,
                p_row_ptr,
                p_col_idx,
                p_vals,
            )
        }
        PostInterpType::LocalQR => local_qr(ctx.r, ctx.agg_of, p_row_ptr, p_col_idx, p_vals),
        PostInterpType::EnergyPolish { sweeps, omega } => {
            if sweeps == 0 {
                return Ok(());
            }
            energy_polish(
                ctx.a.expect("EnergyPolish requires A"),
                ctx.d_inv.expect("EnergyPolish requires diag_inv"),
                p_row_ptr,
                p_col_idx,
                p_vals,
                sweeps,
                omega,
            )
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RankDiagnostics {
    min_col_norm: f64,
    cond_estimate: f64,
    suspect: bool,
    degenerate_cols: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RankFixOutcome {
    Fixed,
    Unfixed,
}

fn p_column_norms2(p: &CsrMatrix<f64>) -> Vec<R> {
    let mut n2 = vec![R::default(); p.ncols()];
    let rp = p.row_ptr();
    let cj = p.col_idx();
    let vv = p.values();
    for i in 0..p.nrows() {
        let (rs, re) = (rp[i], rp[i + 1]);
        for k in rs..re {
            let j = cj[k];
            n2[j] += vv[k] * vv[k];
        }
    }
    n2
}

fn symmetric_eigenvalues(mut m: Mat<f64>) -> Vec<f64> {
    let n = m.nrows();
    if n == 0 {
        return Vec::new();
    }
    let mut iter = 0usize;
    loop {
        let mut max_val = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = m[(i, j)].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 || iter > 64 * n * n {
            break;
        }
        iter += 1;
        let app = m[(p, p)];
        let aqq = m[(q, q)];
        let apq = m[(p, q)];
        if apq.abs() < 1e-30 {
            continue;
        }
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            if k != p && k != q {
                let mkp = m[(k, p)];
                let mkq = m[(k, q)];
                let new_kp = mkp * c - mkq * s;
                let new_kq = mkp * s + mkq * c;
                m[(k, p)] = new_kp;
                m[(p, k)] = new_kp;
                m[(k, q)] = new_kq;
                m[(q, k)] = new_kq;
            }
        }
        let app_new = app * c * c - 2.0 * apq * s * c + aqq * s * s;
        let aqq_new = app * s * s + 2.0 * apq * s * c + aqq * c * c;
        m[(p, p)] = app_new;
        m[(q, q)] = aqq_new;
        m[(p, q)] = 0.0;
        m[(q, p)] = 0.0;
    }
    (0..n).map(|i| m[(i, i)]).collect()
}

fn rank_condition_estimate(p: &CsrMatrix<f64>, s: usize, seed: u64) -> Result<(bool, f64), KError> {
    let nc = p.ncols();
    if nc == 0 {
        return Ok((true, 1.0));
    }
    let n = p.nrows();
    let s = s.max(1).min(nc);
    let mut w = Mat::<f64>::zeros(n, s);
    let mut x = vec![0.0f64; nc];
    let mut y = vec![0.0f64; n];
    let mut omega_cols: Vec<Vec<i8>> = Vec::with_capacity(s);
    for col in 0..s {
        let mut col_vals = vec![0i8; nc];
        for i in 0..nc {
            let mut h = seed
                ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (col as u64).wrapping_mul(0xD00D_F00D_F00D_F00D);
            h ^= h >> 30;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 27;
            h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
            h ^= h >> 31;
            col_vals[i] = if h & 1 == 0 { 1 } else { -1 };
        }
        if nc > 0 {
            'adjust: loop {
                let mut tweaked = false;
                for prev in 0..omega_cols.len() {
                    let mut same = true;
                    let mut neg = true;
                    for i in 0..nc {
                        let v = col_vals[i];
                        let pv = omega_cols[prev][i];
                        if v != pv {
                            same = false;
                        }
                        if v != -pv {
                            neg = false;
                        }
                        if !same && !neg {
                            break;
                        }
                    }
                    if same || neg {
                        let idx = (col + prev) % nc;
                        col_vals[idx] = -col_vals[idx];
                        tweaked = true;
                        break;
                    }
                }
                if !tweaked {
                    break 'adjust;
                }
            }
        }
        for i in 0..nc {
            x[i] = col_vals[i] as f64;
        }
        p.spmv_scaled(1.0, &x, 0.0, &mut y)?;
        for i in 0..n {
            w[(i, col)] = y[i];
        }
        omega_cols.push(col_vals);
    }
    let mut s_mat = Mat::<f64>::zeros(s, s);
    for i in 0..s {
        for j in i..s {
            let mut dot = 0.0f64;
            for k in 0..n {
                dot += w[(k, i)] * w[(k, j)];
            }
            s_mat[(i, j)] = dot;
            s_mat[(j, i)] = dot;
        }
    }
    let mut lam = symmetric_eigenvalues(s_mat);
    lam.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let lam_min = lam.first().copied().unwrap_or(0.0).max(0.0);
    let lam_max = lam.last().copied().unwrap_or(0.0).max(0.0);
    let cond = if lam_min > 0.0 {
        lam_max / lam_min
    } else if lam_max == 0.0 {
        1.0
    } else {
        f64::INFINITY
    };
    let ok = lam_min.is_finite() && lam_max.is_finite() && cond.is_finite();
    Ok((ok, cond))
}

fn check_p_rank_fast(p: &CsrMatrix<f64>, cfg: &AMGConfig) -> Result<RankDiagnostics, KError> {
    if p.ncols() == 0 {
        return Ok(RankDiagnostics::default());
    }
    let norms2 = p_column_norms2(p);
    let mut diag = RankDiagnostics::default();
    let mut min_norm = f64::MAX;
    for (j, &n2) in norms2.iter().enumerate() {
        let norm = n2.sqrt();
        if norm < cfg.rank_min_col_norm {
            diag.degenerate_cols.push(j);
        }
        if norm < min_norm {
            min_norm = norm;
        }
    }
    diag.min_col_norm = if min_norm.is_finite() { min_norm } else { 0.0 };
    let (ok, cond) = rank_condition_estimate(p, cfg.rank_sketch_cols, 0x00C0_FFEE_u64)?;
    diag.cond_estimate = cond;
    let cond_bad = !ok || !cond.is_finite() || cond > cfg.rank_cond_threshold;
    diag.suspect = !diag.degenerate_cols.is_empty() || cond_bad;
    Ok(diag)
}

fn try_fix_rank(
    _level_idx: usize,
    a_l: &CsrMatrix<f64>,
    diag_inv_l: &[f64],
    tp: &TentativeP,
    ctx: &LevelPostContext,
    p_csr: &mut Pcsr,
    cfg: &mut AMGConfig,
) -> Result<RankFixOutcome, KError> {
    let old_drop = cfg.interpolation_truncation;
    let old_cap = cfg.max_elements_per_row;
    cfg.interpolation_truncation = (old_drop * 0.1).min(old_drop);
    if old_cap == 0 {
        cfg.max_elements_per_row = 16;
    } else {
        cfg.max_elements_per_row = old_cap.max(16);
    }

    let mut new_vals = vec![0.0; p_csr.col_idx.len()];
    if tp.num_functions > 1 {
        smooth_sa_values_only_multi(
            a_l,
            diag_inv_l,
            tp,
            cfg.jacobi_omega,
            &p_csr.row_ptr,
            &p_csr.col_idx,
            &mut new_vals,
        )?;
    } else {
        smooth_sa_values_only(
            a_l,
            diag_inv_l,
            tp,
            cfg.jacobi_omega,
            &p_csr.row_ptr,
            &p_csr.col_idx,
            &mut new_vals,
        )?;
    }
    p_csr.vals.copy_from_slice(&new_vals);
    apply_post_interp(cfg, ctx, &p_csr.row_ptr, &p_csr.col_idx, &mut p_csr.vals)?;
    let p_tmp = CsrMatrix::from_csr(
        p_csr.m,
        p_csr.n,
        p_csr.row_ptr.clone(),
        p_csr.col_idx.clone(),
        p_csr.vals.clone(),
    );
    let diag = check_p_rank_fast(&p_tmp, cfg)?;

    cfg.interpolation_truncation = old_drop;
    cfg.max_elements_per_row = old_cap;

    if diag.suspect {
        Ok(RankFixOutcome::Unfixed)
    } else {
        Ok(RankFixOutcome::Fixed)
    }
}

fn galerkin_sample_check(
    a_l: &CsrMatrix<f64>,
    p_l: &CsrMatrix<f64>,
    r_l: &CsrMatrix<f64>,
    a_c: &CsrMatrix<f64>,
    samples: usize,
    tol: f64,
    seed: u64,
) -> Result<(bool, f64), KError> {
    let n = a_l.nrows();
    let nc = a_c.nrows();
    if nc == 0 {
        return Ok((true, 0.0));
    }
    let s = samples.max(1);
    let mut worst = 0.0f64;
    let mut y = vec![0.0f64; nc];
    let mut x = vec![0.0f64; n];
    let mut ax = vec![0.0f64; n];
    let mut u = vec![0.0f64; nc];
    let mut v = vec![0.0f64; nc];
    for t in 0..s {
        for i in 0..nc {
            let h = seed ^ ((t as u64).wrapping_mul(0x9E37) ^ (i as u64).wrapping_mul(0xD00D));
            y[i] = if (h >> 1) & 1 == 0 { 1.0 } else { -1.0 };
        }
        p_l.spmv_scaled(1.0, &y, 0.0, &mut x)?;
        a_l.spmv_scaled(1.0, &x, 0.0, &mut ax)?;
        r_l.spmv_scaled(1.0, &ax, 0.0, &mut u)?;
        a_c.spmv_scaled(1.0, &y, 0.0, &mut v)?;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..nc {
            let d = u[i] - v[i];
            num += d * d;
            den += v[i] * v[i];
        }
        let rel = num.sqrt() / den.sqrt().max(1e-300);
        if rel > worst {
            worst = rel;
        }
    }
    Ok((worst <= tol, worst))
}

fn csr_pattern_hash(a: &CsrMatrix<f64>) -> u64 {
    let mut hasher = DefaultHasher::new();
    a.row_ptr().hash(&mut hasher);
    a.col_idx().hash(&mut hasher);
    hasher.finish()
}

#[cfg(not(feature = "complex"))]
fn pack_message_u64(message: &str) -> (u64, Vec<u64>) {
    let bytes = message.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return (0, Vec::new());
    }
    let words = (len + 7) / 8;
    let mut data = vec![0u64; words];
    for (idx, &b) in bytes.iter().enumerate() {
        let word = idx / 8;
        let shift = (idx % 8) * 8;
        data[word] |= (b as u64) << shift;
    }
    (len as u64, data)
}

#[cfg(not(feature = "complex"))]
fn unpack_message_u64(words: &[u64], len: usize) -> String {
    if len == 0 {
        return String::new();
    }
    let mut bytes = Vec::with_capacity(len);
    for (i, &word) in words.iter().enumerate() {
        let base = i * 8;
        for j in 0..8 {
            let idx = base + j;
            if idx >= len {
                break;
            }
            bytes.push(((word >> (j * 8)) & 0xFF) as u8);
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

#[cfg(not(feature = "complex"))]
fn broadcast_message<C: Comm>(comm: &C, root: usize, message: Option<String>) -> String {
    let rank = comm.rank();
    let size = comm.size();
    if size <= 1 {
        return message.unwrap_or_default();
    }
    if rank == root {
        let msg = message.unwrap_or_default();
        let (len, data) = pack_message_u64(&msg);
        let len_buf = [len];
        let mut reqs = Vec::new();
        for r in 0..size {
            if r == root {
                continue;
            }
            reqs.push(comm.isend_to_u64(&len_buf, r as i32));
        }
        comm.wait_all(&mut reqs);
        if len > 0 {
            let mut data_reqs = Vec::new();
            for r in 0..size {
                if r == root {
                    continue;
                }
                data_reqs.push(comm.isend_to_u64(&data, r as i32));
            }
            comm.wait_all(&mut data_reqs);
        }
        msg
    } else {
        let mut len_buf = [0u64];
        {
            let mut reqs = vec![comm.irecv_from_u64(&mut len_buf, root as i32)];
            comm.wait_all(&mut reqs);
        }
        let len = len_buf[0] as usize;
        if len == 0 {
            return String::new();
        }
        let words = (len + 7) / 8;
        let mut data = vec![0u64; words];
        {
            let mut data_reqs = vec![comm.irecv_from_u64(&mut data, root as i32)];
            comm.wait_all(&mut data_reqs);
        }
        unpack_message_u64(&data, len)
    }
}

#[cfg(not(feature = "complex"))]
fn collect_error_message<C: Comm>(
    comm: &C,
    root: usize,
    local_message: Option<String>,
) -> String {
    let rank = comm.rank();
    let size = comm.size();
    let local = local_message.unwrap_or_default();
    if size <= 1 {
        return local;
    }
    let (len, data) = pack_message_u64(&local);
    if rank != root {
        let len_buf = [len];
        let mut reqs = vec![comm.isend_to_u64(&len_buf, root as i32)];
        comm.wait_all(&mut reqs);
        if len > 0 {
            let mut data_reqs = vec![comm.isend_to_u64(&data, root as i32)];
            comm.wait_all(&mut data_reqs);
        }
        return broadcast_message(comm, root, None);
    }

    let mut len_bufs = vec![[0u64; 1]; size];
    let mut len_reqs = Vec::new();
    for r in 0..size {
        if r == root {
            continue;
        }
        let buf = unsafe { &mut *len_bufs.as_mut_ptr().add(r) };
        len_reqs.push(comm.irecv_from_u64(buf, r as i32));
    }
    comm.wait_all(&mut len_reqs);

    let mut data_bufs: Vec<Vec<u64>> = vec![Vec::new(); size];
    {
        let mut data_reqs = Vec::new();
        for r in 0..size {
            if r == root {
                continue;
            }
            let msg_len = len_bufs[r][0] as usize;
            if msg_len == 0 {
                continue;
            }
            let words = (msg_len + 7) / 8;
            let buf = unsafe { &mut *data_bufs.as_mut_ptr().add(r) };
            *buf = vec![0u64; words];
            data_reqs.push(comm.irecv_from_u64(buf.as_mut_slice(), r as i32));
        }
        comm.wait_all(&mut data_reqs);
    }

    let mut messages = vec![String::new(); size];
    messages[root] = local;
    for r in 0..size {
        if r == root {
            continue;
        }
        let msg_len = len_bufs[r][0] as usize;
        if msg_len == 0 {
            continue;
        }
        messages[r] = unpack_message_u64(&data_bufs[r], msg_len);
    }
    let chosen = messages
        .iter()
        .enumerate()
        .find(|(_, msg)| !msg.is_empty())
        .map(|(_, msg)| msg.clone())
        .unwrap_or_default();
    broadcast_message(comm, root, Some(chosen))
}

#[cfg(not(feature = "complex"))]
fn gather_dist_csr(
    dist: &DistCsrOp,
    root: usize,
) -> Result<Option<CsrMatrix<f64>>, KError> {
    let comm = dist.comm();
    let rank = comm.rank();
    let size = comm.size();
    let row_part = dist.row_partition();
    if row_part.len() != size + 1 {
        return Err(KError::InvalidInput(
            "distributed row partition is inconsistent with communicator size".into(),
        ));
    }
    let local = dist.local_matrix();
    let local_row_ptr = local.row_ptr().to_vec();
    let local_col_idx = local.col_idx().to_vec();
    let local_vals = local.values().to_vec();
    let local_nnz = local_col_idx.len() as u64;
    let mut nnz_counts = Vec::new();
    comm.gather(&[local_nnz], &mut nnz_counts, root);

    if rank != root {
        let row_ptr_u64: Vec<u64> = local_row_ptr.iter().map(|&v| v as u64).collect();
        let col_idx_u64: Vec<u64> = local_col_idx.iter().map(|&v| v as u64).collect();
        let mut reqs = Vec::new();
        reqs.push(comm.isend_to_u64(&row_ptr_u64, root as i32));
        reqs.push(comm.isend_to_u64(&col_idx_u64, root as i32));
        reqs.push(comm.isend_to(&local_vals, root as i32));
        comm.wait_all(&mut reqs);
        return Ok(None);
    }

    let mut recv_row_ptr_u64: Vec<Vec<u64>> = vec![Vec::new(); size];
    let mut recv_col_idx_u64: Vec<Vec<u64>> = vec![Vec::new(); size];
    let mut recv_vals: Vec<Vec<f64>> = vec![Vec::new(); size];
    for r in 0..size {
        if r == root {
            continue;
        }
        let n_local = row_part[r + 1] - row_part[r];
        let nnz = *nnz_counts.get(r).unwrap_or(&0) as usize;
        recv_row_ptr_u64[r] = vec![0u64; n_local + 1];
        recv_col_idx_u64[r] = vec![0u64; nnz];
        recv_vals[r] = vec![0.0; nnz];
        let mut reqs = Vec::with_capacity(3);
        reqs.push(comm.irecv_from_u64(
            recv_row_ptr_u64[r].as_mut_slice(),
            r as i32,
        ));
        reqs.push(comm.irecv_from_u64(
            recv_col_idx_u64[r].as_mut_slice(),
            r as i32,
        ));
        reqs.push(comm.irecv_from(recv_vals[r].as_mut_slice(), r as i32));
        comm.wait_all(&mut reqs);
    }

    let mut row_ptrs: Vec<Vec<usize>> = vec![Vec::new(); size];
    let mut col_idxs: Vec<Vec<usize>> = vec![Vec::new(); size];
    let mut vals: Vec<Vec<f64>> = vec![Vec::new(); size];
    row_ptrs[root] = local_row_ptr;
    col_idxs[root] = local_col_idx;
    vals[root] = local_vals;
    for r in 0..size {
        if r == root {
            continue;
        }
        row_ptrs[r] = recv_row_ptr_u64[r].iter().map(|&v| v as usize).collect();
        col_idxs[r] = recv_col_idx_u64[r].iter().map(|&v| v as usize).collect();
        vals[r] = recv_vals[r].clone();
    }

    let n_global = dist.n_global;
    let mut row_nnz = vec![0usize; n_global];
    for r in 0..size {
        let row_start = row_part[r];
        let n_local = row_part[r + 1] - row_part[r];
        if n_local + 1 != row_ptrs[r].len() {
            return Err(KError::InvalidInput(format!(
                "rank {r} row_ptr length mismatch: expected {}, got {}",
                n_local + 1,
                row_ptrs[r].len()
            )));
        }
        for i in 0..n_local {
            let nnz = row_ptrs[r][i + 1] - row_ptrs[r][i];
            row_nnz[row_start + i] = nnz;
        }
    }
    let mut row_ptr_global = vec![0usize; n_global + 1];
    for i in 0..n_global {
        row_ptr_global[i + 1] = row_ptr_global[i] + row_nnz[i];
    }
    let total_nnz = row_ptr_global[n_global];
    let mut col_idx_global = vec![0usize; total_nnz];
    let mut vals_global = vec![0.0f64; total_nnz];
    let mut next_pos = row_ptr_global.clone();
    for r in 0..size {
        let row_start = row_part[r];
        let n_local = row_part[r + 1] - row_part[r];
        for i in 0..n_local {
            let global_row = row_start + i;
            let mut slot = next_pos[global_row];
            let rs = row_ptrs[r][i];
            let re = row_ptrs[r][i + 1];
            for p in rs..re {
                col_idx_global[slot] = col_idxs[r][p];
                vals_global[slot] = vals[r][p];
                slot += 1;
            }
            next_pos[global_row] = slot;
        }
    }

    Ok(Some(CsrMatrix::from_csr(
        n_global,
        n_global,
        row_ptr_global,
        col_idx_global,
        vals_global,
    )))
}

#[cfg(not(feature = "complex"))]
fn gather_vector(
    comm: &UniverseComm,
    row_part: &[usize],
    root: usize,
    local: &[f64],
) -> Result<Option<Vec<f64>>, KError> {
    // Root-centric gather: every rank sends its owned slice to the root, which
    // assembles the full global vector. Non-root ranks return None.
    let rank = comm.rank();
    let size = comm.size();
    if row_part.len() != size + 1 {
        return Err(KError::InvalidInput(
            "distributed row partition is inconsistent with communicator size".into(),
        ));
    }
    let (start, end) = (row_part[rank], row_part[rank + 1]);
    if local.len() != end.saturating_sub(start) {
        return Err(KError::InvalidInput(
            "distributed vector length does not match local row partition".into(),
        ));
    }
    if rank != root {
        let mut reqs = vec![comm.isend_to(local, root as i32)];
        comm.wait_all(&mut reqs);
        return Ok(None);
    }
    let n_global = *row_part.last().unwrap_or(&0);
    let mut global = vec![0.0f64; n_global];
    global[start..end].copy_from_slice(local);
    for r in 0..size {
        if r == root {
            continue;
        }
        let rs = row_part[r];
        let re = row_part[r + 1];
        let mut reqs = Vec::with_capacity(1);
        reqs.push(comm.irecv_from(&mut global[rs..re], r as i32));
        comm.wait_all(&mut reqs);
    }
    Ok(Some(global))
}

#[cfg(not(feature = "complex"))]
fn scatter_vector(
    comm: &UniverseComm,
    row_part: &[usize],
    root: usize,
    global: Option<&[f64]>,
    local_out: &mut [f64],
) -> Result<(), KError> {
    // Root-centric scatter: the root slices the global vector by row partition
    // and sends each owned segment to its rank. Non-root ranks only receive.
    let rank = comm.rank();
    let size = comm.size();
    if row_part.len() != size + 1 {
        return Err(KError::InvalidInput(
            "distributed row partition is inconsistent with communicator size".into(),
        ));
    }
    let (start, end) = (row_part[rank], row_part[rank + 1]);
    if local_out.len() != end.saturating_sub(start) {
        return Err(KError::InvalidInput(
            "distributed vector length does not match local row partition".into(),
        ));
    }
    if rank == root {
        let global = global.ok_or_else(|| {
            KError::InvalidInput("root rank missing global vector for scatter".into())
        })?;
        if global.len() < *row_part.last().unwrap_or(&0) {
            return Err(KError::InvalidInput(
                "global vector length does not match distributed partition".into(),
            ));
        }
        local_out.copy_from_slice(&global[start..end]);
        let mut reqs = Vec::new();
        for r in 0..size {
            if r == root {
                continue;
            }
            let rs = row_part[r];
            let re = row_part[r + 1];
            reqs.push(comm.isend_to(&global[rs..re], r as i32));
        }
        comm.wait_all(&mut reqs);
        return Ok(());
    }
    let mut reqs = vec![comm.irecv_from(local_out, root as i32)];
    comm.wait_all(&mut reqs);
    Ok(())
}

#[cfg(not(feature = "complex"))]
fn owner_of_row(row_part: &[usize], gcol: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = row_part.len().saturating_sub(2);
    while lo <= hi {
        let mid = (lo + hi) / 2;
        if gcol < row_part[mid + 1] {
            if gcol >= row_part[mid] {
                return mid;
            }
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    lo.min(row_part.len().saturating_sub(2))
}

#[cfg(not(feature = "complex"))]
fn build_amg_halo_plan(
    comm: UniverseComm,
    row_part: Arc<Vec<usize>>,
    row_start: usize,
    row_end: usize,
    local: &CsrMatrix<f64>,
) -> Result<HaloPlan, KError> {
    let rank = comm.rank();
    let mut recv_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let row_ptr = local.row_ptr();
    let col_idx = local.col_idx();
    for row in 0..local.nrows() {
        for idx in row_ptr[row]..row_ptr[row + 1] {
            let gcol = col_idx[idx];
            let owner = owner_of_row(&row_part, gcol);
            if owner != rank {
                recv_map.entry(owner).or_default().push(gcol);
            }
        }
    }
    HaloPlan::new(comm, row_part, row_start, row_end, recv_map)
}

#[cfg(debug_assertions)]
fn debug_check_csr(a: &CsrMatrix<f64>, name: &str) {
    let row_ptr = a.row_ptr();
    let nrows = a.nrows();
    let nnz = a.nnz();
    let col_idx = a.col_idx();
    let vals = a.values();
    debug_assert_eq!(
        row_ptr.len(),
        nrows + 1,
        "{name}: row_ptr.len() mismatch ({} vs {})",
        row_ptr.len(),
        nrows + 1
    );
    debug_assert_eq!(
        col_idx.len(),
        nnz,
        "{name}: col_idx.len() ({}) != nnz ({})",
        col_idx.len(),
        nnz
    );
    debug_assert_eq!(
        vals.len(),
        nnz,
        "{name}: vals.len() ({}) != nnz ({})",
        vals.len(),
        nnz
    );
    debug_assert_eq!(
        row_ptr[nrows], nnz,
        "{name}: row_ptr[nrows] ({}) != nnz ({})",
        row_ptr[nrows], nnz
    );
    let ncols = a.ncols();
    for row in 0..nrows {
        let start = row_ptr[row];
        let end = row_ptr[row + 1];
        debug_assert!(
            start <= end,
            "{name}: row {row} pointers out-of-order ({start}..{end})"
        );
        let mut last_col = None;
        for idx in start..end {
            let col = col_idx[idx];
            debug_assert!(
                col < ncols,
                "{name}: column index {} (row {}) out of bounds (ncols={ncols})",
                col,
                row
            );
            if let Some(prev) = last_col {
                debug_assert!(
                    col >= prev,
                    "{name}: column index decreased at row {}: {} < {}",
                    row,
                    col,
                    prev
                );
            }
            last_col = Some(col);
            let val = vals[idx];
            debug_assert!(
                val.is_finite(),
                "{name}: non-finite value at row {} (idx {}): {}",
                row,
                idx,
                val
            );
        }
    }
}

enum AmgState {
    Uninitialized,
    SymbolicOnly {
        hierarchy: Box<AmgHierarchy>,
        last_structure_id: StructureId,
        pattern_hash: u64,
    },
    Ready {
        hierarchy: Box<AmgHierarchy>,
        last_structure_id: StructureId,
        last_values_id: ValuesId,
        pattern_hash: u64,
    },
}

impl AmgState {
    fn as_ref(&self) -> Option<&AmgHierarchy> {
        match self {
            AmgState::SymbolicOnly { hierarchy, .. } => Some(hierarchy),
            AmgState::Ready { hierarchy, .. } => Some(hierarchy),
            _ => None,
        }
    }
}

struct DistAmgInfo {
    comm: UniverseComm,
    root: usize,
    row_part: Arc<Vec<usize>>,
    n_global: usize,
    local_hierarchy: Option<Box<AmgHierarchy>>,
    local_matrix: Option<Arc<CsrMatrix<f64>>>,
    halo_plan: Option<HaloPlan>,
}

impl DistAmgInfo {
    fn local_range(&self) -> (usize, usize) {
        let rank = self.comm.rank();
        let start = self
            .row_part
            .get(rank)
            .copied()
            .unwrap_or_default();
        let end = self
            .row_part
            .get(rank + 1)
            .copied()
            .unwrap_or(start);
        (start, end)
    }

    fn local_nrows(&self) -> usize {
        let (start, end) = self.local_range();
        end.saturating_sub(start)
    }
}

// ===== Main AMG object =======================================================

pub struct AMG {
    csr: Option<Arc<CsrMatrix<f64>>>,
    state: AmgState,
    cycle_policy: Box<dyn CyclePolicy + Send + Sync>,
    cfg: AMGConfig,
    stats: Option<AmgStats>,
    runtime: Mutex<AmgRuntime>,
    dist: Option<DistAmgInfo>,
}

impl Default for AMG {
    fn default() -> Self {
        let cfg = AMGConfig::default();
        Self {
            csr: None,
            state: AmgState::Uninitialized,
            cycle_policy: Self::make_cycle_policy(&cfg),
            cfg,
            stats: None,
            runtime: Mutex::new(AmgRuntime::default()),
            dist: None,
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
    pub fn with_config(mut cfg: AMGConfig) -> Self {
        if cfg.spd_diag_floor < 0.0 {
            cfg.spd_diag_floor = 0.0;
        }
        Self {
            cycle_policy: Self::make_cycle_policy(&cfg),
            cfg,
            state: AmgState::Uninitialized,
            ..Default::default()
        }
    }

    fn make_cycle_policy(cfg: &AMGConfig) -> Box<dyn CyclePolicy + Send + Sync> {
        match (&cfg.cycle_type, cfg.kcycle.as_ref()) {
            (CycleType::V, None) => Box::new(VPolicy),
            (CycleType::W { gamma }, None) => Box::new(WPolicy {
                gamma: (*gamma).max(2),
            }),
            (base, Some(kc)) => Box::new(KPolicy::new(*base, kc.clone())),
        }
    }

    pub fn extract_coarse_space(&self, opts: &DeflationOptions) -> Result<AmgCoarseSpace, KError> {
        let state = match &self.state {
            AmgState::Ready { hierarchy, .. } => hierarchy,
            _ => return Err(KError::InvalidInput("AMG not set up".into())),
        };
        match opts.z_source {
            ZSource::CoarsestIdentity { cap_k } => {
                let coarse_ix = state.coarsest_ix();
                let n_coarse = state.levels[coarse_ix].a.nrows();
                let k_full = n_coarse;
                let cap = cap_k.unwrap_or(k_full).min(k_full);
                let mut z = Mat::<f64>::zeros(n_coarse, cap);
                for i in 0..cap {
                    z[(i, i)] = 1.0;
                }
                let mut current = z;
                for lvl in (0..coarse_ix).rev() {
                    let p = &state.levels[lvl].p;
                    let mut next = Mat::<f64>::zeros(p.nrows(), cap);
                    csr_spmm_dense(p, current.as_ref(), next.as_mut())?;
                    current = next;
                }
                Ok(AmgCoarseSpace {
                    z: current,
                    local_range: None,
                })
            }
            ZSource::NearNullspace => {
                let finest = &state.levels[0];
                let basis = finest
                    .nns
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("near-nullspace unavailable".into()))?;
                let k = basis.len();
                if k == 0 {
                    return Err(KError::InvalidInput("near-nullspace empty".into()));
                }
                let n = finest.a.nrows();
                let mut z = Mat::<f64>::zeros(n, k);
                for (j, vec) in basis.iter().enumerate() {
                    if vec.len() != n {
                        return Err(KError::InvalidInput(
                            "near-nullspace vector has wrong length".into(),
                        ));
                    }
                    for i in 0..n {
                        z[(i, j)] = vec[i];
                    }
                }
                Ok(AmgCoarseSpace {
                    z,
                    local_range: None,
                })
            }
            ZSource::External => Err(KError::InvalidInput(
                "ZSource::External requires user-provided coarse space".into(),
            )),
        }
    }

    // ---- Setup paths --------------------------------------------------------

    #[cfg(not(feature = "complex"))]
    fn resolve_dist_coarse_strategy(
        &self,
        comm: &UniverseComm,
    ) -> Result<DistCoarseStrategy, KError> {
        let strategy = self.cfg.dist_coarse_strategy;
        match strategy {
            DistCoarseStrategy::None => {
                if comm.size() > 1 {
                    log::warn!(
                        "AMG distributed coarse strategy set to none; falling back to root_gather."
                    );
                    Ok(DistCoarseStrategy::RootGather)
                } else {
                    Ok(DistCoarseStrategy::None)
                }
            }
            DistCoarseStrategy::SuperLuDist => {
                #[cfg(feature = "superlu_dist")]
                {
                    log::warn!(
                        "AMG distributed coarse strategy superlu_dist not wired yet; falling back to root_gather."
                    );
                    Ok(DistCoarseStrategy::RootGather)
                }
                #[cfg(not(feature = "superlu_dist"))]
                {
                    Err(KError::Unsupported(
                        "superlu_dist coarse strategy requires feature superlu_dist".into(),
                    ))
                }
            }
            _ => Ok(strategy),
        }
    }

    fn build_symbolic(&mut self, fine: &CsrMatrix<f64>) -> Result<Box<AmgHierarchy>, KError> {
        // Build the full hierarchy from scratch (symbolic + numeric)
        let mut cfg = self.cfg.clone();
        let primary = build_hierarchy(fine, &mut cfg);
        let (hier, stats, used_cfg) = match primary {
            Ok((hier, stats)) => (hier, stats, cfg),
            Err(primary_err) => {
                let mut fallback_cfg = cfg.clone();
                fallback_cfg.coarsen_type = CoarsenType::RS;
                fallback_cfg.interp_type = InterpType::Classical;
                match build_hierarchy(fine, &mut fallback_cfg) {
                    Ok((hier, stats)) => (hier, stats, fallback_cfg),
                    Err(fallback_err) => {
                        let mut smoother_cfg = cfg.clone();
                        smoother_cfg.coarse_solve = CoarseSolve::Smoother;
                        match build_smoother_only_hierarchy(fine, &mut smoother_cfg) {
                            Ok((hier, stats)) => (hier, stats, smoother_cfg),
                            Err(_) => {
                                let combined = format!(
                                    "AMG setup failed: {primary_err}; fallback failed: {fallback_err}"
                                );
                                return Err(match (&primary_err, &fallback_err) {
                                    (KError::SolveError(_), KError::SolveError(_)) => {
                                        KError::SolveError(combined)
                                    }
                                    _ => KError::InvalidInput(combined),
                                });
                            }
                        }
                    }
                }
            }
        };
        #[cfg(test)]
        BUILD_SYMBOLIC_COUNT.with(|c| c.set(c.get() + 1));
        self.cfg = used_cfg;
        self.cycle_policy = Self::make_cycle_policy(&self.cfg);
        self.stats = stats;
        Ok(Box::new(hier))
    }

    #[cfg(not(feature = "complex"))]
    fn setup_dist(&mut self, dist: &DistCsrOp) -> Result<(), KError> {
        let strategy = self.resolve_dist_coarse_strategy(&dist.comm())?;
        if matches!(strategy, DistCoarseStrategy::LocalPrototype) {
            return self.setup_dist_local(dist);
        }
        let comm = dist.comm();
        let rank = comm.rank();
        let row_part = dist.row_partition();
        let root = 0usize;
        // Distributed setup is root-centric: we gather the global matrix, build the
        // hierarchy on rank 0, and keep non-root ranks in an uninitialized state.
        // Validation that depends on distributed apply (e.g., SPD probing) is disabled
        // under MPI because apply_dist is collective. Errors are synchronized so all
        // ranks return the same failure before any iteration starts.
        self.dist = Some(DistAmgInfo {
            comm: comm.clone(),
            root,
            row_part: row_part.clone(),
            n_global: dist.n_global,
            local_hierarchy: None,
            local_matrix: None,
            halo_plan: None,
        });
        let prev_cfg = self.cfg.clone();
        let mut local_stage: Option<&'static str> = None;
        let mut local_detail: Option<String> = None;
        let mut record_error =
            |local_stage: &mut Option<&'static str>,
             local_detail: &mut Option<String>,
             stage: &'static str,
             err: KError| {
                if local_stage.is_none() {
                    *local_stage = Some(stage);
                    *local_detail = Some(err.to_string());
                }
            };
        let mut setup_state: Option<(
            CsrMatrix<f64>,
            Box<AmgHierarchy>,
            StructureId,
            ValuesId,
            u64,
        )> = None;
        let mut setup_profile: Option<&'static str> = None;
        let global = match gather_dist_csr(dist, root) {
            Ok(global) => global,
            Err(err) => {
                record_error(&mut local_stage, &mut local_detail, "gather_dist_csr", err);
                None
            }
        };
        if local_stage.is_none() && rank == root {
            match global {
                Some(mut fine) => {
                    let cfg = self.cfg.clone();
                    if cfg.conditioning.is_active() {
                        if let Err(err) =
                            apply_csr_transforms("AMG", &mut fine, &cfg.conditioning)
                        {
                            record_error(&mut local_stage, &mut local_detail, "conditioning", err);
                        }
                    }
                    if local_stage.is_none() {
                        let sid = dist.structure_id();
                        let vid = dist.values_id();
                        let pattern_hash = csr_pattern_hash(&fine);
                        match self.build_symbolic(&fine) {
                            Ok(hierarchy) => {
                                setup_profile = Some("strict");
                                setup_state =
                                    Some((fine, hierarchy, sid, vid, pattern_hash));
                            }
                            Err(primary_err) => {
                                let mut permissive_cfg = prev_cfg.clone();
                                permissive_cfg.require_spd = false;
                                permissive_cfg.verify_galerkin = false;
                                permissive_cfg.verify_p_rank = false;
                                permissive_cfg.interp_type = InterpType::Classical;
                                self.cfg = permissive_cfg;
                                match self.build_symbolic(&fine) {
                                    Ok(hierarchy) => {
                                        setup_profile = Some("permissive");
                                        setup_state =
                                            Some((fine, hierarchy, sid, vid, pattern_hash));
                                    }
                                    Err(fallback_err) => {
                                        self.cfg = prev_cfg.clone();
                                        record_error(
                                            &mut local_stage,
                                            &mut local_detail,
                                            "build_symbolic",
                                            KError::InvalidInput(format!(
                                                "strict setup failed: {primary_err}; permissive fallback failed: {fallback_err}"
                                            )),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    record_error(
                        &mut local_stage,
                        &mut local_detail,
                        "gather_dist_csr",
                        KError::InvalidInput("root rank missing assembled CSR matrix".into()),
                    );
                }
            }
        }
        let local_failure = if local_stage.is_none() { 0.0 } else { 1.0 };
        let failure_sum = comm.all_reduce_f64(local_failure);
        if failure_sum > 0.0 {
            let local_message = local_stage.map(|stage| {
                let detail = local_detail
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string());
                format!("stage={stage}: {detail}")
            });
            let error_message = collect_error_message(&comm, root, local_message);
            self.cfg = prev_cfg;
            self.cycle_policy = Self::make_cycle_policy(&self.cfg);
            self.state = AmgState::Uninitialized;
            self.stats = None;
            self.csr = None;
            return Err(KError::InvalidInput(
                format!(
                    "AMG distributed setup failed: {}",
                    if error_message.is_empty() {
                        "unknown error".to_string()
                    } else {
                        error_message
                    }
                ),
            ));
        }
        if rank != root {
            self.state = AmgState::Uninitialized;
            self.stats = None;
            self.csr = None;
            return Ok(());
        }
        let (fine, hierarchy, sid, vid, pattern_hash) = setup_state.ok_or_else(
            || KError::InvalidInput("AMG distributed setup missing hierarchy state".into()),
        )?;
        self.state = AmgState::Ready {
            hierarchy,
            last_structure_id: sid,
            last_values_id: vid,
            pattern_hash,
        };
        self.csr = Some(Arc::new(fine));
        if let Some(profile) = setup_profile
            && profile == "permissive"
            && self.cfg.print_level >= 1
        {
            println!("AMG distributed setup succeeded using permissive configuration.");
        }
        if self.cfg.logging_level >= 2
            && self.cfg.print_level >= 1
            && let Some(s) = self.stats.as_ref()
        {
            print_setup_tables(s);
        }
        Ok(())
    }

    #[cfg(not(feature = "complex"))]
    fn setup_dist_local(&mut self, dist: &DistCsrOp) -> Result<(), KError> {
        let comm = dist.comm();
        let rank = comm.rank();
        let row_part = dist.row_partition();
        let root = 0usize;

        let mut local_stage: Option<&'static str> = None;
        let mut local_detail: Option<String> = None;
        let mut record_error =
            |local_stage: &mut Option<&'static str>,
             local_detail: &mut Option<String>,
             stage: &'static str,
             err: KError| {
                if local_stage.is_none() {
                    *local_stage = Some(stage);
                    *local_detail = Some(err.to_string());
                }
            };

        let local_matrix = dist.local_matrix();
        let local_block = dist.local_block_csr();
        let mut local_hierarchy: Option<Box<AmgHierarchy>> = None;
        let mut halo_plan: Option<HaloPlan> = None;

        if local_stage.is_none() {
            let mut local_amg = AMG::with_config(self.cfg.clone());
            match local_amg.build_symbolic(&local_block) {
                Ok(hier) => {
                    local_hierarchy = Some(hier);
                }
                Err(err) => {
                    record_error(
                        &mut local_stage,
                        &mut local_detail,
                        "build_symbolic(local)",
                        err,
                    );
                }
            }
        }

        if local_stage.is_none() {
            match build_amg_halo_plan(
                comm.clone(),
                row_part.clone(),
                dist.row_start,
                dist.row_end,
                &local_matrix,
            ) {
                Ok(plan) => halo_plan = Some(plan),
                Err(err) => {
                    record_error(&mut local_stage, &mut local_detail, "build_halo_plan", err);
                }
            }
        }

        let local_failure = if local_stage.is_none() { 0.0 } else { 1.0 };
        let failure_sum = comm.all_reduce_f64(local_failure);
        if failure_sum > 0.0 {
            let local_message = local_stage.map(|stage| {
                let detail = local_detail
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string());
                format!("stage={stage}: {detail}")
            });
            let error_message = collect_error_message(&comm, root, local_message);
            self.state = AmgState::Uninitialized;
            self.stats = None;
            self.csr = None;
            return Err(KError::InvalidInput(format!(
                "AMG distributed local setup failed: {}",
                if error_message.is_empty() {
                    "unknown error".to_string()
                } else {
                    error_message
                }
            )));
        }

        self.dist = Some(DistAmgInfo {
            comm: comm.clone(),
            root,
            row_part: row_part.clone(),
            n_global: dist.n_global,
            local_hierarchy,
            local_matrix: Some(Arc::new(local_matrix)),
            halo_plan,
        });
        self.state = AmgState::Uninitialized;
        self.stats = None;
        self.csr = None;

        if self.cfg.print_level >= 1 && self.cfg.logging_level >= 1 {
            log::info!(
                "AMG distributed local prototype setup complete: rank={} local_rows={}",
                rank,
                dist.local_nrows()
            );
        }
        Ok(())
    }

    fn apply_local(&self, side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        if r.len() != z.len() {
            return Err(KError::InvalidInput(format!(
                "AMG.apply: r/z size mismatch: {} vs {}",
                r.len(),
                z.len()
            )));
        }
        if self.cfg.require_spd && side != PcSide::Left {
            return Err(KError::InvalidInput(
                "AMG in SPD mode supports only Left preconditioning for CG-safe use".into(),
            ));
        }
        let h = match &self.state {
            AmgState::Ready { hierarchy, .. } => hierarchy,
            _ => {
                return Err(KError::InvalidInput("AMG not set up".into()));
            }
        };
        self.apply_with_hierarchy(side, r, z, h)
    }

    fn apply_with_hierarchy(
        &self,
        side: PcSide,
        r: &[f64],
        z: &mut [f64],
        h: &AmgHierarchy,
    ) -> Result<(), KError> {
        let mut ws = AMGWorkspace::new(h.finest().a.nrows());
        let do_prof = self.cfg.logging_level >= 2;
        if do_prof {
            let mut cyc = CycleTimings::default();
            let t_all = tic();
            z.fill(R::default());
            self.cycle_profiled(0, r, z, &mut ws, Some(&mut cyc))?;
            cyc.total_cycle = toc(t_all);
            cyc.cycle_type = self.cfg.cycle_type;
            cyc.kcycle = self.cfg.kcycle.clone();
            if let Ok(mut rt) = self.runtime.lock() {
                rt.last_cycle = Some(cyc.clone());
            }
            if self.cfg.print_level >= 2 {
                print_cycle_table(&cyc);
            }
        } else {
            z.fill(R::default());
            self.cycle(0, r, z, &mut ws)?;
        }
        Ok(())
    }

    #[cfg(not(feature = "complex"))]
    fn apply_dist(
        &self,
        side: PcSide,
        r: &[f64],
        z: &mut [f64],
        dist: &DistAmgInfo,
    ) -> Result<(), KError> {
        let do_prof = self.cfg.dist_apply_instrumentation;
        let mut stats = if do_prof {
            let mut stats = DistApplyStats::default();
            stats.mode = self.cfg.dist_coarse_strategy;
            Some(stats)
        } else {
            None
        };

        let strategy = self.resolve_dist_coarse_strategy(&dist.comm)?;
        let result = match strategy {
            DistCoarseStrategy::RootGather => {
                self.apply_dist_root(side, r, z, dist, stats.as_mut())
            }
            DistCoarseStrategy::LocalPrototype => {
                self.apply_dist_local(side, r, z, dist, stats.as_mut())
            }
            DistCoarseStrategy::None => Err(KError::Unsupported(
                "AMG distributed apply requires a coarse strategy (root_gather or local_prototype)"
                    .into(),
            )),
            DistCoarseStrategy::SuperLuDist => {
                self.apply_dist_root(side, r, z, dist, stats.as_mut())
            }
        };

        if do_prof {
            if let Ok(mut rt) = self.runtime.lock() {
                rt.last_dist_apply = stats;
            }
        } else if let Ok(mut rt) = self.runtime.lock() {
            rt.last_dist_apply = None;
        }

        result
    }

    #[cfg(not(feature = "complex"))]
    fn apply_dist_root(
        &self,
        side: PcSide,
        r: &[f64],
        z: &mut [f64],
        dist: &DistAmgInfo,
        mut stats: Option<&mut DistApplyStats>,
    ) -> Result<(), KError> {
        let comm = &dist.comm;
        let root = dist.root;
        let row_part = dist.row_part.as_ref();
        let rank = comm.rank();
        let t_gather = stats.as_ref().map(|_| tic());
        let global_r = gather_vector(comm, row_part, root, r)?;
        if let (Some(stats), Some(t0)) = (stats.as_mut(), t_gather) {
            stats.gather = toc(t0);
        }
        let mut global_z = if rank == root {
            vec![0.0f64; dist.n_global]
        } else {
            Vec::new()
        };
        if rank == root {
            let rhs = global_r.as_ref().ok_or_else(|| {
                KError::InvalidInput("root rank missing gathered RHS".into())
            })?;
            let t_local = stats.as_ref().map(|_| tic());
            self.apply_local(side, rhs, &mut global_z)?;
            if let (Some(stats), Some(t0)) = (stats.as_mut(), t_local) {
                stats.local_apply = toc(t0);
            }
        }
        let global_ref = if rank == root {
            Some(global_z.as_slice())
        } else {
            None
        };
        let t_scatter = stats.as_ref().map(|_| tic());
        scatter_vector(comm, row_part, root, global_ref, z)?;
        if let (Some(stats), Some(t0)) = (stats.as_mut(), t_scatter) {
            stats.scatter = toc(t0);
        }
        Ok(())
    }

    #[cfg(not(feature = "complex"))]
    fn apply_dist_local(
        &self,
        side: PcSide,
        r: &[f64],
        z: &mut [f64],
        dist: &DistAmgInfo,
        mut stats: Option<&mut DistApplyStats>,
    ) -> Result<(), KError> {
        let hierarchy = dist.local_hierarchy.as_ref().ok_or_else(|| {
            KError::InvalidInput("AMG local prototype hierarchy not initialized".into())
        })?;
        if r.len() != z.len() || r.len() != hierarchy.finest().a.nrows() {
            return Err(KError::InvalidInput(
                "AMG local prototype apply length mismatch".into(),
            ));
        }
        let t_local = stats.as_ref().map(|_| tic());
        self.apply_with_hierarchy(side, r, z, hierarchy)?;
        if let (Some(stats), Some(t0)) = (stats.as_mut(), t_local) {
            stats.local_apply = toc(t0);
        }

        if let (Some(halo), Some(local_matrix)) = (
            dist.halo_plan.as_ref(),
            dist.local_matrix.as_ref(),
        ) {
            let t_halo = stats.as_ref().map(|_| tic());
            let req = halo.post_halo(r);
            halo.complete_halo(req);
            if let (Some(stats), Some(t0)) = (stats.as_mut(), t_halo) {
                stats.halo_exchange = toc(t0);
            }
            if self.cfg.dist_coarse_ghost_scale > 0.0 {
                let ghost = halo.ghost_slice_ref();
                let row_ptr = local_matrix.row_ptr();
                let col_idx = local_matrix.col_idx();
                for row in 0..local_matrix.nrows() {
                    let mut acc = 0.0f64;
                    let mut count = 0usize;
                    for idx in row_ptr[row]..row_ptr[row + 1] {
                        let gcol = col_idx[idx];
                        if let Some(&slot) = halo.index.ghost_index_of.get(&gcol) {
                            acc += ghost[slot];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        z[row] += self.cfg.dist_coarse_ghost_scale * acc / (count as f64);
                    }
                }
            }
        }
        Ok(())
    }

    fn ensure_symbolic_structure(
        &mut self,
        fine: &CsrMatrix<f64>,
        sid: StructureId,
        pattern_hash: u64,
    ) -> Result<(), KError> {
        let needs_rebuild = match &self.state {
            AmgState::Uninitialized => true,
            AmgState::SymbolicOnly {
                last_structure_id,
                pattern_hash: hash,
                ..
            } => *last_structure_id != sid || *hash != pattern_hash,
            AmgState::Ready {
                last_structure_id,
                pattern_hash: hash,
                ..
            } => *last_structure_id != sid || *hash != pattern_hash,
        };
        if needs_rebuild {
            let hierarchy = self.build_symbolic(fine)?;
            self.state = AmgState::SymbolicOnly {
                hierarchy,
                last_structure_id: sid,
                pattern_hash,
            };
        }
        Ok(())
    }

    fn refresh_numeric_ready(
        &mut self,
        fine: &CsrMatrix<f64>,
        sid: StructureId,
        vid: ValuesId,
        pattern_hash: u64,
    ) -> Result<(), KError> {
        let mut hierarchy = match std::mem::replace(&mut self.state, AmgState::Uninitialized) {
            AmgState::SymbolicOnly { hierarchy, .. } => hierarchy,
            AmgState::Ready { hierarchy, .. } => hierarchy,
            AmgState::Uninitialized => {
                return Err(KError::InvalidInput(
                    "AMG internal state inconsistent".into(),
                ));
            }
        };
        self.refresh_numeric(fine, &mut hierarchy)?;
        self.state = AmgState::Ready {
            hierarchy,
            last_structure_id: sid,
            last_values_id: vid,
            pattern_hash,
        };
        Ok(())
    }

    fn refresh_numeric(
        &mut self,
        fine: &CsrMatrix<f64>,
        h: &mut AmgHierarchy,
    ) -> Result<(), KError> {
        if h.levels.is_empty() {
            return Err(KError::InvalidInput("AMG hierarchy empty".into()));
        }

        #[cfg(feature = "simd")]
        let spmv_tuning = utils::default_spmv_tuning();

        // Update finest A_0 and diag(A_0)^{-1}
        h.levels[0].a = fine.clone();
        let need_l1 = self.cfg.grid_relax_type.contains(&RelaxType::L1Jacobi);
        let need_cheb = self.cfg.grid_relax_type.contains(&RelaxType::Chebyshev);
        let need_cheb_safe = self.cfg.grid_relax_type.contains(&RelaxType::ChebyshevSafe);
        let need_safe_diag = self
            .cfg
            .grid_relax_type
            .contains(&RelaxType::SafeguardedGaussSeidel)
            || need_cheb_safe;
        let need_ilu0 = self.cfg.grid_relax_type.contains(&RelaxType::Ilu0);
        let need_ras = self.cfg.grid_relax_type.contains(&RelaxType::Ras);
        let allow_safeguard = need_safe_diag || need_ilu0 || need_ras;
        let need_fsai = self.cfg.grid_relax_type.contains(&RelaxType::Fsai);

        h.levels[0].diag_inv =
            diag_inv_from_csr_cfg_fallback(&h.levels[0].a, &self.cfg, allow_safeguard)?;
        let mut trials_current = make_trial_matrix(&self.cfg, h.levels[0].a.nrows())?;
        let mut diag_stats: Vec<AmgLevelStats> = Vec::new();
        diag_stats.push(AmgLevelStats {
            p_min_col_norm: 0.0,
            p_cond_sketched: 0.0,
            galerkin_worst_rel: 0.0,
        });

        // Recompute P_l values, R_l values, and A_{l+1} values using fixed patterns
        for l in 0..h.coarsest_ix() {
            // Recompute P_l values in-place using SA smoother with fixed pattern
            let pr = h.levels[l].p.row_ptr().to_vec();
            let pc = h.levels[l].p.col_idx().to_vec();
            let mut p_new_vals = vec![0.0f64; pc.len()];
            let mut tp_opt: Option<TentativeP> = None;
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
                        InterpType::HE => ClassicalVariant::HE,
                        InterpType::Standard | InterpType::Classical | InterpType::Extended => {
                            ClassicalVariant::Standard
                        }
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
                    num_functions: h.levels[l].num_functions,
                    nns: h.levels[l].nns.clone(),
                    comp_of: h.levels[l].layout.as_ref().map(|lay| lay.comp_of.clone()),
                };
                if let Some(ref rb) = h.levels[l].row_basis {
                    let n_agg = 1 + h.levels[l].agg_of.iter().copied().max().unwrap_or(0);
                    let tn = TentativeNodal {
                        agg_of: tp.agg_of.clone(),
                        n_agg,
                        mfun: tp.num_functions,
                        row_basis: rb.clone(),
                    };
                    smooth_sa_values_only_mf(
                        &h.levels[l].a,
                        &h.levels[l].diag_inv,
                        &tn,
                        self.cfg.jacobi_omega,
                        &pr,
                        &pc,
                        &mut p_new_vals,
                    )?;
                } else {
                    smooth_sa_values_only_multi(
                        &h.levels[l].a,
                        &h.levels[l].diag_inv,
                        &tp,
                        self.cfg.jacobi_omega,
                        &pr,
                        &pc,
                        &mut p_new_vals,
                    )?;
                }
                tp_opt = Some(tp);
            }
            let ctx = LevelPostContext {
                r: h.levels[l].num_functions,
                agg_of: &h.levels[l].agg_of,
                nns: h.levels[l]
                    .nns
                    .as_ref()
                    .map(|v| v.iter().map(|b| b.as_slice()).collect()),
                a: Some(&h.levels[l].a),
                d_inv: Some(&h.levels[l].diag_inv),
            };
            apply_post_interp(&self.cfg, &ctx, &pr, &pc, &mut p_new_vals)?;
            if self.cfg.adaptive_interp
                && self.cfg.adaptive_samples > 0
                && h.levels[l].cf.is_none()
                && h.levels[l + 1].a.nrows() > self.cfg.max_coarse_size
                && tp_opt.as_ref().is_some_and(|tp| tp.num_functions == 1)
            {
                let omega = if self.cfg.adaptive_smooth_omega == 0.0 {
                    self.cfg.jacobi_omega
                } else {
                    self.cfg.adaptive_smooth_omega
                };
                let samples = sample_low_modes(
                    &h.levels[l].a,
                    &h.levels[l].diag_inv,
                    self.cfg.adaptive_samples,
                    self.cfg.adaptive_smooth_steps,
                    omega,
                    0xC0FFEE,
                )?;
                if let Some(ref tp) = tp_opt {
                    let coarse_samples = restrict_samples_to_coarse(
                        &h.levels[l].a,
                        tp,
                        &samples,
                        self.cfg.adaptive_weight_mode,
                    );
                    adaptive_fit_values_only(
                        &pr,
                        &pc,
                        &mut p_new_vals,
                        tp,
                        &samples,
                        &coarse_samples,
                        self.cfg.adaptive_lambda,
                        self.cfg.adaptive_enforce_sum1,
                        self.cfg.interpolation_truncation,
                    )?;
                }
            }
            let mut p_tmp = Pcsr {
                m: h.levels[l].p.nrows(),
                n: h.levels[l].p.ncols(),
                row_ptr: pr,
                col_idx: pc,
                vals: p_new_vals,
            };
            let mut rank_diag = RankDiagnostics::default();
            let check_rank = self.cfg.verify_p_rank && tp_opt.is_some();
            if check_rank {
                let p_view = CsrMatrix::from_csr(
                    p_tmp.m,
                    p_tmp.n,
                    p_tmp.row_ptr.clone(),
                    p_tmp.col_idx.clone(),
                    p_tmp.vals.clone(),
                );
                rank_diag = check_p_rank_fast(&p_view, &self.cfg)?;
                if rank_diag.suspect {
                    let mut cond_report = rank_diag.cond_estimate;
                    match self.cfg.on_rank_failure {
                        RankFallback::RetryLooserInterp => {
                            let tp = tp_opt.as_ref().ok_or_else(|| {
                                KError::InvalidInput(
                                    "AMG: RetryLooserInterp requires SA interpolation".into(),
                                )
                            })?;
                            match try_fix_rank(
                                l,
                                &h.levels[l].a,
                                &h.levels[l].diag_inv,
                                tp,
                                &ctx,
                                &mut p_tmp,
                                &mut self.cfg,
                            )? {
                                RankFixOutcome::Fixed => {
                                    let p_view = CsrMatrix::from_csr(
                                        p_tmp.m,
                                        p_tmp.n,
                                        p_tmp.row_ptr.clone(),
                                        p_tmp.col_idx.clone(),
                                        p_tmp.vals.clone(),
                                    );
                                    rank_diag = check_p_rank_fast(&p_view, &self.cfg)?;
                                    cond_report = rank_diag.cond_estimate;
                                    if rank_diag.suspect {
                                        return Err(KError::InvalidInput(format!(
                                            "AMG: P rank suspect at level {l}, cond≈{cond_report:.3e}"
                                        )));
                                    }
                                }
                                RankFixOutcome::Unfixed => {
                                    return Err(KError::InvalidInput(format!(
                                        "AMG: P rank suspect at level {l}, cond≈{cond_report:.3e}"
                                    )));
                                }
                            }
                        }
                        RankFallback::Abort => {
                            return Err(KError::InvalidInput(format!(
                                "AMG: P rank suspect at level {l}, cond≈{cond_report:.3e}"
                            )));
                        }
                        other => {
                            return Err(KError::InvalidInput(format!(
                                "AMG: rank fallback {other:?} not implemented at level {l}"
                            )));
                        }
                    }
                }
            }
            h.levels[l].p.values_mut().copy_from_slice(&p_tmp.vals);
            // Update R values from P via precomputed transpose mapping
            if self.cfg.keep_transpose {
                let pvals = h.levels[l].p.values().to_vec();
                let p2r = h.levels[l].p2r_pos.clone();
                let rvalsm = h.levels[l].r.values_mut();
                for (pi, &ri) in p2r.iter().enumerate() {
                    rvalsm[ri] = pvals[pi];
                }
                if cfg!(debug_assertions) {
                    let step = (pvals.len() / 7).max(1);
                    for s in (0..pvals.len()).step_by(step) {
                        let ri = p2r[s];
                        let dv = (pvals[s] - rvalsm[ri]).abs();
                        debug_assert!(dv <= 1e-12, "R != P^T at sample {s}");
                    }
                }
            }
            let mut galerkin_worst = 0.0;
            let has_ng = h.levels[l].a_next_pat_ng.is_some();
            let allow_galerkin = self.cfg.verify_galerkin
                && self.cfg.filter_omega <= 0.0
                && !has_ng
                && self.cfg.galerkin_samples > 0;
            // Recompute A_{l+1} values by RAP numeric using fixed pattern
            if let Some(pat_ref) = h.levels[l].a_next_pat.as_ref() {
                let pat = pat_ref.clone();
                let nnz = pat.col_idx.len();
                let r_tmp_storage = if self.cfg.keep_transpose {
                    None
                } else {
                    Some(build_r_from_p(&mut h.levels[l]))
                };
                let r_for_ops = r_tmp_storage.as_ref().unwrap_or(&h.levels[l].r);
                let trials_next = if let Some(ref trials) = trials_current {
                    let mut next = Mat::<f64>::zeros(r_for_ops.nrows(), trials.ncols());
                    restrict_trials(r_for_ops, trials.as_ref(), next.as_mut())?;
                    Some(next)
                } else {
                    None
                };
                let mut vals = vec![0.0; nnz];
                rap_numeric(&pat, r_for_ops, &h.levels[l].a, &h.levels[l].p, &mut vals);
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
                let block_size_next = h.levels[l].num_functions.max(1);
                let mut a_full = CsrMatrix::from_csr(
                    pat.nrows,
                    pat.ncols,
                    pat.row_ptr.clone(),
                    pat.col_idx.clone(),
                    vals,
                );
                let use_ng = has_ng;
                if !use_ng || !self.cfg.filter_after_non_galerkin {
                    apply_trial_compensation(
                        &self.cfg,
                        &mut a_full,
                        trials_next.as_ref(),
                        block_size_next,
                    )?;
                }
                let mut a_coarse = if let (Some(ng_pat), Some(map)) =
                    (&h.levels[l].a_next_pat_ng, &h.levels[l].rap_full2ng_pos)
                {
                    let mut vals_ng = vec![0.0; ng_pat.col_idx.len()];
                    let full_vals = a_full.values();
                    for (k_full, &maybe) in map.iter().enumerate() {
                        if let Some(k_ng) = maybe {
                            vals_ng[k_ng] += full_vals[k_full];
                        }
                    }
                    let mut a_ng = CsrMatrix::from_csr(
                        ng_pat.nrows,
                        ng_pat.ncols,
                        ng_pat.row_ptr.clone(),
                        ng_pat.col_idx.clone(),
                        vals_ng,
                    );
                    if self.cfg.filter_after_non_galerkin {
                        apply_trial_compensation(
                            &self.cfg,
                            &mut a_ng,
                            trials_next.as_ref(),
                            block_size_next,
                        )?;
                    }
                    a_ng
                } else {
                    a_full
                };
                if allow_galerkin {
                    let (ok, worst) = galerkin_sample_check(
                        &h.levels[l].a,
                        &h.levels[l].p,
                        r_for_ops,
                        &a_coarse,
                        self.cfg.galerkin_samples,
                        self.cfg.galerkin_rel_tol,
                        0xBEEF,
                    )?;
                    galerkin_worst = worst;
                    if !ok {
                        let a_fix = rap(r_for_ops, &h.levels[l].a, &h.levels[l].p)?;
                        let (ok2, worst2) = galerkin_sample_check(
                            &h.levels[l].a,
                            &h.levels[l].p,
                            r_for_ops,
                            &a_fix,
                            self.cfg.galerkin_samples,
                            self.cfg.galerkin_rel_tol,
                            0xBEEF,
                        )?;
                        if ok2 {
                            galerkin_worst = worst2;
                            a_coarse = a_fix;
                        } else {
                            return Err(KError::InvalidInput(format!(
                                "AMG: Galerkin identity failed at level {l}: worst rel={worst:.3e} (retry={worst2:.3e})"
                            )));
                        }
                    }
                }
                h.levels[l + 1].a = a_coarse;
                h.levels[l + 1].diag_inv =
                    diag_inv_from_csr_cfg_fallback(&h.levels[l + 1].a, &self.cfg, allow_safeguard)?;
                #[cfg(debug_assertions)]
                debug_check_csr(&h.levels[l + 1].a, "coarse A");
                trials_current = trials_next;
            } else {
                // Safety fallback: full RAP (structure + values)
                let mut _r_tmp_owned: Option<CsrMatrix<f64>> = None;
                let r_used = if self.cfg.keep_transpose {
                    &h.levels[l].r
                } else {
                    _r_tmp_owned = Some(build_r_from_p(&mut h.levels[l]));
                    _r_tmp_owned.as_ref().unwrap()
                };
                let mut a_coarse = rap(r_used, &h.levels[l].a, &h.levels[l].p)?;
                if allow_galerkin {
                    let (ok, worst) = galerkin_sample_check(
                        &h.levels[l].a,
                        &h.levels[l].p,
                        r_used,
                        &a_coarse,
                        self.cfg.galerkin_samples,
                        self.cfg.galerkin_rel_tol,
                        0xBEEF,
                    )?;
                    galerkin_worst = worst;
                    if !ok {
                        let a_fix = rap(r_used, &h.levels[l].a, &h.levels[l].p)?;
                        let (ok2, worst2) = galerkin_sample_check(
                            &h.levels[l].a,
                            &h.levels[l].p,
                            r_used,
                            &a_fix,
                            self.cfg.galerkin_samples,
                            self.cfg.galerkin_rel_tol,
                            0xBEEF,
                        )?;
                        if ok2 {
                            galerkin_worst = worst2;
                            a_coarse = a_fix;
                        } else {
                            return Err(KError::InvalidInput(format!(
                                "AMG: Galerkin identity failed at level {l}: worst rel={worst:.3e} (retry={worst2:.3e})"
                            )));
                        }
                    }
                }
                h.levels[l + 1].diag_inv =
                    diag_inv_from_csr_cfg_fallback(&a_coarse, &self.cfg, allow_safeguard)?;
                h.levels[l + 1].a = a_coarse;
                trials_current = None;
            }
            diag_stats.push(AmgLevelStats {
                p_min_col_norm: rank_diag.min_col_norm,
                p_cond_sketched: rank_diag.cond_estimate,
                galerkin_worst_rel: galerkin_worst,
            });
            if l + 1 == h.coarsest_ix() && matches!(self.cfg.coarse_solve, CoarseSolve::ILU) {
                let levelc = &mut h.levels[l + 1];
                if let Some(m) = &levelc.coarse_solver {
                    m.lock().unwrap().setup(&levelc.a)?;
                } else {
                    let mut solver = CoarseIlu::new(
                        self.cfg.tolerance,
                        levelc.a.nrows().min(self.cfg.max_iterations.max(50)),
                        self.cfg.ilu_drop_tol,
                        self.cfg.ilu_fill_per_row,
                    );
                    solver.setup(&levelc.a)?;
                    levelc.coarse_solver = Some(Mutex::new(Box::new(solver)));
                }
            }
        }

        diag_stats.push(AmgLevelStats {
            p_min_col_norm: 0.0,
            p_cond_sketched: 0.0,
            galerkin_worst_rel: 0.0,
        });
        for lvl in 0..=h.coarsest_ix() {
            let recompute = self.cfg.chebyshev_recompute_esteig || h.levels[lvl].cheb.is_none();
            update_level_caches(
                &self.cfg,
                &mut h.levels[lvl],
                need_l1,
                need_cheb,
                need_safe_diag,
                need_cheb_safe,
                need_ilu0,
                need_ras,
                recompute,
            )?;
            if need_fsai {
                let level = &mut h.levels[lvl];
                if level.fsai.is_none() {
                    let strength_opt = if self.cfg.fsai_use_strength {
                        Some(Strength::from_csr(
                            &level.a,
                            self.cfg.strong_threshold,
                            self.cfg.normalize_strength,
                        ))
                    } else {
                        None
                    };
                    level.fsai = Some(fsai_build_for_level(
                        &self.cfg,
                        &level.a,
                        strength_opt.as_ref(),
                    )?);
                }
                if let Some(mut data) = level.fsai.take() {
                    fsai_refresh_numeric(&level.a, &mut data, self.cfg.fsai_lambda)?;
                    level.fsai = Some(data);
                }
                refresh_mixed_precision_shadows(&self.cfg, level);
            } else {
                h.levels[lvl].fsai = None;
                refresh_mixed_precision_shadows(&self.cfg, &mut h.levels[lvl]);
            }
        }

        #[cfg(feature = "simd")]
        {
            for level in &mut h.levels {
                build_level_spmv_plans(level, &spmv_tuning);
            }
        }

        if self.cfg.logging_level > 0 {
            let mut st = AmgStats::from_hierarchy(h);
            st.levels = collect_level_stats(h, &self.cfg);
            st.diagnostics = diag_stats;
            self.stats = Some(st);
        }
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

    fn jacobi_smooth_sparse_mp(
        omega: f32,
        level: &AMGLevel,
        rhs: &[f64],
        z: &mut [f64],
        iters: usize,
        ws: &mut AMGWorkspace,
        cfg: &AMGConfig,
    ) -> Result<(), KError> {
        if iters == 0 {
            return Ok(());
        }
        let n = level.a.nrows();
        if rhs.len() != n || z.len() != n {
            return Err(KError::InvalidInput("Jacobi: dimension mismatch".into()));
        }
        ws.ensure(n);
        ws.ensure_mixed(n);
        let mp_ws = ws
            .mp
            .as_mut()
            .expect("mixed workspace missing after ensure_mixed");
        mp_ws.temp32[..n]
            .iter_mut()
            .zip(z.iter())
            .for_each(|(dst, &src)| *dst = src as f32);
        mp_ws.residual32[..n]
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(dst, &src)| *dst = src as f32);
        let mut diag_owned = Vec::new();
        let diag_slice: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level
                .diag_inv_f32
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("Jacobi mixed cache missing".into()))?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_diag(n);
                buf.iter_mut()
                    .zip(level.diag_inv.iter())
                    .for_each(|(d, &s)| *d = s as f32);
                diag_owned.extend_from_slice(&buf[..n]);
                &diag_owned
            }
        };
        let mut vals_owned = Vec::new();
        let vals32: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level
                .a_vals_f32
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("Jacobi mixed matrix cache missing".into()))?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_vals(level.a.nnz());
                buf.iter_mut()
                    .zip(level.a.values().iter())
                    .for_each(|(d, &s)| *d = s as f32);
                vals_owned.extend_from_slice(buf.as_slice());
                &vals_owned
            }
        };
        let row_ptr = level.a.row_ptr();
        let col_idx = level.a.col_idx();
        for _ in 0..iters {
            spmv_scaled_f32_on_pattern(
                n,
                row_ptr,
                col_idx,
                vals32,
                1.0,
                &mp_ws.temp32[..n],
                0.0,
                &mut mp_ws.work32[..n],
            );
            for i in 0..n {
                mp_ws.temp32[i] += omega * diag_slice[i] * (mp_ws.residual32[i] - mp_ws.work32[i]);
            }
        }
        z.iter_mut()
            .zip(mp_ws.temp32[..n].iter())
            .for_each(|(dst, &src)| *dst = src as f64);
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

    fn l1_jacobi_mp(
        omega: f32,
        level: &AMGLevel,
        rhs: &[f64],
        z: &mut [f64],
        iters: usize,
        ws: &mut AMGWorkspace,
        cfg: &AMGConfig,
    ) -> Result<(), KError> {
        if iters == 0 {
            return Ok(());
        }
        let n = level.a.nrows();
        if rhs.len() != n || z.len() != n {
            return Err(KError::InvalidInput("L1Jacobi: dimension mismatch".into()));
        }
        ws.ensure(n);
        ws.ensure_mixed(n);
        let mp_ws = ws.mp.as_mut().expect("mixed workspace missing");
        mp_ws.temp32[..n]
            .iter_mut()
            .zip(z.iter())
            .for_each(|(dst, &src)| *dst = src as f32);
        mp_ws.residual32[..n]
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(dst, &src)| *dst = src as f32);
        let mut l1_owned = Vec::new();
        let l1_slice: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level
                .l1_inv_f32
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("L1Jacobi mixed cache missing".into()))?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_l1(n);
                buf.iter_mut()
                    .zip(
                        level
                            .l1_inv
                            .as_ref()
                            .ok_or_else(|| KError::InvalidInput("L1Jacobi cache missing".into()))?
                            .iter(),
                    )
                    .for_each(|(d, &s)| *d = s as f32);
                l1_owned.extend_from_slice(&buf[..n]);
                &l1_owned
            }
        };
        let mut vals_owned = Vec::new();
        let vals32: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level.a_vals_f32.as_ref().ok_or_else(|| {
                KError::InvalidInput("L1Jacobi mixed matrix cache missing".into())
            })?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_vals(level.a.nnz());
                buf.iter_mut()
                    .zip(level.a.values().iter())
                    .for_each(|(d, &s)| *d = s as f32);
                vals_owned.extend_from_slice(buf.as_slice());
                &vals_owned
            }
        };
        let row_ptr = level.a.row_ptr();
        let col_idx = level.a.col_idx();
        for _ in 0..iters {
            spmv_scaled_f32_on_pattern(
                n,
                row_ptr,
                col_idx,
                vals32,
                1.0,
                &mp_ws.temp32[..n],
                0.0,
                &mut mp_ws.work32[..n],
            );
            for i in 0..n {
                mp_ws.temp32[i] += omega * l1_slice[i] * (mp_ws.residual32[i] - mp_ws.work32[i]);
            }
        }
        z.iter_mut()
            .zip(mp_ws.temp32[..n].iter())
            .for_each(|(dst, &src)| *dst = src as f64);
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

    #[cfg(not(feature = "complex"))]
    fn ilu0_smooth(
        omega: f64,
        level: &AMGLevel,
        r: &[f64],
        z: &mut [f64],
        sweeps: usize,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if sweeps == 0 {
            return Ok(());
        }
        let ilu = level
            .ilu0
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("ILU0 cache missing".into()))?;
        let n = level.a.nrows();
        ws.ensure(n);
        for _ in 0..sweeps {
            level.a.spmv_scaled(1.0, z, 0.0, &mut ws.work[..n])?;
            for i in 0..n {
                ws.residual[i] = r[i] - ws.work[i];
            }
            ws.temp[..n].fill(R::default());
            ilu.lock().expect("ILU0 mutex poisoned").apply(
                PcSide::Left,
                &ws.residual[..n],
                &mut ws.temp[..n],
            )?;
            for i in 0..n {
                z[i] += omega * ws.temp[i];
            }
        }
        Ok(())
    }

    #[cfg(feature = "complex")]
    fn ilu0_smooth(
        _omega: f64,
        _level: &AMGLevel,
        _r: &[f64],
        _z: &mut [f64],
        _sweeps: usize,
        _ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        Err(KError::Unsupported(
            "AMG ILU0 smoother requires real scalars".into(),
        ))
    }

    #[cfg(not(feature = "complex"))]
    fn ras_smooth(
        omega: f64,
        level: &AMGLevel,
        r: &[f64],
        z: &mut [f64],
        sweeps: usize,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if sweeps == 0 {
            return Ok(());
        }
        let ras = level
            .ras
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("RAS cache missing".into()))?;
        let n = level.a.nrows();
        ws.ensure(n);
        for _ in 0..sweeps {
            level.a.spmv_scaled(1.0, z, 0.0, &mut ws.work[..n])?;
            for i in 0..n {
                ws.residual[i] = r[i] - ws.work[i];
            }
            ws.temp[..n].fill(R::default());
            ras.lock().expect("RAS mutex poisoned").apply(
                PcSide::Left,
                &ws.residual[..n],
                &mut ws.temp[..n],
            )?;
            for i in 0..n {
                z[i] += omega * ws.temp[i];
            }
        }
        Ok(())
    }

    #[cfg(feature = "complex")]
    fn ras_smooth(
        _omega: f64,
        _level: &AMGLevel,
        _r: &[f64],
        _z: &mut [f64],
        _sweeps: usize,
        _ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        Err(KError::Unsupported(
            "AMG RAS smoother requires real scalars".into(),
        ))
    }

    fn apply_chebyshev(
        a: &CsrMatrix<f64>,
        d_inv: &[f64],
        rhs: &[f64],
        z: &mut [f64],
        degree: usize,
        data: &ChebData,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if degree == 0 {
            return Ok(());
        }
        let n = a.nrows();
        ws.ensure(n);
        let bounds = ChebBounds {
            lam_max: data.lambda_max,
            lam_min: data.lambda_min,
        };
        chebyshev::chebyshev_smooth_csr(
            a,
            d_inv,
            rhs,
            z,
            degree,
            &bounds,
            &mut ws.residual[..n],
            &mut ws.temp[..n],
            &mut ws.work[..n],
        )
    }

    fn fsai_smooth_core(
        g: &CsrMatrix<f64>,
        gt: &CsrMatrix<f64>,
        a: &CsrMatrix<f64>,
        rhs: &[f64],
        z: &mut [f64],
        tau: f64,
        residual: &mut [f64],
        work: &mut [f64],
        tmp: &mut [f64],
    ) -> Result<(), KError> {
        let n = a.nrows();
        a.spmv_scaled(1.0, z, 0.0, work)?;
        for i in 0..n {
            residual[i] = rhs[i] - work[i];
        }
        gt.spmv_scaled(1.0, residual, 0.0, tmp)?;
        g.spmv_scaled(1.0, tmp, 0.0, work)?;
        for i in 0..n {
            z[i] += tau * work[i];
        }
        Ok(())
    }

    fn chebyshev_smooth_csr_mp(
        level: &AMGLevel,
        rhs: &[f64],
        z: &mut [f64],
        degree: usize,
        data: &ChebData,
        ws: &mut AMGWorkspace,
        cfg: &AMGConfig,
    ) -> Result<(), KError> {
        if degree == 0 {
            return Ok(());
        }
        let n = level.a.nrows();
        if rhs.len() != n || z.len() != n {
            return Err(KError::InvalidInput("Chebyshev: dimension mismatch".into()));
        }
        ws.ensure(n);
        ws.ensure_mixed(n);
        let mp_ws = ws.mp.as_mut().expect("mixed workspace missing");
        mp_ws.temp32[..n]
            .iter_mut()
            .zip(z.iter())
            .for_each(|(dst, &src)| *dst = src as f32);
        mp_ws.residual32[..n]
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(dst, &src)| *dst = src as f32);

        let mut vals_owned = Vec::new();
        let vals32: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level.a_vals_f32.as_ref().ok_or_else(|| {
                KError::InvalidInput("Chebyshev mixed matrix cache missing".into())
            })?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_vals(level.a.nnz());
                buf.iter_mut()
                    .zip(level.a.values().iter())
                    .for_each(|(d, &s)| *d = s as f32);
                vals_owned.extend_from_slice(buf.as_slice());
                &vals_owned
            }
        };
        let mut diag_owned = Vec::new();
        let diag32: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level
                .diag_inv_f32
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("Chebyshev mixed diag cache missing".into()))?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_diag(n);
                buf.iter_mut()
                    .zip(level.diag_inv.iter())
                    .for_each(|(d, &s)| *d = s as f32);
                diag_owned.extend_from_slice(&buf[..n]);
                &diag_owned
            }
        };

        let row_ptr = level.a.row_ptr();
        let col_idx = level.a.col_idx();
        spmv_scaled_f32_on_pattern(
            n,
            row_ptr,
            col_idx,
            vals32,
            1.0,
            &mp_ws.temp32[..n],
            0.0,
            &mut mp_ws.work32[..n],
        );
        for i in 0..n {
            mp_ws.residual32[i] -= mp_ws.work32[i];
        }

        let theta = (0.5 * (data.lambda_max + data.lambda_min)).max(1e-12) as f32;
        let delta = (0.5 * (data.lambda_max - data.lambda_min)) as f32;
        let mut alpha = 1.0f32 / theta;

        for i in 0..n {
            mp_ws.fine_corr32[i] = diag32[i] * mp_ws.residual32[i];
        }
        for i in 0..n {
            mp_ws.temp32[i] += alpha * mp_ws.fine_corr32[i];
        }
        spmv_scaled_f32_on_pattern(
            n,
            row_ptr,
            col_idx,
            vals32,
            1.0,
            &mp_ws.fine_corr32[..n],
            0.0,
            &mut mp_ws.work32[..n],
        );
        for i in 0..n {
            mp_ws.residual32[i] -= alpha * mp_ws.work32[i];
        }

        for _ in 1..degree {
            for i in 0..n {
                mp_ws.fine_corr32[i] = diag32[i] * mp_ws.residual32[i];
            }
            let beta = 0.25f32 * delta * delta * alpha;
            alpha = 1.0f32 / (theta - beta);
            for i in 0..n {
                mp_ws.temp32[i] += alpha * mp_ws.fine_corr32[i];
            }
            spmv_scaled_f32_on_pattern(
                n,
                row_ptr,
                col_idx,
                vals32,
                1.0,
                &mp_ws.fine_corr32[..n],
                0.0,
                &mut mp_ws.work32[..n],
            );
            for i in 0..n {
                mp_ws.residual32[i] -= alpha * mp_ws.work32[i];
            }
        }

        z.iter_mut()
            .zip(mp_ws.temp32[..n].iter())
            .for_each(|(dst, &src)| *dst = src as f64);
        Ok(())
    }

    fn fsai_smooth(
        g: &CsrMatrix<f64>,
        gt: &CsrMatrix<f64>,
        a: &CsrMatrix<f64>,
        rhs: &[f64],
        z: &mut [f64],
        tau: f64,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        let n = a.nrows();
        ws.ensure(n);
        let residual = &mut ws.residual[..n];
        let work = &mut ws.work[..n];
        let tmp = &mut ws.fine_corr[..n];
        Self::fsai_smooth_core(g, gt, a, rhs, z, tau, residual, work, tmp)
    }

    fn fsai_smooth_mp(
        level: &AMGLevel,
        data: &FsaiData,
        rhs: &[f64],
        z: &mut [f64],
        tau: f32,
        ws: &mut AMGWorkspace,
        cfg: &AMGConfig,
    ) -> Result<(), KError> {
        let n = level.a.nrows();
        if rhs.len() != n || z.len() != n {
            return Err(KError::InvalidInput("FSAI: dimension mismatch".into()));
        }
        ws.ensure(n);
        ws.ensure_mixed(n);
        let mp_ws = ws.mp.as_mut().expect("mixed workspace missing");
        mp_ws.temp32[..n]
            .iter_mut()
            .zip(z.iter())
            .for_each(|(dst, &src)| *dst = src as f32);
        mp_ws.residual32[..n]
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(dst, &src)| *dst = src as f32);

        let mut a_vals_owned = Vec::new();
        let a_vals32: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level
                .a_vals_f32
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("FSAI mixed matrix cache missing".into()))?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_vals(level.a.nnz());
                buf.iter_mut()
                    .zip(level.a.values().iter())
                    .for_each(|(d, &s)| *d = s as f32);
                a_vals_owned.extend_from_slice(buf.as_slice());
                &a_vals_owned
            }
        };
        let row_ptr = level.a.row_ptr();
        let col_idx = level.a.col_idx();
        spmv_scaled_f32_on_pattern(
            n,
            row_ptr,
            col_idx,
            a_vals32,
            1.0,
            &mp_ws.temp32[..n],
            0.0,
            &mut mp_ws.work32[..n],
        );
        for i in 0..n {
            mp_ws.residual32[i] -= mp_ws.work32[i];
        }

        let mut g_vals_owned = Vec::new();
        let g_vals32: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level
                .fsai_g_vals_f32
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("FSAI mixed G cache missing".into()))?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_fsai_g(data.g.nnz());
                buf.iter_mut()
                    .zip(data.g.values().iter())
                    .for_each(|(d, &s)| *d = s as f32);
                g_vals_owned.extend_from_slice(buf.as_slice());
                &g_vals_owned
            }
        };
        spmv_scaled_f32_on_pattern(
            data.g.nrows(),
            data.g.row_ptr(),
            data.g.col_idx(),
            g_vals32,
            1.0,
            &mp_ws.residual32[..data.g.ncols()],
            0.0,
            &mut mp_ws.work32[..data.g.nrows()],
        );

        let mut gt_vals_owned = Vec::new();
        let gt_vals32: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level
                .fsai_gt_vals_f32
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("FSAI mixed Gt cache missing".into()))?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_fsai_gt(data.gt.nnz());
                buf.iter_mut()
                    .zip(data.gt.values().iter())
                    .for_each(|(d, &s)| *d = s as f32);
                gt_vals_owned.extend_from_slice(buf.as_slice());
                &gt_vals_owned
            }
        };
        spmv_scaled_f32_on_pattern(
            data.gt.nrows(),
            data.gt.row_ptr(),
            data.gt.col_idx(),
            gt_vals32,
            1.0,
            &mp_ws.work32[..data.gt.ncols()],
            0.0,
            &mut mp_ws.coarse_rhs32[..data.gt.nrows()],
        );
        for i in 0..n {
            mp_ws.temp32[i] += tau * mp_ws.coarse_rhs32[i];
        }
        z.iter_mut()
            .zip(mp_ws.temp32[..n].iter())
            .for_each(|(dst, &src)| *dst = src as f64);
        Ok(())
    }

    fn mixed_spmv(
        level: &AMGLevel,
        cfg: &AMGConfig,
        x: &[f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        let n = level.a.nrows();
        if x.len() != n {
            return Err(KError::InvalidInput(
                "mixed SpMV: dimension mismatch".into(),
            ));
        }
        ws.ensure_mixed(n);
        let mp_ws = ws.mp.as_mut().expect("mixed workspace missing");
        mp_ws.temp32[..n]
            .iter_mut()
            .zip(x.iter())
            .for_each(|(dst, &src)| *dst = src as f32);
        let mut vals_owned = Vec::new();
        let vals32: &[f32] = match cfg.mixed_storage {
            MixedStorage::Cached => level
                .a_vals_f32
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("mixed matrix cache missing".into()))?,
            MixedStorage::Transient => {
                let buf = mp_ws.ensure_vals(level.a.nnz());
                buf.iter_mut()
                    .zip(level.a.values().iter())
                    .for_each(|(d, &s)| *d = s as f32);
                vals_owned.extend_from_slice(buf.as_slice());
                &vals_owned
            }
        };
        spmv_scaled_f32_on_pattern(
            n,
            level.a.row_ptr(),
            level.a.col_idx(),
            vals32,
            1.0,
            &mp_ws.temp32[..n],
            0.0,
            &mut mp_ws.work32[..n],
        );
        Ok(())
    }

    fn flexible_relax_supported(relax: RelaxType) -> bool {
        matches!(
            relax,
            RelaxType::Jacobi
                | RelaxType::L1Jacobi
                | RelaxType::SymmetricGaussSeidel
                | RelaxType::Fsai
        )
    }

    fn apply_smoother_as_pc(
        &self,
        relax: RelaxType,
        lvl: &AMGLevel,
        sweeps: usize,
        omega: f64,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        let n = lvl.a.nrows();
        ws.ensure(n);
        let r = &ws.residual[..n];
        let z = &mut ws.fine_corr[..n];
        let work = &mut ws.work[..n];
        if sweeps == 0 {
            z.copy_from_slice(r);
            return Ok(());
        }
        match relax {
            RelaxType::Jacobi => {
                z.fill(R::zero());
                for _ in 0..sweeps {
                    lvl.a.spmv_scaled(1.0, &z[..n], 0.0, work)?;
                    for i in 0..n {
                        z[i] += omega * lvl.diag_inv[i] * (r[i] - work[i]);
                    }
                }
                Ok(())
            }
            RelaxType::L1Jacobi => {
                let l1 = lvl
                    .l1_inv
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("L1Jacobi cache missing".into()))?;
                if l1.len() != n {
                    return Err(KError::InvalidInput(
                        "L1Jacobi cache has incorrect length".into(),
                    ));
                }
                z.fill(R::zero());
                for _ in 0..sweeps {
                    lvl.a.spmv_scaled(1.0, &z[..n], 0.0, work)?;
                    for i in 0..n {
                        z[i] += omega * l1[i] * (r[i] - work[i]);
                    }
                }
                Ok(())
            }
            RelaxType::SymmetricGaussSeidel => {
                z.fill(R::zero());
                Self::sym_gs(omega, &lvl.a, &lvl.diag_inv, r, z, sweeps)?;
                Ok(())
            }
            RelaxType::Fsai => {
                let data = lvl
                    .fsai
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("FSAI cache missing".into()))?;
                ws.fine_corr[..n].fill(R::zero());
                for _ in 0..sweeps {
                    Self::fsai_smooth_core(
                        &data.g,
                        &data.gt,
                        &lvl.a,
                        r,
                        &mut ws.fine_corr[..n],
                        self.cfg.fsai_damping,
                        &mut ws.work[..n],
                        &mut ws.coarse_rhs[..n],
                        &mut ws.temp[..n],
                    )?;
                }
                Ok(())
            }
            other => Err(KError::InvalidInput(format!(
                "RelaxType {other:?} cannot be used as a flexible preconditioner"
            ))),
        }
    }

    fn fcg_presmooth(
        &self,
        lvl: &AMGLevel,
        rhs: &[f64],
        sol: &mut [f64],
        iters: usize,
        rtol: f64,
        pc_sweeps: usize,
        relax: RelaxType,
        omega: f64,
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if iters == 0 {
            return Ok(());
        }
        let n = lvl.a.nrows();
        if rhs.len() != n || sol.len() != n {
            return Err(KError::InvalidInput(
                "fcg_presmooth: dimension mismatch".into(),
            ));
        }
        ws.ensure(n);

        lvl.a.spmv_scaled(1.0, sol, 0.0, &mut ws.work[..n])?;
        for i in 0..n {
            ws.residual[i] = rhs[i] - ws.work[i];
        }

        self.apply_smoother_as_pc(relax, lvl, pc_sweeps, omega, ws)?;
        ws.temp[..n].copy_from_slice(&ws.fine_corr[..n]);
        let mut rho = dot(&ws.residual[..n], &ws.fine_corr[..n]);
        if !rho.is_finite() {
            return Err(KError::InvalidInput(
                "fcg_presmooth: non-finite rho encountered".into(),
            ));
        }
        let mut rnorm_sq = dot(&ws.residual[..n], &ws.residual[..n]);
        let mut tol = -1.0f64;
        if rtol > 0.0 {
            let base = rnorm_sq.sqrt().max(1e-30);
            tol = base * rtol.max(1e-15);
            if base <= tol {
                return Ok(());
            }
        }

        for _ in 0..iters {
            lvl.a
                .spmv_scaled(1.0, &ws.temp[..n], 0.0, &mut ws.work[..n])?;
            let p_ap = dot(&ws.temp[..n], &ws.work[..n]);
            if !p_ap.is_finite() || p_ap.abs() < 1e-300 {
                break;
            }

            let alpha = rho / p_ap;
            for i in 0..n {
                sol[i] += alpha * ws.temp[i];
                ws.residual[i] -= alpha * ws.work[i];
            }

            if tol > 0.0 {
                rnorm_sq = dot(&ws.residual[..n], &ws.residual[..n]);
                if rnorm_sq.sqrt() <= tol {
                    break;
                }
            }

            ws.coarse_rhs[..n].copy_from_slice(&ws.fine_corr[..n]);
            self.apply_smoother_as_pc(relax, lvl, pc_sweeps, omega, ws)?;
            let z = &ws.fine_corr[..n];
            let mut rz_diff = R::default();
            for i in 0..n {
                rz_diff += ws.residual[i] * (z[i] - ws.coarse_rhs[i]);
            }
            let beta = if rho.abs() > R::default() {
                rz_diff / rho
            } else {
                R::default()
            };
            for i in 0..n {
                ws.temp[i] = z[i] + beta * ws.temp[i];
            }
            let rho_new = dot(&ws.residual[..n], z);
            if !rho_new.is_finite() {
                break;
            }
            rho = rho_new;
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
        let use_mp_smooth = cfg
            .mixed_precision
            .map(|mp| mp.smoothers_enabled())
            .unwrap_or(false);
        #[cfg(test)]
        {
            RELAX_CALL_COUNTS.with(|counts| {
                let mut data = counts.get();
                data[phase.ix()] += 1;
                counts.set(data);
            });
        }
        let a = &lvl.a;
        match pol.kind[phase.ix()] {
            RelaxType::Jacobi => {
                if use_mp_smooth {
                    Self::jacobi_smooth_sparse_mp(pol.omega as f32, lvl, rhs, sol, k, ws, cfg)
                } else {
                    Self::jacobi_smooth_sparse(pol.omega, a, &lvl.diag_inv, rhs, sol, k, ws)
                }
            }
            RelaxType::GaussSeidel => {
                if matches!(where_, RelaxWhere::Pre) {
                    Self::gs_forward(1.0, a, &lvl.diag_inv, rhs, sol, k)
                } else {
                    Self::gs_backward(1.0, a, &lvl.diag_inv, rhs, sol, k)
                }
            }
            RelaxType::SafeguardedGaussSeidel => {
                let diag = lvl
                    .diag_inv_safe
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("Safeguarded GS cache missing".into()))?;
                if matches!(where_, RelaxWhere::Pre) {
                    Self::gs_forward(1.0, a, diag, rhs, sol, k)
                } else {
                    Self::gs_backward(1.0, a, diag, rhs, sol, k)
                }
            }
            RelaxType::GaussSeidelBackward => Self::gs_backward(1.0, a, &lvl.diag_inv, rhs, sol, k),
            RelaxType::SymmetricGaussSeidel => Self::sym_gs(1.0, a, &lvl.diag_inv, rhs, sol, k),
            RelaxType::L1Jacobi => {
                if let Some(ref l1) = lvl.l1_inv {
                    if use_mp_smooth {
                        Self::l1_jacobi_mp(pol.omega as f32, lvl, rhs, sol, k, ws, cfg)
                    } else {
                        Self::l1_jacobi(pol.omega, a, l1, rhs, sol, k, ws)
                    }
                } else {
                    Err(KError::InvalidInput("L1Jacobi cache missing".into()))
                }
            }
            RelaxType::Chebyshev => {
                let cheb = lvl
                    .cheb
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("Chebyshev cache missing".into()))?;
                let degree = cfg.chebyshev_degree.max(1);
                for _ in 0..k {
                    if use_mp_smooth {
                        Self::chebyshev_smooth_csr_mp(lvl, rhs, sol, degree, cheb, ws, cfg)?;
                    } else {
                        Self::apply_chebyshev(a, &lvl.diag_inv, rhs, sol, degree, cheb, ws)?;
                    }
                }
                Ok(())
            }
            RelaxType::ChebyshevSafe => {
                if use_mp_smooth {
                    return Err(KError::InvalidInput(
                        "ChebyshevSafe does not support mixed-precision smoothing".into(),
                    ));
                }
                let cheb = lvl
                    .cheb_safe
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("ChebyshevSafe cache missing".into()))?;
                let diag = lvl.diag_inv_safe.as_ref().ok_or_else(|| {
                    KError::InvalidInput("ChebyshevSafe diag cache missing".into())
                })?;
                let degree = cfg.chebyshev_degree.max(1);
                for _ in 0..k {
                    Self::apply_chebyshev(a, diag, rhs, sol, degree, cheb, ws)?;
                }
                Ok(())
            }
            RelaxType::Ilu0 => Self::ilu0_smooth(pol.omega, lvl, rhs, sol, k, ws),
            RelaxType::Ras => Self::ras_smooth(pol.omega, lvl, rhs, sol, k, ws),
            RelaxType::Fsai => {
                let data = lvl
                    .fsai
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("FSAI cache missing".into()))?;
                for _ in 0..k {
                    if use_mp_smooth {
                        Self::fsai_smooth_mp(
                            lvl,
                            data,
                            rhs,
                            sol,
                            cfg.fsai_damping as f32,
                            ws,
                            cfg,
                        )?;
                    } else {
                        Self::fsai_smooth(&data.g, &data.gt, a, rhs, sol, cfg.fsai_damping, ws)?;
                    }
                }
                Ok(())
            }
            other => Err(KError::InvalidInput(format!(
                "RelaxType {other:?} not yet supported"
            ))),
        }
    }

    // ---- Multigrid cycle ----------------------------------------------------

    fn smooth_dispatch(
        relax: RelaxType,
        sweeps: usize,
        lvl: &AMGLevel,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
        cfg: &AMGConfig,
    ) -> Result<(), KError> {
        if sweeps == 0 {
            return Ok(());
        }
        match relax {
            RelaxType::Jacobi => Self::jacobi_smooth_sparse(
                cfg.jacobi_omega,
                &lvl.a,
                &lvl.diag_inv,
                rhs,
                sol,
                sweeps,
                ws,
            ),
            RelaxType::GaussSeidel => {
                Self::gs_forward(1.0, &lvl.a, &lvl.diag_inv, rhs, sol, sweeps)
            }
            RelaxType::SafeguardedGaussSeidel => {
                let diag = lvl
                    .diag_inv_safe
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("Safeguarded GS cache missing".into()))?;
                Self::gs_forward(1.0, &lvl.a, diag, rhs, sol, sweeps)
            }
            RelaxType::GaussSeidelBackward => {
                Self::gs_backward(1.0, &lvl.a, &lvl.diag_inv, rhs, sol, sweeps)
            }
            RelaxType::SymmetricGaussSeidel => {
                Self::sym_gs(1.0, &lvl.a, &lvl.diag_inv, rhs, sol, sweeps)
            }
            RelaxType::L1Jacobi => {
                let l1 = lvl
                    .l1_inv
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("L1Jacobi cache missing".into()))?;
                Self::l1_jacobi(cfg.jacobi_omega, &lvl.a, l1, rhs, sol, sweeps, ws)
            }
            RelaxType::Chebyshev => {
                let cheb = lvl
                    .cheb
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("Chebyshev cache missing".into()))?;
                let degree = cfg.chebyshev_degree.max(1);
                for _ in 0..sweeps {
                    Self::apply_chebyshev(&lvl.a, &lvl.diag_inv, rhs, sol, degree, cheb, ws)?;
                }
                Ok(())
            }
            RelaxType::ChebyshevSafe => {
                let cheb = lvl
                    .cheb_safe
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("ChebyshevSafe cache missing".into()))?;
                let diag = lvl.diag_inv_safe.as_ref().ok_or_else(|| {
                    KError::InvalidInput("ChebyshevSafe diag cache missing".into())
                })?;
                let degree = cfg.chebyshev_degree.max(1);
                for _ in 0..sweeps {
                    Self::apply_chebyshev(&lvl.a, diag, rhs, sol, degree, cheb, ws)?;
                }
                Ok(())
            }
            RelaxType::Ilu0 => Self::ilu0_smooth(cfg.jacobi_omega, lvl, rhs, sol, sweeps, ws),
            RelaxType::Ras => Self::ras_smooth(cfg.jacobi_omega, lvl, rhs, sol, sweeps, ws),
            RelaxType::Fsai => {
                let data = lvl
                    .fsai
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("FSAI cache missing".into()))?;
                for _ in 0..sweeps {
                    Self::fsai_smooth(&data.g, &data.gt, &lvl.a, rhs, sol, cfg.fsai_damping, ws)?;
                }
                Ok(())
            }
            other => Err(KError::InvalidInput(format!(
                "RelaxType {other:?} not yet supported",
            ))),
        }
    }

    fn apply_precond_one_sweep(
        &self,
        level: usize,
        r: &[f64],
        out: &mut [f64],
        work: &mut [f64],
        temp: &mut [f64],
        residual: &mut [f64],
    ) -> Result<(), KError> {
        let h = match &self.state {
            AmgState::Ready { hierarchy, .. } => hierarchy,
            _ => return Err(KError::InvalidInput("AMG not set up".into())),
        };
        let lvl = h
            .levels
            .get(level)
            .ok_or_else(|| KError::InvalidInput("level out of range".into()))?;
        let n = lvl.a.nrows();
        if r.len() != n
            || out.len() != n
            || work.len() != n
            || temp.len() != n
            || residual.len() != n
        {
            return Err(KError::InvalidInput(
                "krylov preconditioner: level size mismatch".into(),
            ));
        }
        out.fill(R::default());

        match self.cfg.relax_type {
            RelaxType::Jacobi => {
                for i in 0..n {
                    out[i] = self.cfg.jacobi_omega * lvl.diag_inv[i] * r[i];
                }
                Ok(())
            }
            RelaxType::GaussSeidel => Self::gs_forward(1.0, &lvl.a, &lvl.diag_inv, r, out, 1),
            RelaxType::SafeguardedGaussSeidel => {
                let diag = lvl
                    .diag_inv_safe
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("Safeguarded GS cache missing".into()))?;
                Self::gs_forward(1.0, &lvl.a, diag, r, out, 1)
            }
            RelaxType::GaussSeidelBackward => {
                Self::gs_backward(1.0, &lvl.a, &lvl.diag_inv, r, out, 1)
            }
            RelaxType::SymmetricGaussSeidel => Self::sym_gs(1.0, &lvl.a, &lvl.diag_inv, r, out, 1),
            RelaxType::L1Jacobi => {
                let l1 = lvl
                    .l1_inv
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("L1Jacobi cache missing".into()))?;
                work.fill(R::default());
                lvl.a.spmv_scaled(1.0, out, 0.0, work)?;
                for i in 0..n {
                    out[i] += self.cfg.jacobi_omega * l1[i] * (r[i] - work[i]);
                }
                Ok(())
            }
            RelaxType::Chebyshev => {
                let cheb = lvl
                    .cheb
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("Chebyshev cache missing".into()))?;
                let bounds = ChebBounds {
                    lam_max: cheb.lambda_max,
                    lam_min: cheb.lambda_min,
                };
                chebyshev::chebyshev_smooth_csr(
                    &lvl.a,
                    &lvl.diag_inv,
                    r,
                    out,
                    self.cfg.chebyshev_degree.max(1),
                    &bounds,
                    residual,
                    temp,
                    work,
                )
            }
            RelaxType::ChebyshevSafe => {
                let cheb = lvl
                    .cheb_safe
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("ChebyshevSafe cache missing".into()))?;
                let diag = lvl.diag_inv_safe.as_ref().ok_or_else(|| {
                    KError::InvalidInput("ChebyshevSafe diag cache missing".into())
                })?;
                let bounds = ChebBounds {
                    lam_max: cheb.lambda_max,
                    lam_min: cheb.lambda_min,
                };
                chebyshev::chebyshev_smooth_csr(
                    &lvl.a,
                    diag,
                    r,
                    out,
                    self.cfg.chebyshev_degree.max(1),
                    &bounds,
                    residual,
                    temp,
                    work,
                )
            }
            #[cfg(not(feature = "complex"))]
            RelaxType::Ilu0 => {
                let ilu = lvl
                    .ilu0
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("ILU0 cache missing".into()))?;
                ilu.lock()
                    .expect("ILU0 mutex poisoned")
                    .apply(PcSide::Left, r, out)
            }
            #[cfg(feature = "complex")]
            RelaxType::Ilu0 => Err(KError::Unsupported(
                "AMG ILU0 smoother requires real scalars".into(),
            )),
            #[cfg(not(feature = "complex"))]
            RelaxType::Ras => {
                let ras = lvl
                    .ras
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("RAS cache missing".into()))?;
                ras.lock()
                    .expect("RAS mutex poisoned")
                    .apply(PcSide::Left, r, out)
            }
            #[cfg(feature = "complex")]
            RelaxType::Ras => Err(KError::Unsupported(
                "AMG RAS smoother requires real scalars".into(),
            )),
            RelaxType::Fsai => {
                let data = lvl
                    .fsai
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("FSAI cache missing".into()))?;
                residual.fill(R::default());
                work.fill(R::default());
                temp.fill(R::default());
                Self::fsai_smooth_core(
                    &data.g,
                    &data.gt,
                    &lvl.a,
                    r,
                    out,
                    self.cfg.fsai_damping,
                    residual,
                    work,
                    temp,
                )
            }
            other => Err(KError::InvalidInput(format!(
                "RelaxType {other:?} not yet supported",
            ))),
        }
    }
    fn krylov_smooth(
        &self,
        algo: KrylovAlgo,
        iters: usize,
        level: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if iters == 0 {
            return Ok(());
        }
        match algo {
            KrylovAlgo::FCG => self.krylov_smooth_pcg(iters, level, rhs, sol, ws),
        }
    }

    fn krylov_smooth_pcg(
        &self,
        iters: usize,
        level: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if iters == 0 {
            return Ok(());
        }
        let h = match &self.state {
            AmgState::Ready { hierarchy, .. } => hierarchy,
            _ => return Err(KError::InvalidInput("AMG not set up".into())),
        };
        let lvl = h
            .levels
            .get(level)
            .ok_or_else(|| KError::InvalidInput("level out of range".into()))?;
        let a = &lvl.a;
        let n = a.nrows();
        if rhs.len() != n || sol.len() != n {
            return Err(KError::InvalidInput(
                "krylov smoother: dimension mismatch".into(),
            ));
        }
        ws.ensure(n);

        a.spmv_scaled(1.0, sol, 0.0, &mut ws.work[..n])?;
        for i in 0..n {
            ws.residual[i] = rhs[i] - ws.work[i];
        }

        ws.k_residual[..n].copy_from_slice(&ws.residual[..n]);
        self.apply_precond_one_sweep(
            level,
            &ws.k_residual[..n],
            &mut ws.k_zeta[..n],
            &mut ws.k_work[..n],
            &mut ws.k_temp[..n],
            &mut ws.k_ap[..n],
        )?;
        ws.k_p[..n].copy_from_slice(&ws.k_zeta[..n]);

        let mut rz_old = dot(&ws.residual[..n], &ws.k_zeta[..n]);
        if !rz_old.is_finite() || rz_old.abs() < 1e-300 {
            return Ok(());
        }

        for _ in 0..iters {
            a.spmv_scaled(1.0, &ws.k_p[..n], 0.0, &mut ws.k_ap[..n])?;
            let denom = dot(&ws.k_p[..n], &ws.k_ap[..n]);
            if !denom.is_finite() || denom.abs() < 1e-300 {
                break;
            }
            let alpha = rz_old / denom;
            for i in 0..n {
                sol[i] += alpha * ws.k_p[i];
                ws.residual[i] -= alpha * ws.k_ap[i];
            }

            ws.k_residual[..n].copy_from_slice(&ws.residual[..n]);
            self.apply_precond_one_sweep(
                level,
                &ws.k_residual[..n],
                &mut ws.k_zeta[..n],
                &mut ws.k_work[..n],
                &mut ws.k_temp[..n],
                &mut ws.k_ap[..n],
            )?;
            let rz_new = dot(&ws.residual[..n], &ws.k_zeta[..n]);
            if !rz_new.is_finite() {
                break;
            }
            if rz_new.abs() < 1e-300 {
                break;
            }
            let beta = rz_new / rz_old;
            for i in 0..n {
                ws.k_p[i] = ws.k_zeta[i] + beta * ws.k_p[i];
            }
            rz_old = rz_new;
        }
        Ok(())
    }

    fn restrict_apply(
        lvl: &AMGLevel,
        fine_res: &[f64],
        coarse_rhs: &mut [f64],
    ) -> Result<(), KError> {
        if lvl.r_row_ptr.is_some() {
            lvl.p.spmv_transpose_scaled(1.0, fine_res, 0.0, coarse_rhs)
        } else {
            lvl.r.spmv_scaled(1.0, fine_res, 0.0, coarse_rhs)
        }
    }

    fn cycle_profiled(
        &self,
        level: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
        mut cyc: Option<&mut CycleTimings>,
    ) -> Result<(), KError> {
        let h = match &self.state {
            AmgState::Ready { hierarchy, .. } => hierarchy,
            _ => return Err(KError::InvalidInput("AMG not set up".into())),
        };
        let lc = h.coarsest_ix();

        let a = &h.levels[level].a;
        let pol = &h.policy;
        let cycle_pol = &*self.cycle_policy;
        let mut lv = CycleLevelTiming {
            level,
            ..Default::default()
        };
        let prof = cyc.is_some();

        if level == lc {
            with_timing(prof, &mut lv.coarse_solve, || {
                let n = a.nrows();
                if matches!(h.coarse_solve, CoarseSolve::Smoother) {
                    ws.ensure(n);
                    return Self::apply_relax(
                        pol,
                        RelaxPhase::Coarsest,
                        RelaxWhere::Pre,
                        &h.levels[level],
                        rhs,
                        sol,
                        ws,
                        &self.cfg,
                    );
                }
                let prefer_dense = matches!(h.coarse_solve, CoarseSolve::DirectDense)
                    || n <= self.cfg.max_coarse_size;
                if prefer_dense {
                    let mut solver = CoarseDenseLu::new();
                    solver.setup(a)?;
                    solver.solve(rhs, sol)
                } else {
                    match h.coarse_solve {
                        CoarseSolve::CG => cg_sparse(
                            a,
                            rhs,
                            sol,
                            self.cfg.tolerance,
                            n.min(self.cfg.max_iterations.max(50)),
                        ),
                        CoarseSolve::ILU => {
                            let levelc = &h.levels[level];
                            if let Some(m) = &levelc.coarse_solver {
                                let mut guard = m.lock().expect("coarse solver mutex poisoned");
                                guard.solve(rhs, sol)
                            } else {
                                let mut solver = CoarseIlu::new(
                                    self.cfg.tolerance,
                                    n.min(self.cfg.max_iterations.max(50)),
                                    self.cfg.ilu_drop_tol,
                                    self.cfg.ilu_fill_per_row,
                                );
                                solver.setup(a)?;
                                solver.solve(rhs, sol)
                            }
                        }
                        CoarseSolve::DirectDense => unreachable!(),
                        CoarseSolve::Smoother => unreachable!(),
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
        let use_mp_residual = self
            .cfg
            .mixed_precision
            .map(|mp| mp.residual_enabled())
            .unwrap_or(false);

        // Pre-smooth
        let phase_pre = if level == 0 {
            RelaxPhase::Fine
        } else {
            RelaxPhase::Down
        };
        with_timing(prof, &mut lv.pre_smooth, || {
            let use_flexible =
                self.cfg.flexible_level == Some(level) && self.cfg.flexible_iters > 0;
            if use_flexible {
                let relax = pol.kind[phase_pre.ix()];
                if !Self::flexible_relax_supported(relax) {
                    return Err(KError::InvalidInput(format!(
                        "RelaxType {relax:?} cannot be used for flexible presmoothing",
                    )));
                }
                self.fcg_presmooth(
                    &h.levels[level],
                    rhs,
                    sol,
                    self.cfg.flexible_iters,
                    self.cfg.flexible_rtol,
                    self.cfg.flexible_pc_sweeps,
                    relax,
                    pol.omega,
                    ws,
                )
            } else {
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
            }
        })?;

        // residual = rhs - A * sol
        if use_mp_residual {
            with_timing(prof, &mut lv.matvec, || {
                Self::mixed_spmv(&h.levels[level], &self.cfg, sol, ws)
            })?;
            with_timing(prof, &mut lv.residual_axpy, || {
                let mp_ws = ws
                    .mp
                    .as_ref()
                    .expect("mixed workspace missing after mixed_spmv");
                #[cfg(feature = "rayon")]
                {
                    for i in 0..n {
                        ws.work[i] = mp_ws.work32[i] as f64;
                    }
                    ws.residual[..n]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, ri)| {
                            *ri = rhs[i] - ws.work[i];
                        });
                }
                #[cfg(not(feature = "rayon"))]
                for i in 0..n {
                    let az = mp_ws.work32[i] as f64;
                    ws.work[i] = az;
                    ws.residual[i] = rhs[i] - az;
                }
            });
        } else {
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
        }

        // r_c = R * residual
        let p = &h.levels[level].p;
        let nc = h.levels[level + 1].a.nrows();

        let mut local_coarse = std::mem::take(&mut ws.coarse_rhs);
        local_coarse.resize(nc, R::zero());
        with_timing(prof, &mut lv.restrict, || {
            Self::restrict_apply(&h.levels[level], &ws.residual[..n], &mut local_coarse[..nc])
        })?;

        let gamma = cycle_pol.gamma_visits(level).max(1);
        for t in 0..gamma {
            let mut zc = vec![R::zero(); nc];
            if level + 1 == lc {
                with_timing(prof, &mut lv.coarse_solve, || {
                    let mut solver = CoarseDenseLu::new();
                    solver.setup(&h.levels[level + 1].a)?;
                    solver.solve(&local_coarse[..nc], &mut zc)
                })?;
            } else {
                self.cycle_profiled(
                    level + 1,
                    &local_coarse[..nc],
                    &mut zc,
                    ws,
                    cyc.as_deref_mut(),
                )?;
            }
            with_timing(prof, &mut lv.prolong, || {
                ws.fine_corr[..n].fill(R::zero());
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
                    Self::restrict_apply(
                        &h.levels[level],
                        &ws.residual[..n],
                        &mut local_coarse[..nc],
                    )
                })?;
            }
        }
        ws.coarse_rhs = local_coarse;

        if let Some((algo, iters)) = cycle_pol.k_presmooth(level) {
            with_timing(prof, &mut lv.post_smooth, || {
                self.krylov_smooth(algo, iters, level, rhs, sol, ws)
            })?;
        }

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

        if let Some((algo, iters)) = cycle_pol.k_postsmooth(level) {
            with_timing(prof, &mut lv.post_smooth, || {
                self.krylov_smooth(algo, iters, level, rhs, sol, ws)
            })?;
        }

        if let Some(c) = cyc {
            c.per_level.push(lv);
        }
        Ok(())
    }

    fn cycle(
        &self,
        level: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        self.cycle_profiled(level, rhs, sol, ws, None)
    }

    #[inline]
    fn v_cycle(
        &self,
        level: usize,
        rhs: &[f64],
        sol: &mut [f64],
        ws: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        self.cycle(level, rhs, sol, ws)
    }

    pub fn fmg_solve(&self, _b: &[f64], _x: &mut [f64]) -> Result<(), KError> {
        Err(KError::NotImplemented(
            "FMG solve not yet implemented".into(),
        ))
    }

    pub fn cascade_solve(&self, _b: &[f64], _x: &mut [f64]) -> Result<(), KError> {
        Err(KError::NotImplemented(
            "Cascade solve not yet implemented".into(),
        ))
    }

    // Convenience to avoid trait ambiguity in examples
    #[cfg(not(feature = "complex"))]
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

    pub fn dist_apply_stats(&self) -> Option<DistApplyStats> {
        if let Ok(rt) = self.runtime.lock() {
            rt.last_dist_apply.clone()
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) fn debug_levels_r_equals_pt(&self) -> bool {
        let h = match &self.state {
            AmgState::Ready { hierarchy, .. } => hierarchy,
            _ => return true,
        };
        if !self.cfg.keep_transpose {
            return true;
        }
        for lvl in 0..h.coarsest_ix() {
            let pvals = h.levels[lvl].p.values();
            let rvals = h.levels[lvl].r.values();
            let map = &h.levels[lvl].p2r_pos;
            if pvals.len() != map.len() || rvals.len() < map.len() {
                return false;
            }
            for (pi, &ri) in map.iter().enumerate() {
                if ri >= rvals.len() {
                    return false;
                }
                if (pvals[pi] - rvals[ri]).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    #[cfg(all(debug_assertions, not(feature = "complex")))]
    fn spd_probe(&self) -> Result<(), KError> {
        if !self.cfg.require_spd {
            return Ok(());
        }
        let h = match &self.state {
            AmgState::Ready { hierarchy, .. } => hierarchy,
            _ => return Err(KError::InvalidInput("AMG not set up".into())),
        };
        let n = h.finest().a.nrows();
        if n == 0 {
            return Ok(());
        }
        let mut x = vec![R::default(); n];
        let mut y = vec![R::default(); n];
        for t in 0..3 {
            for i in 0..n {
                x[i] = ((i + 7919 * t) % 127) as f64 - 63.0;
            }
            y.fill(R::default());
            self.apply(PcSide::Left, &x, &mut y)?;
            let qf = x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>();
            let x_norm2 = x.iter().map(|v| v * v).sum::<f64>();
            let tol = 1e-12_f64.max(1e-10 * x_norm2.abs());
            debug_assert!(
                qf.is_finite() && qf > -tol,
                "Preconditioned operator is not SPD: qf={qf}, tol={tol}"
            );
        }
        Ok(())
    }
}

// ===== Preconditioner trait (new API) =======================================

#[cfg(not(feature = "complex"))]
impl Preconditioner for AMG {
    fn dims(&self) -> (usize, usize) {
        if let Some(dist) = &self.dist {
            let n = dist.local_nrows();
            return (n, n);
        }
        if let AmgState::Ready { hierarchy, .. } = &self.state {
            let n = hierarchy.finest().a.nrows();
            (n, n)
        } else if let Some(csr) = self.csr.as_ref() {
            (csr.nrows(), csr.ncols())
        } else {
            (0, 0)
        }
    }

    fn required_format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Csr
    }

    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.cfg.validate()?;
        if let Some(dist) = op.as_any().downcast_ref::<DistCsrOp>() {
            if op.comm().size() > 1 {
                return self.setup_dist(dist);
            }
        }
        self.dist = None;
        let csr = csr_from_linop(op, self.cfg.drop_tol)?;
        let csr = if self.cfg.conditioning.is_active() {
            let mut local = (*csr).clone();
            apply_csr_transforms("AMG", &mut local, &self.cfg.conditioning)?;
            Arc::new(local)
        } else {
            csr
        };
        let csr_ref = csr.as_ref();
        #[cfg(debug_assertions)]
        debug_check_csr(csr_ref, "setup csr");
        let sid = op.structure_id();
        let vid = op.values_id();
        let pattern_hash = csr_pattern_hash(csr_ref);

        self.ensure_symbolic_structure(csr_ref, sid, pattern_hash)?;

        let need_numeric = match &self.state {
            AmgState::Ready { last_values_id, .. } => *last_values_id != vid,
            AmgState::SymbolicOnly { .. } => true,
            AmgState::Uninitialized => true,
        };

        if need_numeric {
            self.refresh_numeric_ready(csr_ref, sid, vid, pattern_hash)?;
        }

        self.csr = Some(csr.clone());
        if self.cfg.logging_level >= 2
            && self.cfg.print_level >= 1
            && let Some(s) = self.stats.as_ref()
        {
            print_setup_tables(s);
        }
        #[cfg(debug_assertions)]
        if self.cfg.require_spd {
            self.spd_probe()?;
        }
        Ok(())
    }

    fn apply(&self, side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        if let Some(dist) = &self.dist {
            if dist.comm.size() > 1 {
                return self.apply_dist(side, r, z, dist);
            }
        }
        self.apply_local(side, r, z)
    }

    fn capabilities(&self) -> PcCaps {
        let mut caps = PcCaps::default();
        if self.cfg.require_spd {
            caps.is_spd = true;
            caps.side_restriction = Some(PcSide::Left);
        }
        caps
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.cfg.validate()?;
        if let Some(dist) = op.as_any().downcast_ref::<DistCsrOp>() {
            if op.comm().size() > 1 {
                return self.setup_dist(dist);
            }
        }
        self.dist = None;
        let csr = csr_from_linop(op, self.cfg.drop_tol)?;
        let sid = op.structure_id();
        let vid = op.values_id();
        let pattern_hash = csr_pattern_hash(&csr);
        self.ensure_symbolic_structure(csr.as_ref(), sid, pattern_hash)?;
        self.refresh_numeric_ready(csr.as_ref(), sid, vid, pattern_hash)?;
        self.csr = Some(csr.clone());
        if self.cfg.logging_level >= 2
            && self.cfg.print_level >= 1
            && let Some(s) = self.stats.as_ref()
        {
            print_setup_tables(s);
        }
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        self.cfg.validate()?;
        if let Some(dist) = op.as_any().downcast_ref::<DistCsrOp>() {
            if op.comm().size() > 1 {
                return self.setup_dist(dist);
            }
        }
        self.dist = None;
        let csr = csr_from_linop(op, self.cfg.drop_tol)?;
        let sid = op.structure_id();
        let vid = op.values_id();
        let pattern_hash = csr_pattern_hash(&csr);
        let hierarchy = self.build_symbolic(csr.as_ref())?;
        self.state = AmgState::SymbolicOnly {
            hierarchy,
            last_structure_id: sid,
            pattern_hash,
        };
        self.refresh_numeric_ready(csr.as_ref(), sid, vid, pattern_hash)?;
        self.csr = Some(csr.clone());
        if self.cfg.logging_level >= 2
            && self.cfg.print_level >= 1
            && let Some(s) = self.stats.as_ref()
        {
            print_setup_tables(s);
        }
        #[cfg(debug_assertions)]
        if self.cfg.require_spd {
            self.spd_probe()?;
        }
        Ok(())
    }

    fn distributed_support(&self) -> crate::preconditioner::PcDistributedSupport {
        if let Some(dist) = &self.dist {
            if dist.comm.size() > 1 {
                return crate::preconditioner::PcDistributedSupport::Distributed;
            }
        }
        crate::preconditioner::PcDistributedSupport::LocalOnly
    }
}

#[cfg(feature = "complex")]
impl Preconditioner for AMG {
    fn setup(&mut self, _op: &dyn LinOp<S = S>) -> Result<(), KError> {
        Err(KError::Unsupported(
            "AMG does not support complex scalars yet".into(),
        ))
    }

    fn apply(&self, _side: PcSide, _r: &[S], _z: &mut [S]) -> Result<(), KError> {
        Err(KError::Unsupported(
            "AMG does not support complex scalars yet".into(),
        ))
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for AMG {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        <Self as Preconditioner>::dims(self)
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

// ===== Legacy adapter (unchanged external signature) ========================

#[cfg(not(feature = "complex"))]
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
    cfg: &mut AMGConfig,
) -> Result<(AmgHierarchy, Option<AmgStats>), KError> {
    let mut levels: Vec<AMGLevel> = Vec::with_capacity(cfg.max_levels);
    let mut a_cur = fine.clone();
    let do_stats = cfg.logging_level > 0;
    let mut level_stats: Vec<LevelStats> = Vec::new();
    let mut timings: Vec<LevelSetupTiming> = Vec::new();
    let mut diag_stats: Vec<AmgLevelStats> = Vec::new();
    let t_setup_all = if do_stats { Some(tic()) } else { None };
    #[cfg(feature = "simd")]
    let spmv_tuning: SpmvTuning = utils::default_spmv_tuning();

    let need_l1 = cfg.grid_relax_type.contains(&RelaxType::L1Jacobi);
    let need_cheb = cfg.grid_relax_type.contains(&RelaxType::Chebyshev);
    let need_cheb_safe = cfg.grid_relax_type.contains(&RelaxType::ChebyshevSafe);
    let need_safe_diag = cfg
        .grid_relax_type
        .contains(&RelaxType::SafeguardedGaussSeidel)
        || need_cheb_safe;
    let need_ilu0 = cfg.grid_relax_type.contains(&RelaxType::Ilu0);
    let need_ras = cfg.grid_relax_type.contains(&RelaxType::Ras);
    let allow_safeguard = need_safe_diag || need_ilu0 || need_ras;
    let need_fsai = cfg.grid_relax_type.contains(&RelaxType::Fsai);

    // Level 0 (finest)
    let mut lt0 = LevelSetupTiming::default();
    let t = tic();
    let diag0 = diag_inv_from_csr_cfg_fallback(&a_cur, cfg, allow_safeguard)?;
    if do_stats {
        lt0.diag = toc(t);
        lt0.total = lt0.diag;
    }
    let layout0 = if cfg.nodal == NodalMode::Nodal {
        Some(DofLayout::new(a_cur.nrows(), cfg.block_size))
    } else {
        None
    };
    let l0_num_functions = cfg
        .near_nullspace
        .as_ref()
        .map(|nns| nns.basis.len().max(1))
        .or_else(|| layout0.as_ref().map(|layout| layout.block_size.max(1)))
        .unwrap_or_else(|| cfg.num_functions.max(1));
    let l0 = AMGLevel {
        a: a_cur.clone(),
        p: CsrMatrix::identity(a_cur.nrows()),
        r: CsrMatrix::identity(a_cur.nrows()),
        diag_inv: diag0,
        d_sqrt_inv: None,
        l1_inv: None,
        diag_inv_safe: None,
        d_sqrt_inv_safe: None,
        cheb: None,
        cheb_safe: None,
        agg_of: (0..a_cur.nrows()).collect(),
        is_c: Vec::new(),
        cf: None,
        p2r_pos: Vec::new(),
        num_functions: l0_num_functions,
        row_basis: None,
        layout: layout0.clone(),
        nns: cfg.near_nullspace.as_ref().map(|nns| nns.basis.clone()),
        a_next_pat: None,
        a_next_pat_ng: None,
        rap_full2ng_pos: None,
        r_row_ptr: None,
        r_col_idx: None,
        r_vals_scratch: None,
        coarse_solver: None,
        ilu0: None,
        ras: None,
        fsai: None,
        a_vals_f32: None,
        diag_inv_f32: None,
        d_sqrt_inv_f32: None,
        l1_inv_f32: None,
        fsai_g_vals_f32: None,
        fsai_gt_vals_f32: None,
    };
    let mut l0 = l0;
    update_level_caches(
        cfg,
        &mut l0,
        need_l1,
        need_cheb,
        need_safe_diag,
        need_cheb_safe,
        need_ilu0,
        need_ras,
        true,
    )?;
    if need_fsai {
        let strength0 = if cfg.fsai_use_strength {
            Some(Strength::from_csr(
                &l0.a,
                cfg.strong_threshold,
                cfg.normalize_strength,
            ))
        } else {
            None
        };
        l0.fsai = Some(fsai_build_for_level(cfg, &l0.a, strength0.as_ref())?);
        refresh_mixed_precision_shadows(cfg, &mut l0);
    }
    #[cfg(feature = "simd")]
    {
        let tuning = utils::default_spmv_tuning();
        build_level_spmv_plans(&mut l0, &tuning);
    }
    #[cfg(feature = "simd")]
    build_level_spmv_plans(&mut l0, &spmv_tuning);
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
    diag_stats.push(AmgLevelStats {
        p_min_col_norm: 0.0,
        p_cond_sketched: 0.0,
        galerkin_worst_rel: 0.0,
    });

    let mut block_size_cur = if cfg.nodal == NodalMode::Nodal {
        cfg.block_size
    } else {
        1
    };
    let mut trials_current = make_trial_matrix(cfg, a_cur.nrows())?;
    // Drive coarsening: build levels 0..L (inclusive L is coarsest)
    for level in 0..cfg.max_levels {
        let n = a_cur.nrows();
        if n <= cfg.coarse_threshold || n <= cfg.min_coarse_size {
            break;
        }

        let mut lt = LevelSetupTiming::default();

        let layout = if cfg.nodal == NodalMode::Nodal {
            Some(DofLayout::new(n, block_size_cur))
        } else {
            None
        };
        // 1) Strength of connection (sparse)
        let (s, nodal_strength_opt) = with_timing(do_stats, &mut lt.strength, || {
            if let Some(ref lay) = layout {
                let nodal = strength_nodal_from_csr(
                    &a_cur,
                    lay.block_size,
                    cfg.strong_threshold,
                    cfg.normalize_strength,
                );
                let strength = Strength {
                    row_ptr: nodal.row_ptr.clone(),
                    col_idx: nodal.col_idx.clone(),
                };
                (strength, Some(nodal))
            } else {
                (
                    Strength::from_csr(&a_cur, cfg.strong_threshold, cfg.normalize_strength),
                    None,
                )
            }
        });
        // 2) Aggregates
        let mis_k = if level < cfg.agg_num_levels {
            cfg.aggressive_mis_k.max(2)
        } else {
            1
        };
        let agg_algo = match cfg.coarsen_type {
            CoarsenType::RS => AggAlgo::RSGreedy,
            CoarsenType::HMIS => AggAlgo::HMIS,
            CoarsenType::PMIS => AggAlgo::PMIS,
            CoarsenType::Falgout => AggAlgo::Falgout,
        };
        let (agg_node, is_c_node) = with_timing(do_stats, &mut lt.aggregate, || {
            match (layout.as_ref(), nodal_strength_opt.as_ref()) {
                (Some(_), Some(nodal)) => build_aggregates_nodal(
                    nodal,
                    agg_algo,
                    &AggOpts {
                        mis_k,
                        cap_per_row: cfg.max_strong_per_row,
                    },
                ),
                _ => build_aggregates(
                    &s,
                    agg_algo,
                    &AggOpts {
                        mis_k,
                        cap_per_row: cfg.max_strong_per_row,
                    },
                ),
            }
        });
        let (agg, is_c) = if let Some(ref lay) = layout {
            lift_node_aggregates_to_dofs(&agg_node, &is_c_node, lay)
        } else {
            (agg_node, is_c_node)
        };
        let mut nns_basis: Vec<Vec<f64>> = if level == 0 {
            if let Some(ref nns) = cfg.near_nullspace {
                for (k, vec) in nns.basis.iter().enumerate() {
                    if vec.len() != n {
                        return Err(KError::InvalidInput(format!(
                            "AMG: near-nullspace vector {k} has length {}, expected {n}",
                            vec.len()
                        )));
                    }
                }
                nns.basis.clone()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let user_supplied_nns = !nns_basis.is_empty();
        if nns_basis.is_empty() {
            if let Some(ref lay) = layout {
                let mut basis = vec![vec![0.0; n]; lay.block_size.max(1)];
                for i in 0..n {
                    let comp = lay.comp_of[i];
                    basis[comp][i] = 1.0;
                }
                nns_basis = basis;
            } else {
                nns_basis.push(vec![1.0; n]);
            }
        }
        let mut target_functions = if user_supplied_nns {
            nns_basis.len().max(1)
        } else if level == 0 {
            cfg.num_functions.max(1)
        } else if let Some(ref lay) = layout {
            lay.block_size.max(1)
        } else {
            block_size_cur.max(1)
        };
        if !user_supplied_nns {
            if let Some(ref lay) = layout {
                target_functions = target_functions.max(lay.block_size.max(1));
            }
            target_functions = target_functions.max(nns_basis.len().max(1));
            while nns_basis.len() < target_functions {
                nns_basis.push(vec![0.0; n]);
            }
        }
        let num_functions = target_functions;
        let nns_opt = Some(nns_basis.clone());
        let comp_opt = if user_supplied_nns {
            None
        } else {
            layout.as_ref().map(|lay| lay.comp_of.clone())
        };
        let tp = TentativeP {
            n_coarse: 1 + agg.iter().copied().max().unwrap_or(0),
            agg_of: agg.clone(),
            num_functions,
            nns: nns_opt.clone(),
            comp_of: comp_opt.clone(),
        };
        let block_size_next = tp.num_functions.max(1);
        let d = diag_inv_from_csr_cfg_fallback(&a_cur, cfg, allow_safeguard)?;
        let diag_weights: Vec<f64> = d
            .iter()
            .map(|&inv| {
                if inv.abs() > 0.0 {
                    (1.0 / inv).abs().max(1e-30)
                } else {
                    1.0
                }
            })
            .collect();
        let mut tn_opt: Option<TentativeNodal> = None;
        if layout.is_some() {
            let n_agg = tp.n_coarse;
            let mut rows_per_agg: Vec<Vec<usize>> = vec![Vec::new(); n_agg];
            for (row, &g) in agg.iter().enumerate() {
                rows_per_agg[g].push(row);
            }
            let mut row_basis_vec = vec![0.0; n * num_functions];
            for rows in &rows_per_agg {
                orthonormalize_aggregate(
                    rows,
                    &nns_basis,
                    &diag_weights,
                    num_functions,
                    &mut row_basis_vec,
                );
            }
            tn_opt = Some(TentativeNodal {
                agg_of: agg.clone(),
                n_agg,
                mfun: num_functions,
                row_basis: row_basis_vec,
            });
        }
        let s_sym = s.symmetrize();
        let tn_ref = tn_opt.as_ref();
        let (mut p_csr, cf_opt): (Pcsr, Option<CFInfo>) =
            with_timing(do_stats, &mut lt.prolong, || {
                if matches!(
                    cfg.interp_type,
                    InterpType::Direct
                        | InterpType::Standard
                        | InterpType::Extended
                        | InterpType::Classical
                        | InterpType::HE
                ) {
                    let extended = matches!(cfg.interp_type, InterpType::Extended);
                    let (pat, cf) = classical_pattern(&a_cur, &s_sym, &is_c, extended);
                    let mut vals = vec![0.0; pat.col_idx.len()];
                    let params = ClassicalParams {
                        variant: match cfg.interp_type {
                            InterpType::Direct => ClassicalVariant::Direct,
                            InterpType::HE => ClassicalVariant::HE,
                            InterpType::Standard | InterpType::Classical | InterpType::Extended => {
                                ClassicalVariant::Standard
                            }
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
                } else if let Some(tn) = tn_ref {
                    Ok((
                        smooth_tentative_sa_mf(
                            &a_cur,
                            &d,
                            tn,
                            cfg.jacobi_omega,
                            cfg.interpolation_truncation,
                            cfg.max_elements_per_row,
                        ),
                        None,
                    ))
                } else {
                    Ok((
                        smooth_tentative_sa_multi(
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
        let row_basis_for_level = tn_opt.as_ref().map(|tn| tn.row_basis.clone());
        let ctx = LevelPostContext {
            r: num_functions,
            agg_of: &tp.agg_of,
            nns: nns_opt
                .as_ref()
                .map(|v| v.iter().map(|b| b.as_slice()).collect()),
            a: Some(&a_cur),
            d_inv: Some(&d),
        };
        apply_post_interp(cfg, &ctx, &p_csr.row_ptr, &p_csr.col_idx, &mut p_csr.vals)?;
        if cfg.adaptive_interp
            && cfg.adaptive_samples > 0
            && cf_opt.is_none()
            && tp.num_functions == 1
            && p_csr.n > cfg.max_coarse_size
        {
            let omega = if cfg.adaptive_smooth_omega == 0.0 {
                cfg.jacobi_omega
            } else {
                cfg.adaptive_smooth_omega
            };
            let samples = sample_low_modes(
                &a_cur,
                &d,
                cfg.adaptive_samples,
                cfg.adaptive_smooth_steps,
                omega,
                0xC0FFEE,
            )?;
            let coarse_samples =
                restrict_samples_to_coarse(&a_cur, &tp, &samples, cfg.adaptive_weight_mode);
            adaptive_fit_values_only(
                &p_csr.row_ptr,
                &p_csr.col_idx,
                &mut p_csr.vals,
                &tp,
                &samples,
                &coarse_samples,
                cfg.adaptive_lambda,
                cfg.adaptive_enforce_sum1,
                cfg.interpolation_truncation,
            )?;
        }
        let mut p = CsrMatrix::from_csr(
            p_csr.m,
            p_csr.n,
            p_csr.row_ptr.clone(),
            p_csr.col_idx.clone(),
            p_csr.vals.clone(),
        );
        let mut rank_diag = RankDiagnostics::default();
        let check_rank = cfg.verify_p_rank && cf_opt.is_none();
        if check_rank {
            rank_diag = check_p_rank_fast(&p, cfg)?;
            if rank_diag.suspect {
                let mut cond_report = rank_diag.cond_estimate;
                match cfg.on_rank_failure {
                    RankFallback::RetryLooserInterp if cf_opt.is_none() => {
                        match try_fix_rank(level, &a_cur, &d, &tp, &ctx, &mut p_csr, cfg)? {
                            RankFixOutcome::Fixed => {
                                p = CsrMatrix::from_csr(
                                    p_csr.m,
                                    p_csr.n,
                                    p_csr.row_ptr.clone(),
                                    p_csr.col_idx.clone(),
                                    p_csr.vals.clone(),
                                );
                                rank_diag = check_p_rank_fast(&p, cfg)?;
                                cond_report = rank_diag.cond_estimate;
                                if rank_diag.suspect {
                                    return Err(KError::InvalidInput(format!(
                                        "AMG: P rank suspect at level {level}, cond≈{cond_report:.3e}"
                                    )));
                                }
                            }
                            RankFixOutcome::Unfixed => {
                                return Err(KError::InvalidInput(format!(
                                    "AMG: P rank suspect at level {level}, cond≈{cond_report:.3e}"
                                )));
                            }
                        }
                    }
                    RankFallback::Abort => {
                        return Err(KError::InvalidInput(format!(
                            "AMG: P rank suspect at level {level}, cond≈{cond_report:.3e}"
                        )));
                    }
                    other => {
                        return Err(KError::InvalidInput(format!(
                            "AMG: rank fallback {other:?} not implemented at level {level}"
                        )));
                    }
                }
            }
        }
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
        #[cfg(debug_assertions)]
        debug_check_csr(&r, "R pattern");
        debug_assert_eq!(p_csr.col_idx.len(), p2r_pos.len());
        debug_assert_eq!(r_vals.len(), p2r_pos.len());
        for (pi, &ri) in p2r_pos.iter().enumerate() {
            debug_assert!(ri < r_vals.len(), "p2r_pos out of range");
            debug_assert!(
                (p_csr.vals[pi] - r_vals[ri]).abs() <= 1e-12,
                "R != P^T at index {pi}"
            );
        }

        let trials_next = if let Some(ref trials) = trials_current {
            let mut next = Mat::<f64>::zeros(r.nrows(), trials.ncols());
            restrict_trials(&r, trials.as_ref(), next.as_mut())?;
            Some(next)
        } else {
            None
        };

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

        let mut use_ng = cfg.non_galerkin.enabled && (level + 1) >= cfg.non_galerkin.start_level;
        if cfg.require_spd && cfg.forbid_non_galerkin_in_spd {
            use_ng = false;
        }
        let mut a_full = CsrMatrix::from_csr(
            pat.nrows,
            pat.ncols,
            pat.row_ptr.clone(),
            pat.col_idx.clone(),
            a_coarse_vals,
        );
        if !use_ng || !cfg.filter_after_non_galerkin {
            apply_trial_compensation(cfg, &mut a_full, trials_next.as_ref(), block_size_next)?;
        }
        let (mut a_coarse, mut ng_pat_opt, mut map_opt) = if use_ng {
            let (ng_pat, ng_vals, full2ng) = non_galerkin_filter_coarse(
                &pat,
                a_full.values(),
                cfg.non_galerkin.symmetry,
                NgRowFilter {
                    tau_abs: cfg.non_galerkin.drop_abs,
                    tau_rel: cfg.non_galerkin.drop_rel,
                    k_max: cfg.non_galerkin.cap_row,
                    lump_diag: cfg.non_galerkin.lump_diagonal,
                },
            );
            let mut a_ng = CsrMatrix::from_csr(
                ng_pat.nrows,
                ng_pat.ncols,
                ng_pat.row_ptr.clone(),
                ng_pat.col_idx.clone(),
                ng_vals,
            );
            if cfg.filter_after_non_galerkin {
                apply_trial_compensation(cfg, &mut a_ng, trials_next.as_ref(), block_size_next)?;
            }
            (a_ng, Some(ng_pat), Some(full2ng))
        } else {
            (a_full, None, None)
        };
        let mut galerkin_worst = 0.0;
        let allow_galerkin =
            cfg.verify_galerkin && cfg.filter_omega <= 0.0 && !use_ng && cfg.galerkin_samples > 0;
        if allow_galerkin {
            let (ok, worst) = galerkin_sample_check(
                &a_cur,
                &p,
                &r,
                &a_coarse,
                cfg.galerkin_samples,
                cfg.galerkin_rel_tol,
                0xBEEF,
            )?;
            galerkin_worst = worst;
            if !ok {
                let a_fix = rap(&r, &a_cur, &p)?;
                let (ok2, worst2) = galerkin_sample_check(
                    &a_cur,
                    &p,
                    &r,
                    &a_fix,
                    cfg.galerkin_samples,
                    cfg.galerkin_rel_tol,
                    0xBEEF,
                )?;
                if ok2 {
                    galerkin_worst = worst2;
                    a_coarse = a_fix;
                    ng_pat_opt = None;
                    map_opt = None;
                } else {
                    return Err(KError::InvalidInput(format!(
                        "AMG: Galerkin identity failed at level {level}: worst rel={worst:.3e} (retry={worst2:.3e})"
                    )));
                }
            }
        }
        let diag_inv_coarse = with_timing(do_stats, &mut lt.diag, || {
            diag_inv_from_csr_cfg_fallback(&a_coarse, cfg, allow_safeguard)
        })?;
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
        diag_stats.push(AmgLevelStats {
            p_min_col_norm: rank_diag.min_col_norm,
            p_cond_sketched: rank_diag.cond_estimate,
            galerkin_worst_rel: galerkin_worst,
        });

        // Replace previous temporary P/R by actual inter-level transfers and agg mapping
        let mut row_basis_owned = row_basis_for_level;
        if let Some(prev) = levels.last_mut() {
            prev.p = p.clone();
            prev.agg_of = tp.agg_of.clone();
            prev.is_c = is_c.clone();
            prev.cf = cf_opt.clone();
            prev.p2r_pos = p2r_pos;
            prev.num_functions = tp.num_functions;
            prev.row_basis = row_basis_owned.take();
            prev.nns = tp.nns.clone();
            prev.layout = layout.clone();
            prev.a_next_pat = Some(pat.clone());
            prev.a_next_pat_ng = ng_pat_opt.clone();
            prev.rap_full2ng_pos = map_opt;
            if cfg.keep_transpose {
                prev.r = r.clone();
                prev.r_row_ptr = None;
                prev.r_col_idx = None;
                prev.r_vals_scratch = None;
            } else {
                prev.r = CsrMatrix::identity(0);
                prev.r_row_ptr = Some(r_row_ptr);
                prev.r_col_idx = Some(r_col_idx);
                prev.r_vals_scratch = Some(vec![0.0; prev.r_col_idx.as_ref().unwrap().len()]);
            }
            #[cfg(feature = "simd")]
            build_level_spmv_plans(prev, &spmv_tuning);
        }

        // Next level (coarser)
        a_cur = a_coarse.clone();
        trials_current = trials_next;
        block_size_cur = tp.num_functions;
        let mut next_level = AMGLevel {
            a: a_coarse,
            p: CsrMatrix::identity(a_cur.nrows()),
            r: CsrMatrix::identity(a_cur.nrows()),
            diag_inv: diag_inv_coarse,
            d_sqrt_inv: None,
            l1_inv: None,
            diag_inv_safe: None,
            d_sqrt_inv_safe: None,
            cheb: None,
            cheb_safe: None,
            agg_of: (0..a_cur.nrows()).collect(),
            is_c: Vec::new(),
            cf: None,
            p2r_pos: Vec::new(),
            num_functions: 1,
            row_basis: None,
            layout: if cfg.nodal == NodalMode::Nodal {
                Some(DofLayout::new(a_cur.nrows(), block_size_cur))
            } else {
                None
            },
            nns: None,
            a_next_pat: None,
            a_next_pat_ng: None,
            rap_full2ng_pos: None,
            r_row_ptr: None,
            r_col_idx: None,
            r_vals_scratch: None,
            coarse_solver: None,
            ilu0: None,
            ras: None,
            fsai: None,
            a_vals_f32: None,
            diag_inv_f32: None,
            d_sqrt_inv_f32: None,
            l1_inv_f32: None,
            fsai_g_vals_f32: None,
            fsai_gt_vals_f32: None,
        };
        update_level_caches(
            cfg,
            &mut next_level,
            need_l1,
            need_cheb,
            need_safe_diag,
            need_cheb_safe,
            need_ilu0,
            need_ras,
            true,
        )?;
        if need_fsai {
            let strength_coarse = if cfg.fsai_use_strength {
                Some(Strength::from_csr(
                    &next_level.a,
                    cfg.strong_threshold,
                    cfg.normalize_strength,
                ))
            } else {
                None
            };
            next_level.fsai = Some(fsai_build_for_level(
                cfg,
                &next_level.a,
                strength_coarse.as_ref(),
            )?);
            refresh_mixed_precision_shadows(cfg, &mut next_level);
        }
        #[cfg(feature = "simd")]
        build_level_spmv_plans(&mut next_level, &spmv_tuning);
        levels.push(next_level);

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
            if ls_len >= 2
                && let Some(prev) = level_stats.get_mut(ls_len - 2)
            {
                prev.nnz_p = p.nnz();
                prev.nnz_r = r.nnz();
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

    diag_stats.push(AmgLevelStats {
        p_min_col_norm: 0.0,
        p_cond_sketched: 0.0,
        galerkin_worst_rel: 0.0,
    });
    if cfg.non_galerkin.enabled && cfg.non_galerkin.oc_target.is_some() {
        enforce_oc_target(&mut levels, cfg)?;
    }

    let L = levels.len() - 1;
    if matches!(cfg.coarse_solve, CoarseSolve::ILU) {
        let n = levels[L].a.nrows();
        let mut ilu = CoarseIlu::new(
            cfg.tolerance,
            n.min(cfg.max_iterations.max(50)),
            cfg.ilu_drop_tol,
            cfg.ilu_fill_per_row,
        );
        ilu.setup(&levels[L].a)?;
        levels[L].coarse_solver = Some(Mutex::new(Box::new(ilu)));
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
        stats.diagnostics = diag_stats;
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

fn build_smoother_only_hierarchy(
    fine: &CsrMatrix<f64>,
    cfg: &mut AMGConfig,
) -> Result<(AmgHierarchy, Option<AmgStats>), KError> {
    let allow_safeguard = cfg
        .grid_relax_type
        .contains(&RelaxType::SafeguardedGaussSeidel)
        || cfg.grid_relax_type.contains(&RelaxType::ChebyshevSafe)
        || cfg.grid_relax_type.contains(&RelaxType::Ilu0)
        || cfg.grid_relax_type.contains(&RelaxType::Ras);
    let need_l1 = cfg.grid_relax_type.contains(&RelaxType::L1Jacobi);
    let need_cheb = cfg.grid_relax_type.contains(&RelaxType::Chebyshev);
    let need_cheb_safe = cfg.grid_relax_type.contains(&RelaxType::ChebyshevSafe);
    let need_safe_diag = cfg
        .grid_relax_type
        .contains(&RelaxType::SafeguardedGaussSeidel)
        || need_cheb_safe;
    let need_ilu0 = cfg.grid_relax_type.contains(&RelaxType::Ilu0);
    let need_ras = cfg.grid_relax_type.contains(&RelaxType::Ras);
    let need_fsai = cfg.grid_relax_type.contains(&RelaxType::Fsai);

    let diag0 = diag_inv_from_csr_cfg_fallback(fine, cfg, allow_safeguard)?;
    let layout0 = if cfg.nodal == NodalMode::Nodal {
        Some(DofLayout::new(fine.nrows(), cfg.block_size))
    } else {
        None
    };
    let l0_num_functions = cfg
        .near_nullspace
        .as_ref()
        .map(|nns| nns.basis.len().max(1))
        .or_else(|| layout0.as_ref().map(|layout| layout.block_size.max(1)))
        .unwrap_or_else(|| cfg.num_functions.max(1));
    let mut l0 = AMGLevel {
        a: fine.clone(),
        p: CsrMatrix::identity(fine.nrows()),
        r: CsrMatrix::identity(fine.nrows()),
        diag_inv: diag0,
        d_sqrt_inv: None,
        l1_inv: None,
        diag_inv_safe: None,
        d_sqrt_inv_safe: None,
        cheb: None,
        cheb_safe: None,
        agg_of: (0..fine.nrows()).collect(),
        is_c: Vec::new(),
        cf: None,
        p2r_pos: Vec::new(),
        num_functions: l0_num_functions,
        row_basis: None,
        layout: layout0,
        nns: cfg.near_nullspace.as_ref().map(|nns| nns.basis.clone()),
        a_next_pat: None,
        a_next_pat_ng: None,
        rap_full2ng_pos: None,
        r_row_ptr: None,
        r_col_idx: None,
        r_vals_scratch: None,
        coarse_solver: None,
        ilu0: None,
        ras: None,
        fsai: None,
        a_vals_f32: None,
        diag_inv_f32: None,
        d_sqrt_inv_f32: None,
        l1_inv_f32: None,
        fsai_g_vals_f32: None,
        fsai_gt_vals_f32: None,
    };
    update_level_caches(
        cfg,
        &mut l0,
        need_l1,
        need_cheb,
        need_safe_diag,
        need_cheb_safe,
        need_ilu0,
        need_ras,
        true,
    )?;
    if need_fsai {
        let strength0 = if cfg.fsai_use_strength {
            Some(Strength::from_csr(
                &l0.a,
                cfg.strong_threshold,
                cfg.normalize_strength,
            ))
        } else {
            None
        };
        l0.fsai = Some(fsai_build_for_level(cfg, &l0.a, strength0.as_ref())?);
        refresh_mixed_precision_shadows(cfg, &mut l0);
    }

    let hier = AmgHierarchy {
        policy: RelaxPolicy {
            kind: cfg.grid_relax_type,
            sweeps: cfg.num_grid_sweeps,
            omega: cfg.jacobi_omega,
        },
        coarse_solve: CoarseSolve::Smoother,
        levels: vec![l0],
    };

    let stats_opt = if cfg.logging_level > 0 {
        let mut stats = AmgStats::from_hierarchy(&hier);
        stats.levels = vec![LevelStats {
            level: 0,
            n: fine.nrows(),
            nnz_a: fine.nnz(),
            nnz_p: 0,
            nnz_r: 0,
            max_row_sum_a: max_row_sum_abs(fine),
            eff_nnz_a: Some(eff_nnz(fine, cfg.stats_eps)),
        }];
        stats.diagnostics = vec![AmgLevelStats {
            p_min_col_norm: 0.0,
            p_cond_sketched: 0.0,
            galerkin_worst_rel: 0.0,
        }];
        Some(stats)
    } else {
        None
    };

    Ok((hier, stats_opt))
}

fn enforce_oc_target(levels: &mut Vec<AMGLevel>, cfg: &mut AMGConfig) -> Result<(), KError> {
    if let Some(target) = cfg.non_galerkin.oc_target {
        let allow_safeguard = cfg
            .grid_relax_type
            .contains(&RelaxType::SafeguardedGaussSeidel)
            || cfg.grid_relax_type.contains(&RelaxType::ChebyshevSafe)
            || cfg.grid_relax_type.contains(&RelaxType::Ilu0)
            || cfg.grid_relax_type.contains(&RelaxType::Ras);
        let mut trials_current = make_trial_matrix(cfg, levels[0].a.nrows())?;
        for _ in 0..cfg.non_galerkin.oc_max_iter {
            let oc = operator_complexity_estimate(levels);
            if oc <= target {
                break;
            }
            cfg.non_galerkin.drop_abs *= 1.25;
            cfg.non_galerkin.drop_rel = (cfg.non_galerkin.drop_rel * 1.1).min(0.95);
            for l in 0..levels.len() - 1 {
                if let Some(pat_full) = levels[l].a_next_pat.clone() {
                    let r_tmp_storage = if cfg.keep_transpose {
                        None
                    } else {
                        Some(build_r_from_p(&mut levels[l]))
                    };
                    let r_for_ops = r_tmp_storage.as_ref().unwrap_or(&levels[l].r);
                    let trials_next = if let Some(ref trials) = trials_current {
                        let mut next = Mat::<f64>::zeros(r_for_ops.nrows(), trials.ncols());
                        restrict_trials(r_for_ops, trials.as_ref(), next.as_mut())?;
                        Some(next)
                    } else {
                        None
                    };
                    if l + 1 < cfg.non_galerkin.start_level {
                        trials_current = trials_next;
                        continue;
                    }
                    let mut vals_full = vec![0.0; pat_full.col_idx.len()];
                    rap_numeric(
                        &pat_full,
                        r_for_ops,
                        &levels[l].a,
                        &levels[l].p,
                        &mut vals_full,
                    );
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
                            pat_full.nrows,
                            &pat_full.row_ptr,
                            &pat_full.col_idx,
                            &mut vals_full,
                            &mut rf,
                        );
                    }
                    let block_size_next = levels[l].num_functions.max(1);
                    let mut a_full = CsrMatrix::from_csr(
                        pat_full.nrows,
                        pat_full.ncols,
                        pat_full.row_ptr.clone(),
                        pat_full.col_idx.clone(),
                        vals_full,
                    );
                    if !cfg.filter_after_non_galerkin {
                        apply_trial_compensation(
                            cfg,
                            &mut a_full,
                            trials_next.as_ref(),
                            block_size_next,
                        )?;
                    }
                    let (ng_pat, ng_vals, full2ng) = non_galerkin_filter_coarse(
                        &pat_full,
                        a_full.values(),
                        cfg.non_galerkin.symmetry,
                        NgRowFilter {
                            tau_abs: cfg.non_galerkin.drop_abs,
                            tau_rel: cfg.non_galerkin.drop_rel,
                            k_max: cfg.non_galerkin.cap_row,
                            lump_diag: cfg.non_galerkin.lump_diagonal,
                        },
                    );
                    let mut a_ng = CsrMatrix::from_csr(
                        ng_pat.nrows,
                        ng_pat.ncols,
                        ng_pat.row_ptr.clone(),
                        ng_pat.col_idx.clone(),
                        ng_vals,
                    );
                    if cfg.filter_after_non_galerkin {
                        apply_trial_compensation(
                            cfg,
                            &mut a_ng,
                            trials_next.as_ref(),
                            block_size_next,
                        )?;
                    }
                    levels[l].a_next_pat_ng = Some(ng_pat.clone());
                    levels[l].rap_full2ng_pos = Some(full2ng);
                    levels[l + 1].a = a_ng;
                    levels[l + 1].diag_inv =
                        diag_inv_from_csr_cfg_fallback(&levels[l + 1].a, cfg, allow_safeguard)?;
                    trials_current = trials_next;
                } else {
                    trials_current = None;
                }
            }
        }
    }
    Ok(())
}

// ===== Sparse utilities (local; avoid dense on hot path) ====================

fn orthonormalize_aggregate(
    rows: &[usize],
    basis_cols: &[Vec<f64>],
    weights: &[f64],
    mfun: usize,
    row_basis: &mut [f64],
) {
    if rows.is_empty() {
        return;
    }
    let mut q_cols: Vec<Vec<f64>> = Vec::new();
    let limit = basis_cols.len().min(mfun);
    for f in 0..limit {
        let mut col: Vec<f64> = rows.iter().map(|&row_idx| basis_cols[f][row_idx]).collect();
        for prev in 0..q_cols.len() {
            let mut dot = 0.0;
            for (local_ix, &row_idx) in rows.iter().enumerate() {
                let w = weights[row_idx];
                dot += w * col[local_ix] * q_cols[prev][local_ix];
            }
            for local_ix in 0..col.len() {
                col[local_ix] -= dot * q_cols[prev][local_ix];
            }
        }
        let mut norm_sq = 0.0;
        for (local_ix, &row_idx) in rows.iter().enumerate() {
            let w = weights[row_idx];
            let v = col[local_ix];
            norm_sq += w * v * v;
        }
        if norm_sq <= 1e-24 {
            continue;
        }
        let norm = norm_sq.sqrt();
        for val in &mut col {
            *val /= norm;
        }
        q_cols.push(col);
    }
    for (local_ix, &row_idx) in rows.iter().enumerate() {
        for f in 0..mfun {
            let val = q_cols.get(f).map(|col| col[local_ix]).unwrap_or(0.0);
            row_basis[row_idx * mfun + f] = val;
        }
    }
}

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

fn diag_inv_from_csr_with_floor(
    a: &CsrMatrix<f64>,
    floor: f64,
    require_positive: bool,
) -> Result<Vec<f64>, KError> {
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
        if floor > 0.0 && aii <= 0.0 {
            aii += floor;
        }
        if aii.abs() < 1e-14 {
            return Err(KError::SolveError(format!("near-zero diagonal at row {i}")));
        }
        if require_positive && aii <= 0.0 {
            return Err(KError::SolveError(format!(
                "non-positive diagonal at row {i} (value {aii})"
            )));
        }
        d[i] = 1.0 / aii;
    }
    Ok(d)
}

fn diag_inv_from_csr_cfg(a: &CsrMatrix<f64>, cfg: &AMGConfig) -> Result<Vec<f64>, KError> {
    let floor = if cfg.require_spd {
        cfg.spd_diag_floor.max(0.0)
    } else {
        0.0
    };
    diag_inv_from_csr_with_floor(a, floor, cfg.require_spd)
}

fn diag_inv_from_csr_safeguarded(a: &CsrMatrix<f64>) -> Vec<f64> {
    let n = a.nrows();
    let mut inv = vec![0.0; n];
    let row_ptr = a.row_ptr();
    let col_idx = a.col_idx();
    let vals = a.values();
    for i in 0..n {
        let rs = row_ptr[i];
        let re = row_ptr[i + 1];
        let mut diag = 0.0;
        let mut row_sum = 0.0;
        for p in rs..re {
            let v = vals[p];
            row_sum += v.abs();
            if col_idx[p] == i {
                diag = v.abs();
            }
        }
        let denom = diag.max(row_sum).max(1e-30);
        inv[i] = 1.0 / denom;
    }
    inv
}

fn diag_inv_from_csr_cfg_fallback(
    a: &CsrMatrix<f64>,
    cfg: &AMGConfig,
    allow_safeguard: bool,
) -> Result<Vec<f64>, KError> {
    match diag_inv_from_csr_cfg(a, cfg) {
        Ok(v) => Ok(v),
        Err(err) if allow_safeguard && !cfg.require_spd => Ok(diag_inv_from_csr_safeguarded(a)),
        Err(err) => Err(err),
    }
}

fn diag_inv_from_csr(a: &CsrMatrix<f64>) -> Result<Vec<f64>, KError> {
    diag_inv_from_csr_with_floor(a, 0.0, false)
}

fn make_d_sqrt_inv(diag_inv: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; diag_inv.len()];
    for (dst, &d) in out.iter_mut().zip(diag_inv.iter()) {
        *dst = if d > 0.0 { d.sqrt() } else { 0.0 };
    }
    out
}

fn compute_cheb_data(
    cfg: &AMGConfig,
    a: &CsrMatrix<f64>,
    _diag_inv: &[f64],
    d_sqrt_inv: &[f64],
) -> Result<ChebData, KError> {
    let mut lam_max = chebyshev::estimate_lmax_sym(a, d_sqrt_inv, cfg.chebyshev_power_steps)?;
    if !lam_max.is_finite() || lam_max <= 0.0 {
        let mut fallback: f64 = 0.0;
        let row_ptr = a.row_ptr();
        let col_idx = a.col_idx();
        let vals = a.values();
        for i in 0..a.nrows() {
            let rs = row_ptr[i];
            let re = row_ptr[i + 1];
            let mut diag = 0.0f64;
            let mut sum = 0.0f64;
            for p in rs..re {
                let j = col_idx[p];
                let v = vals[p].abs();
                if j == i {
                    diag = v;
                } else {
                    sum += v;
                }
            }
            if diag > 0.0 {
                fallback = fallback.max(sum / diag);
            }
        }
        lam_max = fallback;
    }
    if !lam_max.is_finite() || lam_max <= 0.0 {
        lam_max = 1.0;
    }
    let safety = cfg.chebyshev_safety.max(1.0);
    lam_max *= safety;
    if !lam_max.is_finite() || lam_max <= 0.0 {
        lam_max = safety.max(1.0);
    }
    let ratio = cfg.chebyshev_lower_ratio.clamp(1e-6, 0.99);
    let lam_min = (ratio * lam_max).max(1e-30);
    Ok(ChebData {
        lambda_max: lam_max,
        lambda_min: lam_min,
    })
}

fn update_level_caches(
    cfg: &AMGConfig,
    level: &mut AMGLevel,
    need_l1: bool,
    need_cheb: bool,
    need_safe_diag: bool,
    need_cheb_safe: bool,
    need_ilu0: bool,
    need_ras: bool,
    recompute_cheb: bool,
) -> Result<(), KError> {
    if need_l1 {
        level.l1_inv = Some(l1_diag_inv(&level.a));
    } else {
        level.l1_inv = None;
    }
    if need_cheb {
        let d_sqrt = make_d_sqrt_inv(&level.diag_inv);
        if recompute_cheb || level.cheb.is_none() {
            let cheb = compute_cheb_data(cfg, &level.a, &level.diag_inv, &d_sqrt)?;
            level.cheb = Some(cheb);
        }
        level.d_sqrt_inv = Some(d_sqrt);
    } else {
        level.d_sqrt_inv = None;
        level.cheb = None;
    }
    if need_safe_diag {
        let diag_safe = diag_inv_from_csr_safeguarded(&level.a);
        level.diag_inv_safe = Some(diag_safe);
    } else {
        level.diag_inv_safe = None;
    }
    if need_cheb_safe {
        let diag_safe = level
            .diag_inv_safe
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("ChebyshevSafe cache missing".into()))?;
        let mut d_sqrt = vec![0.0; diag_safe.len()];
        for (dst, &d) in d_sqrt.iter_mut().zip(diag_safe.iter()) {
            *dst = if d > 0.0 { d.sqrt() } else { 0.0 };
        }
        if recompute_cheb || level.cheb_safe.is_none() {
            let cheb = compute_cheb_data(cfg, &level.a, diag_safe, &d_sqrt)?;
            level.cheb_safe = Some(cheb);
        }
        level.d_sqrt_inv_safe = Some(d_sqrt);
    } else {
        level.d_sqrt_inv_safe = None;
        level.cheb_safe = None;
    }
    if need_ilu0 {
        build_ilu0_cache(level)?;
    } else {
        level.ilu0 = None;
    }
    if need_ras {
        build_ras_cache(level)?;
    } else {
        level.ras = None;
    }
    refresh_mixed_precision_shadows(cfg, level);
    Ok(())
}

#[cfg(not(feature = "complex"))]
fn build_ilu0_cache(level: &mut AMGLevel) -> Result<(), KError> {
    let mut cfg = IluCsrConfig::default();
    cfg.kind = IluKind::Ilu0;
    cfg.pivot = PivotStrategy::DiagonalPerturbation;
    cfg.pivot_threshold = 1e-12;
    cfg.diag_perturb_factor = 1e-10;
    cfg.level_sched = false;
    cfg.reordering = ReorderingOptions::default();
    cfg.conditioning = ConditioningOptions::default();
    let mut ilu = IluCsr::new_with_config(cfg);
    let op = crate::matrix::op::CsrOp::new(Arc::new(level.a.clone()));
    ilu.setup(&op)?;
    level.ilu0 = Some(Mutex::new(ilu));
    Ok(())
}

#[cfg(feature = "complex")]
fn build_ilu0_cache(_level: &mut AMGLevel) -> Result<(), KError> {
    Err(KError::Unsupported(
        "AMG ILU0 cache requires real scalars".into(),
    ))
}

#[cfg(not(feature = "complex"))]
fn build_ras_cache(level: &mut AMGLevel) -> Result<(), KError> {
    let cfg = AsmConfig {
        overlap: 1,
        combine: AsmCombine::Restricted,
        local_solver: AsmLocalSolver::ILU,
        local_sweeps: 1,
        weight_partition_of_unity: false,
        deterministic: true,
        nparts: None,
    };
    let mut ras = Asm::with_config(cfg);
    let op = crate::matrix::op::CsrOp::new(Arc::new(level.a.clone()));
    Preconditioner::setup(&mut ras, &op)?;
    level.ras = Some(Mutex::new(ras));
    Ok(())
}

#[cfg(feature = "complex")]
fn build_ras_cache(_level: &mut AMGLevel) -> Result<(), KError> {
    Err(KError::Unsupported(
        "AMG RAS cache requires real scalars".into(),
    ))
}

fn cast_slice_to_f32(src: &[f64]) -> Vec<f32> {
    src.iter().map(|&v| v as f32).collect()
}

fn refresh_mixed_precision_shadows(cfg: &AMGConfig, level: &mut AMGLevel) {
    let Some(mp) = cfg.mixed_precision else {
        level.a_vals_f32 = None;
        level.diag_inv_f32 = None;
        level.d_sqrt_inv_f32 = None;
        level.l1_inv_f32 = None;
        level.fsai_g_vals_f32 = None;
        level.fsai_gt_vals_f32 = None;
        return;
    };
    if cfg.mixed_storage != MixedStorage::Cached {
        level.a_vals_f32 = None;
        level.diag_inv_f32 = None;
        level.d_sqrt_inv_f32 = None;
        level.l1_inv_f32 = None;
        level.fsai_g_vals_f32 = None;
        level.fsai_gt_vals_f32 = None;
        return;
    }
    if mp.residual_enabled() || mp.smoothers_enabled() {
        level.a_vals_f32 = Some(cast_slice_to_f32(level.a.values()));
    } else {
        level.a_vals_f32 = None;
    }
    if mp.smoothers_enabled() {
        level.diag_inv_f32 = Some(cast_slice_to_f32(&level.diag_inv));
        if let Some(ref d) = level.d_sqrt_inv {
            level.d_sqrt_inv_f32 = Some(cast_slice_to_f32(d));
        } else {
            level.d_sqrt_inv_f32 = None;
        }
        if let Some(ref l1) = level.l1_inv {
            level.l1_inv_f32 = Some(cast_slice_to_f32(l1));
        } else {
            level.l1_inv_f32 = None;
        }
        if let Some(ref fsai) = level.fsai {
            level.fsai_g_vals_f32 = Some(cast_slice_to_f32(fsai.g.values()));
            level.fsai_gt_vals_f32 = Some(cast_slice_to_f32(fsai.gt.values()));
        } else {
            level.fsai_g_vals_f32 = None;
            level.fsai_gt_vals_f32 = None;
        }
    } else {
        level.diag_inv_f32 = None;
        level.d_sqrt_inv_f32 = None;
        level.l1_inv_f32 = None;
        level.fsai_g_vals_f32 = None;
        level.fsai_gt_vals_f32 = None;
    }
}

fn csr_lookup(a: &CsrMatrix<f64>, row: usize, col: usize) -> f64 {
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vv = a.values();
    let (rs, re) = (rp[row], rp[row + 1]);
    match ci[rs..re].binary_search(&col) {
        Ok(pos) => vv[rs + pos],
        Err(_) => 0.0,
    }
}

fn gather_dense_submatrix(a: &CsrMatrix<f64>, pattern: &[usize], buf: &mut Vec<f64>) {
    let m = pattern.len();
    buf.resize(m * m, 0.0);
    for (i_local, &i) in pattern.iter().enumerate() {
        for (j_local, &j) in pattern.iter().take(i_local + 1).enumerate() {
            let val = csr_lookup(a, i, j);
            buf[i_local * m + j_local] = val;
            buf[j_local * m + i_local] = val;
        }
    }
}

fn cholesky_factor(mat: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..i {
            let mut sum = mat[i * n + j];
            for k in 0..j {
                sum -= mat[i * n + k] * mat[j * n + k];
            }
            let diag = mat[j * n + j];
            if diag <= 0.0 {
                return false;
            }
            sum /= diag;
            mat[i * n + j] = sum;
        }
        let mut sum = mat[i * n + i];
        for k in 0..i {
            let v = mat[i * n + k];
            sum -= v * v;
        }
        if sum <= 0.0 {
            return false;
        }
        let diag = sum.sqrt();
        mat[i * n + i] = diag;
        for j in (i + 1)..n {
            mat[i * n + j] = 0.0;
        }
    }
    true
}

fn cholesky_solve(mat: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= mat[i * n + k] * y[k];
        }
        let diag = mat[i * n + i];
        y[i] = sum / diag;
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= mat[k * n + i] * x[k];
        }
        let diag = mat[i * n + i];
        x[i] = sum / diag;
    }
    x
}

fn solve_fsai_system(base: &[f64], rhs: &[f64], n: usize, lambda: f64) -> Option<Vec<f64>> {
    let mut attempt = if lambda >= 0.0 { lambda } else { 0.0 };
    let mut mat = vec![0.0; n * n];
    let mut tries = 0;
    while tries < 5 {
        mat.copy_from_slice(base);
        for d in 0..n {
            mat[d * n + d] += attempt;
        }
        if cholesky_factor(&mut mat, n) {
            return Some(cholesky_solve(&mat, rhs, n));
        }
        attempt = if attempt == 0.0 {
            1e-12
        } else {
            attempt * 10.0
        };
        tries += 1;
    }
    None
}

fn prune_pattern_row(a: &CsrMatrix<f64>, row: usize, pattern: &mut Vec<usize>, cap: usize) {
    if pattern.is_empty() {
        pattern.push(row);
    }
    if !pattern.contains(&row) {
        pattern.push(row);
    }
    if cap == 0 {
        pattern.clear();
        pattern.push(row);
        return;
    }
    if pattern.len() <= cap {
        pattern.sort_unstable();
        return;
    }
    let mut entries: Vec<(usize, f64)> = pattern
        .iter()
        .copied()
        .filter(|&col| col != row)
        .map(|col| (col, csr_lookup(a, row, col).abs()))
        .collect();
    entries.sort_unstable_by(
        |a, b| match b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal) {
            CmpOrdering::Equal => a.0.cmp(&b.0),
            other => other,
        },
    );
    let mut kept = Vec::with_capacity(cap.max(1));
    kept.push(row);
    for (col, _) in entries.into_iter().take(cap.saturating_sub(1)) {
        kept.push(col);
    }
    kept.sort_unstable();
    pattern.clear();
    pattern.extend(kept);
}

fn fsai_build_pattern(
    a: &CsrMatrix<f64>,
    strength: Option<&Strength>,
    dist: usize,
    cap: usize,
) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut patterns: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut mark = vec![0usize; n];
    let mut stamp = 1usize;
    let mut frontier: Vec<usize> = Vec::new();
    let mut next: Vec<usize> = Vec::new();
    for i in 0..n {
        let mut acc: Vec<usize> = vec![i];
        frontier.clear();
        frontier.push(i);
        mark[i] = stamp;
        for _ in 0..dist {
            next.clear();
            for &u in &frontier {
                let neighbors: &[usize] = if let Some(g) = strength {
                    let rs = g.row_ptr[u];
                    let re = g.row_ptr[u + 1];
                    &g.col_idx[rs..re]
                } else {
                    let rp = a.row_ptr();
                    let ci = a.col_idx();
                    let (rs, re) = (rp[u], rp[u + 1]);
                    &ci[rs..re]
                };
                for &v in neighbors {
                    if v >= n {
                        continue;
                    }
                    if mark[v] != stamp {
                        mark[v] = stamp;
                        acc.push(v);
                        next.push(v);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            std::mem::swap(&mut frontier, &mut next);
        }
        stamp = stamp.wrapping_add(1);
        acc.sort_unstable();
        acc.dedup();
        prune_pattern_row(a, i, &mut acc, cap);
        patterns.push(acc);
    }
    if cap > 0 {
        let mut additions: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            let current = patterns[i].clone();
            for &j in &current {
                if j >= n || j == i {
                    continue;
                }
                if patterns[j].binary_search(&i).is_err() {
                    additions.push((j, i));
                }
            }
        }
        additions.sort_unstable();
        additions.dedup();
        for (row, col) in additions {
            let pat = &mut patterns[row];
            match pat.binary_search(&col) {
                Ok(_) => {}
                Err(pos) => pat.insert(pos, col),
            }
            prune_pattern_row(a, row, pat, cap);
        }
    }
    patterns
}

fn fsai_enrich_pattern(
    a: &CsrMatrix<f64>,
    pattern: &mut Vec<usize>,
    sol: &[f64],
    cap: usize,
) -> usize {
    if pattern.len() >= cap {
        return 0;
    }
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vv = a.values();
    let mut accum: Vec<(usize, f64)> = Vec::new();
    for (local, &row) in pattern.iter().enumerate() {
        let coeff = sol[local];
        if coeff == 0.0 {
            continue;
        }
        let (rs, re) = (rp[row], rp[row + 1]);
        for idx in rs..re {
            let col = ci[idx];
            if pattern.binary_search(&col).is_ok() {
                continue;
            }
            accum.push((col, -coeff * vv[idx]));
        }
    }
    if accum.is_empty() {
        return 0;
    }
    accum.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut merged: Vec<(usize, f64)> = Vec::new();
    for (col, val) in accum {
        if let Some(last) = merged.last_mut()
            && last.0 == col
        {
            last.1 += val;
            continue;
        }
        merged.push((col, val));
    }
    merged.sort_unstable_by(|a, b| {
        match b
            .1
            .abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(CmpOrdering::Equal)
        {
            CmpOrdering::Equal => a.0.cmp(&b.0),
            other => other,
        }
    });
    let mut added = 0usize;
    let space = cap.saturating_sub(pattern.len());
    for (col, _) in merged.into_iter() {
        if added >= space {
            break;
        }
        if col >= a.ncols() {
            continue;
        }
        if let Err(pos) = pattern.binary_search(&col) {
            pattern.insert(pos, col);
            added += 1;
        }
    }
    added
}

fn fsai_drop_entries(
    a: &CsrMatrix<f64>,
    row: usize,
    pattern: &[usize],
    sol: &[f64],
    drop_tol: f64,
    cap: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut norm = 0.0;
    for &v in sol {
        norm += v * v;
    }
    let norm = norm.sqrt();
    let thr = drop_tol.max(0.0) * norm.max(1e-32);
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for (col, &val) in pattern.iter().zip(sol.iter()) {
        if *col == row {
            let mut keep = val;
            if keep.abs() < thr {
                let diag = csr_lookup(a, row, row);
                keep = if diag.abs() > 0.0 { 1.0 / diag } else { 1.0 };
            }
            cols.push(*col);
            vals.push(keep);
        } else if val.abs() >= thr {
            cols.push(*col);
            vals.push(val);
        }
    }
    if cols.is_empty() {
        let diag = csr_lookup(a, row, row);
        cols.push(row);
        vals.push(if diag.abs() > 0.0 { 1.0 / diag } else { 1.0 });
    }
    let limit = cap.max(1);
    if cols.len() > limit {
        let diag_pos = cols.iter().position(|&c| c == row).unwrap_or(0);
        let mut others: Vec<(usize, f64, usize)> = cols
            .iter()
            .enumerate()
            .filter(|&(idx, &c)| idx != diag_pos && c != row)
            .map(|(idx, &c)| (c, vals[idx].abs(), idx))
            .collect();
        others.sort_unstable_by(
            |a, b| match b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal) {
                CmpOrdering::Equal => a.0.cmp(&b.0),
                other => other,
            },
        );
        let mut keep = vec![diag_pos];
        for (_, _, idx) in others.into_iter().take(limit.saturating_sub(1)) {
            keep.push(idx);
        }
        keep.sort_unstable();
        let mut new_cols = Vec::with_capacity(keep.len());
        let mut new_vals = Vec::with_capacity(keep.len());
        for idx in keep {
            new_cols.push(cols[idx]);
            new_vals.push(vals[idx]);
        }
        cols = new_cols;
        vals = new_vals;
    }
    let mut pairs: Vec<(usize, f64)> = cols.into_iter().zip(vals).collect();
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut out_cols = Vec::with_capacity(pairs.len());
    let mut out_vals = Vec::with_capacity(pairs.len());
    for (c, v) in pairs {
        out_cols.push(c);
        out_vals.push(v);
    }
    (out_cols, out_vals)
}

fn fsai_factor_values(
    a: &CsrMatrix<f64>,
    patterns: &mut [Vec<usize>],
    lambda: f64,
    drop_tol: f64,
    adaptive_passes: usize,
    cap: usize,
) -> Result<CsrMatrix<f64>, KError> {
    let n = a.nrows();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut base = Vec::new();
    let mut rhs = Vec::new();
    row_ptr.push(0);
    for row in 0..n {
        let pat = &mut patterns[row];
        if pat.is_empty() {
            pat.push(row);
        }
        if !pat.contains(&row) {
            pat.push(row);
            pat.sort_unstable();
        }
        let mut pass = 0usize;
        loop {
            let m = pat.len();
            if m == 0 {
                break;
            }
            gather_dense_submatrix(a, pat, &mut base);
            rhs.resize(m, 0.0);
            if let Ok(pos) = pat.binary_search(&row) {
                rhs[pos] = 1.0;
            } else {
                rhs[0] = 1.0;
            }
            let solved = match solve_fsai_system(&base, &rhs, m, lambda) {
                Some(sol) => sol,
                None => {
                    pat.clear();
                    pat.push(row);
                    let diag = csr_lookup(a, row, row);
                    let val = if diag.abs() > 0.0 { 1.0 / diag } else { 1.0 };
                    col_idx.push(row);
                    values.push(val);
                    row_ptr.push(col_idx.len());
                    break;
                }
            };
            if adaptive_passes > 0 && pass < adaptive_passes {
                let added = fsai_enrich_pattern(a, pat, &solved, cap);
                if added > 0 {
                    pass += 1;
                    continue;
                }
            }
            let (cols, vals) = fsai_drop_entries(a, row, pat, &solved, drop_tol, cap);
            pat.clear();
            pat.extend(cols.iter().copied());
            col_idx.extend_from_slice(&cols);
            values.extend_from_slice(&vals);
            row_ptr.push(col_idx.len());
            break;
        }
    }
    Ok(CsrMatrix::from_csr(n, n, row_ptr, col_idx, values))
}

fn fsai_transpose_with_pos(g: &CsrMatrix<f64>) -> (CsrMatrix<f64>, Vec<usize>) {
    let m = g.nrows();
    let n = g.ncols();
    let nnz = g.nnz();
    let mut row_counts = vec![0usize; n + 1];
    for &col in g.col_idx() {
        row_counts[col + 1] += 1;
    }
    for i in 0..n {
        row_counts[i + 1] += row_counts[i];
    }
    let mut col_idx = vec![0usize; nnz];
    let mut vals = vec![0.0f64; nnz];
    let mut next = row_counts.clone();
    let mut map = vec![0usize; nnz];
    for row in 0..m {
        let (rs, re) = (g.row_ptr()[row], g.row_ptr()[row + 1]);
        for idx in rs..re {
            let col = g.col_idx()[idx];
            let dest = next[col];
            col_idx[dest] = row;
            vals[dest] = g.values()[idx];
            map[idx] = dest;
            next[col] += 1;
        }
    }
    (CsrMatrix::from_csr(n, m, row_counts, col_idx, vals), map)
}

fn fsai_build_for_level(
    cfg: &AMGConfig,
    a: &CsrMatrix<f64>,
    strength: Option<&Strength>,
) -> Result<FsaiData, KError> {
    let strength_owned = if cfg.fsai_use_strength {
        if let Some(s) = strength {
            Some(s.symmetrize())
        } else {
            Some(Strength::from_csr(a, cfg.strong_threshold, cfg.normalize_strength).symmetrize())
        }
    } else {
        None
    };
    let mut patterns = fsai_build_pattern(
        a,
        strength_owned.as_ref(),
        cfg.fsai_dist.max(1),
        cfg.fsai_max_per_row.max(1),
    );
    let g = fsai_factor_values(
        a,
        &mut patterns,
        cfg.fsai_lambda,
        cfg.fsai_drop_tol,
        cfg.fsai_adaptive_passes,
        cfg.fsai_max_per_row.max(1),
    )?;
    let (gt, map) = fsai_transpose_with_pos(&g);
    Ok(FsaiData {
        g,
        gt,
        g2gt_pos: map,
    })
}

fn fsai_refresh_numeric(
    a: &CsrMatrix<f64>,
    data: &mut FsaiData,
    lambda: f64,
) -> Result<(), KError> {
    let n = a.nrows();
    let rp = data.g.row_ptr().to_vec();
    let ci = data.g.col_idx().to_vec();
    let mut base = Vec::new();
    let mut rhs = Vec::new();
    {
        let vals = data.g.values_mut();
        for row in 0..n {
            let start = rp[row];
            let end = rp[row + 1];
            if start == end {
                continue;
            }
            let pattern = &ci[start..end];
            gather_dense_submatrix(a, pattern, &mut base);
            rhs.resize(pattern.len(), 0.0);
            if let Ok(pos) = pattern.binary_search(&row) {
                rhs[pos] = 1.0;
            } else if !pattern.is_empty() {
                rhs[0] = 1.0;
            }
            if let Some(sol) = solve_fsai_system(&base, &rhs, pattern.len(), lambda) {
                for (dst, val) in vals[start..end].iter_mut().zip(sol.iter()) {
                    *dst = *val;
                }
            } else {
                for (dst, &col) in vals[start..end].iter_mut().zip(pattern.iter()) {
                    if col == row {
                        let diag = csr_lookup(a, row, row);
                        *dst = if diag.abs() > 0.0 { 1.0 / diag } else { 1.0 };
                    } else {
                        *dst = 0.0;
                    }
                }
            }
        }
    }
    let g_vals = data.g.values();
    let gt_vals = data.gt.values_mut();
    for (src, &dst) in data.g2gt_pos.iter().enumerate() {
        gt_vals[dst] = g_vals[src];
    }
    Ok(())
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

fn compute_anisotropy<M>(a: &M) -> Vec<f64>
where
    M: DenseMatRef<f64> + Sync,
{
    let n = a.nrows();
    #[cfg(feature = "rayon")]
    return (0..n)
        .into_par_iter()
        .map(|i| {
            let diag = a.get(i, i);
            let mut max_off: f64 = 0.0;
            for j in 0..n {
                if i != j {
                    max_off = max_off.max(a.get(i, j).abs());
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
            let diag = a.get(i, i);
            let mut max_off: f64 = 0.0;
            for j in 0..n {
                if i != j {
                    max_off = max_off.max(a.get(i, j).abs());
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

fn compute_adaptive_threshold<M>(a: &M, base_threshold: f64) -> f64
where
    M: DenseMatRef<f64> + Sync,
{
    let anis = compute_anisotropy(a);
    let avg = if anis.is_empty() {
        1.0
    } else {
        anis.iter().sum::<f64>() / anis.len() as f64
    };
    base_threshold * (1.0 + avg.max(0.5))
}

/// S(i,j) = |A_ij| / sqrt(|A_ii| |A_jj|) if above threshold.
fn compute_strength_matrix<M>(a: &M, thr: f64) -> Mat<f64>
where
    M: DenseMatRef<f64>,
{
    let n = a.nrows();
    let mut s = Mat::<f64>::zeros(n, n);
    let mut diag = vec![0.0; n];
    for i in 0..n {
        diag[i] = a.get(i, i).abs();
    }
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let denom = (diag[i] * diag[j]).sqrt();
            if denom > 1e-14 {
                let st = a.get(i, j).abs() / denom;
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
        let mut norm2 = R::default();
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
    x.fill(R::default());

    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut ap = vec![R::default(); n];

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

fn pcg_left_precond<F>(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    maxit: usize,
    mut apply_prec: F,
) -> Result<(), KError>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), KError>,
{
    let n = a.nrows();
    if n == 0 {
        return Ok(());
    }
    x.fill(R::default());

    let mut r = b.to_vec();
    let mut z = vec![R::default(); n];
    let mut p = vec![R::default(); n];
    let mut ap = vec![R::default(); n];

    apply_prec(&r, &mut z)?;
    let mut rz_old = dot(&r, &z);
    let atol = tol.max(1e-12) * rz_old.abs().sqrt().max(1e-30);
    if rz_old.abs().sqrt() < atol {
        return Ok(());
    }
    p.copy_from_slice(&z);

    for _ in 0..maxit {
        a.spmv_scaled(1.0, &p, 0.0, &mut ap)?;
        let denom = dot(&p, &ap);
        if denom.abs() < 1e-30 {
            break;
        }
        let alpha = rz_old / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        apply_prec(&r, &mut z)?;
        let rz_new = dot(&r, &z);
        if rz_new.abs().sqrt() < atol {
            break;
        }
        let beta = rz_new / rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }
    Ok(())
}

#[inline]
fn dot(x: &[R], y: &[R]) -> R {
    let mut s = R::default();
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
pub struct AmgLevelStats {
    pub p_min_col_norm: f64,
    pub p_cond_sketched: f64,
    pub galerkin_worst_rel: f64,
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
    pub kcycle: Option<KCycle>,
}

impl Default for CycleTimings {
    fn default() -> Self {
        Self {
            per_level: Vec::new(),
            total_cycle: Duration::default(),
            cycle_type: CycleType::V,
            kcycle: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DistApplyStats {
    pub mode: DistCoarseStrategy,
    pub local_apply: Duration,
    pub gather: Duration,
    pub scatter: Duration,
    pub halo_exchange: Duration,
    pub reductions: usize,
}

impl Default for DistApplyStats {
    fn default() -> Self {
        Self {
            mode: DistCoarseStrategy::RootGather,
            local_apply: Duration::default(),
            gather: Duration::default(),
            scatter: Duration::default(),
            halo_exchange: Duration::default(),
            reductions: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AmgStats {
    pub grid_complexity: f64,
    pub operator_complexity: f64,
    pub num_levels: usize,
    pub levels: Vec<LevelStats>,
    pub diagnostics: Vec<AmgLevelStats>,
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
            diagnostics: Vec::new(),
            setup: SetupTimings::default(),
            last_cycle: None,
        }
    }
}

#[derive(Default)]
struct AmgRuntime {
    last_cycle: Option<CycleTimings>,
    last_dist_apply: Option<DistApplyStats>,
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
            nnz_r: if i < h.coarsest_ix() {
                if cfg.keep_transpose {
                    lvl.r.nnz()
                } else {
                    lvl.r_row_ptr
                        .as_ref()
                        .and_then(|rp| rp.last().copied())
                        .unwrap_or(0)
                }
            } else {
                0
            },
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
    let mut desc = match c.cycle_type {
        CycleType::V => "V-cycle".to_string(),
        CycleType::W { gamma } => format!("W-cycle(gamma={gamma})"),
    };
    if c.kcycle.is_some() {
        desc.push_str(" + K");
    }
    println!("{desc} timings (ms): level | pre mv axpy R coarse P post");
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

#[cfg(all(test, not(feature = "complex")))]
mod tests {
    use super::*;
    use faer::Mat;
    use std::any::Any;
    use std::cmp::Ordering;
    use std::sync::{Mutex, OnceLock};

    fn relax_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    mod chebyshev;
    mod cycle_policy;
    mod fsai_smoother;
    mod mixed_precision;
    mod nodal_nns;
    mod nodal_strength;
    mod rank_galerkin;

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

    fn ready_hierarchy(amg: &AMG) -> &AmgHierarchy {
        match &amg.state {
            AmgState::Ready { hierarchy, .. } => hierarchy,
            _ => panic!("AMG not ready"),
        }
    }

    fn reset_symbolic_counter() {
        BUILD_SYMBOLIC_COUNT.with(|c| c.set(0));
    }

    fn symbolic_counter() -> usize {
        BUILD_SYMBOLIC_COUNT.with(|c| c.get())
    }

    struct TestLinOp {
        mat: CsrMatrix<f64>,
        sid: StructureId,
        vid: ValuesId,
    }

    impl TestLinOp {
        fn new(mat: CsrMatrix<f64>, sid: StructureId, vid: ValuesId) -> Self {
            Self { mat, sid, vid }
        }

        fn with_values(&self, vid: ValuesId) -> Self {
            Self {
                mat: self.mat.clone(),
                sid: self.sid,
                vid,
            }
        }
    }

    impl LinOp for TestLinOp {
        type S = f64;

        fn dims(&self) -> (usize, usize) {
            (self.mat.nrows(), self.mat.ncols())
        }

        fn matvec(&self, x: &[f64], y: &mut [f64]) {
            crate::matrix::spmv::csr_matvec(&self.mat, x, y).unwrap();
        }

        fn as_any(&self) -> &dyn Any {
            &self.mat
        }

        fn structure_id(&self) -> StructureId {
            self.sid
        }

        fn values_id(&self) -> ValuesId {
            self.vid
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

        let push_acc = |row: usize,
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

    fn l2_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn identity_level() -> AMGLevel {
        let mut level = AMGLevel {
            a: CsrMatrix::identity(1),
            p: CsrMatrix::identity(1),
            r: CsrMatrix::identity(1),
            diag_inv: vec![1.0],
            d_sqrt_inv: None,
            l1_inv: None,
            diag_inv_safe: None,
            d_sqrt_inv_safe: None,
            cheb: None,
            cheb_safe: None,
            agg_of: vec![0],
            is_c: Vec::new(),
            cf: None,
            p2r_pos: vec![],
            num_functions: 1,
            row_basis: None,
            layout: None,
            nns: None,
            a_next_pat: None,
            a_next_pat_ng: None,
            rap_full2ng_pos: None,
            r_row_ptr: None,
            r_col_idx: None,
            r_vals_scratch: None,
            coarse_solver: None,
            ilu0: None,
            ras: None,
            fsai: None,
            a_vals_f32: None,
            diag_inv_f32: None,
            d_sqrt_inv_f32: None,
            l1_inv_f32: None,
            fsai_g_vals_f32: None,
            fsai_gt_vals_f32: None,
        };
        #[cfg(feature = "simd")]
        {
            let tuning = utils::default_spmv_tuning();
            build_level_spmv_plans(&mut level, &tuning);
        }
        level
    }

    fn level_from_matrix(a: &CsrMatrix<f64>) -> AMGLevel {
        let n = a.nrows();
        let mut level = AMGLevel {
            a: a.clone(),
            p: CsrMatrix::identity(n),
            r: CsrMatrix::identity(n),
            diag_inv: diag_inv_from_csr(a).unwrap(),
            d_sqrt_inv: None,
            l1_inv: None,
            diag_inv_safe: None,
            d_sqrt_inv_safe: None,
            cheb: None,
            cheb_safe: None,
            agg_of: vec![0; n.max(1)],
            is_c: Vec::new(),
            cf: None,
            p2r_pos: vec![],
            num_functions: 1,
            row_basis: None,
            layout: None,
            nns: None,
            a_next_pat: None,
            a_next_pat_ng: None,
            rap_full2ng_pos: None,
            r_row_ptr: None,
            r_col_idx: None,
            r_vals_scratch: None,
            coarse_solver: None,
            ilu0: None,
            ras: None,
            fsai: None,
            a_vals_f32: None,
            diag_inv_f32: None,
            d_sqrt_inv_f32: None,
            l1_inv_f32: None,
            fsai_g_vals_f32: None,
            fsai_gt_vals_f32: None,
        };
        #[cfg(feature = "simd")]
        {
            let tuning = utils::default_spmv_tuning();
            build_level_spmv_plans(&mut level, &tuning);
        }
        level
    }

    #[test]
    fn phase_selection_logic() {
        let _guard = relax_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
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
        let mut amg = AMG {
            state: AmgState::Ready {
                hierarchy: Box::new(hier),
                last_structure_id: StructureId(0),
                last_values_id: ValuesId(0),
                pattern_hash: 0,
            },
            ..Default::default()
        };
        amg.cfg.require_spd = false;
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
    fn fcg_presmooth_reduces_residual() {
        let n = 10;
        let a = poisson1d(n);
        let lvl = level_from_matrix(&a);
        let rhs = vec![1.0; n];
        let mut sol = vec![0.0; n];
        let mut ws = AMGWorkspace::new(n);
        let mut amg = AMG::default();
        amg.cfg.require_spd = false;

        amg.fcg_presmooth(
            &lvl,
            &rhs,
            &mut sol,
            5,
            0.0,
            1,
            RelaxType::Jacobi,
            amg.cfg.jacobi_omega,
            &mut ws,
        )
        .unwrap();

        let mut work = vec![0.0; n];
        a.spmv_scaled(1.0, &sol, 0.0, &mut work).unwrap();
        let mut r_out = 0.0;
        for i in 0..n {
            let ri = rhs[i] - work[i];
            r_out += ri * ri;
        }
        let r0 = (rhs.iter().map(|v| v * v).sum::<f64>()).sqrt();
        assert!(r_out.sqrt() < r0);
    }

    #[test]
    fn flexible_presmooth_reduces_relax_calls() {
        let make_amg = || AMG {
            state: AmgState::Ready {
                hierarchy: Box::new(AmgHierarchy {
                    levels: vec![identity_level(), identity_level()],
                    policy: RelaxPolicy {
                        kind: [RelaxType::Jacobi; 4],
                        sweeps: [1, 1, 1, 0],
                        omega: 1.0,
                    },
                    coarse_solve: CoarseSolve::DirectDense,
                }),
                last_structure_id: StructureId(0),
                last_values_id: ValuesId(0),
                pattern_hash: 0,
            },
            ..Default::default()
        };

        let mut amg_std = make_amg();
        amg_std.cfg.require_spd = false;
        reset_relax_counts();
        let rhs = [1.0];
        let mut sol = [0.0];
        amg_std.apply(PcSide::Left, &rhs, &mut sol).unwrap();
        let baseline = get_relax_counts()[RelaxPhase::Fine.ix()];

        let mut amg_flex = make_amg();
        amg_flex.cfg.require_spd = false;
        amg_flex.cfg.flexible_level = Some(0);
        amg_flex.cfg.flexible_iters = 3;
        amg_flex.cfg.flexible_pc_sweeps = 1;
        reset_relax_counts();
        sol[0] = 0.0;
        amg_flex.apply(PcSide::Left, &rhs, &mut sol).unwrap();
        let flex_counts = get_relax_counts()[RelaxPhase::Fine.ix()];
        assert!(flex_counts < baseline);
    }

    #[test]
    fn flexible_presmooth_spd_guard() {
        let a = poisson1d(4);
        let mut amg = AMGBuilder::new()
            .grid_relax_type_all(RelaxType::GaussSeidel)
            .flexible_level(0)
            .flexible_iters(2)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        let err = amg.setup(&a).unwrap_err();
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.contains("flexible presmoothing"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
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

        let ad = a.to_dense().unwrap();
        let pd = p.to_dense().unwrap();
        let rd = r.to_dense().unwrap();
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

    #[test]
    #[cfg(not(feature = "complex"))]
    fn filter_enforces_row_sums() {
        let a = poisson1d(64);
        let mut filtered = AMGBuilder::new()
            .rap_drop_abs(0.05)
            .require_spd(false)
            .filter_omega(1.0)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        let mut baseline = AMGBuilder::new()
            .rap_drop_abs(0.05)
            .require_spd(false)
            .filter_omega(0.0)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        filtered.setup(&a).unwrap();
        baseline.setup(&a).unwrap();
        let h_filtered = ready_hierarchy(&filtered);
        let h_baseline = ready_hierarchy(&baseline);
        assert_eq!(h_filtered.levels.len(), h_baseline.levels.len());
        let mut best_ratio = f64::INFINITY;
        let mut any_significant = false;
        for (lvl_ix, (lvl_filtered, lvl_baseline)) in
            h_filtered.levels.iter().zip(&h_baseline.levels).enumerate()
        {
            if lvl_ix == 0 {
                continue;
            }
            let n = lvl_filtered.a.nrows();
            assert_eq!(n, lvl_baseline.a.nrows());
            let ones = vec![1.0; n];
            let mut sums_filtered = vec![0.0; n];
            let mut sums_baseline = vec![0.0; n];
            lvl_filtered
                .a
                .spmv_scaled(1.0, &ones, 0.0, &mut sums_filtered)
                .unwrap();
            lvl_baseline
                .a
                .spmv_scaled(1.0, &ones, 0.0, &mut sums_baseline)
                .unwrap();
            let max_filtered = sums_filtered.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
            let max_baseline = sums_baseline.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
            if max_baseline > 1e-14 {
                let ratio = max_filtered / max_baseline;
                best_ratio = best_ratio.min(ratio);
                any_significant = true;
            } else {
                assert!(
                    max_filtered <= 1e-14,
                    "level {lvl_ix} filtered row sum {} exceeds tight tolerance",
                    max_filtered
                );
            }
        }
        assert!(
            any_significant,
            "no levels with meaningful baseline row sums"
        );
        assert!(
            best_ratio <= 0.1 + 1e-12,
            "expected at least one level to reduce max row sum by an order of magnitude: best_ratio={}",
            best_ratio
        );
    }
    #[test]
    #[cfg(not(feature = "complex"))]
    fn filter_non_galerkin_preserves_row_sums() {
        let a = poisson1d(64);
        let mut cfg_filtered = AMGConfig::default();
        cfg_filtered.require_spd = false;
        cfg_filtered.rap_truncation_abs = 0.02;
        cfg_filtered.filter_omega = 1.0;
        cfg_filtered.filter_after_non_galerkin = true;
        cfg_filtered.non_galerkin.enabled = true;
        cfg_filtered.non_galerkin.drop_abs = 0.05;
        cfg_filtered.non_galerkin.drop_rel = 0.0;
        cfg_filtered.non_galerkin.cap_row = 0;
        let mut cfg_baseline = cfg_filtered.clone();
        cfg_baseline.filter_omega = 0.0;
        let mut amg_filtered = AMG::with_config(cfg_filtered);
        let mut amg_baseline = AMG::with_config(cfg_baseline);
        amg_filtered.setup(&a).unwrap();
        amg_baseline.setup(&a).unwrap();
        let h_filtered = ready_hierarchy(&amg_filtered);
        let h_baseline = ready_hierarchy(&amg_baseline);
        assert_eq!(h_filtered.levels.len(), h_baseline.levels.len());
        let mut best_ratio = f64::INFINITY;
        let mut any_significant = false;
        for (lvl_ix, (lvl_filtered, lvl_baseline)) in
            h_filtered.levels.iter().zip(&h_baseline.levels).enumerate()
        {
            if lvl_ix == 0 {
                continue;
            }
            let n = lvl_filtered.a.nrows();
            assert_eq!(n, lvl_baseline.a.nrows());
            let ones = vec![1.0; n];
            let mut sums_filtered = vec![0.0; n];
            let mut sums_baseline = vec![0.0; n];
            lvl_filtered
                .a
                .spmv_scaled(1.0, &ones, 0.0, &mut sums_filtered)
                .unwrap();
            lvl_baseline
                .a
                .spmv_scaled(1.0, &ones, 0.0, &mut sums_baseline)
                .unwrap();
            let max_filtered = sums_filtered.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
            let max_baseline = sums_baseline.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
            if max_baseline > 1e-14 {
                let ratio = max_filtered / max_baseline;
                best_ratio = best_ratio.min(ratio);
                any_significant = true;
            } else {
                assert!(
                    max_filtered <= 1e-14,
                    "level {lvl_ix} filtered row sum {} exceeds tight tolerance",
                    max_filtered
                );
            }
        }
        assert!(
            any_significant,
            "no levels with meaningful baseline row sums"
        );
        assert!(
            best_ratio <= 0.1 + 1e-12,
            "expected at least one level to reduce max row sum by an order of magnitude: best_ratio={}",
            best_ratio
        );
    }
    #[test]
    #[cfg(not(feature = "complex"))]
    fn filter_reduces_constant_mode_residual() {
        let a = poisson1d(64);
        let mut cfg_off = AMGConfig::default();
        cfg_off.rap_truncation_abs = 0.02;
        cfg_off.require_spd = false;
        cfg_off.filter_omega = 0.0;
        let mut cfg_on = cfg_off.clone();
        cfg_on.filter_omega = 1.0;
        let mut amg_off = AMG::with_config(cfg_off);
        let mut amg_on = AMG::with_config(cfg_on);
        amg_off.setup(&a).unwrap();
        amg_on.setup(&a).unwrap();
        let h_on = ready_hierarchy(&amg_on);
        let h_off = ready_hierarchy(&amg_off);
        assert!(h_on.coarsest_ix() >= 1);
        let coarse_ix = 1;
        let n_fine = h_on.levels[0].a.nrows();
        let ones_fine = vec![1.0; n_fine];
        let mut rhs_on = vec![0.0; h_on.levels[coarse_ix].a.nrows()];
        h_on.levels[0]
            .r
            .spmv_scaled(1.0, &ones_fine, 0.0, &mut rhs_on)
            .unwrap();
        let mut prod_on = vec![0.0; rhs_on.len()];
        h_on.levels[coarse_ix]
            .a
            .spmv_scaled(1.0, &rhs_on, 0.0, &mut prod_on)
            .unwrap();
        let norm_on = l2_norm(&prod_on);

        let mut rhs_off = vec![0.0; h_off.levels[coarse_ix].a.nrows()];
        h_off.levels[0]
            .r
            .spmv_scaled(1.0, &ones_fine, 0.0, &mut rhs_off)
            .unwrap();
        let mut prod_off = vec![0.0; rhs_off.len()];
        h_off.levels[coarse_ix]
            .a
            .spmv_scaled(1.0, &rhs_off, 0.0, &mut prod_off)
            .unwrap();
        let norm_off = l2_norm(&prod_off);
        assert!(norm_off > 1e-12);
        assert!(norm_on <= norm_off * 0.1 + 1e-10);
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
        a.spmv_scaled(1.0, &z_l1, 0.0, &mut ws_l1.work[..3])
            .unwrap();
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
    #[cfg(not(feature = "complex"))]
    fn refresh_updates_caches() {
        let a = poisson1d(4);
        let mut amg_l1 = AMGBuilder::new()
            .grid_relax_type_all(RelaxType::L1Jacobi)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        amg_l1.setup(&a).unwrap();
        let old = ready_hierarchy(&amg_l1).levels[0].l1_inv.as_ref().unwrap()[0];
        let mut a2 = a.clone();
        let rp = a2.row_ptr();
        for p in rp[0]..rp[1] {
            a2.values_mut()[p] *= 2.0;
        }
        amg_l1.update_numeric(&a2).unwrap();
        let new = ready_hierarchy(&amg_l1).levels[0].l1_inv.as_ref().unwrap()[0];
        assert!((new - old).abs() > 1e-12);

        let mut amg_ch = AMGBuilder::new()
            .grid_relax_type_all(RelaxType::Chebyshev)
            .chebyshev_recompute_esteig(true)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        amg_ch.setup(&a).unwrap();
        let old_l = ready_hierarchy(&amg_ch).levels[0]
            .cheb
            .as_ref()
            .unwrap()
            .lambda_max;
        let old_ds = ready_hierarchy(&amg_ch).levels[0]
            .d_sqrt_inv
            .as_ref()
            .unwrap()[0];
        let mut a3 = a.clone();
        let rp3 = a3.row_ptr();
        for p in rp3[0]..rp3[1] {
            if a3.col_idx()[p] == 0 {
                a3.values_mut()[p] *= 1.5;
            }
        }
        amg_ch.update_numeric(&a3).unwrap();
        let new_l = ready_hierarchy(&amg_ch).levels[0]
            .cheb
            .as_ref()
            .unwrap()
            .lambda_max;
        let new_ds = ready_hierarchy(&amg_ch).levels[0]
            .d_sqrt_inv
            .as_ref()
            .unwrap()[0];
        assert!((new_l - old_l).abs() > 1e-6);
        assert!((new_ds - old_ds).abs() > 1e-12);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn coarse_ilu_reused() {
        let n = 8;
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            row_ptr[i] = col_idx.len();
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
        }
        row_ptr[n] = col_idx.len();
        let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals);

        let mut amg = AMGBuilder::new()
            .relaxation_type(RelaxType::Jacobi)
            .grid_relax_type_all(RelaxType::Jacobi)
            .coarse_solve(CoarseSolve::ILU)
            .require_spd(false)
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        amg.setup(&a).unwrap();

        let setups_before = {
            let h = ready_hierarchy(&amg);
            let lvl = &h.levels[h.coarsest_ix()];
            lvl.coarse_solver
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .nsetups()
        };

        let rhs = vec![1.0; n];
        let mut z = vec![0.0; n];
        amg.apply(PcSide::Left, &rhs, &mut z).unwrap();
        amg.apply(PcSide::Left, &rhs, &mut z).unwrap();

        let setups_after = {
            let h = ready_hierarchy(&amg);
            let lvl = &h.levels[h.coarsest_ix()];
            lvl.coarse_solver
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .nsetups()
        };

        assert_eq!(setups_before, 1);
        assert_eq!(setups_after, 1);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn preserves_num_functions_across_levels() {
        // Build 1D Poisson matrix of size 16
        let n = 16;
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
        let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals);

        // Two-function near nullspace: constant and linear
        let t0 = vec![1.0; n];
        let t1: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let mut amg = AMGBuilder::new()
            .coarse_threshold(1)
            .max_coarse_size(1)
            .near_nullspace(vec![t0, t1])
            .build(&Mat::<f64>::zeros(0, 0))
            .unwrap();
        amg.setup(&a).unwrap();

        let h = ready_hierarchy(&amg);
        for l in 0..h.coarsest_ix() {
            assert_eq!(h.levels[l].num_functions, 2);
        }
    }

    #[test]
    fn apply_before_setup_returns_err() {
        let amg = AMG::default();
        let rhs = vec![1.0, 2.0];
        let mut sol = vec![0.0, 0.0];
        assert!(matches!(
            amg.apply(PcSide::Left, &rhs, &mut sol),
            Err(KError::InvalidInput(_))
        ));
    }

    #[test]
    fn setup_is_idempotent_when_ids_unchanged() {
        reset_symbolic_counter();
        let mat = csr_from_triples(
            3,
            3,
            vec![
                (0, 0, 2.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 2.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 2.0),
            ],
        );
        let op = TestLinOp::new(mat.clone(), StructureId(1), ValuesId(1));
        let mut amg = AMG::default();
        amg.setup(&op).unwrap();
        assert_eq!(symbolic_counter(), 1);
        amg.setup(&op).unwrap();
        assert_eq!(symbolic_counter(), 1);
    }

    #[test]
    fn values_id_change_refreshes_numeric_only() {
        reset_symbolic_counter();
        let mat = csr_from_triples(
            3,
            3,
            vec![
                (0, 0, 2.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 2.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 2.0),
            ],
        );
        let mut amg = AMG::default();
        let op1 = TestLinOp::new(mat.clone(), StructureId(3), ValuesId(3));
        amg.setup(&op1).unwrap();
        assert_eq!(symbolic_counter(), 1);
        let op2 = op1.with_values(ValuesId(4));
        amg.setup(&op2).unwrap();
        assert_eq!(symbolic_counter(), 1);
    }

    #[test]
    fn structure_id_change_rebuilds_symbolic() {
        reset_symbolic_counter();
        let mat1 = csr_from_triples(
            3,
            3,
            vec![
                (0, 0, 2.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 2.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 2.0),
            ],
        );
        let mat2 = csr_from_triples(
            3,
            3,
            vec![
                (0, 0, 3.0),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 3.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 3.0),
                (2, 0, -0.5),
            ],
        );
        let mut amg = AMG::default();
        let op1 = TestLinOp::new(mat1, StructureId(5), ValuesId(5));
        amg.setup(&op1).unwrap();
        assert_eq!(symbolic_counter(), 1);
        let op2 = TestLinOp::new(mat2, StructureId(6), ValuesId(6));
        amg.setup(&op2).unwrap();
        assert_eq!(symbolic_counter(), 2);
    }

    #[test]
    fn transpose_mapping_updates_values() {
        let p = prolong::Pcsr {
            m: 4,
            n: 5,
            row_ptr: vec![0, 2, 4, 6, 7],
            col_idx: vec![0, 2, 1, 3, 0, 4, 2],
            vals: vec![1.0, 2.0, 3.0, 4.0, -1.0, 5.0, 6.0],
        };
        let (rr, rc, _, p2r) = transpose_csr_with_pos(&p);
        let mut r = CsrMatrix::from_csr(p.n, p.m, rr.clone(), rc.clone(), vec![0.0; p.vals.len()]);
        #[cfg(debug_assertions)]
        debug_check_csr(&r, "test keep transpose");
        let mut updated_rc = r.col_idx().to_vec();
        let mut updated_vals = vec![0.0; p.vals.len()];
        for (pi, &ri) in p2r.iter().enumerate() {
            updated_vals[ri] = p.vals[pi] * 2.0;
        }
        let r_updated = CsrMatrix::from_csr(
            p.n,
            p.m,
            rr.clone(),
            updated_rc.to_vec(),
            updated_vals.clone(),
        );
        let mut p_dense = Mat::<f64>::zeros(p.m, p.n);
        for i in 0..p.m {
            for k in p.row_ptr[i]..p.row_ptr[i + 1] {
                p_dense[(i, p.col_idx[k])] = p.vals[k] * 2.0;
            }
        }
        let r_dense = p_dense.transpose().to_owned();
        let mut r_from_updated = Mat::<f64>::zeros(r_updated.nrows(), r_updated.ncols());
        for i in 0..r_updated.nrows() {
            for k in r_updated.row_ptr()[i]..r_updated.row_ptr()[i + 1] {
                r_from_updated[(i, r_updated.col_idx()[k])] = r_updated.values()[k];
            }
        }
        assert_dense_eq(&r_from_updated, &r_dense, 1e-12, 1e-12);
    }

    #[test]
    fn galerkin_sample_check_spot() {
        let a = csr_from_triples(
            2,
            2,
            vec![(0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)],
        );
        let p = CsrMatrix::identity(2);
        let r = CsrMatrix::identity(2);
        let a_coarse = CsrMatrix::identity(2);
        let (ok, worst) = galerkin_sample_check(&a, &p, &r, &a_coarse, 4, 1e-12, 0xFEED).unwrap();
        assert!(ok);
        assert!(worst <= 1e-12);
    }

    #[test]
    fn near_zero_diagonal_aborts_setup() {
        let a = csr_from_triples(2, 2, vec![(0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)]);
        let op = TestLinOp::new(a, StructureId(7), ValuesId(7));
        let mut amg = AMG::default();
        let err = amg.setup(&op).unwrap_err();
        match err {
            KError::SolveError(msg) => assert!(msg.contains("near-zero diagonal")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[cfg(feature = "complex")]
    mod bridge {
        use super::*;
        use crate::algebra::bridge::BridgeScratch;
        use crate::ops::kpc::KPreconditioner;
        use crate::preconditioner::PcSide;

        fn poisson_1d(n: usize) -> CsrMatrix<f64> {
            let mut row_ptr = Vec::with_capacity(n + 1);
            let mut col_idx = Vec::new();
            let mut values = Vec::new();
            row_ptr.push(0);
            for i in 0..n {
                if i > 0 {
                    col_idx.push(i - 1);
                    values.push(-1.0);
                }
                col_idx.push(i);
                values.push(2.0);
                if i + 1 < n {
                    col_idx.push(i + 1);
                    values.push(-1.0);
                }
                row_ptr.push(col_idx.len());
            }
            CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
        }

        #[test]
        #[cfg(not(feature = "complex"))]
        fn apply_s_matches_real_path() {
            let a = poisson_1d(12);
            let mut amg = AMGBuilder::new()
                .relaxation_type(RelaxType::Jacobi)
                .grid_relax_type_all(RelaxType::Jacobi)
                .build(&Mat::<f64>::zeros(0, 0))
                .expect("amg build");
            amg.setup(&a).expect("amg setup");

            let rhs: Vec<f64> = (0..a.nrows()).map(|i| (i as f64).sin()).collect();
            let mut out_real = vec![0.0; rhs.len()];
            amg.apply(PcSide::Left, &rhs, &mut out_real)
                .expect("real amg apply");

            let rhs_s: Vec<S> = rhs.iter().copied().map(S::from_real).collect();
            let mut out_s = vec![S::zero(); rhs_s.len()];
            let mut scratch = BridgeScratch::default();
            amg.apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
                .expect("scalar amg apply");

            for (yr, ys) in out_real.iter().zip(out_s.iter()) {
                assert!((ys.real() - yr).abs() < 1e-10, "real mismatch");
                assert!(ys.imag().abs() < 1e-12, "imag component drift");
            }
        }
    }
}
