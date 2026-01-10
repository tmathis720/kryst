//! PETSc-style options for KSP/PC, parsed via a table-driven engine.
//!
//! Changes:
//! - CLI precedence: CLI > options file(s) > environment > defaults
//! - Added `-options_file <path>` (PETSc-like)
//! - Help now generated from registry: `print_help()`
//! - Boolean flags accept presence or explicit on/off/true/false/1/0

use crate::config::kinds;
use crate::error::KError;
use std::str::FromStr;

use crate::config::options_core::is_help_requested;
use crate::config::options_core::{Sink, Spec, expand_options_files, parse_as};
use crate::config::registry::registry;
use crate::preconditioner::dist::{GlobalPcKind, LocalPcKind, MpiPcOptions};
use crate::preconditioner::ilu::{
    IluConfig, IluType as IluVariant, ReorderingType as IluReorderingType,
    TriSolveType as IluTriSolveType,
};
use crate::preconditioner::ilu_options::{
    IluKind, IluOptions, IterativeSetupType, Overlay, PivotPolicy, ReorderingType, TriSolveType,
};
use crate::utils::conditioning::{ConditioningOptions, ScaleDirection, ScaleNorm};

fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "on"))
}

fn env_lower(key: &str) -> Option<String> {
    std::env::var(key).ok().map(|v| v.to_lowercase())
}

/// Supported CG algorithm variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgVariant {
    /// Classic Hestenes-Stiefel CG.
    Classic,
    /// Pipelined CG that trades local work for fewer global reductions.
    Pipelined,
}

impl Default for CgVariant {
    fn default() -> Self {
        CgVariant::Classic
    }
}

impl FromStr for CgVariant {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "classic" => Ok(CgVariant::Classic),
            "pipelined" | "p-cg" | "pcg" => Ok(CgVariant::Pipelined),
            _ => Err("valid values: classic | pipelined"),
        }
    }
}

/// KSP (Krylov Solver) options.
#[derive(Debug, Default, Clone)]
pub struct KspOptions {
    pub ksp_type: Option<String>,
    pub rtol: Option<f64>,
    pub atol: Option<f64>,
    pub dtol: Option<f64>,
    pub maxits: Option<usize>,
    pub restart: Option<usize>,
    /// Reduction mode for global dot products: "fast" | "deterministic" | "deterministic-accurate"
    pub reduction: Option<String>,
    /// When true, invoke KSP monitors only on rank 0 (MPI).
    pub ksp_monitor_rank0: Option<bool>,
    /// Enable reproducible reductions regardless of the global mode. When true,
    /// the solver selects deterministic MPI reductions and fixed-order local
    /// kernels; combine with `threads = Some(1)` (or `-ksp_threads 1`) for
    /// bit-for-bit equality across runs.
    pub reproducible: Option<bool>,
    /// Select the CG algorithm variant. Defaults to [`CgVariant::Classic`].
    pub cg_variant: Option<CgVariant>,
    // GMRES/FGMRES-specific (backward-compatible; all optional)
    /// Override restart for GMRES; falls back to `restart` if unset
    pub gmres_restart: Option<usize>,
    /// Orthogonalization method for GMRES: "mgs" | "cgs"
    pub gmres_orthog: Option<String>,
    /// If true, perform a second orthogonalization pass (reorthogonalization)
    pub gmres_reorthog: Option<bool>,
    /// GMRES reorthogonalization mode: "never" | "ifneeded" | "always"
    pub gmres_reorth: Option<String>,
    /// GMRES reorthogonalization tolerance used by the if-needed strategy
    pub gmres_reorth_tol: Option<f64>,
    /// If true, treat near-zero residual as a happy breakdown and stop
    pub gmres_happy_breakdown: Option<bool>,
    /// GMRES algorithm variant: "classical" | "pipelined" | "sstep"
    pub gmres_variant: Option<String>,
    /// Block size for s-step GMRES panels
    pub gmres_sstep: Option<usize>,
    /// Conditioning guard for s-step QR (default 1e8)
    pub gmres_sstep_max_cond: Option<f64>,

    /// Override restart for FGMRES; falls back to `restart` if unset
    pub fgmres_restart: Option<usize>,
    /// Orthogonalization method for FGMRES: "mgs" | "cgs"
    pub fgmres_orthog: Option<String>,
    /// If true, perform a second orthogonalization pass (reorthogonalization)
    pub fgmres_reorthog: Option<bool>,
    /// FGMRES reorthogonalization mode: "never" | "ifneeded" | "always"
    pub fgmres_reorth: Option<String>,
    /// FGMRES reorthogonalization tolerance used by the if-needed strategy
    pub fgmres_reorth_tol: Option<f64>,
    /// If true, treat near-zero residual as a happy breakdown and stop
    pub fgmres_happy_breakdown: Option<bool>,
    /// FGMRES algorithm variant: "classical" | "pipelined"
    pub fgmres_variant: Option<String>,
    pub pc_side: Option<String>,
    pub matrix_file: Option<String>,
    pub rhs_file: Option<String>,
    pub min_iter: Option<usize>,
    pub cf_tol: Option<f64>,
    pub skip_real_r_check: Option<bool>,
    pub epsmac: Option<f64>,
    pub guard_zero_residual: Option<f64>,
    pub cg_norm: Option<String>,
    pub cg_pipelined: Option<bool>,
    pub cg_replace_every: Option<usize>,
    pub cg_single_reduction: Option<bool>,
    pub trust_region: Option<f64>,
    pub cg_use_async: Option<bool>,
    pub cg_async_min_n: Option<usize>,
    /// Number of Rayon worker threads (requires the `rayon` feature). Ignored otherwise.
    pub threads: Option<usize>,
    /// Threading mode for Rayon: "context" | "global" | "serial".
    pub threads_mode: Option<String>,
    pub min_len_vec: Option<usize>,
    pub min_rows_spmv: Option<usize>,
    pub chunk_rows_spmv: Option<usize>,
}

/// KSP type tag for option resolution helpers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KspType {
    GMRES,
    FGMRES,
    Other,
}

/// PC options.
#[derive(Debug, Default, Clone)]
pub struct PcOptions {
    /// Preconditioner type (e.g., "jacobi", "ilut").
    pub pc_type: Option<String>,
    /// Level of fill for ILU(k).
    pub ilu_level: Option<usize>,
    /// Degree for Chebyshev smoother.
    pub chebyshev_degree: Option<usize>,
    /// Drop tolerance for ILUT.
    pub ilut_drop_tol: Option<f64>,
    /// Maximum fill for ILUT.
    pub ilut_max_fill: Option<usize>,
    pub ilut_perm_tol: Option<f64>,
    pub ilutp_max_fill: Option<usize>,
    pub ilutp_drop_tol: Option<f64>,
    pub ilutp_perm_tol: Option<f64>,
    pub reorder: Option<String>,
    pub scaling: Option<String>,
    /// Overlap for Additive Schwarz.
    pub asm_overlap: Option<usize>,
    /// ASM mode: "asm" (classical) or "ras" (restricted).
    pub asm_mode: Option<String>,
    /// ASM weighting: "none"|"uniform"|"linear"|"poly:k".
    pub asm_weighting: Option<String>,
    /// Explicit subdomain sizes for ASM.
    pub asm_subdomains: Option<Vec<usize>>,
    pub asm_inner_pc: Option<String>,
    pub chebyshev_lambda_min: Option<f64>,
    pub chebyshev_lambda_max: Option<f64>,
    pub amg_levels: Option<usize>,
    pub amg_strength_threshold: Option<f64>,
    pub amg_nu_pre: Option<usize>,
    pub amg_nu_post: Option<usize>,
    pub amg_coarse_threshold: Option<usize>,
    pub amg_max_coarse_size: Option<usize>,
    pub amg_min_coarse_size: Option<usize>,
    pub amg_truncation_factor: Option<f64>,
    pub amg_max_elements_per_row: Option<usize>,
    pub amg_interpolation_truncation: Option<f64>,
    pub amg_coarsen_type: Option<String>,
    pub amg_interp_type: Option<String>,
    pub amg_relax_type: Option<String>,
    pub amg_logging_level: Option<usize>,
    pub amg_print_level: Option<usize>,
    pub amg_tolerance: Option<f64>,
    pub amg_max_iterations: Option<usize>,
    pub amg_min_iterations: Option<usize>,
    pub amg_ieee_checks: Option<bool>,
    pub amg_optimize_workspace: Option<bool>,
    /// Controls both pre/post smoothing sweeps.
    pub amg_smoother_steps: Option<usize>,
    /// Jacobi omega for smoothing/adaptive routines.
    pub amg_smoother_omega: Option<f64>,
    /// Absolute RAP truncation tolerance.
    pub amg_rap_truncation_abs: Option<f64>,
    /// Cap on entries per RAP row.
    pub amg_rap_max_elements_per_row: Option<usize>,
    /// Keep transpose copy for RAP when possible.
    pub amg_keep_transpose: Option<bool>,
    /// Preserve pivot entries while truncating RAP.
    pub amg_keep_pivot_in_rap: Option<bool>,
    /// Enforce SPD-safe AMG behavior.
    pub amg_require_spd: Option<bool>,
    /// Print the AMG setup statistics when enabled.
    pub amg_print_setup: Option<bool>,
    /// Distributed AMG apply mode: "root" or "local".
    pub amg_dist_apply_mode: Option<String>,
    /// Enable distributed apply instrumentation.
    pub amg_dist_instrumentation: Option<bool>,
    /// Scaling for halo-based coarse correction prototype.
    pub amg_dist_coarse_ghost_scale: Option<f64>,
    /// Chain string, e.g. "jacobi->ilut".
    pub pc_chain: Option<String>,
    /// Structured chain.
    pub chain: Option<Vec<PcOptions>>,
    /// Relaxation factor for SOR, ω ∈ (0, 2).
    pub omega: Option<f64>,
    /// Generic drop tolerance.
    pub drop_tol: Option<f64>,

    pub ilu_type: Option<String>,
    pub ilu_level_of_fill: Option<usize>,
    pub ilu_max_fill_per_row: Option<usize>,
    pub ilu_offdiag_drop_tolerance: Option<f64>,
    /// Reserved for future Schur complement preconditioners; not yet implemented.
    pub ilu_schur_drop_tolerance: Option<f64>,
    pub ilu_reordering_type: Option<String>,
    pub ilu_triangular_solve: Option<String>,
    pub ilu_lower_jacobi_iters: Option<usize>,
    pub ilu_upper_jacobi_iters: Option<usize>,
    pub ilu_tolerance: Option<f64>,
    pub ilu_max_iterations: Option<usize>,
    pub ilu_logging_level: Option<usize>,
    pub ilu_print_level: Option<usize>,
    pub ilu_ieee_checks: Option<bool>,
    pub ilu_pivot_monitoring: Option<bool>,
    pub ilu_optimize_workspace: Option<bool>,
    pub ilu_pivot_threshold: Option<f64>,
    pub ilu_parallel_factorization: Option<bool>,
    pub ilu_parallel_triangular_solve: Option<bool>,
    pub ilu_parallel_chunk_size: Option<usize>,
    pub ilu_par_factor: Option<String>,
    pub ilu_parilu_max_iters: Option<usize>,
    pub ilu_parilu_min_iters: Option<usize>,
    pub ilu_parilu_tol: Option<f64>,
    pub ilu_parilu_omega: Option<f64>,
    pub ilu_distributed: Option<bool>,
    pub ilu_pivot_mode: Option<String>,
    pub ilu_pivot_scale: Option<String>,
    pub ilu_pivot_tau: Option<f64>,
    /// Structured ILU options (new schema).
    pub ilu: IluOptions,
    // ---- Approximate inverse (CSR) ----
    pub approxinv_kind: Option<String>, // "fsai" | "spai"
    pub approxinv_levels: Option<usize>,
    pub approxinv_max_per_col: Option<usize>,
    pub approxinv_drop_tol: Option<f64>,
    pub approxinv_reg: Option<f64>,
    pub approxinv_max_cond: Option<f64>,
    pub approxinv_parallel: Option<bool>,

    pub superlu_pivot_threshold: Option<f64>,
    pub superlu_replace_tiny_pivots: Option<bool>,
    pub superlu_print_level: Option<u8>,
    pub superlu_process_grid: Option<(usize, usize)>,
    pub superlu_column_permutation: Option<String>,
    pub superlu_row_permutation: Option<String>,
    pub superlu_iterative_refinement: Option<String>,
    pub superlu_static_pivoting: Option<bool>,
    pub superlu_panel_size: Option<usize>,
    pub superlu_enable_3d_factorization: Option<bool>,
    pub superlu_process_grid_3d_depth: Option<usize>,
    pub superlu_memory_tradeoff_factor: Option<f64>,
    pub superlu_max_concurrent_panels: Option<usize>,
    pub superlu_async_panel_updates: Option<bool>,
    pub superlu_workspace_memory_limit: Option<usize>,
    pub superlu_aggressive_memory_reuse: Option<bool>,
    pub superlu_preallocation_strategy: Option<String>,
    pub reuse_policy: Option<String>,

    // Additional per-PC knobs
    /// Block size for block Jacobi.
    pub jacobi_block_size: Option<usize>,
    /// Choose the global MPI preconditioner mode.
    pub pc_global: Option<String>,
    /// Choose the local ILU variant for distributed runs.
    pub pc_local: Option<String>,
    /// ILU variant ("iluk", "ilut", ...).
    pub ilu_variant: Option<String>,
    /// Reordering strategy for ILU.
    pub ilu_reordering: Option<String>,

    // SOR
    /// SOR relaxation ω ∈ (0,2).
    pub sor_omega: Option<f64>,
    /// Number of SOR sweeps.
    pub sor_sweeps: Option<usize>,
    /// Use symmetric SOR.
    pub sor_symmetric: Option<bool>,
    /// Matrix side for SOR traversal.
    pub sor_mat_side: Option<String>,

    // Chebyshev
    /// Degree of Chebyshev polynomial.
    pub cheb_degree: Option<usize>,
    /// Lower eigenvalue estimate for Chebyshev.
    pub cheb_eig_lo: Option<f64>,
    /// Upper eigenvalue estimate for Chebyshev.
    pub cheb_eig_hi: Option<f64>,

    // ASM
    /// Hint for ASM subdomain size.
    pub asm_subdomain_size: Option<usize>,
    /// Block solver choice for ASM: "ludense" or "csr" (default: "ludense").
    pub asm_block_solver: Option<String>,

    // AMG
    /// AMG smoother name.
    pub amg_smoother: Option<String>,
    /// Conditioning: fix tiny/missing diagonal entries.
    pub pc_fixdiag: Option<bool>,
    /// Conditioning: shift diagonal by a constant.
    pub pc_shift_diag: Option<f64>,
    /// Conditioning: inject tau * row_norm into diagonal.
    pub pc_diag_inject_tau: Option<f64>,
    /// Conditioning: scale matrix rows/columns.
    pub pc_scale: Option<String>,
    /// Conditioning: scaling norm ("1" or "inf").
    pub pc_scale_norm: Option<String>,
}

/// Side enum kept as-is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcSide {
    Left,
    Right,
    Symmetric,
}

impl FromStr for PcSide {
    type Err = KError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "left" => Ok(PcSide::Left),
            "right" => Ok(PcSide::Right),
            "symmetric" => Ok(PcSide::Symmetric),
            _ => Err(KError::SolveError(format!("Unrecognized pc_side: {s}"))),
        }
    }
}

// ---- Sink impls (table-to-field wiring) ----

macro_rules! set_opt {
    ($slot:expr, $expr:expr) => {{
        *$slot = Some($expr);
        Ok(())
    }};
}

fn ensure_ge_1(name: &str, val: usize) -> Result<usize, KError> {
    if val == 0 {
        Err(KError::SolveError(format!("{name} must be ≥ 1")))
    } else {
        Ok(val)
    }
}

impl Sink for KspOptions {
    fn set_bool(&mut self, key: &str, v: bool) -> Result<(), KError> {
        match key {
            "ksp_skip_real_r_check" => set_opt!(&mut self.skip_real_r_check, v),
            "ksp_cg_pipelined" => {
                self.cg_pipelined = Some(v);
                self.cg_variant = Some(if v {
                    CgVariant::Pipelined
                } else {
                    CgVariant::Classic
                });
                Ok(())
            }
            "ksp_cg_single_reduction" => set_opt!(&mut self.cg_single_reduction, v),
            "ksp_cg_use_async" => set_opt!(&mut self.cg_use_async, v),
            "ksp_gmres_reorthog" => set_opt!(&mut self.gmres_reorthog, v),
            "ksp_gmres_happy_breakdown" => set_opt!(&mut self.gmres_happy_breakdown, v),
            "ksp_fgmres_reorthog" => set_opt!(&mut self.fgmres_reorthog, v),
            "ksp_fgmres_happy_breakdown" => set_opt!(&mut self.fgmres_happy_breakdown, v),
            "ksp_monitor_rank0" => set_opt!(&mut self.ksp_monitor_rank0, v),
            "ksp_reproducible" => set_opt!(&mut self.reproducible, v),
            _ => Err(KError::SolveError(format!("Unknown KSP bool key: {key}"))),
        }
    }

    fn set_val(&mut self, spec: &Spec, v: &str) -> Result<(), KError> {
        match spec.key {
            "ksp_type" => set_opt!(&mut self.ksp_type, v.to_string()),
            "ksp_rtol" => set_opt!(&mut self.rtol, parse_as::<f64>(v, spec)?),
            "ksp_atol" => set_opt!(&mut self.atol, parse_as::<f64>(v, spec)?),
            "ksp_dtol" => set_opt!(&mut self.dtol, parse_as::<f64>(v, spec)?),
            "ksp_max_it" => set_opt!(&mut self.maxits, parse_as::<usize>(v, spec)?),
            // restart flags
            "ksp_restart" => set_opt!(
                &mut self.restart,
                ensure_ge_1("ksp_restart", parse_as::<usize>(v, spec)?)?
            ),
            "ksp_gmres_restart" => set_opt!(
                &mut self.gmres_restart,
                ensure_ge_1("ksp_gmres_restart", parse_as::<usize>(v, spec)?)?
            ),
            // Additional GMRES/FGMRES keys
            "ksp_gmres_orthog" => set_opt!(&mut self.gmres_orthog, v.to_string()),
            "ksp_gmres_reorth" => set_opt!(&mut self.gmres_reorth, v.to_string()),
            "ksp_gmres_reorth_tol" => {
                set_opt!(&mut self.gmres_reorth_tol, parse_as::<f64>(v, spec)?)
            }
            "ksp_gmres_variant" => set_opt!(&mut self.gmres_variant, v.to_string()),
            "ksp_gmres_sstep" => set_opt!(
                &mut self.gmres_sstep,
                ensure_ge_1("ksp_gmres_sstep", parse_as::<usize>(v, spec)?)?
            ),
            "ksp_gmres_sstep_max_cond" => {
                set_opt!(&mut self.gmres_sstep_max_cond, parse_as::<f64>(v, spec)?)
            }
            "ksp_reduction" => set_opt!(&mut self.reduction, v.to_string()),
            "ksp_fgmres_restart" => set_opt!(
                &mut self.fgmres_restart,
                ensure_ge_1("ksp_fgmres_restart", parse_as::<usize>(v, spec)?)?
            ),
            "ksp_fgmres_orthog" => set_opt!(&mut self.fgmres_orthog, v.to_string()),
            "ksp_fgmres_reorth" => set_opt!(&mut self.fgmres_reorth, v.to_string()),
            "ksp_fgmres_reorth_tol" => {
                set_opt!(&mut self.fgmres_reorth_tol, parse_as::<f64>(v, spec)?)
            }
            "ksp_fgmres_variant" => set_opt!(&mut self.fgmres_variant, v.to_string()),
            "ksp_pc_side" => set_opt!(&mut self.pc_side, v.to_string()),
            "matrix" => set_opt!(&mut self.matrix_file, v.to_string()),
            "rhs" => set_opt!(&mut self.rhs_file, v.to_string()),
            "ksp_min_iter" => set_opt!(&mut self.min_iter, parse_as::<usize>(v, spec)?),
            "ksp_cf_tol" => set_opt!(&mut self.cf_tol, parse_as::<f64>(v, spec)?),
            "ksp_epsmac" => set_opt!(&mut self.epsmac, parse_as::<f64>(v, spec)?),
            "ksp_guard_zero_residual" => {
                set_opt!(&mut self.guard_zero_residual, parse_as::<f64>(v, spec)?)
            }
            "ksp_cg_variant" => {
                let variant = CgVariant::from_str(v).map_err(|msg| {
                    KError::SolveError(format!("Unrecognized ksp_cg_variant: {v} ({msg})"))
                })?;
                self.cg_variant = Some(variant);
                self.cg_pipelined = Some(matches!(variant, CgVariant::Pipelined));
                Ok(())
            }
            "ksp_cg_norm" => set_opt!(&mut self.cg_norm, v.to_string()),
            "ksp_cg_replace_every" => {
                set_opt!(&mut self.cg_replace_every, parse_as::<usize>(v, spec)?)
            }
            "ksp_cg_async_min_n" => {
                set_opt!(&mut self.cg_async_min_n, parse_as::<usize>(v, spec)?)
            }
            "ksp_trust_region" => set_opt!(&mut self.trust_region, parse_as::<f64>(v, spec)?),
            "ksp_threads" => set_opt!(
                &mut self.threads,
                ensure_ge_1("ksp_threads", parse_as::<usize>(v, spec)?)?
            ),
            "ksp_threads_mode" => set_opt!(&mut self.threads_mode, v.to_lowercase()),
            "ksp_min_len_vec" => set_opt!(
                &mut self.min_len_vec,
                ensure_ge_1("ksp_min_len_vec", parse_as::<usize>(v, spec)?)?
            ),
            "ksp_min_rows_spmv" => set_opt!(
                &mut self.min_rows_spmv,
                ensure_ge_1("ksp_min_rows_spmv", parse_as::<usize>(v, spec)?)?
            ),
            "ksp_chunk_rows_spmv" => set_opt!(
                &mut self.chunk_rows_spmv,
                ensure_ge_1("ksp_chunk_rows_spmv", parse_as::<usize>(v, spec)?)?
            ),
            "options_file" => Ok(()), // consumed earlier by expansion
            _ => Err(KError::SolveError(format!("Unknown KSP key: {}", spec.key))),
        }
    }

    fn set_pair(&mut self, _spec: &Spec, _a: &str, _b: &str) -> Result<(), KError> {
        Err(KError::SolveError("KSP has no pair-arity flags".into()))
    }
}

impl Sink for PcOptions {
    fn set_bool(&mut self, key: &str, v: bool) -> Result<(), KError> {
        match key {
            "pc_amg" => {
                if v {
                    self.pc_type = Some("amg".to_string());
                }
                Ok(())
            }
            "pc_amg_keep_transpose" => set_opt!(&mut self.amg_keep_transpose, v),
            "pc_amg_keep_pivot_in_rap" => set_opt!(&mut self.amg_keep_pivot_in_rap, v),
            "pc_amg_require_spd" => set_opt!(&mut self.amg_require_spd, v),
            "pc_amg_print_setup" => set_opt!(&mut self.amg_print_setup, v),
            "pc_amg_dist_instrumentation" => set_opt!(&mut self.amg_dist_instrumentation, v),
            "pc_amg_ieee_checks" => set_opt!(&mut self.amg_ieee_checks, v),
            "pc_amg_optimize_workspace" => set_opt!(&mut self.amg_optimize_workspace, v),
            "pc_sor_symmetric" => set_opt!(&mut self.sor_symmetric, v),
            "pc_ilu_ieee_checks" => set_opt!(&mut self.ilu_ieee_checks, v),
            "pc_ilu_pivot_monitoring" => set_opt!(&mut self.ilu_pivot_monitoring, v),
            "pc_ilu_optimize_workspace" => set_opt!(&mut self.ilu_optimize_workspace, v),
            "pc_ilu_parallel_factorization" => {
                set_opt!(&mut self.ilu_parallel_factorization, v)
            }
            "pc_ilu_parallel_trisolve" => {
                set_opt!(&mut self.ilu_parallel_triangular_solve, v)
            }
            "pc_ilu_distributed" => set_opt!(&mut self.ilu_distributed, v),
            "pc_superlu_replace_tiny_pivot" => set_opt!(&mut self.superlu_replace_tiny_pivots, v),
            "pc_superlu_static_pivoting" => set_opt!(&mut self.superlu_static_pivoting, v),
            "pc_superlu_enable_3d_factorization" => {
                set_opt!(&mut self.superlu_enable_3d_factorization, v)
            }
            "pc_superlu_async_panel_updates" => set_opt!(&mut self.superlu_async_panel_updates, v),
            "pc_superlu_aggressive_memory_reuse" => {
                set_opt!(&mut self.superlu_aggressive_memory_reuse, v)
            }
            // Approximate inverse (CSR) options
            "pc_approxinv_parallel" => set_opt!(&mut self.approxinv_parallel, v),
            "pc_fixdiag" => set_opt!(&mut self.pc_fixdiag, v),
            _ => Err(KError::SolveError(format!("Unknown PC bool key: {key}"))),
        }
    }

    fn set_val(&mut self, spec: &Spec, v: &str) -> Result<(), KError> {
        match spec.key {
            "pc_type" => set_opt!(&mut self.pc_type, v.to_string()),
            "pc_global" => set_opt!(&mut self.pc_global, v.to_lowercase()),
            "pc_local" => set_opt!(&mut self.pc_local, v.to_lowercase()),
            "pc_ilu_levels" => set_opt!(&mut self.ilu_level, parse_as::<usize>(v, spec)?),
            "pc_chebyshev_degree" => {
                set_opt!(&mut self.chebyshev_degree, parse_as::<usize>(v, spec)?)
            }
            "pc_ilut_drop_tol" => set_opt!(&mut self.ilut_drop_tol, parse_as::<f64>(v, spec)?),
            "pc_ilut_max_fill" => set_opt!(&mut self.ilut_max_fill, parse_as::<usize>(v, spec)?),
            "pc_ilut_perm_tol" => set_opt!(&mut self.ilut_perm_tol, parse_as::<f64>(v, spec)?),
            "pc_reorder" => set_opt!(&mut self.reorder, v.to_lowercase()),
            "pc_scaling" => set_opt!(&mut self.scaling, v.to_lowercase()),
            "pc_asm_overlap" => set_opt!(&mut self.asm_overlap, parse_as::<usize>(v, spec)?),
            "pc_asm_mode" => set_opt!(&mut self.asm_mode, v.to_lowercase()),
            "pc_asm_weighting" => set_opt!(&mut self.asm_weighting, v.to_lowercase()),
            "pc_asm_block_solver" => set_opt!(&mut self.asm_block_solver, v.to_lowercase()),
            "pc_asm_subdomains" => {
                let parsed: Result<Vec<usize>, _> =
                    v.split(',').map(|s| s.trim().parse()).collect();
                match parsed {
                    Ok(vv) => set_opt!(&mut self.asm_subdomains, vv),
                    Err(_) => Err(KError::SolveError(format!(
                        "Invalid {} value: {}. Use comma-separated usize list",
                        spec.flag, v
                    ))),
                }
            }
            "pc_asm_inner_pc" => set_opt!(&mut self.asm_inner_pc, v.to_lowercase()),
            "pc_chebyshev_lambda_min" => {
                set_opt!(&mut self.chebyshev_lambda_min, parse_as::<f64>(v, spec)?)
            }
            "pc_chebyshev_lambda_max" => {
                set_opt!(&mut self.chebyshev_lambda_max, parse_as::<f64>(v, spec)?)
            }
            "pc_amg_levels" => set_opt!(&mut self.amg_levels, parse_as::<usize>(v, spec)?),
            "pc_amg_strength_threshold" => {
                set_opt!(&mut self.amg_strength_threshold, parse_as::<f64>(v, spec)?)
            }
            "pc_amg_nu_pre" => set_opt!(&mut self.amg_nu_pre, parse_as::<usize>(v, spec)?),
            "pc_amg_nu_post" => set_opt!(&mut self.amg_nu_post, parse_as::<usize>(v, spec)?),
            "pc_amg_coarse_threshold" => {
                set_opt!(&mut self.amg_coarse_threshold, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_max_coarse_size" => {
                set_opt!(&mut self.amg_max_coarse_size, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_min_coarse_size" => {
                set_opt!(&mut self.amg_min_coarse_size, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_truncation_factor" => {
                set_opt!(&mut self.amg_truncation_factor, parse_as::<f64>(v, spec)?)
            }
            "pc_amg_rap_truncation_factor" => {
                set_opt!(&mut self.amg_truncation_factor, parse_as::<f64>(v, spec)?)
            }
            "pc_amg_max_elements_per_row" => set_opt!(
                &mut self.amg_max_elements_per_row,
                parse_as::<usize>(v, spec)?
            ),
            "pc_amg_interp_maxnnz" => set_opt!(
                &mut self.amg_max_elements_per_row,
                parse_as::<usize>(v, spec)?
            ),
            "pc_amg_interpolation_truncation" => set_opt!(
                &mut self.amg_interpolation_truncation,
                parse_as::<f64>(v, spec)?
            ),
            "pc_amg_rap_truncation_abs" => {
                set_opt!(&mut self.amg_rap_truncation_abs, parse_as::<f64>(v, spec)?)
            }
            "pc_amg_rap_maxnnz" => set_opt!(
                &mut self.amg_rap_max_elements_per_row,
                parse_as::<usize>(v, spec)?
            ),
            "pc_amg_coarsen_type" => set_opt!(&mut self.amg_coarsen_type, v.to_lowercase()),
            "pc_amg_coarsen" => set_opt!(&mut self.amg_coarsen_type, v.to_lowercase()),
            "pc_amg_interp_type" => set_opt!(&mut self.amg_interp_type, v.to_lowercase()),
            "pc_amg_interp" => set_opt!(&mut self.amg_interp_type, v.to_lowercase()),
            "pc_amg_relax_type" => set_opt!(&mut self.amg_relax_type, v.to_lowercase()),
            "pc_amg_smoother" => set_opt!(&mut self.amg_smoother, v.to_lowercase()),
            "pc_amg_smoother_steps" => {
                set_opt!(
                    &mut self.amg_smoother_steps,
                    ensure_ge_1("pc_amg_smoother_steps", parse_as::<usize>(v, spec)?)?
                )
            }
            "pc_amg_smoother_omega" => {
                set_opt!(&mut self.amg_smoother_omega, parse_as::<f64>(v, spec)?)
            }
            "pc_amg_logging_level" => {
                set_opt!(&mut self.amg_logging_level, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_print_level" => {
                set_opt!(&mut self.amg_print_level, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_dist_apply_mode" => {
                set_opt!(&mut self.amg_dist_apply_mode, v.to_lowercase())
            }
            "pc_amg_dist_coarse_ghost_scale" => {
                set_opt!(&mut self.amg_dist_coarse_ghost_scale, parse_as::<f64>(v, spec)?)
            }
            "pc_amg_tolerance" => set_opt!(&mut self.amg_tolerance, parse_as::<f64>(v, spec)?),
            "pc_amg_max_iterations" => {
                set_opt!(&mut self.amg_max_iterations, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_min_iterations" => {
                set_opt!(&mut self.amg_min_iterations, parse_as::<usize>(v, spec)?)
            }
            "pc_sor_omega" => set_opt!(&mut self.sor_omega, parse_as::<f64>(v, spec)?),
            "pc_sor_sweeps" => set_opt!(
                &mut self.sor_sweeps,
                ensure_ge_1("pc_sor_sweeps", parse_as::<usize>(v, spec)?)?
            ),
            "pc_sor_mat_side" => {
                kinds::SorMatSideKind::from_str(v)?;
                set_opt!(&mut self.sor_mat_side, v.to_lowercase())
            }
            "pc_chain" => set_opt!(&mut self.pc_chain, v.to_string()),
            "pc_shift_diag" => set_opt!(&mut self.pc_shift_diag, parse_as::<f64>(v, spec)?),
            "pc_diag_inject_tau" => {
                set_opt!(&mut self.pc_diag_inject_tau, parse_as::<f64>(v, spec)?)
            }
            "pc_scale" => set_opt!(&mut self.pc_scale, v.to_lowercase()),
            "pc_scale_norm" => set_opt!(&mut self.pc_scale_norm, v.to_lowercase()),
            "pc_jacobi_block_size" => {
                let val = ensure_ge_1("pc_jacobi_block_size", parse_as::<usize>(v, spec)?)?;
                set_opt!(&mut self.jacobi_block_size, val)
            }
            "pc_ilu_type" => {
                set_opt!(&mut self.ilu_type, v.to_lowercase())?;
                self.update_ilu_kind();
                Ok(())
            }
            "pc_ilu_level_of_fill" => {
                let val = parse_as::<usize>(v, spec)?;
                set_opt!(&mut self.ilu_level_of_fill, val)?;
                self.update_ilu_kind();
                Ok(())
            }
            "pc_ilu_max_fill_per_row" => {
                let val = parse_as::<usize>(v, spec)?;
                set_opt!(&mut self.ilu_max_fill_per_row, val)?;
                self.update_ilu_kind();
                Ok(())
            }
            "pc_ilu_offdiag_drop_tolerance" => {
                let val = parse_as::<f64>(v, spec)?;
                set_opt!(&mut self.ilu_offdiag_drop_tolerance, val)?;
                self.update_ilu_kind();
                Ok(())
            }
            "pc_ilu_schur_drop_tolerance" => {
                let val = parse_as::<f64>(v, spec)?;
                set_opt!(&mut self.ilu_schur_drop_tolerance, val)?;
                self.update_ilu_schur();
                Ok(())
            }
            "pc_ilu_reordering_type" => {
                set_opt!(&mut self.ilu_reordering_type, v.to_lowercase())?;
                self.update_ilu_reordering();
                Ok(())
            }
            "pc_ilu_triangular_solve" => {
                set_opt!(&mut self.ilu_triangular_solve, v.to_lowercase())?;
                self.update_ilu_tri_solve();
                Ok(())
            }
            "pc_ilu_lower_jacobi_iters" => {
                let val = parse_as::<usize>(v, spec)?;
                set_opt!(&mut self.ilu_lower_jacobi_iters, val)?;
                self.update_ilu_tri_solve();
                Ok(())
            }
            "pc_ilu_upper_jacobi_iters" => {
                let val = parse_as::<usize>(v, spec)?;
                set_opt!(&mut self.ilu_upper_jacobi_iters, val)?;
                self.update_ilu_tri_solve();
                Ok(())
            }
            "pc_ilu_tolerance" => {
                let val = parse_as::<f64>(v, spec)?;
                set_opt!(&mut self.ilu_tolerance, val)?;
                self.update_ilu_iterative();
                Ok(())
            }
            "pc_ilu_max_iterations" => {
                let val = parse_as::<usize>(v, spec)?;
                set_opt!(&mut self.ilu_max_iterations, val)?;
                self.update_ilu_iterative();
                Ok(())
            }
            "pc_ilu_logging_level" => {
                let val = parse_as::<usize>(v, spec)?;
                set_opt!(&mut self.ilu_logging_level, val)?;
                self.update_ilu_logging();
                Ok(())
            }
            "pc_ilu_print_level" => {
                set_opt!(&mut self.ilu_print_level, parse_as::<usize>(v, spec)?)
            }
            "pc_ilu_pivot_threshold" => {
                let val = parse_as::<f64>(v, spec)?;
                set_opt!(&mut self.ilu_pivot_threshold, val)?;
                self.update_ilu_pivot();
                Ok(())
            }
            "pc_ilu_par_factor" => {
                let value = v.to_lowercase();
                kinds::IluParFactorKind::from_str(&value)?;
                set_opt!(&mut self.ilu_par_factor, value)?;
                self.update_ilu_parallel_mode();
                Ok(())
            }
            "pc_ilu_parallel_chunk_size" => {
                let val = ensure_ge_1("pc_ilu_parallel_chunk_size", parse_as::<usize>(v, spec)?)?;
                set_opt!(&mut self.ilu_parallel_chunk_size, val)
            }
            "pc_ilu_parilu_iters" => {
                let val = ensure_ge_1("pc_ilu_parilu_iters", parse_as::<usize>(v, spec)?)?;
                set_opt!(&mut self.ilu_parilu_max_iters, val)
            }
            "pc_ilu_parilu_min_iters" => {
                let val = ensure_ge_1("pc_ilu_parilu_min_iters", parse_as::<usize>(v, spec)?)?;
                set_opt!(&mut self.ilu_parilu_min_iters, val)
            }
            "pc_ilu_parilu_tol" => {
                set_opt!(&mut self.ilu_parilu_tol, parse_as::<f64>(v, spec)?)
            }
            "pc_ilu_parilu_omega" => {
                set_opt!(&mut self.ilu_parilu_omega, parse_as::<f64>(v, spec)?)
            }
            "pc_ilu_block_size" => {
                let val = ensure_ge_1("pc_ilu_block_size", parse_as::<usize>(v, spec)?)?;
                set_opt!(&mut self.ilu_parallel_chunk_size, val)
            }
            "pc_ilu_pivot_mode" => set_opt!(&mut self.ilu_pivot_mode, v.to_lowercase()),
            "pc_ilu_pivot_scale" => set_opt!(&mut self.ilu_pivot_scale, v.to_lowercase()),
            "pc_ilu_pivot_tau" => set_opt!(&mut self.ilu_pivot_tau, parse_as::<f64>(v, spec)?),
            "pc_ilutp_max_fill" => {
                set_opt!(&mut self.ilutp_max_fill, parse_as::<usize>(v, spec)?)
            }
            "pc_ilutp_drop_tol" => {
                set_opt!(&mut self.ilutp_drop_tol, parse_as::<f64>(v, spec)?)
            }
            "pc_ilutp_perm_tol" => {
                set_opt!(&mut self.ilutp_perm_tol, parse_as::<f64>(v, spec)?)
            }
            "pc_superlu_pivot_threshold" => {
                set_opt!(&mut self.superlu_pivot_threshold, parse_as::<f64>(v, spec)?)
            }
            "pc_superlu_print_level" => {
                set_opt!(&mut self.superlu_print_level, parse_as::<u8>(v, spec)?)
            }
            "pc_superlu_column_permutation" => {
                set_opt!(&mut self.superlu_column_permutation, v.to_string())
            }
            "pc_superlu_row_permutation" => {
                set_opt!(&mut self.superlu_row_permutation, v.to_string())
            }
            "pc_superlu_iterative_refinement" => {
                set_opt!(&mut self.superlu_iterative_refinement, v.to_string())
            }
            "pc_superlu_panel_size" => {
                set_opt!(&mut self.superlu_panel_size, parse_as::<usize>(v, spec)?)
            }
            "pc_superlu_process_grid_3d_depth" => set_opt!(
                &mut self.superlu_process_grid_3d_depth,
                parse_as::<usize>(v, spec)?
            ),
            "pc_superlu_memory_tradeoff_factor" => set_opt!(
                &mut self.superlu_memory_tradeoff_factor,
                parse_as::<f64>(v, spec)?
            ),
            "pc_superlu_max_concurrent_panels" => set_opt!(
                &mut self.superlu_max_concurrent_panels,
                parse_as::<usize>(v, spec)?
            ),
            "pc_superlu_workspace_memory_limit" => set_opt!(
                &mut self.superlu_workspace_memory_limit,
                parse_as::<usize>(v, spec)?
            ),
            "pc_superlu_preallocation_strategy" => {
                set_opt!(&mut self.superlu_preallocation_strategy, v.to_lowercase())
            }
            "pc_reuse_policy" => set_opt!(&mut self.reuse_policy, v.to_string()),
            // Approximate inverse (CSR) numeric/structure controls
            "pc_approxinv_kind" => set_opt!(&mut self.approxinv_kind, v.to_lowercase()),
            "pc_approxinv_levels" => {
                set_opt!(&mut self.approxinv_levels, parse_as::<usize>(v, spec)?)
            }
            "pc_approxinv_max_per_col" => {
                set_opt!(&mut self.approxinv_max_per_col, parse_as::<usize>(v, spec)?)
            }
            "pc_approxinv_drop_tol" => {
                set_opt!(&mut self.approxinv_drop_tol, parse_as::<f64>(v, spec)?)
            }
            "pc_approxinv_reg" => set_opt!(&mut self.approxinv_reg, parse_as::<f64>(v, spec)?),
            "pc_approxinv_max_cond" => {
                set_opt!(&mut self.approxinv_max_cond, parse_as::<f64>(v, spec)?)
            }
            "options_file" => Ok(()), // consumed earlier
            _ => Err(KError::SolveError(format!("Unknown PC key: {}", spec.key))),
        }
    }

    fn set_pair(&mut self, spec: &Spec, a: &str, b: &str) -> Result<(), KError> {
        match spec.key {
            "pc_superlu_process_grid" => {
                let rows = parse_as::<usize>(a, spec)?;
                let cols = parse_as::<usize>(b, spec)?;
                set_opt!(&mut self.superlu_process_grid, (rows, cols))
            }
            _ => Err(KError::SolveError(format!(
                "Unknown PC pair key: {}",
                spec.key
            ))),
        }
    }
}

// ---- Public constructors / precedence ----

impl KspOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_args(args: &[&str]) -> Result<Self, KError> {
        let mut me = Self::default();
        let reg = registry();
        reg.parse_into(args, &mut me, Some("-ksp_"))?;
        // Also accept -matrix/-rhs (no "-ksp_" prefix)
        reg.parse_into(args, &mut me, Some("-m"))?;
        reg.parse_into(args, &mut me, Some("-r"))?;
        // normalize
        if let Some(ref side) = me.pc_side {
            PcSide::from_str(side)?; // just validate name; value is kept as string
        }
        Ok(me)
    }

    pub fn from_strings(args: &[String]) -> Result<Self, KError> {
        let v: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        Self::from_args(&v)
    }

    pub fn from_env() -> Result<Self, KError> {
        let mut me = Self::default();
        if let Ok(v) = std::env::var("KRYST_KSP_TYPE") {
            me.ksp_type = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_RTOL") {
            me.rtol = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_RTOL: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_KSP_ATOL") {
            me.atol = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_ATOL: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_KSP_DTOL") {
            me.dtol = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_DTOL: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_KSP_MAX_IT") {
            me.maxits = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_MAX_IT: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_KSP_REDUCTION") {
            me.reduction = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_RESTART") {
            let n: usize = v
                .parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_RESTART: {v}")))?;
            me.restart = Some(ensure_ge_1("KRYST_KSP_RESTART", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_GMRES_RESTART") {
            let n: usize = v
                .parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_GMRES_RESTART: {v}")))?;
            me.gmres_restart = Some(ensure_ge_1("KRYST_KSP_GMRES_RESTART", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_GMRES_VARIANT") {
            me.gmres_variant = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_FGMRES_RESTART") {
            let n: usize = v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_KSP_FGMRES_RESTART: {v}"))
            })?;
            me.fgmres_restart = Some(ensure_ge_1("KRYST_KSP_FGMRES_RESTART", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_PC_SIDE") {
            PcSide::from_str(&v)?;
            me.pc_side = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_MATRIX_FILE") {
            me.matrix_file = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_RHS_FILE") {
            me.rhs_file = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_MIN_ITER") {
            me.min_iter = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_MIN_ITER: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CF_TOL") {
            me.cf_tol = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_CF_TOL: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_KSP_SKIP_REAL_R_CHECK") {
            let l = v.to_lowercase();
            me.skip_real_r_check = Some(matches!(l.as_str(), "true" | "1" | "yes" | "on"));
        }
        if let Ok(v) = std::env::var("KRYST_KSP_EPSMAC") {
            me.epsmac = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_EPSMAC: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_KSP_GUARD_ZERO_RESIDUAL") {
            me.guard_zero_residual = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_KSP_GUARD_ZERO_RESIDUAL: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CG_NORM") {
            me.cg_norm = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CG_PIPELINED") {
            let l = v.to_lowercase();
            let pipelined = matches!(l.as_str(), "true" | "1" | "yes" | "on");
            me.cg_pipelined = Some(pipelined);
            me.cg_variant = Some(if pipelined {
                CgVariant::Pipelined
            } else {
                CgVariant::Classic
            });
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CG_VARIANT") {
            let variant = CgVariant::from_str(&v).map_err(|msg| {
                KError::SolveError(format!("Invalid KRYST_KSP_CG_VARIANT: {v} ({msg})"))
            })?;
            me.cg_variant = Some(variant);
            me.cg_pipelined = Some(matches!(variant, CgVariant::Pipelined));
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CG_REPLACE_EVERY") {
            me.cg_replace_every = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_KSP_CG_REPLACE_EVERY: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CG_SINGLE_REDUCTION") {
            let l = v.to_lowercase();
            me.cg_single_reduction = Some(matches!(l.as_str(), "true" | "1" | "yes" | "on"));
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CG_USE_ASYNC") {
            let l = v.to_lowercase();
            me.cg_use_async = Some(matches!(l.as_str(), "true" | "1" | "yes" | "on"));
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CG_ASYNC_MIN_N") {
            me.cg_async_min_n = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_KSP_CG_ASYNC_MIN_N: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_TRUST_REGION") {
            me.trust_region =
                Some(v.parse().map_err(|_| {
                    KError::SolveError(format!("Invalid KRYST_KSP_TRUST_REGION: {v}"))
                })?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_THREADS") {
            let n: usize = v
                .parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_THREADS: {v}")))?;
            me.threads = Some(ensure_ge_1("KRYST_KSP_THREADS", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_THREADS_MODE") {
            me.threads_mode = Some(v.to_lowercase());
        }
        if let Ok(v) = std::env::var("KRYST_KSP_MIN_LEN_VEC") {
            let n: usize = v
                .parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_MIN_LEN_VEC: {v}")))?;
            me.min_len_vec = Some(ensure_ge_1("KRYST_KSP_MIN_LEN_VEC", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_MIN_ROWS_SPMV") {
            let n: usize = v
                .parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_MIN_ROWS_SPMV: {v}")))?;
            me.min_rows_spmv = Some(ensure_ge_1("KRYST_KSP_MIN_ROWS_SPMV", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_KSP_CHUNK_ROWS_SPMV") {
            let n: usize = v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_KSP_CHUNK_ROWS_SPMV: {v}"))
            })?;
            me.chunk_rows_spmv = Some(ensure_ge_1("KRYST_KSP_CHUNK_ROWS_SPMV", n)?);
        }
        Ok(me)
    }

    pub fn overlay_from(&mut self, other: KspOptions) {
        macro_rules! o {
            ($f:ident) => {
                if other.$f.is_some() {
                    self.$f = other.$f;
                }
            };
        }

        o!(ksp_type);
        o!(rtol);
        o!(atol);
        o!(dtol);
        o!(maxits);
        o!(restart);
        o!(reduction);
        o!(ksp_monitor_rank0);
        o!(reproducible);
        o!(cg_variant);

        o!(gmres_restart);
        o!(gmres_orthog);
        o!(gmres_reorthog);
        o!(gmres_reorth);
        o!(gmres_reorth_tol);
        o!(gmres_happy_breakdown);
        o!(gmres_variant);
        o!(gmres_sstep);
        o!(gmres_sstep_max_cond);

        o!(fgmres_restart);
        o!(fgmres_orthog);
        o!(fgmres_reorthog);
        o!(fgmres_reorth);
        o!(fgmres_reorth_tol);
        o!(fgmres_happy_breakdown);
        o!(fgmres_variant);

        o!(pc_side);
        o!(matrix_file);
        o!(rhs_file);

        o!(min_iter);
        o!(cf_tol);
        o!(skip_real_r_check);
        o!(epsmac);
        o!(guard_zero_residual);

        o!(cg_norm);
        o!(cg_pipelined);
        o!(cg_replace_every);
        o!(cg_single_reduction);
        o!(cg_use_async);
        o!(cg_async_min_n);

        o!(trust_region);

        o!(threads);
        o!(threads_mode);
        o!(min_len_vec);
        o!(min_rows_spmv);
        o!(chunk_rows_spmv);
    }

    /// Configure the Rayon worker pool. Requires `feature="rayon"` to take effect.
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = Some(n);
        self
    }

    /// Toggle reproducible reductions for CG/PCG drivers.
    pub fn with_repro(mut self, reproducible: bool) -> Self {
        self.reproducible = Some(reproducible);
        self
    }

    /// Select the CG algorithm variant.
    pub fn with_cg_variant(mut self, variant: CgVariant) -> Self {
        self.cg_variant = Some(variant);
        self.cg_pipelined = Some(matches!(variant, CgVariant::Pipelined));
        self
    }

    /// Set the trust-region radius for CG solvers.
    pub fn with_trust_region(mut self, radius: f64) -> Self {
        self.trust_region = Some(radius);
        self
    }
}

impl KspOptions {
    /// Resolve the effective restart for a given solver type following precedence:
    /// - GMRES: gmres_restart -> restart
    /// - FGMRES: fgmres_restart -> restart
    /// - Others: restart
    pub fn effective_restart_for(&self, ksp_type: KspType) -> Option<usize> {
        match ksp_type {
            KspType::GMRES => self.gmres_restart.or(self.restart),
            KspType::FGMRES => self.fgmres_restart.or(self.restart),
            KspType::Other => self.restart,
        }
    }
}

impl PcOptions {
    fn update_ilu_kind(&mut self) {
        if let Some(ref ty) = self.ilu_type {
            match ty.as_str() {
                "ilu0" => self.ilu.kind = IluKind::ILU0,
                "iluk" => {
                    let k = self.ilu_level_of_fill.unwrap_or(0) as u32;
                    self.ilu.kind = IluKind::ILUK { k };
                }
                "ilut" => {
                    let droptol = self.ilu_offdiag_drop_tolerance.unwrap_or(0.0);
                    let max_fill = self.ilu_max_fill_per_row.unwrap_or(0) as u32;
                    self.ilu.kind = IluKind::ILUT {
                        droptol,
                        max_fill_per_row: max_fill,
                    };
                }
                _ => {}
            }
        }
    }

    fn update_ilu_reordering(&mut self) {
        if let Some(ref t) = self.ilu_reordering_type {
            self.ilu.reordering = match t.as_str() {
                "none" | "natural" => ReorderingType::None,
                "rcm" => ReorderingType::RCM,
                "amd" => ReorderingType::AMD,
                _ => self.ilu.reordering,
            };
        }
    }

    fn update_ilu_tri_solve(&mut self) {
        if let Some(ref t) = self.ilu_triangular_solve {
            if t == "iterative" {
                let l = self.ilu_lower_jacobi_iters.unwrap_or(0) as u32;
                let u = self.ilu_upper_jacobi_iters.unwrap_or(0) as u32;
                self.ilu.tri_solve.kind = TriSolveType::Iterative {
                    lower_jacobi_iters: l,
                    upper_jacobi_iters: u,
                };
            } else {
                self.ilu.tri_solve.kind = TriSolveType::Exact;
            }
        }
    }

    fn update_ilu_parallel_mode(&mut self) {
        if let Some(ref mode) = self.ilu_par_factor {
            let kind =
                kinds::IluParFactorKind::from_str(mode).unwrap_or(kinds::IluParFactorKind::Block);
            let enabled = matches!(
                kind,
                kinds::IluParFactorKind::Block | kinds::IluParFactorKind::ParILU
            );
            self.ilu_parallel_factorization = Some(enabled);
        }
    }

    fn update_ilu_iterative(&mut self) {
        if let Some(tol) = self.ilu_tolerance {
            self.ilu.iterative_setup.tol = tol;
            if matches!(self.ilu.iterative_setup.ty, IterativeSetupType::Disabled) {
                self.ilu.iterative_setup.ty = IterativeSetupType::ParILUFixedPoint;
            }
        }
        if let Some(maxit) = self.ilu_max_iterations {
            self.ilu.iterative_setup.max_iter = maxit as u32;
            if matches!(self.ilu.iterative_setup.ty, IterativeSetupType::Disabled) {
                self.ilu.iterative_setup.ty = IterativeSetupType::ParILUFixedPoint;
            }
        }
    }

    fn update_ilu_logging(&mut self) {
        if let Some(level) = self.ilu_logging_level {
            self.ilu.logging_level = level as u8;
        }
    }

    fn update_ilu_parilu(&mut self) {
        if let Some(iters) = self.ilu_parilu_max_iters {
            self.ilu.iterative_setup.max_iter = iters as u32;
            if matches!(self.ilu.iterative_setup.ty, IterativeSetupType::Disabled) {
                self.ilu.iterative_setup.ty = IterativeSetupType::ParILUFixedPoint;
            }
        }
        if let Some(min_iters) = self.ilu_parilu_min_iters {
            self.ilu.iterative_setup.min_iter = min_iters as u32;
            if matches!(self.ilu.iterative_setup.ty, IterativeSetupType::Disabled) {
                self.ilu.iterative_setup.ty = IterativeSetupType::ParILUFixedPoint;
            }
        }
        if let Some(tol) = self.ilu_parilu_tol {
            self.ilu.iterative_setup.tol = tol;
            if matches!(self.ilu.iterative_setup.ty, IterativeSetupType::Disabled) {
                self.ilu.iterative_setup.ty = IterativeSetupType::ParILUFixedPoint;
            }
        }
        if let Some(omega) = self.ilu_parilu_omega {
            self.ilu.iterative_setup.omega = omega;
            if matches!(self.ilu.iterative_setup.ty, IterativeSetupType::Disabled) {
                self.ilu.iterative_setup.ty = IterativeSetupType::ParILUFixedPoint;
            }
        }
    }

    fn update_ilu_pivot(&mut self) {
        if let Some(tau) = self.ilu_pivot_threshold {
            self.ilu.pivot = PivotPolicy::Threshold { tau };
        }
    }

    fn update_ilu_schur(&mut self) {
        if let Some(dt) = self.ilu_schur_drop_tolerance {
            self.ilu.schur.droptol_s = dt;
        }
    }

    fn sync_ilu_all(&mut self) {
        self.update_ilu_kind();
        self.update_ilu_reordering();
        self.update_ilu_tri_solve();
        self.update_ilu_parallel_mode();
        self.update_ilu_iterative();
        self.update_ilu_parilu();
        self.update_ilu_logging();
        self.update_ilu_pivot();
        self.update_ilu_schur();
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_args(args: &[&str]) -> Result<Self, KError> {
        let mut me = Self::default();
        registry().parse_into(args, &mut me, Some("-pc_"))?;
        // enum validations (centralized)
        if let Some(ref t) = me.reorder {
            kinds::ReorderKind::from_str(t)?;
        }
        if let Some(ref s) = me.scaling {
            kinds::ScalingKind::from_str(s)?;
        }
        if let Some(ref t) = me.ilu_type {
            kinds::IluTypeKind::from_str(t)?;
        }
        if let Some(ref t) = me.ilu_reordering_type {
            kinds::IluReorderKind::from_str(t)?;
        }
        if let Some(ref t) = me.ilu_triangular_solve {
            kinds::IluTriSolveKind::from_str(t)?;
        }
        if let Some(ref t) = me.amg_coarsen_type {
            kinds::AmgCoarsenKind::from_str(t)?;
        }
        if let Some(ref t) = me.amg_interp_type {
            kinds::AmgInterpKind::from_str(t)?;
        }
        if let Some(ref t) = me.amg_relax_type {
            kinds::AmgRelaxKind::from_str(t)?;
        }
        if let Some(ref t) = me.amg_smoother {
            kinds::AmgRelaxKind::from_str(t)?;
        }
        if let Some(ref t) = me.asm_block_solver {
            kinds::AsmBlockSolverKind::from_str(t)?;
        }
        if let Some(ref t) = me.asm_mode {
            kinds::AsmModeKind::from_str(t)?;
        }
        if let Some(ref t) = me.sor_mat_side {
            kinds::SorMatSideKind::from_str(t)?;
        }
        me.sync_ilu_all();
        Ok(me)
    }

    pub fn from_strings(args: &[String]) -> Result<Self, KError> {
        let v: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        Self::from_args(&v)
    }

    pub fn from_env() -> Result<Self, KError> {
        let mut me = Self::default();
        if let Ok(v) = std::env::var("KRYST_PC_TYPE") {
            me.pc_type = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_PC_GLOBAL") {
            me.pc_global = Some(v.to_lowercase());
        }
        if let Ok(v) = std::env::var("KRYST_PC_LOCAL") {
            me.pc_local = Some(v.to_lowercase());
        }
        if let Ok(v) = std::env::var("KRYST_PC_JACOBI_BLOCK_SIZE") {
            me.jacobi_block_size = Some(v.parse().map_err(|_| {
                KError::SolveError(format!(
                    "Invalid KRYST_PC_JACOBI_BLOCK_SIZE: {v}"
                ))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_LEVELS") {
            me.ilu_level =
                Some(v.parse().map_err(|_| {
                    KError::SolveError(format!("Invalid KRYST_PC_ILU_LEVELS: {v}"))
                })?);
        }
        if let Some(v) = env_lower("KRYST_PC_ILU_TYPE") {
            me.ilu_type = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_LEVEL_OF_FILL") {
            me.ilu_level_of_fill = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_LEVEL_OF_FILL: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_MAX_FILL_PER_ROW") {
            me.ilu_max_fill_per_row = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_MAX_FILL_PER_ROW: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_OFFDIAG_DROP_TOL") {
            me.ilu_offdiag_drop_tolerance = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_OFFDIAG_DROP_TOL: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_SCHUR_DROP_TOL") {
            me.ilu_schur_drop_tolerance = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_SCHUR_DROP_TOL: {v}"))
            })?);
        }
        if let Some(v) = env_lower("KRYST_PC_ILU_REORDERING_TYPE") {
            me.ilu_reordering_type = Some(v);
        }
        if let Some(v) = env_lower("KRYST_PC_ILU_TRI_SOLVE") {
            me.ilu_triangular_solve = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_LOWER_JACOBI_ITERS") {
            me.ilu_lower_jacobi_iters = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_LOWER_JACOBI_ITERS: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_UPPER_JACOBI_ITERS") {
            me.ilu_upper_jacobi_iters = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_UPPER_JACOBI_ITERS: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_TOL") {
            me.ilu_tolerance = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_ILU_TOL: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_MAX_ITER") {
            let n: usize = v
                .parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_ILU_MAX_ITER: {v}")))?;
            me.ilu_max_iterations = Some(ensure_ge_1("KRYST_PC_ILU_MAX_ITER", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_PARILU_ITERS") {
            let n: usize = v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_PARILU_ITERS: {v}"))
            })?;
            me.ilu_parilu_max_iters = Some(ensure_ge_1("KRYST_PC_ILU_PARILU_ITERS", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_PARILU_MIN_ITERS") {
            let n: usize = v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_PARILU_MIN_ITERS: {v}"))
            })?;
            me.ilu_parilu_min_iters = Some(ensure_ge_1("KRYST_PC_ILU_PARILU_MIN_ITERS", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_PARILU_TOL") {
            me.ilu_parilu_tol = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_PARILU_TOL: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_PARILU_OMEGA") {
            me.ilu_parilu_omega = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_PARILU_OMEGA: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_CHEBYSHEV_DEGREE") {
            me.chebyshev_degree = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_CHEBYSHEV_DEGREE: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILUT_DROP_TOL") {
            me.ilut_drop_tol =
                Some(v.parse().map_err(|_| {
                    KError::SolveError(format!("Invalid KRYST_PC_ILUT_DROP_TOL: {v}"))
                })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILUT_MAX_FILL") {
            me.ilut_max_fill =
                Some(v.parse().map_err(|_| {
                    KError::SolveError(format!("Invalid KRYST_PC_ILUT_MAX_FILL: {v}"))
                })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILUT_PERM_TOL") {
            me.ilut_perm_tol =
                Some(v.parse().map_err(|_| {
                    KError::SolveError(format!("Invalid KRYST_PC_ILUT_PERM_TOL: {v}"))
                })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_REORDER") {
            me.reorder = Some(v.to_lowercase());
        }
        if let Ok(v) = std::env::var("KRYST_PC_SCALING") {
            me.scaling = Some(v.to_lowercase());
        }
        if let Ok(v) = std::env::var("KRYST_PC_FIXDIAG") {
            let l = v.to_lowercase();
            me.pc_fixdiag = Some(matches!(l.as_str(), "true" | "1" | "yes" | "on"));
        }
        if let Ok(v) = std::env::var("KRYST_PC_SHIFT_DIAG") {
            me.pc_shift_diag = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_SHIFT_DIAG: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_DIAG_INJECT_TAU") {
            me.pc_diag_inject_tau = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_DIAG_INJECT_TAU: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_SCALE") {
            me.pc_scale = Some(v.to_lowercase());
        }
        if let Ok(v) = std::env::var("KRYST_PC_SCALE_NORM") {
            me.pc_scale_norm = Some(v.to_lowercase());
        }
        if let Ok(v) = std::env::var("KRYST_PC_SOR_OMEGA") {
            me.sor_omega = Some(
                v.parse()
                    .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_SOR_OMEGA: {v}")))?,
            );
        }
        if let Ok(v) = std::env::var("KRYST_PC_SOR_SWEEPS") {
            let n: usize = v
                .parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_SOR_SWEEPS: {v}")))?;
            me.sor_sweeps = Some(ensure_ge_1("KRYST_PC_SOR_SWEEPS", n)?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_SOR_SYMMETRIC") {
            let l = v.to_lowercase();
            me.sor_symmetric = Some(matches!(l.as_str(), "true" | "1" | "yes" | "on"));
        }
        if let Ok(v) = std::env::var("KRYST_PC_SOR_MAT_SIDE") {
            kinds::SorMatSideKind::from_str(&v)?;
            me.sor_mat_side = Some(v.to_lowercase());
        }
        if let Some(v) = env_lower("KRYST_PC_ILU_PIVOT_MODE") {
            me.ilu_pivot_mode = Some(v);
        }
        if let Some(v) = env_lower("KRYST_PC_ILU_PIVOT_SCALE") {
            me.ilu_pivot_scale = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_PIVOT_TAU") {
            me.ilu_pivot_tau =
                Some(v.parse().map_err(|_| {
                    KError::SolveError(format!("Invalid KRYST_PC_ILU_PIVOT_TAU: {v}"))
                })?);
        }
        if let Some(v) = env_bool("KRYST_PC_ILU_PARALLEL_FACTORIZATION") {
            me.ilu_parallel_factorization = Some(v);
        }
        if let Some(v) = env_bool("KRYST_PC_ILU_PARALLEL_TRISOLVE") {
            me.ilu_parallel_triangular_solve = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_PARALLEL_CHUNK_SIZE") {
            let n: usize = v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILU_PARALLEL_CHUNK_SIZE: {v}"))
            })?;
            me.ilu_parallel_chunk_size = Some(ensure_ge_1("KRYST_PC_ILU_PARALLEL_CHUNK_SIZE", n)?);
        }
        if let Some(v) = env_lower("KRYST_PC_ILU_PAR_FACTOR") {
            kinds::IluParFactorKind::from_str(&v)?;
            me.ilu_par_factor = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILU_BLOCK_SIZE") {
            let n: usize = v
                .parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_ILU_BLOCK_SIZE: {v}")))?;
            me.ilu_parallel_chunk_size = Some(ensure_ge_1("KRYST_PC_ILU_BLOCK_SIZE", n)?);
        }
        if let Some(v) = env_bool("KRYST_PC_ILU_DISTRIBUTED") {
            me.ilu_distributed = Some(v);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILUTP_MAX_FILL") {
            me.ilutp_max_fill = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILUTP_MAX_FILL: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILUTP_DROP_TOL") {
            me.ilutp_drop_tol = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILUTP_DROP_TOL: {v}"))
            })?);
        }
        if let Ok(v) = std::env::var("KRYST_PC_ILUTP_PERM_TOL") {
            me.ilutp_perm_tol = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_PC_ILUTP_PERM_TOL: {v}"))
            })?);
        }
        me.sync_ilu_all();
        Ok(me)
    }

    pub fn overlay_from(&mut self, other: PcOptions) {
        macro_rules! o {
            ($f:ident) => {
                if other.$f.is_some() {
                    self.$f = other.$f;
                }
            };
        }

        o!(pc_type);
        o!(ilu_level);
        o!(chebyshev_degree);
        o!(ilut_drop_tol);
        o!(ilut_max_fill);
        o!(ilut_perm_tol);
        o!(ilutp_max_fill);
        o!(ilutp_drop_tol);
        o!(ilutp_perm_tol);
        o!(reorder);
        o!(scaling);

        o!(asm_overlap);
        o!(asm_mode);
        o!(asm_weighting);
        o!(asm_subdomains);
        o!(asm_inner_pc);
        o!(asm_block_solver);
        o!(asm_subdomain_size);

        o!(chebyshev_lambda_min);
        o!(chebyshev_lambda_max);

        o!(amg_levels);
        o!(amg_strength_threshold);
        o!(amg_nu_pre);
        o!(amg_nu_post);
        o!(amg_coarse_threshold);
        o!(amg_max_coarse_size);
        o!(amg_min_coarse_size);
        o!(amg_truncation_factor);
        o!(amg_max_elements_per_row);
        o!(amg_interpolation_truncation);
        o!(amg_coarsen_type);
        o!(amg_interp_type);
        o!(amg_relax_type);
        o!(amg_smoother);
        o!(amg_logging_level);
        o!(amg_print_level);
        o!(amg_tolerance);
        o!(amg_max_iterations);
        o!(amg_min_iterations);
        o!(amg_ieee_checks);
        o!(amg_optimize_workspace);
        o!(amg_smoother_steps);
        o!(amg_smoother_omega);
        o!(amg_rap_truncation_abs);
        o!(amg_rap_max_elements_per_row);
        o!(amg_keep_transpose);
        o!(amg_keep_pivot_in_rap);
        o!(amg_require_spd);
        o!(amg_print_setup);
        o!(amg_dist_apply_mode);
        o!(amg_dist_instrumentation);
        o!(amg_dist_coarse_ghost_scale);

        o!(pc_chain);
        o!(chain);
        o!(pc_fixdiag);
        o!(pc_shift_diag);
        o!(pc_diag_inject_tau);
        o!(pc_scale);
        o!(pc_scale_norm);

        o!(sor_omega);
        o!(sor_sweeps);
        o!(sor_symmetric);
        o!(sor_mat_side);

        o!(omega);
        o!(drop_tol);
        o!(reuse_policy);

        o!(approxinv_kind);
        o!(approxinv_levels);
        o!(approxinv_max_per_col);
        o!(approxinv_drop_tol);
        o!(approxinv_reg);
        o!(approxinv_max_cond);
        o!(approxinv_parallel);

        o!(ilu_type);
        o!(ilu_level_of_fill);
        o!(ilu_max_fill_per_row);
        o!(ilu_offdiag_drop_tolerance);
        o!(ilu_schur_drop_tolerance);
        o!(ilu_reordering_type);
        o!(ilu_triangular_solve);
        o!(ilu_lower_jacobi_iters);
        o!(ilu_upper_jacobi_iters);
        o!(ilu_tolerance);
        o!(ilu_max_iterations);
        o!(ilu_logging_level);
        o!(ilu_print_level);
        o!(ilu_ieee_checks);
        o!(ilu_pivot_monitoring);
        o!(ilu_optimize_workspace);
        o!(ilu_pivot_threshold);

        o!(ilu_par_factor);
        o!(ilu_parilu_max_iters);
        o!(ilu_parilu_min_iters);
        o!(ilu_parilu_tol);
        o!(ilu_parilu_omega);

        o!(ilu_parallel_factorization);
        o!(ilu_parallel_triangular_solve);
        o!(ilu_parallel_chunk_size);
        o!(ilu_distributed);

        o!(ilu_pivot_mode);
        o!(ilu_pivot_scale);
        o!(ilu_pivot_tau);

        o!(superlu_pivot_threshold);
        o!(superlu_replace_tiny_pivots);
        o!(superlu_print_level);
        o!(superlu_process_grid);
        o!(superlu_column_permutation);
        o!(superlu_row_permutation);
        o!(superlu_iterative_refinement);
        o!(superlu_static_pivoting);
        o!(superlu_panel_size);
        o!(superlu_enable_3d_factorization);
        o!(superlu_process_grid_3d_depth);
        o!(superlu_memory_tradeoff_factor);
        o!(superlu_max_concurrent_panels);
        o!(superlu_async_panel_updates);
        o!(superlu_workspace_memory_limit);
        o!(superlu_aggressive_memory_reuse);
        o!(superlu_preallocation_strategy);

        o!(jacobi_block_size);
        o!(pc_global);
        o!(pc_local);
        o!(ilu_variant);
        o!(ilu_reordering);

        o!(cheb_degree);
        o!(cheb_eig_lo);
        o!(cheb_eig_hi);

        self.ilu = std::mem::take(&mut self.ilu).overlay(&other.ilu);

        // Keep derived/legacy ↔ structured in sync after overlay.
        self.sync_ilu_all();
    }
}

// ---- Combined parsing with precedence & generated help ----

pub fn help_text() -> String {
    let reg = registry();
    let mut out = String::new();
    out.push_str("Kryst Linear Solver Options\n\n");
    out.push_str("KSP options:\n");
    out.push_str(&reg.help_for_prefix("-ksp_"));
    out.push_str("General:\n");
    out.push_str(&reg.help_for_prefix("-m")); // includes -matrix
    out.push_str(&reg.help_for_prefix("-r")); // includes -rhs
    out.push_str("PC options:\n");
    out.push_str(&reg.help_for_prefix("-pc_"));
    out.push_str("Utility:\n");
    out.push_str("  -options_file <path>              str     Read more options from file\n");
    out
}

pub fn print_help() {
    println!("{}", help_text());
}

/// CLI > options file(s) > env > defaults
pub fn parse_all_options(args: &[String]) -> Result<(KspOptions, PcOptions), KError> {
    let mut args = args.to_vec();

    // Centralized help check: library never exits; return typed signal
    let as_refs_help: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    if is_help_requested(&as_refs_help) {
        return Err(KError::HelpRequested(help_text()));
    }

    // expand options files into the token stream
    args = expand_options_files(args)
        .map_err(|e| KError::SolveError(format!("While expanding -options_file: {e}")))?;

    // start from environment
    let mut ksp_opts = KspOptions::from_env()?;
    let mut pc_opts = PcOptions::from_env()?;

    // parse CLI and override env/options-file (we just parse whole argv once per group)
    let as_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let cli_ksp = KspOptions::from_args(&as_refs)?;
    let cli_pc = PcOptions::from_args(&as_refs)?;

    ksp_opts.overlay_from(cli_ksp);
    pc_opts.overlay_from(cli_pc);

    Ok((ksp_opts, pc_opts))
}

impl PcOptions {
    pub fn conditioning_options(&self) -> Result<ConditioningOptions, KError> {
        let mut opts = ConditioningOptions::default();
        if let Some(v) = self.pc_fixdiag {
            opts.fix_diag = v;
        }
        if let Some(v) = self.pc_shift_diag {
            opts.shift_diag = Some(v);
        }
        if let Some(v) = self.pc_diag_inject_tau {
            opts.diag_inject_tau = Some(v);
        }
        if let Some(ref v) = self.pc_scale {
            opts.scale = Some(ScaleDirection::from_str(v)?);
        }
        if let Some(ref v) = self.pc_scale_norm {
            opts.scale_norm = ScaleNorm::from_str(v)?;
        }
        Ok(opts)
    }

    /// Convert CLI-style options into MPI-specific preconditioner configuration.
    pub fn mpi_pc_options(&self) -> Result<MpiPcOptions, KError> {
        let global = self
            .pc_global
            .as_deref()
            .map(GlobalPcKind::from_str)
            .transpose()?
            .unwrap_or(GlobalPcKind::None);
        let local = self
            .pc_local
            .as_deref()
            .map(LocalPcKind::from_str)
            .transpose()?
            .unwrap_or(LocalPcKind::Ilu);

        if let GlobalPcKind::Ras = global {
            if let Some(mode) = self.asm_mode.as_deref()
                && mode != "ras"
            {
                return Err(KError::InvalidInput(format!(
                    "pc_global=ras requires pc_asm_mode=ras (got {mode})"
                )));
            }
        }

        let mut opts = MpiPcOptions::default();
        opts.global_pc = global;
        opts.local_pc = local;
        opts.ilu_config = build_ilu_config(self)?;
        opts.conditioning = self.conditioning_options()?;

        opts.ilut_fill = self.ilut_max_fill.unwrap_or(opts.ilut_fill);
        if let Some(drop_tol) = self.ilut_drop_tol {
            opts.ilut_drop_tol = drop_tol;
        }
        if let Some(perm_tol) = self.ilut_perm_tol {
            opts.ilut_perm_tol = perm_tol;
        }

        opts.ilutp_max_fill = self.ilutp_max_fill.unwrap_or(opts.ilutp_max_fill);
        if let Some(drop_tol) = self.ilutp_drop_tol {
            opts.ilutp_drop_tol = drop_tol;
        }
        if let Some(perm_tol) = self.ilutp_perm_tol {
            opts.ilutp_perm_tol = perm_tol;
        }

        Ok(opts)
    }
}

fn build_ilu_config(opts: &PcOptions) -> Result<IluConfig, KError> {
    let mut config = IluConfig::default();

    if let Some(ref ty) = opts.ilu_type {
        config.ilu_type = parse_ilu_variant(ty)?;
    }
    if let Some(level) = opts.ilu_level_of_fill {
        config.level_of_fill = level;
    }
    if let Some(max_fill) = opts.ilu_max_fill_per_row {
        config.max_fill_per_row = max_fill;
    }
    if let Some(offdiag) = opts.ilu_offdiag_drop_tolerance {
        config.offdiag_drop_tolerance = offdiag;
    }
    if let Some(schur) = opts.ilu_schur_drop_tolerance {
        config.schur_drop_tolerance = schur;
    }
    if let Some(reordering) = opts.ilu_reordering_type.as_deref() {
        config.reordering_type = parse_ilu_reordering(reordering)?;
    }
    if let Some(tri) = opts.ilu_triangular_solve.as_deref() {
        config.triangular_solve = parse_ilu_tri_solve(tri)?;
    }
    if let Some(lower) = opts.ilu_lower_jacobi_iters {
        config.lower_jacobi_iters = lower;
    }
    if let Some(upper) = opts.ilu_upper_jacobi_iters {
        config.upper_jacobi_iters = upper;
    }
    if let Some(tol) = opts.ilu_tolerance {
        config.tolerance = tol;
    }
    if let Some(max_iter) = opts.ilu_max_iterations {
        config.max_iterations = max_iter;
    }
    if let Some(logging) = opts.ilu_logging_level {
        config.logging_level = logging;
    }
    if let Some(print) = opts.ilu_print_level {
        config.print_level = print;
    }
    if let Some(flag) = opts.ilu_ieee_checks {
        config.ieee_checks = flag;
    }
    if let Some(flag) = opts.ilu_optimize_workspace {
        config.optimize_workspace = flag;
    }
    if let Some(flag) = opts.ilu_parallel_factorization {
        config.enable_parallel_factorization = flag;
    }
    if let Some(flag) = opts.ilu_parallel_triangular_solve {
        config.enable_parallel_triangular_solve = flag;
    }
    if let Some(chunk) = opts.ilu_parallel_chunk_size {
        config.parallel_chunk_size = chunk;
    }
    if let Some(iters) = opts.ilu_parilu_max_iters {
        config.parilu_max_iters = iters;
    }
    if let Some(min_iters) = opts.ilu_parilu_min_iters {
        config.parilu_min_iters = min_iters;
    }
    if let Some(tol) = opts.ilu_parilu_tol {
        config.parilu_tol = tol;
    }
    if let Some(omega) = opts.ilu_parilu_omega {
        config.parilu_omega = omega;
    }

    config.conditioning = opts.conditioning_options()?;

    Ok(config)
}

fn parse_ilu_variant(value: &str) -> Result<IluVariant, KError> {
    match value.to_lowercase().as_str() {
        "ilu0" => Ok(IluVariant::ILU0),
        "iluk" => Ok(IluVariant::ILUK),
        "ilut" => Ok(IluVariant::ILUT),
        "milu0" => Ok(IluVariant::MILU0),
        other => Err(KError::InvalidInput(format!("Unknown ilu_type: {other}"))),
    }
}

fn parse_ilu_reordering(value: &str) -> Result<IluReorderingType, KError> {
    match value.to_lowercase().as_str() {
        "none" => Ok(IluReorderingType::None),
        "natural" => Ok(IluReorderingType::Natural),
        "rcm" => Ok(IluReorderingType::RCM),
        "amd" => Ok(IluReorderingType::AMD),
        other => Err(KError::InvalidInput(format!(
            "Unknown ilu_reordering_type: {other}"
        ))),
    }
}

fn parse_ilu_tri_solve(value: &str) -> Result<IluTriSolveType, KError> {
    match value.to_lowercase().as_str() {
        "exact" => Ok(IluTriSolveType::Exact),
        "jacobi" => Ok(IluTriSolveType::Jacobi),
        "gauss-seidel" | "gaussseidel" => Ok(IluTriSolveType::GaussSeidel),
        other => Err(KError::InvalidInput(format!(
            "Unknown ilu_triangular_solve: {other}"
        ))),
    }
}

// ---- tests: reuse your existing tests; only minor changes below ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::options_core::{Arity, ValueKind};
    use crate::config::registry::SPECS;
    use crate::error::KError;

    struct KrystEnvGuard {
        saved: Vec<(String, String)>,
    }

    impl KrystEnvGuard {
        fn clear() -> Self {
            let mut saved = Vec::new();
            for (k, v) in std::env::vars() {
                if k.starts_with("KRYST_") {
                    saved.push((k.clone(), v));
                    unsafe {
                        std::env::remove_var(k);
                    }
                }
            }
            Self { saved }
        }
    }

    impl Drop for KrystEnvGuard {
        fn drop(&mut self) {
            for (k, v) in self.saved.drain(..) {
                unsafe {
                    std::env::set_var(k, v);
                }
            }
        }
    }

    fn sample_args_for(spec: &Spec) -> Vec<String> {
        let mut out = vec![spec.flag.to_string()];
        match spec.arity {
            Arity::Zero | Arity::OptionalBool => {}
            Arity::One => {
                let v = match spec.key {
                    // KSP strings with “known good” values
                    "ksp_type" => "gmres",
                    "ksp_gmres_orthog" => "mgs",
                    "ksp_gmres_reorth" => "never",
                    "ksp_gmres_variant" => "classical",
                    "ksp_fgmres_variant" => "classical",
                    "ksp_reduction" => "fast",
                    "ksp_pc_side" => "left",
                    "matrix" => "A.mtx",
                    "rhs" => "b.vec",

                    // PC validated enums
                    "pc_type" => "amg",
                    "pc_reorder" => "amd",
                    "pc_scaling" => "diagonal",
                    "pc_asm_mode" => "asm",
                    "pc_asm_block_solver" => "ludense",
                    "pc_sor_mat_side" => "lower",
                    "pc_ilu_type" => "ilut",
                    "pc_ilu_reordering_type" => "rcm",
                    "pc_ilu_triangular_solve" => "exact",
                    "pc_ilu_par_factor" => "parilu",
                    "pc_amg_coarsen_type" => "rs",
                    "pc_amg_interp_type" => "classical",
                    "pc_amg_relax_type" => "jacobi",
                    "pc_amg_smoother" => "jacobi",
                    "pc_amg_coarsen" => "hmis",
                    "pc_amg_interp" => "extended",
                    "pc_amg_interp_maxnnz" => "8",
                    "pc_amg_rap_truncation_factor" => "0.05",
                    "pc_amg_rap_truncation_abs" => "0.0",
                    "pc_amg_rap_maxnnz" => "16",
                    "pc_amg_smoother_steps" => "2",
                    "pc_amg_smoother_omega" => "0.8",
                    "pc_asm_subdomains" => "0,1",

                    // Fallback by kind (must satisfy ensure_ge_1 where applicable)
                    _ => match spec.kind {
                        ValueKind::Bool => "true",
                        ValueKind::Int => "1",
                        ValueKind::UInt => "1",
                        ValueKind::Float => "0.1",
                        ValueKind::Str => "x",
                        ValueKind::Pair(_, _) => unreachable!(),
                    },
                };
                out.push(v.to_string());
            }
            Arity::Two => {
                out.push("1".to_string());
                out.push("1".to_string());
            }
        }
        out
    }

    #[test]
    fn ksp_bool_toggle() {
        let args = vec!["-ksp_skip_real_r_check"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.skip_real_r_check, Some(true));

        let args = vec!["-ksp_skip_real_r_check", "false"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.skip_real_r_check, Some(false));

        let args = vec!["-ksp_cg_pipelined"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.cg_pipelined, Some(true));
        assert_eq!(opts.cg_variant, Some(CgVariant::Pipelined));

        let args = vec!["-ksp_cg_pipelined", "false"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.cg_pipelined, Some(false));
        assert_eq!(opts.cg_variant, Some(CgVariant::Classic));

        let args = vec!["-ksp_cg_use_async"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.cg_use_async, Some(true));

        let args = vec!["-ksp_cg_use_async", "false"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.cg_use_async, Some(false));
    }

    #[test]
    fn help_is_caught_and_returned() {
        let args = vec!["--help".to_string()];
        let err = parse_all_options(&args).unwrap_err();
        match err {
            KError::HelpRequested(text) => assert!(text.contains("Kryst Linear Solver Options")),
            _ => panic!("expected HelpRequested"),
        }
    }

    #[test]
    fn ksp_cli_overrides_options_file_for_missing_fields() {
        // Helper: writes an options file with given content, then parses with CLI args appended
        fn parse_with_layers(file_body: &str, cli: &[&str]) -> KspOptions {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(tmp.path(), file_body).unwrap();
            let mut args: Vec<String> = vec![
                "-options_file".to_string(),
                tmp.path().to_str().unwrap().to_string(),
            ];
            args.extend(cli.iter().map(|s| s.to_string()));
            let (ksp, _pc) = parse_all_options(&args).unwrap();
            ksp
        }

        // GMRES
        let k = parse_with_layers("-ksp_gmres_restart 30\n", &["-ksp_gmres_restart", "60"]);
        assert_eq!(k.gmres_restart, Some(60));

        let k = parse_with_layers("-ksp_gmres_orthog cgs\n", &["-ksp_gmres_orthog", "mgs"]);
        assert_eq!(k.gmres_orthog.as_deref(), Some("mgs"));

        let k = parse_with_layers(
            "-ksp_gmres_reorthog true\n",
            &["-ksp_gmres_reorthog", "false"],
        );
        assert_eq!(k.gmres_reorthog, Some(false));

        let k = parse_with_layers(
            "-ksp_gmres_reorth ifneeded\n",
            &["-ksp_gmres_reorth", "never"],
        );
        assert_eq!(k.gmres_reorth.as_deref(), Some("never"));

        let k = parse_with_layers(
            "-ksp_gmres_reorth_tol 0.6\n",
            &["-ksp_gmres_reorth_tol", "0.25"],
        );
        assert_eq!(k.gmres_reorth_tol, Some(0.25));

        let k = parse_with_layers(
            "-ksp_gmres_happy_breakdown true\n",
            &["-ksp_gmres_happy_breakdown", "false"],
        );
        assert_eq!(k.gmres_happy_breakdown, Some(false));

        // FGMRES
        let k = parse_with_layers("-ksp_fgmres_restart 20\n", &["-ksp_fgmres_restart", "80"]);
        assert_eq!(k.fgmres_restart, Some(80));

        let k = parse_with_layers("-ksp_fgmres_orthog mgs\n", &["-ksp_fgmres_orthog", "cgs"]);
        assert_eq!(k.fgmres_orthog.as_deref(), Some("cgs"));

        let k = parse_with_layers(
            "-ksp_fgmres_reorthog true\n",
            &["-ksp_fgmres_reorthog", "false"],
        );
        assert_eq!(k.fgmres_reorthog, Some(false));

        let k = parse_with_layers(
            "-ksp_fgmres_reorth always\n",
            &["-ksp_fgmres_reorth", "ifneeded"],
        );
        assert_eq!(k.fgmres_reorth.as_deref(), Some("ifneeded"));

        let k = parse_with_layers(
            "-ksp_fgmres_reorth_tol 0.9\n",
            &["-ksp_fgmres_reorth_tol", "0.3"],
        );
        assert_eq!(k.fgmres_reorth_tol, Some(0.3));

        let k = parse_with_layers(
            "-ksp_fgmres_happy_breakdown false\n",
            &["-ksp_fgmres_happy_breakdown", "true"],
        );
        assert_eq!(k.fgmres_happy_breakdown, Some(true));

        // CG
        let k = parse_with_layers(
            "-ksp_cg_norm preconditioned\n",
            &["-ksp_cg_norm", "natural"],
        );
        assert_eq!(k.cg_norm.as_deref(), Some("natural"));

        let k = parse_with_layers(
            "-ksp_cg_single_reduction false\n",
            &["-ksp_cg_single_reduction", "true"],
        );
        assert_eq!(k.cg_single_reduction, Some(true));

        let k = parse_with_layers("-ksp_cg_pipelined true\n", &["-ksp_cg_pipelined", "false"]);
        assert_eq!(k.cg_pipelined, Some(false));
        assert_eq!(k.cg_variant, Some(CgVariant::Classic));

        let k = parse_with_layers(
            "-ksp_cg_async_min_n 5000\n",
            &["-ksp_cg_async_min_n", "2000"],
        );
        assert_eq!(k.cg_async_min_n, Some(2000));

        let k = parse_with_layers(
            "-ksp_cg_replace_every 20\n",
            &["-ksp_cg_replace_every", "10"],
        );
        assert_eq!(k.cg_replace_every, Some(10));

        // Trust region
        let k = parse_with_layers("-ksp_trust_region 0.5\n", &["-ksp_trust_region", "1.5"]);
        assert!(matches!(k.trust_region, Some(v) if (v - 1.5).abs() < 1e-12));
    }

    #[test]
    fn options_file_basic() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "-ksp_type gmres\n-ksp_rtol 1e-8\n").unwrap();
        let args = vec![
            "-options_file".to_string(),
            tmp.path().to_str().unwrap().to_string(),
            "-pc_type".to_string(),
            "jacobi".to_string(),
        ];
        let (ksp, pc) = parse_all_options(&args).unwrap();
        assert_eq!(ksp.ksp_type.as_deref(), Some("gmres"));
        assert_eq!(ksp.rtol, Some(1e-8));
        assert_eq!(pc.pc_type.as_deref(), Some("jacobi"));
    }

    #[test]
    fn nested_options_file_expands_recursively() {
        // tmpA includes tmpB; tmpB sets -pc_type amg
        let a = tempfile::NamedTempFile::new().unwrap();
        let b = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(b.path(), "-pc_type amg\n").unwrap();
        std::fs::write(a.path(), format!("-options_file {}\n", b.path().display())).unwrap();

        let args = vec![
            "-options_file".to_string(),
            a.path().to_string_lossy().into_owned(),
            "-pc_ilut_max_fill".to_string(),
            "20".to_string(),
        ];
        let (_ksp, pc) = parse_all_options(&args).unwrap();
        assert_eq!(pc.pc_type.as_deref(), Some("amg"));
        assert_eq!(pc.ilut_max_fill, Some(20)); // later CLI still overrides file order
    }

    #[test]
    fn options_file_cycle_is_reported() {
        // A includes B; B includes A
        let a = tempfile::NamedTempFile::new().unwrap();
        let b = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(a.path(), format!("-options_file {}\n", b.path().display())).unwrap();
        std::fs::write(b.path(), format!("-options_file {}\n", a.path().display())).unwrap();

        let args = vec![
            "-options_file".to_string(),
            a.path().to_string_lossy().into_owned(),
        ];
        let err = parse_all_options(&args).unwrap_err();
        match err {
            KError::SolveError(msg) => assert!(msg.to_lowercase().contains("cyclic")),
            _ => panic!("expected SolveError for cyclic include"),
        }
    }

    #[test]
    fn relative_path_is_resolved_against_including_file() {
        // create dirA/optsA that includes "nested/optsB" relative to dirA
        let dir = tempfile::tempdir().unwrap();
        let dir_a = dir.path().join("dirA");
        let nested = dir_a.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let a = dir_a.join("optsA");
        let b = nested.join("optsB");

        std::fs::write(&b, "-ksp_type gmres\n").unwrap();
        std::fs::write(&a, "  # comment\n -options_file nested/optsB \n").unwrap();

        let args = vec!["-options_file".to_string(), a.display().to_string()];
        let (ksp, _pc) = parse_all_options(&args).unwrap();
        assert_eq!(ksp.ksp_type.as_deref(), Some("gmres"));
    }

    #[test]
    fn every_spec_affects_parse_all_options() {
        let _guard = KrystEnvGuard::clear();

        let (k0, p0) = parse_all_options(&vec![]).unwrap();
        let baseline = format!("{k0:?}\n{p0:?}");

        for spec in SPECS {
            if spec.flag == "-options_file" {
                continue;
            }
            let args = sample_args_for(spec);
            let (k, p) = parse_all_options(&args)
                .unwrap_or_else(|e| panic!("Spec {} failed to parse/apply: {e:?}", spec.flag));
            let snapshot = format!("{k:?}\n{p:?}");
            assert_ne!(
                snapshot, baseline,
                "Spec {} parsed but had no effect (overlay missing or ignored)",
                spec.flag
            );
        }
    }
}

#[cfg(test)]
mod old_tests {
    use super::*;

    #[test]
    fn test_ksp_options_new() {
        let opts = KspOptions::new();
        assert!(opts.ksp_type.is_none());
        assert!(opts.rtol.is_none());
        assert!(opts.atol.is_none());
        assert!(opts.dtol.is_none());
        assert!(opts.maxits.is_none());
        assert!(opts.restart.is_none());
        assert!(opts.pc_side.is_none());
    }

    #[test]
    fn test_pc_options_new() {
        let opts = PcOptions::new();
        assert!(opts.pc_type.is_none());
        assert!(opts.ilu_level.is_none());
        assert!(opts.chebyshev_degree.is_none());
        assert!(opts.ilut_drop_tol.is_none());
        assert!(opts.ilut_max_fill.is_none());
        assert!(opts.omega.is_none());
        assert!(opts.drop_tol.is_none());
    }

    #[test]
    fn test_ksp_options_from_args_basic() {
        let args = vec!["-ksp_type", "gmres", "-ksp_rtol", "1e-8"];
        let opts = KspOptions::from_args(&args).unwrap();

        assert_eq!(opts.ksp_type, Some("gmres".to_string()));
        assert_eq!(opts.rtol, Some(1e-8));
        assert!(opts.atol.is_none());
    }

    #[test]
    fn test_ksp_options_from_args_all_options() {
        let args = vec![
            "-ksp_type",
            "cg",
            "-ksp_rtol",
            "1e-6",
            "-ksp_atol",
            "1e-12",
            "-ksp_dtol",
            "1e3",
            "-ksp_max_it",
            "1000",
            "-ksp_restart",
            "30",
            "-ksp_pc_side",
            "left",
        ];
        let opts = KspOptions::from_args(&args).unwrap();

        assert_eq!(opts.ksp_type, Some("cg".to_string()));
        assert_eq!(opts.rtol, Some(1e-6));
        assert_eq!(opts.atol, Some(1e-12));
        assert_eq!(opts.dtol, Some(1e3));
        assert_eq!(opts.maxits, Some(1000));
        assert_eq!(opts.restart, Some(30));
        assert_eq!(opts.pc_side, Some("left".to_string()));
    }

    #[test]
    fn test_ksp_options_gmres_advanced() {
        let args = vec![
            "-ksp_type",
            "gmres",
            "-ksp_rtol",
            "1e-8",
            "-ksp_gmres_restart",
            "50",
            "-ksp_min_iter",
            "5",
            "-ksp_cf_tol",
            "0.9",
            "-ksp_skip_real_r_check",
            "true",
            "-ksp_epsmac",
            "1e-15",
            "-ksp_guard_zero_residual",
            "1e-14",
        ];
        let opts = KspOptions::from_args(&args).unwrap();

        assert_eq!(opts.ksp_type, Some("gmres".to_string()));
        assert_eq!(opts.rtol, Some(1e-8));
        assert_eq!(opts.gmres_restart, Some(50));
        assert_eq!(opts.min_iter, Some(5));
        assert_eq!(opts.cf_tol, Some(0.9));
        assert_eq!(opts.skip_real_r_check, Some(true));
        assert_eq!(opts.epsmac, Some(1e-15));
        assert_eq!(opts.guard_zero_residual, Some(1e-14));
    }

    #[test]
    fn cli_sets_ksp_reduction_mode() {
        let opts = KspOptions::from_args(&["-ksp_reduction", "deterministic"]).unwrap();
        assert_eq!(opts.reduction.as_deref(), Some("deterministic"));

        let opts = KspOptions::from_args(&["-ksp_reduction", "deterministic-accurate"]).unwrap();
        assert_eq!(opts.reduction.as_deref(), Some("deterministic-accurate"));
    }

    #[test]
    fn cli_gmres_restart_sets_gmres_field_only() {
        let args = vec!["-ksp_gmres_restart", "50"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.gmres_restart, Some(50));
        assert_eq!(opts.restart, None);
        assert_eq!(opts.fgmres_restart, None);
    }

    #[test]
    fn generic_then_specific_precedence() {
        let args = vec!["-ksp_restart", "30", "-ksp_gmres_restart", "60"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.restart, Some(30));
        assert_eq!(opts.gmres_restart, Some(60));
        assert_eq!(opts.effective_restart_for(KspType::GMRES), Some(60));
        assert_eq!(opts.effective_restart_for(KspType::FGMRES), Some(30));
    }

    #[test]
    fn options_file_and_cli_overlay() {
        // Simulate options file with generic restart, and CLI with fgmres-specific restart
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "-ksp_restart 20\n").unwrap();
        let args = vec![
            "-options_file".to_string(),
            tmp.path().to_str().unwrap().to_string(),
            "-ksp_fgmres_restart".to_string(),
            "40".to_string(),
        ];
        let (ksp, _pc) = parse_all_options(&args).unwrap();
        assert_eq!(ksp.restart, Some(20));
        assert_eq!(ksp.fgmres_restart, Some(40));
        assert_eq!(ksp.effective_restart_for(KspType::FGMRES), Some(40));
        assert_eq!(ksp.effective_restart_for(KspType::GMRES), Some(20));
    }

    #[test]
    fn rejects_zero_restart_values() {
        let bad1 = KspOptions::from_args(&["-ksp_restart", "0"]).err();
        assert!(bad1.is_some());
        let bad2 = KspOptions::from_args(&["-ksp_gmres_restart", "0"]).err();
        assert!(bad2.is_some());
        let bad3 = KspOptions::from_args(&["-ksp_fgmres_restart", "0"]).err();
        assert!(bad3.is_some());
    }

    #[test]
    fn test_ksp_options_gmres_boolean_parsing() {
        // Test various boolean formats for skip_real_r_check
        let test_cases = vec![
            ("true", true),
            ("false", false),
            ("yes", true),
            ("no", false),
            ("on", true),
            ("off", false),
            ("1", true),
            ("0", false),
        ];

        for (bool_str, expected) in test_cases {
            let args = vec!["-ksp_skip_real_r_check", bool_str];
            let opts = KspOptions::from_args(&args).unwrap();
            assert_eq!(
                opts.skip_real_r_check,
                Some(expected),
                "Failed for input: {}",
                bool_str
            );
        }
    }

    #[test]
    fn test_pc_options_from_args_basic() {
        let args = vec!["-pc_type", "jacobi"];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("jacobi".to_string()));
        assert!(opts.ilu_level.is_none());
    }

    #[test]
    fn test_pc_options_from_args_all_options() {
        let args = vec![
            "-pc_type",
            "ilu",
            "-pc_ilu_levels",
            "5",
            "-pc_chebyshev_degree",
            "10",
            "-pc_ilut_drop_tol",
            "1e-4",
            "-pc_ilut_max_fill",
            "20",
        ];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("ilu".to_string()));
        assert_eq!(opts.ilu_level, Some(5));
        assert_eq!(opts.chebyshev_degree, Some(10));
        assert_eq!(opts.ilut_drop_tol, Some(1e-4));
        assert_eq!(opts.ilut_max_fill, Some(20));
    }

    #[test]
    fn test_pc_options_amg_basic() {
        let args = vec![
            "-pc_type",
            "amg",
            "-pc_amg_levels",
            "15",
            "-pc_amg_strength_threshold",
            "0.5",
        ];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("amg".to_string()));
        assert_eq!(opts.amg_levels, Some(15));
        assert_eq!(opts.amg_strength_threshold, Some(0.5));
    }

    #[test]
    fn test_pc_options_amg_comprehensive() {
        let args = vec![
            "-pc_type",
            "amg",
            "-pc_amg_levels",
            "20",
            "-pc_amg_strength_threshold",
            "0.3",
            "-pc_amg_nu_pre",
            "2",
            "-pc_amg_nu_post",
            "2",
            "-pc_amg_coarse_threshold",
            "5",
            "-pc_amg_max_coarse_size",
            "100",
            "-pc_amg_min_coarse_size",
            "2",
            "-pc_amg_truncation_factor",
            "0.1",
            "-pc_amg_max_elements_per_row",
            "8",
            "-pc_amg_interpolation_truncation",
            "0.05",
            "-pc_amg_coarsen_type",
            "hmis",
            "-pc_amg_interp_type",
            "classical",
            "-pc_amg_relax_type",
            "gs",
            "-pc_amg_logging_level",
            "1",
            "-pc_amg_print_level",
            "2",
            "-pc_amg_tolerance",
            "1e-10",
            "-pc_amg_max_iterations",
            "200",
            "-pc_amg_min_iterations",
            "5",
            "-pc_amg_ieee_checks",
            "true",
            "-pc_amg_optimize_workspace",
            "false",
        ];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("amg".to_string()));
        assert_eq!(opts.amg_levels, Some(20));
        assert_eq!(opts.amg_strength_threshold, Some(0.3));
        assert_eq!(opts.amg_nu_pre, Some(2));
        assert_eq!(opts.amg_nu_post, Some(2));
        assert_eq!(opts.amg_coarse_threshold, Some(5));
        assert_eq!(opts.amg_max_coarse_size, Some(100));
        assert_eq!(opts.amg_min_coarse_size, Some(2));
        assert_eq!(opts.amg_truncation_factor, Some(0.1));
        assert_eq!(opts.amg_max_elements_per_row, Some(8));
        assert_eq!(opts.amg_interpolation_truncation, Some(0.05));
        assert_eq!(opts.amg_coarsen_type, Some("hmis".to_string()));
        assert_eq!(opts.amg_interp_type, Some("classical".to_string()));
        assert_eq!(opts.amg_relax_type, Some("gs".to_string()));
        assert_eq!(opts.amg_logging_level, Some(1));
        assert_eq!(opts.amg_print_level, Some(2));
        assert_eq!(opts.amg_tolerance, Some(1e-10));
        assert_eq!(opts.amg_max_iterations, Some(200));
        assert_eq!(opts.amg_min_iterations, Some(5));
        assert_eq!(opts.amg_ieee_checks, Some(true));
        assert_eq!(opts.amg_optimize_workspace, Some(false));
    }

    #[test]
    fn test_pc_options_amg_invalid_coarsen_type() {
        let args = vec!["-pc_amg_coarsen_type", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            let lo = msg.to_lowercase();
            assert!(
                lo.contains("invalid amgcoarsenkind")
                    || lo.contains("invalid amg_coarsen_type")
                    || lo.contains("invalid pc_amg_coarsen_type")
                    || lo.contains("invalid amg")
            );
            assert!(lo.contains("allowed"));
        } else {
            panic!("Expected SolveError for invalid coarsen type");
        }
    }

    #[test]
    fn test_pc_options_amg_invalid_interp_type() {
        let args = vec!["-pc_amg_interp_type", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            let lo = msg.to_lowercase();
            assert!(
                lo.contains("invalid amginterpkind")
                    || lo.contains("invalid amg_interp_type")
                    || lo.contains("invalid pc_amg_interp_type")
            );
            assert!(lo.contains("allowed"));
        } else {
            panic!("Expected SolveError for invalid interpolation type");
        }
    }

    #[test]
    fn test_pc_options_amg_invalid_relax_type() {
        let args = vec!["-pc_amg_relax_type", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            let lo = msg.to_lowercase();
            assert!(
                lo.contains("invalid amgrelaxkind")
                    || lo.contains("invalid amg_relax_type")
                    || lo.contains("invalid pc_amg_relax_type")
            );
            assert!(lo.contains("allowed"));
        } else {
            panic!("Expected SolveError for invalid relaxation type");
        }
    }

    #[test]
    fn test_pc_options_amg_boolean_parsing() {
        // Test true values
        let true_values = vec!["true", "1", "yes", "on"];
        for value in true_values {
            let args = vec!["-pc_amg_ieee_checks", value];
            let opts = PcOptions::from_args(&args).unwrap();
            assert_eq!(opts.amg_ieee_checks, Some(true));
        }

        // Test false values
        let false_values = vec!["false", "0", "no", "off"];
        for value in false_values {
            let args = vec!["-pc_amg_optimize_workspace", value];
            let opts = PcOptions::from_args(&args).unwrap();
            assert_eq!(opts.amg_optimize_workspace, Some(false));
        }
    }

    #[test]
    fn test_pc_options_asm_options() {
        let args = vec![
            "-pc_type",
            "asm",
            "-pc_asm_overlap",
            "2",
            "-pc_asm_subdomains",
            "0,1,2,3",
            "-pc_asm_inner_pc",
            "ilu",
        ];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("asm".to_string()));
        assert_eq!(opts.asm_overlap, Some(2));
        assert_eq!(opts.asm_subdomains, Some(vec![0, 1, 2, 3]));
        assert_eq!(opts.asm_inner_pc, Some("ilu".to_string()));
    }

    #[test]
    fn test_pc_options_chebyshev_options() {
        let args = vec![
            "-pc_type",
            "chebyshev",
            "-pc_chebyshev_degree",
            "5",
            "-pc_chebyshev_lambda_min",
            "0.1",
            "-pc_chebyshev_lambda_max",
            "10.0",
        ];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("chebyshev".to_string()));
        assert_eq!(opts.chebyshev_degree, Some(5));
        assert_eq!(opts.chebyshev_lambda_min, Some(0.1));
        assert_eq!(opts.chebyshev_lambda_max, Some(10.0));
    }

    #[test]
    fn test_pc_options_reorder_and_scaling() {
        let args = vec!["-pc_reorder", "colamd", "-pc_scaling", "diagonal"];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.reorder, Some("colamd".to_string()));
        assert_eq!(opts.scaling, Some("diagonal".to_string()));
    }

    #[test]
    fn test_pc_options_invalid_reorder() {
        let args = vec!["-pc_reorder", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            let lo = msg.to_lowercase();
            assert!(
                lo.contains("invalid reorderkind")
                    || lo.contains("invalid pc_reorder")
                    || lo.contains("invalid reorder")
            );
            assert!(lo.contains("allowed"));
        } else {
            panic!("Expected SolveError for invalid reorder type");
        }
    }

    #[test]
    fn test_pc_options_invalid_scaling() {
        let args = vec!["-pc_scaling", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            let lo = msg.to_lowercase();
            assert!(
                lo.contains("invalid scalingkind")
                    || lo.contains("invalid pc_scaling")
                    || lo.contains("invalid scaling")
            );
            assert!(lo.contains("allowed"));
        } else {
            panic!("Expected SolveError for invalid scaling type");
        }
    }

    #[test]
    fn test_pc_options_chain() {
        let args = vec!["-pc_chain", "jacobi,ilu,amg"];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_chain, Some("jacobi,ilu,amg".to_string()));
    }

    #[test]
    fn test_pc_options_ilu_comprehensive() {
        let args = vec![
            "-pc_type",
            "ilu",
            "-pc_ilu_type",
            "ilut",
            "-pc_ilu_level_of_fill",
            "3",
            "-pc_ilu_max_fill_per_row",
            "50",
            "-pc_ilu_offdiag_drop_tolerance",
            "1e-5",
            "-pc_ilu_schur_drop_tolerance",
            "1e-6",
            "-pc_ilu_reordering_type",
            "rcm",
            "-pc_ilu_triangular_solve",
            "iterative",
            "-pc_ilu_lower_jacobi_iters",
            "2",
            "-pc_ilu_upper_jacobi_iters",
            "3",
            "-pc_ilu_tolerance",
            "1e-8",
            "-pc_ilu_max_iterations",
            "10",
            "-pc_ilu_logging_level",
            "2",
            "-pc_ilu_print_level",
            "1",
            "-pc_ilu_ieee_checks",
            "true",
            "-pc_ilu_pivot_monitoring",
            "false",
            "-pc_ilu_optimize_workspace",
            "true",
            "-pc_ilu_pivot_threshold",
            "1e-10",
        ];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("ilu".to_string()));
        assert_eq!(opts.ilu_type, Some("ilut".to_string()));
        assert_eq!(opts.ilu_level_of_fill, Some(3));
        assert_eq!(opts.ilu_max_fill_per_row, Some(50));
        assert_eq!(opts.ilu_offdiag_drop_tolerance, Some(1e-5));
        assert_eq!(opts.ilu_schur_drop_tolerance, Some(1e-6));
        assert_eq!(opts.ilu_reordering_type, Some("rcm".to_string()));
        assert_eq!(opts.ilu_triangular_solve, Some("iterative".to_string()));
        assert_eq!(opts.ilu_lower_jacobi_iters, Some(2));
        assert_eq!(opts.ilu_upper_jacobi_iters, Some(3));
        assert_eq!(opts.ilu_tolerance, Some(1e-8));
        assert_eq!(opts.ilu_max_iterations, Some(10));
        assert_eq!(opts.ilu_logging_level, Some(2));
        assert_eq!(opts.ilu_print_level, Some(1));
        assert_eq!(opts.ilu_ieee_checks, Some(true));
        assert_eq!(opts.ilu_pivot_monitoring, Some(false));
        assert_eq!(opts.ilu_optimize_workspace, Some(true));
        assert_eq!(opts.ilu_pivot_threshold, Some(1e-10));
        assert!(
            matches!(opts.ilu.kind, IluKind::ILUT { droptol, max_fill_per_row } if (droptol - 1e-5).abs() < 1e-12 && max_fill_per_row == 50)
        );
        assert_eq!(opts.ilu.reordering, ReorderingType::RCM);
        assert!(matches!(
            opts.ilu.tri_solve.kind,
            TriSolveType::Iterative {
                lower_jacobi_iters: 2,
                upper_jacobi_iters: 3
            }
        ));
        assert!((opts.ilu.iterative_setup.tol - 1e-8).abs() < 1e-12);
        assert_eq!(opts.ilu.iterative_setup.max_iter, 10);
        assert_eq!(opts.ilu.logging_level, 2);
        assert!(
            matches!(opts.ilu.pivot, PivotPolicy::Threshold { tau } if (tau - 1e-10).abs() < 1e-12)
        );
    }

    #[test]
    fn test_pc_options_ilu_invalid_type() {
        let args = vec!["-pc_ilu_type", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            let lo = msg.to_lowercase();
            assert!(
                lo.contains("invalid ilutypekind")
                    || lo.contains("invalid pc_ilu_type")
                    || lo.contains("invalid ilu_type")
            );
            assert!(lo.contains("allowed"));
        } else {
            panic!("Expected SolveError for invalid ILU type");
        }
    }

    #[test]
    fn test_pc_options_ilu_invalid_reordering() {
        let args = vec!["-pc_ilu_reordering_type", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            let lo = msg.to_lowercase();
            assert!(
                lo.contains("invalid ilureorderkind")
                    || lo.contains("invalid pc_ilu_reordering_type")
                    || lo.contains("invalid ilu_reordering_type")
            );
            assert!(lo.contains("allowed"));
        } else {
            panic!("Expected SolveError for invalid ILU reordering type");
        }
    }

    #[test]
    fn test_pc_options_ilu_invalid_triangular_solve() {
        let args = vec!["-pc_ilu_triangular_solve", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            let lo = msg.to_lowercase();
            assert!(
                lo.contains("invalid ilutrisolvekind")
                    || lo.contains("invalid pc_ilu_triangular_solve")
                    || lo.contains("invalid ilu_triangular_solve")
            );
            assert!(lo.contains("allowed"));
        } else {
            panic!("Expected SolveError for invalid ILU triangular solve type");
        }
    }

    #[test]
    fn test_pc_options_ilu_boolean_parsing() {
        // Test true values
        let true_values = vec!["true", "1", "yes", "on"];
        for value in true_values {
            let args = vec!["-pc_ilu_ieee_checks", value];
            let opts = PcOptions::from_args(&args).unwrap();
            assert_eq!(opts.ilu_ieee_checks, Some(true));
        }

        // Test false values
        let false_values = vec!["false", "0", "no", "off"];
        for value in false_values {
            let args = vec!["-pc_ilu_pivot_monitoring", value];
            let opts = PcOptions::from_args(&args).unwrap();
            assert_eq!(opts.ilu_pivot_monitoring, Some(false));
        }
    }

    #[test]
    fn test_pc_options_ilu_basic() {
        let args = vec![
            "-pc_type",
            "ilu",
            "-pc_ilu_type",
            "ilu0",
            "-pc_ilu_reordering_type",
            "none",
        ];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("ilu".to_string()));
        assert_eq!(opts.ilu_type, Some("ilu0".to_string()));
        assert_eq!(opts.ilu_reordering_type, Some("none".to_string()));
        assert!(matches!(opts.ilu.kind, IluKind::ILU0));
        assert_eq!(opts.ilu.reordering, ReorderingType::None);
    }

    #[test]
    fn test_pc_options_ilu_par_factor_block() {
        let args = vec!["-pc_ilu_par_factor", "block"];
        let opts = PcOptions::from_args(&args).unwrap();
        assert_eq!(opts.ilu_par_factor, Some("block".to_string()));
        assert_eq!(opts.ilu_parallel_factorization, Some(true));
    }

    #[test]
    fn test_pc_options_ilu_par_factor_invalid() {
        let args = vec!["-pc_ilu_par_factor", "bad"];
        assert!(PcOptions::from_args(&args).is_err());
    }

    #[test]
    fn test_pc_options_ilu_block_size_alias() {
        let args = vec!["-pc_ilu_block_size", "12"];
        let opts = PcOptions::from_args(&args).unwrap();
        assert_eq!(opts.ilu_parallel_chunk_size, Some(12));
    }

    #[test]
    fn test_pc_options_env_par_factor() {
        unsafe {
            std::env::set_var("KRYST_PC_ILU_PAR_FACTOR", "parilu");
        }
        let opts = PcOptions::from_env().unwrap();
        assert_eq!(opts.ilu_par_factor, Some("parilu".to_string()));
        unsafe {
            std::env::remove_var("KRYST_PC_ILU_PAR_FACTOR");
        }
    }

    #[test]
    fn test_pc_options_env_block_size() {
        unsafe {
            std::env::set_var("KRYST_PC_ILU_BLOCK_SIZE", "5");
        }
        let opts = PcOptions::from_env().unwrap();
        assert_eq!(opts.ilu_parallel_chunk_size, Some(5));
        unsafe {
            std::env::remove_var("KRYST_PC_ILU_BLOCK_SIZE");
        }
    }

    #[test]
    fn test_ksp_options_missing_value() {
        let args = vec!["-ksp_type"];
        let result = KspOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            assert!(msg.contains("Missing value for -ksp_type"));
        } else {
            panic!("Expected SolveError for missing value");
        }
    }

    #[test]
    fn test_pc_options_missing_value() {
        let args = vec!["-pc_ilu_levels"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_pc_options_invalid_numeric() {
        let args = vec!["-pc_ilu_levels", "not_a_number"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_pc_side_from_str() {
        assert_eq!(PcSide::from_str("left").unwrap(), PcSide::Left);
        assert_eq!(PcSide::from_str("LEFT").unwrap(), PcSide::Left);
        assert_eq!(PcSide::from_str("right").unwrap(), PcSide::Right);
        assert_eq!(PcSide::from_str("RIGHT").unwrap(), PcSide::Right);
        assert_eq!(PcSide::from_str("symmetric").unwrap(), PcSide::Symmetric);
        assert_eq!(PcSide::from_str("SYMMETRIC").unwrap(), PcSide::Symmetric);

        let result = PcSide::from_str("unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_pc_side_equality() {
        assert_eq!(PcSide::Left, PcSide::Left);
        assert_eq!(PcSide::Right, PcSide::Right);
        assert_eq!(PcSide::Symmetric, PcSide::Symmetric);

        assert_ne!(PcSide::Left, PcSide::Right);
        assert_ne!(PcSide::Left, PcSide::Symmetric);
        assert_ne!(PcSide::Right, PcSide::Symmetric);
    }

    #[test]
    fn test_pc_side_debug() {
        let debug_str = format!("{:?}", PcSide::Left);
        assert!(debug_str.contains("Left"));
    }

    #[test]
    fn test_options_skip_non_ksp_args() {
        let args = vec![
            "program_name",
            "-some_other_option",
            "value",
            "-ksp_type",
            "gmres",
            "-another_option",
            "value2",
            "-ksp_rtol",
            "1e-6",
        ];
        let opts = KspOptions::from_args(&args).unwrap();

        assert_eq!(opts.ksp_type, Some("gmres".to_string()));
        assert_eq!(opts.rtol, Some(1e-6));
    }

    #[test]
    fn test_options_skip_non_pc_args() {
        let args = vec![
            "program_name",
            "-some_option",
            "value",
            "-pc_type",
            "jacobi",
            "-another_option",
            "value2",
        ];
        let opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(opts.pc_type, Some("jacobi".to_string()));
    }

    #[test]
    fn test_ksp_options_from_strings() {
        let args = vec![
            "-ksp_type".to_string(),
            "bicgstab".to_string(),
            "-ksp_rtol".to_string(),
            "1e-7".to_string(),
        ];
        let opts = KspOptions::from_strings(&args).unwrap();

        assert_eq!(opts.ksp_type, Some("bicgstab".to_string()));
        assert_eq!(opts.rtol, Some(1e-7));
    }

    #[test]
    fn test_pc_options_from_strings() {
        let args = vec!["-pc_type".to_string(), "ilu0".to_string()];
        let opts = PcOptions::from_strings(&args).unwrap();

        assert_eq!(opts.pc_type, Some("ilu0".to_string()));
    }

    #[test]
    fn test_options_clone() {
        let mut opts1 = KspOptions::new();
        opts1.ksp_type = Some("gmres".to_string());
        opts1.rtol = Some(1e-8);

        let opts2 = opts1.clone();
        assert_eq!(opts1.ksp_type, opts2.ksp_type);
        assert_eq!(opts1.rtol, opts2.rtol);
    }

    #[test]
    fn test_options_debug() {
        let mut opts = KspOptions::new();
        opts.ksp_type = Some("cg".to_string());

        let debug_str = format!("{:?}", opts);
        assert!(debug_str.contains("cg"));
    }

    #[test]
    fn test_empty_args() {
        let args: Vec<&str> = vec![];
        let ksp_opts = KspOptions::from_args(&args).unwrap();
        let pc_opts = PcOptions::from_args(&args).unwrap();

        // Should be equivalent to default options
        assert!(ksp_opts.ksp_type.is_none());
        assert!(pc_opts.pc_type.is_none());
    }

    #[test]
    fn test_multiple_same_option() {
        // Last occurrence should win
        let args = vec!["-ksp_type", "cg", "-ksp_type", "gmres"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.ksp_type, Some("gmres".to_string()));
    }

    #[test]
    fn test_mixed_ksp_pc_args() {
        let args = vec![
            "-ksp_type",
            "cg",
            "-pc_type",
            "jacobi",
            "-ksp_rtol",
            "1e-6",
            "-pc_ilu_levels",
            "3",
        ];

        let ksp_opts = KspOptions::from_args(&args).unwrap();
        let pc_opts = PcOptions::from_args(&args).unwrap();

        assert_eq!(ksp_opts.ksp_type, Some("cg".to_string()));
        assert_eq!(ksp_opts.rtol, Some(1e-6));
        assert_eq!(pc_opts.pc_type, Some("jacobi".to_string()));
        assert_eq!(pc_opts.ilu_level, Some(3));
    }

    #[test]
    fn test_build_ilu_config_from_options() {
        let mut opts = PcOptions::default();
        opts.ilu_type = Some("iluk".to_string());
        opts.ilu_level_of_fill = Some(2);
        opts.ilu_max_fill_per_row = Some(40);
        opts.ilu_offdiag_drop_tolerance = Some(1e-5);
        opts.ilu_triangular_solve = Some("jacobi".to_string());
        opts.ilu_lower_jacobi_iters = Some(2);
        opts.ilu_upper_jacobi_iters = Some(3);
        opts.ilu_parallel_factorization = Some(true);
        opts.ilu_parallel_triangular_solve = Some(true);
        opts.ilu_parallel_chunk_size = Some(128);

        let config = build_ilu_config(&opts).unwrap();
        assert_eq!(config.ilu_type, IluVariant::ILUK);
        assert_eq!(config.level_of_fill, 2);
        assert_eq!(config.max_fill_per_row, 40);
        assert!((config.offdiag_drop_tolerance - 1e-5).abs() < 1e-12);
        assert_eq!(config.triangular_solve, IluTriSolveType::Jacobi);
        assert_eq!(config.lower_jacobi_iters, 2);
        assert_eq!(config.upper_jacobi_iters, 3);
        assert!(config.enable_parallel_factorization);
        assert!(config.enable_parallel_triangular_solve);
        assert_eq!(config.parallel_chunk_size, 128);
    }

    #[test]
    fn test_ilutp_cli_to_mpi_options() {
        let mut opts = PcOptions::default();
        opts.pc_global = Some("block_jacobi".to_string());
        opts.pc_local = Some("ilutp".to_string());
        opts.ilutp_max_fill = Some(25);
        opts.ilutp_drop_tol = Some(1e-4);
        opts.ilutp_perm_tol = Some(0.2);

        let mpi_opts = opts.mpi_pc_options().unwrap();
        assert_eq!(mpi_opts.global_pc, GlobalPcKind::BlockJacobi);
        assert_eq!(mpi_opts.local_pc, LocalPcKind::Ilutp);
        assert_eq!(mpi_opts.ilutp_max_fill, 25);
        assert!((mpi_opts.ilutp_drop_tol - 1e-4).abs() < 1e-12);
        assert!((mpi_opts.ilutp_perm_tol - 0.2).abs() < 1e-12);
    }
}
