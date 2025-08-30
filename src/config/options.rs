//! PETSc-style options for KSP/PC, parsed via a table-driven engine.
//!
//! Changes:
//! - CLI precedence: CLI > options file(s) > environment > defaults
//! - Added `-options_file <path>` (PETSc-like)
//! - Help now generated from registry: `print_help()`
//! - Boolean flags accept presence or explicit on/off/true/false/1/0

use crate::error::KError;
use std::str::FromStr;

use crate::config::options_core::{Sink, Spec, expand_options_files, parse_as};
use crate::config::registry::registry;

/// KSP (Krylov Solver) options.
#[derive(Debug, Default, Clone)]
pub struct KspOptions {
    pub ksp_type: Option<String>,
    pub rtol: Option<f64>,
    pub atol: Option<f64>,
    pub dtol: Option<f64>,
    pub maxits: Option<usize>,
    pub restart: Option<usize>,
    // GMRES/FGMRES-specific (backward-compatible; all optional)
    /// Override restart for GMRES; falls back to `restart` if unset
    pub gmres_restart: Option<usize>,
    /// Orthogonalization method for GMRES: "mgs" | "cgs"
    pub gmres_orthog: Option<String>,
    /// If true, perform a second orthogonalization pass (reorthogonalization)
    pub gmres_reorthog: Option<bool>,
    /// If true, treat near-zero residual as a happy breakdown and stop
    pub gmres_happy_breakdown: Option<bool>,

    /// Override restart for FGMRES; falls back to `restart` if unset
    pub fgmres_restart: Option<usize>,
    /// Orthogonalization method for FGMRES: "mgs" | "cgs"
    pub fgmres_orthog: Option<String>,
    /// If true, perform a second orthogonalization pass (reorthogonalization)
    pub fgmres_reorthog: Option<bool>,
    /// If true, treat near-zero residual as a happy breakdown and stop
    pub fgmres_happy_breakdown: Option<bool>,
    pub pc_side: Option<String>,
    pub matrix_file: Option<String>,
    pub rhs_file: Option<String>,
    pub min_iter: Option<usize>,
    pub cf_tol: Option<f64>,
    pub skip_real_r_check: Option<bool>,
    pub epsmac: Option<f64>,
    pub guard_zero_residual: Option<f64>,
    pub cg_norm: Option<String>,
    pub cg_single_reduction: Option<bool>,
    pub trust_region: Option<f64>,
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
    pub reorder: Option<String>,
    pub scaling: Option<String>,
    /// Overlap for Additive Schwarz.
    pub asm_overlap: Option<usize>,
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
            _ => Err(KError::SolveError(format!("Unrecognized pc_side: {}", s))),
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

impl Sink for KspOptions {
    fn set_bool(&mut self, key: &str, v: bool) -> Result<(), KError> {
        match key {
            "ksp_skip_real_r_check" => set_opt!(&mut self.skip_real_r_check, v),
            "ksp_cg_single_reduction" => set_opt!(&mut self.cg_single_reduction, v),
            "ksp_gmres_reorthog" => set_opt!(&mut self.gmres_reorthog, v),
            "ksp_gmres_happy_breakdown" => set_opt!(&mut self.gmres_happy_breakdown, v),
            "ksp_fgmres_reorthog" => set_opt!(&mut self.fgmres_reorthog, v),
            "ksp_fgmres_happy_breakdown" => set_opt!(&mut self.fgmres_happy_breakdown, v),
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
            "ksp_gmres_restart" => set_opt!(&mut self.restart, parse_as::<usize>(v, spec)?),
            // Additional GMRES/FGMRES keys
            "ksp_gmres_orthog" => set_opt!(&mut self.gmres_orthog, v.to_string()),
            "ksp_fgmres_restart" => set_opt!(&mut self.fgmres_restart, parse_as::<usize>(v, spec)?),
            "ksp_fgmres_orthog" => set_opt!(&mut self.fgmres_orthog, v.to_string()),
            "ksp_pc_side" => set_opt!(&mut self.pc_side, v.to_string()),
            "matrix" => set_opt!(&mut self.matrix_file, v.to_string()),
            "rhs" => set_opt!(&mut self.rhs_file, v.to_string()),
            "ksp_min_iter" => set_opt!(&mut self.min_iter, parse_as::<usize>(v, spec)?),
            "ksp_cf_tol" => set_opt!(&mut self.cf_tol, parse_as::<f64>(v, spec)?),
            "ksp_epsmac" => set_opt!(&mut self.epsmac, parse_as::<f64>(v, spec)?),
            "ksp_guard_zero_residual" => {
                set_opt!(&mut self.guard_zero_residual, parse_as::<f64>(v, spec)?)
            }
            "ksp_cg_norm" => set_opt!(&mut self.cg_norm, v.to_string()),
            "ksp_trust_region" => set_opt!(&mut self.trust_region, parse_as::<f64>(v, spec)?),
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
            "pc_amg_ieee_checks" => set_opt!(&mut self.amg_ieee_checks, v),
            "pc_amg_optimize_workspace" => set_opt!(&mut self.amg_optimize_workspace, v),
            "pc_ilu_ieee_checks" => set_opt!(&mut self.ilu_ieee_checks, v),
            "pc_ilu_pivot_monitoring" => set_opt!(&mut self.ilu_pivot_monitoring, v),
            "pc_ilu_optimize_workspace" => set_opt!(&mut self.ilu_optimize_workspace, v),
            "pc_superlu_replace_tiny_pivot" => set_opt!(&mut self.superlu_replace_tiny_pivots, v),
            "pc_superlu_static_pivoting" => set_opt!(&mut self.superlu_static_pivoting, v),
            "pc_superlu_enable_3d_factorization" => {
                set_opt!(&mut self.superlu_enable_3d_factorization, v)
            }
            "pc_superlu_async_panel_updates" => set_opt!(&mut self.superlu_async_panel_updates, v),
            "pc_superlu_aggressive_memory_reuse" => {
                set_opt!(&mut self.superlu_aggressive_memory_reuse, v)
            }
            _ => Err(KError::SolveError(format!("Unknown PC bool key: {key}"))),
        }
    }

    fn set_val(&mut self, spec: &Spec, v: &str) -> Result<(), KError> {
        match spec.key {
            "pc_type" => set_opt!(&mut self.pc_type, v.to_string()),
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
            "pc_amg_max_elements_per_row" => set_opt!(
                &mut self.amg_max_elements_per_row,
                parse_as::<usize>(v, spec)?
            ),
            "pc_amg_interpolation_truncation" => set_opt!(
                &mut self.amg_interpolation_truncation,
                parse_as::<f64>(v, spec)?
            ),
            "pc_amg_coarsen_type" => set_opt!(&mut self.amg_coarsen_type, v.to_lowercase()),
            "pc_amg_interp_type" => set_opt!(&mut self.amg_interp_type, v.to_lowercase()),
            "pc_amg_relax_type" => set_opt!(&mut self.amg_relax_type, v.to_lowercase()),
            "pc_amg_logging_level" => {
                set_opt!(&mut self.amg_logging_level, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_print_level" => {
                set_opt!(&mut self.amg_print_level, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_tolerance" => set_opt!(&mut self.amg_tolerance, parse_as::<f64>(v, spec)?),
            "pc_amg_max_iterations" => {
                set_opt!(&mut self.amg_max_iterations, parse_as::<usize>(v, spec)?)
            }
            "pc_amg_min_iterations" => {
                set_opt!(&mut self.amg_min_iterations, parse_as::<usize>(v, spec)?)
            }
            "pc_chain" => set_opt!(&mut self.pc_chain, v.to_string()),
            "pc_ilu_type" => set_opt!(&mut self.ilu_type, v.to_lowercase()),
            "pc_ilu_level_of_fill" => {
                set_opt!(&mut self.ilu_level_of_fill, parse_as::<usize>(v, spec)?)
            }
            "pc_ilu_max_fill_per_row" => {
                set_opt!(&mut self.ilu_max_fill_per_row, parse_as::<usize>(v, spec)?)
            }
            "pc_ilu_offdiag_drop_tolerance" => set_opt!(
                &mut self.ilu_offdiag_drop_tolerance,
                parse_as::<f64>(v, spec)?
            ),
            "pc_ilu_schur_drop_tolerance" => set_opt!(
                &mut self.ilu_schur_drop_tolerance,
                parse_as::<f64>(v, spec)?
            ),
            "pc_ilu_reordering_type" => set_opt!(&mut self.ilu_reordering_type, v.to_lowercase()),
            "pc_ilu_triangular_solve" => set_opt!(&mut self.ilu_triangular_solve, v.to_lowercase()),
            "pc_ilu_lower_jacobi_iters" => set_opt!(
                &mut self.ilu_lower_jacobi_iters,
                parse_as::<usize>(v, spec)?
            ),
            "pc_ilu_upper_jacobi_iters" => set_opt!(
                &mut self.ilu_upper_jacobi_iters,
                parse_as::<usize>(v, spec)?
            ),
            "pc_ilu_tolerance" => set_opt!(&mut self.ilu_tolerance, parse_as::<f64>(v, spec)?),
            "pc_ilu_max_iterations" => {
                set_opt!(&mut self.ilu_max_iterations, parse_as::<usize>(v, spec)?)
            }
            "pc_ilu_logging_level" => {
                set_opt!(&mut self.ilu_logging_level, parse_as::<usize>(v, spec)?)
            }
            "pc_ilu_print_level" => {
                set_opt!(&mut self.ilu_print_level, parse_as::<usize>(v, spec)?)
            }
            "pc_ilu_pivot_threshold" => {
                set_opt!(&mut self.ilu_pivot_threshold, parse_as::<f64>(v, spec)?)
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
        if let Ok(v) = std::env::var("KRYST_KSP_GMRES_RESTART") {
            me.restart = Some(v.parse().map_err(|_| {
                KError::SolveError(format!("Invalid KRYST_KSP_GMRES_RESTART: {v}"))
            })?);
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
        if let Ok(v) = std::env::var("KRYST_KSP_CG_SINGLE_REDUCTION") {
            let l = v.to_lowercase();
            me.cg_single_reduction = Some(matches!(l.as_str(), "true" | "1" | "yes" | "on"));
        }
        if let Ok(v) = std::env::var("KRYST_KSP_TRUST_REGION") {
            me.trust_region =
                Some(v.parse().map_err(|_| {
                    KError::SolveError(format!("Invalid KRYST_KSP_TRUST_REGION: {v}"))
                })?);
        }
        Ok(me)
    }
}

impl PcOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_args(args: &[&str]) -> Result<Self, KError> {
        let mut me = Self::default();
        registry().parse_into(args, &mut me, Some("-pc_"))?;
        // enum validations with friendly messages
        if let Some(ref t) = me.reorder {
            match t.as_str() {
                "none" | "colamd" | "amd" | "rcm" | "cuthill_mckee" => {}
                _ => return Err(KError::SolveError(format!("Invalid reorder type: {t}"))),
            }
        }
        if let Some(ref s) = me.scaling {
            match s.as_str() {
                "none" | "diagonal" | "symmetric" => {}
                _ => return Err(KError::SolveError(format!("Invalid scaling type: {s}"))),
            }
        }
        if let Some(ref t) = me.ilu_type {
            match t.as_str() {
                "ilu0" | "iluk" | "ilut" | "milu0" | "block_jacobi" | "gmres_iluk"
                | "gmres_ilut" => {}
                _ => return Err(KError::SolveError(format!("Invalid ilu_type: {t}"))),
            }
        }
        if let Some(ref t) = me.ilu_reordering_type {
            match t.as_str() {
                "none" | "rcm" | "amd" | "natural" => {}
                _ => {
                    return Err(KError::SolveError(format!(
                        "Invalid ilu_reordering_type: {t}"
                    )));
                }
            }
        }
        if let Some(ref t) = me.ilu_triangular_solve {
            match t.as_str() {
                "exact" | "iterative" => {}
                _ => {
                    return Err(KError::SolveError(format!(
                        "Invalid ilu_triangular_solve: {t}"
                    )));
                }
            }
        }
        if let Some(ref t) = me.amg_coarsen_type {
            match t.as_str() {
                "rs" | "hmis" | "pmis" | "falgout" => {}
                _ => return Err(KError::SolveError(format!("Invalid amg_coarsen_type: {t}"))),
            }
        }
        if let Some(ref t) = me.amg_interp_type {
            match t.as_str() {
                "classical" | "direct" | "multipass" | "extended" | "standard" => {}
                _ => return Err(KError::SolveError(format!("Invalid amg_interp_type: {t}"))),
            }
        }
        if let Some(ref t) = me.amg_relax_type {
            match t.as_str() {
                "jacobi" | "gs" | "gsr" | "sgs" | "hgs" | "l1jacobi" | "chebyshev" => {}
                _ => return Err(KError::SolveError(format!("Invalid amg_relax_type: {t}"))),
            }
        }
        if let Some(ref t) = me.asm_block_solver {
            match t.as_str() {
                "ludense" | "csr" => {}
                _ => return Err(KError::SolveError(format!("Invalid pc_asm_block_solver: {t}"))),
            }
        }
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
        if let Ok(v) = std::env::var("KRYST_PC_ILU_LEVELS") {
            me.ilu_level =
                Some(v.parse().map_err(|_| {
                    KError::SolveError(format!("Invalid KRYST_PC_ILU_LEVELS: {v}"))
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
        Ok(me)
    }
}

// ---- Combined parsing with precedence & generated help ----

pub fn print_help() {
    let reg = registry();
    println!("Kryst Linear Solver Options\n");
    println!("KSP options:");
    print!("{}", reg.help_for_prefix("-ksp_"));
    println!("General:");
    print!("{}", reg.help_for_prefix("-m")); // will include -matrix
    print!("{}", reg.help_for_prefix("-r")); // -rhs
    println!("PC options:");
    print!("{}", reg.help_for_prefix("-pc_"));
    println!("Utility:");
    print!("  -options_file <path>              str     Read more options from file\n");
}

/// CLI > options file(s) > env > defaults
pub fn parse_all_options(args: &[String]) -> Result<(KspOptions, PcOptions), KError> {
    let mut args = args.to_vec();

    // help?
    if args
        .iter()
        .any(|a| a == "-help" || a == "--help" || a == "-h")
    {
        print_help();
        std::process::exit(0);
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

    // overlay helper
    macro_rules! overlay {
        ($lhs:expr, $rhs:expr, $($f:ident),+ $(,)?) => { $( if $rhs.$f.is_some() { $lhs.$f = $rhs.$f; } )+ };
    }

    overlay!(
        ksp_opts,
        cli_ksp,
        ksp_type,
        rtol,
        atol,
        dtol,
        maxits,
        restart,
        pc_side,
        matrix_file,
        rhs_file,
        min_iter,
        cf_tol,
        skip_real_r_check,
        epsmac,
        guard_zero_residual,
    );
    overlay!(
        pc_opts,
        cli_pc,
        pc_type,
        ilu_level,
        chebyshev_degree,
        ilut_drop_tol,
        ilut_max_fill,
        ilut_perm_tol,
        reorder,
        scaling,
        asm_overlap,
        asm_subdomains,
        asm_inner_pc,
        chebyshev_lambda_min,
        chebyshev_lambda_max,
        amg_levels,
        amg_strength_threshold,
        amg_nu_pre,
        amg_nu_post,
        amg_coarse_threshold,
        amg_max_coarse_size,
        amg_min_coarse_size,
        amg_truncation_factor,
        amg_max_elements_per_row,
        amg_interpolation_truncation,
        amg_coarsen_type,
        amg_interp_type,
        amg_relax_type,
        amg_logging_level,
        amg_print_level,
        amg_tolerance,
        amg_max_iterations,
        amg_min_iterations,
        amg_ieee_checks,
        amg_optimize_workspace,
        pc_chain,
        omega,
        drop_tol,
        ilu_type,
        ilu_level_of_fill,
        ilu_max_fill_per_row,
        ilu_offdiag_drop_tolerance,
        ilu_schur_drop_tolerance,
        ilu_reordering_type,
        ilu_triangular_solve,
        ilu_lower_jacobi_iters,
        ilu_upper_jacobi_iters,
        ilu_tolerance,
        ilu_max_iterations,
        ilu_logging_level,
        ilu_print_level,
        ilu_ieee_checks,
        ilu_pivot_monitoring,
        ilu_optimize_workspace,
        ilu_pivot_threshold,
        superlu_pivot_threshold,
        superlu_replace_tiny_pivots,
        superlu_print_level,
        superlu_process_grid,
        superlu_column_permutation,
        superlu_row_permutation,
        superlu_iterative_refinement,
        superlu_static_pivoting,
        superlu_panel_size,
        superlu_enable_3d_factorization,
        superlu_process_grid_3d_depth,
        superlu_memory_tradeoff_factor,
        superlu_max_concurrent_panels,
        superlu_async_panel_updates,
        superlu_workspace_memory_limit,
        superlu_aggressive_memory_reuse,
        superlu_preallocation_strategy,
    );

    Ok((ksp_opts, pc_opts))
}

// ---- tests: reuse your existing tests; only minor changes below ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ksp_bool_toggle() {
        let args = vec!["-ksp_skip_real_r_check"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.skip_real_r_check, Some(true));

        let args = vec!["-ksp_skip_real_r_check", "false"];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.skip_real_r_check, Some(false));
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
            "-ksp_gmres_restart",
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
        assert_eq!(opts.restart, Some(50));
        assert_eq!(opts.min_iter, Some(5));
        assert_eq!(opts.cf_tol, Some(0.9));
        assert_eq!(opts.skip_real_r_check, Some(true));
        assert_eq!(opts.epsmac, Some(1e-15));
        assert_eq!(opts.guard_zero_residual, Some(1e-14));
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
            assert!(msg.contains("Invalid amg_coarsen_type"));
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
            assert!(msg.contains("Invalid amg_interp_type"));
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
            assert!(msg.contains("Invalid amg_relax_type"));
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
            assert!(msg.contains("Invalid reorder type"));
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
            assert!(msg.contains("Invalid scaling type"));
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
    }

    #[test]
    fn test_pc_options_ilu_invalid_type() {
        let args = vec!["-pc_ilu_type", "invalid"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());

        if let Err(KError::SolveError(msg)) = result {
            assert!(msg.contains("Invalid ilu_type"));
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
            assert!(msg.contains("Invalid ilu_reordering_type"));
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
            assert!(msg.contains("Invalid ilu_triangular_solve"));
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
}
