//! PETSc-style command-line options parsing for KSP and PC configuration.
//!
//! This module provides structs and parsing functionality to configure solvers
//! and preconditioners from command-line arguments, similar to PETSc's options database.
//!
//! # Supported Options
//!
//! ## KSP (Krylov Solver) Options
//! - `-ksp_type <solver>` - Solver type (cg, gmres, bicgstab, etc.)
//! - `-ksp_rtol <float>` - Relative tolerance
//! - `-ksp_atol <float>` - Absolute tolerance  
//! - `-ksp_dtol <float>` - Divergence tolerance
//! - `-ksp_max_it <int>` - Maximum iterations
//! - `-ksp_gmres_restart <int>` - Restart parameter for GMRES
//! - `-ksp_pc_side <side>` - Preconditioning side (left, right, symmetric)
//! - `-ksp_min_iter <int>` - Minimum iterations before convergence check (GMRES)
//! - `-ksp_cf_tol <float>` - Convergence factor tolerance for stagnation detection (GMRES)
//! - `-ksp_skip_real_r_check <bool>` - Skip real residual check for performance (GMRES)
//! - `-ksp_epsmac <float>` - IEEE safety epsilon for breakdown protection (GMRES)
//! - `-ksp_guard_zero_residual <float>` - Guard for zero residual to prevent NaN (GMRES)
//!
//! ## PC (Preconditioner) Options
//! - `-pc_type <pc>` - Preconditioner type (jacobi, ilu0, ilu, ilut, none, amg, asm, chebyshev, lu, qr, superlu_dist)
//! - `-pc_ilu_levels <int>` - ILU fill levels (legacy)
//! - `-pc_ilu_type <type>` - ILU factorization type (ilu0, iluk, ilut, milu0, etc.)
//! - `-pc_ilu_level_of_fill <int>` - Level of fill for ILU(k)
//! - `-pc_ilu_triangular_solve <type>` - Triangular solve type (exact, iterative)
//! - `-pc_ilu_reordering_type <type>` - ILU reordering strategy
//! - `-pc_chebyshev_degree <int>` - Chebyshev polynomial degree
//! - `-pc_amg_levels <int>` - AMG coarsening levels
//! - `-pc_asm_overlap <int>` - ASM overlap layers
//!
//! # Usage
//!
//! ```rust,ignore
//! use kryst::config::options::{KspOptions, PcOptions};
//!
//! let args = vec!["-ksp_type", "gmres", "-ksp_rtol", "1e-8", "-pc_type", "jacobi"];
//! let ksp_opts = KspOptions::from_args(&args)?;
//! let pc_opts = PcOptions::from_args(&args)?;
//! ```

use std::str::FromStr;
use crate::error::KError;

/// KSP (Krylov Solver) configuration options from command-line arguments.
#[derive(Debug, Default, Clone)]
pub struct KspOptions {
    /// Solver type (cg, gmres, bicgstab, etc.)
    pub ksp_type: Option<String>,
    /// Relative tolerance for convergence
    pub rtol: Option<f64>,
    /// Absolute tolerance for convergence
    pub atol: Option<f64>,
    /// Divergence tolerance
    pub dtol: Option<f64>,
    /// Maximum number of iterations
    pub maxits: Option<usize>,
    /// Restart parameter for GMRES/FGMRES
    pub restart: Option<usize>,
    /// Preconditioning side (left, right, symmetric)
    pub pc_side: Option<String>,
    /// Matrix file path
    pub matrix_file: Option<String>,
    /// RHS file path
    pub rhs_file: Option<String>,
    /// Minimum iterations before convergence check (GMRES HYPRE feature)
    pub min_iter: Option<usize>,
    /// Convergence factor tolerance for stagnation detection (GMRES HYPRE feature)
    pub cf_tol: Option<f64>,
    /// Skip real residual check for performance (GMRES HYPRE feature)
    pub skip_real_r_check: Option<bool>,
    /// IEEE safety epsilon for breakdown protection (GMRES HYPRE feature)
    pub epsmac: Option<f64>,
    /// Guard for zero residual to prevent NaN (GMRES HYPRE feature)
    pub guard_zero_residual: Option<f64>,
}

/// PC (Preconditioner) configuration options from command-line arguments.
#[derive(Debug, Default, Clone)]
pub struct PcOptions {
    /// Preconditioner type (jacobi, ilu0, none, etc.)
    pub pc_type: Option<String>,
    /// Fill level for ILU preconditioners
    pub ilu_level: Option<usize>,
    /// Polynomial degree for Chebyshev preconditioner
    pub chebyshev_degree: Option<usize>,
    /// Drop tolerance for ILUT
    pub ilut_drop_tol: Option<f64>,
    /// Maximum fill for ILUT
    pub ilut_max_fill: Option<usize>,
    /// Pivot tolerance for ILUTP
    pub ilut_perm_tol: Option<f64>,
    /// Matrix reordering algorithm (colamd, amd, rcm, cuthill_mckee, none)
    pub reorder: Option<String>,
    /// Matrix scaling algorithm (diagonal, symmetric, none)
    pub scaling: Option<String>,
    /// Overlap layers for Additive Schwarz Method (ASM)
    pub asm_overlap: Option<usize>,
    /// Subdomain specification for ASM (default: automatic)
    pub asm_subdomains: Option<Vec<usize>>,
    /// Inner preconditioner type for ASM blocks
    pub asm_inner_pc: Option<String>,
    /// Minimum eigenvalue estimate for Chebyshev
    pub chebyshev_lambda_min: Option<f64>,
    /// Maximum eigenvalue estimate for Chebyshev
    pub chebyshev_lambda_max: Option<f64>,
    /// Number of AMG coarsening levels (HYPRE default: 25)
    pub amg_levels: Option<usize>,
    /// Strength-of-connection threshold for AMG (HYPRE default: 0.25)
    pub amg_strength_threshold: Option<f64>,
    /// Number of pre-smoothing iterations for AMG (HYPRE default: 1)
    pub amg_nu_pre: Option<usize>,
    /// Number of post-smoothing iterations for AMG (HYPRE default: 1)
    pub amg_nu_post: Option<usize>,
    /// Coarse grid threshold - stop coarsening (HYPRE default: 9)
    pub amg_coarse_threshold: Option<usize>,
    /// Maximum coarse grid size (HYPRE default: 9)
    pub amg_max_coarse_size: Option<usize>,
    /// Minimum coarse grid size (HYPRE default: 1)
    pub amg_min_coarse_size: Option<usize>,
    /// Truncation factor for interpolation (HYPRE default: 0.0)
    pub amg_truncation_factor: Option<f64>,
    /// Max elements per row for interpolation (HYPRE default: 0)
    pub amg_max_elements_per_row: Option<usize>,
    /// Interpolation truncation factor (HYPRE default: 0.0)
    pub amg_interpolation_truncation: Option<f64>,
    /// AMG coarsening algorithm (rs, hmis, pmis, falgout)
    pub amg_coarsen_type: Option<String>,
    /// AMG interpolation algorithm (classical, direct, multipass, extended, standard)
    pub amg_interp_type: Option<String>,
    /// AMG relaxation/smoothing type (jacobi, gs, gsr, sgs, hgs, l1jacobi, chebyshev)
    pub amg_relax_type: Option<String>,
    /// AMG logging level (HYPRE style: 0=none, 1=basic, 2=detailed)
    pub amg_logging_level: Option<usize>,
    /// AMG print level (HYPRE style: 0=none, 1=basic, 2=detailed)
    pub amg_print_level: Option<usize>,
    /// AMG convergence tolerance (for standalone AMG solver)
    pub amg_tolerance: Option<f64>,
    /// AMG maximum iterations (for standalone AMG solver)
    pub amg_max_iterations: Option<usize>,
    /// AMG minimum iterations (HYPRE feature)
    pub amg_min_iterations: Option<usize>,
    /// Enable AMG IEEE safety checks
    pub amg_ieee_checks: Option<bool>,
    /// Enable AMG workspace optimization
    pub amg_optimize_workspace: Option<bool>,
    /// Preconditioner chain specification (comma-separated list)
    pub pc_chain: Option<String>,
    /// Relaxation factor ω for SSOR (legacy compatibility)
    pub omega: Option<f64>,
    /// Drop tolerance for ILU(p) (legacy compatibility)
    pub drop_tol: Option<f64>,
    
    // Comprehensive ILU Configuration Options
    /// ILU factorization type (ilu0, iluk, ilut, milu0, block_jacobi, gmres_iluk, gmres_ilut)
    pub ilu_type: Option<String>,
    /// Level of fill for ILU(k) (HYPRE: lfil)
    pub ilu_level_of_fill: Option<usize>,
    /// Maximum nonzeros per row for ILU (HYPRE: maxRowNnz)
    pub ilu_max_fill_per_row: Option<usize>,
    /// Drop tolerance for off-diagonal blocks (HYPRE: droptol[1])
    pub ilu_offdiag_drop_tolerance: Option<f64>,
    /// Drop tolerance for Schur complement (HYPRE: droptol[2])
    pub ilu_schur_drop_tolerance: Option<f64>,
    /// ILU reordering strategy (none, rcm, amd, natural)
    pub ilu_reordering_type: Option<String>,
    /// Triangular solve type (exact, iterative)
    pub ilu_triangular_solve: Option<String>,
    /// Lower triangular Jacobi iterations (HYPRE: lower_jacobi_iters)
    pub ilu_lower_jacobi_iters: Option<usize>,
    /// Upper triangular Jacobi iterations (HYPRE: upper_jacobi_iters)  
    pub ilu_upper_jacobi_iters: Option<usize>,
    /// Tolerance for iterative ILU solve (HYPRE: tol)
    pub ilu_tolerance: Option<f64>,
    /// Maximum iterations for iterative ILU solve (HYPRE: max_iter)
    pub ilu_max_iterations: Option<usize>,
    /// ILU logging level (HYPRE style: 0=none, 1=basic, 2=detailed)
    pub ilu_logging_level: Option<usize>,
    /// ILU print level (HYPRE style: 0=none, 1=basic, 2=detailed)
    pub ilu_print_level: Option<usize>,
    /// Enable ILU IEEE safety checks
    pub ilu_ieee_checks: Option<bool>,
    /// Enable ILU pivot monitoring
    pub ilu_pivot_monitoring: Option<bool>,
    /// Enable ILU workspace optimization
    pub ilu_optimize_workspace: Option<bool>,
    /// ILU pivot threshold for stability
    pub ilu_pivot_threshold: Option<f64>,
    
    // SuperLU_DIST Configuration Options
    /// SuperLU_DIST diagonal pivot threshold (0.0 to 1.0)
    pub superlu_pivot_threshold: Option<f64>,
    /// Whether to replace tiny pivots in SuperLU_DIST
    pub superlu_replace_tiny_pivots: Option<bool>,
    /// SuperLU_DIST print level (0=none, 1=basic, 2=detailed)
    pub superlu_print_level: Option<u8>,
    /// SuperLU_DIST process grid dimensions (rows, cols)
    pub superlu_process_grid: Option<(usize, usize)>,
    /// SuperLU_DIST column permutation strategy
    pub superlu_column_permutation: Option<String>,
    /// SuperLU_DIST row permutation strategy
    pub superlu_row_permutation: Option<String>,
    /// SuperLU_DIST iterative refinement method
    pub superlu_iterative_refinement: Option<String>,
    /// Whether to use static pivoting in SuperLU_DIST
    pub superlu_static_pivoting: Option<bool>,
}

/// Preconditioning side specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcSide {
    /// Left preconditioning: M⁻¹Ax = M⁻¹b
    Left,
    /// Right preconditioning: AM⁻¹y = b, x = M⁻¹y
    Right,
    /// Symmetric preconditioning: L⁻¹AU⁻¹y = L⁻¹b, x = U⁻¹y
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

impl KspOptions {
    /// Create new KspOptions with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse KSP options from command-line arguments.
    ///
    /// # Arguments
    /// * `args` - Command-line arguments (typically from `std::env::args()`)
    ///
    /// # Returns
    /// * `Ok(KspOptions)` with parsed options
    /// * `Err(KError)` if parsing fails
    ///
    /// # Example
    /// ```rust,ignore
    /// let args = vec!["-ksp_type", "gmres", "-ksp_rtol", "1e-8"];
    /// let opts = KspOptions::from_args(&args)?;
    /// assert_eq!(opts.ksp_type, Some("gmres".to_string()));
    /// assert_eq!(opts.rtol, Some(1e-8));
    /// ```
    pub fn from_args(args: &[&str]) -> Result<Self, KError> {
        let mut opts = Self::new();
        let mut i = 0;
        
        while i < args.len() {
            let arg = args[i];
            
            match arg {
                "-ksp_type" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_type".to_string()));
                    }
                    opts.ksp_type = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-ksp_rtol" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_rtol".to_string()));
                    }
                    opts.rtol = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid rtol value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-ksp_atol" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_atol".to_string()));
                    }
                    opts.atol = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid atol value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-ksp_dtol" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_dtol".to_string()));
                    }
                    opts.dtol = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid dtol value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-ksp_max_it" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_max_it".to_string()));
                    }
                    opts.maxits = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid max_it value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-ksp_gmres_restart" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_gmres_restart".to_string()));
                    }
                    opts.restart = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid restart value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-ksp_pc_side" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_pc_side".to_string()));
                    }
                    opts.pc_side = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-matrix" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -matrix".to_string()));
                    }
                    opts.matrix_file = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-rhs" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -rhs".to_string()));
                    }
                    opts.rhs_file = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-ksp_min_iter" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_min_iter".to_string()));
                    }
                    opts.min_iter = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid min_iter value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-ksp_cf_tol" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_cf_tol".to_string()));
                    }
                    opts.cf_tol = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid cf_tol value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-ksp_skip_real_r_check" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_skip_real_r_check".to_string()));
                    }
                    // Parse boolean values flexibly (true/false, yes/no, on/off, 1/0)
                    let val_str = args[i + 1].to_lowercase();
                    let bool_val = match val_str.as_str() {
                        "true" | "yes" | "on" | "1" => true,
                        "false" | "no" | "off" | "0" => false,
                        _ => return Err(KError::SolveError(format!("Invalid boolean value for -ksp_skip_real_r_check: {}", args[i + 1]))),
                    };
                    opts.skip_real_r_check = Some(bool_val);
                    i += 2;
                }
                "-ksp_epsmac" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_epsmac".to_string()));
                    }
                    opts.epsmac = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid epsmac value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-ksp_guard_zero_residual" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -ksp_guard_zero_residual".to_string()));
                    }
                    opts.guard_zero_residual = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid guard_zero_residual value: {}", args[i + 1])))?);
                    i += 2;
                }
                arg if arg.starts_with("-ksp_") => {
                    return Err(KError::SolveError(format!("Unrecognized KSP option: {}", arg)));
                }
                _ => {
                    i += 1; // Skip non-KSP arguments
                }
            }
        }
        
        Ok(opts)
    }

    /// Parse KSP options from string arguments (convenience method).
    pub fn from_strings(args: &[String]) -> Result<Self, KError> {
        let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        Self::from_args(&str_args)
    }

    /// Parse KSP options from environment variables.
    ///
    /// Checks for environment variables like KRYST_KSP_TYPE, KRYST_KSP_RTOL, etc.
    pub fn from_env() -> Result<Self, KError> {
        let mut opts = Self::new();
        
        if let Ok(val) = std::env::var("KRYST_KSP_TYPE") {
            opts.ksp_type = Some(val);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_RTOL") {
            opts.rtol = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_RTOL: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_ATOL") {
            opts.atol = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_ATOL: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_DTOL") {
            opts.dtol = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_DTOL: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_MAX_IT") {
            opts.maxits = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_MAX_IT: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_GMRES_RESTART") {
            opts.restart = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_GMRES_RESTART: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_PC_SIDE") {
            opts.pc_side = Some(val);
        }
        if let Ok(val) = std::env::var("KRYST_MATRIX_FILE") {
            opts.matrix_file = Some(val);
        }
        if let Ok(val) = std::env::var("KRYST_RHS_FILE") {
            opts.rhs_file = Some(val);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_MIN_ITER") {
            opts.min_iter = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_MIN_ITER: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_CF_TOL") {
            opts.cf_tol = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_CF_TOL: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_SKIP_REAL_R_CHECK") {
            let val_lower = val.to_lowercase();
            let bool_val = match val_lower.as_str() {
                "true" | "yes" | "on" | "1" => true,
                "false" | "no" | "off" | "0" => false,
                _ => return Err(KError::SolveError(format!("Invalid boolean value for KRYST_KSP_SKIP_REAL_R_CHECK: {}", val))),
            };
            opts.skip_real_r_check = Some(bool_val);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_EPSMAC") {
            opts.epsmac = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_EPSMAC: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_KSP_GUARD_ZERO_RESIDUAL") {
            opts.guard_zero_residual = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_KSP_GUARD_ZERO_RESIDUAL: {}", val)))?);
        }
        
        Ok(opts)
    }
}

impl PcOptions {
    /// Create new PcOptions with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse PC options from command-line arguments.
    ///
    /// # Arguments
    /// * `args` - Command-line arguments (typically from `std::env::args()`)
    ///
    /// # Returns
    /// * `Ok(PcOptions)` with parsed options
    /// * `Err(KError)` if parsing fails
    ///
    /// # Example
    /// ```rust,ignore
    /// let args = vec!["-pc_type", "jacobi", "-pc_ilu_levels", "2"];
    /// let opts = PcOptions::from_args(&args)?;
    /// assert_eq!(opts.pc_type, Some("jacobi".to_string()));
    /// assert_eq!(opts.ilu_level, Some(2));
    /// ```
    pub fn from_args(args: &[&str]) -> Result<Self, KError> {
        let mut opts = Self::new();
        let mut i = 0;
        
        while i < args.len() {
            let arg = args[i];
            
            match arg {
                "-pc_type" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_type".to_string()));
                    }
                    opts.pc_type = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-pc_ilu_levels" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_levels".to_string()));
                    }
                    opts.ilu_level = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_levels value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_chebyshev_degree" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_chebyshev_degree".to_string()));
                    }
                    opts.chebyshev_degree = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid chebyshev_degree value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilut_drop_tol" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilut_drop_tol".to_string()));
                    }
                    opts.ilut_drop_tol = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilut_drop_tol value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilut_max_fill" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilut_max_fill".to_string()));
                    }
                    opts.ilut_max_fill = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilut_max_fill value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilut_perm_tol" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilut_perm_tol".to_string()));
                    }
                    opts.ilut_perm_tol = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilut_perm_tol value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_reorder" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_reorder".to_string()));
                    }
                    let reorder_type = args[i + 1].to_lowercase();
                    match reorder_type.as_str() {
                        "none" | "colamd" | "amd" | "rcm" | "cuthill_mckee" => {
                            opts.reorder = Some(reorder_type);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid reorder type: {}. Use 'none', 'colamd', 'amd', 'rcm', or 'cuthill_mckee'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_scaling" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_scaling".to_string()));
                    }
                    let scaling_type = args[i + 1].to_lowercase();
                    match scaling_type.as_str() {
                        "none" | "diagonal" | "symmetric" => {
                            opts.scaling = Some(scaling_type);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid scaling type: {}. Use 'none', 'diagonal', or 'symmetric'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_asm_overlap" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_asm_overlap".to_string()));
                    }
                    opts.asm_overlap = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid asm_overlap value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_asm_subdomains" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_asm_subdomains".to_string()));
                    }
                    // Parse comma-separated list of subdomain indices
                    let subdomains: Result<Vec<usize>, _> = args[i + 1]
                        .split(',')
                        .map(|s| s.trim().parse())
                        .collect();
                    opts.asm_subdomains = Some(subdomains
                        .map_err(|_| KError::SolveError(format!("Invalid asm_subdomains value: {}. Use comma-separated indices like '0,1,2'", args[i + 1])))?);
                    i += 2;
                }
                "-pc_asm_inner_pc" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_asm_inner_pc".to_string()));
                    }
                    let inner_pc = args[i + 1].to_lowercase();
                    match inner_pc.as_str() {
                        "jacobi" | "ilu" | "ilut" | "ilutp" => {
                            opts.asm_inner_pc = Some(inner_pc);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid asm_inner_pc type: {}. Use 'jacobi', 'ilu', 'ilut', or 'ilutp'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_chebyshev_lambda_min" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_chebyshev_lambda_min".to_string()));
                    }
                    opts.chebyshev_lambda_min = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid chebyshev_lambda_min value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_chebyshev_lambda_max" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_chebyshev_lambda_max".to_string()));
                    }
                    opts.chebyshev_lambda_max = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid chebyshev_lambda_max value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_levels" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_levels".to_string()));
                    }
                    opts.amg_levels = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_levels value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_strength_threshold" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_strength_threshold".to_string()));
                    }
                    opts.amg_strength_threshold = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_strength_threshold value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_nu_pre" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_nu_pre".to_string()));
                    }
                    opts.amg_nu_pre = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_nu_pre value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_nu_post" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_nu_post".to_string()));
                    }
                    opts.amg_nu_post = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_nu_post value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_coarse_threshold" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_coarse_threshold".to_string()));
                    }
                    opts.amg_coarse_threshold = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_coarse_threshold value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_max_coarse_size" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_max_coarse_size".to_string()));
                    }
                    opts.amg_max_coarse_size = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_max_coarse_size value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_min_coarse_size" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_min_coarse_size".to_string()));
                    }
                    opts.amg_min_coarse_size = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_min_coarse_size value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_truncation_factor" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_truncation_factor".to_string()));
                    }
                    opts.amg_truncation_factor = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_truncation_factor value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_max_elements_per_row" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_max_elements_per_row".to_string()));
                    }
                    opts.amg_max_elements_per_row = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_max_elements_per_row value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_interpolation_truncation" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_interpolation_truncation".to_string()));
                    }
                    opts.amg_interpolation_truncation = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_interpolation_truncation value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_coarsen_type" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_coarsen_type".to_string()));
                    }
                    let coarsen_type = args[i + 1].to_lowercase();
                    match coarsen_type.as_str() {
                        "rs" | "hmis" | "pmis" | "falgout" => {
                            opts.amg_coarsen_type = Some(coarsen_type);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid amg_coarsen_type: {}. Use 'rs', 'hmis', 'pmis', or 'falgout'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_amg_interp_type" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_interp_type".to_string()));
                    }
                    let interp_type = args[i + 1].to_lowercase();
                    match interp_type.as_str() {
                        "classical" | "direct" | "multipass" | "extended" | "standard" => {
                            opts.amg_interp_type = Some(interp_type);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid amg_interp_type: {}. Use 'classical', 'direct', 'multipass', 'extended', or 'standard'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_amg_relax_type" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_relax_type".to_string()));
                    }
                    let relax_type = args[i + 1].to_lowercase();
                    match relax_type.as_str() {
                        "jacobi" | "gs" | "gsr" | "sgs" | "hgs" | "l1jacobi" | "chebyshev" => {
                            opts.amg_relax_type = Some(relax_type);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid amg_relax_type: {}. Use 'jacobi', 'gs', 'gsr', 'sgs', 'hgs', 'l1jacobi', or 'chebyshev'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_amg_logging_level" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_logging_level".to_string()));
                    }
                    opts.amg_logging_level = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_logging_level value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_print_level" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_print_level".to_string()));
                    }
                    opts.amg_print_level = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_print_level value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_tolerance" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_tolerance".to_string()));
                    }
                    opts.amg_tolerance = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_tolerance value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_max_iterations" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_max_iterations".to_string()));
                    }
                    opts.amg_max_iterations = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_max_iterations value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_min_iterations" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_min_iterations".to_string()));
                    }
                    opts.amg_min_iterations = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid amg_min_iterations value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_amg_ieee_checks" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_ieee_checks".to_string()));
                    }
                    let value = args[i + 1].to_lowercase();
                    match value.as_str() {
                        "true" | "1" | "yes" | "on" => opts.amg_ieee_checks = Some(true),
                        "false" | "0" | "no" | "off" => opts.amg_ieee_checks = Some(false),
                        _ => {
                            return Err(KError::SolveError(format!("Invalid amg_ieee_checks value: {}. Use 'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_amg_optimize_workspace" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_amg_optimize_workspace".to_string()));
                    }
                    let value = args[i + 1].to_lowercase();
                    match value.as_str() {
                        "true" | "1" | "yes" | "on" => opts.amg_optimize_workspace = Some(true),
                        "false" | "0" | "no" | "off" => opts.amg_optimize_workspace = Some(false),
                        _ => {
                            return Err(KError::SolveError(format!("Invalid amg_optimize_workspace value: {}. Use 'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_chain" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_chain".to_string()));
                    }
                    opts.pc_chain = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-pc_ilu_type" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_type".to_string()));
                    }
                    let ilu_type = args[i + 1].to_lowercase();
                    match ilu_type.as_str() {
                        "ilu0" | "iluk" | "ilut" | "milu0" | "block_jacobi" | "gmres_iluk" | "gmres_ilut" => {
                            opts.ilu_type = Some(ilu_type);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid ilu_type: {}. Use 'ilu0', 'iluk', 'ilut', 'milu0', 'block_jacobi', 'gmres_iluk', or 'gmres_ilut'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_ilu_level_of_fill" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_level_of_fill".to_string()));
                    }
                    opts.ilu_level_of_fill = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_level_of_fill value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_max_fill_per_row" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_max_fill_per_row".to_string()));
                    }
                    opts.ilu_max_fill_per_row = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_max_fill_per_row value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_offdiag_drop_tolerance" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_offdiag_drop_tolerance".to_string()));
                    }
                    opts.ilu_offdiag_drop_tolerance = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_offdiag_drop_tolerance value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_schur_drop_tolerance" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_schur_drop_tolerance".to_string()));
                    }
                    opts.ilu_schur_drop_tolerance = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_schur_drop_tolerance value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_reordering_type" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_reordering_type".to_string()));
                    }
                    let reordering_type = args[i + 1].to_lowercase();
                    match reordering_type.as_str() {
                        "none" | "rcm" | "amd" | "natural" => {
                            opts.ilu_reordering_type = Some(reordering_type);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid ilu_reordering_type: {}. Use 'none', 'rcm', 'amd', or 'natural'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_ilu_triangular_solve" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_triangular_solve".to_string()));
                    }
                    let solve_type = args[i + 1].to_lowercase();
                    match solve_type.as_str() {
                        "exact" | "iterative" => {
                            opts.ilu_triangular_solve = Some(solve_type);
                        }
                        _ => {
                            return Err(KError::SolveError(format!("Invalid ilu_triangular_solve: {}. Use 'exact' or 'iterative'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_ilu_lower_jacobi_iters" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_lower_jacobi_iters".to_string()));
                    }
                    opts.ilu_lower_jacobi_iters = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_lower_jacobi_iters value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_upper_jacobi_iters" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_upper_jacobi_iters".to_string()));
                    }
                    opts.ilu_upper_jacobi_iters = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_upper_jacobi_iters value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_tolerance" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_tolerance".to_string()));
                    }
                    opts.ilu_tolerance = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_tolerance value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_max_iterations" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_max_iterations".to_string()));
                    }
                    opts.ilu_max_iterations = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_max_iterations value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_logging_level" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_logging_level".to_string()));
                    }
                    opts.ilu_logging_level = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_logging_level value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_print_level" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_print_level".to_string()));
                    }
                    opts.ilu_print_level = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_print_level value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_ilu_ieee_checks" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_ieee_checks".to_string()));
                    }
                    let value = args[i + 1].to_lowercase();
                    match value.as_str() {
                        "true" | "1" | "yes" | "on" => opts.ilu_ieee_checks = Some(true),
                        "false" | "0" | "no" | "off" => opts.ilu_ieee_checks = Some(false),
                        _ => {
                            return Err(KError::SolveError(format!("Invalid ilu_ieee_checks value: {}. Use 'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_ilu_pivot_monitoring" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_pivot_monitoring".to_string()));
                    }
                    let value = args[i + 1].to_lowercase();
                    match value.as_str() {
                        "true" | "1" | "yes" | "on" => opts.ilu_pivot_monitoring = Some(true),
                        "false" | "0" | "no" | "off" => opts.ilu_pivot_monitoring = Some(false),
                        _ => {
                            return Err(KError::SolveError(format!("Invalid ilu_pivot_monitoring value: {}. Use 'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_ilu_optimize_workspace" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_optimize_workspace".to_string()));
                    }
                    let value = args[i + 1].to_lowercase();
                    match value.as_str() {
                        "true" | "1" | "yes" | "on" => opts.ilu_optimize_workspace = Some(true),
                        "false" | "0" | "no" | "off" => opts.ilu_optimize_workspace = Some(false),
                        _ => {
                            return Err(KError::SolveError(format!("Invalid ilu_optimize_workspace value: {}. Use 'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_ilu_pivot_threshold" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_ilu_pivot_threshold".to_string()));
                    }
                    opts.ilu_pivot_threshold = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid ilu_pivot_threshold value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_superlu_pivot_threshold" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_superlu_pivot_threshold".to_string()));
                    }
                    opts.superlu_pivot_threshold = Some(args[i + 1].parse()
                        .map_err(|_| KError::SolveError(format!("Invalid superlu_pivot_threshold value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_superlu_process_grid" => {
                    if i + 2 >= args.len() {
                        return Err(KError::SolveError("Missing values for -pc_superlu_process_grid (requires rows and cols)".to_string()));
                    }
                    let rows = args[i + 1].parse::<usize>()
                        .map_err(|_| KError::SolveError(format!("Invalid process grid rows value: {}", args[i + 1])))?;
                    let cols = args[i + 2].parse::<usize>()
                        .map_err(|_| KError::SolveError(format!("Invalid process grid cols value: {}", args[i + 2])))?;
                    opts.superlu_process_grid = Some((rows, cols));
                    i += 3;
                }
                "-pc_superlu_print_level" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_superlu_print_level".to_string()));
                    }
                    opts.superlu_print_level = Some(args[i + 1].parse::<u8>()
                        .map_err(|_| KError::SolveError(format!("Invalid superlu_print_level value: {}", args[i + 1])))?);
                    i += 2;
                }
                "-pc_superlu_replace_tiny_pivot" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_superlu_replace_tiny_pivot".to_string()));
                    }
                    let value = args[i + 1].to_lowercase();
                    match value.as_str() {
                        "true" | "1" | "yes" | "on" => opts.superlu_replace_tiny_pivots = Some(true),
                        "false" | "0" | "no" | "off" => opts.superlu_replace_tiny_pivots = Some(false),
                        _ => {
                            return Err(KError::SolveError(format!("Invalid superlu_replace_tiny_pivot value: {}. Use 'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                "-pc_superlu_iterative_refinement" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_superlu_iterative_refinement".to_string()));
                    }
                    opts.superlu_iterative_refinement = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-pc_superlu_column_permutation" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_superlu_column_permutation".to_string()));
                    }
                    opts.superlu_column_permutation = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-pc_superlu_row_permutation" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_superlu_row_permutation".to_string()));
                    }
                    opts.superlu_row_permutation = Some(args[i + 1].to_string());
                    i += 2;
                }
                "-pc_superlu_static_pivoting" => {
                    if i + 1 >= args.len() {
                        return Err(KError::SolveError("Missing value for -pc_superlu_static_pivoting".to_string()));
                    }
                    let value = args[i + 1].to_lowercase();
                    match value.as_str() {
                        "true" | "1" | "yes" | "on" => opts.superlu_static_pivoting = Some(true),
                        "false" | "0" | "no" | "off" => opts.superlu_static_pivoting = Some(false),
                        _ => {
                            return Err(KError::SolveError(format!("Invalid superlu_static_pivoting value: {}. Use 'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'", args[i + 1])));
                        }
                    }
                    i += 2;
                }
                arg if arg.starts_with("-pc_") => {
                    return Err(KError::SolveError(format!("Unrecognized PC option: {}", arg)));
                }
                _ => {
                    i += 1; // Skip non-PC arguments
                }
            }
        }
        
        Ok(opts)
    }

    /// Parse PC options from string arguments (convenience method).
    pub fn from_strings(args: &[String]) -> Result<Self, KError> {
        let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        Self::from_args(&str_args)
    }

    /// Parse PC options from environment variables.
    ///
    /// Checks for environment variables like KRYST_PC_TYPE, KRYST_PC_ILU_LEVELS, etc.
    pub fn from_env() -> Result<Self, KError> {
        let mut opts = Self::new();
        
        if let Ok(val) = std::env::var("KRYST_PC_TYPE") {
            opts.pc_type = Some(val);
        }
        if let Ok(val) = std::env::var("KRYST_PC_ILU_LEVELS") {
            opts.ilu_level = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_ILU_LEVELS: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_PC_CHEBYSHEV_DEGREE") {
            opts.chebyshev_degree = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_CHEBYSHEV_DEGREE: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_PC_ILUT_DROP_TOL") {
            opts.ilut_drop_tol = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_ILUT_DROP_TOL: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_PC_ILUT_MAX_FILL") {
            opts.ilut_max_fill = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_ILUT_MAX_FILL: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_PC_ILUT_PERM_TOL") {
            opts.ilut_perm_tol = Some(val.parse()
                .map_err(|_| KError::SolveError(format!("Invalid KRYST_PC_ILUT_PERM_TOL: {}", val)))?);
        }
        if let Ok(val) = std::env::var("KRYST_PC_REORDER") {
            let reorder_type = val.to_lowercase();
            match reorder_type.as_str() {
                "none" | "colamd" | "amd" => {
                    opts.reorder = Some(reorder_type);
                }
                _ => {
                    return Err(KError::SolveError(format!("Invalid KRYST_PC_REORDER: {}. Use 'none', 'colamd', or 'amd'", val)));
                }
            }
        }
        
        Ok(opts)
    }
}

/// Print help information for all supported options.
pub fn print_help() {
    println!("Kryst Linear Solver Options:");
    println!();
    println!("KSP (Krylov Solver) Options:");
    println!("  -ksp_type <solver>         Solver type: cg, pcg, gmres, fgmres, bicgstab, cgs, qmr, tfqmr, minres, cgnr, preonly");
    println!("  -ksp_rtol <float>          Relative convergence tolerance (default: 1e-6)");
    println!("  -ksp_atol <float>          Absolute convergence tolerance (default: 1e-12)");
    println!("  -ksp_dtol <float>          Divergence tolerance (default: 1e3)");
    println!("  -ksp_max_it <int>          Maximum number of iterations (default: 1000)");
    println!("  -ksp_gmres_restart <int>   GMRES restart parameter (default: 50)");
    println!("  -ksp_pc_side <side>        Preconditioning side: left, right, symmetric (default: left)");
    println!();
    println!("GMRES Advanced Options (HYPRE-inspired):");
    println!("  -ksp_min_iter <int>        Minimum iterations before convergence check (default: 0)");
    println!("  -ksp_cf_tol <float>        Convergence factor tolerance for stagnation detection (default: 0.0=disabled)");
    println!("  -ksp_skip_real_r_check <bool>  Skip real residual check for performance (default: false)");
    println!("  -ksp_epsmac <float>        IEEE safety epsilon for breakdown protection (default: 1e-16)");
    println!("  -ksp_guard_zero_residual <float>  Guard for zero residual to prevent NaN (default: 0.0)");
    println!();
    println!("Problem Configuration:");
    println!("  -matrix <path>             Matrix file path (Matrix Market format)");
    println!("  -rhs <path>                RHS vector file path (Matrix Market format)");
    println!();
    println!("PC (Preconditioner) Options:");
    println!("  -pc_type <pc>              Preconditioner type: jacobi, ilu0, none, amg, asm, chebyshev, lu, qr, superlu_dist");
    println!("  -pc_ilu_levels <int>       ILU fill levels (default: 0)");
    println!("  -pc_chebyshev_degree <int> Chebyshev polynomial degree (default: 3)");
    println!("  -pc_ilut_drop_tol <float>  ILUT drop tolerance (default: 1e-3)");
    println!("  -pc_ilut_max_fill <int>    ILUT maximum fill per row (default: 10)");
    println!("  -pc_ilut_perm_tol <float>  ILUTP pivot tolerance (default: 1e-3)");
    println!("  -pc_reorder <type>         Matrix reordering: none, colamd, amd, rcm, cuthill_mckee");
    println!("  -pc_scaling <type>         Matrix scaling: none, diagonal, symmetric");
    println!();
    println!("ASM (Additive Schwarz Method) Options:");
    println!("  -pc_asm_overlap <int>      Overlap layers for ASM (default: 1)");
    println!("  -pc_asm_subdomains <list>  Subdomain indices (comma-separated)");
    println!("  -pc_asm_inner_pc <pc>      Inner preconditioner: jacobi, ilu, ilut, ilutp");
    println!();
    println!("Chebyshev Preconditioner Options:");
    println!("  -pc_chebyshev_lambda_min <float>  Minimum eigenvalue estimate");
    println!("  -pc_chebyshev_lambda_max <float>  Maximum eigenvalue estimate");
    println!();
    println!("AMG (Algebraic Multigrid) Options:");
    println!("  -pc_amg_levels <int>               Number of coarsening levels (default: 25)");
    println!("  -pc_amg_strength_threshold <float> Strength-of-connection threshold (default: 0.25)");
    println!("  -pc_amg_nu_pre <int>              Pre-smoothing iterations (default: 1)");
    println!("  -pc_amg_nu_post <int>             Post-smoothing iterations (default: 1)");
    println!("  -pc_amg_coarse_threshold <int>    Coarse grid threshold (default: 9)");
    println!("  -pc_amg_max_coarse_size <int>     Maximum coarse grid size (default: 9)");
    println!("  -pc_amg_min_coarse_size <int>     Minimum coarse grid size (default: 1)");
    println!("  -pc_amg_truncation_factor <float> Interpolation truncation factor (default: 0.0)");
    println!("  -pc_amg_max_elements_per_row <int> Max elements per row for interpolation (default: 0)");
    println!("  -pc_amg_interpolation_truncation <float> Interpolation truncation (default: 0.0)");
    println!("  -pc_amg_coarsen_type <type>       Coarsening algorithm: rs, hmis, pmis, falgout");
    println!("  -pc_amg_interp_type <type>        Interpolation algorithm: classical, direct, multipass, extended, standard");
    println!("  -pc_amg_relax_type <type>         Relaxation type: jacobi, gs, gsr, sgs, hgs, l1jacobi, chebyshev");
    println!("  -pc_amg_logging_level <int>       Logging level: 0=none, 1=basic, 2=detailed");
    println!("  -pc_amg_print_level <int>         Print level: 0=none, 1=basic, 2=detailed");
    println!("  -pc_amg_tolerance <float>         AMG convergence tolerance (for standalone use)");
    println!("  -pc_amg_max_iterations <int>      AMG maximum iterations (for standalone use)");
    println!("  -pc_amg_min_iterations <int>      AMG minimum iterations");
    println!("  -pc_amg_ieee_checks <bool>        Enable IEEE safety checks (true/false)");
    println!("  -pc_amg_optimize_workspace <bool> Enable workspace optimization (true/false)");
    println!();
    println!("Advanced Options:");
    println!("  -pc_chain <list>           Preconditioner chain (comma-separated)");
    println!();
    println!("ILU (Incomplete LU) Preconditioner Options:");
    println!("  -pc_ilu_type <type>                    ILU factorization type: ilu0, iluk, ilut, milu0, block_jacobi, gmres_iluk, gmres_ilut");
    println!("  -pc_ilu_level_of_fill <int>            Level of fill for ILU(k) (default: 0)");
    println!("  -pc_ilu_max_fill_per_row <int>         Maximum nonzeros per row (default: 0=unlimited)");
    println!("  -pc_ilu_offdiag_drop_tolerance <float> Drop tolerance for off-diagonal blocks (default: 1e-4)");
    println!("  -pc_ilu_schur_drop_tolerance <float>   Drop tolerance for Schur complement (default: 1e-4)");
    println!("  -pc_ilu_reordering_type <type>         Reordering strategy: none, rcm, amd, natural");
    println!("  -pc_ilu_triangular_solve <type>        Triangular solve type: exact, iterative");
    println!("  -pc_ilu_lower_jacobi_iters <int>       Lower triangular Jacobi iterations (default: 1)");
    println!("  -pc_ilu_upper_jacobi_iters <int>       Upper triangular Jacobi iterations (default: 1)");
    println!("  -pc_ilu_tolerance <float>              Tolerance for iterative ILU solve (default: 1e-6)");
    println!("  -pc_ilu_max_iterations <int>           Maximum iterations for iterative ILU solve (default: 1)");
    println!("  -pc_ilu_logging_level <int>            ILU logging level: 0=none, 1=basic, 2=detailed");
    println!("  -pc_ilu_print_level <int>              ILU print level: 0=none, 1=basic, 2=detailed");
    println!("  -pc_ilu_ieee_checks <bool>             Enable ILU IEEE safety checks (true/false)");
    println!("  -pc_ilu_pivot_monitoring <bool>        Enable ILU pivot monitoring (true/false)");
    println!("  -pc_ilu_optimize_workspace <bool>      Enable ILU workspace optimization (true/false)");
    println!("  -pc_ilu_pivot_threshold <float>        ILU pivot threshold for stability (default: 1e-12)");
    println!();
    println!("SuperLU_DIST (Distributed Direct Solver) Preconditioner Options:");
    println!("  -pc_superlu_pivot_threshold <float>    SuperLU_DIST diagonal pivot threshold (0.0-1.0)");
    println!("  -pc_superlu_print_level <int>          SuperLU_DIST print level: 0=none, 1=basic, 2=detailed");
    println!("  -pc_superlu_process_grid <rows> <cols> SuperLU_DIST process grid dimensions");
    println!("  -pc_superlu_replace_tiny_pivot <bool>  Replace tiny pivots in SuperLU_DIST (true/false)");
    println!("  -pc_superlu_iterative_refinement <str> SuperLU_DIST iterative refinement method");
    println!("  -pc_superlu_column_permutation <str>   SuperLU_DIST column permutation strategy");
    println!("  -pc_superlu_row_permutation <str>      SuperLU_DIST row permutation strategy");
    println!("  -pc_superlu_static_pivoting <bool>     Use static pivoting in SuperLU_DIST (true/false)");
    println!();
    println!("Examples:");
    println!("  -ksp_type gmres -ksp_rtol 1e-8 -pc_type jacobi");
    println!("  -ksp_type cg -ksp_max_it 500 -pc_type ilu0 -pc_ilu_levels 2");
    println!("  -ksp_type bicgstab -pc_type amg -pc_amg_levels 10 -pc_amg_strength_threshold 0.5");
    println!("  -ksp_type gmres -pc_type asm -pc_asm_overlap 2 -pc_asm_inner_pc ilu");
    println!("  -ksp_type cg -pc_type ilu -pc_ilu_type ilut -pc_ilu_tolerance 1e-4 -pc_ilu_triangular_solve iterative");
    println!("  -ksp_type gmres -pc_type ilu -pc_ilu_type iluk -pc_ilu_level_of_fill 3 -pc_ilu_reordering_type rcm");
    println!("  -ksp_type preonly -pc_type superlu_dist -pc_superlu_pivot_threshold 0.1 -pc_superlu_process_grid 2 2");
}

/// Check if help is requested in the arguments.
pub fn is_help_requested(args: &[&str]) -> bool {
    args.iter().any(|&arg| arg == "-help" || arg == "--help" || arg == "-h")
}

/// Parse both KSP and PC options from command-line arguments with precedence.
///
/// Precedence order: Command-line > Environment variables > Defaults
pub fn parse_all_options(args: &[String]) -> Result<(KspOptions, PcOptions), KError> {
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    
    // Check for help request
    if is_help_requested(&str_args) {
        print_help();
        std::process::exit(0);
    }
    
    // Parse from environment first (lower precedence)
    let mut ksp_opts = KspOptions::from_env()?;
    let mut pc_opts = PcOptions::from_env()?;
    
    // Parse from command line (higher precedence) and override env vars
    let cli_ksp_opts = KspOptions::from_args(&str_args)?;
    let cli_pc_opts = PcOptions::from_args(&str_args)?;
    
    // Apply CLI options over environment options
    if cli_ksp_opts.ksp_type.is_some() { ksp_opts.ksp_type = cli_ksp_opts.ksp_type; }
    if cli_ksp_opts.rtol.is_some() { ksp_opts.rtol = cli_ksp_opts.rtol; }
    if cli_ksp_opts.atol.is_some() { ksp_opts.atol = cli_ksp_opts.atol; }
    if cli_ksp_opts.dtol.is_some() { ksp_opts.dtol = cli_ksp_opts.dtol; }
    if cli_ksp_opts.maxits.is_some() { ksp_opts.maxits = cli_ksp_opts.maxits; }
    if cli_ksp_opts.restart.is_some() { ksp_opts.restart = cli_ksp_opts.restart; }
    if cli_ksp_opts.pc_side.is_some() { ksp_opts.pc_side = cli_ksp_opts.pc_side; }
    if cli_ksp_opts.matrix_file.is_some() { ksp_opts.matrix_file = cli_ksp_opts.matrix_file; }
    if cli_ksp_opts.rhs_file.is_some() { ksp_opts.rhs_file = cli_ksp_opts.rhs_file; }
    if cli_ksp_opts.min_iter.is_some() { ksp_opts.min_iter = cli_ksp_opts.min_iter; }
    if cli_ksp_opts.cf_tol.is_some() { ksp_opts.cf_tol = cli_ksp_opts.cf_tol; }
    if cli_ksp_opts.skip_real_r_check.is_some() { ksp_opts.skip_real_r_check = cli_ksp_opts.skip_real_r_check; }
    if cli_ksp_opts.epsmac.is_some() { ksp_opts.epsmac = cli_ksp_opts.epsmac; }
    if cli_ksp_opts.guard_zero_residual.is_some() { ksp_opts.guard_zero_residual = cli_ksp_opts.guard_zero_residual; }
    
    if cli_pc_opts.pc_type.is_some() { pc_opts.pc_type = cli_pc_opts.pc_type; }
    if cli_pc_opts.ilu_level.is_some() { pc_opts.ilu_level = cli_pc_opts.ilu_level; }
    if cli_pc_opts.chebyshev_degree.is_some() { pc_opts.chebyshev_degree = cli_pc_opts.chebyshev_degree; }
    if cli_pc_opts.ilut_drop_tol.is_some() { pc_opts.ilut_drop_tol = cli_pc_opts.ilut_drop_tol; }
    if cli_pc_opts.ilut_max_fill.is_some() { pc_opts.ilut_max_fill = cli_pc_opts.ilut_max_fill; }
    if cli_pc_opts.ilut_perm_tol.is_some() { pc_opts.ilut_perm_tol = cli_pc_opts.ilut_perm_tol; }
    if cli_pc_opts.reorder.is_some() { pc_opts.reorder = cli_pc_opts.reorder; }
    if cli_pc_opts.scaling.is_some() { pc_opts.scaling = cli_pc_opts.scaling; }
    if cli_pc_opts.asm_overlap.is_some() { pc_opts.asm_overlap = cli_pc_opts.asm_overlap; }
    if cli_pc_opts.asm_subdomains.is_some() { pc_opts.asm_subdomains = cli_pc_opts.asm_subdomains; }
    if cli_pc_opts.asm_inner_pc.is_some() { pc_opts.asm_inner_pc = cli_pc_opts.asm_inner_pc; }
    if cli_pc_opts.chebyshev_lambda_min.is_some() { pc_opts.chebyshev_lambda_min = cli_pc_opts.chebyshev_lambda_min; }
    if cli_pc_opts.chebyshev_lambda_max.is_some() { pc_opts.chebyshev_lambda_max = cli_pc_opts.chebyshev_lambda_max; }
    if cli_pc_opts.amg_levels.is_some() { pc_opts.amg_levels = cli_pc_opts.amg_levels; }
    if cli_pc_opts.amg_strength_threshold.is_some() { pc_opts.amg_strength_threshold = cli_pc_opts.amg_strength_threshold; }
    if cli_pc_opts.amg_nu_pre.is_some() { pc_opts.amg_nu_pre = cli_pc_opts.amg_nu_pre; }
    if cli_pc_opts.amg_nu_post.is_some() { pc_opts.amg_nu_post = cli_pc_opts.amg_nu_post; }
    if cli_pc_opts.amg_coarse_threshold.is_some() { pc_opts.amg_coarse_threshold = cli_pc_opts.amg_coarse_threshold; }
    if cli_pc_opts.amg_max_coarse_size.is_some() { pc_opts.amg_max_coarse_size = cli_pc_opts.amg_max_coarse_size; }
    if cli_pc_opts.amg_min_coarse_size.is_some() { pc_opts.amg_min_coarse_size = cli_pc_opts.amg_min_coarse_size; }
    if cli_pc_opts.amg_truncation_factor.is_some() { pc_opts.amg_truncation_factor = cli_pc_opts.amg_truncation_factor; }
    if cli_pc_opts.amg_max_elements_per_row.is_some() { pc_opts.amg_max_elements_per_row = cli_pc_opts.amg_max_elements_per_row; }
    if cli_pc_opts.amg_interpolation_truncation.is_some() { pc_opts.amg_interpolation_truncation = cli_pc_opts.amg_interpolation_truncation; }
    if cli_pc_opts.amg_coarsen_type.is_some() { pc_opts.amg_coarsen_type = cli_pc_opts.amg_coarsen_type; }
    if cli_pc_opts.amg_interp_type.is_some() { pc_opts.amg_interp_type = cli_pc_opts.amg_interp_type; }
    if cli_pc_opts.amg_relax_type.is_some() { pc_opts.amg_relax_type = cli_pc_opts.amg_relax_type; }
    if cli_pc_opts.amg_logging_level.is_some() { pc_opts.amg_logging_level = cli_pc_opts.amg_logging_level; }
    if cli_pc_opts.amg_print_level.is_some() { pc_opts.amg_print_level = cli_pc_opts.amg_print_level; }
    if cli_pc_opts.amg_tolerance.is_some() { pc_opts.amg_tolerance = cli_pc_opts.amg_tolerance; }
    if cli_pc_opts.amg_max_iterations.is_some() { pc_opts.amg_max_iterations = cli_pc_opts.amg_max_iterations; }
    if cli_pc_opts.amg_min_iterations.is_some() { pc_opts.amg_min_iterations = cli_pc_opts.amg_min_iterations; }
    if cli_pc_opts.amg_ieee_checks.is_some() { pc_opts.amg_ieee_checks = cli_pc_opts.amg_ieee_checks; }
    if cli_pc_opts.amg_optimize_workspace.is_some() { pc_opts.amg_optimize_workspace = cli_pc_opts.amg_optimize_workspace; }
    if cli_pc_opts.pc_chain.is_some() { pc_opts.pc_chain = cli_pc_opts.pc_chain; }
    if cli_pc_opts.omega.is_some() { pc_opts.omega = cli_pc_opts.omega; }
    if cli_pc_opts.drop_tol.is_some() { pc_opts.drop_tol = cli_pc_opts.drop_tol; }
    if cli_pc_opts.ilu_type.is_some() { pc_opts.ilu_type = cli_pc_opts.ilu_type; }
    if cli_pc_opts.ilu_level_of_fill.is_some() { pc_opts.ilu_level_of_fill = cli_pc_opts.ilu_level_of_fill; }
    if cli_pc_opts.ilu_max_fill_per_row.is_some() { pc_opts.ilu_max_fill_per_row = cli_pc_opts.ilu_max_fill_per_row; }
    if cli_pc_opts.ilu_offdiag_drop_tolerance.is_some() { pc_opts.ilu_offdiag_drop_tolerance = cli_pc_opts.ilu_offdiag_drop_tolerance; }
    if cli_pc_opts.ilu_schur_drop_tolerance.is_some() { pc_opts.ilu_schur_drop_tolerance = cli_pc_opts.ilu_schur_drop_tolerance; }
    if cli_pc_opts.ilu_reordering_type.is_some() { pc_opts.ilu_reordering_type = cli_pc_opts.ilu_reordering_type; }
    if cli_pc_opts.ilu_triangular_solve.is_some() { pc_opts.ilu_triangular_solve = cli_pc_opts.ilu_triangular_solve; }
    if cli_pc_opts.ilu_lower_jacobi_iters.is_some() { pc_opts.ilu_lower_jacobi_iters = cli_pc_opts.ilu_lower_jacobi_iters; }
    if cli_pc_opts.ilu_upper_jacobi_iters.is_some() { pc_opts.ilu_upper_jacobi_iters = cli_pc_opts.ilu_upper_jacobi_iters; }
    if cli_pc_opts.ilu_tolerance.is_some() { pc_opts.ilu_tolerance = cli_pc_opts.ilu_tolerance; }
    if cli_pc_opts.ilu_max_iterations.is_some() { pc_opts.ilu_max_iterations = cli_pc_opts.ilu_max_iterations; }
    if cli_pc_opts.ilu_logging_level.is_some() { pc_opts.ilu_logging_level = cli_pc_opts.ilu_logging_level; }
    if cli_pc_opts.ilu_print_level.is_some() { pc_opts.ilu_print_level = cli_pc_opts.ilu_print_level; }
    if cli_pc_opts.ilu_ieee_checks.is_some() { pc_opts.ilu_ieee_checks = cli_pc_opts.ilu_ieee_checks; }
    if cli_pc_opts.ilu_pivot_monitoring.is_some() { pc_opts.ilu_pivot_monitoring = cli_pc_opts.ilu_pivot_monitoring; }
    if cli_pc_opts.ilu_optimize_workspace.is_some() { pc_opts.ilu_optimize_workspace = cli_pc_opts.ilu_optimize_workspace; }
    if cli_pc_opts.ilu_pivot_threshold.is_some() { pc_opts.ilu_pivot_threshold = cli_pc_opts.ilu_pivot_threshold; }
    
    Ok((ksp_opts, pc_opts))
}

#[cfg(test)]
mod tests {
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
            "-ksp_type", "cg",
            "-ksp_rtol", "1e-6",
            "-ksp_atol", "1e-12", 
            "-ksp_dtol", "1e3",
            "-ksp_max_it", "1000",
            "-ksp_gmres_restart", "30",
            "-ksp_pc_side", "left"
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
            "-ksp_type", "gmres",
            "-ksp_rtol", "1e-8",
            "-ksp_gmres_restart", "50",
            "-ksp_min_iter", "5",
            "-ksp_cf_tol", "0.9",
            "-ksp_skip_real_r_check", "true",
            "-ksp_epsmac", "1e-15",
            "-ksp_guard_zero_residual", "1e-14"
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
            assert_eq!(opts.skip_real_r_check, Some(expected), "Failed for input: {}", bool_str);
        }
    }

    #[test]
    fn test_ksp_options_gmres_invalid_boolean() {
        let args = vec!["-ksp_skip_real_r_check", "invalid"];
        let result = KspOptions::from_args(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid boolean value"));
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
            "-pc_type", "ilu",
            "-pc_ilu_levels", "5",
            "-pc_chebyshev_degree", "10",
            "-pc_ilut_drop_tol", "1e-4",
            "-pc_ilut_max_fill", "20"
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
            "-pc_type", "amg",
            "-pc_amg_levels", "15",
            "-pc_amg_strength_threshold", "0.5"
        ];
        let opts = PcOptions::from_args(&args).unwrap();
        
        assert_eq!(opts.pc_type, Some("amg".to_string()));
        assert_eq!(opts.amg_levels, Some(15));
        assert_eq!(opts.amg_strength_threshold, Some(0.5));
    }

    #[test]
    fn test_pc_options_amg_comprehensive() {
        let args = vec![
            "-pc_type", "amg",
            "-pc_amg_levels", "20",
            "-pc_amg_strength_threshold", "0.3",
            "-pc_amg_nu_pre", "2",
            "-pc_amg_nu_post", "2",
            "-pc_amg_coarse_threshold", "5",
            "-pc_amg_max_coarse_size", "100",
            "-pc_amg_min_coarse_size", "2",
            "-pc_amg_truncation_factor", "0.1",
            "-pc_amg_max_elements_per_row", "8",
            "-pc_amg_interpolation_truncation", "0.05",
            "-pc_amg_coarsen_type", "hmis",
            "-pc_amg_interp_type", "classical",
            "-pc_amg_relax_type", "gs",
            "-pc_amg_logging_level", "1",
            "-pc_amg_print_level", "2",
            "-pc_amg_tolerance", "1e-10",
            "-pc_amg_max_iterations", "200",
            "-pc_amg_min_iterations", "5",
            "-pc_amg_ieee_checks", "true",
            "-pc_amg_optimize_workspace", "false"
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
    fn test_pc_options_amg_invalid_boolean() {
        let args = vec!["-pc_amg_ieee_checks", "maybe"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());
        
        if let Err(KError::SolveError(msg)) = result {
            assert!(msg.contains("Invalid amg_ieee_checks value"));
        } else {
            panic!("Expected SolveError for invalid boolean value");
        }
    }

    #[test]
    fn test_pc_options_asm_options() {
        let args = vec![
            "-pc_type", "asm",
            "-pc_asm_overlap", "2",
            "-pc_asm_subdomains", "0,1,2,3",
            "-pc_asm_inner_pc", "ilu"
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
            "-pc_type", "chebyshev",
            "-pc_chebyshev_degree", "5",
            "-pc_chebyshev_lambda_min", "0.1",
            "-pc_chebyshev_lambda_max", "10.0"
        ];
        let opts = PcOptions::from_args(&args).unwrap();
        
        assert_eq!(opts.pc_type, Some("chebyshev".to_string()));
        assert_eq!(opts.chebyshev_degree, Some(5));
        assert_eq!(opts.chebyshev_lambda_min, Some(0.1));
        assert_eq!(opts.chebyshev_lambda_max, Some(10.0));
    }

    #[test]
    fn test_pc_options_reorder_and_scaling() {
        let args = vec![
            "-pc_reorder", "colamd",
            "-pc_scaling", "diagonal"
        ];
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
            "-pc_type", "ilu",
            "-pc_ilu_type", "ilut",
            "-pc_ilu_level_of_fill", "3",
            "-pc_ilu_max_fill_per_row", "50",
            "-pc_ilu_offdiag_drop_tolerance", "1e-5",
            "-pc_ilu_schur_drop_tolerance", "1e-6",
            "-pc_ilu_reordering_type", "rcm",
            "-pc_ilu_triangular_solve", "iterative",
            "-pc_ilu_lower_jacobi_iters", "2",
            "-pc_ilu_upper_jacobi_iters", "3",
            "-pc_ilu_tolerance", "1e-8",
            "-pc_ilu_max_iterations", "10",
            "-pc_ilu_logging_level", "2",
            "-pc_ilu_print_level", "1",
            "-pc_ilu_ieee_checks", "true",
            "-pc_ilu_pivot_monitoring", "false",
            "-pc_ilu_optimize_workspace", "true",
            "-pc_ilu_pivot_threshold", "1e-10"
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
    fn test_pc_options_ilu_invalid_boolean() {
        let args = vec!["-pc_ilu_ieee_checks", "maybe"];
        let result = PcOptions::from_args(&args);
        assert!(result.is_err());
        
        if let Err(KError::SolveError(msg)) = result {
            assert!(msg.contains("Invalid ilu_ieee_checks value"));
        } else {
            panic!("Expected SolveError for invalid boolean value");
        }
    }

    #[test]
    fn test_pc_options_ilu_basic() {
        let args = vec![
            "-pc_type", "ilu",
            "-pc_ilu_type", "ilu0",
            "-pc_ilu_reordering_type", "none"
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
    fn test_ksp_options_invalid_numeric() {
        let args = vec!["-ksp_rtol", "not_a_number"];
        let result = KspOptions::from_args(&args);
        assert!(result.is_err());
        
        if let Err(KError::SolveError(msg)) = result {
            assert!(msg.contains("Invalid rtol value"));
        } else {
            panic!("Expected SolveError for invalid numeric value");
        }
    }

    #[test]
    fn test_ksp_options_unrecognized_option() {
        let args = vec!["-ksp_unknown", "value"];
        let result = KspOptions::from_args(&args);
        assert!(result.is_err());
        
        if let Err(KError::SolveError(msg)) = result {
            assert!(msg.contains("Unrecognized KSP option: -ksp_unknown"));
        } else {
            panic!("Expected SolveError for unrecognized option");
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
            "-some_other_option", "value",
            "-ksp_type", "gmres",
            "-another_option", "value2",
            "-ksp_rtol", "1e-6"
        ];
        let opts = KspOptions::from_args(&args).unwrap();
        
        assert_eq!(opts.ksp_type, Some("gmres".to_string()));
        assert_eq!(opts.rtol, Some(1e-6));
    }

    #[test]
    fn test_options_skip_non_pc_args() {
        let args = vec![
            "program_name",
            "-some_option", "value",
            "-pc_type", "jacobi",
            "-another_option", "value2"
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
            "1e-7".to_string()
        ];
        let opts = KspOptions::from_strings(&args).unwrap();
        
        assert_eq!(opts.ksp_type, Some("bicgstab".to_string()));
        assert_eq!(opts.rtol, Some(1e-7));
    }

    #[test]
    fn test_pc_options_from_strings() {
        let args = vec![
            "-pc_type".to_string(),
            "ilu0".to_string()
        ];
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
        let args = vec![
            "-ksp_type", "cg",
            "-ksp_type", "gmres"
        ];
        let opts = KspOptions::from_args(&args).unwrap();
        assert_eq!(opts.ksp_type, Some("gmres".to_string()));
    }

    #[test]
    fn test_mixed_ksp_pc_args() {
        let args = vec![
            "-ksp_type", "cg",
            "-pc_type", "jacobi",
            "-ksp_rtol", "1e-6",
            "-pc_ilu_levels", "3"
        ];
        
        let ksp_opts = KspOptions::from_args(&args).unwrap();
        let pc_opts = PcOptions::from_args(&args).unwrap();
        
        assert_eq!(ksp_opts.ksp_type, Some("cg".to_string()));
        assert_eq!(ksp_opts.rtol, Some(1e-6));
        assert_eq!(pc_opts.pc_type, Some("jacobi".to_string()));
        assert_eq!(pc_opts.ilu_level, Some(3));
    }
}
