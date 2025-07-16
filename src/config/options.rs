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
//!
//! ## PC (Preconditioner) Options
//! - `-pc_type <pc>` - Preconditioner type (jacobi, ilu0, none)
//! - `-pc_ilu_levels <int>` - ILU fill levels
//! - `-pc_chebyshev_degree <int>` - Chebyshev polynomial degree
//!
//! # Usage
//!
//! ```rust
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
    /// Relaxation factor ω for SSOR (legacy compatibility)
    pub omega: Option<f64>,
    /// Drop tolerance for ILU(p) (legacy compatibility)
    pub drop_tol: Option<f64>,
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
    /// ```rust
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
    /// ```rust
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
        
        Ok(opts)
    }
}

/// Print help information for all supported options.
pub fn print_help() {
    println!("Kryst Linear Solver Options:");
    println!();
    println!("KSP (Krylov Solver) Options:");
    println!("  -ksp_type <solver>         Solver type: cg, pcg, gmres, bicgstab, cgs, qmr, tfqmr, minres, cgnr, preonly");
    println!("  -ksp_rtol <float>          Relative convergence tolerance (default: 1e-6)");
    println!("  -ksp_atol <float>          Absolute convergence tolerance (default: 1e-12)");
    println!("  -ksp_dtol <float>          Divergence tolerance (default: 1e3)");
    println!("  -ksp_max_it <int>          Maximum number of iterations (default: 1000)");
    println!("  -ksp_gmres_restart <int>   GMRES restart parameter (default: 50)");
    println!("  -ksp_pc_side <side>        Preconditioning side: left, right, symmetric (default: left)");
    println!();
    println!("PC (Preconditioner) Options:");
    println!("  -pc_type <pc>              Preconditioner type: jacobi, ilu0, none");
    println!("  -pc_ilu_levels <int>       ILU fill levels (default: 0)");
    println!("  -pc_chebyshev_degree <int> Chebyshev polynomial degree (default: 3)");
    println!("  -pc_ilut_drop_tol <float>  ILUT drop tolerance (default: 1e-3)");
    println!("  -pc_ilut_max_fill <int>    ILUT maximum fill per row (default: 10)");
    println!();
    println!("Examples:");
    println!("  -ksp_type gmres -ksp_rtol 1e-8 -pc_type jacobi");
    println!("  -ksp_type cg -ksp_max_it 500 -pc_type ilu0");
    println!("  -ksp_type bicgstab -ksp_gmres_restart 100 -pc_type none");
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
    
    if cli_pc_opts.pc_type.is_some() { pc_opts.pc_type = cli_pc_opts.pc_type; }
    if cli_pc_opts.ilu_level.is_some() { pc_opts.ilu_level = cli_pc_opts.ilu_level; }
    if cli_pc_opts.chebyshev_degree.is_some() { pc_opts.chebyshev_degree = cli_pc_opts.chebyshev_degree; }
    if cli_pc_opts.ilut_drop_tol.is_some() { pc_opts.ilut_drop_tol = cli_pc_opts.ilut_drop_tol; }
    if cli_pc_opts.ilut_max_fill.is_some() { pc_opts.ilut_max_fill = cli_pc_opts.ilut_max_fill; }
    
    Ok((ksp_opts, pc_opts))
}
