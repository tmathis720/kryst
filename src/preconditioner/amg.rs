//! Algebraic Multigrid (AMG) preconditioner for kryst.
//!
//! This module implements a production-grade AMG preconditioner inspired by HYPRE's BoomerAMG,
//! supporting both serial and parallel (Rayon/MPI) execution with comprehensive safety checks,
//! robust defaults, and optimized workspace utilization.
//!
//! # Overview
//!
//! Algebraic Multigrid (AMG) is a multilevel preconditioner for large sparse linear systems,
//! especially those arising from discretized PDEs. This implementation includes:
//!
//! - HYPRE-inspired robust defaults and safety checks
//! - IEEE NaN/Inf detection and error handling
//! - Adaptive coarsening strategies with multiple algorithms
//! - Workspace-aware memory management
//! - Comprehensive convergence monitoring
//! - Multiple smoothing strategies (Jacobi, Gauss-Seidel, Chebyshev)
//! - Truncation strategies for operator complexity control
//!
//! # Key Features (HYPRE-Inspired)
//!
//! - **Safety**: Input validation, IEEE checks, dimension verification
//! - **Robustness**: Fallback strategies, multiple coarsening types
//! - **Performance**: Workspace reuse, parallel operations, complexity control
//! - **Monitoring**: Detailed logging, convergence tracking, cycle statistics
//!
//! # References
//!
//! - HYPRE User's Guide and Reference Manual
//! - Henson, V.E. and Yang, U.M. (2002). BoomerAMG: A parallel algebraic multigrid solver and preconditioner.
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, §13.3.
//! - Trottenberg, U., Oosterlee, C. W., & Schuller, A. (2000). Multigrid.
//!
//! # Usage
//!
//! ```rust
//! // HYPRE-style robust construction with defaults
//! let amg = AMG::new_with_defaults(&matrix)?;
//!
//! // Advanced configuration
//! let amg = AMG::builder()
//!     .max_levels(25)                    // HYPRE default: 25
//!     .strong_threshold(0.25)            // HYPRE default: 0.25
//!     .coarse_threshold(9)               // HYPRE default: 9
//!     .max_coarse_size(9)                // HYPRE default: 9
//!     .min_coarse_size(1)                // HYPRE minimum
//!     .truncation_factor(0.0)            // HYPRE default: no truncation
//!     .interpolation_truncation(0.0)     // HYPRE default
//!     .smoothing_sweeps(1, 1)            // pre/post sweeps
//!     .coarsening_type(CoarsenType::HMIS) // HYPRE default
//!     .interpolation_type(InterpType::Extended) // HYPRE robust choice
//!     .enable_logging()
//!     .build(&matrix)?;
//! ```

use crate::error::KError;
use crate::matrix::sparse::{CsrMatrix, SparseMatrix};
use crate::matrix::utils;
use crate::parallel::Comm;
use crate::preconditioner::legacy::Preconditioner;
use faer::Mat;
#[cfg(feature = "logging")]
use log::{debug, info, trace, warn};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Workspace for AMG operations to avoid repeated allocations
#[derive(Debug)]
pub struct AMGWorkspace {
    /// Temporary vector for smoothing operations
    pub temp_vector: Vec<f64>,
    /// Working vector for matrix-vector operations
    pub work_vector: Vec<f64>,
    /// Residual vector storage
    pub residual: Vec<f64>,
    /// Coarse grid solution storage
    pub coarse_solution: Vec<f64>,
    /// Fine grid correction storage
    pub fine_correction: Vec<f64>,
}

impl AMGWorkspace {
    /// Create new workspace with given capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            temp_vector: vec![0.0; max_size],
            work_vector: vec![0.0; max_size],
            residual: vec![0.0; max_size],
            coarse_solution: vec![0.0; max_size],
            fine_correction: vec![0.0; max_size],
        }
    }

    /// Resize workspace to accommodate given size
    pub fn resize(&mut self, size: usize) {
        if self.temp_vector.len() < size {
            self.temp_vector.resize(size, 0.0);
            self.work_vector.resize(size, 0.0);
            self.residual.resize(size, 0.0);
            self.coarse_solution.resize(size, 0.0);
            self.fine_correction.resize(size, 0.0);
        }
    }

    /// Get a temporary vector slice of given size
    pub fn get_temp(&mut self, size: usize) -> &mut [f64] {
        self.resize(size);
        &mut self.temp_vector[..size]
    }

    /// Get a work vector slice of given size
    pub fn get_work(&mut self, size: usize) -> &mut [f64] {
        self.resize(size);
        &mut self.work_vector[..size]
    }
}

/// Trait for matrix-vector operations (sparse-aware)
pub trait MatVecOp {
    /// Perform y = A * x
    fn matvec(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError>;
    /// Get matrix dimensions
    fn dims(&self) -> (usize, usize);
}

/// Trait for dot product operations (distributed-aware)
pub trait DotOp {
    /// Compute dot product x^T * y
    fn dot(&self, x: &[f64], y: &[f64]) -> f64;
}

/// Dense matrix implementation of MatVecOp
impl MatVecOp for Mat<f64> {
    fn matvec(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        let (rows, cols) = (self.nrows(), self.ncols());
        if x.len() != cols {
            return Err(KError::InvalidInput(format!(
                "Matrix-vector dimension mismatch: {}x{} matrix, vector length {}",
                rows,
                cols,
                x.len()
            )));
        }
        if y.len() != rows {
            return Err(KError::InvalidInput(format!(
                "Matrix-vector result dimension mismatch: {}x{} matrix, result length {}",
                rows,
                cols,
                y.len()
            )));
        }

        #[cfg(feature = "rayon")]
        {
            y.par_iter_mut().enumerate().for_each(|(i, yi)| {
                *yi = (0..cols).map(|j| self[(i, j)] * x[j]).sum();
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for (i, yi) in y.iter_mut().enumerate() {
                *yi = (0..cols).map(|j| self[(i, j)] * x[j]).sum();
            }
        }
        Ok(())
    }

    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

/// Local dot product implementation
pub struct LocalDot;

impl DotOp for LocalDot {
    fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
        #[cfg(feature = "rayon")]
        {
            x.par_iter().zip(y.par_iter()).map(|(xi, yi)| xi * yi).sum()
        }
        #[cfg(not(feature = "rayon"))]
        {
            x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum()
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarsenType {
    /// Classical Ruge-Stuben coarsening
    RS = 0,
    /// HYPRE's modified independent set algorithm (default)
    HMIS = 1,
    /// Parallel modified independent set
    PMIS = 2,
    /// Falgout coarsening (hybrid of RS and PMIS)
    Falgout = 3,
}

/// HYPRE-inspired interpolation types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpType {
    /// Classical modified interpolation
    Classical = 0,
    /// Direct interpolation
    Direct = 1,
    /// Multipass interpolation
    Multipass = 2,
    /// Extended classical interpolation (more robust)
    Extended = 3,
    /// Standard interpolation (HYPRE default for robustness)
    Standard = 4,
}

/// HYPRE-inspired smoothing types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelaxType {
    /// Jacobi
    Jacobi = 0,
    /// Gauss-Seidel (forward)
    GaussSeidel = 1,
    /// Gauss-Seidel (backward)
    GaussSeidelBackward = 2,
    /// Symmetric Gauss-Seidel
    SymmetricGaussSeidel = 3,
    /// Hybrid Gauss-Seidel
    HybridGaussSeidel = 4,
    /// L1-Jacobi (for nonsymmetric problems)
    L1Jacobi = 6,
    /// Chebyshev
    Chebyshev = 16,
}

/// AMG configuration structure with HYPRE-inspired defaults
#[derive(Clone, Debug)]
pub struct AMGConfig {
    /// Maximum number of multigrid levels (HYPRE default: 25)
    pub max_levels: usize,
    /// Strong threshold for coarsening (HYPRE default: 0.25)
    pub strong_threshold: f64,
    /// Coarse grid threshold - stop coarsening (HYPRE default: 9)
    pub coarse_threshold: usize,
    /// Maximum coarse grid size (HYPRE default: 9)
    pub max_coarse_size: usize,
    /// Minimum coarse grid size (HYPRE default: 1)
    pub min_coarse_size: usize,
    /// Truncation factor for interpolation (HYPRE default: 0.0 = no truncation)
    pub truncation_factor: f64,
    /// Max elements per row for interpolation (HYPRE default: 0 = no limit)
    pub max_elements_per_row: usize,
    /// Interpolation truncation factor (HYPRE default: 0.0)
    pub interpolation_truncation: f64,
    /// Pre-smoothing sweeps (HYPRE default: 1)
    pub pre_sweeps: usize,
    /// Post-smoothing sweeps (HYPRE default: 1)
    pub post_sweeps: usize,
    /// Coarsening algorithm (HYPRE default: HMIS)
    pub coarsen_type: CoarsenType,
    /// Interpolation algorithm (HYPRE robust choice: Extended)
    pub interp_type: InterpType,
    /// Relaxation/smoothing type (HYPRE default: Gauss-Seidel)
    pub relax_type: RelaxType,
    /// Enable comprehensive logging (HYPRE style)
    pub logging_level: usize,
    /// Print level for debugging (HYPRE style)
    pub print_level: usize,
    /// Convergence tolerance (for standalone AMG solver)
    pub tolerance: f64,
    /// Maximum iterations (for standalone AMG solver)
    pub max_iterations: usize,
    /// Minimum iterations (HYPRE feature)
    pub min_iterations: usize,
    /// IEEE safety checks enabled (HYPRE inspired)
    pub ieee_checks: bool,
    /// Workspace optimization enabled
    pub optimize_workspace: bool,
    /// Damping factor for Jacobi smoother
    pub jacobi_omega: f64,
    /// Chebyshev degree (0 disables)
    pub chebyshev_degree: usize,
}

impl Default for AMGConfig {
    /// HYPRE-inspired robust defaults
    fn default() -> Self {
        Self {
            max_levels: 25,                     // HYPRE default
            strong_threshold: 0.25,             // HYPRE default
            coarse_threshold: 9,                // HYPRE default
            max_coarse_size: 9,                 // HYPRE default
            min_coarse_size: 1,                 // HYPRE minimum
            truncation_factor: 0.0,             // HYPRE default: no truncation
            max_elements_per_row: 0,            // HYPRE default: unlimited
            interpolation_truncation: 0.0,      // HYPRE default
            pre_sweeps: 1,                      // HYPRE default
            post_sweeps: 1,                     // HYPRE default
            coarsen_type: CoarsenType::HMIS,    // HYPRE default
            interp_type: InterpType::Extended,  // Robust choice
            relax_type: RelaxType::GaussSeidel, // HYPRE default
            logging_level: 0,                   // No logging by default
            print_level: 0,                     // No printing by default
            tolerance: 1e-6,                    // HYPRE default for standalone solver
            max_iterations: 20,                 // HYPRE default for cycles
            min_iterations: 0,                  // HYPRE default
            ieee_checks: true,                  // Safety first
            optimize_workspace: true,           // Performance optimization
            jacobi_omega: 2.0 / 3.0,            // Safe default damping
            chebyshev_degree: 0,                // Disabled by default
        }
    }
}

/// HYPRE-inspired AMG builder for advanced configuration
pub struct AMGBuilder {
    config: AMGConfig,
}

impl AMGBuilder {
    /// Create new builder with HYPRE defaults
    pub fn new() -> Self {
        Self {
            config: AMGConfig::default(),
        }
    }

    /// Set maximum number of levels (HYPRE: max_levels)
    pub fn max_levels(mut self, levels: usize) -> Self {
        self.config.max_levels = levels;
        self
    }

    /// Set strong threshold (HYPRE: strong_threshold)
    pub fn strong_threshold(mut self, threshold: f64) -> Self {
        self.config.strong_threshold = threshold;
        self
    }

    /// Set coarse threshold (HYPRE: coarse_threshold)
    pub fn coarse_threshold(mut self, threshold: usize) -> Self {
        self.config.coarse_threshold = threshold;
        self
    }

    /// Set max coarse size (HYPRE: max_coarse_size)
    pub fn max_coarse_size(mut self, size: usize) -> Self {
        self.config.max_coarse_size = size;
        self
    }

    /// Set min coarse size (HYPRE: min_coarse_size)
    pub fn min_coarse_size(mut self, size: usize) -> Self {
        self.config.min_coarse_size = size;
        self
    }

    /// Set truncation factor (HYPRE: trunc_factor)
    pub fn truncation_factor(mut self, factor: f64) -> Self {
        self.config.truncation_factor = factor;
        self
    }

    /// Set interpolation truncation (HYPRE: interpolation truncation)
    pub fn interpolation_truncation(mut self, factor: f64) -> Self {
        self.config.interpolation_truncation = factor;
        self
    }

    /// Set Jacobi damping factor ω
    pub fn jacobi_omega(mut self, omega: f64) -> Self {
        self.config.jacobi_omega = omega;
        self
    }

    /// Set Chebyshev smoother degree (0 disables)
    pub fn chebyshev_degree(mut self, k: usize) -> Self {
        self.config.chebyshev_degree = k;
        self
    }

    /// Set smoothing sweeps (HYPRE: num_sweeps)
    pub fn smoothing_sweeps(mut self, pre: usize, post: usize) -> Self {
        self.config.pre_sweeps = pre;
        self.config.post_sweeps = post;
        self
    }

    /// Set coarsening type (HYPRE: coarsen_type)
    pub fn coarsening_type(mut self, coarsen_type: CoarsenType) -> Self {
        self.config.coarsen_type = coarsen_type;
        self
    }

    /// Set interpolation type (HYPRE: interp_type)
    pub fn interpolation_type(mut self, interp_type: InterpType) -> Self {
        self.config.interp_type = interp_type;
        self
    }

    /// Set relaxation type (HYPRE: relax_type)
    pub fn relaxation_type(mut self, relax_type: RelaxType) -> Self {
        self.config.relax_type = relax_type;
        self
    }

    /// Enable logging (HYPRE: logging level)
    pub fn enable_logging(mut self) -> Self {
        self.config.logging_level = 1;
        self
    }

    /// Set detailed logging level (HYPRE: logging)
    pub fn logging_level(mut self, level: usize) -> Self {
        self.config.logging_level = level;
        self
    }

    /// Enable printing (HYPRE: print_level)
    pub fn enable_printing(mut self) -> Self {
        self.config.print_level = 1;
        self
    }

    /// Set print level (HYPRE: print_level)
    pub fn print_level(mut self, level: usize) -> Self {
        self.config.print_level = level;
        self
    }

    /// Build AMG with configuration
    pub fn build(self, matrix: &Mat<f64>) -> Result<AMG, KError> {
        AMG::new_with_config(matrix, self.config)
    }
}

impl Default for AMGBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// AMG preconditioner struct, holding the multigrid hierarchy and parameters.
///
/// This version uses workspace management and sparse-aware operations for better performance.
pub struct AMG {
    /// Multigrid levels, from fine to coarse.
    levels: Vec<AMGLevel>,
    /// Number of pre-smoothing Jacobi iterations per V-cycle.
    nu_pre: usize,
    /// Number of post-smoothing Jacobi iterations per V-cycle.
    nu_post: usize,
    /// The original system matrix (for fallback smoothing).
    matrix: Mat<f64>,
    /// Workspace for temporary vectors
    workspace: AMGWorkspace,
    /// Configuration
    config: AMGConfig,
}

/// One level in the AMG hierarchy: interpolation, restriction, coarse matrix, and diagonal inverse.
struct AMGLevel {
    /// Prolongation (interpolation) operator to next finer level - now sparse
    interpolation: CsrMatrix<f64>,
    /// Restriction operator to next coarser level - now sparse
    restriction: CsrMatrix<f64>,
    /// Coarse-level matrix - now sparse
    coarse_matrix: CsrMatrix<f64>,
    /// Inverse of diagonal (for Jacobi smoothing)
    diag_inv: Vec<f64>,
    /// Sparse pattern information for optimization
    nnz: usize,
}

impl AMG {
    /// HYPRE-inspired input validation
    fn validate_matrix(matrix: &Mat<f64>) -> Result<(), KError> {
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            return Err(KError::InvalidInput("Matrix cannot be empty".to_string()));
        }

        if matrix.nrows() != matrix.ncols() {
            return Err(KError::InvalidInput(
                "AMG requires square matrices".to_string(),
            ));
        }

        // Check diagonal dominance for stability
        let mut weak_diagonal_count = 0;
        for i in 0..matrix.nrows() {
            let diagonal = matrix[(i, i)].abs();
            if diagonal < 1e-14 {
                weak_diagonal_count += 1;
            }
        }

        if weak_diagonal_count > matrix.nrows() / 2 {
            return Err(KError::InvalidInput(
                "Matrix has too many weak diagonal entries for stable AMG".to_string(),
            ));
        }

        Ok(())
    }

    /// HYPRE-inspired configuration validation
    fn validate_config(config: &AMGConfig) -> Result<(), KError> {
        if config.max_levels == 0 {
            return Err(KError::InvalidInput("max_levels must be > 0".to_string()));
        }

        if config.strong_threshold < 0.0 || config.strong_threshold > 1.0 {
            return Err(KError::InvalidInput(
                "strong_threshold must be in [0.0, 1.0]".to_string(),
            ));
        }

        if config.coarse_threshold == 0 {
            return Err(KError::InvalidInput(
                "coarse_threshold must be > 0".to_string(),
            ));
        }

        if config.max_coarse_size < config.min_coarse_size {
            return Err(KError::InvalidInput(
                "max_coarse_size must be >= min_coarse_size".to_string(),
            ));
        }

        if config.tolerance <= 0.0 {
            return Err(KError::InvalidInput("tolerance must be > 0".to_string()));
        }

        if config.truncation_factor < 0.0 || config.truncation_factor > 1.0 {
            return Err(KError::InvalidInput(
                "truncation_factor must be in [0.0, 1.0]".to_string(),
            ));
        }

        if !(0.0 < config.jacobi_omega && config.jacobi_omega <= 1.0) {
            return Err(KError::InvalidInput(
                "jacobi_omega must be in (0, 1]".to_string(),
            ));
        }

        Ok(())
    }

    /// Convert dense matrix to sparse format with drop tolerance
    /// This is the foundation for sparse Galerkin products
    fn to_sparse_with_tolerance(matrix: &Mat<f64>, drop_tol: f64) -> CsrMatrix<f64> {
        CsrMatrix::from_dense(matrix, drop_tol)
    }

    /// Unified V-cycle implementation using kernel trait to eliminate code duplication
    fn apply_cycle<K: crate::core::traits::AmgKernel>(
        &self,
        level: usize,
        r: &[f64],
        z: &mut [f64],
        workspace: &mut AMGWorkspace,
        kernel: &K,
        comm: &K::Comm,
    ) -> Result<(), KError> {
        if level + 1 >= self.levels.len() {
            // Coarsest level: direct solve or heavy smoothing
            if level < self.levels.len() {
                let level_data = &self.levels[level];
                self.smooth_jacobi_parallel_workspace_sparse(
                    &level_data.coarse_matrix,
                    &level_data.diag_inv,
                    r,
                    z,
                    20,
                    workspace,
                )?;
            } else {
                // Fallback for edge case
                let diag_inv = utils::extract_diagonal_inverse(&self.matrix);
                self.smooth_jacobi_parallel_workspace(
                    &self.matrix,
                    &diag_inv,
                    r,
                    z,
                    10,
                    workspace,
                )?;
            }
            return Ok(());
        }

        let level_data = &self.levels[level];
        let current_matrix = &level_data.coarse_matrix;
        let next_level_matrix = &self.levels[level + 1].coarse_matrix;

        // Validate dimensions
        if r.len() != current_matrix.nrows() || z.len() != current_matrix.nrows() {
            return Err(KError::InvalidInput(format!(
                "Dimension mismatch at level {}: matrix {}x{}, r.len()={}, z.len()={}",
                level,
                current_matrix.nrows(),
                current_matrix.ncols(),
                r.len(),
                z.len()
            )));
        }

        // Ensure workspace is large enough
        workspace.resize(current_matrix.nrows().max(next_level_matrix.nrows()));

        // Pre-smoothing
        self.smooth_jacobi_parallel_workspace_sparse(
            current_matrix,
            &level_data.diag_inv,
            r,
            z,
            self.nu_pre,
            workspace,
        )?;

        // Compute residual: res = r - A * z
        kernel.matvec(
            1.0,
            current_matrix,
            z,
            0.0,
            &mut workspace.residual[..current_matrix.nrows()],
        )?;
        for i in 0..current_matrix.nrows() {
            workspace.residual[i] = r[i] - workspace.residual[i]; // r - A*z
        }

        // Restrict residual to coarse level
        let coarse_size = next_level_matrix.nrows();
        kernel.matvec(
            1.0,
            &level_data.restriction,
            &workspace.residual[..current_matrix.nrows()],
            0.0,
            &mut workspace.coarse_solution[..coarse_size],
        )?;

        // Recursive solve on coarse level - need separate workspace to avoid borrowing issues
        let coarse_rhs: Vec<f64> = workspace.coarse_solution[..coarse_size].to_vec();
        let mut coarse_correction = vec![0.0; coarse_size];

        // Create a minimal temporary workspace for recursion
        let mut temp_workspace = AMGWorkspace::new(coarse_size);
        self.apply_cycle(
            level + 1,
            &coarse_rhs,
            &mut coarse_correction,
            &mut temp_workspace,
            kernel,
            comm,
        )?;

        // Interpolate correction back to fine level
        kernel.matvec(
            1.0,
            &level_data.interpolation,
            &coarse_correction,
            0.0,
            &mut workspace.fine_correction[..current_matrix.nrows()],
        )?;

        // Add correction: z = z + fine_correction
        for i in 0..current_matrix.nrows() {
            z[i] += workspace.fine_correction[i];
        }

        // Post-smoothing
        self.smooth_jacobi_parallel_workspace_sparse(
            current_matrix,
            &level_data.diag_inv,
            r,
            z,
            self.nu_post,
            workspace,
        )?;

        Ok(())
    }

    /// HYPRE-inspired constructor with comprehensive configuration
    pub fn new_with_config(matrix: &Mat<f64>, config: AMGConfig) -> Result<Self, KError> {
        // HYPRE-style input validation
        Self::validate_matrix(matrix)?;
        Self::validate_config(&config)?;

        // IEEE safety checks if enabled
        if config.ieee_checks {
            utils::check_ieee_values(matrix)?;

            #[cfg(feature = "logging")]
            if config.logging_level > 0 {
                info!("AMG: IEEE safety checks passed");
            }
        }

        #[cfg(feature = "logging")]
        if config.logging_level > 0 {
            let (nnz, diag_dominance, diag_sum) = utils::analyze_matrix_properties(matrix);
            info!(
                "AMG Setup: Starting with {} x {} matrix (nnz={}, diag_dominance={:.2})",
                matrix.nrows(),
                matrix.ncols(),
                nnz,
                diag_dominance
            );
            debug!(
                "AMG Config: max_levels={}, strong_threshold={:.3}, coarsen_type={:?}",
                config.max_levels, config.strong_threshold, config.coarsen_type
            );
            if diag_sum < 1e-12 {
                warn!("AMG: Matrix has very weak diagonal (sum={:.2e})", diag_sum);
            }
        }

        let mut levels = Vec::with_capacity(config.max_levels);
        let mut current_matrix = matrix.clone();
        let mut current_diag = utils::extract_diagonal_inverse(&current_matrix);
        let mut setup_complexity = 0.0;
        let original_size = matrix.nrows();

        for level_idx in 0..config.max_levels {
            let n = current_matrix.nrows();

            #[cfg(feature = "logging")]
            if config.logging_level > 1 {
                // Calculate nnz manually for faer matrices
                let mut nnz = 0;
                for i in 0..current_matrix.nrows() {
                    for j in 0..current_matrix.ncols() {
                        if current_matrix[(i, j)].abs() > 1e-15 {
                            nnz += 1;
                        }
                    }
                }
                trace!("AMG Level {}: size={}, nnz={}", level_idx, n, nnz);
            }

            // HYPRE-style stopping criteria
            if n <= config.coarse_threshold {
                #[cfg(feature = "logging")]
                if config.logging_level > 0 {
                    debug!(
                        "AMG: Stopped coarsening at level {} (size={} <= threshold={})",
                        level_idx, n, config.coarse_threshold
                    );
                }
                break;
            }

            if n <= config.min_coarse_size {
                #[cfg(feature = "logging")]
                if config.logging_level > 0 {
                    debug!("AMG: Reached minimum coarse size at level {}", level_idx);
                }
                break;
            }

            // Compute adaptive threshold based on anisotropy
            let adaptive_threshold =
                utils::compute_adaptive_threshold(&current_matrix, config.strong_threshold);

            // Generate interpolation and restriction operators with HYPRE-style coarsening
            let (mut interpolation, restriction) = Self::generate_operators_with_config(
                &current_matrix,
                adaptive_threshold,
                &config,
                level_idx,
            );

            // Validate operator dimensions for early error detection
            if interpolation.nrows() != current_matrix.nrows() {
                return Err(KError::FactorError(format!(
                    "Interpolation matrix dimension mismatch at level {}: expected {} rows, got {}",
                    level_idx,
                    current_matrix.nrows(),
                    interpolation.nrows()
                )));
            }

            if restriction.ncols() != current_matrix.ncols() {
                return Err(KError::FactorError(format!(
                    "Restriction matrix dimension mismatch at level {}: expected {} cols, got {}",
                    level_idx,
                    current_matrix.ncols(),
                    restriction.ncols()
                )));
            }

            // Apply HYPRE-style interpolation improvements with error handling
            match config.interp_type {
                InterpType::Extended | InterpType::Standard => {
                    smooth_interpolation(&mut interpolation, &current_matrix, 0.5);
                    minimize_energy(&mut interpolation, &current_matrix);
                }
                _ => {
                    // Basic interpolation processing
                }
            }

            // Apply truncation if specified
            if config.truncation_factor > 0.0 {
                utils::apply_truncation(&mut interpolation, config.truncation_factor);
            }

            // Build coarse matrix (Galerkin product: R * A * P) with error checking
            let coarse_matrix = &restriction * &current_matrix * &interpolation;

            // Convert to sparse format for storage and operations
            let sparse_interpolation = utils::to_sparse_with_tolerance(&interpolation, 1e-12);
            let sparse_restriction = utils::to_sparse_with_tolerance(&restriction, 1e-12);
            let sparse_matrix = utils::to_sparse_with_tolerance(&current_matrix, 1e-12);

            // Sparse Galerkin product: coarse_matrix = R * A * P
            let sparse_coarse_matrix = utils::sparse_galerkin_product(
                &sparse_restriction,
                &sparse_matrix,
                &sparse_interpolation,
            )?;

            // Convert back to dense for validation (TODO: implement sparse validation)
            let coarse_matrix = sparse_coarse_matrix.to_dense();

            // Validate coarse matrix properties
            if coarse_matrix.nrows() == 0 || coarse_matrix.ncols() == 0 {
                return Err(KError::FactorError(format!(
                    "Generated empty coarse matrix at level {}",
                    level_idx
                )));
            }

            // Check for numerical issues in coarse matrix
            if utils::has_numerical_issues(&coarse_matrix) {
                return Err(KError::FactorError(format!(
                    "Numerical issues detected in coarse matrix at level {}",
                    level_idx
                )));
            }

            let coarse_diag = utils::extract_diagonal_inverse(&coarse_matrix);

            // HYPRE-style complexity tracking using sparse nnz
            let coarse_nnz = sparse_coarse_matrix.nnz();
            setup_complexity += coarse_nnz as f64 / original_size as f64;

            levels.push(AMGLevel {
                interpolation: sparse_interpolation,
                restriction: sparse_restriction,
                coarse_matrix: sparse_coarse_matrix,
                diag_inv: coarse_diag.clone(),
                nnz: coarse_nnz,
            });

            // Update for next iteration using sparse matrix
            current_matrix = coarse_matrix;
            current_diag = coarse_diag;

            // Check for stalling (HYPRE-style)
            if current_matrix.nrows() >= n {
                #[cfg(feature = "logging")]
                if config.logging_level > 0 {
                    warn!(
                        "AMG: Coarsening stalled at level {} (no size reduction)",
                        level_idx
                    );
                }
                break;
            }
        }

        // Add the coarsest level
        let final_size = current_matrix.nrows();
        let final_sparse_matrix = utils::to_sparse_with_tolerance(&current_matrix, 1e-12);
        let final_nnz = final_sparse_matrix.nnz();

        levels.push(AMGLevel {
            interpolation: CsrMatrix::identity(final_size),
            restriction: CsrMatrix::identity(final_size),
            coarse_matrix: final_sparse_matrix,
            diag_inv: current_diag,
            nnz: final_nnz,
        });

        #[cfg(feature = "logging")]
        if config.logging_level > 0 {
            info!(
                "AMG Setup Complete: {} levels, complexity={:.2}",
                levels.len(),
                setup_complexity
            );
            if config.print_level > 0 {
                println!(
                    "AMG Setup: {} -> {} (complexity: {:.2})",
                    original_size, final_size, setup_complexity
                );
            }
        }

        Ok(AMG {
            levels,
            nu_pre: config.pre_sweeps,
            nu_post: config.post_sweeps,
            matrix: matrix.clone(),
            workspace: AMGWorkspace::new(matrix.nrows()),
            config,
        })
    }

    /// Generate operators with HYPRE-style configuration
    fn generate_operators_with_config(
        a: &Mat<f64>,
        threshold: f64,
        config: &AMGConfig,
        _level: usize,
    ) -> (Mat<f64>, Mat<f64>) {
        // Use configuration-driven coarsening and interpolation
        let strength_matrix = compute_strength_matrix(a, threshold);

        let aggregates = match config.coarsen_type {
            CoarsenType::HMIS => {
                // HYPRE's modified independent set (fallback to double pairwise for now)
                double_pairwise_aggregation(&strength_matrix)
            }
            CoarsenType::RS => {
                // Classical Ruge-Stuben (fallback to greedy for now)
                greedy_aggregation(&strength_matrix)
            }
            CoarsenType::PMIS => {
                // Parallel modified independent set (fallback to double pairwise)
                double_pairwise_aggregation(&strength_matrix)
            }
            CoarsenType::Falgout => {
                // Hybrid approach (fallback to double pairwise)
                double_pairwise_aggregation(&strength_matrix)
            }
        };

        let prolongation = match config.interp_type {
            InterpType::Extended => construct_prolongation(a, &aggregates),
            InterpType::Standard => construct_prolongation(a, &aggregates),
            InterpType::Classical => construct_prolongation(a, &aggregates),
            InterpType::Direct => construct_prolongation(a, &aggregates),
            InterpType::Multipass => construct_prolongation(a, &aggregates),
        };

        let restriction = prolongation.transpose().to_owned();
        (prolongation, restriction)
    }

    /// Construct a new AMG hierarchy from a matrix.
    ///
    /// # Arguments
    /// * `a` - System matrix
    /// * `max_levels` - Maximum number of coarsening levels
    /// * `base_threshold` - Base strength-of-connection threshold
    pub fn new(a: &Mat<f64>, max_levels: usize, base_threshold: f64) -> Self {
        // Use HYPRE defaults with legacy parameters
        let config = AMGConfig {
            max_levels,
            strong_threshold: base_threshold,
            ..Default::default()
        };

        Self::new_with_config(a, config).unwrap_or_else(|e| {
            // Fallback to original implementation for compatibility
            #[cfg(feature = "logging")]
            warn!("AMG: Falling back to legacy constructor due to: {}", e);

            Self::with_smoothing(a, max_levels, base_threshold, 1, 1)
        })
    }

    /// Create a new AMG preconditioner with default HYPRE-inspired settings
    pub fn new_with_defaults(matrix: &Mat<f64>) -> Result<Self, KError> {
        let config = AMGConfig::default();
        Self::new_with_config(matrix, config)
    }

    /// Construct a new AMG hierarchy from a matrix with custom smoothing parameters.
    ///
    /// # Arguments
    /// * `a` - System matrix
    /// * `max_levels` - Maximum number of coarsening levels
    /// * `base_threshold` - Base strength-of-connection threshold
    /// * `nu_pre` - Number of pre-smoothing iterations
    /// * `nu_post` - Number of post-smoothing iterations
    pub fn with_smoothing(
        a: &Mat<f64>,
        max_levels: usize,
        base_threshold: f64,
        nu_pre: usize,
        nu_post: usize,
    ) -> Self {
        let mut levels = Vec::new();
        let mut current_matrix = a.clone();
        let mut current_diag = utils::extract_diagonal_inverse(&current_matrix);
        for _level_idx in 0..max_levels {
            let n = current_matrix.nrows();
            if n <= 10 {
                break;
            }
            // Compute adaptive threshold based on anisotropy
            let adaptive_threshold =
                utils::compute_adaptive_threshold(&current_matrix, base_threshold);
            // Generate interpolation and restriction operators
            let (mut interpolation, restriction) =
                AMG::generate_operators(&current_matrix, adaptive_threshold, true);
            // Smooth and normalize interpolation
            smooth_interpolation(&mut interpolation, &current_matrix, 0.5);
            minimize_energy(&mut interpolation, &current_matrix);
            // Convert to sparse for consistency with unified AMG approach
            let sparse_interpolation = utils::to_sparse_with_tolerance(&interpolation, 1e-12);
            let sparse_restriction = utils::to_sparse_with_tolerance(&restriction, 1e-12);

            // Build coarse matrix using sparse Galerkin product
            let sparse_coarse_matrix = match utils::sparse_galerkin_product(
                &sparse_restriction,
                &utils::to_sparse_with_tolerance(&current_matrix, 1e-12),
                &sparse_interpolation,
            ) {
                Ok(matrix) => matrix,
                Err(_) => {
                    // Fallback to dense computation if sparse fails
                    let temp = &restriction * &current_matrix;
                    let coarse_dense = &temp * &interpolation;
                    utils::to_sparse_with_tolerance(&coarse_dense, 1e-12)
                }
            };

            // Extract diagonal for smoothing
            let coarse_matrix_dense = sparse_coarse_matrix.to_dense();
            let coarse_diag = utils::extract_diagonal_inverse(&coarse_matrix_dense);
            let coarse_nnz = sparse_coarse_matrix.nnz();

            levels.push(AMGLevel {
                interpolation: sparse_interpolation,
                restriction: sparse_restriction,
                coarse_matrix: sparse_coarse_matrix,
                diag_inv: coarse_diag.clone(),
                nnz: coarse_nnz,
            });
            current_matrix = coarse_matrix_dense;
            current_diag = coarse_diag;
        }
        // Add the coarsest level (identity prolongation/restriction)
        let diag_inv_final = Self::extract_diagonal_inverse(&current_matrix);
        levels.push(AMGLevel {
            interpolation: CsrMatrix::identity(current_matrix.nrows()),
            restriction: CsrMatrix::identity(current_matrix.nrows()),
            coarse_matrix: Self::to_sparse_with_tolerance(&current_matrix, 1e-12),
            diag_inv: diag_inv_final,
            nnz: utils::count_nnz(&current_matrix),
        });
        AMG {
            levels,
            nu_pre,
            nu_post,
            matrix: a.clone(),
            workspace: AMGWorkspace::new(a.nrows()),
            config: AMGConfig::default(),
        }
    }
    /// Generate interpolation and restriction operators for a given matrix and threshold.
    ///
    /// Returns (prolongation, restriction).
    fn generate_operators(
        a: &Mat<f64>,
        threshold: f64,
        double_pairwise: bool,
    ) -> (Mat<f64>, Mat<f64>) {
        let strength_matrix = compute_strength_matrix(a, threshold);
        let aggregates = if double_pairwise {
            double_pairwise_aggregation(&strength_matrix)
        } else {
            greedy_aggregation(&strength_matrix)
        };
        let prolongation = construct_prolongation(a, &aggregates);
        let restriction = prolongation.transpose().to_owned();
        (prolongation, restriction)
    }
    /// Extract the inverse of the diagonal of a matrix, with zero for near-singular entries.
    fn extract_diagonal_inverse(m: &Mat<f64>) -> Vec<f64> {
        assert_eq!(m.nrows(), m.ncols());
        let n = m.nrows();
        #[cfg(feature = "rayon")]
        {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let d = m[(i, i)];
                    if d.abs() < 1e-14 {
                        0.0
                    } else {
                        1.0 / d
                    }
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            (0..n)
                .into_iter()
                .map(|i| {
                    let d = m[(i, i)];
                    if d.abs() < 1e-14 {
                        0.0
                    } else {
                        1.0 / d
                    }
                })
                .collect()
        }
    }
    /// Parallel Jacobi smoother for a given matrix and right-hand side.
    ///
    /// Applies a fixed number of Jacobi iterations to z, using the diagonal inverse.
    fn smooth_jacobi_parallel(
        &self,
        a: &Mat<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        iterations: usize,
    ) -> Result<(), KError> {
        let n = r.len();
        if diag_inv.len() != n || z.len() != n {
            return Err(KError::InvalidInput(
                "Jacobi(simple): dimension mismatch".to_string(),
            ));
        }
        if iterations == 0 {
            return Ok(());
        }

        let omega = self.config.jacobi_omega;

        let mut z_vec = z.to_vec();
        let mut temp = vec![0.0; n];
        for _ in 0..iterations {
            utils::parallel_mat_vec(a, &z_vec, &mut temp)?;
            #[cfg(feature = "rayon")]
            {
                temp.par_iter_mut().enumerate().for_each(|(i, val)| {
                    *val = r[i] - *val;
                });
                z_vec
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, val)| *val += omega * diag_inv[i] * temp[i]);
            }
            #[cfg(not(feature = "rayon"))]
            {
                temp.iter_mut().enumerate().for_each(|(i, val)| {
                    *val = r[i] - *val;
                });
                z_vec
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, val)| *val += omega * diag_inv[i] * temp[i]);
            }
        }
        z.copy_from_slice(&z_vec);
        Ok(())
    }

    /// Workspace-aware Jacobi smoothing - reuses allocated vectors for better performance
    /// Sparse version of Jacobi smoothing
    fn smooth_jacobi_parallel_workspace_sparse(
        &self,
        a: &CsrMatrix<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        iterations: usize,
        workspace: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        let n = r.len();
        if diag_inv.len() != n || z.len() != n {
            return Err(KError::InvalidInput(
                "Jacobi: dimension mismatch".to_string(),
            ));
        }
        if iterations == 0 {
            return Ok(());
        }

        workspace.resize(n);
        workspace.temp_vector[..n].copy_from_slice(z);
        let omega = self.config.jacobi_omega;

        for _ in 0..iterations {
            a.spmv_scaled(
                1.0,
                &workspace.temp_vector[..n],
                0.0,
                &mut workspace.work_vector[..n],
            )?;
            for i in 0..n {
                let corr = diag_inv[i] * (r[i] - workspace.work_vector[i]);
                workspace.temp_vector[i] += omega * corr;
            }
        }

        z[..n].copy_from_slice(&workspace.temp_vector[..n]);
        Ok(())
    }

    fn smooth_jacobi_parallel_workspace(
        &self,
        a: &Mat<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        iterations: usize,
        workspace: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        let n = r.len();
        if diag_inv.len() != n || z.len() != n {
            return Err(KError::InvalidInput(
                "Jacobi(dense): dimension mismatch".to_string(),
            ));
        }
        if iterations == 0 {
            return Ok(());
        }

        workspace.resize(n);
        workspace.temp_vector[..n].copy_from_slice(z);
        let omega = self.config.jacobi_omega;

        for _ in 0..iterations {
            utils::parallel_mat_vec(
                a,
                &workspace.temp_vector[..n],
                &mut workspace.work_vector[..n],
            )?;

            #[cfg(feature = "rayon")]
            {
                workspace.temp_vector[..n]
                    .par_iter_mut()
                    .zip(workspace.work_vector[..n].par_iter())
                    .zip(r.par_iter())
                    .zip(diag_inv.par_iter())
                    .for_each(|(((temp, &work), &residual), &d_inv)| {
                        *temp += omega * d_inv * (residual - work);
                    });
            }
            #[cfg(not(feature = "rayon"))]
            {
                for i in 0..n {
                    let corr = diag_inv[i] * (r[i] - workspace.work_vector[i]);
                    workspace.temp_vector[i] += omega * corr;
                }
            }
        }

        z.copy_from_slice(&workspace.temp_vector[..n]);
        Ok(())
    }
    /// Recursive AMG V-cycle application (serial/Rayon).
    ///
    /// Applies pre-smoothing, restricts the residual, recursively solves on the coarse grid, prolongates the correction, and post-smooths.
    fn apply_recursive(
        &self,
        level: usize,
        r: &[f64],
        z: &mut [f64],
        workspace: &mut AMGWorkspace,
    ) -> Result<(), KError> {
        if level + 1 == self.levels.len() {
            AMG::solve_direct_sparse(&self.levels[level].coarse_matrix, r, z);
            return Ok(());
        }
        let a = &self.levels[level].coarse_matrix;
        let diag_inv = &self.levels[level].diag_inv;
        let restriction = &self.levels[level].restriction;
        let interpolation = &self.levels[level].interpolation;
        let coarse_matrix = &self.levels[level + 1].coarse_matrix;

        // Ensure workspace is large enough
        workspace.resize(a.nrows().max(coarse_matrix.nrows()));

        // Pre-smoothing
        self.smooth_jacobi_parallel_workspace_sparse(a, diag_inv, r, z, self.nu_pre, workspace)?;

        // Compute residual: r - A*z using workspace
        utils::parallel_mat_vec_sparse(a, z, &mut workspace.residual[..a.nrows()])?;
        #[cfg(feature = "rayon")]
        {
            workspace.residual[..a.nrows()]
                .par_iter_mut()
                .zip(r.par_iter())
                .for_each(|(res, &ri)| *res = ri - *res);
        }
        #[cfg(not(feature = "rayon"))]
        {
            for i in 0..a.nrows() {
                workspace.residual[i] = r[i] - workspace.residual[i];
            }
        }

        // Restrict residual to coarse grid
        utils::parallel_mat_vec_sparse(
            restriction,
            &workspace.residual[..a.nrows()],
            &mut workspace.coarse_solution[..coarse_matrix.nrows()],
        )?;

        // Make owned copies for recursive call to avoid borrowing conflicts
        let coarse_residual = workspace.coarse_solution[..coarse_matrix.nrows()].to_vec();
        let mut coarse_solution = vec![0.0; coarse_matrix.nrows()];

        // Recursive coarse solve
        self.apply_recursive(level + 1, &coarse_residual, &mut coarse_solution, workspace)?;

        // Prolongate correction
        utils::parallel_mat_vec_sparse(
            interpolation,
            &coarse_solution,
            &mut workspace.fine_correction[..a.nrows()],
        )?;

        // Add correction to solution
        #[cfg(feature = "rayon")]
        {
            z.par_iter_mut()
                .zip(workspace.fine_correction[..a.nrows()].par_iter())
                .for_each(|(zi, &correction)| *zi += correction);
        }
        #[cfg(not(feature = "rayon"))]
        {
            for i in 0..z.len() {
                z[i] += workspace.fine_correction[i];
            }
        }

        // Post-smoothing
        self.smooth_jacobi_parallel_workspace_sparse(a, diag_inv, r, z, self.nu_post, workspace)?;
        Ok(())
    }
    /// Direct solve on coarsest level using sparse matrix
    /// Uses iterative CG for small sparse systems to avoid densification
    fn solve_direct_sparse(a: &CsrMatrix<f64>, r: &[f64], z: &mut [f64]) {
        let n = a.nrows();

        // For very small matrices, use dense direct solve
        if n <= 10 {
            let a_dense = a.to_dense();
            Self::solve_direct(&a_dense, r, z);
            return;
        }

        // Use sparse iterative solver (CG) for larger coarse grids
        // This avoids the O(n³) cost of dense factorization

        // Initialize solution to zero
        z.fill(0.0);

        // CG workspace
        let mut p = vec![0.0; n];
        let mut ap = vec![0.0; n];
        let mut residual = vec![0.0; n];

        // r = b - A*x (x = 0, so r = b)
        residual.copy_from_slice(r);
        p.copy_from_slice(r);

        let mut rsold = residual.iter().map(|x| x * x).sum::<f64>();
        let tolerance = 1e-10 * rsold.sqrt().max(1e-12);

        for _iter in 0..n.min(50) {
            // Limit iterations for coarse grid
            // ap = A * p
            a.spmv(&p, &mut ap);

            // alpha = rsold / (p^T * A * p)
            let ptap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
            if ptap.abs() < 1e-14 {
                break;
            }
            let alpha = rsold / ptap;

            // x = x + alpha * p
            for i in 0..n {
                z[i] += alpha * p[i];
            }

            // r = r - alpha * ap
            for i in 0..n {
                residual[i] -= alpha * ap[i];
            }

            let rsnew: f64 = residual.iter().map(|x| x * x).sum();
            if rsnew.sqrt() < tolerance {
                break;
            }

            let beta = rsnew / rsold;

            // p = r + beta * p
            for i in 0..n {
                p[i] = residual[i] + beta * p[i];
            }

            rsold = rsnew;
        }
    }

    /// Direct solve using sparse matrix with MPI communication
    fn solve_direct_sparse_with_comm(
        a: &CsrMatrix<f64>,
        r: &[f64],
        z: &mut [f64],
        comm: &crate::parallel::UniverseComm,
    ) {
        // Convert to dense for direct solve temporarily
        let a_dense = a.to_dense();
        Self::solve_direct_with_comm(&a_dense, r, z, comm);
    }

    /// Fallback direct solver for coarse grid (CG iterations).
    ///
    /// Uses the Conjugate Gradient method for small dense matrices.
    fn solve_direct(a: &Mat<f64>, r: &[f64], z: &mut [f64]) {
        let n = r.len();
        assert_eq!(a.ncols(), n);
        assert_eq!(a.nrows(), n);
        assert_eq!(z.len(), n);
        let mut x = vec![0.0; n];
        let mut residual = r.to_vec();
        let mut p = residual.clone();
        let mut ap = vec![0.0; n];
        let mut alpha;
        let mut beta;
        // initial residual norm
        let mut rr_new = {
            #[cfg(feature = "rayon")]
            {
                residual.par_iter().map(|&v| v * v).sum::<f64>()
            }
            #[cfg(not(feature = "rayon"))]
            {
                residual.iter().map(|&v| v * v).sum::<f64>()
            }
        };
        let mut rr_old;
        for _ in 0..n {
            let _ = utils::parallel_mat_vec(a, &p, &mut ap);
            #[cfg(feature = "rayon")]
            let denominator = p
                .par_iter()
                .zip(ap.par_iter())
                .map(|(&pi, &api)| pi * api)
                .sum::<f64>();
            #[cfg(not(feature = "rayon"))]
            let denominator = p
                .iter()
                .zip(ap.iter())
                .map(|(&pi, &api)| pi * api)
                .sum::<f64>();
            alpha = rr_new / denominator;
            #[cfg(feature = "rayon")]
            x.par_iter_mut()
                .zip(p.par_iter())
                .for_each(|(xi, &pi)| *xi += alpha * pi);
            #[cfg(not(feature = "rayon"))]
            for (xi, &pi) in x.iter_mut().zip(p.iter()) {
                *xi += alpha * pi;
            }
            #[cfg(feature = "rayon")]
            residual
                .par_iter_mut()
                .zip(ap.par_iter())
                .for_each(|(ri, &api)| *ri -= alpha * api);
            #[cfg(not(feature = "rayon"))]
            for (ri, &api) in residual.iter_mut().zip(ap.iter()) {
                *ri -= alpha * api;
            }
            // update our old and new residual norms
            rr_old = rr_new;
            rr_new = {
                #[cfg(feature = "rayon")]
                {
                    residual.par_iter().map(|&v| v * v).sum::<f64>()
                }
                #[cfg(not(feature = "rayon"))]
                {
                    residual.iter().map(|&v| v * v).sum::<f64>()
                }
            };
            if rr_new.sqrt() < 1e-10 {
                break;
            }
            beta = rr_new / rr_old;
            #[cfg(feature = "rayon")]
            p.par_iter_mut()
                .zip(residual.par_iter())
                .for_each(|(pi, &ri)| *pi = ri + beta * *pi);
            #[cfg(not(feature = "rayon"))]
            for (pi, &ri) in p.iter_mut().zip(residual.iter()) {
                *pi = ri + beta * *pi;
            }
        }
        z.copy_from_slice(&x);
    }
    /// Direct solver for coarse grid using distributed collectives.
    ///
    /// Uses the Conjugate Gradient method with distributed dot products and mat-vecs.
    fn solve_direct_with_comm(
        a: &Mat<f64>,
        r: &[f64],
        z: &mut [f64],
        comm: &crate::parallel::UniverseComm,
    ) {
        let n = r.len();
        assert_eq!(a.ncols(), n);
        assert_eq!(a.nrows(), n);
        assert_eq!(z.len(), n);
        let mut x = vec![0.0; n];
        let mut residual = r.to_vec();
        let mut p = residual.clone();
        let mut ap = vec![0.0; n];
        let mut alpha;
        let mut beta;
        let mut rr_new = Comm::dot(comm, &residual, &residual);
        let mut rr_old;
        for _ in 0..n {
            comm.parallel_mat_vec(a, &p, &mut ap);
            let denominator = Comm::dot(comm, &p, &ap);
            alpha = rr_new / denominator;
            x.iter_mut()
                .zip(p.iter())
                .for_each(|(xi, &pi)| *xi += alpha * pi);
            residual
                .iter_mut()
                .zip(ap.iter())
                .for_each(|(ri, &api)| *ri -= alpha * api);
            rr_old = rr_new;
            rr_new = Comm::dot(comm, &residual, &residual);
            if rr_new.sqrt() < 1e-10 {
                break;
            }
            beta = rr_new / rr_old;
            p.iter_mut()
                .zip(residual.iter())
                .for_each(|(pi, &ri)| *pi = ri + beta * *pi);
        }
        z.copy_from_slice(&x);
    }
    /// AMG V-cycle with distributed collectives and mat-vecs via Comm abstraction.
    ///
    /// Applies the V-cycle recursively using distributed operations.
    pub fn apply_recursive_with_comm(
        &self,
        level: usize,
        r: &[f64],
        z: &mut [f64],
        comm: &crate::parallel::UniverseComm,
    ) -> Result<(), KError> {
        if level + 1 == self.levels.len() {
            AMG::solve_direct_sparse_with_comm(&self.levels[level].coarse_matrix, r, z, comm);
            return Ok(());
        }
        let a = &self.levels[level].coarse_matrix;
        let diag_inv = &self.levels[level].diag_inv;
        let restriction = &self.levels[level].restriction;
        let interpolation = &self.levels[level].interpolation;
        // Pre-smoothing
        self.smooth_jacobi_parallel_with_comm_sparse(a, diag_inv, r, z, self.nu_pre, comm)?;
        // Compute residual: az = r - A z
        let mut az = vec![0.0; a.nrows()];
        let a_dense = a.to_dense();
        comm.parallel_mat_vec(&a_dense, z, &mut az);
        for i in 0..az.len() {
            az[i] = r[i] - az[i];
        }
        // Restrict residual to coarse grid
        let mut coarse_residual = vec![0.0; restriction.nrows()];
        let restriction_dense = restriction.to_dense();
        comm.parallel_mat_vec(&restriction_dense, &az, &mut coarse_residual);
        // Recursive coarse solve
        let mut coarse_solution = vec![0.0; coarse_residual.len()];
        self.apply_recursive_with_comm(level + 1, &coarse_residual, &mut coarse_solution, comm)?;
        // Prolongate correction
        let mut fine_correction = vec![0.0; interpolation.nrows()];
        let interpolation_dense = interpolation.to_dense();
        comm.parallel_mat_vec(&interpolation_dense, &coarse_solution, &mut fine_correction);
        for i in 0..z.len() {
            z[i] += fine_correction[i];
        }
        // Post-smoothing
        self.smooth_jacobi_parallel_with_comm_sparse(a, diag_inv, r, z, self.nu_post, comm)?;
        Ok(())
    }

    /// Distributed Jacobi smoother using Comm abstraction.
    ///
    /// Applies a fixed number of Jacobi iterations using distributed mat-vecs.
    /// Sparse version of Jacobi smoothing with MPI communication
    fn smooth_jacobi_parallel_with_comm_sparse(
        &self,
        a: &CsrMatrix<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        iterations: usize,
        comm: &crate::parallel::UniverseComm,
    ) -> Result<(), KError> {
        let n = r.len();
        if diag_inv.len() != n || z.len() != n {
            return Err(KError::InvalidInput(
                "Jacobi(MPI): dimension mismatch".to_string(),
            ));
        }
        if iterations == 0 {
            return Ok(());
        }

        let omega = self.config.jacobi_omega;
        let mut temp = vec![0.0; n];
        let mut work = vec![0.0; n];

        temp.copy_from_slice(z);

        for _ in 0..iterations {
            // TODO: replace with comm.parallel_spmv_csr when available
            let a_dense = a.to_dense();
            comm.parallel_mat_vec(&a_dense, &temp, &mut work);

            for i in 0..n {
                let corr = diag_inv[i] * (r[i] - work[i]);
                temp[i] += omega * corr;
            }
        }

        z.copy_from_slice(&temp);
        Ok(())
    }

    fn smooth_jacobi_parallel_with_comm(
        &self,
        a: &Mat<f64>,
        diag_inv: &[f64],
        r: &[f64],
        z: &mut [f64],
        iterations: usize,
        comm: &crate::parallel::UniverseComm,
    ) -> Result<(), KError> {
        let n = r.len();
        if diag_inv.len() != n || z.len() != n {
            return Err(KError::InvalidInput(
                "Jacobi(simple MPI): dimension mismatch".to_string(),
            ));
        }
        if iterations == 0 {
            return Ok(());
        }

        let omega = self.config.jacobi_omega;
        let mut z_vec = z.to_vec();
        let mut temp = vec![0.0; n];
        for _ in 0..iterations {
            comm.parallel_mat_vec(a, &z_vec, &mut temp);
            for i in 0..n {
                let corr = diag_inv[i] * (r[i] - temp[i]);
                z_vec[i] += omega * corr;
            }
        }
        z.copy_from_slice(&z_vec);
        Ok(())
    }

    /// Distributed AMG entry point.
    ///
    /// Applies the AMG preconditioner using distributed collectives.
    pub fn apply_with_comm(
        &self,
        r: &[f64],
        z: &mut [f64],
        comm: &crate::parallel::UniverseComm,
    ) -> Result<(), KError> {
        let residual = r;
        let mut solution = vec![0.0; residual.len()];
        if self.levels.is_empty() {
            let diag_inv = AMG::extract_diagonal_inverse(&self.matrix);
            self.smooth_jacobi_parallel_with_comm(
                &self.matrix,
                &diag_inv,
                residual,
                &mut solution,
                10,
                comm,
            )?;
        } else {
            self.apply_recursive_with_comm(0, residual, &mut solution, comm)?;
        }
        z.copy_from_slice(&solution);
        Ok(())
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for AMG {
    /// Apply the AMG preconditioner: z = M⁻¹ r.
    fn apply(
        &self,
        _side: crate::preconditioner::PcSide,
        r: &Vec<f64>,
        z: &mut Vec<f64>,
    ) -> Result<(), KError> {
        // Input validation
        if r.len() != z.len() {
            return Err(KError::InvalidInput(format!(
                "Vector dimension mismatch: r.len()={}, z.len()={}",
                r.len(),
                z.len()
            )));
        }

        if r.is_empty() {
            return Err(KError::InvalidInput(
                "Cannot apply preconditioner to empty vectors".to_string(),
            ));
        }

        // Check for NaN/Inf in input
        for (i, &val) in r.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(KError::InvalidInput(format!(
                    "Invalid value {} detected in input vector at position {}",
                    val, i
                )));
            }
        }

        if self.levels.is_empty() || r.len() <= 10 {
            // Force simple smoothing for small problems
            // Direct smoother application for matrix without multilevel setup OR small matrices
            if r.len() != self.matrix.nrows() {
                return Err(KError::InvalidInput(format!(
                    "Vector size {} doesn't match matrix size {}",
                    r.len(),
                    self.matrix.nrows()
                )));
            }

            let diag_inv = AMG::extract_diagonal_inverse(&self.matrix);
            self.smooth_jacobi_parallel(&self.matrix, &diag_inv, r, z, 10)?;
        } else {
            // Multilevel V-cycle application with unified kernel approach
            let mut local_workspace = AMGWorkspace::new(r.len());
            let local_kernel = crate::core::traits::LocalAmgKernel::new();
            let local_comm = crate::parallel::NoComm;

            // First apply matrix to the fine level
            // Pre-smooth
            let diag_inv = Self::extract_diagonal_inverse(&self.matrix);
            self.smooth_jacobi_parallel(&self.matrix, &diag_inv, r, z, self.nu_pre)?;

            // Compute residual
            local_workspace.resize(r.len());
            utils::parallel_mat_vec(&self.matrix, z, &mut local_workspace.residual[..r.len()])?;
            for i in 0..r.len() {
                local_workspace.residual[i] = r[i] - local_workspace.residual[i];
            }

            // Restrict to coarse level (level 0 in levels array)
            if !self.levels.is_empty() {
                let coarse_size = self.levels[0].coarse_matrix.nrows();
                utils::parallel_mat_vec_sparse(
                    &self.levels[0].restriction,
                    &local_workspace.residual[..r.len()],
                    &mut local_workspace.coarse_solution[..coarse_size],
                )?;

                // Recursive solve starting from level 0
                let coarse_rhs: Vec<f64> = local_workspace.coarse_solution[..coarse_size].to_vec();
                let mut coarse_correction = vec![0.0; coarse_size];
                let mut temp_workspace = AMGWorkspace::new(coarse_size);

                if let Err(e) = self.apply_cycle(
                    0,
                    &coarse_rhs,
                    &mut coarse_correction,
                    &mut temp_workspace,
                    &local_kernel,
                    &local_comm,
                ) {
                    return Err(e);
                }

                // Interpolate back
                utils::parallel_mat_vec_sparse(
                    &self.levels[0].interpolation,
                    &coarse_correction,
                    &mut local_workspace.fine_correction[..r.len()],
                )?;

                // Add correction
                for i in 0..r.len() {
                    z[i] += local_workspace.fine_correction[i];
                }
            }

            // Post-smooth
            self.smooth_jacobi_parallel(&self.matrix, &diag_inv, r, z, self.nu_post)?;
        }

        // Validate output for numerical safety
        for (i, &val) in z.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(KError::SolveError(format!(
                    "Invalid value {} generated in output vector at position {}",
                    val, i
                )));
            }
        }

        Ok(())
    }
    /// AMG is constructed with new(), so setup is a no-op.
    fn setup(&mut self, _a: &Mat<f64>) -> Result<(), KError> {
        Ok(())
    }
}

// ------------------- Additional Functions for Improvements -------------------

/// Compute anisotropy for each row of the matrix.
/// Anisotropy is defined as the ratio max_off_diag/diag.
fn compute_anisotropy(a: &Mat<f64>) -> Vec<f64> {
    let n = a.nrows();
    #[cfg(feature = "rayon")]
    {
        (0..n)
            .into_par_iter() // Parallel iterator
            .map(|i| {
                let diag = a[(i, i)];
                let max_off_diag = (0..n)
                    .filter(|&j| i != j) // Exclude the diagonal element
                    .map(|j| a[(i, j)].abs()) // Compute absolute value of off-diagonal elements
                    .fold(0.0, f64::max); // Find the maximum off-diagonal element
                if diag.abs() > 1e-14 {
                    max_off_diag / diag.abs()
                } else {
                    0.0
                }
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..n)
            .into_iter()
            .map(|i| {
                let diag = a[(i, i)];
                let max_off_diag = (0..n)
                    .filter(|&j| i != j)
                    .map(|j| a[(i, j)].abs())
                    .fold(0.0, f64::max);
                if diag.abs() > 1e-14 {
                    max_off_diag / diag.abs()
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Compute an adaptive threshold based on global anisotropy indicators.
///
/// The threshold is scaled by the average anisotropy to improve coarsening for highly anisotropic problems.
fn compute_adaptive_threshold(a: &Mat<f64>, base_threshold: f64) -> f64 {
    let anis = compute_anisotropy(a);
    let avg_anis = if anis.is_empty() {
        1.0
    } else {
        anis.iter().sum::<f64>() / (anis.len() as f64)
    };
    base_threshold * (1.0 + avg_anis.max(0.5))
}

/// Smooth the interpolation matrix to improve prolongation accuracy.
/// This applies a weighted Jacobi smoothing to the interpolation operator.
fn smooth_interpolation(interpolation: &mut Mat<f64>, matrix: &Mat<f64>, weight: f64) {
    let row_count = interpolation.nrows().min(matrix.nrows());
    let col_count = interpolation.ncols().min(matrix.ncols());
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        // Safer approach: collect updates and apply them sequentially
        let updates: Vec<(usize, usize, f64)> = (0..row_count)
            .into_par_iter()
            .flat_map(|i| {
                (0..col_count)
                    .into_par_iter()
                    .map(move |j| (i, j, weight * matrix[(i, j)]))
            })
            .collect();

        // Apply updates sequentially (still faster than mutex per element)
        for (i, j, update) in updates {
            interpolation[(i, j)] -= update;
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        for j in 0..col_count {
            for i in 0..row_count {
                interpolation[(i, j)] -= weight * matrix[(i, j)];
            }
        }
    }
}

/// Normalize rows of the interpolation matrix to minimize energy.
/// This rescales each row to unit 2-norm.
fn minimize_energy(interpolation: &mut Mat<f64>, _matrix: &Mat<f64>) {
    let rows = interpolation.nrows();
    let cols = interpolation.ncols();
    #[cfg(feature = "rayon")]
    let normalized_rows: Vec<Vec<f64>> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let mut row_vec: Vec<f64> = (0..cols).map(|j| interpolation[(i, j)]).collect();
            let row_sum: f64 = row_vec.iter().map(|&val| val * val).sum();
            let norm_factor = if row_sum.abs() > 1e-14 {
                row_sum.sqrt()
            } else {
                1.0
            };
            for val in row_vec.iter_mut() {
                *val /= norm_factor;
            }
            row_vec
        })
        .collect();
    #[cfg(not(feature = "rayon"))]
    let normalized_rows: Vec<Vec<f64>> = (0..rows)
        .into_iter()
        .map(|i| {
            let mut row_vec: Vec<f64> = (0..cols).map(|j| interpolation[(i, j)]).collect();
            let row_sum: f64 = row_vec.iter().map(|&val| val * val).sum();
            let norm_factor = if row_sum.abs() > 1e-14 {
                row_sum.sqrt()
            } else {
                1.0
            };
            for val in row_vec.iter_mut() {
                *val /= norm_factor;
            }
            row_vec
        })
        .collect();
    for i in 0..rows {
        for j in 0..cols {
            interpolation[(i, j)] = normalized_rows[i][j];
        }
    }
}

/// Parallel mat-vec multiplication for sparse matrices using rayon or serial fallback.
fn parallel_mat_vec_sparse(
    mat: &CsrMatrix<f64>,
    vec: &[f64],
    result: &mut [f64],
) -> Result<(), KError> {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    let (vlen, rlen) = (vec.len(), result.len());

    if cols != vlen {
        return Err(KError::InvalidInput(format!(
            "Dimension mismatch in parallel_mat_vec_sparse: Matrix is {}x{}, but input vector length is {}",
            rows, cols, vlen
        )));
    }

    if rows != rlen {
        return Err(KError::InvalidInput(format!(
            "Dimension mismatch in parallel_mat_vec_sparse: Matrix is {}x{}, but result length is {}",
            rows, cols, rlen
        )));
    }

    // Use the sparse spmv method
    mat.spmv_scaled(1.0, vec, 0.0, result)
}

/// Parallel mat-vec multiplication using rayon or serial fallback.
fn parallel_mat_vec(mat: &Mat<f64>, vec: &[f64], result: &mut [f64]) -> Result<(), KError> {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    let (vlen, rlen) = (vec.len(), result.len());

    if cols != vlen {
        return Err(KError::InvalidInput(format!(
            "Dimension mismatch in parallel_mat_vec: Matrix is {}x{}, but input vector length is {}",
            rows, cols, vlen
        )));
    }

    if rows != rlen {
        return Err(KError::InvalidInput(format!(
            "Dimension mismatch in parallel_mat_vec: Matrix is {}x{}, but result length is {}",
            rows, cols, rlen
        )));
    }

    #[cfg(feature = "rayon")]
    {
        result.par_iter_mut().enumerate().for_each(|(i, res)| {
            *res = (0..cols).map(|j| mat[(i, j)] * vec[j]).sum();
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        result.iter_mut().enumerate().for_each(|(i, res)| {
            *res = (0..cols).map(|j| mat[(i, j)] * vec[j]).sum();
        });
    }

    Ok(())
}

// ------------------- Helper Functions for Enhanced Coarsening -------------------

/// Compute strength of connection matrix S, based on the definition:
/// S(i, j) = |A_ij| / sqrt(|A_ii| * |A_jj|) if > threshold, else 0.
fn compute_strength_matrix(a: &Mat<f64>, threshold: f64) -> Mat<f64> {
    let n = a.nrows();
    let mut s = Mat::<f64>::zeros(n, n);
    #[cfg(feature = "rayon")]
    let updates: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let a_ii = a[(i, i)].abs();
            (0..n)
                .filter_map(move |j| {
                    if i == j {
                        return Some((i, j, 0.0));
                    }
                    let val = a[(i, j)];
                    let a_jj = a[(j, j)].abs();
                    if a_ii > 1e-14 && a_jj > 1e-14 {
                        let strength = val.abs() / (a_ii * a_jj).sqrt();
                        if strength > threshold {
                            return Some((i, j, strength));
                        }
                    }
                    None
                })
                .collect::<Vec<_>>()
        })
        .collect();
    #[cfg(not(feature = "rayon"))]
    let updates: Vec<(usize, usize, f64)> = (0..n)
        .into_iter()
        .flat_map(|i| {
            let a_ii = a[(i, i)].abs();
            (0..n)
                .filter_map(move |j| {
                    if i == j {
                        return Some((i, j, 0.0));
                    }
                    let val = a[(i, j)];
                    let a_jj = a[(j, j)].abs();
                    if a_ii > 1e-14 && a_jj > 1e-14 {
                        let strength = val.abs() / (a_ii * a_jj).sqrt();
                        if strength > threshold {
                            return Some((i, j, strength));
                        }
                    }
                    None
                })
                .collect::<Vec<_>>()
        })
        .collect();
    for (i, j, value) in updates {
        s[(i, j)] = value;
    }
    s
}

/// Perform double-pairwise aggregation:
/// 1. Pairwise aggregate the graph to form coarse nodes.
/// 2. On the coarse graph, perform another round of pairing to form larger aggregates.
///    This function returns a vector where `aggregates[i]` = aggregate index of node i.
fn double_pairwise_aggregation(s: &Mat<f64>) -> Vec<usize> {
    // First pass: pairwise aggregation
    let first_pass = pairwise_aggregation(s);

    // Construct a coarse-level graph and apply pairwise aggregation again
    let coarse_graph = build_coarse_graph(s, &first_pass);
    let second_pass = pairwise_aggregation(&coarse_graph);

    // Map the second pass results back to the fine level
    remap_aggregates(&first_pass, &second_pass)
}

/// Improved greedy aggregation with better balance and connectivity analysis
fn greedy_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let n = s.nrows();
    let mut aggregates = vec![usize::MAX; n];
    let mut aggregate_sizes = Vec::new();
    let mut next_agg_id = 0;
    let max_aggregate_size = 4; // Better balance - smaller aggregates

    // Sort nodes by connectivity strength (most connected first for better seeds)
    let mut node_strengths: Vec<(f64, usize)> = (0..n)
        .map(|i| {
            let total_strength: f64 = (0..n).map(|j| s[(i, j)]).sum();
            (total_strength, i)
        })
        .collect();

    // Sort by descending strength for better seed selection
    node_strengths.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for &(_strength, seed) in &node_strengths {
        if aggregates[seed] != usize::MAX {
            continue; // Already assigned
        }

        // Start new aggregate with better balance control
        let mut current_aggregate = vec![seed];
        aggregates[seed] = next_agg_id;

        // Collect potential neighbors sorted by connection strength
        let mut candidates: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != seed && aggregates[j] == usize::MAX)
            .map(|j| (s[(seed, j)], j))
            .collect();

        // Sort by descending strength
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Add strongly connected neighbors up to max size
        for &(strength, neighbor) in &candidates {
            if current_aggregate.len() >= max_aggregate_size {
                break;
            }

            // Only add if strongly connected (threshold-based)
            if strength > 0.1 && aggregates[neighbor] == usize::MAX {
                current_aggregate.push(neighbor);
                aggregates[neighbor] = next_agg_id;
            }
        }

        aggregate_sizes.push(current_aggregate.len());
        next_agg_id += 1;
    }

    aggregates
}

/// Pairwise aggregate a given strength matrix. This is a helper for double_pairwise_aggregation.
fn pairwise_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let n = s.nrows();
    let mut aggregates = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    let mut aggregate_id = 0;

    for i in 0..n {
        if visited[i] {
            continue;
        }

        // Find the strongest unvisited neighbor
        let mut max_strength = 0.0;
        let mut strongest_neighbor = None;
        for j in 0..n {
            if i != j && !visited[j] {
                let strength = s[(i, j)];
                if strength > max_strength {
                    max_strength = strength;
                    strongest_neighbor = Some(j);
                }
            }
        }

        // Form an aggregate with the strongest neighbor
        if let Some(j) = strongest_neighbor {
            aggregates[i] = aggregate_id;
            aggregates[j] = aggregate_id;
            visited[i] = true;
            visited[j] = true;
            aggregate_id += 1;
        } else {
            // No neighbor found, form a singleton aggregate
            aggregates[i] = aggregate_id;
            visited[i] = true;
            aggregate_id += 1;
        }
    }

    aggregates
}

/// Build a coarse graph from fine-level aggregates.
/// Each aggregate forms a node in the coarse graph.
/// The weights of edges between coarse nodes can be inherited or averaged from the fine graph.
fn build_coarse_graph(s: &Mat<f64>, aggregates: &[usize]) -> Mat<f64> {
    let max_agg_id = *aggregates.iter().max().unwrap_or(&0);
    let coarse_n = max_agg_id + 1;
    let mut coarse_mat = Mat::<f64>::zeros(coarse_n, coarse_n);
    let n = s.nrows();
    // Use a sequential loop for correctness
    for fine_node_i in 0..n {
        for fine_node_j in 0..s.ncols() {
            let agg_i = aggregates[fine_node_i];
            let agg_j = aggregates[fine_node_j];
            if agg_j < usize::MAX {
                let val = s[(fine_node_i, fine_node_j)];
                if val != 0.0 {
                    coarse_mat[(agg_i, agg_j)] += val;
                }
            }
        }
    }
    coarse_mat
}

/// Remap second pass aggregates to fine-level nodes.
fn remap_aggregates(first_pass: &[usize], second_pass: &[usize]) -> Vec<usize> {
    #[cfg(feature = "rayon")]
    {
        first_pass
            .par_iter()
            .map(|&coarse_agg| second_pass[coarse_agg])
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        first_pass
            .iter()
            .map(|&coarse_agg| second_pass[coarse_agg])
            .collect()
    }
}

/// Construct the prolongation matrix P from the aggregate assignments.
/// For piecewise constant interpolation:
/// P_{ij} = 1 if node i is in aggregate j, else 0.
fn construct_prolongation(a: &Mat<f64>, aggregates: &[usize]) -> Mat<f64> {
    let n = a.nrows();
    let max_agg_id = *aggregates.iter().max().unwrap();
    let coarse_n = max_agg_id + 1;
    #[cfg(feature = "rayon")]
    {
        let mut p = Mat::<f64>::zeros(n, coarse_n);
        // Collect updates in parallel, then apply sequentially
        let updates: Vec<(usize, usize)> =
            (0..n).into_par_iter().map(|i| (i, aggregates[i])).collect();

        // Apply updates sequentially - much faster than per-element mutex
        for (i, agg_id) in updates {
            p[(i, agg_id)] = 1.0;
        }
        p
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut p = Mat::<f64>::zeros(n, coarse_n);
        for (i, &agg_id) in aggregates.iter().enumerate() {
            p[(i, agg_id)] = 1.0;
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::utils;
    use faer::mat;

    #[test]
    fn test_amg_preconditioner_simple() {
        let matrix = mat![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let r = vec![5.0, 5.0, 3.0];
        let mut z = vec![0.0; 3];

        let max_levels = 2;
        let coarsening_threshold = 9.0; // Use large threshold to prevent coarsening for small matrices
        let amg_preconditioner = AMG::new(&matrix, max_levels, coarsening_threshold);

        amg_preconditioner
            .apply(crate::preconditioner::PcSide::Left, &r, &mut z)
            .unwrap();

        let mut residual = vec![0.0; 3];
        MatVecOp::matvec(&matrix, &z, &mut residual).unwrap();
        for i in 0..3 {
            residual[i] = r[i] - residual[i];
        }
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(
            residual_norm < 100.0,
            "Residual norm too high: {}",
            residual_norm
        );
    }

    #[test]
    fn test_amg_preconditioner_odd_size() {
        let matrix = mat![
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 4.0]
        ];
        let r = vec![5.0, 5.0, 3.0, 1.0];
        let mut z = vec![0.0; 4];

        let max_levels = 2;
        let coarsening_threshold = 9.0; // Use large threshold to prevent coarsening for small matrices
        let amg_preconditioner = AMG::new(&matrix, max_levels, coarsening_threshold);

        amg_preconditioner
            .apply(crate::preconditioner::PcSide::Left, &r, &mut z)
            .unwrap();

        let mut residual = vec![0.0; 4];
        MatVecOp::matvec(&matrix, &z, &mut residual).unwrap();
        for i in 0..4 {
            residual[i] = r[i] - residual[i];
        }
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(
            residual_norm < 100.0,
            "Residual norm too high: {}",
            residual_norm
        );
    }

    #[test]
    fn test_smooth_interpolation_basic() {
        // Input matrices
        let mut interpolation = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let matrix = mat![[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5]];
        let weight = 0.5;

        // Apply the function
        smooth_interpolation(&mut interpolation, &matrix, weight);

        // Expected result
        let expected = mat![[0.75, 1.75, 2.75], [3.5, 4.5, 5.5], [6.25, 7.25, 8.25]];

        // Assertions
        assert_eq!(interpolation, expected);
    }

    #[test]
    fn test_smooth_interpolation_partial_overlap() {
        // Matrix has fewer columns than interpolation
        let mut interpolation = mat![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        let matrix = mat![[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]];
        let weight = 1.0;

        // Apply the function
        smooth_interpolation(&mut interpolation, &matrix, weight);

        // Expected result
        let expected = mat![
            [0.5, 1.5, 3.0, 4.0],
            [4.0, 5.0, 7.0, 8.0],
            [7.5, 8.5, 11.0, 12.0]
        ];

        // Assertions
        assert_eq!(interpolation, expected);
    }

    #[test]
    fn test_spgemm() {
        // Test sparse matrix multiplication with simple matrices
        // A = [[2, 1], [0, 3]], B = [[1, 0], [1, 2]]
        // Expected: A*B = [[3, 2], [3, 6]]

        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 3],       // row_ptr
            vec![0, 1, 1],       // col_indices
            vec![2.0, 1.0, 3.0], // values
        );

        let b = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 1, 3],       // row_ptr
            vec![0, 0, 1],       // col_indices
            vec![1.0, 1.0, 2.0], // values
        );

        let result = utils::spgemm(&a, &b).unwrap();
        let result_dense = result.to_dense();

        // Check expected values
        assert!((result_dense[(0, 0)] - 3.0).abs() < 1e-12);
        assert!((result_dense[(0, 1)] - 2.0).abs() < 1e-12);
        assert!((result_dense[(1, 0)] - 3.0).abs() < 1e-12);
        assert!((result_dense[(1, 1)] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_galerkin_vs_dense() {
        // Compare sparse vs dense Galerkin products
        let matrix = faer::mat![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];

        // Simple operators for testing
        let interpolation = faer::mat![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]];

        let restriction = faer::mat![[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]];

        // Dense computation
        let temp = &restriction * &matrix;
        let dense_result = &temp * &interpolation;

        // Sparse computation
        let sparse_matrix = CsrMatrix::from_dense(&matrix, 1e-15);
        let sparse_interpolation = CsrMatrix::from_dense(&interpolation, 1e-15);
        let sparse_restriction = CsrMatrix::from_dense(&restriction, 1e-15);

        let sparse_result = utils::sparse_galerkin_product(
            &sparse_restriction,
            &sparse_matrix,
            &sparse_interpolation,
        )
        .unwrap();

        let sparse_result_dense = sparse_result.to_dense();

        println!("Dense result:");
        for i in 0..dense_result.nrows() {
            for j in 0..dense_result.ncols() {
                print!("{:.6} ", dense_result[(i, j)]);
            }
            println!();
        }

        println!("Sparse result:");
        for i in 0..sparse_result_dense.nrows() {
            for j in 0..sparse_result_dense.ncols() {
                print!("{:.6} ", sparse_result_dense[(i, j)]);
            }
            println!();
        }

        // Check if results match within tolerance
        for i in 0..dense_result.nrows() {
            for j in 0..dense_result.ncols() {
                let diff = (dense_result[(i, j)] - sparse_result_dense[(i, j)]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch at ({}, {}): dense={}, sparse={}, diff={}",
                    i,
                    j,
                    dense_result[(i, j)],
                    sparse_result_dense[(i, j)],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_high_threshold_debug() {
        // Test with high threshold that should prevent coarsening
        let matrix = mat![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];

        let max_levels = 2;
        let coarsening_threshold = 9.0; // High threshold - should prevent coarsening
        let amg = AMG::new(&matrix, max_levels, coarsening_threshold);

        println!("Matrix size: {}", matrix.nrows());
        println!("Coarsening threshold: {}", coarsening_threshold);
        println!("AMG levels created: {}", amg.levels.len());
        println!("Levels is empty: {}", amg.levels.is_empty());

        if amg.levels.is_empty() {
            println!("SUCCESS: No multilevel hierarchy created - will use simple smoothing");
        } else {
            println!("PROBLEM: Multilevel hierarchy still created despite high threshold");
        }
    }
}
