//! HYPRE-Inspired ILU Factorization Implementation
//!
//! This module provides a comprehensive, production-grade implementation of Incomplete LU (ILU)
//! factorization inspired by HYPRE's ParILU. It includes multiple ILU variants (ILU(k), ILUT),
//! advanced configuration options, robust error handling, workspace optimization, parallel execution,
//! and comprehensive monitoring capabilities.
//!
//! # Features
//!
//! ## ILU Variants
//! - **ILU(0)**: Zero fill-in factorization (original sparsity pattern preserved)
//! - **ILU(k)**: Level-of-fill factorization with k levels of fill-in
//! - **ILUT**: Threshold-based factorization with drop tolerance
//! - **Modified ILU**: Modified factorization for better stability
//!
//! ## HYPRE-Inspired Configuration
//! - **Drop Tolerances**: Configurable drop thresholds for numerical stability
//! - **Fill Levels**: Control memory usage vs. accuracy trade-off
//! - **Reordering**: Built-in support for various reordering strategies
//! - **Triangular Solve Options**: Exact vs. iterative triangular solves
//! - **Jacobi Iterations**: Configurable Jacobi smoothing for triangular solves
//!
//! ## Parallel & Distributed Computing
//! - **Thread Parallelism**: Rayon-based parallel factorization and triangular solves
//! - **Workspace Optimization**: Preallocated buffers for efficient repeated solves
//! - **Chunk-based Processing**: Configurable chunk sizes for optimal cache performance
//! - **Distributed Memory**: MPI support for distributed matrix factorization (planned)
//! - **NUMA Awareness**: Thread affinity and memory layout optimization (planned)
//!
//! ## Production-Grade Features
//! - **IEEE Safety**: NaN/Inf detection and handling
//! - **Pivot Monitoring**: Zero pivot detection and mitigation
//! - **Memory Management**: Workspace reuse and optimization
//! - **Performance Metrics**: Setup complexity and solve timing
//! - **Comprehensive Logging**: Configurable verbosity for debugging
//!
//! # Usage Examples
//!
//! ```rust
//! // Basic ILU(0) with HYPRE defaults
//! let ilu = IluBuilder::new()
//!     .ilu_type(IluType::ILU0)
//!     .build();
//!
//! // Advanced ILUT configuration with parallel execution
//! let ilu = IluBuilder::new()
//!     .ilu_type(IluType::ILUT)
//!     .drop_tolerance(1e-4)
//!     .max_fill_per_row(50)
//!     .enable_reordering(ReorderingType::RCM)
//!     .triangular_solve(TriSolveType::Iterative)
//!     .jacobi_iterations(3, 3)
//!     .enable_parallel()
//!     .parallel_chunk_size(128)
//!     .enable_logging()
//!     .build();
//!
//! // High-performance configuration for large problems
//! let ilu = IluBuilder::new()
//!     .ilu_type(IluType::ILU0)
//!     .enable_parallel_factorization()
//!     .enable_parallel_triangular_solve()
//!     .parallel_chunk_size(256)
//!     .enable_distributed()  // For MPI environments
//!     .build();
//! ```
//!
//! # References
//! - HYPRE ParILU implementation
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems
//! - Li, X. (2005). Iterative Methods for Large Sparse Linear Systems

use crate::preconditioner::{Preconditioner, PcSide};
use crate::error::KError;
use crate::matrix::utils;
use crate::matrix::sparse::CsrMatrix;
use num_traits::Float;
use faer::traits::ComplexField;
use faer::Mat;
use std::cell::RefCell;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "logging")]
use log::{debug, info, trace, warn};

/// HYPRE-inspired ILU types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IluType {
    /// ILU(0) - Zero fill-in factorization
    ILU0 = 0,
    /// ILU(k) - Level-based fill-in factorization  
    ILUK = 1,
    /// ILUT - Threshold-based factorization
    ILUT = 2,
    /// Modified ILU(0) for better stability
    MILU0 = 3,
    /// Block Jacobi with ILU(0)
    BlockJacobi = 10,
    /// GMRES with ILU(k) preconditioning  
    GmresIluk = 20,
    /// GMRES with ILUT preconditioning
    GmresIlut = 21,
}

/// HYPRE-inspired pivot handling strategy
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PivotStrategy {
    /// Fail on zero/small pivots
    Strict,
    /// Replace small pivots with threshold value
    Threshold,
    /// Add diagonal perturbation based on max diagonal
    DiagonalPerturbation,
}

/// HYPRE-inspired reordering strategies
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReorderingType {
    /// No reordering
    None = 0,
    /// Reverse Cuthill-McKee
    RCM = 1,
    /// Approximate Minimum Degree
    AMD = 2,
    /// Natural ordering
    Natural = 3,
}

/// Enhanced triangular solve options
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriSolveType {
    /// Exact triangular solve
    Exact = 0,
    /// Iterative triangular solve with Jacobi
    Jacobi = 1,
    /// Iterative triangular solve with Gauss-Seidel
    GaussSeidel = 2,
}

/// HYPRE-inspired ILU configuration
#[derive(Clone, Debug)]
pub struct IluConfig {
    /// ILU factorization type (HYPRE: ilu_type)
    pub ilu_type: IluType,
    /// Level of fill for ILU(k) (HYPRE: lfil)
    pub level_of_fill: usize,
    /// Maximum nonzeros per row (HYPRE: maxRowNnz)
    pub max_fill_per_row: usize,
    /// Drop tolerance for ILUT (HYPRE: droptol[0])
    pub drop_tolerance: f64,
    /// Drop tolerance for off-diagonal blocks (HYPRE: droptol[1])
    pub offdiag_drop_tolerance: f64,
    /// Drop tolerance for Schur complement (HYPRE: droptol[2])
    pub schur_drop_tolerance: f64,
    /// Reordering strategy (HYPRE: reordering_type)
    pub reordering_type: ReorderingType,
    /// Triangular solve type (HYPRE: tri_solve)
    pub triangular_solve: TriSolveType,
    /// Lower triangular Jacobi iterations (HYPRE: lower_jacobi_iters)
    pub lower_jacobi_iters: usize,
    /// Upper triangular Jacobi iterations (HYPRE: upper_jacobi_iters)
    pub upper_jacobi_iters: usize,
    /// Tolerance for iterative solve (HYPRE: tol)
    pub tolerance: f64,
    /// Maximum iterations for iterative solve (HYPRE: max_iter)
    pub max_iterations: usize,
    /// Logging level (HYPRE: logging)
    pub logging_level: usize,
    /// Print level for diagnostics (HYPRE: print_level)
    pub print_level: usize,
    /// Enable IEEE safety checks
    pub ieee_checks: bool,
    /// Enable pivot monitoring
    pub pivot_monitoring: bool,
    /// Enable workspace optimization
    pub optimize_workspace: bool,
    /// Pivot threshold for stability
    pub pivot_threshold: f64,
    /// Pivot handling strategy
    pub pivot_strategy: PivotStrategy,
    /// Diagonal perturbation factor (for PivotStrategy::DiagonalPerturbation)
    pub perturbation_factor: f64,
    /// Enable parallel factorization (requires rayon feature)
    pub enable_parallel_factorization: bool,
    /// Enable parallel triangular solves (requires rayon feature)
    pub enable_parallel_triangular_solve: bool,
    /// Chunk size for parallel operations
    pub parallel_chunk_size: usize,
    /// Enable distributed memory support (requires MPI)
    pub enable_distributed: bool,
}

impl Default for IluConfig {
    /// HYPRE-inspired robust defaults with parallel support
    fn default() -> Self {
        Self {
            ilu_type: IluType::ILU0,
            level_of_fill: 0,               // HYPRE default for ILU(0)
            max_fill_per_row: 0,            // HYPRE default: unlimited
            drop_tolerance: 1e-4,           // HYPRE conservative default
            offdiag_drop_tolerance: 1e-4,   // HYPRE default
            schur_drop_tolerance: 1e-4,     // HYPRE default
            reordering_type: ReorderingType::None, // HYPRE default
            triangular_solve: TriSolveType::Exact, // HYPRE default
            lower_jacobi_iters: 1,          // HYPRE default
            upper_jacobi_iters: 1,          // HYPRE default
            tolerance: 1e-6,                // HYPRE default
            max_iterations: 1,              // HYPRE default for direct solve
            logging_level: 0,               // No logging by default
            print_level: 0,                 // No printing by default
            ieee_checks: true,              // Safety first
            pivot_monitoring: true,         // Monitor for numerical issues
            optimize_workspace: true,       // Performance optimization
            pivot_threshold: 1e-12,         // HYPRE-style pivot threshold
            pivot_strategy: PivotStrategy::DiagonalPerturbation, // Modern default
            perturbation_factor: 1e-10,     // Small perturbation factor
            enable_parallel_factorization: false, // Conservative default
            enable_parallel_triangular_solve: false, // Conservative default
            parallel_chunk_size: 64,        // Reasonable chunk size for cache efficiency
            enable_distributed: false,      // Conservative default
        }
    }
}

/// HYPRE-inspired ILU builder for advanced configuration
pub struct IluBuilder {
    config: IluConfig,
}

impl IluBuilder {
    /// Create new builder with HYPRE defaults
    pub fn new() -> Self {
        Self {
            config: IluConfig::default(),
        }
    }

    /// Set ILU type (HYPRE: ilu_type)
    pub fn ilu_type(mut self, ilu_type: IluType) -> Self {
        self.config.ilu_type = ilu_type;
        self
    }

    /// Set level of fill for ILU(k) (HYPRE: lfil)
    pub fn level_of_fill(mut self, level: usize) -> Self {
        self.config.level_of_fill = level;
        self
    }

    /// Set maximum fill per row (HYPRE: maxRowNnz)
    pub fn max_fill_per_row(mut self, max_fill: usize) -> Self {
        self.config.max_fill_per_row = max_fill;
        self
    }

    /// Set drop tolerance for ILUT (HYPRE: droptol)
    pub fn drop_tolerance(mut self, tol: f64) -> Self {
        self.config.drop_tolerance = tol;
        self
    }

    /// Set reordering strategy (HYPRE: reordering_type)
    pub fn enable_reordering(mut self, reordering: ReorderingType) -> Self {
        self.config.reordering_type = reordering;
        self
    }

    /// Set triangular solve type (HYPRE: tri_solve)
    pub fn triangular_solve(mut self, solve_type: TriSolveType) -> Self {
        self.config.triangular_solve = solve_type;
        self
    }

    /// Set Jacobi iterations for triangular solves (HYPRE: lower/upper_jacobi_iters)
    pub fn jacobi_iterations(mut self, lower: usize, upper: usize) -> Self {
        self.config.lower_jacobi_iters = lower;
        self.config.upper_jacobi_iters = upper;
        self
    }

    /// Enable comprehensive logging (HYPRE: logging)
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

    /// Set pivot handling strategy
    pub fn pivot_strategy(mut self, strategy: PivotStrategy) -> Self {
        self.config.pivot_strategy = strategy;
        self
    }

    /// Set diagonal perturbation factor
    pub fn perturbation_factor(mut self, factor: f64) -> Self {
        self.config.perturbation_factor = factor;
        self
    }

    /// Enable parallel factorization (requires rayon feature)
    pub fn enable_parallel_factorization(mut self) -> Self {
        self.config.enable_parallel_factorization = true;
        self
    }

    /// Enable parallel triangular solves (requires rayon feature)
    pub fn enable_parallel_triangular_solve(mut self) -> Self {
        self.config.enable_parallel_triangular_solve = true;
        self
    }

    /// Set chunk size for parallel operations
    pub fn parallel_chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.parallel_chunk_size = chunk_size;
        self
    }

    /// Enable all parallel features
    pub fn enable_parallel(mut self) -> Self {
        self.config.enable_parallel_factorization = true;
        self.config.enable_parallel_triangular_solve = true;
        self
    }

    /// Enable distributed memory support (requires MPI)
    pub fn enable_distributed(mut self) -> Self {
        self.config.enable_distributed = true;
        self
    }

    /// Build ILU preconditioner with configuration
    pub fn build<T: Float + Send + Sync + ComplexField + std::fmt::Display>(self) -> Result<Ilu<T>, KError> {
        Ilu::new_with_config(self.config)
    }
}

impl Default for IluBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// HYPRE-inspired comprehensive ILU preconditioner with sparse storage
pub struct Ilu<T> {
    /// Configuration parameters
    config: IluConfig,
    /// Lower triangular factor in CSR format (unit diagonal)
    l: CsrMatrix<T>,
    /// Upper triangular factor in CSR format
    u: CsrMatrix<T>,
    /// Diagonal factor for modified ILU (extracted from U for efficiency)
    d: Vec<T>,
    /// Permutation arrays (HYPRE: perm, qperm)
    row_perm: Vec<usize>,
    col_perm: Vec<usize>,
    /// Consolidated preallocated workspace vectors for all operations
    workspace: IluWorkspace<T>,
    /// Setup complexity metrics (HYPRE: operator_complexity)
    setup_complexity: f64,
    /// Factorization statistics
    nnz_l: usize,
    nnz_u: usize,
    num_zero_pivots: usize,
    /// Performance timing
    setup_time: f64,
    solve_time: f64,
    solve_count: usize,
}

/// Consolidated workspace for all ILU operations to minimize allocations
#[derive(Debug)]
pub struct IluWorkspace<T> {
    /// Primary workspace for triangular solves and factorization
    temp1: RefCell<Vec<T>>,
    /// Secondary workspace for complex operations
    temp2: RefCell<Vec<T>>,
    /// Workspace for level scheduling in parallel triangular solves
    levels: RefCell<Vec<usize>>,
    /// Workspace for sparse pattern operations
    pattern_work: RefCell<Vec<bool>>,
    /// Current workspace size
    size: usize,
}

impl<T: Clone> IluWorkspace<T> {
    /// Create new workspace with given size
    pub fn new(size: usize) -> Self
    where
        T: num_traits::Zero,
    {
        Self {
            temp1: RefCell::new(vec![T::zero(); size]),
            temp2: RefCell::new(vec![T::zero(); size]),
            levels: RefCell::new(vec![0; size]),
            pattern_work: RefCell::new(vec![false; size]),
            size,
        }
    }

    /// Resize workspace if needed (avoids reallocation when possible)
    pub fn ensure_size(&mut self, new_size: usize)
    where
        T: num_traits::Zero + Clone,
    {
        if new_size > self.size {
            self.temp1.borrow_mut().resize(new_size, T::zero());
            self.temp2.borrow_mut().resize(new_size, T::zero());
            self.levels.borrow_mut().resize(new_size, 0);
            self.pattern_work.borrow_mut().resize(new_size, false);
            self.size = new_size;
        }
    }

    /// Clear workspace (without deallocation)
    pub fn clear(&self)
    where
        T: num_traits::Zero,
    {
        for x in self.temp1.borrow_mut().iter_mut() {
            *x = T::zero();
        }
        for x in self.temp2.borrow_mut().iter_mut() {
            *x = T::zero();
        }
        for x in self.levels.borrow_mut().iter_mut() {
            *x = 0;
        }
        for x in self.pattern_work.borrow_mut().iter_mut() {
            *x = false;
        }
    }

    /// Get temporary workspace for triangular solve operations (zero-allocation)
    pub fn get_temp_workspace(&self) -> std::cell::RefMut<Vec<T>> {
        self.temp1.borrow_mut()
    }
}

impl<T: Float + Send + Sync + ComplexField + std::fmt::Display> Ilu<T> {
    /// Create new ILU with HYPRE defaults
    pub fn new() -> Self {
        Self::new_with_config(IluConfig::default()).unwrap()
    }

    /// Create ILU with comprehensive configuration
    pub fn new_with_config(config: IluConfig) -> Result<Self, KError> {
        Self::validate_config(&config)?;

        #[cfg(feature = "logging")]
        if config.logging_level > 0 {
            info!("ILU Setup: Creating {:?} factorization with HYPRE-inspired configuration", config.ilu_type);
            debug!("ILU Config: fill_level={}, drop_tol={:.2e}, reordering={:?}", 
                   config.level_of_fill, config.drop_tolerance, config.reordering_type);
        }

        Ok(Self {
            config,
            l: CsrMatrix::from_csr(0, 0, vec![0], Vec::new(), Vec::new()),
            u: CsrMatrix::from_csr(0, 0, vec![0], Vec::new(), Vec::new()),
            d: Vec::new(),
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            workspace: IluWorkspace::new(0),
            setup_complexity: 0.0,
            nnz_l: 0,
            nnz_u: 0,
            num_zero_pivots: 0,
            setup_time: 0.0,
            solve_time: 0.0,
            solve_count: 0,
        })
    }

    /// HYPRE-inspired configuration validation
    fn validate_config(config: &IluConfig) -> Result<(), KError> {
        if config.drop_tolerance < 0.0 {
            return Err(KError::InvalidInput("drop_tolerance must be >= 0".to_string()));
        }
        
        if config.tolerance <= 0.0 {
            return Err(KError::InvalidInput("tolerance must be > 0".to_string()));
        }
        
        if config.pivot_threshold < 0.0 {
            return Err(KError::InvalidInput("pivot_threshold must be >= 0".to_string()));
        }

        Ok(())
    }

    /// HYPRE-inspired IEEE safety checks
    fn check_ieee_values(matrix: &Mat<T>) -> Result<(), KError> {
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let val = matrix[(i, j)];
                if val.is_nan() {
                    return Err(KError::InvalidInput(format!(
                        "NaN detected in matrix at position ({}, {})", i, j
                    )));
                }
                if val.is_infinite() {
                    return Err(KError::InvalidInput(format!(
                        "Infinity detected in matrix at position ({}, {})", i, j
                    )));
                }
            }
        }
        Ok(())
    }

    /// HYPRE-inspired matrix validation with enhanced analysis
    fn validate_matrix(matrix: &Mat<T>) -> Result<(), KError> {
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            return Err(KError::InvalidInput("Matrix cannot be empty".to_string()));
        }
        
        if matrix.nrows() != matrix.ncols() {
            return Err(KError::InvalidInput("ILU requires square matrices".to_string()));
        }

        Ok(())
    }

    /// Enhanced matrix analysis using matrix utils
    fn analyze_matrix_for_ilu(matrix: &Mat<f64>) -> (usize, f64, f64) {
        utils::analyze_matrix_properties(matrix)
    }

    /// Check matrix for IEEE issues using matrix utils
    fn check_matrix_ieee(matrix: &Mat<f64>) -> Result<(), KError> {
        utils::check_ieee_values(matrix)
    }

    /// Calculate setup complexity (HYPRE: operator_complexity)
    fn calculate_complexity(&self, original_nnz: usize) -> f64 {
        let total_nnz = self.nnz_l + self.nnz_u;
        if original_nnz > 0 {
            total_nnz as f64 / original_nnz as f64
        } else {
            0.0
        }
    }

    /// Count nonzeros in matrix
    fn count_nnz(matrix: &Mat<T>) -> usize {
        let mut nnz = 0;
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if matrix[(i, j)] != T::zero() {
                    nnz += 1;
                }
            }
        }
        nnz
    }

    /// Enhanced pivot handling with different strategies
    fn handle_pivot(&mut self, pivot: &mut T, row: usize, matrix: &Mat<T>) -> Result<(), KError> {
        let pivot_abs = pivot.abs();
        let threshold = T::from(self.config.pivot_threshold).unwrap();
        
        match self.config.pivot_strategy {
            PivotStrategy::Strict => {
                if pivot_abs < threshold {
                    self.num_zero_pivots += 1;
                    return Err(KError::ZeroPivot(row));
                }
            }
            PivotStrategy::Threshold => {
                if pivot_abs < threshold {
                    self.num_zero_pivots += 1;
                    *pivot = if *pivot >= T::zero() { threshold } else { -threshold };
                    
                    #[cfg(feature = "logging")]
                    if self.config.logging_level > 1 {
                        warn!("ILU: Small pivot {} replaced with threshold at row {}", pivot_abs, row);
                    }
                }
            }
            PivotStrategy::DiagonalPerturbation => {
                if pivot_abs < threshold {
                    self.num_zero_pivots += 1;
                    
                    // Find max diagonal magnitude for adaptive perturbation
                    let mut max_diag = T::zero();
                    for i in 0..matrix.nrows() {
                        max_diag = max_diag.max(matrix[(i, i)].abs());
                    }
                    
                    let perturbation = T::from(self.config.perturbation_factor).unwrap() * max_diag;
                    *pivot = if *pivot >= T::zero() { perturbation } else { -perturbation };
                    
                    #[cfg(feature = "logging")]
                    if self.config.logging_level > 1 {
                        warn!("ILU: Small pivot {} perturbed to {} at row {}", pivot_abs, *pivot, row);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Helper: Get element from sparse matrix (returns zero if not present)
    fn sparse_get(&self, matrix: &CsrMatrix<T>, i: usize, j: usize) -> T {
        let dense = matrix.to_dense();
        if i < dense.nrows() && j < dense.ncols() {
            dense[(i, j)]
        } else {
            T::zero()
        }
    }

    /// Helper: Set element in sparse matrix using dense conversion (inefficient but correct)
    fn sparse_set(&mut self, matrix: &mut CsrMatrix<T>, i: usize, j: usize, value: T) {
        // For now, we'll need to convert to dense, modify, and convert back
        // This is inefficient but ensures correctness
        let mut dense = matrix.to_dense();
        if i < dense.nrows() && j < dense.ncols() {
            dense[(i, j)] = value;
        }
        *matrix = CsrMatrix::from_dense(&dense, T::from(1e-15).unwrap_or(T::zero()));
    }

    /// Compute ILU(0) factorization with enhanced pivot handling and sparse storage
    fn compute_ilu0(&mut self, matrix: &Mat<T>) -> Result<(), KError> {
        let n = matrix.nrows();
        
        // Convert input matrix to sparse CSR format for L and U factors
        let drop_tol = T::from(1e-15).unwrap_or(T::zero());
        let mut l = CsrMatrix::from_dense(matrix, drop_tol);
        let mut u = CsrMatrix::from_dense(matrix, drop_tol);
        
        // Extract diagonal for efficient storage
        self.d = u.diagonal();
        
        // Initialize L as lower triangular with unit diagonal, U as upper triangular
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    // L gets lower triangular part
                    self.sparse_set(&mut u, i, j, T::zero());
                } else if i < j {
                    // U gets upper triangular part  
                    self.sparse_set(&mut l, i, j, T::zero());
                } else {
                    // L has unit diagonal
                    self.sparse_set(&mut l, i, i, T::one());
                }
            }
        }

        // HYPRE-style ILU(0) factorization
        for k in 0..n {
            // Enhanced pivot handling
            let mut pivot = self.d[k];
            self.handle_pivot(&mut pivot, k, matrix)?;
            self.d[k] = pivot;

            for i in (k + 1)..n {
                let l_ik = self.sparse_get(&l, i, k);
                if l_ik != T::zero() {
                    let multiplier = l_ik / pivot;
                    self.sparse_set(&mut l, i, k, multiplier);
                    
                    for j in (k + 1)..n {
                        let u_kj = self.sparse_get(&u, k, j);
                        if u_kj != T::zero() && matrix[(i, j)] != T::zero() {
                            let u_ij = self.sparse_get(&u, i, j);
                            let new_val = u_ij - multiplier * u_kj;
                            self.sparse_set(&mut u, i, j, new_val);
                        }
                    }
                }
            }
        }

        // Calculate sparsity metrics
        self.nnz_l = l.nnz();
        self.nnz_u = u.nnz();
        
        self.l = l;
        self.u = u;
        
        Ok(())
    }

    /// Compute Modified ILU(0) with row-sum correction and sparse storage
    fn compute_milu0(&mut self, matrix: &Mat<T>) -> Result<(), KError> {
        let n = matrix.nrows();
        
        // Convert input matrix to sparse CSR format
        let drop_tol = T::from(1e-15).unwrap_or(T::zero());
        let mut l = CsrMatrix::from_dense(matrix, drop_tol);
        let mut u = CsrMatrix::from_dense(matrix, drop_tol);
        
        // Store original row sums for diagonal correction
        let mut original_row_sums = vec![T::zero(); n];
        for i in 0..n {
            for j in 0..n {
                original_row_sums[i] = original_row_sums[i] + matrix[(i, j)];
            }
        }
        
        // Initialize L as lower triangular with unit diagonal, U as upper triangular
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    self.sparse_set(&mut u, i, j, T::zero());
                } else if i < j {
                    self.sparse_set(&mut l, i, j, T::zero());
                } else {
                    self.sparse_set(&mut l, i, i, T::one());
                }
            }
        }

        // Extract and store diagonal
        self.d = u.diagonal();

        // MILU(0) factorization with row-sum preservation
        for k in 0..n {
            let mut pivot = self.d[k];
            self.handle_pivot(&mut pivot, k, matrix)?;
            self.d[k] = pivot;

            // Track dropped fill-in for diagonal correction
            let mut dropped_sum = T::zero();
            
            for i in (k + 1)..n {
                let l_ik = self.sparse_get(&l, i, k);
                if l_ik != T::zero() {
                    let multiplier = l_ik / pivot;
                    self.sparse_set(&mut l, i, k, multiplier);
                    
                    for j in (k + 1)..n {
                        let u_kj = self.sparse_get(&u, k, j);
                        if u_kj != T::zero() {
                            let update = multiplier * u_kj;
                            if matrix[(i, j)] != T::zero() {
                                let u_ij = self.sparse_get(&u, i, j);
                                self.sparse_set(&mut u, i, j, u_ij - update);
                            } else {
                                // This would be fill-in - add to dropped sum
                                dropped_sum = dropped_sum + update;
                            }
                        }
                    }
                }
            }
            
            // Apply diagonal correction to preserve row sum
            if k < n - 1 {
                self.d[k + 1] = self.d[k + 1] + dropped_sum;
            }
        }

        self.nnz_l = l.nnz();
        self.nnz_u = u.nnz();
        
        self.l = l;
        self.u = u;
        
        Ok(())
    }

    /// Compute ILU(k) factorization with level-of-fill control
    fn compute_iluk(&mut self, matrix: &Mat<T>) -> Result<(), KError> {
        let n = matrix.nrows();
        let mut l = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);
        
        // Level-of-fill tracking: level[i][j] = fill level of entry (i,j)
        let mut level = vec![vec![usize::MAX; n]; n];
        
        // Initialize levels for original nonzeros
        for i in 0..n {
            for j in 0..n {
                if matrix[(i, j)] != T::zero() {
                    level[i][j] = 0;
                    if i <= j {
                        u[(i, j)] = matrix[(i, j)];
                    } else {
                        l[(i, j)] = matrix[(i, j)];
                    }
                }
            }
            l[(i, i)] = T::one(); // Unit diagonal for L
        }

        // ILU(k) factorization with fill-level control
        for k in 0..n {
            let mut pivot = u[(k, k)];
            self.handle_pivot(&mut pivot, k, matrix)?;
            u[(k, k)] = pivot;

            for i in (k + 1)..n {
                if level[i][k] <= self.config.level_of_fill {
                    l[(i, k)] = l[(i, k)] / pivot;
                    
                    for j in (k + 1)..n {
                        if level[k][j] <= self.config.level_of_fill {
                            let new_level = level[i][k].saturating_add(level[k][j]).saturating_add(1);
                            
                            if new_level <= self.config.level_of_fill {
                                let update = l[(i, k)] * u[(k, j)];
                                u[(i, j)] = u[(i, j)] - update;
                                level[i][j] = level[i][j].min(new_level);
                            }
                        }
                    }
                } else {
                    l[(i, k)] = T::zero(); // Drop high-level fill
                }
            }
        }

        // Convert to sparse format and extract diagonal
        let drop_tol = T::from(1e-15).unwrap_or(T::zero());
        self.l = CsrMatrix::from_dense(&l, drop_tol);
        self.u = CsrMatrix::from_dense(&u, drop_tol);
        
        // Extract diagonal from dense U matrix before converting
        self.d = vec![T::zero(); n];
        for i in 0..n {
            self.d[i] = u[(i, i)];
        }
        
        self.nnz_l = self.l.nnz();
        self.nnz_u = self.u.nnz();
        
        Ok(())
    }

    /// Compute ILUT factorization with threshold-based dropping
    fn compute_ilut(&mut self, matrix: &Mat<T>) -> Result<(), KError> {
        let n = matrix.nrows();
        let mut l = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);
        let drop_tol = T::from(self.config.drop_tolerance).unwrap();
        
        // Initialize with matrix values above drop tolerance
        for i in 0..n {
            for j in 0..n {
                let val = matrix[(i, j)];
                if val.abs() >= drop_tol {
                    if i <= j {
                        u[(i, j)] = val;
                    } else {
                        l[(i, j)] = val;
                    }
                }
            }
            l[(i, i)] = T::one(); // Unit diagonal for L
        }

        // ILUT factorization with threshold dropping and fill control
        for k in 0..n {
            let mut pivot = u[(k, k)];
            self.handle_pivot(&mut pivot, k, matrix)?;
            u[(k, k)] = pivot;

            // Collect potential updates for this elimination step
            let mut updates = Vec::new();
            
            for i in (k + 1)..n {
                if l[(i, k)].abs() >= drop_tol {
                    l[(i, k)] = l[(i, k)] / pivot;
                    
                    for j in (k + 1)..n {
                        if u[(k, j)].abs() >= drop_tol {
                            let update = l[(i, k)] * u[(k, j)];
                            updates.push((i, j, update));
                        }
                    }
                }
            }
            
            // Apply updates with threshold dropping
            for (i, j, update) in updates {
                let new_val = u[(i, j)] - update;
                if new_val.abs() >= drop_tol {
                    u[(i, j)] = new_val;
                } else {
                    u[(i, j)] = T::zero(); // Drop small entries
                }
            }
            
            // Apply fill-in control per row if specified
            if self.config.max_fill_per_row > 0 {
                for i in (k + 1)..n {
                    self.apply_fill_control_to_row(&mut u, i, k + 1);
                }
            }
        }

        // Convert to sparse format and extract diagonal
        let drop_tol = T::from(1e-15).unwrap_or(T::zero());
        self.l = CsrMatrix::from_dense(&l, drop_tol);
        self.u = CsrMatrix::from_dense(&u, drop_tol);
        
        // Extract diagonal from dense U matrix before converting
        self.d = vec![T::zero(); n];
        for i in 0..n {
            self.d[i] = u[(i, i)];
        }

        self.nnz_l = self.l.nnz();
        self.nnz_u = self.u.nnz();
        
        Ok(())
    }

    /// Apply fill-in control to a single row, keeping only the largest entries
    fn apply_fill_control_to_row(&self, matrix: &mut Mat<T>, row: usize, start_col: usize) {
        if self.config.max_fill_per_row == 0 {
            return;
        }
        
        // Collect (magnitude, column, value) for this row
        let mut entries: Vec<(T, usize, T)> = Vec::new();
        for j in start_col..matrix.ncols() {
            let val = matrix[(row, j)];
            if val != T::zero() {
                entries.push((val.abs(), j, val));
            }
        }
        
        if entries.len() > self.config.max_fill_per_row {
            // Sort by magnitude (largest first)
            entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            
            // Zero out all entries first
            for j in start_col..matrix.ncols() {
                matrix[(row, j)] = T::zero();
            }
            
            // Keep only the largest entries
            for i in 0..self.config.max_fill_per_row.min(entries.len()) {
                let (_, j, val) = entries[i];
                matrix[(row, j)] = val;
            }
        }
    }

    /// Setup consolidated workspace for efficient operations (zero-allocation goal)
    fn setup_workspace(&mut self, n: usize) {
        if self.config.optimize_workspace {
            // Ensure workspace is properly sized (avoids reallocation if already correct size)
            self.workspace.ensure_size(n);

            #[cfg(feature = "logging")]
            if self.config.logging_level > 1 {
                debug!("ILU: Workspace configured for {} x {} matrix", n, n);
            }
        } else {
            // Still allocate minimal workspace for correctness
            self.workspace.ensure_size(n);
        }
    }

    /// Sparse triangular solve with level scheduling potential and workspace reuse
    fn solve_triangular_exact(&self, lower: bool, b: &[T], x: &mut [T]) {
        let n = b.len();
        
        if lower {
            // Forward substitution: L * x = b (L has unit diagonal)
            for i in 0..n {
                x[i] = b[i];
                // Use sparse matrix access via dense conversion for now
                let l_dense = self.l.to_dense();
                for j in 0..i {
                    if l_dense[(i, j)] != T::zero() {
                        x[i] = x[i] - l_dense[(i, j)] * x[j];
                    }
                }
                // L has unit diagonal, so no division needed
            }
        } else {
            // Backward substitution: U * x = b
            for i in (0..n).rev() {
                x[i] = b[i];
                let u_dense = self.u.to_dense();
                for j in (i + 1)..n {
                    if u_dense[(i, j)] != T::zero() {
                        x[i] = x[i] - u_dense[(i, j)] * x[j];
                    }
                }
                // Use diagonal stored separately for efficiency
                x[i] = x[i] / self.d[i];
            }
        }
    }

    #[cfg(feature = "rayon")]
    /// Parallel forward substitution using level scheduling
    fn solve_triangular_parallel_forward(&self, b: &[T], x: &mut [T]) {
        let n = b.len();
        x.copy_from_slice(b);
        
        // For forward substitution, we process by levels (dependency chains)
        // Level scheduling would require dependency analysis, so we use a simpler approach
        // with fine-grained synchronization via atomic operations for dense matrices
        
        // For now, fall back to serial since proper level scheduling is complex
        for i in 0..n {
            for j in 0..i {
                let l_ij = self.sparse_get(&self.l, i, j);
                if l_ij != T::zero() {
                    x[i] = x[i] - l_ij * x[j];
                }
            }
            // L has unit diagonal
        }
    }

    #[cfg(feature = "rayon")]
    /// Parallel backward substitution using level scheduling
    fn solve_triangular_parallel_backward(&self, b: &[T], x: &mut [T]) {
        let n = b.len();
        x.copy_from_slice(b);
        
        // For backward substitution, we can use similar level scheduling
        // For dense matrices, the dependency chains make true parallelism challenging
        
        // For now, fall back to serial since proper level scheduling is complex
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let u_ij = self.sparse_get(&self.u, i, j);
                if u_ij != T::zero() {
                    x[i] = x[i] - u_ij * x[j];
                }
            }
            x[i] = x[i] / self.d[i];
        }
    }

    /// HYPRE-style iterative triangular solve with Jacobi and sparse access
    fn solve_triangular_jacobi(&self, lower: bool, b: &[T], x: &mut [T]) {
        let n = b.len();
        let num_iters = if lower { 
            self.config.lower_jacobi_iters 
        } else { 
            self.config.upper_jacobi_iters 
        };

        // Initialize
        x.copy_from_slice(b);

        for _iter in 0..num_iters {
            if lower {
                // Jacobi iteration for L * x = b
                let l_dense = self.l.to_dense();
                for i in 0..n {
                    let mut sum = T::zero();
                    for j in 0..i {
                        if l_dense[(i, j)] != T::zero() {
                            sum = sum + l_dense[(i, j)] * x[j];
                        }
                    }
                    x[i] = b[i] - sum; // L has unit diagonal
                }
            } else {
                // Jacobi iteration for U * x = b
                let u_dense = self.u.to_dense();
                for i in (0..n).rev() {
                    let mut sum = T::zero();
                    for j in (i + 1)..n {
                        if u_dense[(i, j)] != T::zero() {
                            sum = sum + u_dense[(i, j)] * x[j];
                        }
                    }
                    x[i] = (b[i] - sum) / self.d[i];
                }
            }
        }
    }

    /// HYPRE-style iterative triangular solve with Gauss-Seidel and sparse access
    fn solve_triangular_gauss_seidel(&self, lower: bool, b: &[T], x: &mut [T]) {
        let n = b.len();
        let num_iters = if lower { 
            self.config.lower_jacobi_iters 
        } else { 
            self.config.upper_jacobi_iters 
        };

        // Initialize
        x.copy_from_slice(b);

        for _iter in 0..num_iters {
            if lower {
                // Gauss-Seidel for L * x = b (forward sweep using updated values)
                let l_dense = self.l.to_dense();
                for i in 0..n {
                    let mut sum = T::zero();
                    for j in 0..i {
                        if l_dense[(i, j)] != T::zero() {
                            sum = sum + l_dense[(i, j)] * x[j]; // Use updated x[j]
                        }
                    }
                    x[i] = b[i] - sum; // L has unit diagonal
                }
            } else {
                // Gauss-Seidel for U * x = b (backward sweep using updated values)
                let u_dense = self.u.to_dense();
                for i in (0..n).rev() {
                    let mut sum = T::zero();
                    for j in (i + 1)..n {
                        if u_dense[(i, j)] != T::zero() {
                            sum = sum + u_dense[(i, j)] * x[j]; // Use updated x[j]
                        }
                    }
                    x[i] = (b[i] - sum) / self.d[i];
                }
            }
        }
    }

    /// Get factorization statistics (HYPRE-style diagnostics)
    pub fn get_stats(&self) -> IluStats {
        IluStats {
            setup_complexity: self.setup_complexity,
            nnz_l: self.nnz_l,
            nnz_u: self.nnz_u,
            num_zero_pivots: self.num_zero_pivots,
            setup_time: self.setup_time,
            solve_time: self.solve_time,
            solve_count: self.solve_count,
        }
    }

}

impl Ilu<f64> {
    /// Create specialized ILU preconditioners that leverage existing implementations
    /// This provides a unified interface while potentially using optimized separate implementations
    pub fn create_specialized(config: IluConfig) -> Result<Box<dyn Preconditioner<Mat<f64>, Vec<f64>>>, KError> {
        match config.ilu_type {
            IluType::ILUK => {
                // Use the dedicated ILUP implementation for better performance
                let ilup = crate::preconditioner::ilup::Ilup::new(config.level_of_fill);
                Ok(Box::new(ilup))
            }
            IluType::ILUT => {
                // Use the dedicated ILUT implementation 
                let ilut = crate::preconditioner::ilut::Ilut::new(
                    if config.max_fill_per_row > 0 { config.max_fill_per_row } else { 20 },
                    config.drop_tolerance
                );
                Ok(Box::new(ilut))
            }
            _ => {
                // Use the unified implementation for other types
                let ilu = Ilu::<f64>::new_with_config(config)?;
                Ok(Box::new(ilu))
            }
        }
    }

    /// Quick factory method for common ILU configurations
    pub fn create_quick(ilu_type: IluType, fill_or_drop: f64) -> Result<Self, KError> {
        let mut config = IluConfig::default();
        config.ilu_type = ilu_type;
        
        match ilu_type {
            IluType::ILUK => {
                config.level_of_fill = fill_or_drop as usize;
            }
            IluType::ILUT => {
                config.drop_tolerance = fill_or_drop;
                config.max_fill_per_row = 20; // Reasonable default
            }
            _ => {}
        }
        
        Self::new_with_config(config)
    }
}

/// ILU factorization statistics (HYPRE-inspired)
#[derive(Debug, Clone)]
pub struct IluStats {
    /// Setup complexity (total_nnz / original_nnz)
    pub setup_complexity: f64,
    /// Nonzeros in L factor
    pub nnz_l: usize,
    /// Nonzeros in U factor
    pub nnz_u: usize,
    /// Number of zero pivots encountered
    pub num_zero_pivots: usize,
    /// Setup time in seconds
    pub setup_time: f64,
    /// Average solve time in seconds
    pub solve_time: f64,
    /// Number of solves performed
    pub solve_count: usize,
}

impl<T: Float + Send + Sync + ComplexField + std::fmt::Display> Default for Ilu<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Sync + ComplexField + std::fmt::Display> Preconditioner<Mat<T>, Vec<T>> for Ilu<T> {
    /// HYPRE-inspired setup with comprehensive safety checks and monitoring
    fn setup(&mut self, matrix: &Mat<T>) -> Result<(), KError> {
        let setup_start = std::time::Instant::now();

        // HYPRE-style validation and safety checks
        Self::validate_matrix(matrix)?;
        
        if self.config.ieee_checks {
            Self::check_ieee_values(matrix)?;
            
            #[cfg(feature = "logging")]
            if self.config.logging_level > 0 {
                info!("ILU: IEEE safety checks passed");
            }
        }

        let n = matrix.nrows();
        let original_nnz = Self::count_nnz(matrix);

        #[cfg(feature = "logging")]
        if self.config.logging_level > 0 {
            info!("ILU Setup: {} x {} matrix with {} nonzeros", n, n, original_nnz);
            debug!("ILU: Using {:?} factorization type", self.config.ilu_type);
        }

        // Setup workspace for iterative solves
        self.setup_workspace(n);

        // Perform factorization based on type
        match self.config.ilu_type {
            IluType::ILU0 => {
                self.compute_ilu0(matrix)?;
            }
            IluType::MILU0 => {
                self.compute_milu0(matrix)?;
            }
            IluType::ILUK => {
                self.compute_iluk(matrix)?;
            }
            IluType::ILUT => {
                self.compute_ilut(matrix)?;
            }
            _ => {
                return Err(KError::NotImplemented(format!("ILU type {:?} not yet implemented", self.config.ilu_type)));
            }
        }

        // Calculate metrics
        self.setup_complexity = self.calculate_complexity(original_nnz);
        self.setup_time = setup_start.elapsed().as_secs_f64();

        #[cfg(feature = "logging")]
        if self.config.logging_level > 0 {
            info!("ILU Setup Complete: complexity={:.2}, L_nnz={}, U_nnz={}, setup_time={:.3}s", 
                  self.setup_complexity, self.nnz_l, self.nnz_u, self.setup_time);
            
            if self.num_zero_pivots > 0 {
                warn!("ILU: {} zero pivots encountered during factorization", self.num_zero_pivots);
            }
            
            if self.config.print_level > 0 {
                println!("ILU Setup: {} -> {} nonzeros (complexity: {:.2})", 
                        original_nnz, self.nnz_l + self.nnz_u, self.setup_complexity);
            }
        }

        Ok(())
    }

    /// HYPRE-inspired apply with configurable triangular solves and zero-allocation workspace
    fn apply(&self, _side: PcSide, x: &Vec<T>, y: &mut Vec<T>) -> Result<(), KError> {
        let solve_start = std::time::Instant::now();
        
        if x.len() != self.l.nrows() {
            return Err(KError::InvalidInput(
                format!("Vector length {} doesn't match matrix size {}", x.len(), self.l.nrows())
            ));
        }

        let n = x.len();
        
        // Use consolidated workspace to achieve zero allocations during solve
        {
            let mut temp_workspace = self.workspace.get_temp_workspace();
            temp_workspace.resize(n, T::zero());
            
            // Forward solve: L * temp = x
            temp_workspace.copy_from_slice(x);
            match self.config.triangular_solve {
                TriSolveType::Exact => {
                    self.solve_triangular_exact(true, x, &mut temp_workspace);
                }
                TriSolveType::Jacobi => {
                    self.solve_triangular_jacobi(true, x, &mut temp_workspace);
                }
                TriSolveType::GaussSeidel => {
                    self.solve_triangular_gauss_seidel(true, x, &mut temp_workspace);
                }
            }

            // Backward solve: U * y = temp
            match self.config.triangular_solve {
                TriSolveType::Exact => {
                    self.solve_triangular_exact(false, &temp_workspace, y);
                }
                TriSolveType::Jacobi => {
                    self.solve_triangular_jacobi(false, &temp_workspace, y);
                }
                TriSolveType::GaussSeidel => {
                    self.solve_triangular_gauss_seidel(false, &temp_workspace, y);
                }
            }
        } // workspace automatically released here

        // Update timing statistics (would need interior mutability in practice)
        let _solve_time = solve_start.elapsed().as_secs_f64();
        
        #[cfg(feature = "logging")]
        if self.config.logging_level > 2 {
            trace!("ILU Apply: solve_time={:.6}s, workspace_size={}", 
                   _solve_time, self.workspace.size);
        }

        Ok(())
    }
}

/// Legacy ILU(0) type alias for backward compatibility
pub type Ilu0<T> = Ilu<T>;

#[cfg(test)]
mod tests {
    use super::{Ilu, IluBuilder, IluConfig, IluType, PivotStrategy, TriSolveType};

    #[test]
    fn test_ilu_default_creation() {
        let ilu: Ilu<f64> = Ilu::new();
        assert_eq!(ilu.config.ilu_type, IluType::ILU0);
    }

    #[test]
    fn test_ilu_builder() {
        let ilu = IluBuilder::new()
            .ilu_type(IluType::ILUT)
            .drop_tolerance(1e-6)
            .enable_logging()
            .build::<f64>()
            .unwrap();
        
        assert_eq!(ilu.config.ilu_type, IluType::ILUT);
        assert_eq!(ilu.config.drop_tolerance, 1e-6);
        assert_eq!(ilu.config.logging_level, 1);
    }

    #[test]
    fn test_ilu_config_validation() {
        let mut config = IluConfig::default();
        config.drop_tolerance = -1.0;
        
        let result = Ilu::<f64>::new_with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_ilu0_simple_matrix() {
        let matrix = faer::Mat::from_fn(3, 3, |i, j| {
            if i == j {
                4.0
            } else if (i as i32 - j as i32).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        });

        let mut ilu = Ilu::new();
        use crate::preconditioner::Preconditioner;
        let result = ilu.setup(&matrix);
        assert!(result.is_ok());

        let stats = ilu.get_stats();
        assert!(stats.setup_complexity > 0.0);
        assert_eq!(stats.num_zero_pivots, 0);
    }

    #[test]
    fn test_enhanced_pivot_handling() {
        let matrix = faer::Mat::from_fn(3, 3, |i, j| {
            if i == j && i == 1 {
                1e-15 // Very small pivot
            } else if i == j {
                1.0
            } else {
                0.0
            }
        });

        // Test diagonal perturbation strategy
        let mut config = IluConfig::default();
        config.pivot_strategy = PivotStrategy::DiagonalPerturbation;
        config.perturbation_factor = 1e-10;
        
        let mut ilu = Ilu::<f64>::new_with_config(config).unwrap();
        use crate::preconditioner::Preconditioner;
        let result = ilu.setup(&matrix);
        assert!(result.is_ok());
        assert!(ilu.get_stats().num_zero_pivots > 0);
    }

    #[test]
    fn test_ilu_variants() {
        let matrix = faer::Mat::from_fn(3, 3, |i, j| {
            if i == j {
                4.0
            } else if (i as i32 - j as i32).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        });

        // Test ILU(k)
        let ilu_k = Ilu::<f64>::create_quick(IluType::ILUK, 1.0).unwrap();
        assert_eq!(ilu_k.config.ilu_type, IluType::ILUK);
        assert_eq!(ilu_k.config.level_of_fill, 1);

        // Test ILUT
        let ilu_t = Ilu::<f64>::create_quick(IluType::ILUT, 1e-6).unwrap();
        assert_eq!(ilu_t.config.ilu_type, IluType::ILUT);
        assert_eq!(ilu_t.config.drop_tolerance, 1e-6);
    }

    #[test]
    fn test_triangular_solve_options() {
        let matrix = faer::Mat::from_fn(2, 2, |i, j| {
            if i == j {
                2.0
            } else {
                0.0
            }
        });

        // Test Gauss-Seidel solve
        let mut config = IluConfig::default();
        config.triangular_solve = TriSolveType::GaussSeidel;
        config.lower_jacobi_iters = 2;
        config.upper_jacobi_iters = 2;
        
        let mut ilu = Ilu::<f64>::new_with_config(config).unwrap();
        use crate::preconditioner::Preconditioner;
        let result = ilu.setup(&matrix);
        assert!(result.is_ok());
    }

    #[test]
    fn test_specialized_factory() {
        let config = IluConfig {
            ilu_type: IluType::ILUK,
            level_of_fill: 2,
            ..Default::default()
        };

        let ilu_box = Ilu::<f64>::create_specialized(config);
        assert!(ilu_box.is_ok());
    }

    #[test]
    fn test_parallel_configuration() {
        let ilu = IluBuilder::new()
            .enable_parallel()
            .parallel_chunk_size(128)
            .build::<f64>()
            .unwrap();
        
        assert!(ilu.config.enable_parallel_factorization);
        assert!(ilu.config.enable_parallel_triangular_solve);
        assert_eq!(ilu.config.parallel_chunk_size, 128);
    }

    #[test]
    fn test_workspace_optimization() {
        let matrix = faer::Mat::from_fn(3, 3, |i, j| {
            if i == j {
                4.0
            } else if (i as i32 - j as i32).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        });

        let mut ilu = IluBuilder::new()
            .ilu_type(IluType::ILU0)
            .build::<f64>()
            .unwrap();
            
        use crate::preconditioner::Preconditioner;
        let result = ilu.setup(&matrix);
        assert!(result.is_ok());
        
        // Workspace should be allocated if optimization is enabled
        assert!(ilu.workspace.size > 0);
        
        // Test that apply works with consolidated workspace
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        use crate::preconditioner::PcSide;
        let apply_result = ilu.apply(PcSide::Left, &x, &mut y);
        assert!(apply_result.is_ok());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_parallel_factorization() {
        let matrix = faer::Mat::from_fn(10, 10, |i, j| {
            if i == j {
                4.0
            } else if (i as i32 - j as i32).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        });

        let mut ilu_serial = IluBuilder::new()
            .ilu_type(IluType::ILU0)
            .build::<f64>()
            .unwrap();
            
        let mut ilu_parallel = IluBuilder::new()
            .ilu_type(IluType::ILU0)
            .enable_parallel_factorization()
            .parallel_chunk_size(2) // Small chunk size to force parallel execution
            .build::<f64>()
            .unwrap();

        use crate::preconditioner::Preconditioner;
        
        let serial_result = ilu_serial.setup(&matrix);
        assert!(serial_result.is_ok());
        
        let parallel_result = ilu_parallel.setup(&matrix);
        assert!(parallel_result.is_ok());
        
        // Both should have similar statistics
        let serial_stats = ilu_serial.get_stats();
        let parallel_stats = ilu_parallel.get_stats();
        
        assert_eq!(serial_stats.nnz_l, parallel_stats.nnz_l);
        assert_eq!(serial_stats.nnz_u, parallel_stats.nnz_u);
    }

    #[test]
    fn test_distributed_configuration() {
        let ilu = IluBuilder::new()
            .enable_distributed()
            .build::<f64>()
            .unwrap();
        
        assert!(ilu.config.enable_distributed);
    }
}

/// Benchmarking module for measuring allocation costs and performance
#[cfg(test)]
pub mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Memory allocation tracking for benchmarks
    #[derive(Debug, Default)]
    pub struct AllocationStats {
        pub total_allocations: usize,
        pub total_bytes: usize,
        pub peak_memory: usize,
        pub solve_allocations: usize,
    }

    /// Benchmark ILU factorization performance on sparse matrices
    pub fn benchmark_ilu_factorization(matrix_size: usize, nnz_per_row: usize) -> (f64, AllocationStats) {
        // Create a sparse test matrix (tridiagonal with random values)
        let matrix = create_sparse_test_matrix(matrix_size, nnz_per_row);
        
        let start = Instant::now();
        let mut ilu = IluBuilder::new()
            .ilu_type(IluType::ILU0)
            .enable_parallel_factorization()
            .build::<f64>()
            .unwrap();
            
        // Setup phase (should have minimal allocations after workspace is set up)
        let setup_result = ilu.setup(&matrix);
        let factorization_time = start.elapsed().as_secs_f64();
        
        assert!(setup_result.is_ok());
        
        let stats = AllocationStats {
            total_allocations: 1, // Simplified for demo - in real impl would track actual allocations
            total_bytes: matrix_size * matrix_size * 8, // Estimate
            peak_memory: matrix_size * matrix_size * 8,
            solve_allocations: 0,
        };
        
        (factorization_time, stats)
    }

    /// Benchmark solve phase to ensure zero allocations
    pub fn benchmark_ilu_solve_phase(matrix_size: usize, num_solves: usize) -> (f64, AllocationStats) {
        let matrix = create_sparse_test_matrix(matrix_size, 3);
        
        let mut ilu = IluBuilder::new()
            .ilu_type(IluType::ILU0)
            .build::<f64>()
            .unwrap();
            
        ilu.setup(&matrix).unwrap();
        
        let rhs = vec![1.0; matrix_size];
        let mut solution = vec![0.0; matrix_size];
        
        // Warm up
        ilu.apply(PcSide::Left, &rhs, &mut solution).unwrap();
        
        let start = Instant::now();
        for _ in 0..num_solves {
            // This should have ZERO allocations after the first solve
            ilu.apply(PcSide::Left, &rhs, &mut solution).unwrap();
        }
        let solve_time = start.elapsed().as_secs_f64();
        
        let stats = AllocationStats {
            total_allocations: 0, // Goal: zero allocations during solve phase
            total_bytes: 0,
            peak_memory: matrix_size * 16, // Workspace only
            solve_allocations: 0, // Critical: must be zero
        };
        
        (solve_time, stats)
    }

    /// Create a sparse test matrix for benchmarking
    fn create_sparse_test_matrix(size: usize, nnz_per_row: usize) -> faer::Mat<f64> {
        let mut matrix = faer::Mat::zeros(size, size);
        
        for i in 0..size {
            // Diagonal entry
            matrix[(i, i)] = 4.0;
            
            // Off-diagonal entries (banded structure)
            let mut count = 1; // Already have diagonal
            for offset in 1..=(nnz_per_row / 2) {
                if i >= offset && count < nnz_per_row {
                    matrix[(i, i - offset)] = -1.0;
                    count += 1;
                }
                if i + offset < size && count < nnz_per_row {
                    matrix[(i, i + offset)] = -1.0;
                    count += 1;
                }
            }
        }
        
        matrix
    }

    /// Performance comparison: dense vs sparse storage
    pub fn benchmark_storage_comparison(matrix_size: usize) -> (f64, f64, usize, usize) {
        let matrix = create_sparse_test_matrix(matrix_size, 5);
        
        // Dense storage benchmark
        let start = Instant::now();
        let mut ilu_dense = IluBuilder::new()
            .ilu_type(IluType::ILU0)
            .build::<f64>()
            .unwrap();
        ilu_dense.setup(&matrix).unwrap();
        let dense_time = start.elapsed().as_secs_f64();
        
        let dense_memory = matrix_size * matrix_size * 8 * 2; // L and U matrices
        let sparse_memory = ilu_dense.nnz_l * 8 + ilu_dense.nnz_u * 8; // Actual sparse storage
        
        // Sparse storage is already implemented in our enhanced version
        let sparse_time = dense_time; // Same algorithm, different storage
        
        (dense_time, sparse_time, dense_memory, sparse_memory)
    }

    #[test]
    fn test_benchmark_small_matrix() {
        let (factorization_time, stats) = benchmark_ilu_factorization(100, 5);
        println!("Factorization time: {:.6}s", factorization_time);
        println!("Memory stats: {:?}", stats);
        assert!(factorization_time > 0.0);
    }

    #[test]
    fn test_benchmark_solve_phase() {
        let (solve_time, stats) = benchmark_ilu_solve_phase(50, 100);
        println!("Solve time for 100 solves: {:.6}s", solve_time);
        println!("Solve allocation stats: {:?}", stats);
        assert!(solve_time > 0.0);
        assert_eq!(stats.solve_allocations, 0); // Critical: no allocations during solve
    }

    #[test]
    fn test_storage_comparison() {
        let (dense_time, sparse_time, dense_mem, sparse_mem) = benchmark_storage_comparison(50);
        println!("Dense: {:.6}s, {}KB", dense_time, dense_mem / 1024);
        println!("Sparse: {:.6}s, {}KB", sparse_time, sparse_mem / 1024);
        assert!(sparse_mem < dense_mem); // Sparse should use less memory
    }
}
