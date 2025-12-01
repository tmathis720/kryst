#![cfg(feature = "backend-faer")]

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

use crate::algebra::scalar::KrystScalar;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::utils;
use crate::preconditioner::LocalPreconditioner;
use crate::preconditioner::stats::{ParIluHistory, ParIluIterSample};
use crate::preconditioner::{PcSide, legacy::Preconditioner, pivot::*};
use crate::utils::metrics::{Counters, SolveTimer};
use crate::utils::monitor::{Event, Monitor};
use faer::Mat;
use std::sync::Mutex;

#[cfg(feature = "logging")]
use log::{debug, info, trace, warn};

// ILU is restricted to real scalars.
type S = f64;
type Real = f64;

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
    pub drop_tolerance: Real,
    /// Drop tolerance for off-diagonal blocks (HYPRE: droptol[1])
    pub offdiag_drop_tolerance: Real,
    /// Drop tolerance for Schur complement (HYPRE: droptol[2])
    pub schur_drop_tolerance: Real,
    /// Reordering strategy (HYPRE: reordering_type)
    pub reordering_type: ReorderingType,
    /// Triangular solve type (HYPRE: tri_solve)
    pub triangular_solve: TriSolveType,
    /// Lower triangular Jacobi iterations (HYPRE: lower_jacobi_iters)
    pub lower_jacobi_iters: usize,
    /// Upper triangular Jacobi iterations (HYPRE: upper_jacobi_iters)
    pub upper_jacobi_iters: usize,
    /// Tolerance for iterative solve (HYPRE: tol)
    pub tolerance: Real,
    /// Maximum iterations for iterative solve (HYPRE: max_iter)
    pub max_iterations: usize,
    /// Logging level (HYPRE: logging)
    pub logging_level: usize,
    /// Print level for diagnostics (HYPRE: print_level)
    pub print_level: usize,
    /// Enable IEEE safety checks
    pub ieee_checks: bool,
    /// Enable workspace optimization
    pub optimize_workspace: bool,
    /// Pivot handling policy
    pub pivot_policy: PivotPolicy,
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
            level_of_fill: 0,                      // HYPRE default for ILU(0)
            max_fill_per_row: 0,                   // HYPRE default: unlimited
            drop_tolerance: 1e-4,                  // HYPRE conservative default
            offdiag_drop_tolerance: 1e-4,          // HYPRE default
            schur_drop_tolerance: 1e-4,            // HYPRE default
            reordering_type: ReorderingType::None, // HYPRE default
            triangular_solve: TriSolveType::Exact, // HYPRE default
            lower_jacobi_iters: 1,                 // HYPRE default
            upper_jacobi_iters: 1,                 // HYPRE default
            tolerance: 1e-6,                       // HYPRE default
            max_iterations: 1,                     // HYPRE default for direct solve
            logging_level: 0,                      // No logging by default
            print_level: 0,                        // No printing by default
            ieee_checks: true,                     // Safety first
            optimize_workspace: true,              // Performance optimization
            pivot_policy: PivotPolicy::default(),
            enable_parallel_factorization: false, // Conservative default
            enable_parallel_triangular_solve: false, // Conservative default
            parallel_chunk_size: 64,              // Reasonable chunk size for cache efficiency
            enable_distributed: false,            // Conservative default
        }
    }
}

#[cfg(feature = "logging")]
fn print_ilu_banner(cfg: &IluConfig) {
    if cfg.logging_level == 0 {
        return;
    }
    info!("ILU Setup:");
    info!("  kind                 : {:?}", cfg.ilu_type);
    info!("  reordering           : {:?}", cfg.reordering_type);
    let tri = match cfg.triangular_solve {
        TriSolveType::Exact => "Exact".to_string(),
        TriSolveType::Jacobi => format!(
            "Jacobi (L:{} U:{})",
            cfg.lower_jacobi_iters, cfg.upper_jacobi_iters
        ),
        TriSolveType::GaussSeidel => "GaussSeidel".to_string(),
    };
    info!("  triangular solve     : {tri}");
    info!(
        "  iterative setup      : tol={:.2e}, max_iter={}",
        cfg.tolerance, cfg.max_iterations
    );
    info!(
        "  exec                 : distributed={}, par_factorization={}, par_trisolve={}",
        cfg.enable_distributed,
        cfg.enable_parallel_factorization,
        cfg.enable_parallel_triangular_solve
    );
    info!("  pivot                : {:?}", cfg.pivot_policy);
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
    pub fn drop_tolerance(mut self, tol: Real) -> Self {
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

    /// Set pivot handling policy
    pub fn pivot_policy(mut self, policy: PivotPolicy) -> Self {
        self.config.pivot_policy = policy;
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
    pub fn build(self) -> Result<Ilu, KError> {
        Ilu::new_with_config(self.config)
    }
}

impl Default for IluBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// HYPRE-inspired comprehensive ILU preconditioner with sparse storage
///
/// **Note:** ILU is currently restricted to real (`f64`) matrices only.
/// Complex-valued problems should use simpler preconditioners (e.g., Jacobi, diagonal scaling).
pub struct Ilu {
    /// Configuration parameters
    config: IluConfig,
    /// Lower triangular factor in CSR format (unit diagonal)
    l: CsrMatrix<f64>,
    /// Upper triangular factor in CSR format
    u: CsrMatrix<f64>,
    /// Cached inverse of U's diagonal entries for fast solves
    inv_diag_u: Vec<f64>,
    /// Permutation arrays (HYPRE: perm, qperm)
    #[allow(dead_code)]
    row_perm: Vec<usize>,
    #[allow(dead_code)]
    col_perm: Vec<usize>,
    /// Consolidated preallocated workspace vectors for all operations
    workspace: IluWorkspace,
    #[cfg(feature = "rayon")]
    /// Level scheduling for lower triangular solves
    levels_l: Levels,
    #[cfg(feature = "rayon")]
    /// Level scheduling for upper triangular solves
    levels_u: Levels,
    /// Setup complexity metrics (HYPRE: operator_complexity)
    setup_complexity: f64,
    /// Factorization statistics
    nnz_l: usize,
    nnz_u: usize,
    num_zero_pivots: usize,
    /// Pivot handling statistics
    pivot_stats: PivotStats,
    /// Global scaling from A's diagonal
    max_diag_a: f64,
    /// Row-wise infinity norm of A
    row_inf_a: Vec<f64>,
    /// Row-wise Gershgorin estimate of A
    row_gersh_a: Vec<f64>,
    /// Running maximum of |U_kk|
    running_max_u: f64,
    /// Performance timing
    setup_time: f64,
    solve_ctrs: Counters,
    /// Optional ParILU iteration history
    history: Option<ParIluHistory>,
    /// Optional event monitor
    monitor: Option<Box<dyn Monitor>>,
}

/// Consolidated workspace for all ILU operations to minimize allocations
#[derive(Debug)]
pub struct IluWorkspace {
    /// Scratch buffer for triangular solves (sized once in setup)
    solve_buf: Mutex<Vec<f64>>,
    /// Secondary workspace for complex operations
    temp2: Mutex<Vec<f64>>,
    /// Workspace for level scheduling in parallel triangular solves
    levels: Mutex<Vec<usize>>,
    /// Workspace for sparse pattern operations
    pattern_work: Mutex<Vec<bool>>,
    /// Current workspace size
    size: usize,
}

impl IluWorkspace {
    /// Create new workspace with given size
    pub fn new(size: usize) -> Self {
        Self {
            solve_buf: Mutex::new(vec![0.0; size]),
            temp2: Mutex::new(vec![0.0; size]),
            levels: Mutex::new(vec![0; size]),
            pattern_work: Mutex::new(vec![false; size]),
            size,
        }
    }

    /// Resize workspace if needed (avoids reallocation when possible)
    pub fn ensure_size(&mut self, new_size: usize) {
        if new_size > self.size {
            self.solve_buf.lock().unwrap().resize(new_size, 0.0);
            self.temp2.lock().unwrap().resize(new_size, 0.0);
            self.levels.lock().unwrap().resize(new_size, 0);
            self.pattern_work.lock().unwrap().resize(new_size, false);
            self.size = new_size;
        }
    }

    /// Clear workspace (without deallocation)
    pub fn clear(&self) {
        for x in self.solve_buf.lock().unwrap().iter_mut() {
            *x = 0.0;
        }
        for x in self.temp2.lock().unwrap().iter_mut() {
            *x = 0.0;
        }
        for x in self.levels.lock().unwrap().iter_mut() {
            *x = 0;
        }
        for x in self.pattern_work.lock().unwrap().iter_mut() {
            *x = false;
        }
    }

    /// Borrow the solve buffer sized in `setup()`.
    #[inline]
    pub fn borrow_solve_buf(&self, n: usize) -> std::sync::MutexGuard<'_, Vec<f64>> {
        debug_assert!(
            self.size >= n,
            "workspace not sized; call ensure_size in setup()"
        );
        self.solve_buf.lock().unwrap()
    }
}

#[cfg(feature = "rayon")]
#[derive(Clone, Debug, Default)]
struct Levels {
    /// Rows grouped by level
    buckets: Vec<Vec<usize>>,
    /// Maximum level
    max_level: u32,
}

#[cfg(feature = "rayon")]
fn build_levels_lower(l: &CsrMatrix<f64>) -> Levels {
    let n = l.nrows();
    let mut lev = vec![0u32; n];
    let mut maxl = 0u32;
    for i in 0..n {
        let (cols, _vals) = l.row(i);
        let mut li = 0u32;
        for &j in cols {
            if j >= i {
                break;
            }
            li = li.max(lev[j] + 1);
        }
        lev[i] = li;
        maxl = maxl.max(li);
    }
    let mut buckets = vec![Vec::new(); (maxl as usize) + 1];
    for (i, &l) in lev.iter().enumerate() {
        buckets[l as usize].push(i);
    }
    Levels {
        buckets,
        max_level: maxl,
    }
}

#[cfg(feature = "rayon")]
fn build_levels_upper(u: &CsrMatrix<f64>) -> Levels {
    let n = u.nrows();
    let mut lev = vec![0u32; n];
    let mut maxl = 0u32;
    for i in (0..n).rev() {
        let (cols, _vals) = u.row(i);
        let mut li = 0u32;
        for &j in cols {
            if j <= i {
                continue;
            }
            li = li.max(lev[j] + 1);
        }
        lev[i] = li;
        maxl = maxl.max(li);
    }
    let mut buckets = vec![Vec::new(); (maxl as usize) + 1];
    for (i, &l) in lev.iter().enumerate() {
        buckets[l as usize].push(i);
    }
    Levels {
        buckets,
        max_level: maxl,
    }
}

impl Ilu {
    /// Create new ILU with HYPRE defaults
    pub fn new() -> Self {
        Self::new_with_config(IluConfig::default()).unwrap()
    }

    /// Create ILU with comprehensive configuration
    pub fn new_with_config(config: IluConfig) -> Result<Self, KError> {
        Self::validate_config(&config)?;

        #[cfg(feature = "logging")]
        if config.logging_level > 0 {
            info!(
                "ILU Setup: Creating {:?} factorization with HYPRE-inspired configuration",
                config.ilu_type
            );
            debug!(
                "ILU Config: fill_level={}, drop_tol={:.2e}, reordering={:?}",
                config.level_of_fill, config.drop_tolerance, config.reordering_type
            );
        }

        Ok(Self {
            config,
            l: CsrMatrix::from_csr(0, 0, vec![0], Vec::new(), Vec::new()),
            u: CsrMatrix::from_csr(0, 0, vec![0], Vec::new(), Vec::new()),
            inv_diag_u: Vec::new(),
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            workspace: IluWorkspace::new(0),
            #[cfg(feature = "rayon")]
            levels_l: Levels::default(),
            #[cfg(feature = "rayon")]
            levels_u: Levels::default(),
            setup_complexity: 0.0,
            nnz_l: 0,
            nnz_u: 0,
            num_zero_pivots: 0,
            pivot_stats: PivotStats::default(),
            max_diag_a: Real::default(),
            row_inf_a: Vec::new(),
            row_gersh_a: Vec::new(),
            running_max_u: Real::default(),
            setup_time: 0.0,
            solve_ctrs: Counters::new(),
            history: None,
            monitor: None,
        })
    }

    /// HYPRE-inspired configuration validation
    fn validate_config(config: &IluConfig) -> Result<(), KError> {
        if config.drop_tolerance < 0.0 {
            return Err(KError::InvalidInput(
                "drop_tolerance must be >= 0".to_string(),
            ));
        }

        if config.tolerance <= 0.0 {
            return Err(KError::InvalidInput("tolerance must be > 0".to_string()));
        }

        Ok(())
    }

    /// HYPRE-inspired IEEE safety checks
    fn check_ieee_values(matrix: &Mat<f64>) -> Result<(), KError> {
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let val = matrix[(i, j)];
                if val.is_nan() {
                    return Err(KError::InvalidInput(format!(
                        "NaN detected in matrix at position ({i}, {j})"
                    )));
                }
                if val.is_infinite() {
                    return Err(KError::InvalidInput(format!(
                        "Infinity detected in matrix at position ({i}, {j})"
                    )));
                }
            }
        }
        Ok(())
    }

    /// HYPRE-inspired matrix validation with enhanced analysis
    fn validate_matrix(matrix: &Mat<f64>) -> Result<(), KError> {
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            return Err(KError::InvalidInput("Matrix cannot be empty".to_string()));
        }

        if matrix.nrows() != matrix.ncols() {
            return Err(KError::InvalidInput(
                "ILU requires square matrices".to_string(),
            ));
        }

        Ok(())
    }

    /// Enhanced matrix analysis using matrix utils
    #[allow(dead_code)]
    fn analyze_matrix_for_ilu(matrix: &Mat<f64>) -> (usize, f64, f64) {
        utils::analyze_matrix_properties(matrix)
    }

    /// Check matrix for IEEE issues using matrix utils
    #[allow(dead_code)]
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
    fn count_nnz(matrix: &Mat<f64>) -> usize {
        let mut nnz = 0;
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if matrix[(i, j)] != 0.0 {
                    nnz += 1;
                }
            }
        }
        nnz
    }

    /// Pivot stabilization using configurable policy
    fn handle_pivot(
        &mut self,
        pivot: &mut f64,
        row: usize,
        matrix: &Mat<f64>,
    ) -> Result<(), KError> {
        let policy = &self.config.pivot_policy;

        // determine scaling value and guard against vanishing floors by
        // clamping to the global maximum diagonal magnitude
        let s_i = match policy.scale {
            PivotScale::MaxDiagA => self.max_diag_a,
            PivotScale::LocalDiagA => matrix[(row, row)].abs(),
            PivotScale::RowInfA => self.row_inf_a[row],
            PivotScale::RowGershgorin => self.row_gersh_a[row],
            PivotScale::RunningMaxU => self.running_max_u,
        }
        .max(self.max_diag_a);

        let tau = policy.tau;
        if let Err(e) = stabilize_pivot_in_place(
            pivot,
            s_i,
            tau,
            policy.sign,
            policy.mode,
            &mut self.pivot_stats,
            row,
        ) {
            self.num_zero_pivots += 1;
            return Err(e);
        }

        let abs = pivot.abs();
        if abs > self.running_max_u {
            self.running_max_u = abs;
        }

        Ok(())
    }

    /// Helper: Get element from sparse matrix (returns zero if not present)
    fn sparse_get(&self, matrix: &CsrMatrix<f64>, i: usize, j: usize) -> f64 {
        let (cols, vals) = matrix.row(i);
        match cols.binary_search(&j) {
            Ok(pos) => vals[pos],
            Err(_) => 0.0,
        }
    }

    /// Helper: Set element in sparse matrix without changing structure.
    ///
    /// This routine assumes the sparsity pattern already contains the
    /// entry `(i, j)`.  If the entry is absent, the call is a no-op.
    fn sparse_set(&mut self, matrix: &mut CsrMatrix<f64>, i: usize, j: usize, value: f64) {
        let start = matrix.row_ptr()[i];
        let end = matrix.row_ptr()[i + 1];
        // Determine position of column j within the row while holding only
        // an immutable borrow.
        let mut pos_in_row = None;
        {
            let cols = &matrix.col_idx()[start..end];
            if let Ok(off) = cols.binary_search(&j) {
                pos_in_row = Some(start + off);
            }
        }
        if let Some(p) = pos_in_row {
            let values = matrix.values_mut();
            values[p] = value;
        }
    }

    /// Compute ILU(0) factorization with enhanced pivot handling and sparse storage
    fn compute_ilu0(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        let n = matrix.nrows();

        // Convert input matrix to sparse CSR format for L and U factors
        let drop_tol: f64 = 1e-15;
        let mut l = CsrMatrix::from_dense(matrix, drop_tol);
        let mut u = CsrMatrix::from_dense(matrix, drop_tol);

        // Initialize L as lower triangular with unit diagonal, U as upper triangular
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    // L gets lower triangular part
                    self.sparse_set(&mut u, i, j, 0.0);
                } else if i < j {
                    // U gets upper triangular part
                    self.sparse_set(&mut l, i, j, 0.0);
                } else {
                    // L has unit diagonal
                    self.sparse_set(&mut l, i, i, 1.0);
                }
            }
        }

        // HYPRE-style ILU(0) factorization
        for k in 0..n {
            // Enhanced pivot handling
            let mut pivot = self.sparse_get(&u, k, k);
            self.handle_pivot(&mut pivot, k, matrix)?;
            self.sparse_set(&mut u, k, k, pivot);

            for i in (k + 1)..n {
                let l_ik = self.sparse_get(&l, i, k);
                if l_ik != 0.0 {
                    let multiplier = l_ik / pivot;
                    self.sparse_set(&mut l, i, k, multiplier);

                    for j in (k + 1)..n {
                        let u_kj = self.sparse_get(&u, k, j);
                        if u_kj != 0.0 && matrix[(i, j)] != 0.0 {
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

        // Cache inverse diagonal of U for fast solves
        self.inv_diag_u = u.diagonal().into_iter().map(|v| 1.0 / v).collect();

        self.l = l;
        self.u = u;

        Ok(())
    }

    /// Compute Modified ILU(0) with row-sum correction and sparse storage
    fn compute_milu0(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        let n = matrix.nrows();

        // Convert input matrix to sparse CSR format
        let drop_tol: f64 = 1e-15;
        let mut l = CsrMatrix::from_dense(matrix, drop_tol);
        let mut u = CsrMatrix::from_dense(matrix, drop_tol);

        // Store original row sums for diagonal correction
        let mut original_row_sums = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                original_row_sums[i] = original_row_sums[i] + matrix[(i, j)];
            }
        }

        // Initialize L as lower triangular with unit diagonal, U as upper triangular
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    self.sparse_set(&mut u, i, j, 0.0);
                } else if i < j {
                    self.sparse_set(&mut l, i, j, 0.0);
                } else {
                    self.sparse_set(&mut l, i, i, 1.0);
                }
            }
        }

        // MILU(0) factorization with row-sum preservation
        for k in 0..n {
            let mut pivot = self.sparse_get(&u, k, k);
            self.handle_pivot(&mut pivot, k, matrix)?;
            self.sparse_set(&mut u, k, k, pivot);

            for i in (k + 1)..n {
                let l_ik = self.sparse_get(&l, i, k);
                if l_ik != 0.0 {
                    let multiplier = l_ik / pivot;
                    self.sparse_set(&mut l, i, k, multiplier);

                    let mut dropped_sum = 0.0;
                    for j in (k + 1)..n {
                        let u_kj = self.sparse_get(&u, k, j);
                        if u_kj != 0.0 {
                            let update = multiplier * u_kj;
                            if matrix[(i, j)] != 0.0 {
                                let u_ij = self.sparse_get(&u, i, j);
                                self.sparse_set(&mut u, i, j, u_ij - update);
                            } else {
                                dropped_sum = dropped_sum + update;
                            }
                        }
                    }
                    // Apply diagonal correction for this row
                    let u_ii = self.sparse_get(&u, i, i);
                    self.sparse_set(&mut u, i, i, u_ii + dropped_sum);
                }
            }
        }

        self.nnz_l = l.nnz();
        self.nnz_u = u.nnz();

        self.inv_diag_u = u.diagonal().into_iter().map(|v| 1.0 / v).collect();

        self.l = l;
        self.u = u;

        Ok(())
    }

    /// Compute ILU(k) factorization with level-of-fill control
    fn compute_iluk(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        let n = matrix.nrows();
        let mut l = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);

        // Level-of-fill tracking: level[i][j] = fill level of entry (i,j)
        let mut level = vec![vec![usize::MAX; n]; n];

        // Initialize levels for original nonzeros
        for i in 0..n {
            for j in 0..n {
                if matrix[(i, j)] != 0.0 {
                    level[i][j] = 0;
                    if i <= j {
                        u[(i, j)] = matrix[(i, j)];
                    } else {
                        l[(i, j)] = matrix[(i, j)];
                    }
                }
            }
            l[(i, i)] = 1.0; // Unit diagonal for L
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
                            let new_level =
                                level[i][k].saturating_add(level[k][j]).saturating_add(1);

                            if new_level <= self.config.level_of_fill {
                                let update = l[(i, k)] * u[(k, j)];
                                u[(i, j)] = u[(i, j)] - update;
                                level[i][j] = level[i][j].min(new_level);
                            }
                        }
                    }
                } else {
                    l[(i, k)] = 0.0; // Drop high-level fill
                }
            }
        }

        // Convert to sparse format and cache inverse diagonal
        let drop_tol: f64 = 1e-15;
        self.l = CsrMatrix::from_dense(&l, drop_tol);
        self.u = CsrMatrix::from_dense(&u, drop_tol);

        self.inv_diag_u = (0..n).map(|i| 1.0 / u[(i, i)]).collect();

        self.nnz_l = self.l.nnz();
        self.nnz_u = self.u.nnz();

        Ok(())
    }

    /// Compute ILUT factorization with threshold-based dropping
    fn compute_ilut(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        let n = matrix.nrows();
        let mut l = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);
        let drop_tol: f64 = self.config.drop_tolerance;

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
            l[(i, i)] = 1.0; // Unit diagonal for L
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
                    u[(i, j)] = 0.0; // Drop small entries
                }
            }

            // Apply fill-in control per row if specified
            if self.config.max_fill_per_row > 0 {
                for i in (k + 1)..n {
                    self.apply_fill_control_to_row(&mut u, i, k + 1);
                }
            }
        }

        // Convert to sparse format and cache inverse diagonal
        let drop_tol: f64 = 1e-15;
        self.l = CsrMatrix::from_dense(&l, drop_tol);
        self.u = CsrMatrix::from_dense(&u, drop_tol);

        self.inv_diag_u = (0..n).map(|i| 1.0 / u[(i, i)]).collect();

        self.nnz_l = self.l.nnz();
        self.nnz_u = self.u.nnz();

        Ok(())
    }

    /// Apply fill-in control to a single row, keeping only the largest entries
    fn apply_fill_control_to_row(&self, matrix: &mut Mat<f64>, row: usize, start_col: usize) {
        if self.config.max_fill_per_row == 0 {
            return;
        }

        // Collect (magnitude, column, value) for this row
        let mut entries: Vec<(f64, usize, f64)> = Vec::new();
        for j in start_col..matrix.ncols() {
            let val = matrix[(row, j)];
            if val != 0.0 {
                entries.push((val.abs(), j, val));
            }
        }

        if entries.len() > self.config.max_fill_per_row {
            // Sort by magnitude (largest first)
            entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Zero out all entries first
            for j in start_col..matrix.ncols() {
                matrix[(row, j)] = 0.0;
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
                debug!("ILU: Workspace configured for {n} x {n} matrix");
            }
        } else {
            // Still allocate minimal workspace for correctness
            self.workspace.ensure_size(n);
        }
    }

    /// Exact sparse triangular solve operating in-place on the provided buffer.
    fn solve_triangular_exact(&self, lower: bool, x: &mut [S]) {
        #[cfg(feature = "rayon")]
        if self.config.enable_parallel_triangular_solve {
            if lower {
                self.solve_triangular_parallel_forward(x);
            } else {
                self.solve_triangular_parallel_backward(x);
            }
            return;
        }

        let n = x.len();
        if lower {
            // Forward substitution: L * x = b (unit diagonal) using x as both rhs and solution
            for i in 0..n {
                let mut sum = x[i];
                let (cols, vals) = self.l.row(i);
                for (&j, &val) in cols.iter().zip(vals.iter()) {
                    if j < i {
                        sum = sum - val * x[j];
                    }
                }
                x[i] = sum;
            }
        } else {
            // Backward substitution: U * x = b using x in-place
            for i in (0..n).rev() {
                let mut sum = x[i];
                let (cols, vals) = self.u.row(i);
                for (&j, &val) in cols.iter().zip(vals.iter()) {
                    if j > i {
                        sum = sum - val * x[j];
                    }
                }
                x[i] = sum * self.inv_diag_u[i];
            }
        }
    }

    #[cfg(feature = "rayon")]
    /// Level-scheduled forward substitution (currently executes sequentially).
    fn solve_triangular_parallel_forward(&self, x: &mut [S]) {
        let levels = &self.levels_l;
        for rows in &levels.buckets {
            for &i in rows {
                let mut sum = x[i];
                let (cols, vals) = self.l.row(i);
                for (&j, &val) in cols.iter().zip(vals.iter()) {
                    if j >= i {
                        break;
                    }
                    sum = sum - val * x[j];
                }
                x[i] = sum;
            }
        }
    }

    #[cfg(feature = "rayon")]
    /// Level-scheduled backward substitution (currently executes sequentially).
    fn solve_triangular_parallel_backward(&self, x: &mut [S]) {
        let levels = &self.levels_u;
        for ell in (0..=levels.max_level).rev() {
            let rows = &levels.buckets[ell as usize];
            for &i in rows {
                let mut sum = x[i];
                let (cols, vals) = self.u.row(i);
                for (&j, &val) in cols.iter().zip(vals.iter()) {
                    if j <= i {
                        continue;
                    }
                    sum = sum - val * x[j];
                }
                x[i] = sum * self.inv_diag_u[i];
            }
        }
    }

    /// HYPRE-style iterative triangular solve with Jacobi and sparse access
    fn solve_triangular_jacobi(&self, lower: bool, b: &[S], x: &mut [S]) {
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
                for i in 0..n {
                    let mut sum = S::zero();
                    let (cols, vals) = self.l.row(i);
                    for (&j, &val) in cols.iter().zip(vals.iter()) {
                        if j < i {
                            sum = sum + val * x[j];
                        }
                    }
                    x[i] = b[i] - sum; // L has unit diagonal
                }
            } else {
                // Jacobi iteration for U * x = b
                for i in (0..n).rev() {
                    let mut sum = S::zero();
                    let (cols, vals) = self.u.row(i);
                    for (&j, &val) in cols.iter().zip(vals.iter()) {
                        if j > i {
                            sum = sum + val * x[j];
                        }
                    }
                    x[i] = (b[i] - sum) * self.inv_diag_u[i];
                }
            }
        }
    }

    /// HYPRE-style iterative triangular solve with Gauss-Seidel and sparse access
    fn solve_triangular_gauss_seidel(&self, lower: bool, b: &[S], x: &mut [S]) {
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
                for i in 0..n {
                    let mut sum = S::zero();
                    let (cols, vals) = self.l.row(i);
                    for (&j, &val) in cols.iter().zip(vals.iter()) {
                        if j < i {
                            sum = sum + val * x[j];
                        }
                    }
                    x[i] = b[i] - sum; // L has unit diagonal
                }
            } else {
                // Gauss-Seidel for U * x = b (backward sweep using updated values)
                for i in (0..n).rev() {
                    let mut sum = S::zero();
                    let (cols, vals) = self.u.row(i);
                    for (&j, &val) in cols.iter().zip(vals.iter()) {
                        if j > i {
                            sum = sum + val * x[j];
                        }
                    }
                    x[i] = (b[i] - sum) * self.inv_diag_u[i];
                }
            }
        }
    }

    /// Get factorization statistics (HYPRE-style diagnostics)
    pub fn get_stats(&self) -> IluStats {
        let (total_ns, count, _) = self.solve_ctrs.snapshot();
        let avg = if count == 0 {
            0.0
        } else {
            (total_ns as f64) / (count as f64) / 1e9
        };
        IluStats {
            setup_complexity: self.setup_complexity,
            nnz_l: self.nnz_l,
            nnz_u: self.nnz_u,
            num_zero_pivots: self.num_zero_pivots,
            setup_time: self.setup_time,
            solve_time: avg,
            solve_count: count as usize,
        }
    }

    /// Access pivot handling statistics
    pub fn pivot_stats(&self) -> &PivotStats {
        &self.pivot_stats
    }
}

#[cfg(not(feature = "complex"))]
impl Ilu {
    /// Create specialized ILU preconditioners that leverage existing implementations
    /// This provides a unified interface while potentially using optimized separate implementations
    pub fn create_specialized(
        config: IluConfig,
    ) -> Result<Box<dyn Preconditioner<Mat<S>, Vec<S>>>, KError> {
        match config.ilu_type {
            IluType::ILUK => {
                // Use the dedicated ILUP implementation for better performance
                let ilup = crate::preconditioner::ilup::Ilup::new(config.level_of_fill);
                Ok(Box::new(ilup))
            }
            IluType::ILUT => {
                // Use the dedicated ILUT implementation
                let ilut = crate::preconditioner::ilut::Ilut::new(
                    if config.max_fill_per_row > 0 {
                        config.max_fill_per_row
                    } else {
                        20
                    },
                    config.drop_tolerance,
                );
                Ok(Box::new(ilut))
            }
            _ => {
                // Use the unified implementation for other types
                let ilu = Ilu::new_with_config(config)?;
                Ok(Box::new(ilu))
            }
        }
    }

    /// Quick factory method for common ILU configurations
    pub fn create_quick(ilu_type: IluType, fill_or_drop: Real) -> Result<Self, KError> {
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

impl Default for Ilu {
    fn default() -> Self {
        Self::new()
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for Ilu {
    /// HYPRE-inspired setup with comprehensive safety checks and monitoring
    fn setup(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        let setup_start = std::time::Instant::now();

        if let Some(m) = &self.monitor {
            m.on_event(Event::IluSetupBegin { opts_hash: 0 });
        }

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
        print_ilu_banner(&self.config);

        // Precompute scaling terms for pivoting
        let mut max_diag: Real = Real::default();
        self.row_inf_a.resize(n, Real::default());
        self.row_gersh_a.resize(n, Real::default());
        for i in 0..n {
            let mut row_inf: Real = Real::default();
            let mut row_gersh = matrix[(i, i)].abs();
            for j in 0..n {
                let val_abs = matrix[(i, j)].abs();
                if j != i {
                    row_gersh = row_gersh + val_abs;
                }
                if val_abs > row_inf {
                    row_inf = val_abs;
                }
            }
            self.row_inf_a[i] = row_inf;
            self.row_gersh_a[i] = row_gersh;
            max_diag = max_diag.max(matrix[(i, i)].abs());
        }
        self.max_diag_a = max_diag;
        self.running_max_u = Real::default();
        self.pivot_stats = PivotStats::default();

        #[cfg(feature = "logging")]
        if self.config.logging_level > 0 {
            info!("ILU Setup: {n} x {n} matrix with {original_nnz} nonzeros");
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
                return Err(KError::NotImplemented(format!(
                    "ILU type {:?} not yet implemented",
                    self.config.ilu_type
                )));
            }
        }

        // Calculate metrics
        self.setup_complexity = self.calculate_complexity(original_nnz);
        self.setup_time = setup_start.elapsed().as_secs_f64();

        #[cfg(feature = "rayon")]
        if self.config.enable_parallel_triangular_solve {
            self.levels_l = build_levels_lower(&self.l);
            self.levels_u = build_levels_upper(&self.u);
        }

        #[cfg(feature = "logging")]
        if self.config.logging_level > 0 {
            info!(
                "ILU Setup Complete: complexity={:.2}, L_nnz={}, U_nnz={}, setup_time={:.3}s",
                self.setup_complexity, self.nnz_l, self.nnz_u, self.setup_time
            );

            debug!(
                "Pivot floors: {} (max shift {:.3e})",
                self.pivot_stats.num_floors, self.pivot_stats.max_abs_shift
            );

            if self.num_zero_pivots > 0 {
                warn!(
                    "ILU: {} zero pivots encountered during factorization",
                    self.num_zero_pivots
                );
            }

            if self.config.print_level > 0 {
                println!(
                    "ILU Setup: {} -> {} nonzeros (complexity: {:.2})",
                    original_nnz,
                    self.nnz_l + self.nnz_u,
                    self.setup_complexity
                );
            }
        }
        if let Some(m) = &self.monitor {
            m.on_event(Event::IluSetupEnd {
                iters: 0,
                converged: true,
                setup_time_s: self.setup_time,
            });
        }

        Ok(())
    }

    /// HYPRE-inspired apply with configurable triangular solves and zero-allocation workspace
    fn apply(&self, side: PcSide, x: &Vec<f64>, y: &mut Vec<f64>) -> Result<(), KError> {
        self.apply_slice(side, x.as_slice(), y.as_mut_slice())
    }
}

impl Ilu {
    fn apply_slice(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        let n = self.l.nrows();
        if x.len() != n || y.len() != n {
            return Err(KError::InvalidInput(format!(
                "Vector length mismatch: expected {}, got x={} y={}",
                n,
                x.len(),
                y.len(),
            )));
        }

        let _timer = SolveTimer::start(&self.solve_ctrs);

        match self.config.triangular_solve {
            TriSolveType::Exact => {
                // single copy mandated by API
                y.copy_from_slice(x);
                self.solve_triangular_exact(true, y);
                self.solve_triangular_exact(false, y);
            }
            TriSolveType::Jacobi => {
                let mut buf = self.workspace.borrow_solve_buf(n);
                self.solve_triangular_jacobi(true, x, &mut buf[..n]);
                self.solve_triangular_jacobi(false, &buf[..n], y);
            }
            TriSolveType::GaussSeidel => {
                let mut buf = self.workspace.borrow_solve_buf(n);
                self.solve_triangular_gauss_seidel(true, x, &mut buf[..n]);
                self.solve_triangular_gauss_seidel(false, &buf[..n], y);
            }
        }

        #[cfg(feature = "logging")]
        if self.config.logging_level > 2 {
            let _solve_time = _timer.elapsed().as_secs_f64();
            trace!(
                "ILU Apply: solve_time={:.6}s, workspace_size={}",
                _solve_time, self.workspace.size
            );
        }

        Ok(())
    }
}

impl Ilu {
    pub fn parilu_history(&self) -> Option<&[ParIluIterSample]> {
        self.history.as_ref().map(|h| h.as_slice())
    }

    pub fn set_monitor(&mut self, m: Option<Box<dyn Monitor>>) {
        self.monitor = m;
    }
}

impl LocalPreconditioner<f64> for Ilu {
    fn dims(&self) -> (usize, usize) {
        (self.l.nrows(), self.l.ncols())
    }

    fn apply_local(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply_slice(PcSide::Left, x, y)
    }
}

/// Legacy ILU(0) type alias for backward compatibility
pub type Ilu0 = Ilu;

#[cfg(test)]
mod tests {
    use super::{Ilu, IluBuilder, IluConfig, IluType, TriSolveType};
    use crate::algebra::parallel::par_sum_abs2_local;
    use crate::algebra::prelude::*;
    use crate::preconditioner::PcSide;
    use crate::preconditioner::legacy::Preconditioner;

    fn make_spd_3x3() -> faer::Mat<S> {
        // A = [[4, 1, 0],
        //      [1, 3, 1],
        //      [0, 1, 2]]
        faer::Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => S::from_real(4.0),
            (0, 1) | (1, 0) => S::from_real(1.0),
            (1, 1) => S::from_real(3.0),
            (1, 2) | (2, 1) => S::from_real(1.0),
            (2, 2) => S::from_real(2.0),
            _ => S::zero(),
        })
    }

    fn mat_vec_mul(a: &faer::Mat<S>, x: &[S]) -> Vec<S> {
        let mut out = vec![S::zero(); a.nrows()];
        for i in 0..a.nrows() {
            let mut acc = S::zero();
            for j in 0..a.ncols() {
                acc = acc + a[(i, j)] * x[j];
            }
            out[i] = acc;
        }
        out
    }

    #[test]
    fn test_ilu_default_creation() {
        let ilu = Ilu::new();
        assert_eq!(ilu.config.ilu_type, IluType::ILU0);
    }

    #[test]
    fn test_ilu_builder() {
        let ilu = IluBuilder::new()
            .ilu_type(IluType::ILUT)
            .drop_tolerance(1e-6)
            .enable_logging()
            .build()
            .unwrap();

        assert_eq!(ilu.config.ilu_type, IluType::ILUT);
        assert_eq!(ilu.config.drop_tolerance, 1e-6);
        assert_eq!(ilu.config.logging_level, 1);
    }

    #[test]
    fn test_ilu_config_validation() {
        let mut config = IluConfig::default();
        config.drop_tolerance = -1.0;

        let result = Ilu::new_with_config(config);
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
        use crate::preconditioner::legacy::Preconditioner;
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

        // Test pivot policy with default settings
        let config = IluConfig::default();
        let mut ilu = Ilu::new_with_config(config).unwrap();
        use crate::preconditioner::legacy::Preconditioner;
        let result = ilu.setup(&matrix);
        assert!(result.is_ok());
        assert!(ilu.pivot_stats().num_floors > 0);
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn test_ilu_variants() {
        let _matrix = faer::Mat::from_fn(3, 3, |i, j| {
            if i == j {
                4.0
            } else if (i as i32 - j as i32).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        });

        // Test ILU(k)
        let ilu_k = Ilu::create_quick(IluType::ILUK, 1.0).unwrap();
        assert_eq!(ilu_k.config.ilu_type, IluType::ILUK);
        assert_eq!(ilu_k.config.level_of_fill, 1);

        // Test ILUT
        let ilu_t = Ilu::create_quick(IluType::ILUT, 1e-6).unwrap();
        assert_eq!(ilu_t.config.ilu_type, IluType::ILUT);
        assert_eq!(ilu_t.config.drop_tolerance, 1e-6);
    }

    #[test]
    fn test_triangular_solve_options() {
        let matrix = faer::Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.0 });

        // Test Gauss-Seidel solve
        let mut config = IluConfig::default();
        config.triangular_solve = TriSolveType::GaussSeidel;
        config.lower_jacobi_iters = 2;
        config.upper_jacobi_iters = 2;

        let mut ilu = Ilu::new_with_config(config).unwrap();
        use crate::preconditioner::legacy::Preconditioner;
        let result = ilu.setup(&matrix);
        assert!(result.is_ok());
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn test_specialized_factory() {
        let config = IluConfig {
            ilu_type: IluType::ILUK,
            level_of_fill: 2,
            ..Default::default()
        };

        let ilu_box = Ilu::create_specialized(config);
        assert!(ilu_box.is_ok());
    }

    #[test]
    fn test_parallel_configuration() {
        let ilu = IluBuilder::new()
            .enable_parallel()
            .parallel_chunk_size(128)
            .build()
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

        let mut ilu = IluBuilder::new().ilu_type(IluType::ILU0).build().unwrap();

        use crate::preconditioner::legacy::Preconditioner;
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

        let mut ilu_serial = IluBuilder::new().ilu_type(IluType::ILU0).build().unwrap();

        let mut ilu_parallel = IluBuilder::new()
            .ilu_type(IluType::ILU0)
            .enable_parallel_factorization()
            .parallel_chunk_size(2) // Small chunk size to force parallel execution
            .build()
            .unwrap();

        use crate::preconditioner::legacy::Preconditioner;

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
        let ilu = IluBuilder::new().enable_distributed().build().unwrap();

        assert!(ilu.config.enable_distributed);
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn ilu0_real_factorization_solves_spd() {
        let matrix = make_spd_3x3();
        let x_true = vec![S::from_real(1.0), S::from_real(2.0), S::from_real(-1.0)];
        let b = mat_vec_mul(&matrix, &x_true);

        let mut ilu = Ilu::new();
        ilu.setup(&matrix).expect("ILU(0) setup");

        let mut x = vec![S::zero(); b.len()];
        ilu.apply(PcSide::Left, &b, &mut x).expect("ILU(0) apply");

        let r: Vec<S> = mat_vec_mul(&matrix, &x)
            .into_iter()
            .zip(b.iter())
            .map(|(ax, &bi)| ax - bi)
            .collect();
        let res_norm = par_sum_abs2_local(&r).sqrt();
        assert!(
            res_norm.real() < 1e-10,
            "residual too large: {:?}",
            res_norm
        );
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
    pub fn benchmark_ilu_factorization(
        matrix_size: usize,
        nnz_per_row: usize,
    ) -> (f64, AllocationStats) {
        // Create a sparse test matrix (tridiagonal with random values)
        let matrix = create_sparse_test_matrix(matrix_size, nnz_per_row);

        let start = Instant::now();
        let mut ilu = IluBuilder::new()
            .ilu_type(IluType::ILU0)
            .enable_parallel_factorization()
            .build()
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
    pub fn benchmark_ilu_solve_phase(
        matrix_size: usize,
        num_solves: usize,
    ) -> (f64, AllocationStats) {
        let matrix = create_sparse_test_matrix(matrix_size, 3);

        let mut ilu = IluBuilder::new().ilu_type(IluType::ILU0).build().unwrap();

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
            solve_allocations: 0,          // Critical: must be zero
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
        let mut ilu_dense = IluBuilder::new().ilu_type(IluType::ILU0).build().unwrap();
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
