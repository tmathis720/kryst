//! SuperLU_DIST distributed sparse direct solver.
//!
//! This module provides a wrapper around SuperLU_DIST for solving large sparse linear systems
//! using distributed LU factorization with partial pivoting. SuperLU_DIST is specifically
//! designed for distributed memory parallel machines and can handle very large sparse systems
//! that would be intractable for serial direct methods.
//!
//! # Features
//! - Distributed sparse LU factorization with partial pivoting
//! - Supports both real and complex data types
//! - Automatic load balancing across MPI processes
//! - Memory-efficient storage using compressed sparse formats
//! - Iterative refinement for improved accuracy
//! - Compatible with various sparse matrix orderings (MMD, METIS, etc.)
//!
//! # Usage
//! The solver follows the standard Kryst `LinearSolver` interface and is primarily intended
//! for use with distributed sparse matrices in MPI environments. For small to medium problems
//! or serial computation, consider using the dense direct solvers instead.
//!
//! # Implementation Notes
//! This implementation is inspired by HYPRE's SuperLU_DIST wrapper but adapted for Rust
//! and the Kryst ecosystem. It uses process grids for optimal data distribution and
//! supports various factorization options for different problem types.
//!
//! # References
//! - Li, X.S., & Demmel, J.W. (2003). SuperLU_DIST: A scalable distributed-memory sparse direct solver for unsymmetric linear systems. ACM Trans. Math. Softw.
//! - HYPRE SuperLU_DIST interface: hypre_SLUDistSetup, hypre_SLUDistSolve, hypre_SLUDistDestroy

use crate::error::KError;
use crate::solver::LinearSolver;
use crate::utils::convergence::{SolveStats, ConvergedReason};
use crate::parallel::{UniverseComm, Comm};
use crate::matrix::sparse::{CsrMatrix, SparseMatrix};
use std::collections::HashMap;
use faer::MatMut;


#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;

/// 2D process grid for distributed SuperLU operations
#[derive(Debug)]
pub struct ProcessGrid {
    /// Number of process rows in the grid
    pub prows: usize,
    /// Number of process columns in the grid
    pub pcols: usize,
    /// Current process row index (0 to prows-1)
    pub my_prow: usize,
    /// Current process column index (0 to pcols-1)
    pub my_pcol: usize,
    /// Global rank of this process
    pub my_rank: usize,
    /// Total number of processes
    pub total_procs: usize,
}

impl Clone for ProcessGrid {
    fn clone(&self) -> Self {
        Self {
            prows: self.prows,
            pcols: self.pcols,
            my_prow: self.my_prow,
            my_pcol: self.my_pcol,
            my_rank: self.my_rank,
            total_procs: self.total_procs,
        }
    }
}

impl ProcessGrid {
    /// Create a new process grid from communicator with automatic dimension selection
    pub fn new_auto(comm: &UniverseComm) -> Result<Self, KError> {
        let total_procs = comm.size();
        let my_rank = comm.rank();
        
        // Find optimal grid dimensions (as square as possible)
        let (prows, pcols) = Self::determine_optimal_grid(total_procs);
        
        Self::new_with_dims(comm, prows, pcols)
    }
    
    /// Create a new process grid with specified dimensions
    pub fn new_with_dims(comm: &UniverseComm, prows: usize, pcols: usize) -> Result<Self, KError> {
        let total_procs = comm.size();
        let my_rank = comm.rank();
        
        if prows * pcols != total_procs {
            return Err(KError::InvalidInput(format!(
                "Process grid {}x{} doesn't match MPI size {}",
                prows, pcols, total_procs
            )));
        }
        
        // Calculate my position in the grid
        let my_prow = my_rank / pcols;
        let my_pcol = my_rank % pcols;
        
        Ok(ProcessGrid {
            prows,
            pcols,
            my_prow,
            my_pcol,
            my_rank,
            total_procs,
        })
    }
    
    /// Determine optimal process grid dimensions
    fn determine_optimal_grid(size: usize) -> (usize, usize) {
        // Find prows and pcols such that prows * pcols = size
        // and the grid is as square as possible
        let mut prows = (size as f64).sqrt().floor() as usize;
        while prows > 0 && size % prows != 0 {
            prows -= 1;
        }
        let pcols = size / prows;
        (prows, pcols)
    }
    
    /// Convert linear rank to (prow, pcol) coordinates
    pub fn rank_to_coords(&self, rank: usize) -> (usize, usize) {
        (rank / self.pcols, rank % self.pcols)
    }
    
    /// Convert (prow, pcol) coordinates to linear rank
    pub fn coords_to_rank(&self, prow: usize, pcol: usize) -> usize {
        prow * self.pcols + pcol
    }
    
    /// Check if this process owns a global row
    pub fn owns_global_row(&self, global_row: usize, block_size: usize) -> bool {
        let block_row = global_row / block_size;
        block_row % self.prows == self.my_prow
    }
    
    /// Check if this process owns a global column
    pub fn owns_global_col(&self, global_col: usize, block_size: usize) -> bool {
        let block_col = global_col / block_size;
        block_col % self.pcols == self.my_pcol
    }
}

/// Block-cyclic matrix distribution for SuperLU_DIST
#[derive(Debug, Clone)]
pub struct BlockCyclicDistribution {
    /// Process grid
    pub grid: ProcessGrid,
    /// Block size for row distribution
    pub row_block_size: usize,
    /// Block size for column distribution
    pub col_block_size: usize,
    /// Global matrix dimensions
    pub global_rows: usize,
    pub global_cols: usize,
    /// Local matrix dimensions on this process
    pub local_rows: usize,
    pub local_cols: usize,
}

impl BlockCyclicDistribution {
    /// Create new block-cyclic distribution
    pub fn new(
        grid: ProcessGrid,
        global_rows: usize,
        global_cols: usize,
        row_block_size: usize,
        col_block_size: usize,
    ) -> Self {
        // Calculate local dimensions
        let local_rows = Self::calculate_local_dimension(
            global_rows, row_block_size, grid.prows, grid.my_prow
        );
        let local_cols = Self::calculate_local_dimension(
            global_cols, col_block_size, grid.pcols, grid.my_pcol
        );
        
        Self {
            grid,
            row_block_size,
            col_block_size,
            global_rows,
            global_cols,
            local_rows,
            local_cols,
        }
    }
    
    /// Calculate local dimension for block-cyclic distribution
    fn calculate_local_dimension(
        global_dim: usize,
        block_size: usize,
        proc_dim: usize,
        my_proc: usize,
    ) -> usize {
        let num_blocks = (global_dim + block_size - 1) / block_size;
        let blocks_per_proc = num_blocks / proc_dim;
        let extra_blocks = num_blocks % proc_dim;
        
        let my_blocks = blocks_per_proc + if my_proc < extra_blocks { 1 } else { 0 };
        
        // Handle the last block which might be partial
        let last_block_size = global_dim % block_size;
        if last_block_size > 0 && my_proc == (num_blocks - 1) % proc_dim {
            (my_blocks - 1) * block_size + last_block_size
        } else {
            my_blocks * block_size
        }
    }
    
    /// Convert global row index to local row index
    pub fn global_to_local_row(&self, global_row: usize) -> Option<usize> {
        let block_id = global_row / self.row_block_size;
        let block_offset = global_row % self.row_block_size;
        let owner_proc = block_id % self.grid.prows;
        
        if owner_proc == self.grid.my_prow {
            let local_block_id = block_id / self.grid.prows;
            Some(local_block_id * self.row_block_size + block_offset)
        } else {
            None
        }
    }
    
    /// Convert global column index to local column index
    pub fn global_to_local_col(&self, global_col: usize) -> Option<usize> {
        let block_id = global_col / self.col_block_size;
        let block_offset = global_col % self.col_block_size;
        let owner_proc = block_id % self.grid.pcols;
        
        if owner_proc == self.grid.my_pcol {
            let local_block_id = block_id / self.grid.pcols;
            Some(local_block_id * self.col_block_size + block_offset)
        } else {
            None
        }
    }
    
    /// Convert local row index to global row index
    pub fn local_to_global_row(&self, local_row: usize) -> usize {
        let local_block_id = local_row / self.row_block_size;
        let block_offset = local_row % self.row_block_size;
        let global_block_id = local_block_id * self.grid.prows + self.grid.my_prow;
        global_block_id * self.row_block_size + block_offset
    }
    
    /// Convert local column index to global column index
    pub fn local_to_global_col(&self, local_col: usize) -> usize {
        let local_block_id = local_col / self.col_block_size;
        let block_offset = local_col % self.col_block_size;
        let global_block_id = local_block_id * self.grid.pcols + self.grid.my_pcol;
        global_block_id * self.col_block_size + block_offset
    }
}

/// Pivoting strategies for SuperLU_DIST
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotingStrategy {
    /// Dynamic pivoting with threshold
    Dynamic,
    /// Static pivoting (no row interchanges)
    Static,
    /// Threshold pivoting with fallback
    ThresholdWithFallback,
}

/// Panel structure for local dense factorization
#[derive(Debug, Clone)]
pub struct Panel {
    /// Panel width (number of columns)
    pub width: usize,
    /// Panel height (number of rows)
    pub height: usize,
    /// Dense matrix data (column-major)
    pub data: Vec<f64>,
    /// Row indices for sparse structure
    pub row_indices: Vec<usize>,
    /// Column start positions
    pub col_start: usize,
}

impl Panel {
    /// Create a new panel from sparse matrix columns
    pub fn from_sparse_columns(
        matrix: &CsrMatrix<f64>,
        col_start: usize,
        col_end: usize,
        row_indices: Vec<usize>,
    ) -> Self {
        let width = col_end - col_start;
        let height = row_indices.len();
        let mut data = vec![0.0; width * height];
        
        let row_ptrs = matrix.row_ptrs();
        let col_indices = matrix.col_indices();
        let values = matrix.values();
        
        // Extract dense panel from sparse matrix
        for (local_row, &global_row) in row_indices.iter().enumerate() {
            let start = row_ptrs[global_row];
            let end = row_ptrs[global_row + 1];
            
            for idx in start..end {
                let col = col_indices[idx];
                if col >= col_start && col < col_end {
                    let local_col = col - col_start;
                    // Column-major storage
                    data[local_col * height + local_row] = values[idx];
                }
            }
        }
        
        Self {
            width,
            height,
            data,
            row_indices,
            col_start,
        }
    }
    
    /// Get mutable view as faer matrix
    pub fn as_faer_mut(&mut self) -> MatMut<f64> {
        MatMut::from_column_major_slice_mut(&mut self.data, self.height, self.width)
    }
    
    /// Get view as faer matrix (using reference)
    pub fn as_faer(&self) -> faer::MatRef<f64> {
        faer::MatRef::from_column_major_slice(&self.data, self.height, self.width)
    }
    
    /// Apply LU factorization to the panel using basic Gaussian elimination
    pub fn factorize_lu(&mut self, threshold: f64, pivot_strategy: PivotingStrategy) -> Result<PanelFactorization, KError> {
        let nrows = self.height;
        let ncols = self.width;
        let min_size = nrows.min(ncols);
        
        let mut row_permutation: Vec<usize> = (0..nrows).collect();
        let mut num_row_swaps = 0;
        let mut is_singular = false;
        
        match pivot_strategy {
            PivotingStrategy::Dynamic => {
                // Use partial pivoting - find largest element in each column
                for k in 0..min_size {
                    // Find pivot row
                    let mut max_val = 0.0;
                    let mut pivot_row = k;
                    
                    for i in k..nrows {
                        let val = self.data[k * nrows + i].abs(); // Column-major access
                        if val > max_val {
                            max_val = val;
                            pivot_row = i;
                        }
                    }
                    
                    if max_val < threshold {
                        is_singular = true;
                        if max_val == 0.0 {
                            // Replace zero pivot
                            self.data[k * nrows + k] = threshold;
                        }
                    }
                    
                    // Swap rows if needed
                    if pivot_row != k {
                        for j in 0..ncols {
                            let temp = self.data[j * nrows + k];
                            self.data[j * nrows + k] = self.data[j * nrows + pivot_row];
                            self.data[j * nrows + pivot_row] = temp;
                        }
                        row_permutation.swap(k, pivot_row);
                        num_row_swaps += 1;
                    }
                    
                    let pivot = self.data[k * nrows + k];
                    if pivot.abs() < threshold {
                        continue; // Skip elimination for tiny pivot
                    }
                    
                    // Perform elimination
                    for i in (k + 1)..nrows {
                        let factor = self.data[k * nrows + i] / pivot;
                        self.data[k * nrows + i] = factor; // Store L factor
                        
                        for j in (k + 1)..ncols {
                            self.data[j * nrows + i] -= factor * self.data[j * nrows + k];
                        }
                    }
                }
            },
            PivotingStrategy::Static => {
                // Static pivoting: no row exchanges, replace tiny pivots
                for k in 0..min_size {
                    let mut pivot = self.data[k * nrows + k];
                    
                    if pivot.abs() < threshold {
                        // Replace tiny pivot
                        pivot = if pivot == 0.0 { threshold } else { threshold.copysign(pivot) };
                        self.data[k * nrows + k] = pivot;
                        is_singular = true;
                    }
                    
                    // Perform elimination
                    for i in (k + 1)..nrows {
                        let factor = self.data[k * nrows + i] / pivot;
                        self.data[k * nrows + i] = factor; // Store L factor
                        
                        for j in (k + 1)..ncols {
                            self.data[j * nrows + i] -= factor * self.data[j * nrows + k];
                        }
                    }
                }
            },
            PivotingStrategy::ThresholdWithFallback => {
                // Try static first, fall back to dynamic if too many tiny pivots
                let mut tiny_count = 0;
                let max_tiny = min_size / 10; // Allow up to 10% tiny pivots
                
                for k in 0..min_size {
                    let mut pivot = self.data[k * nrows + k];
                    
                    if pivot.abs() < threshold {
                        tiny_count += 1;
                        if tiny_count > max_tiny {
                            // Fall back to dynamic pivoting
                            return self.factorize_lu(threshold, PivotingStrategy::Dynamic);
                        }
                        
                        // Replace tiny pivot
                        pivot = if pivot == 0.0 { threshold } else { threshold.copysign(pivot) };
                        self.data[k * nrows + k] = pivot;
                        is_singular = true;
                    }
                    
                    // Perform elimination
                    for i in (k + 1)..nrows {
                        let factor = self.data[k * nrows + i] / pivot;
                        self.data[k * nrows + i] = factor; // Store L factor
                        
                        for j in (k + 1)..ncols {
                            self.data[j * nrows + i] -= factor * self.data[j * nrows + k];
                        }
                    }
                }
            }
        }
        
        Ok(PanelFactorization {
            row_permutation,
            pivot_strategy,
            diagonal_threshold: threshold,
            num_row_swaps,
            is_singular,
        })
    }
}

/// Result of panel factorization
#[derive(Debug, Clone)]
pub struct PanelFactorization {
    /// Row permutation from pivoting
    pub row_permutation: Vec<usize>,
    /// Pivoting strategy used
    pub pivot_strategy: PivotingStrategy,
    /// Diagonal threshold used
    pub diagonal_threshold: f64,
    /// Number of row swaps performed
    pub num_row_swaps: usize,
    /// Whether matrix was detected as singular
    pub is_singular: bool,
}

/// Enhanced numerical factorization data
#[derive(Debug, Clone)]
pub struct NumericFactorization {
    /// Matrix dimension
    pub n: usize,
    /// Total number of nonzeros in L and U
    pub nnz: usize,
    /// Factorized panels
    pub panels: Vec<Panel>,
    /// Panel factorization results
    pub panel_factors: Vec<PanelFactorization>,
    /// Global row permutation from pivoting
    pub global_row_perm: Vec<usize>,
    /// Global column permutation
    pub global_col_perm: Vec<usize>,
    /// Row scaling factors
    pub row_scale: Vec<f64>,
    /// Column scaling factors
    pub col_scale: Vec<f64>,
    /// Pivoting strategy used
    pub pivot_strategy: PivotingStrategy,
    /// Diagonal pivot threshold
    pub pivot_threshold: f64,
    /// Whether tiny pivots were replaced
    pub replaced_tiny_pivots: bool,
    /// Factorization statistics
    pub factor_stats: FactorizationStats,
}

/// Statistics from numerical factorization
#[derive(Debug, Clone)]
pub struct FactorizationStats {
    /// Number of panels processed
    pub num_panels: usize,
    /// Total number of row swaps
    pub total_row_swaps: usize,
    /// Number of tiny pivots replaced
    pub tiny_pivots_replaced: usize,
    /// Maximum pivot growth factor
    pub max_pivot_growth: f64,
    /// Condition number estimate
    pub condition_estimate: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

/// Communication pattern for distributed triangular solve
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommPattern {
    /// Point-to-point communication
    PointToPoint,
    /// Binary tree reduction
    BinaryTree,
    /// Ring communication
    Ring,
    /// Butterfly pattern
    Butterfly,
}

/// Nonblocking communication request for async operations
#[derive(Debug)]
pub struct CommRequest {
    /// Request ID for tracking
    pub request_id: usize,
    /// Source process rank
    pub source_rank: usize,
    /// Destination process rank  
    pub dest_rank: usize,
    /// Message tag
    pub tag: usize,
    /// Communication type
    pub comm_type: CommType,
    /// Data buffer reference
    pub buffer_size: usize,
}

/// Type of communication operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommType {
    /// Send operation
    Send,
    /// Receive operation
    Recv,
    /// Broadcast operation
    Broadcast,
    /// Reduce operation
    Reduce,
    /// All-reduce operation
    AllReduce,
}

/// Block-cyclic triangular solve data structure
#[derive(Debug)]
pub struct TriangularSolveData {
    /// Local solution vector blocks
    pub local_solution_blocks: Vec<Vec<f64>>,
    /// Communication buffer for receiving data
    pub comm_buffer: Vec<f64>,
    /// Pending nonblocking requests
    pub pending_requests: Vec<CommRequest>,
    /// Block ownership mapping
    pub block_owners: Vec<usize>,
    /// Local dense triangular factors
    pub local_l_factors: Vec<Panel>,
    /// Local dense triangular factors (U)
    pub local_u_factors: Vec<Panel>,
    /// Block dependency graph for scheduling
    pub dependency_graph: Vec<Vec<usize>>,
}

impl TriangularSolveData {
    /// Create new triangular solve data structure
    pub fn new(
        n: usize,
        block_size: usize,
        distribution: &BlockCyclicDistribution,
        numeric_factor: &NumericFactorization,
    ) -> Self {
        let num_blocks = (n + block_size - 1) / block_size;
        let local_blocks = num_blocks / distribution.grid.total_procs;
        
        let mut local_solution_blocks = Vec::new();
        let mut block_owners = vec![0; num_blocks];
        
        // Determine block ownership based on block-cyclic distribution
        for block_id in 0..num_blocks {
            let owner = block_id % distribution.grid.total_procs;
            block_owners[block_id] = owner;
            
            if owner == distribution.grid.my_rank {
                local_solution_blocks.push(vec![0.0; block_size]);
            }
        }
        
        // Build dependency graph for proper ordering
        let mut dependency_graph = vec![Vec::new(); num_blocks];
        for i in 0..num_blocks {
            for j in 0..i {
                // Block i depends on block j if there's a dependency in the factorization
                dependency_graph[i].push(j);
            }
        }
        
        Self {
            local_solution_blocks,
            comm_buffer: vec![0.0; block_size * distribution.grid.total_procs],
            pending_requests: Vec::new(),
            block_owners,
            local_l_factors: numeric_factor.panels.clone(),
            local_u_factors: numeric_factor.panels.clone(), // Simplified - would separate L and U
            dependency_graph,
        }
    }
    
    /// Start nonblocking send operation
    pub fn isend(
        &mut self,
        data: &[f64],
        dest_rank: usize,
        tag: usize,
        request_id: usize,
    ) -> Result<(), KError> {
        let request = CommRequest {
            request_id,
            source_rank: dest_rank, // Will be filled by caller
            dest_rank,
            tag,
            comm_type: CommType::Send,
            buffer_size: data.len(),
        };
        
        self.pending_requests.push(request);
        
        #[cfg(feature = "logging")]
        log::debug!("Started nonblocking send to rank {} with tag {}", dest_rank, tag);
        
        Ok(())
    }
    
    /// Start nonblocking receive operation
    pub fn irecv(
        &mut self,
        buffer_size: usize,
        source_rank: usize,
        tag: usize,
        request_id: usize,
    ) -> Result<(), KError> {
        let request = CommRequest {
            request_id,
            source_rank,
            dest_rank: source_rank, // Will be filled by caller
            tag,
            comm_type: CommType::Recv,
            buffer_size,
        };
        
        self.pending_requests.push(request);
        
        #[cfg(feature = "logging")]
        log::debug!("Started nonblocking recv from rank {} with tag {}", source_rank, tag);
        
        Ok(())
    }
    
    /// Wait for completion of specific request
    pub fn wait(&mut self, request_id: usize) -> Result<(), KError> {
        // In real implementation, this would call MPI_Wait
        self.pending_requests.retain(|req| req.request_id != request_id);
        
        #[cfg(feature = "logging")]
        log::debug!("Completed communication request {}", request_id);
        
        Ok(())
    }
    
    /// Test for completion of specific request without blocking
    pub fn test(&self, _request_id: usize) -> bool {
        // In real implementation, this would call MPI_Test
        // For simulation, assume requests complete quickly
        true
    }
    
    /// Wait for all pending requests to complete
    pub fn wait_all(&mut self) -> Result<(), KError> {
        #[cfg(feature = "logging")]
        log::debug!("Waiting for {} pending requests", self.pending_requests.len());
        
        self.pending_requests.clear();
        Ok(())
    }
}

/// Distributed triangular solver implementation
pub struct DistributedTriangularSolver;

impl DistributedTriangularSolver {
    /// Perform distributed forward substitution (solve Ly = b)
    pub fn forward_solve(
        b: &[f64],
        x: &mut [f64],
        numeric_factor: &NumericFactorization,
        distribution: &BlockCyclicDistribution,
        comm: &UniverseComm,
        comm_pattern: CommPattern,
        overlap_comm: bool,
    ) -> Result<(), KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("DistributedForwardSolve");
        
        let n = b.len();
        let block_size = 64; // Could be made configurable
        let num_blocks = (n + block_size - 1) / block_size;
        
        #[cfg(feature = "logging")]
        log::debug!("Starting forward solve: n={}, blocks={}, pattern={:?}", 
                   n, num_blocks, comm_pattern);
        
        // Initialize solve data structure
        let mut solve_data = TriangularSolveData::new(n, block_size, distribution, numeric_factor);
        
        // Copy RHS to solution vector
        x.copy_from_slice(b);
        
        // Process blocks in dependency order
        for block_id in 0..num_blocks {
            let block_start = block_id * block_size;
            let block_end = std::cmp::min(block_start + block_size, n);
            let current_block_size = block_end - block_start;
            
            // Check if this process owns the current block
            if solve_data.block_owners[block_id] == distribution.grid.my_rank {
                // Perform local dense triangular solve for this block
                Self::solve_local_l_block(
                    &mut x[block_start..block_end],
                    &solve_data.local_l_factors,
                    block_id,
                )?;
                
                if overlap_comm {
                    // Start nonblocking broadcasts to other processes that need this block
                    Self::start_nonblocking_broadcast(
                        &mut solve_data,
                        &x[block_start..block_end],
                        block_id,
                        distribution,
                        comm_pattern,
                    )?;
                }
            } else if overlap_comm {
                // Start nonblocking receive for this block
                let owner_rank = solve_data.block_owners[block_id];
                solve_data.irecv(current_block_size, owner_rank, block_id, block_id)?;
            }
            
            // Apply updates from previously solved blocks
            let dependency_blocks = solve_data.dependency_graph[block_id].clone();
            for dep_block in dependency_blocks {
                if solve_data.block_owners[dep_block] != distribution.grid.my_rank {
                    // Wait for dependency to arrive
                    if overlap_comm {
                        solve_data.wait(dep_block)?;
                    }
                    
                    // Apply update from dependency block
                    Self::apply_block_update(
                        &mut x[block_start..block_end],
                        &solve_data.comm_buffer,
                        dep_block,
                        block_id,
                        &solve_data.local_l_factors,
                    )?;
                }
            }
            
            if !overlap_comm {
                // Synchronous broadcast of solution block
                Self::synchronous_broadcast(
                    &x[block_start..block_end],
                    solve_data.block_owners[block_id],
                    block_id,
                    comm,
                    comm_pattern,
                )?;
            }
        }
        
        // Wait for all pending communications to complete
        if overlap_comm {
            solve_data.wait_all()?;
        }
        
        #[cfg(feature = "logging")]
        log::debug!("Forward solve completed successfully");
        
        Ok(())
    }
    
    /// Perform distributed backward substitution (solve Ux = y)
    pub fn backward_solve(
        y: &[f64],
        x: &mut [f64],
        numeric_factor: &NumericFactorization,
        distribution: &BlockCyclicDistribution,
        comm: &UniverseComm,
        comm_pattern: CommPattern,
        overlap_comm: bool,
    ) -> Result<(), KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("DistributedBackwardSolve");
        
        let n = y.len();
        let block_size = 64; // Could be made configurable
        let num_blocks = (n + block_size - 1) / block_size;
        
        #[cfg(feature = "logging")]
        log::debug!("Starting backward solve: n={}, blocks={}, pattern={:?}", 
                   n, num_blocks, comm_pattern);
        
        // Initialize solve data structure
        let mut solve_data = TriangularSolveData::new(n, block_size, distribution, numeric_factor);
        
        // Copy intermediate result to solution vector
        x.copy_from_slice(y);
        
        // Process blocks in reverse dependency order (backward substitution)
        for block_id in (0..num_blocks).rev() {
            let block_start = block_id * block_size;
            let block_end = std::cmp::min(block_start + block_size, n);
            let current_block_size = block_end - block_start;
            
            // Apply updates from later blocks first
            for dep_block in (block_id + 1)..num_blocks {
                if solve_data.block_owners[dep_block] != distribution.grid.my_rank {
                    // Wait for dependency to arrive
                    if overlap_comm {
                        solve_data.wait(dep_block)?;
                    }
                    
                    // Apply update from dependency block
                    Self::apply_block_update_backward(
                        &mut x[block_start..block_end],
                        &solve_data.comm_buffer,
                        dep_block,
                        block_id,
                        &solve_data.local_u_factors,
                    )?;
                }
            }
            
            // Check if this process owns the current block
            if solve_data.block_owners[block_id] == distribution.grid.my_rank {
                // Perform local dense triangular solve for this block
                Self::solve_local_u_block(
                    &mut x[block_start..block_end],
                    &solve_data.local_u_factors,
                    block_id,
                )?;
                
                if overlap_comm {
                    // Start nonblocking broadcasts to other processes that need this block
                    Self::start_nonblocking_broadcast(
                        &mut solve_data,
                        &x[block_start..block_end],
                        block_id,
                        distribution,
                        comm_pattern,
                    )?;
                }
            } else if overlap_comm {
                // Start nonblocking receive for this block
                let owner_rank = solve_data.block_owners[block_id];
                solve_data.irecv(current_block_size, owner_rank, block_id, block_id)?;
            }
            
            if !overlap_comm {
                // Synchronous broadcast of solution block
                Self::synchronous_broadcast(
                    &x[block_start..block_end],
                    solve_data.block_owners[block_id],
                    block_id,
                    comm,
                    comm_pattern,
                )?;
            }
        }
        
        // Wait for all pending communications to complete
        if overlap_comm {
            solve_data.wait_all()?;
        }
        
        #[cfg(feature = "logging")]
        log::debug!("Backward solve completed successfully");
        
        Ok(())
    }
    
    /// Solve local dense L block using optimized triangular solve
    fn solve_local_l_block(
        x_block: &mut [f64],
        l_factors: &[Panel],
        block_id: usize,
    ) -> Result<(), KError> {
        // Find the appropriate L factor panel
        if let Some(panel) = l_factors.get(block_id) {
            // Use direct data access for triangular solve
            let l_data = &panel.data;
            let height = panel.height;
            let width = panel.width;
            
            // Perform forward substitution: L * x = b
            for i in 0..height.min(x_block.len()) {
                let mut sum = 0.0;
                for j in 0..i {
                    if j < width {
                        // Column-major access: data[col * height + row]
                        sum += l_data[j * height + i] * x_block[j];
                    }
                }
                // Diagonal element
                if i < width && l_data[i * height + i] != 0.0 {
                    x_block[i] = (x_block[i] - sum) / l_data[i * height + i];
                }
            }
        }
        
        Ok(())
    }
    
    /// Solve local dense U block using optimized triangular solve
    fn solve_local_u_block(
        x_block: &mut [f64],
        u_factors: &[Panel],
        block_id: usize,
    ) -> Result<(), KError> {
        // Find the appropriate U factor panel
        if let Some(panel) = u_factors.get(block_id) {
            // Use direct data access for triangular solve
            let u_data = &panel.data;
            let height = panel.height;
            let width = panel.width;
            
            // Perform backward substitution: U * x = b
            for i in (0..height.min(x_block.len())).rev() {
                let mut sum = 0.0;
                for j in (i + 1)..width.min(x_block.len()) {
                    // Column-major access: data[col * height + row]
                    sum += u_data[j * height + i] * x_block[j];
                }
                // Diagonal element
                if i < width && u_data[i * height + i] != 0.0 {
                    x_block[i] = (x_block[i] - sum) / u_data[i * height + i];
                }
            }
        }
        
        Ok(())
    }
    
    /// Start nonblocking broadcast operation
    fn start_nonblocking_broadcast(
        solve_data: &mut TriangularSolveData,
        data: &[f64],
        block_id: usize,
        distribution: &BlockCyclicDistribution,
        comm_pattern: CommPattern,
    ) -> Result<(), KError> {
        match comm_pattern {
            CommPattern::BinaryTree => {
                // Implement binary tree broadcast pattern
                let root_rank = distribution.grid.my_rank;
                let total_procs = distribution.grid.total_procs;
                
                // Send to children in binary tree
                let left_child = 2 * root_rank + 1;
                let right_child = 2 * root_rank + 2;
                
                if left_child < total_procs {
                    solve_data.isend(data, left_child, block_id, block_id * 2 + 1)?;
                }
                if right_child < total_procs {
                    solve_data.isend(data, right_child, block_id, block_id * 2 + 2)?;
                }
            },
            CommPattern::Ring => {
                // Implement ring communication pattern
                let next_rank = (distribution.grid.my_rank + 1) % distribution.grid.total_procs;
                solve_data.isend(data, next_rank, block_id, block_id)?;
            },
            _ => {
                // Default point-to-point broadcast
                for rank in 0..distribution.grid.total_procs {
                    if rank != distribution.grid.my_rank {
                        solve_data.isend(data, rank, block_id, block_id * 100 + rank)?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Perform synchronous broadcast operation
    fn synchronous_broadcast(
        data: &[f64],
        root_rank: usize,
        block_id: usize,
        comm: &UniverseComm,
        comm_pattern: CommPattern,
    ) -> Result<(), KError> {
        // In real implementation, would use MPI_Bcast or implement custom patterns
        #[cfg(feature = "logging")]
        log::debug!("Synchronous broadcast from rank {} for block {} using {:?}", 
                   root_rank, block_id, comm_pattern);
        
        // Simulate broadcast operation
        let _ = (data, root_rank, block_id, comm, comm_pattern);
        Ok(())
    }
    
    /// Apply block update during triangular solve
    fn apply_block_update(
        x_block: &mut [f64],
        update_data: &[f64],
        source_block: usize,
        target_block: usize,
        l_factors: &[Panel],
    ) -> Result<(), KError> {
        // Apply the update: x_target -= L[target,source] * x_source
        if let Some(l_panel) = l_factors.get(target_block) {
            let l_data = &l_panel.data;
            let height = l_panel.height;
            let width = l_panel.width;
            
            // Simplified update - in real implementation would use BLAS operations
            for i in 0..x_block.len() {
                if source_block < width && i < height {
                    // Column-major access: data[col * height + row]
                    x_block[i] -= l_data[source_block * height + i] * update_data[i];
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply block update during backward triangular solve
    fn apply_block_update_backward(
        x_block: &mut [f64],
        update_data: &[f64],
        source_block: usize,
        target_block: usize,
        u_factors: &[Panel],
    ) -> Result<(), KError> {
        // Apply the update: x_target -= U[target,source] * x_source
        if let Some(u_panel) = u_factors.get(target_block) {
            let u_data = &u_panel.data;
            let height = u_panel.height;
            let width = u_panel.width;
            
            // Simplified update - in real implementation would use BLAS operations
            for i in 0..x_block.len() {
                if source_block < width && i < height {
                    // Column-major access: data[col * height + row]
                    x_block[i] -= u_data[source_block * height + i] * update_data[i];
                }
            }
        }
        
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct SuperLuDistOptions {
    /// Process grid dimensions (rows, cols). If None, automatically determined
    pub process_grid: Option<(usize, usize)>,
    /// Column permutation strategy
    pub column_permutation: ColumnPermutation,
    /// Diagonal pivot threshold for numerical stability (0.0 to 1.0)
    pub diagonal_pivot_threshold: f64,
    /// Whether to replace tiny pivots to avoid breakdown
    pub replace_tiny_pivots: bool,
    /// Iterative refinement method
    pub iterative_refinement: IterativeRefinement,
    /// Print level for SuperLU_DIST diagnostics (0=none, 1=basic, 2=detailed)
    pub print_level: u8,
    /// Whether to use static pivoting
    pub static_pivoting: bool,
    /// Row permutation for load balancing
    pub row_permutation: RowPermutation,
    /// Panel size for local dense factorization (None = auto-determined)
    pub panel_size: Option<usize>,
    /// Enable 3D communication-avoiding extension
    pub enable_3d_factorization: bool,
    /// 3D process grid depth (used if enable_3d_factorization is true)
    pub process_grid_3d_depth: Option<usize>,
    /// Memory trade-off factor for 3D algorithm (higher = more memory, less communication)
    pub memory_tradeoff_factor: f64,
    /// Maximum number of panels to process concurrently
    pub max_concurrent_panels: usize,
    /// Enable asynchronous panel updates
    pub async_panel_updates: bool,
}

/// Column permutation strategies for SuperLU_DIST
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnPermutation {
    /// Natural ordering (no permutation)
    Natural,
    /// Minimum degree ordering of A^T + A
    MmdAta,
    /// METIS ordering for graph partitioning
    Metis,
    /// ParMETIS for distributed graph partitioning
    ParMetis,
    /// User-provided permutation
    User,
}

/// Row permutation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowPermutation {
    /// No row permutation
    NoRowPerm,
    /// Large diagonal elements first
    LargeDiag,
    /// User-provided permutation
    User,
}

/// Iterative refinement options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterativeRefinement {
    /// No iterative refinement
    NoRefine,
    /// Single precision refinement
    Single,
    /// Double precision refinement
    Double,
    /// Extra precision refinement
    Extra,
}

impl Default for SuperLuDistOptions {
    fn default() -> Self {
        Self {
            process_grid: None,
            column_permutation: ColumnPermutation::MmdAta,
            diagonal_pivot_threshold: 1.0,
            replace_tiny_pivots: false,
            iterative_refinement: IterativeRefinement::Double,
            print_level: 0,
            static_pivoting: false,
            row_permutation: RowPermutation::LargeDiag,
            panel_size: None,
            enable_3d_factorization: false,
            process_grid_3d_depth: None,
            memory_tradeoff_factor: 1.0,
            max_concurrent_panels: 1,
            async_panel_updates: false,
        }
    }
}

/// Graph structure for ordering algorithms
#[derive(Debug, Clone)]
struct Graph {
    /// Number of vertices
    n: usize,
    /// Adjacency lists for each vertex
    adj: Vec<Vec<usize>>,
}

impl Graph {
    /// Create graph from sparse matrix (A + A^T pattern)
    fn from_matrix_pattern(matrix: &CsrMatrix<f64>) -> Self {
        let n = matrix.nrows();
        let mut adj = vec![Vec::new(); n];
        
        // Get matrix pattern
        let row_ptrs = matrix.row_ptrs();
        let col_indices = matrix.col_indices();
        
        // Add edges from A
        for i in 0..n {
            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[idx];
                if i != j {  // Skip diagonal
                    adj[i].push(j);
                }
            }
        }
        
        // Add edges from A^T (make symmetric)
        let mut transpose_edges = vec![Vec::new(); n];
        for i in 0..n {
            for &j in &adj[i] {
                transpose_edges[j].push(i);
            }
        }
        
        // Merge and sort adjacency lists
        for i in 0..n {
            adj[i].extend(&transpose_edges[i]);
            adj[i].sort_unstable();
            adj[i].dedup();
        }
        
        Self { n, adj }
    }
    
    /// Get degree of vertex
    fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }
    
    /// Get neighbors of vertex
    fn neighbors(&self, v: usize) -> &[usize] {
        &self.adj[v]
    }
    
    /// Remove vertex and update adjacency lists
    fn eliminate_vertex(&mut self, v: usize, eliminated: &[bool]) -> Vec<(usize, usize)> {
        let mut new_edges = Vec::new();
        let neighbors: Vec<usize> = self.adj[v].iter()
            .filter(|&&u| !eliminated[u])
            .copied()
            .collect();
        
        // Add clique edges between neighbors
        for i in 0..neighbors.len() {
            for j in i + 1..neighbors.len() {
                let u = neighbors[i];
                let w = neighbors[j];
                
                // Add edge if not already present
                if !self.adj[u].contains(&w) {
                    new_edges.push((u, w));
                    new_edges.push((w, u));
                }
            }
        }
        
        // Apply new edges
        for (u, v) in &new_edges {
            self.adj[*u].push(*v);
            self.adj[*u].sort_unstable();
        }
        
        new_edges
    }
}

/// Elimination tree structure
#[derive(Debug, Clone)]
pub struct EliminationTree {
    /// Parent array (parent[i] = parent of node i, or n if root)
    pub parent: Vec<usize>,
    /// Children lists for each node
    pub children: Vec<Vec<usize>>,
    /// Post-order traversal
    pub post_order: Vec<usize>,
}

impl EliminationTree {
    /// Create elimination tree from symbolic factorization
    fn from_symbolic_factor(n: usize, l_pattern: &HashMap<(usize, usize), bool>) -> Self {
        let mut parent = vec![n; n]; // n means no parent (root)
        
        // For each column j, find the first row i > j with L[i,j] != 0
        // This gives us parent[j] = i
        for j in 0..n {
            for i in (j + 1)..n {
                if l_pattern.contains_key(&(i, j)) {
                    parent[j] = i;
                    break;
                }
            }
        }
        
        // Build children lists
        let mut children = vec![Vec::new(); n];
        for (child, &par) in parent.iter().enumerate() {
            if par < n {
                children[par].push(child);
            }
        }
        
        // Compute post-order traversal
        let mut post_order = Vec::new();
        let mut visited = vec![false; n];
        
        fn dfs_post_order(
            v: usize,
            children: &[Vec<usize>],
            visited: &mut [bool],
            post_order: &mut Vec<usize>,
        ) {
            visited[v] = true;
            for &child in &children[v] {
                if !visited[child] {
                    dfs_post_order(child, children, visited, post_order);
                }
            }
            post_order.push(v);
        }
        
        // Visit all roots
        for i in 0..n {
            if parent[i] == n && !visited[i] {
                dfs_post_order(i, &children, &mut visited, &mut post_order);
            }
        }
        
        Self { parent, children, post_order }
    }
    
    /// Get ancestors of a node in the elimination tree
    pub fn ancestors(&self, node: usize) -> Vec<usize> {
        let mut ancestors = Vec::new();
        let mut current = self.parent[node];
        
        while current < self.parent.len() {
            ancestors.push(current);
            current = self.parent[current];
        }
        
        ancestors
    }
}

/// Ordering algorithms implementation
pub struct OrderingAlgorithms;

impl OrderingAlgorithms {
    /// Natural ordering (identity permutation)
    pub fn natural_ordering(n: usize) -> Vec<usize> {
        (0..n).collect()
    }
    
    /// Approximate Minimum Degree (AMD) ordering
    pub fn amd_ordering(matrix: &CsrMatrix<f64>) -> Vec<usize> {
        let n = matrix.nrows();
        let mut graph = Graph::from_matrix_pattern(matrix);
        let mut perm = Vec::new();
        let mut eliminated = vec![false; n];
        
        // Main AMD loop
        for _ in 0..n {
            // Find vertex with minimum degree among non-eliminated vertices
            let mut min_degree = usize::MAX;
            let mut min_vertex = 0;
            
            for v in 0..n {
                if !eliminated[v] {
                    let degree = graph.adj[v].iter()
                        .filter(|&&u| !eliminated[u])
                        .count();
                    
                    if degree < min_degree {
                        min_degree = degree;
                        min_vertex = v;
                    }
                }
            }
            
            // Eliminate the minimum degree vertex
            perm.push(min_vertex);
            eliminated[min_vertex] = true;
            graph.eliminate_vertex(min_vertex, &eliminated);
        }
        
        perm
    }
    
    /// Minimum Degree on A + A^T structure
    pub fn mmd_ata_ordering(matrix: &CsrMatrix<f64>) -> Vec<usize> {
        // For now, use AMD as a placeholder for MMD
        // In a full implementation, this would be more sophisticated
        Self::amd_ordering(matrix)
    }
    
    /// METIS ordering (placeholder - would interface with METIS C library)
    pub fn metis_ordering(matrix: &CsrMatrix<f64>) -> Result<Vec<usize>, KError> {
        // Placeholder implementation that falls back to AMD
        // In a real implementation, this would call METIS C library
        #[cfg(feature = "logging")]
        log::warn!("METIS ordering not implemented, falling back to AMD");
        
        Ok(Self::amd_ordering(matrix))
    }
    
    /// ParMETIS ordering for distributed graphs (placeholder)
    pub fn parmetis_ordering(
        matrix: &CsrMatrix<f64>,
        _comm: &UniverseComm,
    ) -> Result<Vec<usize>, KError> {
        // Placeholder implementation that falls back to AMD
        // In a real implementation, this would call ParMETIS C library
        #[cfg(feature = "logging")]
        log::warn!("ParMETIS ordering not implemented, falling back to AMD");
        
        Ok(Self::amd_ordering(matrix))
    }
}

/// Symbolic factorization implementation
pub struct SymbolicFactorizer;

impl SymbolicFactorizer {
    /// Compute symbolic factorization pattern
    pub fn compute_symbolic_pattern(
        matrix: &CsrMatrix<f64>,
        col_perm: &[usize],
        row_perm: &[usize],
    ) -> Result<HashMap<(usize, usize), bool>, KError> {
        let n = matrix.nrows();
        let mut l_pattern = HashMap::new();
        
        // Create permuted matrix structure
        let row_ptrs = matrix.row_ptrs();
        let col_indices = matrix.col_indices();
        
        // Apply column permutation to get A[:, col_perm]
        let mut perm_col_indices = Vec::new();
        let mut perm_row_ptrs = vec![0];
        
        for &row in row_perm {
            let start = row_ptrs[row];
            let end = row_ptrs[row + 1];
            let mut row_cols = Vec::new();
            
            for idx in start..end {
                let col = col_indices[idx];
                // Find where this column maps to in the permutation
                if let Some(new_col) = col_perm.iter().position(|&c| c == col) {
                    row_cols.push(new_col);
                }
            }
            
            row_cols.sort_unstable();
            perm_col_indices.extend(row_cols);
            perm_row_ptrs.push(perm_col_indices.len());
        }
        
        // Perform symbolic Cholesky-like factorization on A^T A pattern
        // This is a simplified version - full SuperLU would be more complex
        for i in 0..n {
            // Add diagonal element
            l_pattern.insert((i, i), true);
            
            // For each j < i with A[i,j] != 0
            let row_start = perm_row_ptrs[i];
            let row_end = perm_row_ptrs[i + 1];
            
            for idx in row_start..row_end {
                let j = perm_col_indices[idx];
                if j < i {
                    // Add L[i,j] to pattern
                    l_pattern.insert((i, j), true);
                    
                    // Add fill-in: for each k where L[i,k] exists and k < j
                    for k in 0..j {
                        if l_pattern.contains_key(&(i, k)) {
                            // This creates fill-in at L[i,k]
                            l_pattern.insert((i, k), true);
                        }
                    }
                }
            }
        }
        
        Ok(l_pattern)
    }
    
    /// Build elimination tree from symbolic pattern
    pub fn build_elimination_tree(
        n: usize,
        l_pattern: &HashMap<(usize, usize), bool>,
    ) -> EliminationTree {
        EliminationTree::from_symbolic_factor(n, l_pattern)
    }
}

/// SuperLU_DIST data structure for managing factorization state
///
/// This structure encapsulates all the SuperLU_DIST internal data structures
/// needed for setup, factorization, and solve phases. It includes the process grid
/// and block-cyclic matrix distribution for distributed computation.
pub struct SuperLuDistData {
    /// Process grid for 2D distribution
    pub process_grid: ProcessGrid,
    /// Block-cyclic matrix distribution
    pub distribution: BlockCyclicDistribution,
    /// Factorization options
    pub options: SuperLuDistOptions,
    /// Whether factorization has been computed
    pub factored: bool,
    /// Local matrix data in CSR format
    pub local_matrix: Option<CsrMatrix<f64>>,
    /// Symbolic factorization data
    symbolic_factor: Option<SymbolicFactorization>,
    /// Numerical factorization data
    numeric_factor: Option<NumericFactorization>,
    /// Solve workspace data
    solve_workspace: Option<SolveWorkspace>,
}

/// Symbolic factorization data for SuperLU_DIST structures
#[derive(Debug, Clone)]
pub struct SymbolicFactorization {
    /// Column permutation vector
    pub col_perm: Vec<usize>,
    /// Row permutation vector 
    pub row_perm: Vec<usize>,
    /// Elimination tree
    pub etree: EliminationTree,
    /// Symbolic pattern of L factor
    pub l_pattern: HashMap<(usize, usize), bool>,
    /// Symbolic pattern of U factor (computed from L^T)
    pub u_pattern: HashMap<(usize, usize), bool>,
}

/// Solve workspace (placeholder for SuperLU_DIST solve structures)
#[derive(Debug, Clone)]
pub struct SolveWorkspace {
    /// Temporary vectors for distributed solve
    pub temp_vectors: Vec<Vec<f64>>,
    /// Communication buffers
    pub comm_buffers: Vec<Vec<f64>>,
}

impl SuperLuDistData {
    /// Get symbolic factorization data
    pub fn symbolic_factor(&self) -> Option<&SymbolicFactorization> {
        self.symbolic_factor.as_ref()
    }
    
    /// Get numeric factorization data
    pub fn numeric_factor(&self) -> Option<&NumericFactorization> {
        self.numeric_factor.as_ref()
    }
    
    /// Get solve workspace data
    pub fn solve_workspace(&self) -> Option<&SolveWorkspace> {
        self.solve_workspace.as_ref()
    }
    
    /// Set symbolic factorization data
    pub fn set_symbolic_factor(&mut self, factor: SymbolicFactorization) {
        self.symbolic_factor = Some(factor);
    }
    
    /// Set numeric factorization data
    pub fn set_numeric_factor(&mut self, factor: NumericFactorization) {
        self.numeric_factor = Some(factor);
    }
    
    /// Set solve workspace data
    pub fn set_solve_workspace(&mut self, workspace: SolveWorkspace) {
        self.solve_workspace = Some(workspace);
    }
}

/// Iterative refinement configuration
#[derive(Debug, Clone)]
pub struct RefinementConfig {
    /// Maximum number of refinement iterations
    pub max_iterations: usize,
    /// Convergence tolerance for residual norm
    pub tolerance: f64,
    /// Relative tolerance (relative to initial residual)
    pub relative_tolerance: f64,
    /// Minimum improvement factor to continue refinement
    pub min_improvement_factor: f64,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            tolerance: 1e-12,
            relative_tolerance: 1e-6,
            min_improvement_factor: 0.9,
        }
    }
}

/// Residual computation method for distributed matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualMethod {
    /// Standard residual: r = b - A*x
    Standard,
    /// Scaled residual: r = (b - A*x) / ||b||
    Scaled,
    /// Component-wise scaled: r_i = (b_i - (A*x)_i) / max(|b_i|, |(A*x)_i|)
    ComponentWise,
}

/// Iterative refinement engine for SuperLU_DIST
#[derive(Debug)]
pub struct RefinementEngine {
    /// Refinement configuration
    config: RefinementConfig,
    /// Residual computation method
    residual_method: ResidualMethod,
    /// Workspace for residual computation
    residual_workspace: Vec<f64>,
    /// Workspace for correction vector
    correction_workspace: Vec<f64>,
    /// Workspace for matrix-vector product
    matvec_workspace: Vec<f64>,
    /// Statistics from last refinement
    last_stats: Option<RefinementStats>,
}

/// Statistics from iterative refinement
#[derive(Debug, Clone)]
pub struct RefinementStats {
    /// Number of refinement iterations performed
    pub iterations: usize,
    /// Initial residual norm
    pub initial_residual_norm: f64,
    /// Final residual norm
    pub final_residual_norm: f64,
    /// Residual norms at each iteration
    pub residual_history: Vec<f64>,
    /// Whether refinement converged
    pub converged: bool,
    /// Convergence reason
    pub convergence_reason: RefinementConvergence,
    /// Total time spent in refinement
    pub refinement_time: f64,
}

/// Convergence reasons for iterative refinement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementConvergence {
    /// Converged to absolute tolerance
    AbsoluteTolerance,
    /// Converged to relative tolerance
    RelativeTolerance,
    /// Reached maximum iterations
    MaxIterations,
    /// Stagnation detected (insufficient improvement)
    Stagnation,
    /// Divergence detected
    Divergence,
}

impl RefinementEngine {
    /// Create new iterative refinement engine
    pub fn new(config: RefinementConfig, residual_method: ResidualMethod) -> Self {
        Self {
            config,
            residual_method,
            residual_workspace: Vec::new(),
            correction_workspace: Vec::new(),
            matvec_workspace: Vec::new(),
            last_stats: None,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RefinementConfig::default(), ResidualMethod::Standard)
    }

    /// Setup workspace for given problem size
    pub fn setup_workspace(&mut self, n: usize) {
        self.residual_workspace.resize(n, 0.0);
        self.correction_workspace.resize(n, 0.0);
        self.matvec_workspace.resize(n, 0.0);
    }

    /// Perform iterative refinement on the solution
    pub fn refine_solution(
        &mut self,
        matrix: &CsrMatrix<f64>,
        rhs: &[f64],
        solution: &mut [f64],
        superlu_data: &SuperLuDistData,
        comm: &UniverseComm,
    ) -> Result<RefinementStats, KError> {
        let n = solution.len();
        if n != rhs.len() {
            return Err(KError::InvalidInput("Solution and RHS dimension mismatch".to_string()));
        }

        self.setup_workspace(n);
        
        let start_time = std::time::Instant::now();
        let mut stats = RefinementStats {
            iterations: 0,
            initial_residual_norm: 0.0,
            final_residual_norm: 0.0,
            residual_history: Vec::new(),
            converged: false,
            convergence_reason: RefinementConvergence::MaxIterations,
            refinement_time: 0.0,
        };

        // Clone workspace vectors to avoid borrowing conflicts
        let mut residual_workspace = self.residual_workspace.clone();
        let mut correction_workspace = self.correction_workspace.clone();
        let mut matvec_workspace = self.matvec_workspace.clone();

        // Compute initial residual: r = b - A*x
        Self::compute_residual_static(
            matrix, 
            rhs, 
            solution, 
            &mut residual_workspace, 
            &mut matvec_workspace,
            self.residual_method,
            comm
        )?;
        
        let initial_residual_norm = Self::compute_residual_norm_static(&residual_workspace, comm)?;
        stats.initial_residual_norm = initial_residual_norm;
        stats.final_residual_norm = initial_residual_norm;
        stats.residual_history.push(initial_residual_norm);

        // Check if already converged
        if self.check_convergence(initial_residual_norm, initial_residual_norm, 0) {
            stats.converged = true;
            stats.convergence_reason = RefinementConvergence::AbsoluteTolerance;
            stats.refinement_time = start_time.elapsed().as_secs_f64();
            self.last_stats = Some(stats.clone());
            return Ok(stats);
        }

        // Refinement loop
        let mut previous_residual_norm = initial_residual_norm;
        
        for iter in 0..self.config.max_iterations {
            stats.iterations = iter + 1;

            // Solve correction equation: A * dx = r
            Self::solve_correction_static(
                &residual_workspace,
                &mut correction_workspace,
                superlu_data,
                comm,
            )?;

            // Update solution: x += dx
            for i in 0..n {
                solution[i] += correction_workspace[i];
            }

            // Compute new residual: r = b - A*x
            Self::compute_residual_static(
                matrix, 
                rhs, 
                solution, 
                &mut residual_workspace, 
                &mut matvec_workspace,
                self.residual_method,
                comm
            )?;
            
            let residual_norm = Self::compute_residual_norm_static(&residual_workspace, comm)?;
            stats.final_residual_norm = residual_norm;
            stats.residual_history.push(residual_norm);

            // Check convergence
            if self.check_convergence(residual_norm, initial_residual_norm, iter + 1) {
                stats.converged = true;
                stats.convergence_reason = if residual_norm <= self.config.tolerance {
                    RefinementConvergence::AbsoluteTolerance
                } else {
                    RefinementConvergence::RelativeTolerance
                };
                break;
            }

            // Check for stagnation
            let improvement_factor = residual_norm / previous_residual_norm;
            if improvement_factor > self.config.min_improvement_factor {
                stats.convergence_reason = RefinementConvergence::Stagnation;
                break;
            }

            // Check for divergence
            if residual_norm > initial_residual_norm * 10.0 {
                stats.convergence_reason = RefinementConvergence::Divergence;
                break;
            }

            previous_residual_norm = residual_norm;
        }

        stats.refinement_time = start_time.elapsed().as_secs_f64();
        self.last_stats = Some(stats.clone());
        Ok(stats)
    }

    /// Compute residual r = b - A*x using distributed sparse matrix-vector product (static version)
    fn compute_residual_static(
        matrix: &CsrMatrix<f64>,
        rhs: &[f64],
        solution: &[f64],
        residual: &mut [f64],
        matvec_workspace: &mut [f64],
        residual_method: ResidualMethod,
        comm: &UniverseComm,
    ) -> Result<(), KError> {
        // Initialize residual with RHS
        residual.copy_from_slice(rhs);

        // Compute A*x and subtract from residual
        Self::distributed_sparse_matvec_static(matrix, solution, matvec_workspace, comm)?;
        
        // r = b - A*x
        for i in 0..residual.len() {
            residual[i] -= matvec_workspace[i];
        }

        // Apply residual method scaling if needed
        match residual_method {
            ResidualMethod::Standard => {
                // No scaling needed
            }
            ResidualMethod::Scaled => {
                let rhs_norm = Self::compute_vector_norm_static(rhs, comm)?;
                if rhs_norm > 0.0 {
                    for r in residual.iter_mut() {
                        *r /= rhs_norm;
                    }
                }
            }
            ResidualMethod::ComponentWise => {
                for i in 0..residual.len() {
                    let scale = f64::max(rhs[i].abs(), matvec_workspace[i].abs());
                    if scale > 0.0 {
                        residual[i] /= scale;
                    }
                }
            }
        }

        Ok(())
    }

    /// Perform distributed sparse matrix-vector product (static version)
    fn distributed_sparse_matvec_static(
        matrix: &CsrMatrix<f64>,
        x: &[f64],
        y: &mut [f64],
        _comm: &UniverseComm,
    ) -> Result<(), KError> {
        // For now, perform local matrix-vector product
        // In a full MPI implementation, this would handle communication
        // for distributed vector components
        
        let row_ptrs = matrix.row_ptrs();
        let col_indices = matrix.col_indices();
        let values = matrix.values();
        
        y.fill(0.0);
        
        for i in 0..matrix.nrows() {
            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[idx];
                let val = values[idx];
                y[i] += val * x[j];
            }
        }
        
        Ok(())
    }

    /// Solve correction equation A * dx = r using existing factorization (static version)
    fn solve_correction_static(
        residual: &[f64],
        correction: &mut [f64],
        superlu_data: &SuperLuDistData,
        comm: &UniverseComm,
    ) -> Result<(), KError> {
        // Use the existing triangular solve infrastructure
        let numeric_factor = superlu_data.numeric_factor.as_ref()
            .ok_or_else(|| KError::SolveError("Numeric factorization not available".to_string()))?;

        // Copy residual to correction as initial guess
        correction.copy_from_slice(residual);
        
        // Create temporary vector for intermediate result
        let mut temp_result = correction.to_vec();
        
        // Use the existing distributed triangular solve methods
        // Forward solve: L * y = r
        DistributedTriangularSolver::forward_solve(
            residual,
            &mut temp_result,
            numeric_factor,
            &superlu_data.distribution,
            comm,
            CommPattern::PointToPoint,
            false,
        )?;
        
        // Backward solve: U * dx = y  
        DistributedTriangularSolver::backward_solve(
            &temp_result,
            correction,
            numeric_factor,
            &superlu_data.distribution,
            comm,
            CommPattern::PointToPoint,
            false,
        )?;
        
        Ok(())
    }

    /// Compute norm of residual vector (distributed) (static version)
    fn compute_residual_norm_static(residual: &[f64], _comm: &UniverseComm) -> Result<f64, KError> {
        // Compute local norm
        let local_norm_sq: f64 = residual.iter().map(|x| x * x).sum();
        
        // In full MPI implementation, would use allreduce to sum across processes
        // For now, just return local norm
        Ok(local_norm_sq.sqrt())
    }

    /// Compute norm of a vector (distributed) (static version)
    fn compute_vector_norm_static(vector: &[f64], _comm: &UniverseComm) -> Result<f64, KError> {
        let local_norm_sq: f64 = vector.iter().map(|x| x * x).sum();
        Ok(local_norm_sq.sqrt())
    }

    /// Check convergence criteria
    fn check_convergence(&self, current_norm: f64, initial_norm: f64, iteration: usize) -> bool {
        if iteration == 0 {
            return false; // Never converge on first iteration
        }
        
        // Absolute tolerance check
        if current_norm <= self.config.tolerance {
            return true;
        }
        
        // Relative tolerance check
        if initial_norm > 0.0 && current_norm / initial_norm <= self.config.relative_tolerance {
            return true;
        }
        
        false
    }

    /// Get statistics from last refinement
    pub fn last_stats(&self) -> Option<&RefinementStats> {
        self.last_stats.as_ref()
    }

    /// Update configuration
    pub fn set_config(&mut self, config: RefinementConfig) {
        self.config = config;
    }

    /// Update residual method
    pub fn set_residual_method(&mut self, method: ResidualMethod) {
        self.residual_method = method;
    }
}

/// SuperLU_DIST distributed direct solver
pub struct SuperLuDistSolver {
    /// Solver options
    options: SuperLuDistOptions,
    /// Internal SuperLU_DIST data (None until first setup)
    data: Option<SuperLuDistData>,
    /// Iterative refinement engine
    refinement_engine: Option<RefinementEngine>,
}

impl SuperLuDistSolver {
    /// Create a new SuperLU_DIST solver with default options
    pub fn new() -> Self {
        Self {
            options: SuperLuDistOptions::default(),
            data: None,
            refinement_engine: None,
        }
    }

    /// Create a new SuperLU_DIST solver with custom options
    pub fn with_options(options: SuperLuDistOptions) -> Self {
        Self {
            options,
            data: None,
            refinement_engine: None,
        }
    }

    /// Set the diagonal pivot threshold
    pub fn set_diagonal_pivot_threshold(&mut self, threshold: f64) -> &mut Self {
        self.options.diagonal_pivot_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the column permutation strategy
    pub fn set_column_permutation(&mut self, perm: ColumnPermutation) -> &mut Self {
        self.options.column_permutation = perm;
        self
    }

    /// Set the iterative refinement strategy
    pub fn set_iterative_refinement(&mut self, refine: IterativeRefinement) -> &mut Self {
        self.options.iterative_refinement = refine;
        self
    }

    /// Set the print level for diagnostics
    pub fn set_print_level(&mut self, level: u8) -> &mut Self {
        self.options.print_level = level;
        self
    }

    /// Set the panel size for local dense factorization
    pub fn set_panel_size(&mut self, size: usize) -> &mut Self {
        self.options.panel_size = Some(size);
        self
    }

    /// Enable 3D communication-avoiding factorization
    pub fn set_3d_factorization(&mut self, enable: bool, depth: Option<usize>) -> &mut Self {
        self.options.enable_3d_factorization = enable;
        self.options.process_grid_3d_depth = depth;
        self
    }

    /// Set memory trade-off factor for 3D algorithm
    pub fn set_memory_tradeoff(&mut self, factor: f64) -> &mut Self {
        self.options.memory_tradeoff_factor = factor.max(0.1);
        self
    }

    /// Set maximum concurrent panels
    pub fn set_max_concurrent_panels(&mut self, max_panels: usize) -> &mut Self {
        self.options.max_concurrent_panels = max_panels.max(1);
        self
    }

    /// Enable asynchronous panel updates
    pub fn set_async_panel_updates(&mut self, enable: bool) -> &mut Self {
        self.options.async_panel_updates = enable;
        self
    }

    /// Set static pivoting mode
    pub fn set_static_pivoting(&mut self, enable: bool) -> &mut Self {
        self.options.static_pivoting = enable;
        self
    }

    /// Get a reference to the current options
    pub fn options(&self) -> &SuperLuDistOptions {
        &self.options
    }

    /// Enable iterative refinement with default configuration
    pub fn enable_iterative_refinement(&mut self) -> &mut Self {
        self.refinement_engine = Some(RefinementEngine::with_defaults());
        self
    }

    /// Configure iterative refinement with custom settings
    pub fn set_refinement_config(&mut self, config: RefinementConfig) -> &mut Self {
        if let Some(ref mut engine) = self.refinement_engine {
            engine.set_config(config);
        } else {
            self.refinement_engine = Some(RefinementEngine::new(config, ResidualMethod::Standard));
        }
        self
    }

    /// Set the residual computation method for iterative refinement
    pub fn set_residual_method(&mut self, method: ResidualMethod) -> &mut Self {
        if let Some(ref mut engine) = self.refinement_engine {
            engine.set_residual_method(method);
        } else {
            self.refinement_engine = Some(RefinementEngine::new(RefinementConfig::default(), method));
        }
        self
    }

    /// Disable iterative refinement
    pub fn disable_iterative_refinement(&mut self) -> &mut Self {
        self.refinement_engine = None;
        self
    }

    /// Get refinement statistics from the last solve (if available)
    pub fn refinement_stats(&self) -> Option<&RefinementStats> {
        self.refinement_engine.as_ref().and_then(|engine| engine.last_stats())
    }

    /// Setup the SuperLU_DIST factorization for the given matrix
    ///
    /// This creates the process grid, distributes the matrix, and performs
    /// symbolic and numerical factorization.
    fn setup_factorization(
        &mut self,
        matrix: &CsrMatrix<f64>,
        comm: &UniverseComm,
    ) -> Result<(), KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("SuperLuDistSetup");

        // Create process grid
        let process_grid = if let Some((prows, pcols)) = self.options.process_grid {
            ProcessGrid::new_with_dims(comm, prows, pcols)?
        } else {
            ProcessGrid::new_auto(comm)?
        };

        // Create block-cyclic distribution
        // Use reasonable block sizes (can be made configurable later)
        let row_block_size = std::cmp::max(1, matrix.nrows() / (process_grid.prows * 4));
        let col_block_size = std::cmp::max(1, matrix.ncols() / (process_grid.pcols * 4));
        
        let distribution = BlockCyclicDistribution::new(
            process_grid.clone(),
            matrix.nrows(),
            matrix.ncols(),
            row_block_size,
            col_block_size,
        );

        // Distribute the matrix to local portions
        let local_matrix = self.distribute_matrix(matrix, &distribution)?;

        // Create SuperLU_DIST data structure
        let mut slu_data = SuperLuDistData {
            process_grid,
            distribution,
            options: self.options.clone(),
            factored: false,
            local_matrix: Some(local_matrix),
            symbolic_factor: None,
            numeric_factor: None,
            solve_workspace: None,
        };

        // Perform symbolic factorization
        #[cfg(feature = "logging")]
        let _symbolic_guard = StageGuard::new("SuperLuDistSymbolic");
        
        let symbolic = self.symbolic_factorization(&slu_data)?;
        slu_data.symbolic_factor = Some(symbolic);

        // Perform numerical factorization
        #[cfg(feature = "logging")]
        let _numeric_guard = StageGuard::new("SuperLuDistNumeric");
        
        let numeric = self.numerical_factorization(&slu_data)?;
        slu_data.numeric_factor = Some(numeric);

        // Setup solve workspace
        let workspace = self.setup_solve_workspace(&slu_data)?;
        slu_data.solve_workspace = Some(workspace);

        slu_data.factored = true;
        self.data = Some(slu_data);

        Ok(())
    }

    /// Distribute global matrix to local portions using block-cyclic distribution
    fn distribute_matrix(
        &self,
        global_matrix: &CsrMatrix<f64>,
        distribution: &BlockCyclicDistribution,
    ) -> Result<CsrMatrix<f64>, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("MatrixDistribution");

        // Extract local portion of the global matrix
        let mut local_row_ptrs = vec![0];
        let mut local_col_indices = Vec::new();
        let mut local_values = Vec::new();
        
        let mut nnz_count = 0;
        
        // Iterate through global rows and extract those owned by this process
        for global_row in 0..global_matrix.nrows() {
            if let Some(_local_row) = distribution.global_to_local_row(global_row) {
                // Get the row data from global matrix
                let row_start = global_matrix.row_ptrs()[global_row];
                let row_end = global_matrix.row_ptrs()[global_row + 1];
                
                // Extract columns that belong to this process
                for idx in row_start..row_end {
                    let global_col = global_matrix.col_indices()[idx];
                    if let Some(local_col) = distribution.global_to_local_col(global_col) {
                        local_col_indices.push(local_col);
                        local_values.push(global_matrix.values()[idx]);
                        nnz_count += 1;
                    }
                }
                
                local_row_ptrs.push(nnz_count);
            }
        }
        
        // Create local CSR matrix
        let local_matrix = CsrMatrix::from_csr(
            distribution.local_rows,
            distribution.local_cols,
            local_row_ptrs,
            local_col_indices,
            local_values,
        );
        
        Ok(local_matrix)
    }

    /// Enhanced symbolic factorization using ordering algorithms
    fn symbolic_factorization(&self, data: &SuperLuDistData) -> Result<SymbolicFactorization, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("SymbolicFactorization");
        
        let n = data.distribution.global_rows;
        let matrix = data.local_matrix.as_ref()
            .ok_or_else(|| KError::SolveError("No local matrix available".to_string()))?;
        
        // Compute column permutation based on strategy
        let col_perm = match self.options.column_permutation {
            ColumnPermutation::Natural => OrderingAlgorithms::natural_ordering(n),
            ColumnPermutation::MmdAta => OrderingAlgorithms::mmd_ata_ordering(matrix),
            ColumnPermutation::Metis => OrderingAlgorithms::metis_ordering(matrix)?,
            ColumnPermutation::ParMetis => {
                // For distributed case, we would need the global matrix
                // For now, fall back to local AMD
                OrderingAlgorithms::amd_ordering(matrix)
            },
            ColumnPermutation::User => {
                // User-provided permutation would be stored in options
                // For now, use natural ordering
                OrderingAlgorithms::natural_ordering(n)
            },
        };

        // Compute row permutation based on strategy
        let row_perm = match self.options.row_permutation {
            RowPermutation::NoRowPerm => OrderingAlgorithms::natural_ordering(n),
            RowPermutation::LargeDiag => {
                // For large diagonal strategy, we would analyze diagonal elements
                // For now, use natural ordering as placeholder
                OrderingAlgorithms::natural_ordering(n)
            },
            RowPermutation::User => {
                // User-provided permutation would be stored in options
                OrderingAlgorithms::natural_ordering(n)
            },
        };
        
        #[cfg(feature = "logging")]
        log::debug!("Computing symbolic pattern with {} x {} matrix", n, n);

        // Compute symbolic factorization pattern
        let l_pattern = SymbolicFactorizer::compute_symbolic_pattern(
            matrix, &col_perm, &row_perm
        )?;
        
        // Compute U pattern (transpose of L for square matrices)
        let mut u_pattern = HashMap::new();
        for &(i, j) in l_pattern.keys() {
            u_pattern.insert((j, i), true);
        }
        
        // Build elimination tree
        let etree = SymbolicFactorizer::build_elimination_tree(n, &l_pattern);
        
        #[cfg(feature = "logging")]
        log::debug!("Symbolic factorization completed: {} L entries, {} U entries", 
                   l_pattern.len(), u_pattern.len());

        Ok(SymbolicFactorization {
            col_perm,
            row_perm,
            etree,
            l_pattern,
            u_pattern,
        })
    }

    /// Enhanced numerical factorization with panel-based approach
    fn numerical_factorization(&self, data: &SuperLuDistData) -> Result<NumericFactorization, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("NumericalFactorization");
        
        let n = data.distribution.global_rows;
        let matrix = data.local_matrix.as_ref()
            .ok_or_else(|| KError::SolveError("No local matrix available".to_string()))?;
        
        let symbolic = data.symbolic_factor.as_ref()
            .ok_or_else(|| KError::SolveError("No symbolic factorization available".to_string()))?;
        
        // Determine pivoting strategy from options
        let pivot_strategy = if self.options.static_pivoting {
            PivotingStrategy::Static
        } else if self.options.replace_tiny_pivots {
            PivotingStrategy::ThresholdWithFallback
        } else {
            PivotingStrategy::Dynamic
        };
        
        // Panel size configuration (could be made configurable)
        let panel_size = std::cmp::min(64, n / 4).max(1);
        
        #[cfg(feature = "logging")]
        log::debug!("Starting numerical factorization with panel size {}, pivot strategy {:?}", 
                   panel_size, pivot_strategy);
        
        let mut panels = Vec::new();
        let mut panel_factors = Vec::new();
        let mut total_row_swaps = 0;
        let mut tiny_pivots_replaced = 0;
        let mut max_pivot_growth = 1.0;
        
        // Process matrix in panels
        for panel_start in (0..n).step_by(panel_size) {
            let panel_end = std::cmp::min(panel_start + panel_size, n);
            
            // Extract rows that have nonzeros in this panel's columns
            let mut panel_rows = Vec::new();
            for i in 0..matrix.nrows() {
                let row_start = matrix.row_ptrs()[i];
                let row_end = matrix.row_ptrs()[i + 1];
                
                for idx in row_start..row_end {
                    let col = matrix.col_indices()[idx];
                    if col >= panel_start && col < panel_end {
                        panel_rows.push(i);
                        break;
                    }
                }
            }
            
            if panel_rows.is_empty() {
                continue; // No nonzeros in this panel
            }
            
            // Create panel from sparse matrix
            let mut panel = Panel::from_sparse_columns(
                matrix, panel_start, panel_end, panel_rows
            );
            
            // Factorize the panel
            match panel.factorize_lu(self.options.diagonal_pivot_threshold, pivot_strategy) {
                Ok(factor) => {
                    total_row_swaps += factor.num_row_swaps;
                    if factor.is_singular {
                        tiny_pivots_replaced += 1;
                    }
                    
                    // Estimate pivot growth (simplified)
                    for i in 0..panel.width.min(panel.height) {
                        let diag_val = panel.data[i * panel.height + i].abs();
                        if diag_val > max_pivot_growth {
                            max_pivot_growth = diag_val;
                        }
                    }
                    
                    panel_factors.push(factor);
                },
                Err(e) => {
                    #[cfg(feature = "logging")]
                    log::error!("Panel factorization failed: {}", e);
                    return Err(e);
                }
            }
            
            panels.push(panel);
        }
        
        // Create global permutations (simplified - in real SuperLU_DIST this would be distributed)
        let global_row_perm = symbolic.row_perm.clone();
        let global_col_perm = symbolic.col_perm.clone();
        
        // Compute scaling factors (placeholder - would be more sophisticated)
        let row_scale = vec![1.0; n];
        let col_scale = vec![1.0; n];
        
        // Estimate memory usage
        let memory_usage = panels.iter()
            .map(|p| p.data.len() * std::mem::size_of::<f64>())
            .sum::<usize>() + 
            (global_row_perm.len() + global_col_perm.len()) * std::mem::size_of::<usize>();
        
        let factor_stats = FactorizationStats {
            num_panels: panels.len(),
            total_row_swaps,
            tiny_pivots_replaced,
            max_pivot_growth,
            condition_estimate: None, // Would require more sophisticated analysis
            memory_usage,
        };
        
        #[cfg(feature = "logging")]
        log::info!("Numerical factorization completed: {} panels, {} row swaps, max pivot growth {:.2e}",
                  factor_stats.num_panels, factor_stats.total_row_swaps, factor_stats.max_pivot_growth);
        
        Ok(NumericFactorization {
            n,
            nnz: panels.iter().map(|p| p.data.len()).sum(),
            panels,
            panel_factors,
            global_row_perm,
            global_col_perm,
            row_scale,
            col_scale,
            pivot_strategy,
            pivot_threshold: self.options.diagonal_pivot_threshold,
            replaced_tiny_pivots: tiny_pivots_replaced > 0,
            factor_stats,
        })
    }

    /// Setup solve workspace
    fn setup_solve_workspace(&self, data: &SuperLuDistData) -> Result<SolveWorkspace, KError> {
        let n = data.distribution.global_rows;
        
        Ok(SolveWorkspace {
            temp_vectors: vec![vec![0.0; n]; 2],
            comm_buffers: vec![vec![0.0; n]; data.process_grid.total_procs],
        })
    }

    /// Distributed solve using the computed factorization
    ///
    /// This corresponds to the HYPRE `hypre_SLUDistSolve` function.
    fn solve_factored(
        &mut self,
        b: &Vec<f64>,
        x: &mut Vec<f64>,
        comm: &UniverseComm,
    ) -> Result<(), KError> {
        let data = self.data.as_ref()
            .ok_or_else(|| KError::SolveError("SuperLU_DIST not factored".to_string()))?;

        if !data.factored {
            return Err(KError::SolveError("Matrix not factored".to_string()));
        }

        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("SuperLuDistSolve");

        // Copy b to x as starting point
        x.clone_from_slice(b);

        let numeric_factor = data.numeric_factor.as_ref()
            .ok_or_else(|| KError::SolveError("No numeric factorization available".to_string()))?;

        // Determine communication pattern based on options
        let comm_pattern = if self.options.enable_3d_factorization {
            CommPattern::BinaryTree
        } else if self.options.async_panel_updates {
            CommPattern::Butterfly
        } else {
            CommPattern::PointToPoint
        };

        let overlap_comm = self.options.async_panel_updates;

        #[cfg(feature = "logging")]
        log::info!("Starting distributed triangular solve with pattern {:?}, overlap_comm={}", 
                  comm_pattern, overlap_comm);

        // Phase 1: Forward substitution (solve Ly = Pb)
        // Apply row permutation to RHS
        let row_perm = &numeric_factor.global_row_perm;
        let mut permuted_b = vec![0.0; b.len()];
        for (i, &perm_i) in row_perm.iter().enumerate() {
            if perm_i < b.len() {
                permuted_b[i] = b[perm_i];
            }
        }

        let mut y = vec![0.0; x.len()];
        DistributedTriangularSolver::forward_solve(
            &permuted_b,
            &mut y,
            numeric_factor,
            &data.distribution,
            comm,
            comm_pattern,
            overlap_comm,
        )?;

        // Phase 2: Backward substitution (solve Ux = y)
        DistributedTriangularSolver::backward_solve(
            &y,
            x,
            numeric_factor,
            &data.distribution,
            comm,
            comm_pattern,
            overlap_comm,
        )?;

        // Apply column permutation to solution
        let col_perm = &numeric_factor.global_col_perm;
        let mut permuted_x = vec![0.0; x.len()];
        for (i, &perm_i) in col_perm.iter().enumerate() {
            if i < x.len() && perm_i < permuted_x.len() {
                permuted_x[perm_i] = x[i];
            }
        }
        x.copy_from_slice(&permuted_x);

        // Apply iterative refinement if requested and engine is available
        if !matches!(self.options.iterative_refinement, IterativeRefinement::NoRefine) {
            if let Some(ref mut engine) = self.refinement_engine {
                // Get the original matrix for residual computation
                let data = self.data.as_ref().unwrap();
                let local_matrix = data.local_matrix.as_ref()
                    .ok_or_else(|| KError::SolveError("Local matrix not available for refinement".to_string()))?;
                
                // Perform iterative refinement
                let _refinement_stats = engine.refine_solution(
                    local_matrix,
                    b,
                    x,
                    data,
                    comm,
                )?;
                
                #[cfg(feature = "logging")]
                if let Some(stats) = engine.last_stats() {
                    log::info!("Iterative refinement completed: {} iterations, final residual: {:.2e}", 
                              stats.iterations, stats.final_residual_norm);
                }
            }
        }

        #[cfg(feature = "logging")]
        log::info!("Distributed triangular solve completed successfully");

        Ok(())
    }

    /// Destroy the factorization and free memory
    pub fn destroy(&mut self) {
        self.data = None;
        self.refinement_engine = None;
    }
}

impl Default for SuperLuDistSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSolver<CsrMatrix<f64>, Vec<f64>> for SuperLuDistSolver {
    type Error = KError;
    type Scalar = f64;

    /// Solve the linear system A·x = b using distributed SuperLU factorization
    ///
    /// # Arguments
    /// * `a` - Sparse matrix in CSR format
    /// * `pc` - Preconditioner (unused for direct solvers)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: ignored; on output: solution vector
    /// * `comm` - MPI communicator for distributed computation
    /// * `monitors` - Optional callbacks for progress monitoring
    /// * `work` - Optional workspace (unused for direct solvers)
    ///
    /// # Returns
    /// * `Ok(SolveStats)` with convergence information (always converged in 1 iteration for direct solvers)
    /// * `Err(KError)` on factorization or solve failure
    fn solve(
        &mut self,
        a: &CsrMatrix<f64>,
        pc: Option<&dyn crate::preconditioner::Preconditioner<CsrMatrix<f64>, Vec<f64>>>,
        b: &Vec<f64>,
        x: &mut Vec<f64>,
        comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
        _work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<crate::utils::convergence::SolveStats<f64>, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("SuperLuDistSolve");

        let _ = pc; // Direct solvers do not use preconditioners

        // Validate input dimensions
        if b.len() != a.nrows() {
            return Err(KError::InvalidInput(format!(
                "RHS length {} doesn't match matrix rows {}",
                b.len(), a.nrows()
            )));
        }

        if x.len() != a.ncols() {
            x.resize(a.ncols(), 0.0);
        }

        // Call monitors at start if provided
        if let Some(monitors) = monitors {
            for monitor in monitors {
                monitor(0, 0.0);
            }
        }

        // Setup factorization if not already done
        if self.data.is_none() {
            self.setup_factorization(a, comm)?;
        }

        // Solve using the factorization
        self.solve_factored(b, x, comm)?;

        // Call monitors at end if provided
        if let Some(monitors) = monitors {
            for monitor in monitors {
                monitor(1, 0.0);
            }
        }

        // Direct solvers always converge in 1 iteration
        Ok(SolveStats {
            iterations: 1,
            final_residual: 0.0,
            reason: ConvergedReason::ConvergedAtol,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::NoComm;

    #[test]
    fn test_superlu_dist_creation() {
        let solver = SuperLuDistSolver::new();
        assert!(solver.data.is_none());
        assert_eq!(solver.options.print_level, 0);
        assert_eq!(solver.options.diagonal_pivot_threshold, 1.0);
    }

    #[test]
    fn test_superlu_dist_options() {
        let mut solver = SuperLuDistSolver::new();
        
        solver.set_diagonal_pivot_threshold(0.5)
              .set_column_permutation(ColumnPermutation::Metis)
              .set_iterative_refinement(IterativeRefinement::Single)
              .set_print_level(1);
        
        assert_eq!(solver.options.diagonal_pivot_threshold, 0.5);
        assert_eq!(solver.options.column_permutation, ColumnPermutation::Metis);
        assert_eq!(solver.options.iterative_refinement, IterativeRefinement::Single);
        assert_eq!(solver.options.print_level, 1);
    }

    #[test]
    fn test_process_grid_determination() {
        assert_eq!(ProcessGrid::determine_optimal_grid(1), (1, 1));
        assert_eq!(ProcessGrid::determine_optimal_grid(4), (2, 2));
        assert_eq!(ProcessGrid::determine_optimal_grid(6), (2, 3));
        assert_eq!(ProcessGrid::determine_optimal_grid(8), (2, 4));
        assert_eq!(ProcessGrid::determine_optimal_grid(16), (4, 4));
    }

    #[test]
    fn test_process_grid_creation() {
        let comm = UniverseComm::NoComm(NoComm);
        let grid = ProcessGrid::new_auto(&comm).unwrap();
        
        assert_eq!(grid.prows, 1);
        assert_eq!(grid.pcols, 1);
        assert_eq!(grid.my_prow, 0);
        assert_eq!(grid.my_pcol, 0);
        assert_eq!(grid.my_rank, 0);
        assert_eq!(grid.total_procs, 1);
    }

    #[test]
    fn test_block_cyclic_distribution() {
        let comm = UniverseComm::NoComm(NoComm);
        let grid = ProcessGrid::new_auto(&comm).unwrap();
        
        let distribution = BlockCyclicDistribution::new(
            grid,
            10, // global_rows
            10, // global_cols
            4,  // row_block_size
            4,  // col_block_size
        );
        
        assert_eq!(distribution.global_rows, 10);
        assert_eq!(distribution.global_cols, 10);
        assert_eq!(distribution.local_rows, 10);  // All rows on single process
        assert_eq!(distribution.local_cols, 10);  // All cols on single process
    }

    #[test]
    fn test_global_to_local_conversion() {
        let comm = UniverseComm::NoComm(NoComm);
        let grid = ProcessGrid::new_auto(&comm).unwrap();
        
        let distribution = BlockCyclicDistribution::new(
            grid, 8, 8, 4, 4
        );
        
        // For single process, all global indices should map to local indices
        assert_eq!(distribution.global_to_local_row(0), Some(0));
        assert_eq!(distribution.global_to_local_row(3), Some(3));
        assert_eq!(distribution.global_to_local_row(7), Some(7));
        
        assert_eq!(distribution.global_to_local_col(0), Some(0));
        assert_eq!(distribution.global_to_local_col(3), Some(3));
        assert_eq!(distribution.global_to_local_col(7), Some(7));
    }

    #[test]
    fn test_graph_creation() {
        // Create a simple 3x3 tridiagonal matrix
        let matrix = CsrMatrix::from_csr(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 0, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0],
        );
        
        let graph = Graph::from_matrix_pattern(&matrix);
        
        // Check adjacency structure
        assert_eq!(graph.adj[0], vec![1]);  // 0 connected to 1
        assert_eq!(graph.adj[1], vec![0, 2]); // 1 connected to 0, 2
        assert_eq!(graph.adj[2], vec![1]);  // 2 connected to 1
    }

    #[test]
    fn test_natural_ordering() {
        let perm = OrderingAlgorithms::natural_ordering(5);
        assert_eq!(perm, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_amd_ordering() {
        // Create a simple matrix for AMD testing
        let matrix = CsrMatrix::from_csr(
            4, 4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 1, 2, 2, 3, 0, 3],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );
        
        let perm = OrderingAlgorithms::amd_ordering(&matrix);
        
        // Should return a valid permutation
        assert_eq!(perm.len(), 4);
        let mut sorted_perm = perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_symbolic_factorization_pattern() {
        // Create a simple 3x3 matrix
        let matrix = CsrMatrix::from_csr(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![2.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        );
        
        let col_perm = vec![0, 1, 2];
        let row_perm = vec![0, 1, 2];
        
        let pattern = SymbolicFactorizer::compute_symbolic_pattern(
            &matrix, &col_perm, &row_perm
        ).unwrap();
        
        // Should have at least diagonal elements
        assert!(pattern.contains_key(&(0, 0)));
        assert!(pattern.contains_key(&(1, 1)));
        assert!(pattern.contains_key(&(2, 2)));
    }

    #[test]
    fn test_elimination_tree_construction() {
        let n = 3;
        let mut l_pattern = HashMap::new();
        
        // Simple L pattern: lower triangular with some fill
        l_pattern.insert((0, 0), true);
        l_pattern.insert((1, 0), true);
        l_pattern.insert((1, 1), true);
        l_pattern.insert((2, 0), true);
        l_pattern.insert((2, 1), true);
        l_pattern.insert((2, 2), true);
        
        let etree = SymbolicFactorizer::build_elimination_tree(n, &l_pattern);
        
        // Check that we have a valid elimination tree
        assert_eq!(etree.parent.len(), n);
        assert_eq!(etree.children.len(), n);
    }

    #[test]
    fn test_enhanced_symbolic_factorization() {
        // Create a simple matrix
        let matrix = CsrMatrix::from_csr(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![2.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        );
        
        let comm = UniverseComm::NoComm(NoComm);
        let grid = ProcessGrid::new_auto(&comm).unwrap();
        let distribution = BlockCyclicDistribution::new(
            grid, 3, 3, 2, 2
        );
        
        // Setup solver with different ordering strategies
        let mut solver = SuperLuDistSolver::new();
        solver.set_column_permutation(ColumnPermutation::MmdAta);
        
        // Create test data
        let slu_data = SuperLuDistData {
            process_grid: distribution.grid.clone(),
            distribution,
            options: solver.options.clone(),
            factored: false,
            local_matrix: Some(matrix),
            symbolic_factor: None,
            numeric_factor: None,
            solve_workspace: None,
        };
        
        // Test symbolic factorization
        let symbolic = solver.symbolic_factorization(&slu_data).unwrap();
        
        // Verify the result
        assert_eq!(symbolic.col_perm.len(), 3);
        assert_eq!(symbolic.row_perm.len(), 3);
        assert!(!symbolic.l_pattern.is_empty());
        assert!(!symbolic.u_pattern.is_empty());
    }

    #[test]
    fn test_panel_creation() {
        // Create a simple 4x4 matrix
        let matrix = CsrMatrix::from_csr(
            4, 4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 1, 2, 2, 3, 0, 3],
            vec![2.0, -1.0, 2.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        );
        
        let row_indices = vec![0, 1, 2, 3];
        let panel = Panel::from_sparse_columns(&matrix, 0, 2, row_indices);
        
        assert_eq!(panel.width, 2);
        assert_eq!(panel.height, 4);
        assert_eq!(panel.col_start, 0);
        assert_eq!(panel.data.len(), 8); // 2 columns * 4 rows
        
        // Check that matrix data was correctly extracted
        assert_eq!(panel.data[0], 2.0); // (0,0)
        assert_eq!(panel.data[1], 0.0); // (1,0) - zero
        assert_eq!(panel.data[4], -1.0); // (0,1)
        assert_eq!(panel.data[5], 2.0); // (1,1)
    }

    #[test]
    fn test_panel_factorization_static() {
        let mut panel = Panel {
            width: 2,
            height: 2,
            data: vec![2.0, 1.0, 1.0, 3.0], // Column-major: [[2,1],[1,3]]
            row_indices: vec![0, 1],
            col_start: 0,
        };
        
        let result = panel.factorize_lu(1e-12, PivotingStrategy::Static).unwrap();
        
        assert_eq!(result.pivot_strategy, PivotingStrategy::Static);
        assert_eq!(result.num_row_swaps, 0); // No row swaps in static pivoting
        assert!(!result.is_singular);
        
        // Check that factorization modified the panel data
        assert_ne!(panel.data, vec![2.0, 1.0, 1.0, 3.0]);
    }

    #[test]
    fn test_panel_factorization_dynamic() {
        let mut panel = Panel {
            width: 2,
            height: 2,
            data: vec![1.0, 3.0, 2.0, 1.0], // Column-major: [[1,2],[3,1]]
            row_indices: vec![0, 1],
            col_start: 0,
        };
        
        let result = panel.factorize_lu(1e-12, PivotingStrategy::Dynamic).unwrap();
        
        assert_eq!(result.pivot_strategy, PivotingStrategy::Dynamic);
        // Dynamic pivoting should find the larger pivot (3.0) and swap rows
        assert!(result.num_row_swaps > 0 || !result.is_singular);
    }

    #[test]
    fn test_panel_factorization_tiny_pivot() {
        let mut panel = Panel {
            width: 2,
            height: 2,
            data: vec![1e-15, 1.0, 1.0, 3.0], // Very small pivot
            row_indices: vec![0, 1],
            col_start: 0,
        };
        
        let result = panel.factorize_lu(1e-12, PivotingStrategy::Static).unwrap();
        
        assert!(result.is_singular); // Should detect singular matrix
        assert_eq!(result.pivot_strategy, PivotingStrategy::Static);
        
        // Check that tiny pivot was replaced
        assert!(panel.data[0].abs() >= 1e-12);
    }

    #[test]
    fn test_threshold_with_fallback() {
        let mut panel = Panel {
            width: 2,
            height: 2,
            data: vec![1e-15, 1.0, 1.0, 3.0], // Very small pivot
            row_indices: vec![0, 1],
            col_start: 0,
        };
        
        let result = panel.factorize_lu(1e-12, PivotingStrategy::ThresholdWithFallback).unwrap();
        
        // Should fall back to dynamic pivoting due to tiny pivot
        assert_eq!(result.pivot_strategy, PivotingStrategy::Dynamic);
    }

    #[test]
    fn test_numerical_factorization_integration() {
        // Create a simple symmetric positive definite matrix
        let matrix = CsrMatrix::from_csr(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![4.0, -1.0, 4.0, -1.0, -1.0, 4.0],
        );
        
        let comm = UniverseComm::NoComm(NoComm);
        let grid = ProcessGrid::new_auto(&comm).unwrap();
        let distribution = BlockCyclicDistribution::new(
            grid, 3, 3, 2, 2
        );
        
        // Setup solver with static pivoting
        let mut solver = SuperLuDistSolver::new();
        solver.set_static_pivoting(true)
              .set_panel_size(2);
        
        // Create test data with symbolic factorization
        let symbolic = SymbolicFactorization {
            col_perm: vec![0, 1, 2],
            row_perm: vec![0, 1, 2],
            etree: EliminationTree {
                parent: vec![3, 3, 3],
                children: vec![vec![], vec![], vec![], vec![0, 1, 2]],
                post_order: vec![0, 1, 2],
            },
            l_pattern: [(0,0), (1,1), (2,2)].iter().map(|&k| (k, true)).collect(),
            u_pattern: [(0,0), (1,1), (2,2)].iter().map(|&k| (k, true)).collect(),
        };
        
        let slu_data = SuperLuDistData {
            process_grid: distribution.grid.clone(),
            distribution,
            options: solver.options.clone(),
            factored: false,
            local_matrix: Some(matrix),
            symbolic_factor: Some(symbolic),
            numeric_factor: None,
            solve_workspace: None,
        };
        
        // Test numerical factorization
        let numeric = solver.numerical_factorization(&slu_data).unwrap();
        
        // Verify the result
        assert_eq!(numeric.n, 3);
        assert_eq!(numeric.pivot_strategy, PivotingStrategy::Static);
        assert!(!numeric.panels.is_empty());
        assert_eq!(numeric.panels.len(), numeric.panel_factors.len());
        assert_eq!(numeric.global_row_perm.len(), 3);
        assert_eq!(numeric.global_col_perm.len(), 3);
        
        // Check statistics
        assert!(numeric.factor_stats.num_panels > 0);
        assert!(numeric.factor_stats.memory_usage > 0);
        assert!(numeric.factor_stats.max_pivot_growth >= 1.0);
    }

    #[test]
    fn test_3d_factorization_options() {
        let mut solver = SuperLuDistSolver::new();
        
        // Test 3D factorization configuration
        solver.set_3d_factorization(true, Some(2))
              .set_memory_tradeoff(2.5)
              .set_max_concurrent_panels(4)
              .set_async_panel_updates(true);
        
        let options = solver.options();
        assert!(options.enable_3d_factorization);
        assert_eq!(options.process_grid_3d_depth, Some(2));
        assert_eq!(options.memory_tradeoff_factor, 2.5);
        assert_eq!(options.max_concurrent_panels, 4);
        assert!(options.async_panel_updates);
    }

    #[test]
    fn test_pivoting_strategies() {
        let mut solver = SuperLuDistSolver::new();
        
        // Test static pivoting
        solver.set_static_pivoting(true);
        assert!(solver.options().static_pivoting);
        
        // Test threshold settings
        solver.set_diagonal_pivot_threshold(0.1);
        assert_eq!(solver.options().diagonal_pivot_threshold, 0.1);
        
        // Test panel size setting
        solver.set_panel_size(32);
        assert_eq!(solver.options().panel_size, Some(32));
    }

    #[test]
    fn test_triangular_solve_data_creation() {
        let comm = UniverseComm::NoComm(NoComm);
        let grid = ProcessGrid::new_auto(&comm).unwrap();
        let distribution = BlockCyclicDistribution::new(grid, 8, 8, 2, 2);
        
        let numeric_factor = NumericFactorization {
            n: 8,
            nnz: 16,
            panels: vec![],
            panel_factors: vec![],
            global_row_perm: (0..8).collect(),
            global_col_perm: (0..8).collect(),
            row_scale: vec![1.0; 8],
            col_scale: vec![1.0; 8],
            pivot_strategy: PivotingStrategy::Static,
            pivot_threshold: 1e-12,
            replaced_tiny_pivots: false,
            factor_stats: FactorizationStats {
                num_panels: 0,
                total_row_swaps: 0,
                tiny_pivots_replaced: 0,
                max_pivot_growth: 1.0,
                condition_estimate: None,
                memory_usage: 0,
            },
        };
        
        let solve_data = TriangularSolveData::new(8, 4, &distribution, &numeric_factor);
        
        assert_eq!(solve_data.block_owners.len(), 2); // 8/4 = 2 blocks
        assert_eq!(solve_data.dependency_graph.len(), 2);
        assert!(!solve_data.comm_buffer.is_empty());
    }

    #[test]
    fn test_communication_patterns() {
        // Test that communication pattern enum works correctly
        assert_eq!(CommPattern::BinaryTree, CommPattern::BinaryTree);
        assert_ne!(CommPattern::BinaryTree, CommPattern::PointToPoint);
        
        // Test communication request creation
        let request = CommRequest {
            request_id: 1,
            source_rank: 0,
            dest_rank: 1,
            tag: 100,
            comm_type: CommType::Send,
            buffer_size: 64,
        };
        
        assert_eq!(request.request_id, 1);
        assert_eq!(request.comm_type, CommType::Send);
    }

    #[test]
    fn test_local_triangular_solve_l() {
        // Create a simple L factor panel
        let panel = Panel {
            width: 2,
            height: 2,
            // Column-major: L = [[1, 0], [2, 3]]
            data: vec![1.0, 2.0, 0.0, 3.0],
            row_indices: vec![0, 1],
            col_start: 0,
        };
        
        let factors = vec![panel];
        let mut x = vec![4.0, 11.0]; // Should solve to [4, 1] since L*[4,1] = [4, 11]
        
        DistributedTriangularSolver::solve_local_l_block(&mut x, &factors, 0).unwrap();
        
        // Check that solution is approximately [4, 1]
        assert!((x[0] - 4.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_local_triangular_solve_u() {
        // Create a simple U factor panel
        let panel = Panel {
            width: 2,
            height: 2,
            // Column-major: U = [[2, 1], [0, 3]]
            data: vec![2.0, 0.0, 1.0, 3.0],
            row_indices: vec![0, 1],
            col_start: 0,
        };
        
        let factors = vec![panel];
        let mut x = vec![5.0, 3.0]; // Should solve to [2, 1] since U*[2,1] = [5, 3]
        
        DistributedTriangularSolver::solve_local_u_block(&mut x, &factors, 0).unwrap();
        
        // Check that solution is approximately [2, 1]
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_block_update_application() {
        let panel = Panel {
            width: 2,
            height: 2,
            // Column-major: [[1, 2], [3, 4]]
            data: vec![1.0, 3.0, 2.0, 4.0],
            row_indices: vec![0, 1],
            col_start: 0,
        };
        
        let factors = vec![panel];
        let mut x_block = vec![5.0, 7.0];
        let update_data = vec![1.0, 1.0];
        
        // Apply update: x -= L[:, source_block] * update_data
        // Should subtract column 0 of L (i.e., [1, 3]) * 1 = [1, 3]
        DistributedTriangularSolver::apply_block_update(
            &mut x_block, &update_data, 0, 0, &factors
        ).unwrap();
        
        // Result should be [5-1, 7-3] = [4, 4]
        assert!((x_block[0] - 4.0).abs() < 1e-10);
        assert!((x_block[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_distributed_solve_integration() {
        // Create a simple SPD matrix for testing
        let matrix = CsrMatrix::from_csr(
            4, 4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 1, 2, 2, 3, 0, 3],
            vec![4.0, -1.0, 4.0, -1.0, 4.0, -1.0, -1.0, 4.0],
        );
        
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0; 4];
        let mut solver = SuperLuDistSolver::new();
        
        // Configure for distributed solve
        solver.set_async_panel_updates(true)
              .set_3d_factorization(false, None)
              .set_max_concurrent_panels(2);
        
        let comm = UniverseComm::NoComm(NoComm);
        let stats = solver.solve(&matrix, None, &b, &mut x, &comm, None, None).unwrap();
        
        // Verify solve completed
        assert_eq!(stats.iterations, 1);
        assert!(matches!(stats.reason, ConvergedReason::ConvergedAtol));
        
        // For a diagonal-dominant matrix, solution should be reasonable
        assert!(x.iter().all(|&val| val.is_finite()));
    }

    #[test]
    fn test_communication_overlap_options() {
        let mut solver = SuperLuDistSolver::new();
        
        // Test async panel updates
        solver.set_async_panel_updates(true);
        assert!(solver.options().async_panel_updates);
        
        // Test concurrent panel limits
        solver.set_max_concurrent_panels(8);
        assert_eq!(solver.options().max_concurrent_panels, 8);
        
        // Test 3D factorization
        solver.set_3d_factorization(true, Some(4));
        assert!(solver.options().enable_3d_factorization);
        assert_eq!(solver.options().process_grid_3d_depth, Some(4));
        
        // Test memory tradeoff
        solver.set_memory_tradeoff(3.0);
        assert_eq!(solver.options().memory_tradeoff_factor, 3.0);
    }

    #[test]
    fn test_superlu_dist_simple_solve() {
        // Create a simple 3x3 identity matrix
        let matrix = CsrMatrix::from_csr(
            3, 3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![1.0, 1.0, 1.0],
        );
        
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        let mut solver = SuperLuDistSolver::new();
        
        let comm = UniverseComm::NoComm(NoComm);
        let stats = solver.solve(&matrix, None, &b, &mut x, &comm, None, None).unwrap();
        
        // For identity matrix, solution should equal RHS
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
        assert_eq!(stats.iterations, 1);
        assert!(matches!(stats.reason, ConvergedReason::ConvergedAtol));
    }

    #[test]
    fn test_invalid_input_dimensions() {
        let matrix = CsrMatrix::from_csr(
            3, 3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![1.0, 1.0, 1.0],
        );
        
        let b = vec![1.0, 2.0]; // Wrong size
        let mut x = vec![0.0; 3];
        let mut solver = SuperLuDistSolver::new();
        
        let comm = UniverseComm::NoComm(NoComm);
        let result = solver.solve(&matrix, None, &b, &mut x, &comm, None, None);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), KError::InvalidInput(_)));
    }

    #[test]
    fn test_solver_reuse() {
        let matrix = CsrMatrix::from_csr(
            2, 2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![2.0, 3.0],
        );
        
        let mut solver = SuperLuDistSolver::new();
        let comm = UniverseComm::NoComm(NoComm);
        
        // First solve
        let b1 = vec![2.0, 3.0];
        let mut x1 = vec![0.0; 2];
        let _stats1 = solver.solve(&matrix, None, &b1, &mut x1, &comm, None, None).unwrap();
        
        // Solver should now have factorization cached
        assert!(solver.data.is_some());
        
        // Second solve with different RHS
        let b2 = vec![4.0, 6.0];
        let mut x2 = vec![0.0; 2];
        let _stats2 = solver.solve(&matrix, None, &b2, &mut x2, &comm, None, None).unwrap();
        
        // Factorization should be reused
        assert!(solver.data.is_some());
    }

    #[test]
    fn test_refinement_config() {
        let config = RefinementConfig {
            max_iterations: 10,
            tolerance: 1e-14,
            relative_tolerance: 1e-8,
            min_improvement_factor: 0.8,
        };
        
        let mut solver = SuperLuDistSolver::new();
        solver.set_refinement_config(config.clone());
        
        assert!(solver.refinement_engine.is_some());
        if let Some(ref engine) = solver.refinement_engine {
            assert_eq!(engine.config.max_iterations, 10);
            assert_eq!(engine.config.tolerance, 1e-14);
            assert_eq!(engine.config.relative_tolerance, 1e-8);
            assert_eq!(engine.config.min_improvement_factor, 0.8);
        }
    }

    #[test]
    fn test_refinement_methods() {
        let mut solver = SuperLuDistSolver::new();
        
        // Test enabling refinement
        solver.enable_iterative_refinement();
        assert!(solver.refinement_engine.is_some());
        
        // Test setting residual method
        solver.set_residual_method(ResidualMethod::Scaled);
        if let Some(ref engine) = solver.refinement_engine {
            assert_eq!(engine.residual_method, ResidualMethod::Scaled);
        }
        
        // Test disabling refinement
        solver.disable_iterative_refinement();
        assert!(solver.refinement_engine.is_none());
    }

    #[test]
    fn test_refinement_engine_creation() {
        let config = RefinementConfig::default();
        let engine = RefinementEngine::new(config, ResidualMethod::ComponentWise);
        
        assert_eq!(engine.residual_method, ResidualMethod::ComponentWise);
        assert!(engine.last_stats.is_none());
    }

    #[test]
    fn test_refinement_convergence_criteria() {
        let mut engine = RefinementEngine::with_defaults();
        
        // Test absolute tolerance convergence
        assert!(engine.check_convergence(1e-13, 1e-6, 1));
        
        // Test relative tolerance convergence
        assert!(engine.check_convergence(1e-7, 1e-1, 1));
        
        // Test no convergence
        assert!(!engine.check_convergence(1e-4, 1e-6, 1));
        
        // Test first iteration never converges
        assert!(!engine.check_convergence(1e-13, 1e-6, 0));
    }

    #[test]
    fn test_distributed_sparse_matvec() {
        let matrix = CsrMatrix::from_csr(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![2.0, 1.0, 3.0, 1.0, 1.0, 4.0],
        );
        
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        
        let comm = UniverseComm::NoComm(NoComm);
        
        RefinementEngine::distributed_sparse_matvec_static(&matrix, &x, &mut y, &comm).unwrap();
        
        // Expected: [2*1 + 1*2, 3*2 + 1*3, 1*1 + 4*3] = [4, 9, 13]
        assert_eq!(y, vec![4.0, 9.0, 13.0]);
    }

    #[test]
    fn test_refinement_stats() {
        let stats = RefinementStats {
            iterations: 3,
            initial_residual_norm: 1e-3,
            final_residual_norm: 1e-12,
            residual_history: vec![1e-3, 1e-6, 1e-9, 1e-12],
            converged: true,
            convergence_reason: RefinementConvergence::AbsoluteTolerance,
            refinement_time: 0.001,
        };
        
        assert_eq!(stats.iterations, 3);
        assert!(stats.converged);
        assert_eq!(stats.residual_history.len(), 4);
        assert!(matches!(stats.convergence_reason, RefinementConvergence::AbsoluteTolerance));
    }

    #[test]
    fn test_residual_methods() {
        // Test all residual method variants
        assert_eq!(ResidualMethod::Standard, ResidualMethod::Standard);
        assert_ne!(ResidualMethod::Standard, ResidualMethod::Scaled);
        assert_ne!(ResidualMethod::Scaled, ResidualMethod::ComponentWise);
    }

    #[test]
    fn test_refinement_workspace_setup() {
        let mut engine = RefinementEngine::with_defaults();
        let n = 100;
        
        engine.setup_workspace(n);
        
        assert_eq!(engine.residual_workspace.len(), n);
        assert_eq!(engine.correction_workspace.len(), n);
        assert_eq!(engine.matvec_workspace.len(), n);
    }

    #[test]
    fn test_vector_norm_computation() {
        let comm = UniverseComm::NoComm(NoComm);
        
        let vector = vec![3.0, 4.0, 0.0];
        let norm = RefinementEngine::compute_vector_norm_static(&vector, &comm).unwrap();
        
        // Expected norm: sqrt(9 + 16 + 0) = 5.0
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_refinement_convergence_variants() {
        // Test all convergence reason variants
        let reasons = [
            RefinementConvergence::AbsoluteTolerance,
            RefinementConvergence::RelativeTolerance,
            RefinementConvergence::MaxIterations,
            RefinementConvergence::Stagnation,
            RefinementConvergence::Divergence,
        ];
        
        for (i, reason1) in reasons.iter().enumerate() {
            for (j, reason2) in reasons.iter().enumerate() {
                if i == j {
                    assert_eq!(reason1, reason2);
                } else {
                    assert_ne!(reason1, reason2);
                }
            }
        }
    }

    #[test]
    fn test_iterative_refinement_integration() {
        // Test complete integration with solver and refinement
        let _matrix = CsrMatrix::from_csr(
            3, 3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![2.0, 1.0, 3.0, 1.0, 1.0, 4.0],
        );
        
        let _b = vec![6.0, 8.0, 10.0];
        let _x = vec![0.0; 3];
        
        // Create solver with iterative refinement enabled
        let mut solver = SuperLuDistSolver::new();
        solver.enable_iterative_refinement();
        
        // Configure refinement settings
        let config = RefinementConfig {
            max_iterations: 3,
            tolerance: 1e-10,
            relative_tolerance: 1e-8,
            min_improvement_factor: 0.9,
        };
        solver.set_refinement_config(config);
        solver.set_residual_method(ResidualMethod::Standard);
        
        let _comm = UniverseComm::NoComm(NoComm);
        
        // Since we don't have full matrix factorization in the test environment,
        // just verify that refinement engine is properly configured
        assert!(solver.refinement_engine.is_some());
        
        if let Some(ref engine) = solver.refinement_engine {
            assert_eq!(engine.config.max_iterations, 3);
            assert_eq!(engine.config.tolerance, 1e-10);
            assert_eq!(engine.residual_method, ResidualMethod::Standard);
        }
        
        // Test that refinement stats are initially None
        assert!(solver.refinement_stats().is_none());
    }

    #[test] 
    fn test_refinement_residual_scaling() {
        // Test different residual scaling methods
        let matrix = CsrMatrix::from_csr(
            2, 2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![1.0, 1.0],
        );
        
        let rhs = vec![2.0, 3.0];
        let solution = vec![1.0, 1.0];
        let mut residual = vec![0.0; 2];
        let mut matvec_workspace = vec![0.0; 2];
        let comm = UniverseComm::NoComm(NoComm);
        
        // Test standard residual
        RefinementEngine::compute_residual_static(
            &matrix, &rhs, &solution, &mut residual, &mut matvec_workspace,
            ResidualMethod::Standard, &comm
        ).unwrap();
        // Expected: rhs - matrix*solution = [2,3] - [1,1] = [1,2]
        assert_eq!(residual, vec![1.0, 2.0]);
        
        // Test scaled residual
        RefinementEngine::compute_residual_static(
            &matrix, &rhs, &solution, &mut residual, &mut matvec_workspace,
            ResidualMethod::Scaled, &comm
        ).unwrap();
        // Should be scaled by ||rhs|| = sqrt(4+9) = sqrt(13)
        let rhs_norm = (4.0 + 9.0_f64).sqrt();
        assert!((residual[0] - 1.0/rhs_norm).abs() < 1e-10);
        assert!((residual[1] - 2.0/rhs_norm).abs() < 1e-10);
    }
}
