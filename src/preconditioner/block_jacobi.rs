// Block-Jacobi preconditioner implementation
//
// This module implements the Block-Jacobi preconditioner, which divides the matrix into blocks
// and applies an exact or approximate inverse to each block independently. This is useful for
// accelerating the convergence of iterative solvers, especially when the matrix has a natural block structure.
//
// Each block is factorized (LU) and stored for efficient repeated application.
//
// # Usage
//
// 1. Create a `BlockJacobi` with a list of block index sets.
// 2. Call `setup` with the system matrix to factorize each block.
// 3. Use `apply` to apply the preconditioner to a vector.

use crate::core::traits::{RowPattern, MatrixGet};
use crate::solver::direct_lu::{LuSolver};
use crate::solver::LinearSolver;

/// Block-Jacobi preconditioner
///
/// Stores the block structure and the LU factorization for each block.
///
/// - `blocks`: List of index sets, each representing a block (list of row/column indices)
/// - `block_factors`: For each block, stores the indices and the corresponding LU solver
pub struct BlockJacobi<T> {
    /// List of block index sets (each block is a list of row/column indices)
    pub blocks: Vec<Vec<usize>>,
    /// For each block: (indices, LU solver for the block)
    pub block_factors: Vec<(Vec<usize>, LuSolver<T>)>, // (indices, LU solver)
}

impl BlockJacobi<f64> {
    /// Setup the Block-Jacobi preconditioner by factorizing each block.
    ///
    /// For each block, extracts the submatrix, factorizes it with LU, and stores the solver.
    ///
    /// # Arguments
    /// * `a` - The system matrix (must support row access and element access)
    pub fn setup<M: RowPattern + MatrixGet<f64> + crate::matrix::dense::DenseMatrix<f64>>(&mut self, a: &M) {
        self.block_factors.clear();
        for block in &self.blocks {
            let n = block.len();
            // Extract the n x n block submatrix
            let mut data = vec![0.0; n * n];
            for (ii, &i) in block.iter().enumerate() {
                let row = a.row_indices(i);
                for (jj, &j) in block.iter().enumerate() {
                    // Only fill if the entry exists in the original matrix
                    if row.contains(&j) {
                        data[jj * n + ii] = a.get(i, j);
                    }
                }
            }
            // Create a dense matrix for the block
            let amat = crate::matrix::dense::DenseMatrix::from_raw(n, n, data);
            let mut lusolver = LuSolver::<f64>::new();
            // Factorize the block (dummy solve to trigger factorization)
            let _ = LinearSolver::solve(&mut lusolver, &amat, None, &vec![0.0; n], &mut vec![0.0; n]);
            self.block_factors.push((block.clone(), lusolver));
        }
    }
    /// Apply the Block-Jacobi preconditioner: z = M⁻¹ r
    ///
    /// For each block, solves the block system and writes the result into the corresponding entries of z.
    ///
    /// # Arguments
    /// * `r` - Input vector (right-hand side)
    /// * `z` - Output vector (solution, overwritten)
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        // Zero out the output vector
        for zi in z.iter_mut() { *zi = 0.0; }
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            use std::sync::Arc;
            use std::sync::Mutex;
            let z_arc = Arc::new(Mutex::new(z));
            self.block_factors.par_iter().for_each(|(indices, lusolver)| {
                // Extract the block of r corresponding to this block
                let mut r_block = Vec::with_capacity(indices.len());
                for &i in indices { r_block.push(r[i]); }
                let mut x_block = vec![0.0; indices.len()];
                // Solve the block system
                lusolver.solve_cached(&r_block, &mut x_block);
                // Write the solution back to the correct entries in z
                let mut z_guard = z_arc.lock().unwrap();
                for (&i, &xi) in indices.iter().zip(x_block.iter()) {
                    z_guard[i] = xi;
                }
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for (indices, lusolver) in &self.block_factors {
                // Extract the block of r corresponding to this block
                let mut r_block = Vec::with_capacity(indices.len());
                for &i in indices { r_block.push(r[i]); }
                let mut x_block = vec![0.0; indices.len()];
                // Solve the block system
                lusolver.solve_cached(&r_block, &mut x_block);
                // Write the solution back to the correct entries in z
                for (&i, &xi) in indices.iter().zip(x_block.iter()) {
                    z[i] = xi;
                }
            }
        }
    }
}
