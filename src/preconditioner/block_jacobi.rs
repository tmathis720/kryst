// Block-Jacobi preconditioner implementation

use crate::core::traits::{RowPattern, MatrixGet};
use crate::solver::direct_lu::{LuSolver};
use crate::solver::LinearSolver;

/// Block-Jacobi preconditioner
pub struct BlockJacobi<T> {
    pub blocks: Vec<Vec<usize>>,
    pub block_factors: Vec<(Vec<usize>, LuSolver<T>)>, // (indices, LU solver)
}

impl BlockJacobi<f64> {
    /// Setup: factor each block using LuSolver
    pub fn setup<M: RowPattern + MatrixGet<f64> + crate::matrix::dense::DenseMatrix<f64>>(&mut self, a: &M) {
        self.block_factors.clear();
        for block in &self.blocks {
            let n = block.len();
            let mut data = vec![0.0; n * n];
            for (ii, &i) in block.iter().enumerate() {
                let row = a.row_indices(i);
                for (jj, &j) in block.iter().enumerate() {
                    if row.contains(&j) {
                        data[jj * n + ii] = a.get(i, j);
                    }
                }
            }
            let amat = crate::matrix::dense::DenseMatrix::from_raw(n, n, data);
            let mut lusolver = LuSolver::<f64>::new();
            let _ = LinearSolver::solve(&mut lusolver, &amat, None, &vec![0.0; n], &mut vec![0.0; n]);
            self.block_factors.push((block.clone(), lusolver));
        }
    }
    /// Apply: z = M⁻¹ r
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        for zi in z.iter_mut() { *zi = 0.0; }
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            use std::sync::Arc;
            use std::sync::Mutex;
            let z_arc = Arc::new(Mutex::new(z));
            self.block_factors.par_iter().for_each(|(indices, lusolver)| {
                let mut r_block = Vec::with_capacity(indices.len());
                for &i in indices { r_block.push(r[i]); }
                let mut x_block = vec![0.0; indices.len()];
                lusolver.solve_cached(&r_block, &mut x_block);
                let mut z_guard = z_arc.lock().unwrap();
                for (&i, &xi) in indices.iter().zip(x_block.iter()) {
                    z_guard[i] = xi;
                }
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for (indices, lusolver) in &self.block_factors {
                let mut r_block = Vec::with_capacity(indices.len());
                for &i in indices { r_block.push(r[i]); }
                let mut x_block = vec![0.0; indices.len()];
                lusolver.solve_cached(&r_block, &mut x_block);
                for (&i, &xi) in indices.iter().zip(x_block.iter()) {
                    z[i] = xi;
                }
            }
        }
    }
}
