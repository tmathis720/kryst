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

use crate::core::traits::{MatrixGet, RowPattern};
#[cfg(not(feature = "dense-direct"))]
use crate::matrix::op::CsrOp;
#[cfg(not(feature = "dense-direct"))]
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::PcSide;
#[cfg(not(feature = "dense-direct"))]
use crate::preconditioner::Preconditioner; // bring trait into scope for IluCsr::setup/apply
#[cfg(not(feature = "dense-direct"))]
use crate::preconditioner::ilu_csr::{
    IluCsr, IluCsrConfig, IluKind, PivotStrategy, ReorderingOptions,
};
#[cfg(feature = "dense-direct")]
use crate::solver::direct_lu::LuSolver;
#[cfg(feature = "dense-direct")]
use crate::solver::legacy::LinearSolver;
#[cfg(not(feature = "dense-direct"))]
use std::marker::PhantomData;

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
    #[cfg(feature = "dense-direct")]
    pub block_factors: Vec<(Vec<usize>, LuSolver<T>)>, // (indices, LU solver)
    #[cfg(not(feature = "dense-direct"))]
    pub block_factors_ilu: Vec<(Vec<usize>, std::sync::Arc<IluCsr>)>,
    #[cfg(not(feature = "dense-direct"))]
    _marker: PhantomData<T>,
}

impl BlockJacobi<f64> {
    pub fn dims(&self) -> (usize, usize) {
        let n = self
            .blocks
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |idx| idx + 1);
        (n, n)
    }

    /// Setup the Block-Jacobi preconditioner by factorizing each block.
    ///
    /// For each block, extracts the submatrix, factorizes it with LU, and stores the solver.
    ///
    /// # Arguments
    /// * `a` - The system matrix (must support row access and element access)
    #[cfg(feature = "dense-direct")]
    pub fn setup<M: RowPattern + MatrixGet<f64> + crate::matrix::dense::DenseMatrix<f64>>(
        &mut self,
        a: &M,
    ) {
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
            let _ = LinearSolver::solve(
                &mut lusolver,
                &amat,
                None,
                &vec![0.0; n],
                &mut vec![0.0; n],
                PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            );
            self.block_factors.push((block.clone(), lusolver));
        }
    }
    #[cfg(not(feature = "dense-direct"))]
    pub fn setup<M: RowPattern + MatrixGet<f64>>(&mut self, a: &M) {
        self.block_factors_ilu.clear();
        let cfg = IluCsrConfig {
            kind: IluKind::Ilu0,
            pivot: PivotStrategy::DiagonalPerturbation,
            pivot_threshold: 1e-12,
            diag_perturb_factor: 1e-10,
            level_sched: cfg!(feature = "rayon"),
            numeric_update_fixed: true,
            logging: 0,
            reordering: ReorderingOptions::default(),
        };
        for block in &self.blocks {
            let n = block.len();
            let mut row_ptr = Vec::with_capacity(n + 1);
            let mut col_idx = Vec::new();
            let mut values = Vec::new();
            row_ptr.push(0);
            for &i in block {
                let row = a.row_indices(i);
                for (jj, &j) in block.iter().enumerate() {
                    if row.contains(&j) {
                        col_idx.push(jj);
                        values.push(a.get(i, j));
                    }
                }
                row_ptr.push(col_idx.len());
            }
            let csr =
                std::sync::Arc::new(CsrMatrix::<f64>::from_csr(n, n, row_ptr, col_idx, values));
            let mut ilu = IluCsr::new_with_config(cfg.clone());
            let op = CsrOp::new(csr.clone());
            let _ = ilu.setup(&op);
            self.block_factors_ilu
                .push((block.clone(), std::sync::Arc::new(ilu)));
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
        for zi in z.iter_mut() {
            *zi = 0.0;
        }
        #[cfg(all(feature = "rayon", feature = "dense-direct"))]
        {
            use rayon::prelude::*;
            use std::sync::Arc;
            use std::sync::Mutex;
            let z_arc = Arc::new(Mutex::new(z));
            self.block_factors
                .par_iter()
                .for_each(|(indices, lusolver)| {
                    // Extract the block of r corresponding to this block
                    let mut r_block = Vec::with_capacity(indices.len());
                    for &i in indices {
                        r_block.push(r[i]);
                    }
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
        #[cfg(all(not(feature = "rayon"), feature = "dense-direct"))]
        {
            for (indices, lusolver) in &self.block_factors {
                // Extract the block of r corresponding to this block
                let mut r_block = Vec::with_capacity(indices.len());
                for &i in indices {
                    r_block.push(r[i]);
                }
                let mut x_block = vec![0.0; indices.len()];
                // Solve the block system
                lusolver.solve_cached(&r_block, &mut x_block);
                // Write the solution back to the correct entries in z
                for (&i, &xi) in indices.iter().zip(x_block.iter()) {
                    z[i] = xi;
                }
            }
        }
        #[cfg(not(feature = "dense-direct"))]
        {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                use std::sync::{Arc, Mutex};
                let z_arc = Arc::new(Mutex::new(z));
                self.block_factors_ilu
                    .par_iter()
                    .for_each(|(indices, ilu)| {
                        let mut r_blk = Vec::with_capacity(indices.len());
                        for &i in indices {
                            r_blk.push(r[i]);
                        }
                        let mut x_blk = vec![0.0; indices.len()];
                        let _ = ilu.apply(PcSide::Left, &r_blk, &mut x_blk);
                        let mut z_guard = z_arc.lock().unwrap();
                        for (&i, &xi) in indices.iter().zip(x_blk.iter()) {
                            z_guard[i] = xi;
                        }
                    });
            }
            #[cfg(not(feature = "rayon"))]
            {
                for (indices, ilu) in &self.block_factors_ilu {
                    let mut r_blk = Vec::with_capacity(indices.len());
                    for &i in indices {
                        r_blk.push(r[i]);
                    }
                    let mut x_blk = vec![0.0; indices.len()];
                    let _ = ilu.apply(PcSide::Left, &r_blk, &mut x_blk);
                    for (&i, &xi) in indices.iter().zip(x_blk.iter()) {
                        z[i] = xi;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::bridge::BridgeScratch;
    use crate::algebra::prelude::*;
    use crate::core::traits::{MatrixGet, RowPattern};
    use crate::error::KError;
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::PcSide;
    use std::marker::PhantomData;

    struct TestDiagMatrix {
        diag: Vec<f64>,
        pattern: Vec<Vec<usize>>,
    }

    impl TestDiagMatrix {
        fn new(diag: Vec<f64>) -> Self {
            let pattern = (0..diag.len()).map(|i| vec![i]).collect();
            Self { diag, pattern }
        }
    }

    impl RowPattern for TestDiagMatrix {
        fn row_indices(&self, i: usize) -> &[usize] {
            &self.pattern[i]
        }
    }

    impl MatrixGet<f64> for TestDiagMatrix {
        fn get(&self, i: usize, j: usize) -> f64 {
            if i == j { self.diag[i] } else { 0.0 }
        }
    }

    impl crate::preconditioner::Preconditioner for BlockJacobi<f64> {
        fn dims(&self) -> (usize, usize) {
            Self::dims(self)
        }

        fn setup(&mut self, _a: &dyn crate::matrix::op::LinOp<S = f64>) -> Result<(), KError> {
            Ok(())
        }

        fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
            BlockJacobi::apply(self, x, y);
            Ok(())
        }
    }

    #[test]
    fn apply_bridge_matches_real() {
        let mut pc = BlockJacobi {
            blocks: vec![vec![0], vec![1]],
            #[cfg(feature = "dense-direct")]
            block_factors: Vec::new(),
            #[cfg(not(feature = "dense-direct"))]
            block_factors_ilu: Vec::new(),
            #[cfg(not(feature = "dense-direct"))]
            _marker: PhantomData,
        };

        let a = TestDiagMatrix::new(vec![4.0, 9.0]);
        pc.setup(&a);

        let rhs_real = vec![8.0, 18.0];
        let mut out_real = vec![0.0; rhs_real.len()];
        pc.apply(&rhs_real, &mut out_real);

        let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
        let mut out_s = vec![S::zero(); rhs_real.len()];
        let mut scratch = BridgeScratch::default();
        let wrapper = crate::ops::wrap::as_s_pc(&pc);
        wrapper
            .apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
            .expect("block jacobi bridge apply");

        for (ys, yr) in out_s.iter().zip(out_real.iter()) {
            assert!((ys.real() - yr).abs() < 1e-12);
        }
    }
}
