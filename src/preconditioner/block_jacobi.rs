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

#[cfg(feature = "complex")]
use crate::algebra::bridge::{BridgeScratch, copy_real_into_scalar, copy_scalar_to_real_in};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::core::traits::{MatrixGet, RowPattern};
use crate::error::KError;
#[cfg(all(not(feature = "dense-direct"), not(feature = "complex")))]
use crate::matrix::op::CsrOp;
#[cfg(not(feature = "complex"))]
use crate::matrix::sparse::CsrMatrix;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::PcSide;
#[cfg(not(feature = "complex"))]
use crate::preconditioner::Preconditioner;
#[cfg(all(not(feature = "dense-direct"), not(feature = "complex")))]
use crate::preconditioner::ilu_csr::{
    IluCsr, IluCsrConfig, IluKind, PivotStrategy, ReorderingOptions,
};
use crate::utils::conditioning::ConditioningOptions;
#[cfg(feature = "dense-direct")]
use crate::solver::direct_lu::LuSolver;
#[cfg(feature = "dense-direct")]
use crate::solver::legacy::LinearSolver;
#[cfg(feature = "dense-direct")]
use faer::Mat;

/// Block-Jacobi preconditioner
///
/// Stores the block structure and the LU factorization for each block.
///
/// - `blocks`: List of index sets, each representing a block (list of row/column indices)
/// - `block_factors`: For each block, stores the indices and the corresponding LU solver
pub struct BlockJacobi {
    /// List of block index sets (each block is a list of row/column indices)
    pub blocks: Vec<Vec<usize>>,
    /// Contiguous block size (used to construct blocks when none are provided).
    pub block_size: usize,
    /// For each block: (indices, LU solver for the block)
    #[cfg(feature = "dense-direct")]
    pub block_factors: Vec<(Vec<usize>, LuSolver)>, // (indices, LU solver)
    #[cfg(all(not(feature = "dense-direct"), not(feature = "complex")))]
    pub block_factors_ilu: Vec<(Vec<usize>, std::sync::Arc<IluCsr>)>,
}

impl BlockJacobi {
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

    fn ensure_blocks(&mut self, n: usize) {
        if !self.blocks.is_empty() || n == 0 {
            return;
        }
        let block = self.block_size.max(1);
        let mut start = 0;
        while start < n {
            let end = (start + block).min(n);
            self.blocks.push((start..end).collect());
            start = end;
        }
    }

    /// Setup the Block-Jacobi preconditioner by factorizing each block.
    ///
    /// For each block, extracts the submatrix, factorizes it with LU, and stores the solver.
    ///
    /// # Arguments
    /// * `a` - The system matrix (must support row access and element access)
    #[cfg(feature = "dense-direct")]
    pub fn setup<M>(&mut self, a: &M)
    where
        M: RowPattern + MatrixGet<S>,
    {
        self.block_factors.clear();
        for block in &self.blocks {
            let n = block.len();
            // Extract the n x n block submatrix
            let mut data = vec![R::zero(); n * n];
            for (ii, &i) in block.iter().enumerate() {
                let row = a.row_indices(i);
                for (jj, &j) in block.iter().enumerate() {
                    // Only fill if the entry exists in the original matrix
                    if row.contains(&j) {
                        data[jj * n + ii] = a.get(i, j).real();
                    }
                }
            }
            // Create a dense matrix for the block
            let amat = Mat::from_fn(n, n, |i, j| data[j * n + i]);
            let mut lusolver = LuSolver::new();
            // Factorize the block (dummy solve to trigger factorization)
            let _ = lusolver.solve(
                &amat,
                None,
                &vec![R::zero(); n],
                &mut vec![R::zero(); n],
                PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            );
            self.block_factors.push((block.clone(), lusolver));
        }
    }
    #[cfg(all(not(feature = "dense-direct"), not(feature = "complex")))]
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
            conditioning: ConditioningOptions::default(),
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
    #[cfg(feature = "dense-direct")]
    fn setup_from_dense(&mut self, a: &Mat<f64>) {
        self.block_factors.clear();
        for block in &self.blocks {
            let n = block.len();
            let mut data = vec![R::zero(); n * n];
            for (ii, &i) in block.iter().enumerate() {
                for (jj, &j) in block.iter().enumerate() {
                    data[jj * n + ii] = a[(i, j)];
                }
            }
            let amat = Mat::from_fn(n, n, |i, j| data[j * n + i]);
            let mut lusolver = LuSolver::new();
            let _ = lusolver.solve(
                &amat,
                None,
                &vec![R::zero(); n],
                &mut vec![R::zero(); n],
                PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            );
            self.block_factors.push((block.clone(), lusolver));
        }
    }
    #[cfg(feature = "dense-direct")]
    fn setup_from_csr_dense(&mut self, a: &CsrMatrix<f64>) {
        self.block_factors.clear();
        for block in &self.blocks {
            let n = block.len();
            let mut data = vec![R::zero(); n * n];
            let mut map = std::collections::HashMap::with_capacity(n);
            for (local, &global) in block.iter().enumerate() {
                map.insert(global, local);
            }
            for (ii, &i) in block.iter().enumerate() {
                let (cols, vals) = a.row(i);
                for (&col, &val) in cols.iter().zip(vals.iter()) {
                    if let Some(&jj) = map.get(&col) {
                        data[jj * n + ii] = val;
                    }
                }
            }
            let amat = Mat::from_fn(n, n, |i, j| data[j * n + i]);
            let mut lusolver = LuSolver::new();
            let _ = lusolver.solve(
                &amat,
                None,
                &vec![R::zero(); n],
                &mut vec![R::zero(); n],
                PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            );
            self.block_factors.push((block.clone(), lusolver));
        }
    }
    #[cfg(all(not(feature = "dense-direct"), not(feature = "complex")))]
    fn setup_from_csr_ilu(&mut self, a: &CsrMatrix<f64>) {
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
            conditioning: ConditioningOptions::default(),
        };
        for block in &self.blocks {
            let n = block.len();
            let mut map = std::collections::HashMap::with_capacity(n);
            for (local, &global) in block.iter().enumerate() {
                map.insert(global, local);
            }
            let mut row_ptr = Vec::with_capacity(n + 1);
            let mut col_idx = Vec::new();
            let mut values = Vec::new();
            row_ptr.push(0);
            for &i in block {
                let (cols, vals) = a.row(i);
                for (&col, &val) in cols.iter().zip(vals.iter()) {
                    if let Some(&jj) = map.get(&col) {
                        col_idx.push(jj);
                        values.push(val);
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
    ///
    /// **Note:** When `S != f64` (complex), this panics - use KPreconditioner trait instead.
    pub fn apply(&self, _r: &[S], _z: &mut [S]) {
        #[cfg(feature = "complex")]
        {
            panic!("BlockJacobi::apply called with S != f64; use KPreconditioner trait");
        }
        #[cfg(not(feature = "complex"))]
        {
            // When complex is disabled, S == f64, so this is safe
            let r_f64 = unsafe { std::mem::transmute::<&[S], &[f64]>(_r) };
            let z_f64 = unsafe { std::mem::transmute::<&mut [S], &mut [f64]>(_z) };
            self.apply_real(r_f64, z_f64);
        }
    }

    /// Internal real-valued apply implementation
    fn apply_real(&self, r: &[f64], z: &mut [f64]) {
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
        #[cfg(all(not(feature = "dense-direct"), not(feature = "complex")))]
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

#[cfg(not(feature = "complex"))]
impl Preconditioner for BlockJacobi {
    fn dims(&self) -> (usize, usize) {
        self.dims()
    }

    fn setup(&mut self, a: &dyn crate::matrix::op::LinOp<S = S>) -> Result<(), KError> {
        if let Some(dist) = a.as_any().downcast_ref::<crate::matrix::DistCsrOp>() {
            let n_local = dist.local_nrows();
            self.ensure_blocks(n_local);
            #[cfg(feature = "dense-direct")]
            {
                let local = dist.local_block_dense();
                self.setup_from_dense(&local);
                return Ok(());
            }
            #[cfg(not(feature = "dense-direct"))]
            {
                let local = dist.local_block_csr();
                self.setup_from_csr_ilu(&local);
                return Ok(());
            }
        }

        #[cfg(feature = "dense-direct")]
        {
            if let Some(mat) = a.as_any().downcast_ref::<Mat<f64>>() {
                self.ensure_blocks(mat.nrows());
                self.setup_from_dense(mat);
                return Ok(());
            }
            if let Some(dense_op) = a.as_any().downcast_ref::<crate::matrix::op::DenseOp<f64>>() {
                let mat = dense_op.inner();
                self.ensure_blocks(mat.nrows());
                self.setup_from_dense(mat);
                return Ok(());
            }
        }

        let csr = crate::matrix::convert::csr_from_linop(a, 0.0)?;
        self.ensure_blocks(csr.nrows());
        #[cfg(feature = "dense-direct")]
        {
            self.setup_from_csr_dense(&csr);
        }
        #[cfg(not(feature = "dense-direct"))]
        {
            self.setup_from_csr_ilu(&csr);
        }
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(format!(
                "BlockJacobi::apply dimension mismatch: x.len()={}, y.len()={}",
                x.len(),
                y.len()
            )));
        }
        BlockJacobi::apply(self, x, y);
        Ok(())
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for BlockJacobi {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        self.dims()
    }

    fn apply_s(
        &self,
        _side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(format!(
                "BlockJacobi::apply_s dimension mismatch: x.len()={}, y.len()={}",
                x.len(),
                y.len()
            )));
        }

        let n = x.len();
        scratch.with_pair(n, |xr, yr| {
            copy_scalar_to_real_in(x, xr);
            self.apply_real(xr, yr);
            copy_real_into_scalar(yr, y);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "complex")]
    use crate::algebra::bridge::BridgeScratch;
    use crate::core::traits::{Indexing, MatVec, MatrixGet, RowPattern};
    use crate::error::KError;
    #[cfg(feature = "complex")]
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::PcSide;

    struct TestDiagMatrix {
        diag: Vec<S>,
        pattern: Vec<Vec<usize>>,
    }

    impl TestDiagMatrix {
        fn new(diag: Vec<S>) -> Self {
            let pattern = (0..diag.len()).map(|i| vec![i]).collect();
            Self { diag, pattern }
        }
    }

    impl RowPattern for TestDiagMatrix {
        fn row_indices(&self, i: usize) -> &[usize] {
            &self.pattern[i]
        }
    }

    impl MatrixGet<S> for TestDiagMatrix {
        fn get(&self, i: usize, j: usize) -> S {
            if i == j { self.diag[i] } else { S::zero() }
        }
    }

    impl Indexing for TestDiagMatrix {
        fn nrows(&self) -> usize {
            self.diag.len()
        }
    }

    impl MatVec<Vec<S>> for TestDiagMatrix {
        fn matvec(&self, x: &Vec<S>, y: &mut Vec<S>) {
            assert_eq!(
                x.len(),
                self.diag.len(),
                "TestDiagMatrix::matvec input length mismatch"
            );
            if y.len() != self.diag.len() {
                y.resize(self.diag.len(), S::zero());
            }
            for (i, diag) in self.diag.iter().enumerate() {
                y[i] = diag * x[i];
            }
        }
    }

    impl crate::matrix::dense::DenseMatrix for TestDiagMatrix {
        fn from_raw(nrows: usize, ncols: usize, data: Vec<S>) -> Self {
            assert_eq!(
                nrows, ncols,
                "TestDiagMatrix::from_raw expects a square matrix"
            );
            assert_eq!(
                data.len(),
                nrows * ncols,
                "TestDiagMatrix::from_raw received inconsistent storage"
            );
            let diag = (0..nrows).map(|i| data[i + i * nrows]).collect::<Vec<_>>();
            TestDiagMatrix::new(diag)
        }
    }

    #[cfg(all(feature = "complex", feature = "dense-direct"))]
    #[test]
    fn apply_s_matches_real_path() {
        let mut pc = BlockJacobi {
            blocks: vec![vec![0], vec![1]],
            block_size: 0,
            #[cfg(feature = "dense-direct")]
            block_factors: Vec::new(),
            #[cfg(not(feature = "dense-direct"))]
            block_factors_ilu: Vec::new(),
        };

        let a = TestDiagMatrix::new(vec![S::from_real(4.0), S::from_real(9.0)]);
        pc.setup(&a);

        let rhs_real = vec![S::from_real(8.0), S::from_real(18.0)];
        let mut out_real = vec![R::default(); rhs_real.len()];
        pc.apply(&rhs_real, &mut out_real);

        let rhs_s: Vec<S> = rhs_real.iter().copied().collect();
        let mut out_s = vec![S::zero(); rhs_real.len()];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
            .expect("block jacobi apply_s");

        for (ys, yr) in out_s.iter().zip(out_real.iter()) {
            assert!((ys.real() - yr).abs() < 1e-12);
        }
    }
}
