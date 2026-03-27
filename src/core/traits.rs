//! Core linear-algebra traits for kryst.

/// Matrix–vector product: y ← A x.
pub trait MatVec<V> {
    /// Compute y = A · x.
    fn matvec(&self, x: &V, y: &mut V);
}

/// Matrix–transpose–vector product: y ← Aᵗ x (A^H for complex).
pub trait MatTransVec<V> {
    /// Compute y = Aᵗ · x (or A^H · x in complex builds).
    fn mattransvec(&self, x: &V, y: &mut V);
}

// Blanket implementations of MatVec/MatTransVec for LinOp types using Vec storage.
use crate::algebra::parallel::{self, par_dot_conj_local, par_sum_abs2_local};
use crate::algebra::prelude::*;
use crate::algebra::scalar::{KrystScalar, copy_real_to_scalar_in, copy_scalar_to_real_in};
use crate::core::block::BlockVec;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::LocalPreconditioner;

impl<L> MatVec<Vec<S>> for L
where
    L: LinOp<S = S> + ?Sized,
{
    fn matvec(&self, x: &Vec<S>, y: &mut Vec<S>) {
        LinOp::matvec(self, &x[..], &mut y[..]);
    }
}

impl<L> MatTransVec<Vec<S>> for L
where
    L: LinOp<S = S> + ?Sized,
{
    fn mattransvec(&self, x: &Vec<S>, y: &mut Vec<S>) {
        if !LinOp::supports_transpose(self) {
            panic!("t_matvec not supported");
        }
        LinOp::t_matvec(self, &x[..], &mut y[..]);
    }
}

#[cfg(feature = "complex")]
impl<L> MatVec<Vec<f64>> for L
where
    L: LinOp<S = f64> + ?Sized,
{
    fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
        LinOp::matvec(self, &x[..], &mut y[..]);
    }
}

#[cfg(feature = "complex")]
impl<L> MatTransVec<Vec<f64>> for L
where
    L: LinOp<S = f64> + ?Sized,
{
    fn mattransvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
        if !LinOp::supports_transpose(self) {
            panic!("t_matvec not supported");
        }
        LinOp::t_matvec(self, &x[..], &mut y[..]);
    }
}

/// Real-valued block operator used by block Krylov variants while remaining matrix-free.
///
/// This trait is intentionally defined in terms of `f64` scalars. In complex builds the helper
/// conversions (`copy_scalar_to_real_in`) drop imaginary parts, so `apply_many`/`apply` work
/// only on the real components of the block vectors. Use this trait for real preconditioners
/// or diagnostics inside complex solvers rather than for fully complex-valued operators.
pub trait BlockOp {
    /// Apply the operator to multiple columns at once. Default implementation calls
    /// [`apply`](Self::apply) per column to remain format agnostic.
    fn apply_many(&self, x: &BlockVec, y: &mut BlockVec) -> Result<(), KError> {
        if x.ncols() != y.ncols() {
            return Err(KError::InvalidInput(format!(
                "apply_many column mismatch: {} vs {}",
                x.ncols(),
                y.ncols()
            )));
        }
        let mut x_real = vec![0.0; x.nrows()];
        let mut y_real = vec![0.0; y.nrows()];
        // In complex builds `copy_scalar_to_real_in` discards imaginary components.
        for c in 0..x.ncols() {
            copy_scalar_to_real_in(x.col(c), &mut x_real);
            copy_scalar_to_real_in(y.col(c), &mut y_real);
            self.apply(&x_real, &mut y_real)?;
            copy_real_to_scalar_in(&y_real, y.col_mut(c));
        }
        Ok(())
    }

    /// Apply the operator to a single column.
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError>;

    /// Apply the transpose of the operator if available.
    fn apply_t(&self, _x: &[f64], _y: &mut [f64]) -> Result<(), KError> {
        Err(KError::Unsupported("transpose not available"))
    }
}

impl<T> BlockOp for T
where
    T: LinOp<S = f64> + ?Sized,
{
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        LinOp::try_matvec(self, x, y)
    }

    fn apply_t(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if !LinOp::supports_transpose(self) {
            return Err(KError::Unsupported(
                "LinOp::t_matvec called but transpose not supported",
            ));
        }
        LinOp::t_matvec(self, x, y);
        Ok(())
    }
}

/// Inner products & norms.
pub trait InnerProduct<V> {
    /// Associated scalar type.
    type Scalar: Copy + PartialOrd + From<f64> + Into<f64>;
    /// Compute dot(x, y) with communicator support for parallel reductions.
    fn dot(&self, x: &V, y: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar;
    /// Compute ‖x‖₂ with communicator support for parallel reductions.
    fn norm(&self, x: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar {
        let local_sq = self.dot(x, x, comm);
        let global_sq = comm.all_reduce_f64(local_sq.into());
        (global_sq.sqrt()).into()
    }
}

/// Uniform indexing into vectors (dense or sparse).
pub trait Indexing {
    /// Number of rows (or length for a vector).
    fn nrows(&self) -> usize;
}

/// Matrix shape trait: provides nrows/ncols for matrices and vectors.
pub trait MatShape {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Trait for extracting the sparsity pattern of a matrix row.
pub trait RowPattern {
    /// Returns the column indices of nonzeros in row i.
    fn row_indices(&self, i: usize) -> &[usize];
}

/// Trait for extracting elements from a matrix.
pub trait MatrixGet<T> {
    /// Get the element at position (i, j).
    fn get(&self, i: usize, j: usize) -> T;
}

/// Trait for extracting a submatrix by index set (for block/ASM preconditioners).
pub trait SubmatrixExtract: Sized {
    /// Stored scalar type for the matrix.
    type S;

    /// Extract a submatrix defined by row and column index sets.
    fn extract_submatrix(&self, rows: &[usize], cols: &[usize]) -> Self;

    /// Convenience helper for square sub-blocks that use the same index set.
    fn submatrix(&self, indices: &[usize]) -> Self {
        self.extract_submatrix(indices, indices)
    }
}

/// Sparse-aware matrix-vector operations for AMG and iterative solvers
pub trait MatVecOp<T> {
    /// Compute y = alpha * A * x + beta * y
    fn mat_vec(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) -> Result<(), crate::error::KError>;

    /// Compute y = alpha * A^T * x + beta * y (transpose operation)
    fn mat_vec_trans(
        &self,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
    ) -> Result<(), crate::error::KError>;

    /// Get the number of rows
    fn nrows(&self) -> usize;

    /// Get the number of columns  
    fn ncols(&self) -> usize;
}

/// Sparse-aware dot product operations
pub trait DotOp<T> {
    /// Compute the dot product x^T * y
    fn dot(&self, x: &[T], y: &[T]) -> T;

    /// Compute the 2-norm of a vector
    fn norm2(&self, x: &[T]) -> T;
}

/// Implementation for sparse CSR matrices over the crate scalar.
impl MatVecOp<S> for crate::matrix::sparse::CsrMatrix<S> {
    fn mat_vec(&self, alpha: S, x: &[S], beta: S, y: &mut [S]) -> Result<(), crate::error::KError> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".to_string(),
            ));
        }

        let rp = self.row_ptr();
        let cj = self.col_idx();
        let vv = self.values();

        #[cfg(debug_assertions)]
        {
            assert_eq!(rp.len(), self.nrows() + 1);
            assert!(rp.windows(2).all(|w| w[0] <= w[1]));
            let nnz = *rp.last().unwrap();
            assert_eq!(cj.len(), nnz);
            assert_eq!(vv.len(), nnz);
        }

        if alpha == S::zero() {
            if beta == S::zero() {
                for v in y.iter_mut() {
                    *v = S::zero();
                }
            } else if beta != S::one() {
                for v in y.iter_mut() {
                    *v *= beta;
                }
            }
            return Ok(());
        }

        let m = self.nrows();
        if beta == S::zero() {
            for i in 0..m {
                let rs = rp[i];
                let re = rp[i + 1];
                let mut acc = S::zero();
                for p in rs..re {
                    let j = cj[p];
                    acc = vv[p].mul_add(x[j], acc);
                }
                y[i] = alpha * acc;
            }
        } else if beta == S::one() {
            for i in 0..m {
                let rs = rp[i];
                let re = rp[i + 1];
                let mut acc = S::zero();
                for p in rs..re {
                    let j = cj[p];
                    acc = vv[p].mul_add(x[j], acc);
                }
                y[i] += alpha * acc;
            }
        } else {
            for i in 0..m {
                let rs = rp[i];
                let re = rp[i + 1];
                let mut acc = S::zero();
                for p in rs..re {
                    let j = cj[p];
                    acc = vv[p].mul_add(x[j], acc);
                }
                y[i] = alpha * acc + beta * y[i];
            }
        }

        Ok(())
    }

    fn mat_vec_trans(
        &self,
        alpha: S,
        x: &[S],
        beta: S,
        y: &mut [S],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.nrows() || y.len() != self.ncols() {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".to_string(),
            ));
        }

        let rp = self.row_ptr();
        let cj = self.col_idx();
        let vv = self.values();

        if alpha == S::zero() {
            if beta == S::zero() {
                for v in y.iter_mut() {
                    *v = S::zero();
                }
            } else {
                for v in y.iter_mut() {
                    *v *= beta;
                }
            }
            return Ok(());
        }

        if beta == S::zero() {
            for v in y.iter_mut() {
                *v = S::zero();
            }
        } else if beta != S::one() {
            for v in y.iter_mut() {
                *v *= beta;
            }
        }

        let m = self.nrows();
        for i in 0..m {
            let xi = x[i];
            if xi == S::zero() {
                continue;
            }
            let start = rp[i];
            let end = rp[i + 1];
            for k in start..end {
                let j = cj[k];
                y[j] += alpha * vv[k] * xi;
            }
        }

        Ok(())
    }

    fn nrows(&self) -> usize {
        self.nrows()
    }
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

#[cfg(feature = "backend-faer")]
impl MatVecOp<S> for faer::Mat<S> {
    fn mat_vec(&self, alpha: S, x: &[S], beta: S, y: &mut [S]) -> Result<(), crate::error::KError> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".into(),
            ));
        }

        if beta == S::zero() {
            for value in y.iter_mut() {
                *value = S::zero();
            }
        } else if beta != S::one() {
            for value in y.iter_mut() {
                *value *= beta;
            }
        }

        let m = self.nrows();
        let n = self.ncols();
        for i in 0..m {
            let mut acc = S::zero();
            for j in 0..n {
                acc = self[(i, j)].mul_add(x[j], acc);
            }
            y[i] += alpha * acc;
        }
        Ok(())
    }

    fn mat_vec_trans(
        &self,
        alpha: S,
        x: &[S],
        beta: S,
        y: &mut [S],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.nrows() || y.len() != self.ncols() {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".into(),
            ));
        }

        if beta == S::zero() {
            for value in y.iter_mut() {
                *value = S::zero();
            }
        } else if beta != S::one() {
            for value in y.iter_mut() {
                *value *= beta;
            }
        }

        let m = self.nrows();
        let n = self.ncols();
        for j in 0..n {
            let mut acc = S::zero();
            for i in 0..m {
                acc = self[(i, j)].mul_add(x[i], acc);
            }
            y[j] += alpha * acc;
        }
        Ok(())
    }

    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }
}

/// Standard dot product implementation
pub struct StandardDotOp;

impl DotOp<S> for StandardDotOp {
    fn dot(&self, x: &[S], y: &[S]) -> S {
        par_dot_conj_local(x, y)
    }

    fn norm2(&self, x: &[S]) -> S {
        S::from_real(par_sum_abs2_local(x).sqrt())
    }
}

#[cfg(feature = "complex")]
impl DotOp<f64> for StandardDotOp {
    fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn norm2(&self, x: &[f64]) -> f64 {
        self.dot(x, x).sqrt()
    }
}

/// Unified kernel trait for local vs distributed operations
/// Provides a consistent interface for AMG operations that can work
/// both in single-process (local) and multi-process (MPI) scenarios
pub trait KernelOp<T: KrystScalar> {
    /// The communicator type for this kernel (e.g., UniverseComm for MPI, () for local)
    type Comm: crate::parallel::Comm;

    /// Matrix-vector product with communicator support: y = alpha * A * x + beta * y
    fn kernel_mat_vec(
        &self,
        matrix: &dyn MatVecOp<T>,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>;

    /// Transpose matrix-vector product: y = alpha * A^T * x + beta * y (A^H for complex).
    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<T>,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>;

    /// Global dot product with reduction across processes
    fn kernel_dot(&self, x: &[T], y: &[T], comm: &Self::Comm) -> T;

    /// Global norm computation with reduction
    fn kernel_norm2(&self, x: &[T], comm: &Self::Comm) -> T;

    /// Vector operations: y = alpha * x + beta * y
    fn kernel_axpby(&self, alpha: T, x: &[T], beta: T, y: &mut [T]);

    /// Copy operation: y = x
    fn kernel_copy(&self, x: &[T], y: &mut [T]);

    /// Scale operation: x = alpha * x
    fn kernel_scale(&self, alpha: T, x: &mut [T]);

    /// Apply a local preconditioner: y = M^{-1} x.
    fn kernel_apply_preconditioner<P>(
        &self,
        preconditioner: &P,
        x: &[T],
        y: &mut [T],
        _comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>
    where
        P: LocalPreconditioner<T>,
    {
        let (m, n) = preconditioner.dims();
        if (m != 0 && x.len() != m) || (n != 0 && y.len() != n) {
            return Err(crate::error::KError::InvalidInput(format!(
                "Preconditioner dimension mismatch: dims=({m}, {n}), x.len()={}, y.len()={}",
                x.len(),
                y.len()
            )));
        }

        preconditioner.apply_local(x, y)
    }
}

/// Local (single-process) kernel implementation
pub struct LocalKernel;

impl KernelOp<S> for LocalKernel {
    type Comm = crate::parallel::NoComm;

    fn kernel_mat_vec(
        &self,
        matrix: &dyn MatVecOp<S>,
        alpha: S,
        x: &[S],
        beta: S,
        y: &mut [S],
        _comm: &Self::Comm,
    ) -> Result<(), crate::error::KError> {
        // For local operations, no communication needed
        matrix.mat_vec(alpha, x, beta, y)
    }

    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<S>,
        alpha: S,
        x: &[S],
        beta: S,
        y: &mut [S],
        _comm: &Self::Comm,
    ) -> Result<(), crate::error::KError> {
        matrix.mat_vec_trans(alpha, x, beta, y)
    }

    fn kernel_dot(&self, x: &[S], y: &[S], _comm: &Self::Comm) -> S {
        par_dot_conj_local(x, y)
    }

    fn kernel_norm2(&self, x: &[S], _comm: &Self::Comm) -> S {
        let norm_sq = par_sum_abs2_local(x);
        S::from_real(norm_sq.sqrt())
    }

    fn kernel_axpby(&self, alpha: S, x: &[S], beta: S, y: &mut [S]) {
        parallel::par_axpby(x, alpha, y, beta);
    }

    fn kernel_copy(&self, x: &[S], y: &mut [S]) {
        parallel::par_copy(x, y);
    }

    fn kernel_scale(&self, alpha: S, x: &mut [S]) {
        parallel::par_scale(alpha, x);
    }
}

/// Distributed (MPI) kernel implementation for future use
/// Currently a placeholder that delegates to local operations
#[cfg(feature = "mpi")]
pub struct DistributedKernel;

#[cfg(feature = "mpi")]
impl KernelOp<S> for DistributedKernel {
    type Comm = crate::parallel::UniverseComm;

    fn kernel_mat_vec(
        &self,
        matrix: &dyn MatVecOp<S>,
        alpha: S,
        x: &[S],
        beta: S,
        y: &mut [S],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError> {
        let mut local = vec![S::zero(); y.len()];
        matrix.mat_vec(alpha, x, S::zero(), &mut local)?;
        comm.allreduce_sum_scalars(&mut local);

        if beta == S::zero() {
            parallel::par_copy(&local, y);
        } else {
            parallel::par_axpby(&local, S::one(), y, beta);
        }

        Ok(())
    }

    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<S>,
        alpha: S,
        x: &[S],
        beta: S,
        y: &mut [S],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError> {
        let mut local = vec![S::zero(); y.len()];
        matrix.mat_vec_trans(alpha, x, S::zero(), &mut local)?;
        comm.allreduce_sum_scalars(&mut local);

        if beta == S::zero() {
            parallel::par_copy(&local, y);
        } else {
            parallel::par_axpby(&local, S::one(), y, beta);
        }

        Ok(())
    }

    fn kernel_dot(&self, x: &[S], y: &[S], comm: &Self::Comm) -> S {
        let local_dot = par_dot_conj_local(x, y);
        comm.allreduce_sum_scalar(local_dot)
    }

    fn kernel_norm2(&self, x: &[S], comm: &Self::Comm) -> S {
        let local_sq = par_sum_abs2_local(x);
        let global_sq = comm.allreduce_sum_real(local_sq);
        S::from_real(global_sq.sqrt())
    }

    fn kernel_axpby(&self, alpha: S, x: &[S], beta: S, y: &mut [S]) {
        parallel::par_axpby(x, alpha, y, beta);
    }

    fn kernel_copy(&self, x: &[S], y: &mut [S]) {
        parallel::par_copy(x, y);
    }

    fn kernel_scale(&self, alpha: S, x: &mut [S]) {
        parallel::par_scale(alpha, x);
    }
}

/// Unified AMG kernel trait to eliminate code duplication between local and MPI variants
pub trait AmgKernel {
    /// Associated communicator type
    type Comm: crate::parallel::Comm;

    /// Matrix-vector multiplication with alpha/beta scaling
    fn matvec<M>(
        &self,
        alpha: S,
        a: &M,
        x: &[S],
        beta: S,
        y: &mut [S],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>
    where
        M: MatVecOp<S>;

    /// Global dot product with communicator reduction
    fn dot(&self, x: &[S], y: &[S], comm: &Self::Comm) -> S;

    /// Global norm with communicator reduction
    fn norm(&self, x: &[S], comm: &Self::Comm) -> S {
        S::from_real(self.dot(x, x, comm).abs().sqrt())
    }

    /// Vector scaling: x = alpha * x
    fn scale(&self, alpha: S, x: &mut [S]);

    /// Vector copy: y = x
    fn copy(&self, x: &[S], y: &mut [S]);

    /// AXPY operation: y = alpha * x + y
    fn axpy(&self, alpha: S, x: &[S], y: &mut [S]);

    /// Apply a local preconditioner: y = M^{-1} x.
    fn apply_preconditioner<P>(
        &self,
        preconditioner: &P,
        x: &[S],
        y: &mut [S],
        _comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>
    where
        P: LocalPreconditioner<S>,
    {
        let (m, n) = preconditioner.dims();
        if (m != 0 && x.len() != m) || (n != 0 && y.len() != n) {
            return Err(crate::error::KError::InvalidInput(format!(
                "Preconditioner dimension mismatch: dims=({m}, {n}), x.len()={}, y.len()={}",
                x.len(),
                y.len()
            )));
        }

        preconditioner.apply_local(x, y)
    }
}

/// Local (single-process) AMG kernel implementation
pub struct LocalAmgKernel;

impl LocalAmgKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LocalAmgKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl AmgKernel for LocalAmgKernel {
    type Comm = crate::parallel::NoComm;

    fn matvec<M>(
        &self,
        alpha: S,
        a: &M,
        x: &[S],
        beta: S,
        y: &mut [S],
        _comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>
    where
        M: MatVecOp<S>,
    {
        a.mat_vec(alpha, x, beta, y)
    }

    fn dot(&self, x: &[S], y: &[S], _comm: &Self::Comm) -> S {
        par_dot_conj_local(x, y)
    }

    fn scale(&self, alpha: S, x: &mut [S]) {
        parallel::par_scale(alpha, x);
    }

    fn copy(&self, x: &[S], y: &mut [S]) {
        parallel::par_copy(x, y);
    }

    fn axpy(&self, alpha: S, x: &[S], y: &mut [S]) {
        parallel::par_axpy(x, alpha, y);
    }
}

#[cfg(feature = "mpi")]
/// Distributed (MPI) AMG kernel implementation
pub struct DistributedAmgKernel;

#[cfg(feature = "mpi")]
impl DistributedAmgKernel {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "mpi")]
impl Default for DistributedAmgKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "mpi")]
impl AmgKernel for DistributedAmgKernel {
    type Comm = crate::parallel::UniverseComm;

    fn matvec<M>(
        &self,
        alpha: S,
        a: &M,
        x: &[S],
        beta: S,
        y: &mut [S],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>
    where
        M: MatVecOp<S>,
    {
        let mut local = vec![S::zero(); y.len()];
        a.mat_vec(alpha, x, S::zero(), &mut local)?;
        comm.allreduce_sum_scalars(&mut local);

        if beta == S::zero() {
            parallel::par_copy(&local, y);
        } else {
            parallel::par_axpby(&local, S::one(), y, beta);
        }

        Ok(())
    }

    fn dot(&self, x: &[S], y: &[S], comm: &Self::Comm) -> S {
        let local_dot = par_dot_conj_local(x, y);
        comm.allreduce_sum_scalar(local_dot)
    }

    fn scale(&self, alpha: S, x: &mut [S]) {
        parallel::par_scale(alpha, x);
    }

    fn copy(&self, x: &[S], y: &mut [S]) {
        parallel::par_copy(x, y);
    }

    fn axpy(&self, alpha: S, x: &[S], y: &mut [S]) {
        parallel::par_axpy(x, alpha, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::CsrMatrix;
    use crate::parallel::NoComm;

    // Simple test to verify traits can be imported and used
    #[test]
    fn test_traits_exist() {
        // This test just verifies that all traits compile and can be referenced
        // More comprehensive tests would require mock implementations

        // Test that trait bounds can be specified
        fn _test_matvec_bound<T, V>(_: &T)
        where
            T: MatVec<V>,
        {
        }
        fn _test_mattransvec_bound<T, V>(_: &T)
        where
            T: MatTransVec<V>,
        {
        }
        fn _test_inner_product_bound<T, V>(_: &T)
        where
            T: InnerProduct<V>,
        {
        }
        fn _test_indexing_bound<T>(_: &T)
        where
            T: Indexing,
        {
        }
        fn _test_mat_shape_bound<T>(_: &T)
        where
            T: MatShape,
        {
        }
        fn _test_row_pattern_bound<T>(_: &T)
        where
            T: RowPattern,
        {
        }
        fn _test_matrix_get_bound<T, U>(_: &T)
        where
            T: MatrixGet<U>,
        {
        }
        fn _test_submatrix_extract_bound<T>(_: &T)
        where
            T: SubmatrixExtract,
        {
        }

        // All traits should compile
        assert!(true);
    }

    #[test]
    fn test_inner_product_scalar_trait_bounds() {
        // Test that the associated Scalar type has the required bounds
        fn _check_scalar_bounds<T: Copy + PartialOrd + From<f64> + Into<f64>>() {}

        // Use the real partner of S for bound checking
        _check_scalar_bounds::<R>();

        assert!(true);
    }

    #[test]
    fn test_trait_names_and_methods() {
        // Verify method names exist by checking trait signatures
        trait TestMatVec<V> {
            fn matvec(&self, x: &V, y: &mut V);
        }

        trait TestMatTransVec<V> {
            fn mattransvec(&self, x: &V, y: &mut V);
        }

        trait TestInnerProduct<V> {
            type Scalar: Copy;
            fn dot(&self, x: &V, y: &V, _comm: &impl crate::parallel::Comm) -> Self::Scalar;
        }
        struct Dummy;

        impl TestMatVec<Vec<S>> for Dummy {
            fn matvec(&self, _x: &Vec<S>, _y: &mut Vec<S>) {}
        }

        impl TestMatTransVec<Vec<S>> for Dummy {
            fn mattransvec(&self, _x: &Vec<S>, _y: &mut Vec<S>) {}
        }

        impl TestInnerProduct<Vec<S>> for Dummy {
            type Scalar = S;
            fn dot(
                &self,
                _x: &Vec<S>,
                _y: &Vec<S>,
                _comm: &impl crate::parallel::Comm,
            ) -> Self::Scalar {
                S::zero()
            }
        }

        fn _use_traits<
            T: TestMatVec<Vec<S>> + TestMatTransVec<Vec<S>> + TestInnerProduct<Vec<S>>,
        >() {
        }
        _use_traits::<Dummy>();

        let dummy = Dummy;
        let comm = crate::parallel::NoComm;
        let v = vec![S::zero(); 1];
        let mut y = vec![S::zero(); 1];
        dummy.matvec(&v, &mut y);
        dummy.mattransvec(&v, &mut y);
        let _ = dummy.dot(&v, &v, &comm);

        // All method signatures should compile without panicking.
    }

    #[test]
    fn csr_matvec_happy_path() {
        // 2x3 CSR: row_ptr=[0,2,3], col_idx=[0,2,1], val=[1,4,5]
        // A = [1 0 4; 0 5 0]
        let a = CsrMatrix::from_csr(
            2,
            3,
            vec![0, 2, 3],
            vec![0, 2, 1],
            vec![S::from_real(1.0), S::from_real(4.0), S::from_real(5.0)],
        );
        let x = [S::from_real(10.0), S::from_real(20.0), S::from_real(30.0)];
        let mut y = [S::zero(); 2];
        MatVecOp::mat_vec(&a, S::one(), &x, S::zero(), &mut y).unwrap();
        let expected = [S::from_real(130.0), S::from_real(100.0)];
        for (got, target) in y.iter().zip(expected.iter()) {
            assert!(((*got) - *target).abs() < 1e-12);
        }
        // with scaling
        let mut y2 = [S::from_real(1.0), S::from_real(2.0)];
        MatVecOp::mat_vec(&a, S::from_real(2.0), &x, S::from_real(3.0), &mut y2).unwrap();
        // 2*A*x + 3*y0
        let expect0 = S::from_real(2.0 * 130.0 + 3.0 * 1.0);
        let expect1 = S::from_real(2.0 * 100.0 + 3.0 * 2.0);
        assert!((y2[0] - expect0).abs() < 1e-12);
        assert!((y2[1] - expect1).abs() < 1e-12);
    }

    struct DummyPc;

    impl LocalPreconditioner for DummyPc {
        fn dims(&self) -> (usize, usize) {
            (2, 2)
        }

        fn apply_local(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
            y.copy_from_slice(x);
            Ok(())
        }
    }

    #[test]
    fn kernel_apply_preconditioner_respects_dims() {
        let pc = DummyPc;
        let kernel = LocalKernel;
        let comm = NoComm;

        let mut y = vec![S::zero(); 2];
        kernel
            .kernel_apply_preconditioner(&pc, &[S::one(), S::from_real(2.0)], &mut y, &comm)
            .unwrap();
        assert_eq!(y, vec![S::one(), S::from_real(2.0)]);

        let err = kernel.kernel_apply_preconditioner(&pc, &[S::one()], &mut y, &comm);
        assert!(matches!(err, Err(KError::InvalidInput(_))));
    }
}
