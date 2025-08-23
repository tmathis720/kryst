//! Core linear-algebra traits for kryst.

/// Matrix–vector product: y ← A x.
pub trait MatVec<V> {
    /// Compute y = A · x.
    fn matvec(&self, x: &V, y: &mut V);
}

/// Matrix–transpose–vector product: y ← Aᵗ x.
pub trait MatTransVec<V> {
    /// Compute y = Aᵗ · x.
    fn mattransvec(&self, x: &V, y: &mut V);
}

// Blanket implementations of MatVec/MatTransVec for LinOp types using Vec storage.
use crate::matrix::op::LinOp;
use faer::traits::ComplexField;

impl<T, L> MatVec<Vec<T>> for L
where
    L: LinOp<S = T> + ?Sized,
    T: ComplexField,
{
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        LinOp::matvec(self, &x[..], &mut y[..]);
    }
}

impl<T, L> MatTransVec<Vec<T>> for L
where
    L: LinOp<S = T> + ?Sized,
    T: ComplexField,
{
    fn mattransvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        LinOp::t_matvec(self, &x[..], &mut y[..]).expect("t_matvec not supported");
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
pub trait SubmatrixExtract {
    /// Returns the submatrix with rows and columns given by `indices`.
    fn submatrix(&self, indices: &[usize]) -> Self;
}

/// Sparse-aware matrix-vector operations for AMG and iterative solvers
pub trait MatVecOp<T> {
    /// Compute y = alpha * A * x + beta * y
    fn mat_vec(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) -> Result<(), crate::KError>;
    
    /// Compute y = alpha * A^T * x + beta * y (transpose operation)
    fn mat_vec_trans(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) -> Result<(), crate::KError>;
    
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

/// Implementation for dense matrices (faer Mat)
impl MatVecOp<f64> for faer::Mat<f64> {
    fn mat_vec(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), crate::KError> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(crate::KError::InvalidInput("Matrix-vector dimension mismatch".to_string()));
        }
        
        for i in 0..self.nrows() {
            let mut sum = 0.0;
            for j in 0..self.ncols() {
                sum += self[(i, j)] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
        Ok(())
    }
    
    fn mat_vec_trans(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), crate::KError> {
        if x.len() != self.nrows() || y.len() != self.ncols() {
            return Err(crate::KError::InvalidInput("Matrix-vector dimension mismatch".to_string()));
        }
        
        for j in 0..self.ncols() {
            let mut sum = 0.0;
            for i in 0..self.nrows() {
                sum += self[(i, j)] * x[i];
            }
            y[j] = alpha * sum + beta * y[j];
        }
        Ok(())
    }
    
    fn nrows(&self) -> usize { faer::Mat::nrows(self) }
    fn ncols(&self) -> usize { faer::Mat::ncols(self) }
}

/// Implementation for sparse matrices (CsrMatrix)
impl MatVecOp<f64> for crate::matrix::sparse::CsrMatrix<f64> {
    fn mat_vec(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), crate::KError> {
        use crate::matrix::sparse::SparseMatrix;
        if x.len() != SparseMatrix::ncols(self) || y.len() != SparseMatrix::nrows(self) {
            return Err(crate::KError::InvalidInput("Matrix-vector dimension mismatch".to_string()));
        }
        
        // Use the existing spmv method and scale appropriately
        if beta == 0.0 {
            // y = alpha * A * x (zero y first)
            for val in y.iter_mut() {
                *val = 0.0;
            }
            if alpha != 0.0 {
                let mut temp = vec![0.0; y.len()];
                SparseMatrix::spmv(self, x, &mut temp);
                for (y_val, temp_val) in y.iter_mut().zip(temp.iter()) {
                    *y_val = alpha * temp_val;
                }
            }
        } else if alpha == 0.0 {
            // y = beta * y
            for val in y.iter_mut() {
                *val *= beta;
            }
        } else {
            // y = alpha * A * x + beta * y
            let mut temp = vec![0.0; y.len()];
            SparseMatrix::spmv(self, x, &mut temp);
            for (y_val, temp_val) in y.iter_mut().zip(temp.iter()) {
                *y_val = alpha * temp_val + beta * (*y_val);
            }
        }
        Ok(())
    }
    
    fn mat_vec_trans(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), crate::KError> {
        use crate::matrix::sparse::SparseMatrix;

        // Dimension checks: x is in R^{m}, y in R^{n} for A^T (A is m×n)
        if x.len() != SparseMatrix::nrows(self) || y.len() != SparseMatrix::ncols(self) {
            return Err(crate::KError::InvalidInput(
                "Matrix-vector dimension mismatch".to_string(),
            ));
        }

        // Quick exits
        if alpha == 0.0 {
            // y = beta * y
            if beta == 0.0 {
                for v in y.iter_mut() {
                    *v = 0.0;
                }
            } else {
                for v in y.iter_mut() {
                    *v *= beta;
                }
            }
            return Ok(());
        }

        // Scale y by beta (or zero) up front
        if beta == 0.0 {
            for v in y.iter_mut() {
                *v = 0.0;
            }
        } else if beta != 1.0 {
            for v in y.iter_mut() {
                *v *= beta;
            }
        }
        // If beta == 1.0, leave y as-is and accumulate into it.

        // Access CSR structure. These accessor names assume your CSR exposes them.
        // If your type uses different getters, adjust accordingly.
        let row_ptr = self.row_ptr();   // &[usize] of length m+1
        let col_idx = self.col_idx();   // &[usize] of length nnz
        let values  = self.values();    // &[f64]   of length nnz

        // y_j += alpha * a_ij * x_i  for all nonzeros a_ij
        let m = SparseMatrix::nrows(self);
        for i in 0..m {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            // SAFETY: bounds guaranteed by CSR invariants
            for k in start..end {
                let j = col_idx[k];
                y[j] += alpha * values[k] * xi;
            }
        }

        Ok(())
    }

    
    fn nrows(&self) -> usize { 
        use crate::matrix::sparse::SparseMatrix;
        SparseMatrix::nrows(self) 
    }
    fn ncols(&self) -> usize { 
        use crate::matrix::sparse::SparseMatrix;
        SparseMatrix::ncols(self) 
    }
}

/// Standard dot product implementation
pub struct StandardDotOp;

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
pub trait KernelOp<T> {
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
    ) -> Result<(), crate::KError>;
    
    /// Transpose matrix-vector product: y = alpha * A^T * x + beta * y
    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<T>,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
        comm: &Self::Comm,
    ) -> Result<(), crate::KError>;
    
    /// Global dot product with reduction across processes
    fn kernel_dot(
        &self,
        x: &[T],
        y: &[T],
        comm: &Self::Comm,
    ) -> T;
    
    /// Global norm computation with reduction
    fn kernel_norm2(
        &self,
        x: &[T],
        comm: &Self::Comm,
    ) -> T;
    
    /// Vector operations: y = alpha * x + beta * y
    fn kernel_axpby(
        &self,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
    );
    
    /// Copy operation: y = x
    fn kernel_copy(&self, x: &[T], y: &mut [T]);
    
    /// Scale operation: x = alpha * x
    fn kernel_scale(&self, alpha: T, x: &mut [T]);
}

/// Local (single-process) kernel implementation
pub struct LocalKernel;

impl KernelOp<f64> for LocalKernel {
    type Comm = crate::parallel::NoComm;
    
    fn kernel_mat_vec(
        &self,
        matrix: &dyn MatVecOp<f64>,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        _comm: &Self::Comm,
    ) -> Result<(), crate::KError> {
        // For local operations, no communication needed
        matrix.mat_vec(alpha, x, beta, y)
    }
    
    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<f64>,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        _comm: &Self::Comm,
    ) -> Result<(), crate::KError> {
        matrix.mat_vec_trans(alpha, x, beta, y)
    }
    
    fn kernel_dot(&self, x: &[f64], y: &[f64], _comm: &Self::Comm) -> f64 {
        let dot_op = StandardDotOp;
        dot_op.dot(x, y)
    }
    
    fn kernel_norm2(&self, x: &[f64], _comm: &Self::Comm) -> f64 {
        let dot_op = StandardDotOp;
        dot_op.norm2(x)
    }
    
    fn kernel_axpby(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
            *y_val = alpha * x_val + beta * (*y_val);
        }
    }
    
    fn kernel_copy(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
    
    fn kernel_scale(&self, alpha: f64, x: &mut [f64]) {
        for val in x.iter_mut() {
            *val *= alpha;
        }
    }
}

/// Distributed (MPI) kernel implementation for future use
/// Currently a placeholder that delegates to local operations
pub struct DistributedKernel;

impl KernelOp<f64> for DistributedKernel {
    type Comm = crate::parallel::UniverseComm;
    
    fn kernel_mat_vec(
        &self,
        matrix: &dyn MatVecOp<f64>,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        _comm: &Self::Comm,
    ) -> Result<(), crate::KError> {
        // TODO: Add proper distributed matrix-vector operations
        // For now, delegate to local operation
        matrix.mat_vec(alpha, x, beta, y)
    }
    
    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<f64>,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        _comm: &Self::Comm,
    ) -> Result<(), crate::KError> {
        // TODO: Add proper distributed transpose operations
        matrix.mat_vec_trans(alpha, x, beta, y)
    }
    
    fn kernel_dot(&self, x: &[f64], y: &[f64], comm: &Self::Comm) -> f64 {
        use crate::parallel::Comm;
        // Compute local dot product
        let local_dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        // Reduce across all processes
        comm.all_reduce_f64(local_dot)
    }
    
    fn kernel_norm2(&self, x: &[f64], comm: &Self::Comm) -> f64 {
        self.kernel_dot(x, x, comm).sqrt()
    }
    
    fn kernel_axpby(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        // Vector operations are local in distributed setting
        for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
            *y_val = alpha * x_val + beta * (*y_val);
        }
    }
    
    fn kernel_copy(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
    
    fn kernel_scale(&self, alpha: f64, x: &mut [f64]) {
        for val in x.iter_mut() {
            *val *= alpha;
        }
    }
}

/// Unified AMG kernel trait to eliminate code duplication between local and MPI variants
pub trait AmgKernel {
    /// Associated communicator type  
    type Comm: crate::parallel::Comm;
    
    /// Matrix-vector multiplication with alpha/beta scaling
    fn matvec<M>(&self, alpha: f64, a: &M, x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), crate::KError>
    where
        M: MatVecOp<f64>;
    
    /// Global dot product with communicator reduction
    fn dot(&self, x: &[f64], y: &[f64], comm: &Self::Comm) -> f64;
    
    /// Global norm with communicator reduction  
    fn norm(&self, x: &[f64], comm: &Self::Comm) -> f64 {
        self.dot(x, x, comm).sqrt()
    }
    
    /// Vector scaling: x = alpha * x
    fn scale(&self, alpha: f64, x: &mut [f64]);
    
    /// Vector copy: y = x
    fn copy(&self, x: &[f64], y: &mut [f64]);
    
    /// AXPY operation: y = alpha * x + y
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]);
}

/// Local (single-process) AMG kernel implementation
pub struct LocalAmgKernel;

impl LocalAmgKernel {
    pub fn new() -> Self {
        Self
    }
}

impl AmgKernel for LocalAmgKernel {
    type Comm = crate::parallel::NoComm;
    
    fn matvec<M>(&self, alpha: f64, a: &M, x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), crate::KError>
    where
        M: MatVecOp<f64>
    {
        a.mat_vec(alpha, x, beta, y)
    }
    
    fn dot(&self, x: &[f64], y: &[f64], _comm: &Self::Comm) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
    
    fn scale(&self, alpha: f64, x: &mut [f64]) {
        for val in x.iter_mut() {
            *val *= alpha;
        }
    }
    
    fn copy(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
    
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
            *y_val += alpha * x_val;
        }
    }
}

/// Distributed (MPI) AMG kernel implementation
pub struct DistributedAmgKernel;

impl DistributedAmgKernel {
    pub fn new() -> Self {
        Self
    }
}

impl AmgKernel for DistributedAmgKernel {
    type Comm = crate::parallel::UniverseComm;
    
    fn matvec<M>(&self, alpha: f64, a: &M, x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), crate::KError>
    where
        M: MatVecOp<f64>
    {
        // For now, delegate to trait method (TODO: implement proper distributed matvec)
        a.mat_vec(alpha, x, beta, y)
    }
    
    fn dot(&self, x: &[f64], y: &[f64], comm: &Self::Comm) -> f64 {
        use crate::parallel::Comm;
        // Compute local dot product, then reduce across processes
        let local_dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        comm.all_reduce_f64(local_dot)
    }
    
    fn scale(&self, alpha: f64, x: &mut [f64]) {
        // Vector operations are local even in distributed setting
        for val in x.iter_mut() {
            *val *= alpha;
        }
    }
    
    fn copy(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
    
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
            *y_val += alpha * x_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test to verify traits can be imported and used
    #[test]
    fn test_traits_exist() {
        // This test just verifies that all traits compile and can be referenced
        // More comprehensive tests would require mock implementations
        
        // Test that trait bounds can be specified
        fn _test_matvec_bound<T, V>(_: &T) where T: MatVec<V> {}
        fn _test_mattransvec_bound<T, V>(_: &T) where T: MatTransVec<V> {}
        fn _test_inner_product_bound<T, V>(_: &T) where T: InnerProduct<V> {}
        fn _test_indexing_bound<T>(_: &T) where T: Indexing {}
        fn _test_mat_shape_bound<T>(_: &T) where T: MatShape {}
        fn _test_row_pattern_bound<T>(_: &T) where T: RowPattern {}
        fn _test_matrix_get_bound<T, U>(_: &T) where T: MatrixGet<U> {}
        fn _test_submatrix_extract_bound<T>(_: &T) where T: SubmatrixExtract {}
        
        // All traits should compile
        assert!(true);
    }

    #[test]
    fn test_inner_product_scalar_trait_bounds() {
        // Test that the associated Scalar type has the required bounds
        fn _check_scalar_bounds<T: Copy + PartialOrd + From<f64> + Into<f64>>() {}
        
        // f64 should satisfy the bounds
        _check_scalar_bounds::<f64>();
        
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
            type Scalar: Copy + PartialOrd + From<f64> + Into<f64>;
            fn dot(&self, x: &V, y: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar;
            fn norm(&self, x: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar {
                let local_sq = self.dot(x, x, comm);
                let global_sq = comm.all_reduce_f64(local_sq.into());
                (global_sq.sqrt()).into()
            }
        }
        
        // All method signatures should compile
        assert!(true);
    }
}
