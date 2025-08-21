// SparseMatrix trait and implementations (CSR, CSC)

/// A read‐only sparse matrix supporting y = A * x.
pub trait SparseMatrix<T> {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Compute y = A * x.  `x.len() == ncols()`, `y.len() == nrows()`.
    fn spmv(&self, x: &[T], y: &mut [T]);
}

use faer::sparse::{
    SymbolicSparseRowMat,    // owning symbolic CSR alias
    SparseRowMat,            // owning numeric CSR alias
    //CreationError,           // error type for builders
};
use faer::traits::ComplexField;
//use faer::sparse::linalg::matmul::sparse_dense_matmul;

/// CSR matrix wrapper for Faer sparse matrices.
#[derive(Clone)]
pub struct CsrMatrix<T> {
    inner: SparseRowMat<usize, T>,
}

impl<T: ComplexField + Copy + num_traits::Zero + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T>> CsrMatrix<T> {
    /// Build a CSR from raw row‐ptr, col‐idx, and values.
    pub fn from_csr(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        // Build symbolic structure; second argument `None` means "no separate row_nnz":
        let symbolic = SymbolicSparseRowMat::new_checked(
            nrows,
            ncols,
            row_ptr,
            None,      // optional row_nnz: Option<Vec<usize>>
            col_idx,
        );
        // Attach the numerical values:
        let inner = SparseRowMat::new(symbolic, values);
        Self { inner }
    }

    /// Convert from dense faer::Mat to sparse CSR format with drop tolerance
    pub fn from_dense(dense: &faer::Mat<T>, drop_tol: T) -> Self 
    where T: PartialOrd + std::ops::Neg<Output = T>
    {
        let nrows = dense.nrows();
        let ncols = dense.ncols();
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        
        for i in 0..nrows {
            for j in 0..ncols {
                let val = dense[(i, j)];
                // Use comparison with tolerance
                if val > drop_tol || val < -drop_tol {
                    col_idx.push(j);
                    values.push(val);
                }
            }
            row_ptr.push(col_idx.len());
        }
        
        Self::from_csr(nrows, ncols, row_ptr, col_idx, values)
    }

    /// Create an identity matrix of size n x n
    pub fn identity(n: usize) -> Self 
    where T: num_traits::One
    {
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let values: Vec<T> = vec![T::one(); n];
        
        Self::from_csr(n, n, row_ptr, col_idx, values)
    }

    /// Convert to dense faer::Mat for use with dense solvers.
    pub fn to_dense(&self) -> faer::Mat<T> {
        self.inner.to_dense()
    }

    /// Get matrix dimensions
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// Get number of nonzeros  
    pub fn nnz(&self) -> usize {
        self.inner.compute_nnz()
    }

    /// Extract diagonal as a vector
    pub fn diagonal(&self) -> Vec<T> {
        let n = self.nrows().min(self.ncols());
        let mut diag = vec![T::zero(); n];
        
        for i in 0..n {
            let row_start = self.inner.row_ptr()[i];
            let row_end = self.inner.row_ptr()[i + 1];
            
            for idx in row_start..row_end {
                if self.inner.col_idx()[idx] == i {
                    diag[i] = self.inner.val()[idx];
                    break;
                }
            }
        }
        
        diag
    }

    /// Sparse matrix-vector product: y = alpha * A * x + beta * y
    pub fn spmv_scaled(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) -> Result<(), crate::KError> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(crate::KError::InvalidInput(format!(
                "Dimension mismatch in spmv: A={}x{}, x.len()={}, y.len()={}",
                self.nrows(), self.ncols(), x.len(), y.len()
            )));
        }

        for i in 0..self.nrows() {
            let row_start = self.inner.row_ptr()[i];
            let row_end = self.inner.row_ptr()[i + 1];
            
            let mut sum = T::zero();
            for idx in row_start..row_end {
                let j = self.inner.col_idx()[idx];
                sum = sum + self.inner.val()[idx] * x[j];
            }
            
            y[i] = alpha * sum + beta * y[i];
        }
        
        Ok(())
    }
    
    /// Access to row pointers (indices into col_indices and values arrays)
    /// Note: This is a simplified implementation using dense conversion
    pub fn to_row_ptr_vec(&self) -> Vec<usize> {
        // For now, we'll reconstruct CSR from dense (inefficient but works)
        let dense = self.to_dense();
        let mut row_ptrs = vec![0];
        let mut nnz = 0;
        
        for i in 0..self.inner.nrows() {
            for j in 0..self.inner.ncols() {
                if dense[(i, j)] != T::zero() {
                    nnz += 1;
                }
            }
            row_ptrs.push(nnz);
        }
        row_ptrs
    }
    
    /// Access to column indices array
    /// Note: This is a simplified implementation using dense conversion
    pub fn to_col_idx_vec(&self) -> Vec<usize> {
        let dense = self.to_dense();
        let mut col_indices = Vec::new();
        
        for i in 0..self.inner.nrows() {
            for j in 0..self.inner.ncols() {
                if dense[(i, j)] != T::zero() {
                    col_indices.push(j);
                }
            }
        }
        col_indices
    }
    
    /// Access to values array
    /// Note: This is a simplified implementation using dense conversion
    pub fn to_values_vec(&self) -> Vec<T> {
        let dense = self.to_dense();
        let mut values = Vec::new();
        
        for i in 0..self.inner.nrows() {
            for j in 0..self.inner.ncols() {
                let val = dense[(i, j)];
                if val != T::zero() {
                    values.push(val);
                }
            }
        }
        values
    }

        /// Borrow the CSR row pointer array (length = nrows + 1).
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        self.inner.row_ptr()
    }

    /// Borrow the CSR column index array (length = nnz).
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        self.inner.col_idx()
    }

    /// Borrow the CSR value array (length = nnz).
    #[inline]
    pub fn values(&self) -> &[T] {
        self.inner.val()
    }
}

impl<T: ComplexField + Copy + num_traits::One + num_traits::Zero> SparseMatrix<T> for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    fn spmv(&self, x: &[T], y: &mut [T]) {
        // Simple implementation using direct sparse matrix-vector product
        // Reset y to zero
        for i in 0..y.len() {
            y[i] = T::zero();
        }
        
        // Sparse matrix-vector multiplication
        for i in 0..self.inner.nrows() {
            let row_start = self.inner.row_ptr()[i];
            let row_end = self.inner.row_ptr()[i + 1];
            for idx in row_start..row_end {
                let j = self.inner.col_idx()[idx];
                y[i] = y[i] + self.inner.val()[idx] * x[j];
            }
        }
    }
}

// Implement MatVec trait for CsrMatrix to work with Kryst solvers
use crate::core::traits::Indexing;

// Implement Indexing trait for CsrMatrix to work with preconditioners
impl<T: ComplexField + Copy + num_traits::One + num_traits::Zero> Indexing for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        SparseMatrix::nrows(self)
    }
}

use crate::core::traits::SubmatrixExtract;

impl<T: ComplexField + Copy + num_traits::Zero + num_traits::One + PartialEq + PartialOrd> SubmatrixExtract for CsrMatrix<T> {
    fn submatrix(&self, indices: &[usize]) -> Self {
        let dense = self.inner.to_dense();
        let n = indices.len();
        let sub = faer::Mat::from_fn(n, n, |i, j| dense[(indices[i], indices[j])]);
        // Convert dense submatrix to CSR (inefficient fallback)
        let mut row_ptr = vec![0; n + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let v = sub[(i, j)];
                if v != T::zero() {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr[i + 1] = col_idx.len();
        }
        CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
    }
}

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use rayon::iter::IntoParallelRefMutIterator;

#[cfg(feature = "rayon")]
impl<T: ComplexField + Copy + num_traits::One + num_traits::Zero + Send + Sync> CsrMatrix<T> {
    /// Parallel SpMV using Rayon.
    pub fn spmv_parallel(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), SparseMatrix::nrows(self));
        let dense = self.inner.to_dense();
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = T::zero();
            for j in 0..self.ncols() {
                sum = sum + dense[(i, j)] * x[j];
            }
            *yi = sum;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_spmv() {
        // 3×3 identity in CSR: row_ptr=[0,1,2,3], col_idx=[0,1,2], vals=[1,1,1]
        let m = CsrMatrix::from_csr(3, 3, vec![0,1,2,3], vec![0,1,2], vec![1.0,1.0,1.0]);
        let x = vec![2.0, 3.0, 5.0];
        let mut y = vec![0.0; 3];
        m.spmv_scaled(1.0, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn simple_pattern() {
        // 2×3 matrix [[1,2,0],[0,3,4]]
        let m = CsrMatrix::from_csr(
            2, 3,
            vec![0,2,4],
            vec![0,1,1,2],
            vec![1.0,2.0,3.0,4.0],
        );
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];
        m.spmv_scaled(1.0, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, vec![3.0, 7.0]);
    }
}
