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

pub struct CsrMatrix<T> {
    inner: SparseRowMat<usize, T>,
}

impl<T: ComplexField + Copy> CsrMatrix<T> {
    /// Build a CSR from raw row‐ptr, col‐idx, and values.
    pub fn from_csr(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        // Build symbolic structure; second argument `None` means “no separate row_nnz”:
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
}

impl<T: ComplexField + Copy + num_traits::One> SparseMatrix<T> for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        let x_mat = faer::Mat::<T>::from_fn(self.ncols(), 1, |i, _| x[i]);
        let mut y_mat = faer::Mat::<T>::zeros(self.nrows(), 1);
        // Fallback: convert to dense and multiply (since sparse_dense_matmul expects CSC)
        let dense = self.inner.to_dense();
        y_mat.copy_from(&dense * &x_mat);
        for i in 0..y.len() {
            y[i] = y_mat[(i, 0)];
        }
    }
}

use crate::core::traits::SubmatrixExtract;

impl<T: ComplexField + Copy + num_traits::One + num_traits::Zero> SubmatrixExtract for CsrMatrix<T> {
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
    /// Parallel SpMV using Rayon
    pub fn spmv_parallel(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
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
        m.spmv(&x, &mut y);
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
        m.spmv(&x, &mut y);
        assert_eq!(y, vec![3.0, 7.0]);
    }
}
