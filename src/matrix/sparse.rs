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
    SparseRowMat, // owning numeric CSR alias
    //CreationError,           // error type for builders
    SymbolicSparseRowMat, // owning symbolic CSR alias
};
use faer::traits::ComplexField;
//use faer::sparse::linalg::matmul::sparse_dense_matmul;

/// CSR matrix wrapper for Faer sparse matrices.
///
/// Use [`row_ptr`], [`col_idx`], and [`values`] to access the raw CSR
/// structure without incurring dense conversions. Sparse matrix products are
/// available through [`crate::matrix::utils::spgemm`] and
/// [`crate::matrix::utils::spgemm_with_drop_tol`]; the Galerkin triple product
/// composes these CSR kernels directly.
#[derive(Clone)]
pub struct CsrMatrix<T> {
    inner: SparseRowMat<usize, T>,
    /// Cached position of the diagonal entry in each row.
    ///
    /// `diag_pos[i]` stores `Some(k)` if column `i` appears in row `i` and
    /// `k` is the index into `values()`; otherwise `None` if the diagonal is
    /// structurally zero.  This enables O(1) access to the diagonal for
    /// factorization and triangular solve kernels without converting to a
    /// dense representation.
    diag_pos: Vec<Option<usize>>,
}

impl<
    T: ComplexField
        + Copy
        + num_traits::Zero
        + PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>,
> CsrMatrix<T>
{
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
            nrows, ncols, row_ptr, None, // optional row_nnz: Option<Vec<usize>>
            col_idx,
        );
        // Attach the numerical values:
        let inner = SparseRowMat::new(symbolic, values);
        let mut this = Self {
            inner,
            diag_pos: Vec::new(),
        };
        this.build_diag_pos();
        this
    }

    /// Convert from dense faer::Mat to sparse CSR format with drop tolerance
    pub fn from_dense(dense: &faer::Mat<T>, drop_tol: T) -> Self
    where
        T: PartialOrd + std::ops::Neg<Output = T>,
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

    /// Convert from an owned dense `faer::Mat<T>` to sparse CSR format with
    /// drop tolerance without cloning the matrix. This is useful when the
    /// caller already owns the dense matrix and can move it into this call.
    ///
    /// This method mirrors `from_dense(&faer::Mat<T>, ...)` but takes the
    /// dense matrix by value so callers avoid an extra clone.
    pub fn from_dense_owned(dense: faer::Mat<T>, drop_tol: T) -> Self
    where
        T: PartialOrd + std::ops::Neg<Output = T>,
    {
        // Reuse the same implementation but work with the owned matrix.
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
    where
        T: num_traits::One,
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
    pub fn spmv_scaled(
        &self,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(crate::error::KError::InvalidInput(format!(
                "Dimension mismatch in spmv: A={}x{}, x.len()={}, y.len()={}",
                self.nrows(),
                self.ncols(),
                x.len(),
                y.len()
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

    /// Borrow a row of the matrix as CSR slices `(col_idx, values)`.
    #[inline]
    pub fn row(&self, i: usize) -> (&[usize], &[T]) {
        let start = self.inner.row_ptr()[i];
        let end = self.inner.row_ptr()[i + 1];
        (
            &self.inner.col_idx()[start..end],
            &self.inner.val()[start..end],
        )
    }

    /// Mutable borrow of the values of row `i`.
    ///
    /// The column indices remain immutable; the structure of the CSR matrix
    /// is fixed.  This is intended for in-place numeric operations such as
    /// ILU factorizations where the sparsity pattern does not change.
    #[inline]
    pub fn row_values_mut(&mut self, i: usize) -> &mut [T] {
        let start = self.inner.row_ptr()[i];
        let end = self.inner.row_ptr()[i + 1];
        &mut self.inner.val_mut()[start..end]
    }

    /// Immutable access to the diagonal entry of row `i` if present.
    #[inline]
    pub fn diag_ref(&self, i: usize) -> Option<&T> {
        self.diag_pos[i].map(|k| &self.values()[k])
    }

    /// Mutable access to the diagonal entry of row `i` if present.
    #[inline]
    pub fn diag_mut(&mut self, i: usize) -> Option<&mut T> {
        if let Some(k) = self.diag_pos[i] {
            Some(&mut self.values_mut()[k])
        } else {
            None
        }
    }

    /// Rebuild the cached diagonal positions.  Call after any operation that
    /// may have modified the sparsity structure (in this module we construct
    /// CSR matrices with sorted, unique rows, so this function is typically
    /// only required once at creation).
    pub fn build_diag_pos(&mut self) {
        let n = self.nrows();
        self.diag_pos.resize(n, None);
        for i in 0..n {
            let start = self.inner.row_ptr()[i];
            let end = self.inner.row_ptr()[i + 1];
            // Binary search for column i within row i.
            if let Ok(off) = self.inner.col_idx()[start..end].binary_search(&i) {
                self.diag_pos[i] = Some(start + off);
            } else {
                self.diag_pos[i] = None;
            }
        }
    }

    /// Mutably borrow the CSR value array (length = nnz).
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        self.inner.val_mut()
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
use std::collections::HashMap;

impl<T: ComplexField + Copy + num_traits::Zero + num_traits::One + PartialEq + PartialOrd>
    SubmatrixExtract for CsrMatrix<T>
{
    fn submatrix(&self, indices: &[usize]) -> Self {
        // Efficient CSR-based submatrix extraction that selects rows and
        // columns corresponding to `indices`, returning an n x n CSR whose
        // local column indices are remapped to 0..n-1.
        let n = indices.len();
        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        // Build a map from global column -> local column index
        let mut g2l: HashMap<usize, usize> = HashMap::with_capacity(n);
        for (l, &g) in indices.iter().enumerate() {
            g2l.insert(g, l);
        }

        let rp = self.inner.row_ptr();
        let cj = self.inner.col_idx();
        let vv = self.inner.val();

        for &g_row in indices {
            let rs = rp[g_row];
            let re = rp[g_row + 1];
            for p in rs..re {
                let gcol = cj[p];
                if let Some(&lcol) = g2l.get(&gcol) {
                    col_idx.push(lcol);
                    values.push(vv[p]);
                }
            }
            row_ptr.push(col_idx.len());
        }

        CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
    }
}

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "rayon")]
impl<T> CsrMatrix<T>
where
    T: ComplexField
        + Copy
        + num_traits::Zero
        + PartialOrd
        + Send
        + Sync
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>,
{
    /// Parallel SpMV using CSR structure directly.
    pub fn spmv_parallel(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        let rp = self.row_ptr();
        let cj = self.col_idx();
        let vv = self.values();
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = T::zero();
            let rs = rp[i];
            let re = rp[i + 1];
            for p in rs..re {
                let j = cj[p];
                sum = sum + vv[p] * x[j];
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
        let m = CsrMatrix::from_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]);
        let x = vec![2.0, 3.0, 5.0];
        let mut y = vec![0.0; 3];
        m.spmv_scaled(1.0, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn simple_pattern() {
        // 2×3 matrix [[1,2,0],[0,3,4]]
        let m = CsrMatrix::from_csr(
            2,
            3,
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];
        m.spmv_scaled(1.0, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, vec![3.0, 7.0]);
    }
}
