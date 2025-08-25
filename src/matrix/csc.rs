use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::traits::ComplexField;

/// CSC matrix wrapper for Faer sparse matrices.
/// Stores owning symbolic and numeric data via `SparseColMat`.
#[derive(Clone)]
pub struct CscMatrix<T> {
    inner: SparseColMat<usize, T>,
}

impl<T: ComplexField + Copy + num_traits::Zero> CscMatrix<T> {
    /// Build a CSC from raw col-pointer, row-index, and values.
    pub fn from_csc(
        nrows: usize,
        ncols: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        let symbolic = SymbolicSparseColMat::new_checked(nrows, ncols, col_ptr, None, row_idx);
        let inner = SparseColMat::new(symbolic, values);
        Self { inner }
    }

    /// Convert from dense `faer::Mat` to CSC format with drop tolerance.
    pub fn from_dense(dense: &faer::Mat<T>, drop_tol: T) -> Self
    where
        T: PartialOrd + std::ops::Neg<Output = T>,
    {
        let m = dense.nrows();
        let n = dense.ncols();
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for j in 0..n {
            for i in 0..m {
                let v = dense[(i, j)];
                if v > drop_tol || v < -drop_tol {
                    row_idx.push(i);
                    values.push(v);
                }
            }
            col_ptr.push(row_idx.len());
        }
        Self::from_csc(m, n, col_ptr, row_idx, values)
    }

    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    pub fn nnz(&self) -> usize {
        self.inner.compute_nnz()
    }

    pub fn to_dense(&self) -> faer::Mat<T> {
        self.inner.to_dense()
    }

    #[inline]
    pub fn col_ptr(&self) -> &[usize] {
        self.inner.col_ptr()
    }
    #[inline]
    pub fn row_idx(&self) -> &[usize] {
        self.inner.row_idx()
    }
    #[inline]
    pub fn values(&self) -> &[T] {
        self.inner.val()
    }
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        self.inner.val_mut()
    }
}
