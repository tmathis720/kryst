use crate::algebra::prelude::*;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};

/// CSC matrix wrapper for Faer sparse matrices.
/// Stores owning symbolic and numeric data via `SparseColMat`.
#[derive(Clone)]
pub struct CscMatrix<T> {
    inner: SparseColMat<usize, T>,
}

impl<T> CscMatrix<T> {
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
}

impl<T> CscMatrix<T> {
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    pub fn nnz(&self) -> usize {
        self.inner.compute_nnz()
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

impl<T> CscMatrix<T>
where
    T: KrystScalar<Real = f64>,
{
    /// Convert to dense `faer::Mat` with real entries. Works for real scalars only.
    ///
    /// # Errors
    /// Returns `KError::Unsupported` when called with complex scalars to avoid
    /// silently discarding imaginary components.
    pub fn to_dense(&self) -> Result<faer::Mat<f64>, crate::error::KError> {
        if crate::algebra::scalar::is_complex_scalar::<T>() {
            return Err(crate::error::KError::Unsupported(
                "CSC to_dense is real-only; complex scalars are unsupported",
            ));
        }
        let m = self.nrows();
        let n = self.ncols();
        let mut dense = faer::Mat::from_fn(m, n, |_, _| f64::default());
        let cp = self.col_ptr();
        let ri = self.row_idx();
        let vv = self.values();
        for j in 0..n {
            for p in cp[j]..cp[j + 1] {
                let row = ri[p];
                dense[(row, j)] = vv[p].real();
            }
        }
        Ok(dense)
    }

    /// Convert from dense `faer::Mat` (with real entries) to CSC format with drop tolerance.
    ///
    /// # Errors
    /// Returns `KError::Unsupported` when called with complex scalars.
    pub fn from_dense(dense: &faer::Mat<f64>, drop_tol: f64) -> Result<Self, crate::error::KError> {
        if crate::algebra::scalar::is_complex_scalar::<T>() {
            return Err(crate::error::KError::Unsupported(
                "CSC from_dense is real-only; complex scalars are unsupported",
            ));
        }
        let m = dense.nrows();
        let n = dense.ncols();
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for j in 0..n {
            for i in 0..m {
                let v = dense[(i, j)];
                if v.abs() >= drop_tol {
                    row_idx.push(i);
                    values.push(T::from_real(v));
                }
            }
            col_ptr.push(row_idx.len());
        }
        Ok(Self::from_csc(m, n, col_ptr, row_idx, values))
    }
}

impl<T> CscMatrix<T>
where
    T: KrystScalar,
{
    /// Sparse matrix-vector multiply: `y = A * x`.
    ///
    /// Sequential implementation that updates `y` in place.
    /// Requires `x.len() == ncols` and `y.len() == nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        y.fill(T::zero());
        let cp = self.col_ptr();
        let ri = self.row_idx();
        let vv: &[T] = self.values();
        for j in 0..self.ncols() {
            let xj = x[j];
            for p in cp[j]..cp[j + 1] {
                let row = ri[p];
                y[row] = y[row] + vv[p] * xj;
            }
        }
    }

    /// Transpose sparse matrix-vector multiply: `y = A^T * x`.
    /// Uses `A^H` in complex builds.
    ///
    /// Sequential implementation mirroring [`spmv`]. The output slice is fully
    /// overwritten, so callers do not need to zero-initialize it before calling
    /// this routine.
    pub fn t_matvec(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.nrows());
        assert_eq!(y.len(), self.ncols());
        let cp = self.col_ptr();
        let ri = self.row_idx();
        let vv: &[T] = self.values();

        for (j, yj) in y.iter_mut().enumerate() {
            let mut acc = T::zero();
            for p in cp[j]..cp[j + 1] {
                let row = ri[p];
                acc = acc + vv[p].conj() * x[row];
            }
            *yj = acc;
        }
    }
}

#[cfg(feature = "rayon")]
impl<T> CscMatrix<T>
where
    T: KrystScalar,
{
    /// Parallel sparse matrix-vector multiply: `y = A * x`.
    ///
    /// Each thread accumulates into a private buffer to avoid write conflicts
    /// on the output vector.
    pub fn spmv_parallel(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        use rayon::prelude::*;

        let m = self.nrows();
        let cp = self.col_ptr();
        let ri = self.row_idx();
        let vv: &[T] = self.values();

        let result = (0..self.ncols())
            .into_par_iter()
            .fold(
                || vec![T::zero(); m],
                |mut accum, j| {
                    let xj = x[j];
                    for p in cp[j]..cp[j + 1] {
                        let row = ri[p];
                        accum[row] = accum[row] + vv[p] * xj;
                    }
                    accum
                },
            )
            .reduce(
                || vec![T::zero(); m],
                |mut a, b| {
                    for i in 0..m {
                        a[i] = a[i] + b[i];
                    }
                    a
                },
            );

        y.copy_from_slice(&result);
    }
}

#[cfg(all(test, feature = "backend-faer", feature = "complex"))]
mod tests {
    use super::*;

    #[test]
    fn dense_conversions_reject_complex_scalars() {
        let csc = CscMatrix::from_csc(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![
                S::from_parts(1.0, 0.5),
                S::from_parts(2.0, -1.0),
                S::one(),
                S::zero(),
            ],
        );
        let err = match csc.to_dense() {
            Err(err) => err,
            Ok(_) => panic!("expected complex to_dense to return Err"),
        };
        assert!(matches!(err, crate::error::KError::Unsupported(_)));

        let dense = faer::Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let err = match CscMatrix::<S>::from_dense(&dense, 0.0) {
            Err(err) => err,
            Ok(_) => panic!("expected complex from_dense to return Err"),
        };
        assert!(matches!(err, crate::error::KError::Unsupported(_)));
    }
}
