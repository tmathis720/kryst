use crate::algebra::prelude::*;
use crate::core::traits::{Indexing, MatShape, MatVec, MatrixGet, SubmatrixExtract};
use crate::matrix::dense_api::{DenseMatMut, DenseMatRef, DenseMatShape};
use faer::Mat;

/// Blanket impl so any Faer `Mat<S>` is a `DenseMatrix`.
pub trait DenseMatrix: MatVec<Vec<S>> + Indexing {
    /// Construct from raw column-major storage.
    fn from_raw(nrows: usize, ncols: usize, data: Vec<S>) -> Self;
}

impl<T: KrystScalar> MatrixGet<T> for Mat<T> {
    #[inline]
    fn get(&self, i: usize, j: usize) -> T {
        self[(i, j)]
    }
}

impl<T> DenseMatShape for Mat<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

impl<T: KrystScalar> DenseMatRef<T> for Mat<T> {
    #[inline]
    fn get(&self, i: usize, j: usize) -> T {
        self[(i, j)]
    }

    #[inline]
    fn col_major_data(&self) -> Option<&[T]> {
        let view = self.as_ref();
        view.try_as_col_major().map(|cm| unsafe {
            let len = self.nrows() * self.ncols();
            std::slice::from_raw_parts(cm.as_ptr(), len)
        })
    }
}

impl<T: KrystScalar> DenseMatMut<T> for Mat<T> {
    #[inline]
    fn set(&mut self, i: usize, j: usize, val: T) {
        self[(i, j)] = val;
    }

    #[inline]
    fn col_major_data_mut(&mut self) -> Option<&mut [T]> {
        let len = self.nrows() * self.ncols();
        let view = self.as_mut();
        view.try_as_col_major_mut()
            .map(|cm| unsafe { std::slice::from_raw_parts_mut(cm.as_ptr() as *mut T, len) })
    }
}

impl DenseMatrix for Mat<S> {
    #[inline]
    fn from_raw(nrows: usize, ncols: usize, data: Vec<S>) -> Self {
        Mat::from_fn(nrows, ncols, |i, j| data[j * nrows + i])
    }
}

impl<T: Clone> SubmatrixExtract for Mat<T> {
    type S = T;

    fn extract_submatrix(&self, rows: &[usize], cols: &[usize]) -> Self {
        let m = rows.len();
        let n = cols.len();
        Mat::from_fn(m, n, |i, j| self[(rows[i], cols[j])].clone())
    }
}

impl<T> MatShape for Mat<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}
