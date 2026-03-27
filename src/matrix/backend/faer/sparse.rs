use crate::algebra::prelude::*;
use crate::matrix::csc::CscMatrix;
use crate::matrix::sparse_api::{CscMatMut, CscMatRef};

impl<S: KrystScalar> CscMatRef<S> for CscMatrix<S> {
    #[inline]
    fn nrows(&self) -> usize {
        CscMatrix::nrows(self)
    }

    #[inline]
    fn ncols(&self) -> usize {
        CscMatrix::ncols(self)
    }

    #[inline]
    fn col_ptr(&self) -> &[usize] {
        CscMatrix::col_ptr(self)
    }

    #[inline]
    fn row_idx(&self) -> &[usize] {
        CscMatrix::row_idx(self)
    }

    #[inline]
    fn values(&self) -> &[S] {
        CscMatrix::values(self)
    }
}

impl<S: KrystScalar> CscMatMut<S> for CscMatrix<S> {
    #[inline]
    fn values_mut(&mut self) -> &mut [S] {
        CscMatrix::values_mut(self)
    }
}
