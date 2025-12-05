use crate::algebra::prelude::*;
use crate::matrix::csc::CscMatrix;
use crate::matrix::sparse_api::{CscMatMut, CscMatRef};

impl<S: KrystScalar> CscMatRef<S> for CscMatrix<S> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }

    #[inline]
    fn col_ptr(&self) -> &[usize] {
        self.col_ptr()
    }

    #[inline]
    fn row_idx(&self) -> &[usize] {
        self.row_idx()
    }

    #[inline]
    fn values(&self) -> &[S] {
        self.values()
    }
}

impl<S: KrystScalar> CscMatMut<S> for CscMatrix<S> {
    #[inline]
    fn values_mut(&mut self) -> &mut [S] {
        self.values_mut()
    }
}
