use crate::algebra::prelude::*;
use crate::matrix::csc::CscMatrix;
use crate::matrix::csr::CsrMatrix as ScalarCsrMatrix;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::sparse_api::{CscMatMut, CscMatRef, CsrMatMut, CsrMatRef};

impl<T: KrystScalar> CsrMatRef<T> for CsrMatrix<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }

    #[inline]
    fn row_ptr(&self) -> &[usize] {
        self.row_ptr()
    }

    #[inline]
    fn col_idx(&self) -> &[usize] {
        self.col_idx()
    }

    #[inline]
    fn values(&self) -> &[T] {
        self.values()
    }
}

impl<T: KrystScalar> CsrMatMut<T> for CsrMatrix<T> {
    #[inline]
    fn values_mut(&mut self) -> &mut [T] {
        self.values_mut()
    }
}

impl<S: KrystScalar> CsrMatRef<S> for ScalarCsrMatrix<S> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }

    #[inline]
    fn row_ptr(&self) -> &[usize] {
        self.row_ptr()
    }

    #[inline]
    fn col_idx(&self) -> &[usize] {
        self.col_idx()
    }

    #[inline]
    fn values(&self) -> &[S] {
        self.values()
    }
}

impl<S: KrystScalar> CsrMatMut<S> for ScalarCsrMatrix<S> {
    #[inline]
    fn values_mut(&mut self) -> &mut [S] {
        self.values_mut()
    }

    #[inline]
    fn row_ptr_mut(&mut self) -> &mut [usize] {
        self.row_ptr_mut()
    }

    #[inline]
    fn col_idx_mut(&mut self) -> &mut [usize] {
        self.col_idx_mut()
    }
}

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
