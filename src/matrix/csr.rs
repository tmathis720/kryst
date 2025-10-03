use crate::algebra::prelude::*;

/// Compressed Sparse Row matrix with scalar entries of type `S`.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix<S: KrystScalar> {
    pub nrows: usize,
    pub ncols: usize,
    /// CSR `rowptr` length = nrows + 1
    pub rowptr: Vec<usize>,
    /// Column indices for each nonzero (same length as `values`)
    pub colind: Vec<usize>,
    /// Nonzero values
    pub values: Vec<S>,
}

impl<S: KrystScalar> CsrMatrix<S> {
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    pub fn new(
        nrows: usize,
        ncols: usize,
        rowptr: Vec<usize>,
        colind: Vec<usize>,
        values: Vec<S>,
    ) -> Self {
        debug_assert_eq!(rowptr.len(), nrows + 1);
        debug_assert_eq!(colind.len(), values.len());
        Self {
            nrows,
            ncols,
            rowptr,
            colind,
            values,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.rowptr.len() == self.nrows + 1
            && self.colind.len() == self.values.len()
            && self.rowptr.windows(2).all(|w| w[0] <= w[1])
            && self.colind.iter().all(|&j| j < self.ncols)
    }
}

/// Keep a convenient alias for explicitly-real use sites (e.g., file I/O).
pub type CsrMatrix64 = CsrMatrix<f64>;
