use crate::algebra::prelude::*;

use crate::matrix::spmv::scalar::{spmv_scaled_csr, spmv_t_scaled_csr};

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

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn rowptr(&self) -> &[usize] {
        &self.rowptr
    }

    #[inline]
    pub fn rowptr_mut(&mut self) -> &mut [usize] {
        &mut self.rowptr
    }

    /// Row pointer accessor mirroring the legacy CSR wrapper API.
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        self.rowptr()
    }

    /// Mutable row pointer accessor matching the legacy CSR wrapper API.
    #[inline]
    pub fn row_ptr_mut(&mut self) -> &mut [usize] {
        self.rowptr_mut()
    }

    #[inline]
    pub fn colind(&self) -> &[usize] {
        &self.colind
    }

    #[inline]
    pub fn colind_mut(&mut self) -> &mut [usize] {
        &mut self.colind
    }

    /// Column index accessor matching the historic naming used throughout the
    /// sparse module.
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        self.colind()
    }

    #[inline]
    pub fn col_idx_mut(&mut self) -> &mut [usize] {
        self.colind_mut()
    }

    #[inline]
    pub fn values(&self) -> &[S] {
        &self.values
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut [S] {
        &mut self.values
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

    /// Computes `y = A * x` using the scalar CSR kernel.
    #[inline]
    pub fn spmv(&self, x: &[S], y: &mut [S]) {
        self.spmv_scaled(S::one(), x, S::zero(), y);
    }

    /// Computes `y = alpha * A * x + beta * y` using the scalar CSR kernel.
    #[inline]
    pub fn spmv_scaled(&self, alpha: S, x: &[S], beta: S, y: &mut [S]) {
        spmv_scaled_csr(
            self.nrows,
            &self.rowptr,
            &self.colind,
            &self.values,
            alpha,
            x,
            beta,
            y,
        );
    }

    /// Computes `y = A^T * x` using the scalar CSR transpose kernel.
    #[inline]
    pub fn spmv_t(&self, x: &[S], y: &mut [S]) {
        self.spmv_t_scaled(S::one(), x, S::zero(), y);
    }

    /// Computes `y = alpha * A^T * x + beta * y` using the scalar CSR transpose kernel.
    #[inline]
    pub fn spmv_t_scaled(&self, alpha: S, x: &[S], beta: S, y: &mut [S]) {
        spmv_t_scaled_csr(
            self.nrows,
            &self.rowptr,
            &self.colind,
            &self.values,
            alpha,
            x,
            beta,
            y,
        );
    }
}

impl<S> CsrMatrix<S>
where
    S: KrystScalar<Real = f64>,
{
    /// Builds a scalar-aware CSR matrix from a real-valued sparse matrix.
    ///
    /// This helper clones the structure and lifts the numeric values into the
    /// active scalar domain via [`KrystScalar::from_real`], enabling solvers to
    /// reuse existing `f64` sparsity patterns in complex builds without
    /// reassembling them from scratch.
    pub fn from_real_csr(real: &crate::matrix::sparse::CsrMatrix<f64>) -> Self {
        let values = real.values().iter().copied().map(S::from_real).collect();
        Self::new(
            real.nrows(),
            real.ncols(),
            real.row_ptr().to_vec(),
            real.col_idx().to_vec(),
            values,
        )
    }
}

/// Keep a convenient alias for explicitly-real use sites (e.g., file I/O).
pub type CsrMatrix64 = CsrMatrix<f64>;
