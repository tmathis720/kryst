//! Backend-neutral sparse storage traits used by SpMV and utilities.

use crate::algebra::scalar::KrystScalar;

/// Read-only CSR interface.
pub trait CsrMatRef<S: KrystScalar> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn row_ptr(&self) -> &[usize];
    fn col_idx(&self) -> &[usize];
    fn values(&self) -> &[S];
}

/// Mutable CSR interface.
pub trait CsrMatMut<S: KrystScalar>: CsrMatRef<S> {
    fn values_mut(&mut self) -> &mut [S];

    #[inline]
    fn row_ptr_mut(&mut self) -> &mut [usize] {
        unimplemented!("row_ptr_mut not supported by this backend")
    }

    #[inline]
    fn col_idx_mut(&mut self) -> &mut [usize] {
        unimplemented!("col_idx_mut not supported by this backend")
    }
}

/// Read-only CSC interface.
pub trait CscMatRef<S: KrystScalar> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn col_ptr(&self) -> &[usize];
    fn row_idx(&self) -> &[usize];
    fn values(&self) -> &[S];
    fn t_matvec(&self, x: &[S], y: &mut [S]) {
        assert_eq!(x.len(), self.nrows());
        assert_eq!(y.len(), self.ncols());
        y.fill(S::zero());
        let cp = self.col_ptr();
        let ri = self.row_idx();
        let vv = self.values();
        for col in 0..self.ncols() {
            for ptr in cp[col]..cp[col + 1] {
                let row = ri[ptr];
                y[col] = y[col] + vv[ptr] * x[row];
            }
        }
    }
}

/// Mutable CSC interface.
pub trait CscMatMut<S: KrystScalar>: CscMatRef<S> {
    fn values_mut(&mut self) -> &mut [S];
}
