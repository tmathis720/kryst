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
}

/// Mutable CSC interface.
pub trait CscMatMut<S: KrystScalar>: CscMatRef<S> {
    fn values_mut(&mut self) -> &mut [S];
}
