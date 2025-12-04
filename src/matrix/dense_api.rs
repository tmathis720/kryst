//! Backend-agnostic dense matrix traits.
//!
//! These traits describe the minimal capabilities required by generic dense
//! utilities without committing to a specific backend type such as `faer::Mat`.

use crate::algebra::scalar::KrystScalar;

/// Shared shape access for dense matrix references.
pub trait DenseMatShape {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
}

/// Read-only dense matrix interface.
pub trait DenseMatRef<S: KrystScalar>: DenseMatShape {
    /// Get entry (i, j).
    fn get(&self, i: usize, j: usize) -> S;

    /// Optional fast view of contiguous column-major storage.
    #[inline]
    fn col_major_data(&self) -> Option<&[S]> {
        None
    }
}

/// Mutable dense matrix interface.
pub trait DenseMatMut<S: KrystScalar>: DenseMatRef<S> {
    /// Set entry (i, j).
    fn set(&mut self, i: usize, j: usize, val: S);

    /// Optional fast mutable view of contiguous column-major storage.
    #[inline]
    fn col_major_data_mut(&mut self) -> Option<&mut [S]> {
        None
    }
}
