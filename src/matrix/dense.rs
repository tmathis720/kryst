//! Dense‐matrix API on top of Faer.
//!
//! This module provides the `DenseMatrix` trait and its implementation for the `faer::Mat<T>` type,
//! enabling construction from raw column-major storage.

use crate::core::traits::{Indexing, MatVec};
use faer::Mat;

/// Blanket impl so any Faer Mat<T> is a DenseMatrix.
pub trait DenseMatrix<T>: MatVec<Vec<T>> + Indexing {
    /// Construct from raw column-major storage.
    fn from_raw(nrows: usize, ncols: usize, data: Vec<T>) -> Self;
}

impl<T: Copy + num_traits::Float> DenseMatrix<T> for Mat<T> {
    fn from_raw(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        Mat::from_fn(nrows, ncols, |i, j| data[j * nrows + i])
    }
}
