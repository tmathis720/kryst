//! Dense‐matrix API on top of Faer.
//!
//! This module provides the `DenseMatrix` trait and its implementation for the `faer::Mat<T>` type,
//! enabling construction from raw column-major storage.

use crate::algebra::prelude::*;
use crate::core::traits::{Indexing, MatShape, MatVec, SubmatrixExtract};
use faer::Mat;

impl crate::core::traits::MatrixGet<S> for Mat<S> {
    fn get(&self, i: usize, j: usize) -> S {
        self[(i, j)]
    }
}

/// Blanket impl so any Faer Mat<S> is a DenseMatrix.
pub trait DenseMatrix: MatVec<Vec<S>> + Indexing {
    /// Construct from raw column-major storage.
    fn from_raw(nrows: usize, ncols: usize, data: Vec<S>) -> Self;
}

impl DenseMatrix for Mat<S> {
    fn from_raw(nrows: usize, ncols: usize, data: Vec<S>) -> Self {
        Mat::from_fn(nrows, ncols, |i, j| data[j * nrows + i])
    }
}

impl SubmatrixExtract for Mat<S> {
    fn submatrix(&self, indices: &[usize]) -> Self {
        let n = indices.len();
        Mat::from_fn(n, n, |i, j| self[(indices[i], indices[j])])
    }
}

impl MatShape for Mat<S> {
    fn nrows(&self) -> usize {
        self.nrows()
    }
    fn ncols(&self) -> usize {
        self.ncols()
    }
}
