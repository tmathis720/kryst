//! Wrappers for faer dense matrix types and vector operations.
//!
//! This module provides implementations of core linear algebra traits for `faer::Mat`, `faer::MatRef`, and `Vec<T>`,
//! enabling their use in generic iterative solvers and preconditioners. It also provides parallel and distributed
//! inner product operations, supporting both single-threaded, multi-threaded (Rayon), and MPI-based distributed environments.
//!
//! # Features
//! - Matrix-vector and matrix-transpose-vector multiplication for `faer` dense matrices.
//! - Inner product and norm operations for vectors, with optional Rayon parallelism.
//! - Distributed inner product and norm for MPI-enabled builds.
//! - Indexing trait implementations for vectors and matrices.
//!
//! # Usage
//! These wrappers allow the use of `faer` matrices and Rust vectors as generic types in the KrylovKit solver framework.
//!
//! # References
//! - [faer crate documentation](https://docs.rs/faer)
//! - [num-traits crate documentation](https://docs.rs/num-traits)

use crate::core::traits::{Indexing, InnerProduct, MatTransVec, MatVec};
use faer::{Mat, MatRef};
use num_traits::{Float, FromPrimitive};

/// Implements matrix-vector multiplication for `faer::Mat`.
///
/// Computes `y = A * x` where `A` is a dense matrix, `x` and `y` are vectors.
impl<T: Float> MatVec<Vec<T>> for Mat<T> {
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.nrows(), y.len(), "Output vector y has incorrect length");
        assert_eq!(self.ncols(), x.len(), "Input vector x has incorrect length");
        for i in 0..self.nrows() {
            y[i] = T::zero();
            for j in 0..self.ncols() {
                y[i] = y[i] + self[(i, j)] * x[j];
            }
        }
    }
}

/// Implements matrix-vector multiplication for a matrix reference (`faer::MatRef`).
impl<'a, T: Float> MatVec<Vec<T>> for MatRef<'a, T> {
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.nrows(), y.len(), "Output vector y has incorrect length");
        assert_eq!(self.ncols(), x.len(), "Input vector x has incorrect length");
        for i in 0..self.nrows() {
            y[i] = T::zero();
            for j in 0..self.ncols() {
                y[i] = y[i] + self[(i, j)] * x[j];
            }
        }
    }
}

/// Implements matrix-transpose-vector multiplication for `faer::Mat`.
///
/// Computes `y = A^T * x` where `A` is a dense matrix, `x` and `y` are vectors.
impl<T: Float> MatTransVec<Vec<T>> for Mat<T> {
    fn mattransvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.ncols(), y.len(), "Output vector y has incorrect length");
        assert_eq!(self.nrows(), x.len(), "Input vector x has incorrect length");
        for j in 0..self.ncols() {
            y[j] = T::zero();
            for i in 0..self.nrows() {
                y[j] = y[j] + self[(i, j)] * x[i];
            }
        }
    }
}

/// Implements matrix-transpose-vector multiplication for a matrix reference (`faer::MatRef`).
impl<'a, T: Float> MatTransVec<Vec<T>> for MatRef<'a, T> {
    fn mattransvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.ncols(), y.len(), "Output vector y has incorrect length");
        assert_eq!(self.nrows(), x.len(), "Input vector x has incorrect length");
        for j in 0..self.ncols() {
            y[j] = T::zero();
            for i in 0..self.nrows() {
                y[j] = y[j] + self[(i, j)] * x[i];
            }
        }
    }
}

/// Implements inner product and norm for vectors, with optional Rayon parallelism.
///
/// If the `rayon` feature is enabled, uses parallel iterators for performance.
impl<T: Float + From<f64> + Send + Sync> InnerProduct<Vec<T>> for () {
    type Scalar = T;
    /// Computes the dot product of two vectors: `x^T y`.
    fn dot(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            x.as_slice()
                .par_iter()
                .zip(y.as_slice().par_iter())
                .map(|(xi, yi)| *xi * *yi)
                .reduce(|| T::zero(), |acc, v| acc + v)
        }
        #[cfg(not(feature = "rayon"))]
        {
            x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| *xi * *yi)
                .fold(T::zero(), |acc, v| acc + v)
        }
    }
    /// Computes the Euclidean norm of a vector: `||x||_2`.
    fn norm(&self, x: &Vec<T>) -> T {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            x.as_slice()
                .par_iter()
                .map(|xi| *xi * *xi)
                .reduce(|| T::zero(), |acc, v| acc + v)
                .sqrt()
        }
        #[cfg(not(feature = "rayon"))]
        {
            x.iter()
                .map(|xi| *xi * *xi)
                .fold(T::zero(), |acc, v| acc + v)
                .sqrt()
        }
    }
}

/// Distributed inner product and norm for MPI-enabled builds.
///
/// This struct is only available if the `mpi` feature is enabled. It wraps a communicator and provides
/// collective dot product and norm operations across distributed memory processes.
#[cfg(feature = "mpi")]
pub struct DistributedInnerProduct<'a, C: crate::parallel::Comm> {
    /// Reference to the communicator implementing the `Comm` trait.
    pub comm: &'a C,
}

#[cfg(feature = "mpi")]
impl<'a, C: crate::parallel::Comm> DistributedInnerProduct<'a, C> {
    /// Computes the distributed dot product of two slices, reducing across all processes.
    pub fn dot<T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero>(&self, x: &[T], y: &[T]) -> T {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");
        // Convert local dot product to f64 for reduction
        let local: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| (a * b).to_f64().unwrap_or(0.0)).sum();
        let global = self.comm.all_reduce(local);
        T::from_f64(global).unwrap_or(T::zero())
    }
    /// Computes the distributed Euclidean norm of a slice, reducing across all processes.
    pub fn norm<T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::Float>(&self, x: &[T]) -> T {
        let local: f64 = x.iter().map(|&a| (a * a).to_f64().unwrap_or(0.0)).sum();
        let global = self.comm.all_reduce(local);
        T::from_f64(global.sqrt()).unwrap_or(T::zero())
    }
}

/// Implements the `Indexing` trait for `Vec<T>`, treating a vector as a column vector.
impl<T> Indexing for Vec<T> {
    /// Returns the number of rows (length) of the vector.
    fn nrows(&self) -> usize {
        self.len()
    }
}

/// Implements the `Indexing` trait for `faer::Mat`, returning the number of rows.
impl<T> Indexing for Mat<T> {
    fn nrows(&self) -> usize {
        self.nrows()
    }
}
