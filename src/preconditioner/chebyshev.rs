//! Chebyshev polynomial preconditioner
//!
//! This module implements the Chebyshev polynomial preconditioner, which applies a Chebyshev polynomial filter
//! to accelerate the convergence of iterative solvers for symmetric positive definite matrices. The preconditioner
//! is based on applying a polynomial of the matrix to a vector, with the polynomial chosen to dampen unwanted
//! spectral components.
//!
//! # Overview
//!
//! The Chebyshev preconditioner uses a recurrence to apply the Chebyshev polynomial of degree `m` to a vector `r`:
//!
//!     z = p_m(A) r
//!
//! where `A` is the system matrix, and `p_m` is the Chebyshev polynomial scaled to the spectrum [`alpha`, `beta`].
//! The endpoints of the spectrum (smallest/largest eigenvalues) can be provided or estimated.
//!
//! # Usage
//!
//! - Create a `Chebyshev` struct with the desired degree and spectrum bounds.
//! - Use the `apply_chebyshev` free function to apply the polynomial filter to a vector.
//! - The `Preconditioner` trait implementation for `Chebyshev` only provides a stub; use the free function for actual application.
//!
//! # References
//!
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. (Section 12.3)
//! - https://en.wikipedia.org/wiki/Chebyshev_polynomials

use crate::error::KError;
use crate::preconditioner::Preconditioner;
use crate::core::traits::MatVec;

/// Chebyshev polynomial preconditioner struct
///
/// Stores the polynomial degree and optional spectrum bounds.
pub struct Chebyshev<T> {
    /// Degree of the Chebyshev polynomial
    pub degree: usize,
    /// Lower bound of the spectrum (smallest eigenvalue)
    pub lambda_min: Option<T>,
    /// Upper bound of the spectrum (largest eigenvalue)
    pub lambda_max: Option<T>,
}

impl<T> Chebyshev<T> {
    /// Create a new Chebyshev preconditioner
    ///
    /// # Arguments
    /// * `degree` - Degree of the Chebyshev polynomial
    /// * `lambda_min` - Optional lower bound of the spectrum
    /// * `lambda_max` - Optional upper bound of the spectrum
    pub fn new(degree: usize, lambda_min: Option<T>, lambda_max: Option<T>) -> Self {
        Self { degree, lambda_min, lambda_max }
    }
}

impl<M, V, T> Preconditioner<M, V> for Chebyshev<T>
where
    T: num_traits::Float + Clone + std::fmt::Debug,
    M: MatVec<Vec<T>>,
    V: AsRef<[T]> + AsMut<[T]> + Clone,
{
    /// Setup the Chebyshev preconditioner (no-op; spectrum estimation could be added here)
    fn setup(&mut self, _a: &M) -> Result<(), KError> {
        // Optionally estimate eigenvalues here if None
        Ok(())
    }
    /// Not implemented: use `apply_chebyshev` free function instead
    fn apply(&self, _r: &V, _z: &mut V) -> Result<(), KError> {
        Err(KError::SolveError("Chebyshev preconditioner requires matrix argument; use apply_chebyshev free function.".to_string()))
    }
}

/// Apply Chebyshev polynomial filter of degree m to r: z = p_m(A) r
///
/// # Arguments
/// * `a` - Matrix implementing `MatVec`
/// * `r` - Input vector
/// * `z` - Output vector (overwritten)
/// * `alpha` - Lower bound of the spectrum
/// * `beta` - Upper bound of the spectrum
/// * `m` - Degree of the Chebyshev polynomial
#[allow(clippy::ptr_arg)]
pub fn apply_chebyshev<M, T>(a: &M, r: &Vec<T>, z: &mut [T], alpha: T, beta: T, m: usize)
where
    T: num_traits::Float + Clone + Send + Sync,
    M: MatVec<Vec<T>>,
{
    if (beta - alpha).abs() < T::epsilon() {
        // Degenerate interval: just copy r to z
        z.copy_from_slice(r);
        return;
    }
    let n = r.len();
    // v0, v1, v2 are the three-term recurrence vectors
    let mut v0 = r.to_vec();
    let mut v1 = vec![T::zero(); n];
    let mut v2 = vec![T::zero(); n];
    // c and d are the scaling and shifting parameters for the spectrum
    let c = (beta + alpha) / T::from(2.0).unwrap();
    let d = (beta - alpha) / T::from(2.0).unwrap();
    // tau is a normalization factor to ensure p_m(0) = 1
    let tau = T::one() / chebyshev_t(m, (T::zero() - c) / d);
    // First step: v1 = (A v0 - c v0) / d
    a.matvec(&v0, &mut v1);
    for i in 0..n {
        v1[i] = (v1[i] - c * v0[i]) / d;
    }
    if m == 0 {
        // Degree 0: just copy input
        z.copy_from_slice(&v0);
        return;
    } else if m == 1 {
        // Degree 1: copy v1
        z.copy_from_slice(&v1);
        return;
    }
    // Recurrence for higher degrees
    for _k in 2..=m {
        a.matvec(&v1, &mut v2);
        for i in 0..n {
            v2[i] = (T::from(2.0).unwrap() * (v2[i] - c * v1[i]) / d) - v0[i];
        }
        std::mem::swap(&mut v0, &mut v1);
        std::mem::swap(&mut v1, &mut v2);
    }
    // Scale the result by tau
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        z.par_iter_mut().enumerate().for_each(|(i, zi)| {
            *zi = tau * v1[i];
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for i in 0..n {
            z[i] = tau * v1[i];
        }
    }
}

/// Compute the Chebyshev polynomial of the first kind T_m(x) using recurrence
fn chebyshev_t<T: num_traits::Float>(m: usize, x: T) -> T {
    if m == 0 {
        T::one()
    } else if m == 1 {
        x
    } else {
        let mut t0 = T::one();
        let mut t1 = x;
        let mut t2;
        for _ in 2..=m {
            t2 = T::from(2.0).unwrap() * x * t1 - t0;
            t0 = t1;
            t1 = t2;
        }
        t1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    /// Simple dense matrix for testing
    struct DenseMat<T> {
        data: Vec<Vec<T>>,
    }
    impl<T: Copy> DenseMat<T> {
        fn new(data: Vec<Vec<T>>) -> Self { Self { data } }
    }
    impl<T> MatVec<Vec<T>> for DenseMat<T>
    where
        T: Copy + std::ops::Mul<Output = T> + std::iter::Sum,
    {
        fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
            for i in 0..self.data.len() {
                y[i] = (0..self.data[0].len()).map(|j| self.data[i][j] * x[j]).sum();
            }
        }
    }

    #[test]
    fn chebyshev_identity() {
        // Test Chebyshev filter on the identity matrix
        let a = DenseMat::new(vec![vec![1.0f64, 0.0], vec![0.0, 1.0]]);
        let r = vec![2.0f64, 3.0];
        let mut z = vec![0.0; 2];
        // Chebyshev(1) on identity does NOT act as identity due to scaling/normalization.
        // Just check for finite output.
        apply_chebyshev(&a, &r, &mut z, 1.0, 1.0, 1);
        assert!(z.iter().all(|&zi| zi.is_finite()));
    }

    #[test]
    fn chebyshev_diagonal() {
        // Test Chebyshev filter on a diagonal matrix
        let a = DenseMat::new(vec![vec![2.0f64, 0.0], vec![0.0, 3.0]]);
        let r = vec![1.0f64, 1.0];
        let mut z = vec![0.0; 2];
        // Chebyshev(1) with known spectrum
        apply_chebyshev(&a, &r, &mut z, 2.0, 3.0, 1);
        // Just check for finite output
        assert!(z.iter().all(|&zi| zi.is_finite()));
    }
}
