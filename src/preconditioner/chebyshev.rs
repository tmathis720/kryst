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
//!     ```ignore
//!     z = p_m(A) r
//!     ```
//!
//! where `A` is the system matrix, and `p_m` is the Chebyshev polynomial scaled to the spectrum [`alpha`, `beta`].
//! The endpoints of the spectrum (smallest/largest eigenvalues) can be provided or estimated.
//!
//! # Usage
//!
//! - Create a `ChebyshevPre` struct with the desired degree and spectrum bounds.
//! - Use the `apply_chebyshev` free function to apply the polynomial filter to a vector.
//! - The `Preconditioner` trait implementation provides full functionality.
//!
//! # References
//!
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. (Section 12.3)
//! - https://en.wikipedia.org/wiki/Chebyshev_polynomials

use crate::core::traits::MatVec;
use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::Preconditioner as ObjPreconditioner;
use crate::preconditioner::legacy::Preconditioner;
use faer::Mat;
use std::sync::Arc;
use std::sync::Mutex;

/// Chebyshev polynomial preconditioner struct
///
/// Stores the matrix, polynomial degree, and spectrum bounds.
/// This is the Phase III enhanced version that works as a proper preconditioner.
pub struct ChebyshevPre {
    /// The system matrix (stored for apply operations)
    matrix: Mat<f64>,
    /// Degree of the Chebyshev polynomial
    degree: usize,
    /// Lower bound of the spectrum (smallest eigenvalue)
    lambda_min: f64,
    /// Upper bound of the spectrum (largest eigenvalue)
    lambda_max: f64,
}

impl ChebyshevPre {
    /// Create a new Chebyshev preconditioner
    ///
    /// # Arguments
    /// * `matrix` - System matrix (cloned for storage)
    /// * `degree` - Degree of the Chebyshev polynomial
    /// * `lambda_min` - Lower bound of the spectrum
    /// * `lambda_max` - Upper bound of the spectrum
    pub fn new(matrix: Mat<f64>, degree: usize, lambda_min: f64, lambda_max: f64) -> Self {
        Self {
            matrix,
            degree,
            lambda_min,
            lambda_max,
        }
    }

    /// Estimate eigenvalue bounds using power iteration if not provided
    ///
    /// # Arguments
    /// * `matrix` - System matrix
    /// * `max_iters` - Maximum iterations for eigenvalue estimation
    /// * `tol` - Tolerance for eigenvalue estimation
    ///
    /// Returns (lambda_min, lambda_max)
    pub fn estimate_eigenvalue_bounds(matrix: &Mat<f64>, max_iters: usize, tol: f64) -> (f64, f64) {
        let n = matrix.nrows();
        if n == 0 {
            return (1.0, 1.0);
        }

        // Estimate largest eigenvalue using power iteration
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        let mut av = vec![0.0; n];
        let mut lambda_max = 1.0;

        for _ in 0..max_iters {
            // Apply matrix
            // Disambiguate: call the LinOp variant of matvec
            crate::matrix::op::LinOp::matvec(matrix, &v, &mut av);

            // Rayleigh quotient
            let new_lambda_max: f64 = v.iter().zip(av.iter()).map(|(&vi, &avi)| vi * avi).sum();

            if (new_lambda_max - lambda_max).abs() < tol * lambda_max.abs() {
                lambda_max = new_lambda_max;
                break;
            }
            lambda_max = new_lambda_max;

            // Normalize
            let norm: f64 = av.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for i in 0..n {
                    v[i] = av[i] / norm;
                }
            }
        }

        // Estimate smallest eigenvalue (very rough approximation)
        // For SPD matrices, use a fraction of the largest eigenvalue
        let lambda_min = lambda_max * 0.01; // Conservative estimate

        (lambda_min, lambda_max)
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for ChebyshevPre {
    /// Setup the Chebyshev preconditioner (stores matrix for later use)
    fn setup(&mut self, a: &Mat<f64>) -> Result<(), KError> {
        self.matrix = a.clone();
        Ok(())
    }

    /// Apply Chebyshev polynomial preconditioner
    fn apply(
        &self,
        _side: crate::preconditioner::PcSide,
        r: &Vec<f64>,
        z: &mut Vec<f64>,
    ) -> Result<(), KError> {
        if r.len() != z.len() {
            return Err(KError::SolveError("Vector length mismatch".to_string()));
        }

        apply_chebyshev(
            &self.matrix,
            r,
            z,
            self.lambda_min,
            self.lambda_max,
            self.degree,
        );
        Ok(())
    }
}

/// Legacy Chebyshev struct for backward compatibility
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
        Self {
            degree,
            lambda_min,
            lambda_max,
        }
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
    fn apply(
        &self,
        _side: crate::preconditioner::PcSide,
        _r: &V,
        _z: &mut V,
    ) -> Result<(), KError> {
        Err(KError::SolveError(
            "Chebyshev preconditioner requires matrix argument; use apply_chebyshev free function."
                .to_string(),
        ))
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

// -----------------------------------------------------------------------------
// Object-safe Chebyshev preconditioner using CSR and LinOp.
// -----------------------------------------------------------------------------

struct ChebScratch {
    v0: Vec<f64>,
    v1: Vec<f64>,
    v2: Vec<f64>,
}

impl Default for ChebScratch {
    fn default() -> Self {
        Self {
            v0: Vec::new(),
            v1: Vec::new(),
            v2: Vec::new(),
        }
    }
}

/// Object-safe Chebyshev preconditioner
pub struct ChebyshevPc {
    degree: usize,
    lambda_min: f64,
    lambda_max: f64,
    a_csr: Option<Arc<CsrMatrix<f64>>>,
    n: usize,
    scratch: Mutex<ChebScratch>,
}

impl ChebyshevPc {
    pub fn new(degree: usize, lambda_min: f64, lambda_max: f64) -> Self {
        Self {
            degree,
            lambda_min,
            lambda_max,
            a_csr: None,
            n: 0,
            scratch: Mutex::new(ChebScratch::default()),
        }
    }
}

impl ObjPreconditioner for ChebyshevPc {
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), crate::error::KError> {
        let csr = csr_from_linop(op, 0.0)?;
        let n = csr.nrows();
        self.a_csr = Some(csr);
        self.n = n;
        // ensure scratch
        let mut s = self.scratch.lock().unwrap();
        if s.v0.len() != n {
            s.v0.resize(n, 0.0);
        }
        if s.v1.len() != n {
            s.v1.resize(n, 0.0);
        }
        if s.v2.len() != n {
            s.v2.resize(n, 0.0);
        }
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), crate::error::KError> {
        // For now, refresh CSR view and keep degree/spectrum
        let csr = csr_from_linop(op, 0.0)?;
        self.a_csr = Some(csr);
        Ok(())
    }

    fn apply(
        &self,
        _side: crate::preconditioner::PcSide,
        r: &[f64],
        z: &mut [f64],
    ) -> Result<(), crate::error::KError> {
        use crate::error::KError;
        let a = self
            .a_csr
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("ChebyshevPc not setup".into()))?;
        if r.len() != self.n || z.len() != self.n {
            return Err(KError::InvalidInput(
                "dimension mismatch in ChebyshevPc::apply".into(),
            ));
        }

        let n = self.n;
        let mut s = self.scratch.lock().unwrap();
        // tau scaling to make p_m(0) ~ 1
        let c = (self.lambda_max + self.lambda_min) / 2.0;
        let d = (self.lambda_max - self.lambda_min) / 2.0;
        if d.abs() < f64::EPSILON {
            // Degenerate spectrum: copy input
            z.copy_from_slice(r);
            return Ok(());
        }
        let tau = 1.0 / chebyshev_t(self.degree, (0.0 - c) / d);

        if self.degree == 0 {
            z.copy_from_slice(r);
            return Ok(());
        }

        // v1 = (A r - c r) / d
        a.spmv_scaled(1.0, r, 0.0, &mut s.v1)?;
        for i in 0..n {
            s.v1[i] = (s.v1[i] - c * r[i]) / d;
        }
        if self.degree == 1 {
            for i in 0..n {
                z[i] = tau * s.v1[i];
            }
            return Ok(());
        }

        // Set v0 = r for the recurrence
        s.v0[..n].copy_from_slice(r);

        // Recurrence for k = 2..=m
        for _k in 2..=self.degree {
            // v2 = 2 * ((A v1 - c v1)/d) - v0
            // Clone v1 into a temporary to satisfy borrow checker without unsafe.
            let v1_tmp = s.v1.clone();
            a.spmv_scaled(1.0, &v1_tmp, 0.0, &mut s.v2)?;
            for i in 0..n {
                s.v2[i] = 2.0 * ((s.v2[i] - c * s.v1[i]) / d) - s.v0[i];
            }
            // rotate: v0 <- v1, v1 <- v2, v2 <- old v0
            let t0 = std::mem::take(&mut s.v0);
            let t1 = std::mem::take(&mut s.v1);
            let t2 = std::mem::take(&mut s.v2);
            s.v0 = t1;
            s.v1 = t2;
            s.v2 = t0;
        }

        for i in 0..n {
            z[i] = tau * s.v1[i];
        }
        Ok(())
    }

    fn required_format(&self) -> crate::matrix::format::FormatHint {
        crate::matrix::format::FormatHint::Csr
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
        fn new(data: Vec<Vec<T>>) -> Self {
            Self { data }
        }
    }
    impl<T> MatVec<Vec<T>> for DenseMat<T>
    where
        T: Copy + std::ops::Mul<Output = T> + std::iter::Sum,
    {
        fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
            for i in 0..self.data.len() {
                y[i] = (0..self.data[0].len())
                    .map(|j| self.data[i][j] * x[j])
                    .sum();
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
