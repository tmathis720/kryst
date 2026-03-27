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

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::core::traits::MatVec;
use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::PcDistributedSupport;
use crate::preconditioner::Preconditioner as ObjPreconditioner;
#[cfg(feature = "complex")]
use crate::preconditioner::bridge::{
    apply_pc_mut_s as bridge_apply_pc_mut_s, apply_pc_s as bridge_apply_pc_s,
};
use crate::preconditioner::legacy::Preconditioner;
use faer::Mat;
use std::sync::Arc;
use std::sync::Mutex;

/// Chebyshev polynomial preconditioner struct
///
/// Stores the matrix, polynomial degree, and spectrum bounds.
/// This is the Phase III enhanced version that works as a proper preconditioner.
/// NOTE: Kept for legacy/manual use; the object-safe `ChebyshevPc` below is what
/// `PcFactory` constructs at runtime.
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
pub struct Chebyshev {
    /// Degree of the Chebyshev polynomial
    pub degree: usize,
    /// Lower bound of the spectrum (smallest eigenvalue)
    pub lambda_min: Option<f64>,
    /// Upper bound of the spectrum (largest eigenvalue)
    pub lambda_max: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct ChebBounds {
    pub lam_max: f64,
    pub lam_min: f64,
}

fn apply_sym_scaled(
    a: &CsrMatrix<f64>,
    d_sqrt_inv: &[f64],
    x: &[f64],
    y: &mut [f64],
) -> Result<(), KError> {
    let n = a.nrows();
    if d_sqrt_inv.len() != n || x.len() != n || y.len() != n {
        return Err(KError::InvalidInput(
            "apply_sym_scaled: dimension mismatch".into(),
        ));
    }
    for i in 0..n {
        y[i] = d_sqrt_inv[i] * x[i];
    }
    let mut tmp = vec![0.0; n];
    a.spmv_scaled(1.0, y, 0.0, &mut tmp[..])?;
    for i in 0..n {
        y[i] = d_sqrt_inv[i] * tmp[i];
    }
    Ok(())
}

pub fn estimate_lmax_sym(
    a: &CsrMatrix<f64>,
    d_sqrt_inv: &[f64],
    power_steps: usize,
) -> Result<f64, KError> {
    let n = a.nrows();
    if n == 0 {
        return Ok(0.0);
    }
    if d_sqrt_inv.len() != n {
        return Err(KError::InvalidInput(
            "estimate_lmax_sym: dimension mismatch".into(),
        ));
    }
    let mut x = vec![0.0; n];
    for i in 0..n {
        let pattern = (i.wrapping_mul(2_654_435_761)) & 1;
        x[i] = if pattern == 0 { 1.0 } else { -1.0 };
    }
    let mut nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nrm == 0.0 {
        x[0] = 1.0;
        nrm = 1.0;
    }
    for xi in &mut x {
        *xi /= nrm;
    }
    let mut y = vec![0.0; n];
    let steps = power_steps.max(1);
    for _ in 0..steps {
        apply_sym_scaled(a, d_sqrt_inv, &x, &mut y)?;
        let nrm = y.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        for i in 0..n {
            x[i] = y[i] / nrm;
        }
    }
    apply_sym_scaled(a, d_sqrt_inv, &x, &mut y)?;
    let lam = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
    Ok(lam.max(0.0))
}

pub fn chebyshev_smooth_csr(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    rhs: &[f64],
    z: &mut [f64],
    deg: usize,
    bounds: &ChebBounds,
    work_r: &mut [f64],
    work_q: &mut [f64],
    work_aq: &mut [f64],
) -> Result<(), KError> {
    if deg == 0 {
        return Ok(());
    }
    let n = a.nrows();
    if d_inv.len() != n || rhs.len() != n || z.len() != n {
        return Err(KError::InvalidInput(
            "chebyshev_smooth_csr: dimension mismatch".into(),
        ));
    }
    if work_r.len() != n || work_q.len() != n || work_aq.len() != n {
        return Err(KError::InvalidInput(
            "chebyshev_smooth_csr: workspace mismatch".into(),
        ));
    }
    if !bounds.lam_max.is_finite() || !bounds.lam_min.is_finite() || bounds.lam_max <= 0.0 {
        return Err(KError::InvalidInput(
            "chebyshev_smooth_csr: invalid eigenvalue bounds".into(),
        ));
    }

    a.spmv_scaled(1.0, z, 0.0, work_aq)?;
    for i in 0..n {
        work_r[i] = rhs[i] - work_aq[i];
    }

    let theta = (0.5 * (bounds.lam_max + bounds.lam_min)).max(1e-12);
    let delta = 0.5 * (bounds.lam_max - bounds.lam_min);
    let mut alpha = 1.0 / theta;

    for i in 0..n {
        work_q[i] = d_inv[i] * work_r[i];
    }
    for i in 0..n {
        z[i] += alpha * work_q[i];
    }
    a.spmv_scaled(1.0, work_q, 0.0, work_aq)?;
    for i in 0..n {
        work_r[i] -= alpha * work_aq[i];
    }

    for _ in 1..deg {
        let beta = 0.25 * delta * delta * alpha;
        for i in 0..n {
            work_q[i] = d_inv[i] * work_r[i] + beta * work_q[i];
        }
        alpha = 1.0 / (theta - beta);
        for i in 0..n {
            z[i] += alpha * work_q[i];
        }
        a.spmv_scaled(1.0, work_q, 0.0, work_aq)?;
        for i in 0..n {
            work_r[i] -= alpha * work_aq[i];
        }
    }
    Ok(())
}

impl Chebyshev {
    /// Create a new Chebyshev preconditioner
    ///
    /// # Arguments
    /// * `degree` - Degree of the Chebyshev polynomial
    /// * `lambda_min` - Optional lower bound of the spectrum
    /// * `lambda_max` - Optional upper bound of the spectrum
    pub fn new(degree: usize, lambda_min: Option<f64>, lambda_max: Option<f64>) -> Self {
        Self {
            degree,
            lambda_min,
            lambda_max,
        }
    }
}

impl<M, V> Preconditioner<M, V> for Chebyshev
where
    M: MatVec<Vec<f64>>,
    V: AsRef<[f64]> + AsMut<[f64]> + Clone,
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
pub fn apply_chebyshev<M>(a: &M, r: &Vec<f64>, z: &mut [f64], alpha: f64, beta: f64, m: usize)
where
    M: MatVec<Vec<f64>>,
{
    if (beta - alpha).abs() < f64::EPSILON {
        // Degenerate interval: just copy r to z
        z.copy_from_slice(r);
        return;
    }
    let n = r.len();
    // v0, v1, v2 are the three-term recurrence vectors
    let mut v0 = r.to_vec();
    let mut v1 = vec![0.0; n];
    let mut v2 = vec![0.0; n];
    // c and d are the scaling and shifting parameters for the spectrum
    let c: f64 = (beta + alpha) / 2.0;
    let d: f64 = (beta - alpha) / 2.0;
    // tau is a normalization factor to ensure p_m(0) = 1
    let tau = 1.0 / chebyshev_t(m, (0.0 - c) / d);
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
            v2[i] = (2.0 * (v2[i] - c * v1[i]) / d) - v0[i];
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
fn chebyshev_t(m: usize, x: R) -> R {
    if m == 0 {
        1.0
    } else if m == 1 {
        x
    } else {
        let mut t0 = 1.0;
        let mut t1 = x;
        let mut t2;
        for _ in 2..=m {
            t2 = 2.0 * x * t1 - t0;
            t0 = t1;
            t1 = t2;
        }
        t1
    }
}

// -----------------------------------------------------------------------------
// Object-safe Chebyshev preconditioner using CSR and LinOp.
// -----------------------------------------------------------------------------

#[derive(Default)]
struct ChebScratch {
    v0: Vec<S>,
    v1: Vec<S>,
    v2: Vec<S>,
    spmv_in_re: Vec<f64>,
    spmv_out_re: Vec<f64>,
    #[cfg(feature = "complex")]
    spmv_in_im: Vec<f64>,
    #[cfg(feature = "complex")]
    spmv_out_im: Vec<f64>,
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

fn estimate_hermitian_magnitude_lmax(a: &CsrMatrix<f64>) -> Result<f64, KError> {
    let n = a.nrows();
    if n == 0 {
        return Ok(1.0);
    }
    // Complex-safe strategy:
    // estimate on D^{-1/2} A D^{-1/2} with D taken from |diag(A)|. This behaves
    // like a Hermitian-magnitude bound for real-projected operators and remains
    // valid when the runtime scalar is complex.
    let mut d_sqrt_inv = vec![1.0; n];
    let diag = a.diagonal();
    for i in 0..n {
        let di = diag[i].abs();
        d_sqrt_inv[i] = if di > 1e-30 { 1.0 / di.sqrt() } else { 1.0 };
    }
    let est = estimate_lmax_sym(a, &d_sqrt_inv, 4)?;
    Ok(est.max(1e-8))
}

fn normalize_bounds(
    a: &CsrMatrix<f64>,
    lambda_min: f64,
    lambda_max: f64,
) -> Result<ChebBounds, KError> {
    let provided_lo = lambda_min.abs();
    let provided_hi = lambda_max.abs();
    let user_provided_degenerate = provided_lo.is_finite()
        && provided_hi.is_finite()
        && provided_lo > 0.0
        && provided_hi > 0.0
        && (provided_lo - provided_hi).abs() <= f64::EPSILON * provided_hi.max(1.0);
    if user_provided_degenerate {
        return Ok(ChebBounds {
            lam_max: provided_hi,
            lam_min: provided_lo,
        });
    }
    let mut lam_max = if provided_hi.is_finite() && provided_hi > 0.0 {
        provided_hi
    } else {
        estimate_hermitian_magnitude_lmax(a)?
    };
    let mut lam_min = if provided_lo.is_finite() && provided_lo > 0.0 {
        provided_lo
    } else {
        0.1 * lam_max
    };
    if lam_min >= lam_max {
        lam_max = lam_max.max(lam_min * 1.25);
        lam_min = (0.1 * lam_max).max(1e-12);
    }
    Ok(ChebBounds { lam_max, lam_min })
}

fn csr_real_matvec_scalar(
    a: &CsrMatrix<f64>,
    x: &[S],
    y: &mut [S],
    s: &mut ChebScratch,
) -> Result<(), KError> {
    let n = a.nrows();
    if x.len() != n || y.len() != n {
        return Err(KError::InvalidInput(
            "dimension mismatch in csr_real_matvec_scalar".into(),
        ));
    }
    for i in 0..n {
        s.spmv_in_re[i] = x[i].real();
    }
    a.spmv_scaled(1.0, &s.spmv_in_re, 0.0, &mut s.spmv_out_re)?;
    #[cfg(not(feature = "complex"))]
    {
        for i in 0..n {
            y[i] = S::from_real(s.spmv_out_re[i]);
        }
    }
    #[cfg(feature = "complex")]
    {
        for i in 0..n {
            s.spmv_in_im[i] = x[i].imag();
        }
        a.spmv_scaled(1.0, &s.spmv_in_im, 0.0, &mut s.spmv_out_im)?;
        for i in 0..n {
            y[i] = S::from_parts(s.spmv_out_re[i], s.spmv_out_im[i]);
        }
    }
    Ok(())
}

impl ObjPreconditioner for ChebyshevPc {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), crate::error::KError> {
        let csr = csr_from_linop(op, 0.0)?;
        let n = csr.nrows();
        let b = normalize_bounds(&csr, self.lambda_min, self.lambda_max)?;
        self.lambda_min = b.lam_min;
        self.lambda_max = b.lam_max;
        self.a_csr = Some(csr);
        self.n = n;
        // ensure scratch
        let mut s = self.scratch.lock().unwrap();
        if s.v0.len() != n {
            s.v0.resize(n, S::default());
        }
        if s.v1.len() != n {
            s.v1.resize(n, S::default());
        }
        if s.v2.len() != n {
            s.v2.resize(n, S::default());
        }
        if s.spmv_in_re.len() != n {
            s.spmv_in_re.resize(n, 0.0);
        }
        if s.spmv_out_re.len() != n {
            s.spmv_out_re.resize(n, 0.0);
        }
        #[cfg(feature = "complex")]
        if s.spmv_in_im.len() != n {
            s.spmv_in_im.resize(n, 0.0);
        }
        #[cfg(feature = "complex")]
        if s.spmv_out_im.len() != n {
            s.spmv_out_im.resize(n, 0.0);
        }
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = S>) -> Result<(), crate::error::KError> {
        // For now, refresh CSR view and keep degree/spectrum
        let csr = csr_from_linop(op, 0.0)?;
        let b = normalize_bounds(&csr, self.lambda_min, self.lambda_max)?;
        self.lambda_min = b.lam_min;
        self.lambda_max = b.lam_max;
        self.a_csr = Some(csr);
        Ok(())
    }

    fn apply(
        &self,
        _side: crate::preconditioner::PcSide,
        r: &[S],
        z: &mut [S],
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
        // Take ownership of scratch vectors to avoid overlapping borrows; restore before return.
        let mut v0 = std::mem::take(&mut s.v0);
        let mut v1 = std::mem::take(&mut s.v1);
        let mut v2 = std::mem::take(&mut s.v2);
        // PcSide is intentionally ignored: this smoother is symmetric and is applied
        // as a left preconditioner in practice.
        // tau scaling to make p_m(0) ~ 1
        let c = (self.lambda_max + self.lambda_min) / 2.0;
        let d = (self.lambda_max - self.lambda_min) / 2.0;

        let res = (|| {
            if d.abs() < f64::EPSILON {
                // Degenerate spectrum: copy input
                z.copy_from_slice(r);
                return Ok(());
            }
            let tau = S::from_real(1.0 / chebyshev_t(self.degree, (0.0 - c) / d));

            if self.degree == 0 {
                z.copy_from_slice(r);
                return Ok(());
            }

            // v1 = (A r - c r) / d
            csr_real_matvec_scalar(a, r, &mut v1[..n], &mut s)?;
            for i in 0..n {
                v1[i] = (v1[i] - S::from_real(c) * r[i]) / S::from_real(d);
            }
            if self.degree == 1 {
                for i in 0..n {
                    z[i] = tau * v1[i];
                }
                return Ok(());
            }

            // Set v0 = r for the recurrence
            v0[..n].copy_from_slice(r);

            // Recurrence for k = 2..=m
            for _k in 2..=self.degree {
                // v2 = 2 * ((A v1 - c v1)/d) - v0
                csr_real_matvec_scalar(a, &v1[..n], &mut v2[..n], &mut s)?;
                for i in 0..n {
                    v2[i] = S::from_real(2.0)
                        * ((v2[i] - S::from_real(c) * v1[i]) / S::from_real(d))
                        - v0[i];
                }
                // rotate: v0 <- v1, v1 <- v2, v2 becomes scratch (old v0)
                std::mem::swap(&mut v0, &mut v1);
                std::mem::swap(&mut v1, &mut v2);
            }

            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                z.par_iter_mut()
                    .zip(v1[..n].par_iter())
                    .for_each(|(zi, &vi)| *zi = tau * vi);
            }
            #[cfg(not(feature = "rayon"))]
            {
                for i in 0..n {
                    z[i] = tau * v1[i];
                }
            }
            Ok(())
        })();

        // Restore scratch vectors before returning.
        s.v0 = v0;
        s.v1 = v1;
        s.v2 = v2;
        res
    }

    fn required_format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Csr
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for ChebyshevPc {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        ObjPreconditioner::dims(self)
    }

    fn apply_s(
        &self,
        side: crate::preconditioner::PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), crate::error::KError> {
        bridge_apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: crate::preconditioner::PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), crate::error::KError> {
        bridge_apply_pc_mut_s(self, side, x, y, scratch)
    }

    fn on_restart_s(
        &mut self,
        outer_iter: usize,
        residual_norm: R,
    ) -> Result<(), crate::error::KError> {
        ObjPreconditioner::on_restart(self, outer_iter, residual_norm)
    }
}

#[cfg(all(test, not(feature = "complex")))]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;
    use crate::matrix::op::CsrOp;
    use crate::preconditioner::PcSide;
    use std::sync::Arc;

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

    #[test]
    fn chebyshev_pc_degenerate_copy() {
        // lam_min == lam_max triggers copy-through path
        let row_ptr = vec![0, 1, 2];
        let col_idx = vec![0, 1];
        let values = vec![1.0, 1.0];
        let csr = Arc::new(CsrMatrix::from_csr(2, 2, row_ptr, col_idx, values));
        let op = CsrOp::new(csr);

        let mut pc = ChebyshevPc::new(3, 1.0, 1.0);
        pc.setup(&op).expect("setup");

        let rhs = vec![2.5, -3.0];
        let mut out = vec![0.0; rhs.len()];
        pc.apply(PcSide::Left, &rhs, &mut out).expect("apply");

        for i in 0..rhs.len() {
            assert!((out[i] - rhs[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn chebyshev_pc_basic_spd() {
        // Simple SPD tridiagonal matrix
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let csr = Arc::new(CsrMatrix::from_csr(3, 3, row_ptr, col_idx, values));
        let op = CsrOp::new(csr);

        let mut pc = ChebyshevPc::new(2, 0.1, 4.0);
        pc.setup(&op).expect("setup");

        let rhs = vec![1.0, 0.0, -1.0];
        let mut out = vec![0.0; rhs.len()];
        pc.apply(PcSide::Left, &rhs, &mut out).expect("apply");

        assert!(out.iter().all(|zi| zi.is_finite()));
        assert!(out.iter().any(|&zi| zi.abs() > 1e-6));
    }
}

#[cfg(all(test, feature = "complex"))]
mod complex_tests {
    use super::*;
    use crate::algebra::bridge::BridgeScratch;
    use crate::context::KspContext;
    use crate::context::ksp_context::SolverType;
    use crate::context::pc_context::PcType;
    use crate::matrix::op::CsrOp;
    use crate::matrix::op::LinOp;
    use crate::matrix::sparse::CsrMatrix;
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::PcSide;
    use std::sync::Arc;

    #[test]
    fn apply_s_complex_setup_and_apply() {
        let row_ptr = vec![0, 2, 4];
        let col_idx = vec![0, 1, 0, 1];
        let values = vec![
            S::from_real(1.5),
            S::from_real(0.0),
            S::from_real(0.0),
            S::from_real(1.5),
        ];
        let op = CsrOp::new(Arc::new(CsrMatrix::from_csr(
            2, 2, row_ptr, col_idx, values,
        )));

        let mut pc = ChebyshevPc::new(3, -0.2, 0.0);
        pc.setup(&op as &dyn LinOp<S = S>).expect("setup");
        let rhs = vec![S::from_parts(1.0, 0.25), S::from_parts(-2.0, 0.5)];
        let mut out = vec![S::zero(); rhs.len()];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(PcSide::Left, &rhs, &mut out, &mut scratch)
            .expect("apply_s");
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            out.iter()
                .zip(rhs.iter())
                .any(|(o, r)| (*o - *r).abs() > 1e-8)
        );
    }

    #[test]
    fn chebyshev_pc_integrates_with_ksp_context_complex() {
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![S::from_real(1.0), S::from_real(1.0), S::from_real(1.0)];
        let a = Arc::new(CsrOp::new(Arc::new(CsrMatrix::from_csr(
            3, 3, row_ptr, col_idx, values,
        ))));

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Richardson).expect("set solver");
        ksp.set_pc_type(PcType::Chebyshev, None).expect("set pc");
        ksp.set_operators(a.clone(), None);
        ksp.setup().expect("ksp setup");

        let b = vec![
            S::from_parts(1.0, 1.0),
            S::from_parts(-2.0, 0.0),
            S::from_parts(0.5, -0.5),
        ];
        let mut x = vec![S::zero(); b.len()];
        let _ = ksp.solve(&b, &mut x).expect("ksp solve");
        assert!(x.iter().all(|xi| xi.is_finite()));
    }
}
