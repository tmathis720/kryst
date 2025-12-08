#![cfg(feature = "backend-faer")]

//! Incomplete LU factorization with threshold and pivoting (ILUTP).
//!
//! This module implements the ILUTP algorithm which combines threshold-based dropping
//! with partial pivoting to improve stability for difficult matrices such as those
//! arising from discretized Navier-Stokes equations.
//! The factorization is kept in `f64`, but a `KPreconditioner` bridge exposes the same
//! functionality to complex solvers via `BridgeScratch`.
//!
//! For non-pivoting ILUT on `faer::Mat<f64>`, prefer the canonical `Ilu` implementation
//! with `IluType::ILUT`.
//!
//! The algorithm follows the approach described in Saad's "Iterative Methods for
//! Sparse Linear Systems" with modifications for numerical stability.

#[cfg(feature = "complex")]
use crate::algebra::bridge::{BridgeScratch, copy_real_into_scalar, copy_scalar_to_real_in};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::{LocalPreconditioner, legacy::Preconditioner};
use faer::Mat;
use std::cmp::Ordering;

/// ILUTP preconditioner with threshold control and partial pivoting.
///
/// This preconditioner is particularly effective for nonsymmetric, indefinite
/// matrices arising from discretized Navier-Stokes equations.
/// It implements [`LocalPreconditioner`] and is designed to stay on the local rank
/// so that MPI routines can wrap it without additional communication.
#[derive(Debug)]
pub struct Ilutp {
    /// Lower triangular factor
    l_factor: Mat<f64>,
    /// Upper triangular factor
    u_factor: Mat<f64>,
    /// Row permutation vector
    row_perm: Vec<usize>,
    /// Maximum fill-in per row
    max_fill: usize,
    /// Drop tolerance for small elements
    drop_tol: f64,
    /// Pivot tolerance (threshold for pivoting)
    perm_tol: f64,
}

impl Ilutp {
    /// Create a new ILUTP preconditioner with default parameters.
    pub fn new() -> Self {
        Self {
            l_factor: Mat::zeros(0, 0),
            u_factor: Mat::zeros(0, 0),
            row_perm: Vec::new(),
            max_fill: 10,
            drop_tol: 1e-4,
            perm_tol: 0.1,
        }
    }

    /// Create a new ILUTP preconditioner with custom parameters.
    pub fn with_params(max_fill: usize, drop_tol: f64, perm_tol: f64) -> Self {
        Self {
            l_factor: Mat::zeros(0, 0),
            u_factor: Mat::zeros(0, 0),
            row_perm: Vec::new(),
            max_fill,
            drop_tol,
            perm_tol,
        }
    }

    /// Set the maximum fill-in per row.
    pub fn set_max_fill(&mut self, max_fill: usize) {
        self.max_fill = max_fill;
    }

    /// Set the drop tolerance.
    pub fn set_drop_tolerance(&mut self, drop_tol: f64) {
        self.drop_tol = drop_tol;
    }

    /// Set the pivot tolerance.
    pub fn set_pivot_tolerance(&mut self, perm_tol: f64) {
        self.perm_tol = perm_tol;
    }

    /// Compute ILUTP factorization with simplified algorithm.
    fn compute_factorization(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(KError::SolveError(
                "Matrix must be square for ILUTP".to_string(),
            ));
        }

        // Initialize working matrix as a copy
        let mut work_matrix = matrix.clone();

        // Initialize permutation as identity
        self.row_perm = (0..n).collect();

        // Initialize L and U factors
        let mut l_factor = Mat::zeros(n, n);
        let mut u_factor = Mat::zeros(n, n);

        // Set diagonal of L to 1
        for i in 0..n {
            l_factor[(i, i)] = 1.0;
        }

        // Simplified ILUTP factorization
        for k in 0..n {
            // Find pivot row (simplified pivoting with zero avoidance)
            let mut pivot_row = k;
            let mut max_val = work_matrix[(k, k)].abs();

            // Look for a better pivot if current is too small
            if max_val < 1e-12 {
                for i in (k + 1)..n {
                    let val = work_matrix[(i, k)].abs();
                    if val > max_val {
                        max_val = val;
                        pivot_row = i;
                    }
                }
            } else {
                // Standard partial pivoting
                for i in (k + 1)..n {
                    let val = work_matrix[(i, k)].abs();
                    if val > max_val && val > self.perm_tol * max_val {
                        max_val = val;
                        pivot_row = i;
                    }
                }
            }

            // Swap rows if needed
            if pivot_row != k {
                self.row_perm.swap(k, pivot_row);
                for j in 0..n {
                    let temp = work_matrix[(k, j)];
                    work_matrix[(k, j)] = work_matrix[(pivot_row, j)];
                    work_matrix[(pivot_row, j)] = temp;
                }
            }

            let pivot = work_matrix[(k, k)];
            if pivot.abs() < 1e-12 {
                // TODO: consider reusing `preconditioner::pivot` handling (e.g., `PivotPolicy`)
                // for consistent stabilization instead of this ad-hoc regularization.
                // Handle near-zero pivot by regularization
                work_matrix[(k, k)] = if pivot >= 0.0 { 1e-12 } else { -1e-12 };
                eprintln!(
                    "Warning: Near-zero pivot at row {} regularized from {} to {}",
                    k,
                    pivot,
                    work_matrix[(k, k)]
                );
            }

            // Store U factor
            for j in k..n {
                u_factor[(k, j)] = work_matrix[(k, j)];
            }

            // Elimination with threshold dropping
            for i in (k + 1)..n {
                if work_matrix[(i, k)].abs() > self.drop_tol {
                    let multiplier = work_matrix[(i, k)] / pivot;
                    l_factor[(i, k)] = multiplier;

                    // Collect row elements for fill-in control
                    let mut row_elements = Vec::new();
                    for j in (k + 1)..n {
                        let new_val = work_matrix[(i, j)] - multiplier * work_matrix[(k, j)];
                        if new_val.abs() > self.drop_tol {
                            row_elements.push((j, new_val));
                        }
                    }

                    // Apply fill-in control
                    if row_elements.len() > self.max_fill {
                        row_elements.sort_by(|a, b| {
                            b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal)
                        });
                        row_elements.truncate(self.max_fill);
                    }

                    // Clear row and set selected elements
                    for j in (k + 1)..n {
                        work_matrix[(i, j)] = 0.0;
                    }
                    for (j, val) in row_elements {
                        work_matrix[(i, j)] = val;
                    }
                }
            }
        }

        self.l_factor = l_factor;
        self.u_factor = u_factor;
        Ok(())
    }

    /// Forward substitution: solve Ly = Pb
    fn forward_solve(&self, b: &[f64], y: &mut [f64]) -> Result<(), KError> {
        let n = self.l_factor.nrows();

        // Apply row permutation
        for i in 0..n {
            y[i] = b[self.row_perm[i]];
        }

        // Forward substitution: Ly = Pb
        for i in 0..n {
            for j in 0..i {
                y[i] -= self.l_factor[(i, j)] * y[j];
            }
            // L has unit diagonal, so no division needed
        }

        Ok(())
    }

    /// Backward substitution: solve Ux = y
    fn backward_solve(&self, y: &[f64], x: &mut [f64]) -> Result<(), KError> {
        let n = self.u_factor.nrows();
        x.copy_from_slice(y);

        // Backward substitution: Ux = y
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= self.u_factor[(i, j)] * x[j];
            }
            let diag = self.u_factor[(i, i)];
            if diag.abs() < 1e-14 {
                return Err(KError::SolveError(format!(
                    "Singular U factor at diagonal {i}"
                )));
            }
            x[i] /= diag;
        }

        Ok(())
    }

    fn apply_slice(&self, input: &[f64], output: &mut [f64]) -> Result<(), KError> {
        let n = input.len();
        let expected = self.l_factor.nrows();
        if output.len() != n || expected != n {
            return Err(KError::InvalidInput(format!(
                "Ilutp::apply_slice dimension mismatch: expected {}, got input.len()={}, output.len()={}",
                expected,
                n,
                output.len()
            )));
        }

        let mut temp = vec![0.0; n];
        self.forward_solve(input, &mut temp)?;
        self.backward_solve(&temp, output)?;
        Ok(())
    }
}

impl Default for Ilutp {
    fn default() -> Self {
        Self::new()
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for Ilutp {
    /// Setup the ILUTP preconditioner by computing the factorization.
    fn setup(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        self.compute_factorization(matrix)
    }

    /// Apply the ILUTP preconditioner: solve M⁻¹x = b where M ≈ A.
    fn apply(
        &self,
        _side: crate::preconditioner::PcSide,
        input: &Vec<f64>,
        output: &mut Vec<f64>,
    ) -> Result<(), KError> {
        self.apply_slice(input, output)
    }
}

impl LocalPreconditioner<f64> for Ilutp {
    fn dims(&self) -> (usize, usize) {
        let n = self.l_factor.nrows();
        (n, n)
    }

    fn apply_local(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        let (n, _) = LocalPreconditioner::<f64>::dims(self);
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        self.apply_slice(x, y)
    }
}

#[cfg(test)]
impl Ilutp {
    pub fn test_max_fill(&self) -> usize {
        self.max_fill
    }

    pub fn test_drop_tolerance(&self) -> f64 {
        self.drop_tol
    }

    pub fn test_perm_tolerance(&self) -> f64 {
        self.perm_tol
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for Ilutp {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        let n = self.l_factor.nrows();
        (n, n)
    }

    fn apply_s(
        &self,
        _side: crate::preconditioner::PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        let (rows, cols) = LocalPreconditioner::<f64>::dims(self);
        let n = x.len();
        if x.len() != y.len() || rows != n || cols != n {
            return Err(KError::InvalidInput(format!(
                "Ilutp::apply_s dimension mismatch: expected {}x{}, got x.len()={} y.len()={}",
                rows,
                cols,
                x.len(),
                y.len()
            )));
        }
        scratch.with_pair(n, |xr, yr| {
            copy_scalar_to_real_in(x, xr);
            self.apply_slice(xr, yr)?;
            copy_real_into_scalar(yr, y);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn test_ilutp_basic() {
        let mut ilutp = Ilutp::new();

        // Test matrix: simple 3x3 diagonally dominant
        let mut matrix = Mat::zeros(3, 3);
        matrix[(0, 0)] = 4.0;
        matrix[(0, 1)] = -1.0;
        matrix[(1, 0)] = -1.0;
        matrix[(1, 1)] = 4.0;
        matrix[(1, 2)] = -1.0;
        matrix[(2, 1)] = -1.0;
        matrix[(2, 2)] = 4.0;

        // Setup should succeed
        assert!(ilutp.setup(&matrix).is_ok());

        // Test application
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        assert!(
            ilutp
                .apply(crate::preconditioner::PcSide::Left, &input, &mut output)
                .is_ok()
        );
    }

    #[test]
    fn test_ilutp_with_custom_params() {
        let mut ilutp = Ilutp::with_params(5, 1e-6, 0.01);

        // Test parameter setting
        ilutp.set_max_fill(15);
        ilutp.set_drop_tolerance(1e-8);
        ilutp.set_pivot_tolerance(0.1);

        assert_eq!(ilutp.max_fill, 15);
        assert_eq!(ilutp.drop_tol, 1e-8);
        assert_eq!(ilutp.perm_tol, 0.1);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn apply_s_matches_real_path() {
        use crate::algebra::bridge::BridgeScratch;
        use crate::algebra::prelude::*;
        use crate::ops::kpc::KPreconditioner;

        let mut ilutp = Ilutp::with_params(4, 1e-6, 0.05);

        let mut matrix = Mat::zeros(2, 2);
        matrix[(0, 0)] = 5.0;
        matrix[(0, 1)] = -1.0;
        matrix[(1, 0)] = -2.0;
        matrix[(1, 1)] = 6.0;
        ilutp.setup(&matrix).unwrap();

        let rhs_real = vec![3.0f64, -1.0];
        let mut out_real = vec![0.0; rhs_real.len()];
        ilutp
            .apply(
                crate::preconditioner::PcSide::Left,
                &rhs_real,
                &mut out_real,
            )
            .expect("ilutp real apply");

        let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
        let mut out_s = vec![S::zero(); rhs_s.len()];
        let mut scratch = BridgeScratch::default();
        ilutp
            .apply_s(
                crate::preconditioner::PcSide::Left,
                &rhs_s,
                &mut out_s,
                &mut scratch,
            )
            .expect("ilutp apply_s");

        for (ys, yr) in out_s.iter().zip(out_real.iter()) {
            assert!((ys.real() - yr).abs() < 1e-10);
        }
    }
}
