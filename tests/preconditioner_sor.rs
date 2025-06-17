//! SOR/SSOR preconditioner tests for kryst
//!
//! This module contains unit tests for the Successive Over-Relaxation (SOR) and Symmetric SOR (SSOR)
//! preconditioners implemented in the kryst library. The tests verify correct application of the preconditioner
//! to identity and tridiagonal matrices, as well as correct formatting and finite output.
//!
//! - SOR is an iterative method for solving linear systems, often used as a preconditioner.
//! - SSOR is a symmetric variant that applies both forward and backward sweeps.
//!
//! The tests use the `faer` crate for matrix construction and `approx` for floating-point comparisons.

use approx::assert_relative_eq;
use kryst::preconditioner::{Sor, MatSorType, Preconditioner};
use faer::Mat;

/// Constructs a tridiagonal matrix of size `n` with subdiagonal `a`, diagonal `b`, and superdiagonal `c`.
///
/// # Arguments
/// * `n` - Size of the matrix (n x n)
/// * `a` - Value for subdiagonal entries
/// * `b` - Value for diagonal entries
/// * `c` - Value for superdiagonal entries
fn make_tridiag(n: usize, a: f64, b: f64, c: f64) -> Mat<f64> {
    let mut mat = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        if i > 0 { mat[(i, i-1)] = a; }
        mat[(i, i)] = b;
        if i+1 < n { mat[(i, i+1)] = c; }
    }
    mat
}

/// Constructs an identity matrix of size `n`.
fn make_eye(n: usize) -> Mat<f64> {
    let mut mat = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        mat[(i, i)] = 1.0;
    }
    mat
}

/// Test that applying SOR to the identity matrix returns the input vector unchanged.
#[test]
fn test_sor_identity() {
    let n = 5;
    let a = make_eye(n);
    // SOR with omega=1.0, 1 sweep, forward (lower) application
    let mut sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.0, 1, 1, MatSorType::APPLY_LOWER, 0.0);
    sor.setup(&a).unwrap();
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    sor.apply(&x, &mut y).unwrap();
    // Output should match input exactly
    assert_relative_eq!(x.as_slice(), y.as_slice(), epsilon=1e-12);
}

/// Test SOR on a tridiagonal matrix with a forward sweep.
///
/// Compares the result to a manually computed expected SOR sweep.
#[test]
fn test_sor_tridiag_forward() {
    let n = 5;
    let a = make_tridiag(n, -1.0, 4.0, -1.0);
    // SOR with omega=1.0, 1 sweep, forward (lower) application
    let mut sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.0, 1, 1, MatSorType::APPLY_LOWER, 0.0);
    sor.setup(&a).unwrap();
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    sor.apply(&x, &mut y).unwrap();
    // Compute expected SOR sweep (forward, in-place)
    let mut expected = vec![0.0; n];
    for i in 0..n {
        let left = if i > 0 { expected[i-1] } else { 0.0 };
        let right = if i+1 < n { x[i+1] } else { 0.0 };
        expected[i] = (x[i] + left + right) / 4.0;
    }
    // Compare each entry with a tight tolerance
    for i in 0..n {
        assert!((y[i] - expected[i]).abs() < 1e-12_f64, "SOR mismatch at i={}: got {}, expected {}", i, y[i], expected[i]);
    }
}

/// Test SSOR (symmetric SOR) on a tridiagonal matrix.
///
/// Verifies that the output is finite for all entries.
#[test]
fn test_ssor_tridiag() {
    let n = 5;
    let a = make_tridiag(n, -1.0, 4.0, -1.0);
    // SSOR: symmetric sweep
    let mut sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.0, 1, 1, MatSorType::SYMMETRIC_SWEEP, 0.0);
    sor.setup(&a).unwrap();
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    sor.apply(&x, &mut y).unwrap();
    // All output values should be finite
    assert!(y.iter().all(|&v| (v as f64).is_finite()));
}

/// Test the Display implementation for SOR.
///
/// Checks that the formatted string contains the expected parameter values.
#[test]
fn test_sor_display() {
    let sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.5, 2, 1, MatSorType::APPLY_LOWER, 0.1);
    let s = format!("{}", sor);
    assert!(s.contains("SOR(omega=1.5"));
}
