#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
//! Tests for core dense matrix operations: matrix-vector multiplication, dot product, and norm.
//!
//! These tests verify the correctness of the MatVec and InnerProduct trait implementations
//! for dense matrices and vectors, using random and fixed data.

use approx::assert_abs_diff_eq;
use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::core::traits::{InnerProduct, MatVec};
use rand::Rng;

/// Test matrix-vector multiplication for a small random dense matrix.
///
/// This test constructs a random 5x5 matrix and a random vector, computes the matrix-vector
/// product using the MatVec trait, and checks the result against a manual computation.
#[test]
fn matvec_random_small() {
    let n = 5;
    let mut rng = rand::thread_rng();
    let vals: Vec<R> = (0..n * n).map(|_| rng.r#gen()).collect();
    // Use from_fn to build a column-major matrix
    let a = Mat::<R>::from_fn(n, n, |i, j| vals[j * n + i]);
    let x: Vec<R> = (0..n).map(|_| rng.r#gen()).collect();
    let mut y = vec![R::default(); n];
    a.matvec(&x, &mut y);

    // check y[i] == sum_j A[i,j]*x[j]
    for i in 0..n {
        let expected = (0..n).map(|j| vals[j * n + i] * x[j]).sum::<R>();
        assert_abs_diff_eq!(y[i], expected, epsilon = 1e-12);
    }
}

/// Test dot product and Euclidean norm for small vectors.
///
/// This test verifies that the InnerProduct trait correctly computes the dot product and
/// the Euclidean norm (L2 norm) for two small vectors, comparing against manual calculations.
#[test]
fn dot_and_norm() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let x = vec![R::from(1.0), R::from(2.0), R::from(3.0)];
    let y = vec![R::from(4.0), R::from(-5.0), R::from(6.0)];
    let ip = ();
    let dot = ip.dot(&x, &y, &comm);
    assert_abs_diff_eq!(
        dot,
        R::from(1.0) * R::from(4.0) + R::from(2.0) * R::from(-5.0) + R::from(3.0) * R::from(6.0),
        epsilon = 1e-12
    );
    let norm_x = ip.norm(&x, &comm);
    let expected_norm =
        ((R::from(1.0)).powi(2) + R::from(2.0).powi(2) + R::from(3.0).powi(2)).sqrt();
    assert_abs_diff_eq!(norm_x, expected_norm, epsilon = 1e-12);
}
