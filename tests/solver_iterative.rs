//! Tests for iterative solvers (CG, GMRES) vs direct solvers on random matrices.
//!
//! This module verifies that the implemented iterative solvers (Conjugate Gradient and GMRES)
//! produce solutions that closely match those from direct solvers (LU and QR) on small random
//! systems. The tests use random SPD and non-symmetric matrices, and compare the results
//! elementwise within a tight tolerance.

use faer::Mat;
use faer::linalg::solvers::SolveCore;
use kryst::solver::{CgSolver, GmresSolver, LinearSolver};
use rand::Rng;
use approx::assert_abs_diff_eq;

/// Helper function to generate a random symmetric positive definite (SPD) matrix `A` and a random right-hand side `b`.
///
/// The SPD matrix is constructed as `A = Mᵀ M + I`, where `M` is a random matrix and `I` is the identity.
/// This ensures that `A` is symmetric and positive definite, suitable for CG.
fn random_spd(n: usize) -> (faer::Mat<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n*n).map(|_| rng.r#gen()).collect();
    let m = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let m_t = m.transpose();
    let a = &m_t * &m + Mat::<f64>::identity(n, n);
    let b: Vec<f64> = (0..n).map(|_| rng.r#gen()).collect();
    (a, b)
}

/// Test that the Conjugate Gradient (CG) solver produces a solution matching the direct LU solver on a random SPD system.
///
/// - Generates a random SPD matrix and right-hand side.
/// - Solves with CG and checks for convergence.
/// - Solves with direct LU and compares the solutions elementwise.
#[test]
fn cg_vs_direct_on_spd() {
    let n = 10;
    let (a, b) = random_spd(n);
    let mut x_cg = vec![0.0; n];
    let mut solver = CgSolver::new(1e-8, 1000);
    let stats = solver.solve(&a, None, &b, &mut x_cg).unwrap();
    assert!(stats.converged);
    // Direct solve using LU decomposition
    let mut x_direct = b.clone();
    let lus = faer::linalg::solvers::FullPivLu::new(a.as_ref());
    let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_direct, n, 1);
    lus.solve_in_place_with_conj(faer::Conj::No, x_mat);
    // Compare each element of the solutions
    for i in 0..n {
        assert_abs_diff_eq!(x_cg[i], x_direct[i], epsilon = 1e-6);
    }
}

/// Test that the GMRES solver produces a solution matching the direct QR solver on a random non-symmetric system.
///
/// - Generates a random (possibly non-symmetric) matrix and right-hand side.
/// - Solves with GMRES and checks for convergence.
/// - Solves with direct QR and compares the solutions elementwise.
#[test]
fn gmres_vs_direct_on_nonsymmetric() {
    let n = 10;
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n*n).map(|_| rng.r#gen()).collect();
    let a = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let b: Vec<f64> = (0..n).map(|_| rng.r#gen()).collect();
    let mut x_gmres = vec![0.0; n];
    let mut solver = GmresSolver::new(100, 1e-8, 1000);
    let stats = solver.solve(&a, None, &b, &mut x_gmres).unwrap();
    assert!(stats.converged);
    // Direct solve using QR decomposition
    let mut x_direct = b.clone();
    let qr = faer::linalg::solvers::Qr::new(a.as_ref());
    let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_direct, n, 1);
    qr.solve_in_place_with_conj(faer::Conj::No, x_mat);
    // Compare each element of the solutions
    for i in 0..n {
        assert_abs_diff_eq!(x_gmres[i], x_direct[i], epsilon = 1e-6);
    }
}
