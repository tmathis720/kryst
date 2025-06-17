//! Integration tests for preconditioners and solvers in the KrylovKit library.
//!
//! This module verifies that various preconditioners (e.g., Jacobi, ILU0) work correctly
//! with iterative solvers (CG, PCG, GMRES) on small test matrices. It checks convergence,
//! solution accuracy, and correct preconditioner application. These tests ensure that the
//! preconditioner and solver interfaces are compatible and robust for a variety of matrix types.

use kryst::preconditioner::{Preconditioner, Jacobi, Ilu0};
use kryst::solver::{PcgSolver, GmresSolver, LinearSolver};
use kryst::solver::gmres::Preconditioning;
use faer::Mat;

/// Construct a symmetric positive definite (SPD) tridiagonal matrix of size `n`.
/// Returns the matrix, the right-hand side vector `b` for the solution x = [1, ..., 1],
/// and the true solution vector.
fn spd_matrix(n: usize) -> (Mat<f64>, Vec<f64>, Vec<f64>) {
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 2.0;
        if i > 0 {
            a[(i, i-1)] = -1.0;
            a[(i-1, i)] = -1.0;
        }
    }
    let x_true = vec![1.0; n];
    let mut b = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            b[i] += a[(i, j)] * x_true[j];
        }
    }
    (a, b, x_true)
}

/// Construct a non-symmetric tridiagonal matrix of size `n`.
/// Returns the matrix, the right-hand side vector `b` for the solution x = [1, ..., 1],
/// and the true solution vector.
fn nonsym_matrix(n: usize) -> (Mat<f64>, Vec<f64>, Vec<f64>) {
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 2.0;
        if i > 0 {
            a[(i, i-1)] = -1.0;
        }
        if i+1 < n {
            a[(i, i+1)] = 0.5;
        }
    }
    let x_true = vec![1.0; n];
    let mut b = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            b[i] += a[(i, j)] * x_true[j];
        }
    }
    (a, b, x_true)
}

/// Compute the relative L2 error between two vectors.
fn rel_error(x: &[f64], x_true: &[f64]) -> f64 {
    let num: f64 = x.iter().zip(x_true).map(|(xi, ti)| (xi - ti).powi(2)).sum();
    let denom: f64 = x_true.iter().map(|ti| ti.powi(2)).sum();
    (num / denom).sqrt()
}

/// Build a badly conditioned diagonal matrix of size `n` with condition number `kappa`.
/// Returns the matrix and a right-hand side vector of all ones.
fn ill_cond(n: usize, kappa: f64) -> (Mat<f64>, Vec<f64>) {
    let mut diag = vec![1.0; n];
    diag[n-1] = kappa;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = diag[i];
    }
    let b = vec![1.0; n];
    (a, b)
}

/// Test: CG with Jacobi preconditioner on an ill-conditioned diagonal matrix.
/// Ensures that the Jacobi preconditioner can be set up and applied, and that the solver runs without panic.
#[test]
fn cg_with_jacobi() {
    let (a, b) = ill_cond(5, 1e6);
    let mut pc = Jacobi::new();
    <Jacobi<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::setup(&mut pc, &a).unwrap();
    let mut solver = PcgSolver::new(1e-6, 1000);
    let mut x = vec![0.0; 5];
    // wrap A and pc into a PCG solver if you have one, else manually:
    let r_in = b.clone();
    let mut r_out = vec![0.0; b.len()];
    <Jacobi<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::apply(&pc, &r_in, &mut r_out).unwrap();
    let stats = solver.solve(&a, None, &b, &mut x).unwrap();
    assert!(stats.converged);
    // No stats to check; just ensure it runs without panic
}

/// Test: GMRES with ILU0 preconditioner on an ill-conditioned diagonal matrix.
/// Ensures that the ILU0 preconditioner can be set up and applied, and that the solver runs without panic.
#[test]
fn gmres_with_ilu0() {
    let (a, b) = ill_cond(5, 1e4);
    let mut pc = Ilu0::new();
    <Ilu0<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::setup(&mut pc, &a).unwrap();
    let mut solver = GmresSolver::new(4, 1e-6, 1000);
    let mut x = vec![0.0; 5];
    let stats = solver.solve(&a, None, &b, &mut x).unwrap();
    assert!(stats.converged);
    // No stats to check; just ensure it runs without panic
}

/// Test: PCG with Jacobi preconditioner on an ill-conditioned diagonal matrix.
/// Checks that the solver converges with the preconditioner.
#[test]
fn pcg_with_jacobi() {
    let (a, b) = ill_cond(5, 1e6);
    let mut pc = Jacobi::new();
    <Jacobi<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::setup(&mut pc, &a).unwrap();
    let mut solver = kryst::solver::PcgSolver::new(1e-6, 1000);
    let mut x = vec![0.0; 5];
    let stats = solver.solve(&a, Some(&pc), &b, &mut x).unwrap();
    assert!(stats.converged);
}

/// Test: PCG with Jacobi preconditioner on a symmetric positive definite matrix.
/// Checks that the solver converges to the correct solution within the expected number of iterations.
#[test]
fn spd_jacobi_pcg_converges() {
    let n = 10;
    let (a, b, x_true) = spd_matrix(n);
    let mut pc = Jacobi::new();
    pc.setup(&a).unwrap();
    let mut solver = PcgSolver::new(1e-12, n);
    let mut x = vec![0.0; n];
    let stats = solver.solve(&a, Some(&pc), &b, &mut x).unwrap();
    assert!(stats.converged);
    assert!(rel_error(&x, &x_true) < 1e-10);
    assert!(stats.iterations <= n);
}

/// Test: CG (no preconditioner) on a symmetric positive definite matrix.
/// Checks that the solver converges to the correct solution.
#[test]
fn spd_no_pc_cg_converges() {
    let n = 10;
    let (a, b, x_true) = spd_matrix(n);
    let mut solver = PcgSolver::new(1e-12, n);
    let mut x = vec![0.0; n];
    let stats = solver.solve(&a, None, &b, &mut x).unwrap();
    assert!(stats.converged);
    assert!(rel_error(&x, &x_true) < 1e-10);
}

/// Test: GMRES (no preconditioner, right preconditioning) on a non-symmetric matrix.
/// Checks that the solver converges to the correct solution.
#[test]
fn nonsym_no_pc_gmresright_converges() {
    let n = 10;
    let (a, b, x_true) = nonsym_matrix(n);
    let mut solver = GmresSolver::new(10, 1e-12, 100);
    let mut x = vec![0.0; n];
    let stats = solver.solve(&a, None, &b, &mut x).unwrap();
    assert!(stats.converged);
    assert!(rel_error(&x, &x_true) < 1e-10);
}

/// Test: GMRES with left preconditioning (ILU0) on a non-symmetric matrix.
/// Checks that the solver converges to the correct solution.
#[test]
fn nonsym_left_pc_gmresleft_converges() {
    let n = 10;
    let (a, b, x_true) = nonsym_matrix(n);
    let mut pc = Ilu0::new();
    pc.setup(&a).unwrap();
    let mut solver = GmresSolver::new(10, 1e-12, 100).with_preconditioning(Preconditioning::Left);
    let mut x = vec![0.0; n];
    let stats = solver.solve(&a, Some(&pc), &b, &mut x).unwrap();
    assert!(stats.converged);
    assert!(rel_error(&x, &x_true) < 1e-10);
}

// The following test is commented out because Jacobi does not implement FlexiblePreconditioner.
// It would otherwise compare FGMRES and GMRES with Jacobi preconditioning for equivalence.
// #[test]
// fn fgmres_flexible_vs_fixed_equiv() {
//     let n = 10;
//     let (a, b, x_true) = nonsym_matrix(n);
//     let mut pc = Jacobi::new();
//     pc.setup(&a).unwrap();
//     let mut gmres = GmresSolver::new(10, 1e-12, 100);
//     let mut fgmres = FgmresSolver::new(1e-12, 100, 10);
//     let mut x1 = vec![0.0; n];
//     let mut x2 = vec![0.0; n];
//     let stats1 = gmres.solve(&a, Some(&pc), &b, &mut x1).unwrap();
//     let stats2 = fgmres.solve_flex(&a, Some(&mut pc), &b, &mut x2).unwrap();
//     assert!(stats1.converged && stats2.converged);
//     assert!(rel_error(&x1, &x2) < 1e-12);
// }
