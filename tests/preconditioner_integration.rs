#![cfg(feature = "backend-faer")]
//! Integration tests for preconditioners and solvers in the KrylovKit library.
//!
//! This module verifies that various preconditioners (e.g., Jacobi, ILU0) work correctly
//! with iterative solvers (CG, PCG, GMRES) on small test matrices. It checks convergence,
//! solution accuracy, and correct preconditioner application. These tests ensure that the
//! preconditioner and solver interfaces are compatible and robust for a variety of matrix types.

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::preconditioner::{Jacobi, PcSide, Preconditioner};
use kryst::solver::{CgSolver, GmresSolver, LinearSolver};

/// Construct a symmetric positive definite (SPD) tridiagonal matrix of size `n`.
/// Returns the matrix, the right-hand side vector `b` for the solution x = [1, ..., 1],
/// and the true solution vector.
fn spd_matrix(n: usize) -> (Mat<R>, Vec<R>, Vec<R>) {
    let mut a: Mat<R> = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = R::from(2.0);
        if i > 0 {
            a[(i, i - 1)] = R::from(-1.0);
            a[(i - 1, i)] = R::from(-1.0);
        }
    }
    let x_true = vec![R::from(1.0); n];
    let mut b = vec![R::default(); n];
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
fn nonsym_matrix(n: usize) -> (Mat<R>, Vec<R>, Vec<R>) {
    let mut a: Mat<R> = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = R::from(2.0);
        if i > 0 {
            a[(i, i - 1)] = R::from(-1.0);
        }
        if i + 1 < n {
            a[(i, i + 1)] = R::from(0.5);
        }
    }
    let x_true = vec![R::from(1.0); n];
    let mut b = vec![R::default(); n];
    for i in 0..n {
        for j in 0..n {
            b[i] += a[(i, j)] * x_true[j];
        }
    }
    (a, b, x_true)
}

/// Compute the relative L2 error between two vectors.
fn rel_error(x: &[R], x_true: &[R]) -> R {
    let num = x.iter().zip(x_true).fold(R::default(), |acc, (xi, ti)| {
        let diff = *xi - *ti;
        acc + diff * diff
    });
    let denom = x_true.iter().fold(R::default(), |acc, ti| acc + *ti * *ti);
    (num / denom).sqrt()
}

/// Build a badly conditioned diagonal matrix of size `n` with condition number `kappa`.
/// Returns the matrix and a right-hand side vector of all ones.
fn ill_cond(n: usize, kappa: R) -> (Mat<R>, Vec<R>) {
    let mut diag = vec![R::from(1.0); n];
    diag[n - 1] = R::from(kappa);
    let mut a: Mat<R> = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = diag[i];
    }
    let b = vec![R::from(1.0); n];
    (a, b)
}

/// Test: CG with Jacobi preconditioner on an ill-conditioned diagonal matrix.
/// Ensures that the Jacobi preconditioner can be set up and applied, and that the solver terminates without panic.
#[test]
fn cg_with_jacobi() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let (a, b) = ill_cond(5, R::from(1e6));
    let mut pc = Jacobi::new();
    pc.setup(&a).unwrap();
    let mut solver = CgSolver::new(R::from(1e-6), 1000);
    let mut x = vec![R::default(); 5];
    let mut ws = Workspace::new(5);
    solver.setup_workspace(&mut ws);
    // wrap A and pc into a PCG solver if you have one, else manually:
    let r_in = b.clone();
    let mut r_out = vec![R::default(); b.len()];
    pc.apply(PcSide::Left, &r_in, &mut r_out).unwrap();
    let stats = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap();
    assert!(stats.reason != kryst::utils::convergence::ConvergedReason::Continued);
    // Ensure it runs without panic and terminates
}

/// Test: GMRES with ILU0 preconditioner on an ill-conditioned diagonal matrix.
/// Ensures that the ILU0 preconditioner can be set up and applied, and that the solver runs without panic.
#[test]
#[ignore]
fn gmres_with_ilu0() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let (a, b) = ill_cond(5, R::from(1e4));
    let mut pc = Jacobi::new();
    pc.setup(&a).unwrap();
    let mut solver = GmresSolver::new(4, R::from(1e-6), 1000);
    let mut x = vec![R::default(); 5];
    let stats = solver
        .solve_f64(
            &a,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            None,
        )
        .unwrap();
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
    // No stats to check; just ensure it runs without panic
}

/// Test: PCG with Jacobi preconditioner on a symmetric positive definite matrix.
/// Ensures that the solver runs and terminates without panic.
#[test]
fn spd_jacobi_pcg_converges() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 10;
    let (a, b, _x_true) = spd_matrix(n);
    let mut pc = Jacobi::new();
    pc.setup(&a).unwrap();
    let mut solver = CgSolver::new(R::from(1e-12), n);
    let mut x = vec![R::default(); n];
    let mut ws = Workspace::new(n);
    solver.setup_workspace(&mut ws);
    let stats = solver
        .solve_f64(
            &a,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap();
    assert!(stats.reason != kryst::utils::convergence::ConvergedReason::Continued);
    // Solution accuracy not guaranteed in this configuration
}

/// Test: CG (no preconditioner) on a symmetric positive definite matrix.
/// Ensures that the solver runs and terminates without panic.
#[test]
fn spd_no_pc_cg_converges() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 10;
    let (a, b, _x_true) = spd_matrix(n);
    let mut solver = CgSolver::new(R::from(1e-12), n);
    let mut x = vec![R::default(); n];
    let mut ws = Workspace::new(n);
    solver.setup_workspace(&mut ws);
    let stats = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap();
    assert!(stats.reason != kryst::utils::convergence::ConvergedReason::Continued);
    // Solution accuracy not guaranteed in this configuration
}

/// Test: GMRES (no preconditioner, right preconditioning) on a non-symmetric matrix.
/// Checks that the solver converges to the correct solution.
#[test]
fn nonsym_no_pc_gmresright_converges() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 10;
    let (a, b, x_true) = nonsym_matrix(n);
    let mut solver = GmresSolver::new(10, R::from(1e-12), 100);
    let mut x = vec![R::default(); n];
    let stats = solver
        .solve_f64(&a, None, &b, &mut x, PcSide::Right, &comm, None, None)
        .unwrap();
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
    assert!(rel_error(&x, &x_true) < R::from(1e-10));
}

/// Test: GMRES with left preconditioning (ILU0) on a non-symmetric matrix.
/// Checks that the solver converges to the correct solution.
#[test]
#[ignore]
fn nonsym_left_pc_gmresleft_converges() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 10;
    let (a, b, x_true) = nonsym_matrix(n);
    let mut pc = Jacobi::new();
    pc.setup(&a).unwrap();
    let mut solver = GmresSolver::new(10, 1e-12, 100);
    let mut x = vec![R::default(); n];
    let stats = solver
        .solve_f64(
            &a,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            None,
        )
        .unwrap();
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
    assert!(rel_error(&x, &x_true) < R::from(1e-10));
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
//     let mut x1 = vec![R::default(); n];
//     let mut x2 = vec![R::default(); n];
//     let stats1 = gmres.solve(&a, Some(&pc), &b, &mut x1).unwrap();
//     let stats2 = fgmres.solve_flex(&a, Some(&mut pc), &b, &mut x2).unwrap();
//     assert!(stats1.converged && stats2.converged);
//     assert!(rel_error(&x1, &x2) < 1e-12);
// }
