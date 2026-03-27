#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
//! Tests for iterative solvers (CG, GMRES) vs direct solvers on random matrices.
//!
//! This module verifies that the implemented iterative solvers (Conjugate Gradient and GMRES)
//! produce solutions that closely match those from direct solvers (LU and QR) on small random
//! systems. The tests use random SPD and non-symmetric matrices, and compare the results
//! elementwise within a tight tolerance.

use approx::assert_abs_diff_eq;
use faer::Mat;
use faer::linalg::solvers::SolveCore;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::preconditioner::PcSide;
use kryst::solver::{CgSolver, GmresSolver, LinearSolver};
use rand::{RngExt, rng};

/// Helper function to generate a random symmetric positive definite (SPD) matrix `A` and a random right-hand side `b`.
///
/// The SPD matrix is constructed as `A = Mᵀ M + I`, where `M` is a random matrix and `I` is the identity.
/// This ensures that `A` is symmetric and positive definite, suitable for CG.
fn random_spd(n: usize) -> (faer::Mat<R>, Vec<R>) {
    let mut rng = rng();
    let data: Vec<R> = (0..n * n).map(|_| rng.random()).collect();
    let m = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let m_t = m.transpose();
    let a = &m_t * &m + Mat::<R>::identity(n, n);
    let b: Vec<R> = (0..n).map(|_| rng.random()).collect();
    (a, b)
}

/// Test that the Conjugate Gradient (CG) solver produces a solution matching the direct LU solver on a random SPD system.
///
/// - Generates a random SPD matrix and right-hand side.
/// - Solves with CG and checks for convergence.
/// - Solves with direct LU and compares the solutions elementwise.
#[test]
fn cg_vs_direct_on_spd() {
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 10;
    let (a, b) = random_spd(n);
    let mut x_cg = vec![R::default(); n];
    let mut solver = CgSolver::new(1e-8, 1000);
    let mut ws = Workspace::new(n);
    solver.setup_workspace(&mut ws);
    let stats = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x_cg,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap();
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
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
    let comm = kryst::parallel::UniverseComm::NoComm(kryst::parallel::NoComm);
    let n = 10;
    let mut rng = rng();
    let data: Vec<R> = (0..n * n).map(|_| rng.random()).collect();
    // Shift the diagonal to avoid near-singular random systems that can make
    // convergence flaky for strict stopping criteria.
    let a = Mat::from_fn(n, n, |i, j| {
        let mut v = data[j * n + i];
        if i == j {
            v += n as R;
        }
        v
    });
    let b: Vec<R> = (0..n).map(|_| rng.random()).collect();
    let mut x_gmres = vec![R::default(); n];
    let mut solver = GmresSolver::new(100, 1e-8, 1000);
    let stats = solver
        .solve_f64(&a, None, &b, &mut x_gmres, PcSide::Left, &comm, None, None)
        .unwrap();
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
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
