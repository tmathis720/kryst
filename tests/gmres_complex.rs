#![cfg(feature = "complex")]

mod support;

use approx::assert_abs_diff_eq;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::fgmres::FgmresVariant;
use kryst::solver::gmres::GmresVariant;
use kryst::solver::{FgmresSolver, GmresSolver};
use kryst::utils::convergence::ConvergedReason;
use support::complex_dense::diagonally_dominant_system;

const SEED: u64 = 0xDEC0DED5EED;
const ROW_SCALE: f64 = 0.15;
const DIAG_SHIFT: f64 = 1.0;

#[test]
fn gmres_solves_random_complex_system() {
    let n = 8;
    let (op, x_true, b) = diagonally_dominant_system(n, SEED, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = GmresSolver::new(12, 1e-10, 200);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let stats = solver
        .solve(
            &op,
            None::<&dyn kryst::ops::kpc::KPreconditioner<Scalar = S>>,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("GMRES solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let err = op.residual_norm(&x, &b);
    assert!(err < 1e-9, "GMRES residual too large: {err:e}");

    for (approx, exact) in x.iter().zip(x_true.iter()) {
        assert_abs_diff_eq!(approx.real(), exact.real(), epsilon = 1e-7);
        assert_abs_diff_eq!(approx.imag(), exact.imag(), epsilon = 1e-7);
    }
}

#[test]
fn gmres_pipelined_solves_random_complex_system() {
    let n = 8;
    let (op, _x_true, b) = diagonally_dominant_system(n, SEED, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = GmresSolver::new(12, 1e-10, 200);
    solver.set_variant(GmresVariant::Pipelined);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let stats = solver
        .solve(
            &op,
            None::<&dyn kryst::ops::kpc::KPreconditioner<Scalar = S>>,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("GMRES pipelined solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let err = op.residual_norm(&x, &b);
    assert!(err.is_finite(), "pipelined GMRES residual not finite");
}

#[test]
fn fgmres_solves_random_complex_system() {
    let n = 6;
    let (op, _x_true, b) = diagonally_dominant_system(n, SEED, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = FgmresSolver::new(1e-10, 200, 10);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let stats = solver
        .solve_k(
            &op,
            None::<&mut dyn kryst::ops::kpc::KPreconditioner<Scalar = S>>,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("FGMRES solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let err = op.residual_norm(&x, &b);
    assert!(err < 1e-9, "FGMRES residual too large: {err:e}");
}

#[test]
fn fgmres_pipelined_solves_random_complex_system() {
    let n = 6;
    let (op, _x_true, b) = diagonally_dominant_system(n, SEED, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = FgmresSolver::new(1e-10, 200, 10);
    solver.set_variant(FgmresVariant::Pipelined);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let stats = solver
        .solve_k(
            &op,
            None::<&mut dyn kryst::ops::kpc::KPreconditioner<Scalar = S>>,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("FGMRES pipelined solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let err = op.residual_norm(&x, &b);
    assert!(err.is_finite(), "pipelined FGMRES residual not finite");
}

#[test]
fn gmres_with_sor_preconditioner_complex() {
    use kryst::matrix::sparse::CsrMatrix;
    use kryst::ops::kpc::KPreconditioner;
    use kryst::preconditioner::sor::{MatSorType, SorPc};

    let n = 6;
    let (op, _x_true, b) = diagonally_dominant_system(n, SEED + 7, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = GmresSolver::new(8, 1e-9, 120);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        col_idx.push(i);
        vals.push(S::from_real(2.0));
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(S::from_parts(-0.2, 0.1));
        }
        row_ptr.push(col_idx.len());
    }
    let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals);
    let mut sor = SorPc::new(1.0, 1, MatSorType::APPLY_LOWER, 0.0);
    kryst::preconditioner::Preconditioner::setup(&mut sor, &a).unwrap();

    let stats = solver
        .solve(
            &op,
            Some(&sor as &dyn KPreconditioner<Scalar = S>),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("GMRES+SOR solve");
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));
}
