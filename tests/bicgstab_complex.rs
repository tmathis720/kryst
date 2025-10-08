#![cfg(feature = "complex")]

mod support;

use approx::assert_abs_diff_eq;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::BiCgStabSolver;
use kryst::utils::convergence::ConvergedReason;
use support::complex_dense::diagonally_dominant_system;

const SEED: u64 = 0xB1C6_57AB_u64;
const ROW_SCALE: f64 = 0.2;
const DIAG_SHIFT: f64 = 1.0;

#[test]
fn bicgstab_solves_random_complex_system() {
    let n = 7;
    let (op, x_true, b) = diagonally_dominant_system(n, SEED, ROW_SCALE, DIAG_SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = BiCgStabSolver::new(1e-10, 300);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let stats = solver
        .solve(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("BiCGStab solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let err = op.residual_norm(&x, &b);
    assert!(err < 1e-9, "BiCGStab residual too large: {err:e}");

    for (approx, exact) in x.iter().zip(x_true.iter()) {
        assert_abs_diff_eq!(approx.real(), exact.real(), epsilon = 1e-7);
        assert_abs_diff_eq!(approx.imag(), exact.imag(), epsilon = 1e-7);
    }
}
