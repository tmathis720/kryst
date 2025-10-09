#![cfg(feature = "complex")]

mod support;

use approx::assert_abs_diff_eq;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::MinresSolver;
use kryst::utils::convergence::ConvergedReason;
use support::complex_dense::hermitian_pos_def_system;

const SEED: u64 = 0x51D3_DEAD_u64;
const SHIFT: f64 = 1.5;

#[test]
fn minres_solves_random_hpd_system() {
    let n = 7;
    let (op, x_true, b) = hermitian_pos_def_system(n, SEED, SHIFT);
    let comm = UniverseComm::NoComm(NoComm);
    let mut solver = MinresSolver::new(1e-10, 250);
    let mut work = Workspace::new(n);
    let mut x = vec![S::zero(); n];

    let stats = solver
        .solve_k(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut work),
        )
        .expect("MINRES solve");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let err = op.residual_norm(&x, &b);
    assert!(err < 1e-9, "MINRES residual too large: {err:e}");

    for (approx, exact) in x.iter().zip(x_true.iter()) {
        assert_abs_diff_eq!(approx.real(), exact.real(), epsilon = 1e-7);
        assert_abs_diff_eq!(approx.imag(), exact.imag(), epsilon = 1e-7);
    }
}
