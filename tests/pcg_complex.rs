#![cfg(feature = "complex")]

mod support;

use approx::assert_abs_diff_eq;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::Workspace;
use kryst::ops::kpc::KPreconditioner;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::pcg::{PcgSolver, PcgVariant};
use support::complex_dense::hermitian_pos_def_system;

#[test]
fn pcg_solve_k_complex_hpd_classic_and_pipelined() {
    let n = 8;
    let (op, x_true, b) = hermitian_pos_def_system(n, 0x5EED_CAFE, 3.0);
    let comm = UniverseComm::NoComm(NoComm);

    for variant in [
        PcgVariant::Classic,
        PcgVariant::Pipelined { replace_every: 3 },
    ] {
        let mut solver = PcgSolver::new(1e-11, 4 * n).with_variant(variant);
        let mut work = Workspace::new(n);
        let mut x = vec![S::zero(); n];

        let stats = solver
            .solve_k(
                &op,
                None::<&dyn KPreconditioner<Scalar = S>>,
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut work),
            )
            .expect("complex PCG solve_k");

        let residual = op.residual_norm(&x, &b);
        assert!(
            stats.reason.is_converged() || residual < 1e-9,
            "PCG {variant:?} did not converge: reason={:?}, residual={residual:e}",
            stats.reason
        );
        assert!(
            residual < 1e-9,
            "PCG {variant:?} residual too large: {residual:e}"
        );

        for (actual, expected) in x.iter().zip(x_true.iter()) {
            assert_abs_diff_eq!(actual.real(), expected.real(), epsilon = 1e-7);
            assert_abs_diff_eq!(actual.imag(), expected.imag(), epsilon = 1e-7);
        }
    }
}
