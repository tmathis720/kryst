use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::gmres::{GmresSolver, GmresVariant};

use super::util;

fn solve_with_variant(
    a: &crate::matrix::sparse::CsrMatrix<f64>,
    b: &[R],
    variant: GmresVariant,
    restart: usize,
) -> Result<(Vec<R>, usize, R), KError> {
    let mut solver = GmresSolver::new(restart, 1e-8, 2_000);
    solver.set_variant(variant);
    let mut x: Vec<R> = vec![R::default(); b.len()];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let stats = solver.solve_f64(a, None, b, &mut x, PcSide::Left, &comm, None, Some(&mut ws))?;
    let rtrue = util::true_residual_norm(a, &x, b);
    Ok((x, stats.iterations, rtrue))
}

#[test]
fn gmres_pipelined_tracks_classical_convergence() -> Result<(), KError> {
    let a = util::nonsym_convdiff_2d(10, 5.0);
    let b: Vec<R> = util::rhs_random(a.nrows(), 7);
    let restart = 20;
    let bnorm: R = util::vec_norm(&b).max(R::from(1e-32));

    let (_x_classic, it_classic, res_classic) =
        solve_with_variant(&a, &b, GmresVariant::Classical, restart)?;
    let (_x_pipe, it_pipe, res_pipe) =
        solve_with_variant(&a, &b, GmresVariant::Pipelined, restart)?;

    assert!(res_classic <= R::from(1e-8) * bnorm + R::from(1e-10));
    assert!(res_pipe <= R::from(1e-8) * bnorm + R::from(1e-10));
    assert!((it_classic as isize - it_pipe as isize).abs() as usize <= restart);
    Ok(())
}

#[test]
fn gmres_sstep_converges_on_spd() -> Result<(), KError> {
    let a = util::spd_poisson2d(6);
    let b: Vec<R> = util::rhs_random(a.nrows(), 4);
    let restart = 12;
    let bnorm: R = util::vec_norm(&b).max(R::from(1e-32));

    let (_x_sstep, _it_sstep, res_sstep) = solve_with_variant(
        &a,
        &b,
        GmresVariant::SStep {
            s: 3,
            reorth: crate::context::ksp_context::ReorthPolicy::IfNeeded,
            max_cond: 1e8,
        },
        restart,
    )?;

    assert!(res_sstep <= R::from(1e-8) * bnorm + R::from(1e-10));
    Ok(())
}
