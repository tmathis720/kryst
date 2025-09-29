use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::LinearSolver;
use crate::solver::gmres::{GmresSolver, GmresVariant};

use super::util;

fn solve_with_variant(
    a: &crate::matrix::sparse::CsrMatrix<f64>,
    b: &[f64],
    variant: GmresVariant,
    restart: usize,
) -> Result<(Vec<f64>, usize, f64), KError> {
    let mut solver = GmresSolver::new(restart, 1e-8, 2_000);
    solver.set_variant(variant);
    let mut x = vec![0.0; b.len()];
    let mut ws = Workspace::default();
    let op: &dyn crate::matrix::op::LinOp<S = f64> = a;
    let comm = UniverseComm::NoComm(NoComm);
    let stats = solver.solve(
        op,
        None,
        b,
        &mut x,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;
    let rtrue = util::true_residual_norm(op, &x, b);
    Ok((x, stats.iterations, rtrue))
}

#[test]
fn gmres_pipelined_tracks_classical_convergence() -> Result<(), KError> {
    let a = util::nonsym_convdiff_2d(10, 5.0);
    let b = util::rhs_random(a.nrows(), 7);
    let restart = 20;
    let bnorm = util::vec_norm(&b).max(1e-32);

    let (_x_classic, it_classic, res_classic) =
        solve_with_variant(&a, &b, GmresVariant::Classical, restart)?;
    let (_x_pipe, it_pipe, res_pipe) =
        solve_with_variant(&a, &b, GmresVariant::Pipelined, restart)?;

    assert!(res_classic <= 1e-8 * bnorm + 1e-10);
    assert!(res_pipe <= 1e-8 * bnorm + 1e-10);
    assert!((it_classic as isize - it_pipe as isize).abs() as usize <= restart);
    Ok(())
}

#[test]
fn gmres_sstep_reports_not_implemented() {
    let mut solver = GmresSolver::new(10, 1e-8, 100);
    solver.set_variant(GmresVariant::SStep {
        s: 3,
        reorth: crate::context::ksp_context::ReorthPolicy::IfNeeded,
        max_cond: 1e8,
    });
    let a = util::spd_poisson2d(6);
    let b = vec![0.0; a.nrows()];
    let mut x = vec![0.0; a.nrows()];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    let err = solver
        .solve(
            op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap_err();
    assert!(matches!(err, KError::NotImplemented(_)));
}
