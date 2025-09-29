use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::preconditioner::jacobi::Jacobi;
use crate::solver::LinearSolver;
use crate::solver::gmres::{GmresSolver, GmresVariant};
use crate::solver::pcg::{PcgSolver, PcgVariant};

use super::util;

#[test]
fn pipelined_cg_uses_single_reduction_per_iteration() -> Result<(), KError> {
    crate::utils::reduction::install_test_counter(true);
    let a = util::spd_poisson2d(10);
    let b = util::rhs_random(a.nrows(), 5);
    let mut solver = PcgSolver::new(1e-8, 5_000);
    solver.set_variant(PcgVariant::Pipelined { replace_every: 0 });
    let mut ws = Workspace::default();
    let mut pc = Jacobi::new();
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    pc.setup(op)?;
    let comm = UniverseComm::NoComm(NoComm);
    let mut x = vec![0.0; a.nrows()];
    let stats = solver.solve(
        op,
        Some(&mut pc),
        &b,
        &mut x,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;
    let counters = crate::utils::reduction::take_test_counter();
    crate::utils::reduction::install_test_counter(false);
    assert!((counters.allreduces as isize - stats.iterations as isize).abs() <= 2);
    assert!(counters.allreduces >= stats.iterations.saturating_sub(1));
    assert!(stats.counters.num_global_reductions >= stats.iterations.saturating_sub(1));
    Ok(())
}

#[test]
fn gmres_classic_reduction_count_within_expected_bounds() -> Result<(), KError> {
    crate::utils::reduction::install_test_counter(true);
    let a = util::nonsym_convdiff_2d(8, 4.0);
    let b = util::rhs_random(a.nrows(), 17);
    let mut solver = GmresSolver::new(12, 1e-8, 500);
    solver.set_variant(GmresVariant::Classical);
    let mut ws = Workspace::default();
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    let comm = UniverseComm::NoComm(NoComm);
    let mut x = vec![0.0; a.nrows()];
    let stats = solver.solve(
        op,
        None,
        &b,
        &mut x,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;
    let counters = crate::utils::reduction::take_test_counter();
    crate::utils::reduction::install_test_counter(false);
    assert!(counters.allreduces >= stats.iterations);
    assert!(counters.allreduces <= stats.iterations + solver.restart + 4);
    assert!(stats.counters.num_global_reductions <= counters.allreduces + 2);
    Ok(())
}
