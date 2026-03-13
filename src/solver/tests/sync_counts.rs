#![cfg(not(feature = "complex"))]

use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::jacobi::Jacobi;
use crate::solver::LinearSolver;
use crate::solver::bicgstab::{BiCgStabSolver, BiCgStabVariant};
use crate::solver::gmres::{GmresSolver, GmresVariant};
use crate::solver::pcg::{PcgSolver, PcgVariant};

use super::util;

#[test]
fn pipelined_cg_uses_single_reduction_per_iteration() -> Result<(), KError> {
    crate::utils::reduction::install_test_counter(true);
    let a = util::spd_poisson2d(10);
    let b: Vec<R> = util::rhs_random(a.nrows(), 5);
    let mut solver = PcgSolver::new(1e-8, 5_000);
    solver.set_variant(PcgVariant::Pipelined { replace_every: 0 });
    let mut ws = Workspace::default();
    let mut pc = Jacobi::new();
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    pc.setup(op)?;
    let comm = UniverseComm::NoComm(NoComm);
    let mut x: Vec<R> = vec![R::default(); a.nrows()];
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
    let expected = 2 * stats.iterations + 2; // initial dot/norm plus per-iteration reductions
    if counters.allreduces > 0 {
        assert!(
            counters.allreduces >= expected,
            "unexpected allreduce count: iters={} allreduces={} expected>={}",
            stats.iterations,
            counters.allreduces,
            expected
        );
        assert!(
            counters.allreduces <= expected + 6,
            "unexpected allreduce count: iters={} allreduces={} expected<={}",
            stats.iterations,
            counters.allreduces,
            expected + 6
        );
    }
    assert!(
        stats.counters.num_global_reductions >= expected,
        "solver-reported reductions {} < expected {}",
        stats.counters.num_global_reductions,
        expected
    );
    Ok(())
}

#[test]
fn gmres_classic_reduction_count_within_expected_bounds() -> Result<(), KError> {
    crate::utils::reduction::install_test_counter(true);
    let a = util::nonsym_convdiff_2d(8, 4.0);
    let b: Vec<R> = util::rhs_random(a.nrows(), 17);
    let mut solver = GmresSolver::new(12, 1e-8, 500);
    solver.set_variant(GmresVariant::Classical);
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let mut x: Vec<R> = vec![R::default(); a.nrows()];
    let stats = solver.solve_f64(
        &a,
        None,
        &b,
        &mut x,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;
    crate::utils::reduction::take_test_counter();
    crate::utils::reduction::install_test_counter(false);
    let reported = stats.counters.num_global_reductions;
    if reported > 0 {
        assert!(reported >= stats.iterations);
        let upper_bound = 2 * stats.iterations + solver.restart + 8;
        assert!(
            reported <= upper_bound,
            "reported reductions {reported} exceeds upper bound {upper_bound} (iters={}, restart={})",
            stats.iterations,
            solver.restart
        );
    }
    Ok(())
}

#[test]
fn bicgstab_lowsync_reduces_reported_syncs_vs_classic() -> Result<(), KError> {
    let a = util::nonsym_convdiff_2d(8, 3.0);
    let b: Vec<R> = util::rhs_random(a.nrows(), 9);
    let comm = UniverseComm::NoComm(NoComm);

    let mut classic = BiCgStabSolver::new(1e-8, 200);
    classic.set_variant(BiCgStabVariant::Classic);
    let mut xc = vec![R::default(); a.nrows()];
    let mut ws = Workspace::default();
    let stats_classic = classic.solve_f64(
        &a,
        None,
        &b,
        &mut xc,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;

    let mut lowsync = BiCgStabSolver::new(1e-8, 200);
    lowsync.set_variant(BiCgStabVariant::LowSync);
    let mut xl = vec![R::default(); a.nrows()];
    let mut ws = Workspace::default();
    let stats_lowsync = lowsync.solve_f64(
        &a,
        None,
        &b,
        &mut xl,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws),
    )?;

    let b_norm2 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();
    assert!(stats_lowsync.final_residual <= 1e-6 * b_norm2 + 1e-8);
    assert!(
        stats_lowsync.counters.num_global_reductions
            <= stats_classic.counters.num_global_reductions,
        "expected lowsync reductions <= classic (lowsync={}, classic={})",
        stats_lowsync.counters.num_global_reductions,
        stats_classic.counters.num_global_reductions
    );
    Ok(())
}
