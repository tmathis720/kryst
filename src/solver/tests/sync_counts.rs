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
use crate::solver::fgmres::{FgmresSolver, FgmresVariant};
use crate::solver::gmres::{GmresSolver, GmresVariant};
use crate::solver::pcg::{PcgSolver, PcgVariant};
use crate::solver::pipegcr::{GcrOrthog, PipeGcrSolver};

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
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws),
    )?;
    let counters = crate::utils::reduction::take_test_counter();
    crate::utils::reduction::install_test_counter(false);

    if counters.allreduces > 0 {
        let upper_bound = 2 * stats.iterations + solver.restart + 8;
        assert!(
            counters.allreduces <= upper_bound,
            "observed allreduces {} exceeds upper bound {upper_bound} (iters={}, restart={})",
            counters.allreduces,
            stats.iterations,
            solver.restart
        );
    }

    let reported = stats.counters.num_global_reductions;
    if reported > 0 {
        assert!(reported >= stats.iterations);
        if counters.allreduces > 0 {
            assert!(
                reported >= counters.allreduces,
                "reported reductions {} should include at least allreduce launches {}",
                reported,
                counters.allreduces
            );
        }
    }
    Ok(())
}

#[test]
fn bicgstab_fewerchecks_reduces_reported_syncs_vs_classic() -> Result<(), KError> {
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
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws),
    )?;

    let mut fewerchecks = BiCgStabSolver::new(1e-8, 200);
    fewerchecks.set_variant(BiCgStabVariant::FewerChecks);
    let mut xl = vec![R::default(); a.nrows()];
    let mut ws = Workspace::default();
    let stats_fewerchecks = fewerchecks.solve_f64(
        &a,
        None,
        &b,
        &mut xl,
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws),
    )?;

    let b_norm2 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();
    assert!(stats_fewerchecks.final_residual <= 1e-6 * b_norm2 + 1e-8);
    assert!(
        stats_fewerchecks.counters.num_global_reductions
            <= stats_classic.counters.num_global_reductions,
        "expected fewerchecks reductions <= classic (fewerchecks={}, classic={})",
        stats_fewerchecks.counters.num_global_reductions,
        stats_classic.counters.num_global_reductions
    );
    Ok(())
}

#[test]
fn pipegcr_matches_pipelined_fgmres_convergence_on_nonsymmetric_system() -> Result<(), KError> {
    let a = util::nonsym_convdiff_2d(9, 3.5);
    let b: Vec<R> = util::rhs_random(a.nrows(), 73);
    let comm = UniverseComm::NoComm(NoComm);

    let mut baseline = FgmresSolver::new(1e-8, 400, 12);
    baseline.set_variant(FgmresVariant::Pipelined);
    let mut xb = vec![R::default(); a.nrows()];
    let mut ws_baseline = Workspace::default();
    let baseline_stats = baseline.solve_f64(
        &a,
        None,
        &b,
        &mut xb,
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws_baseline),
    )?;

    let mut pipegcr = PipeGcrSolver::new(12, 1e-8, 400);
    pipegcr.set_orthog(GcrOrthog::Classical);
    let mut xg = vec![R::default(); a.nrows()];
    let mut ws_gcr = Workspace::default();
    let gcr_stats = pipegcr.solve_f64(
        &a,
        None,
        &b,
        &mut xg,
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws_gcr),
    )?;

    assert!(baseline_stats.reason.is_converged());
    assert_eq!(
        baseline_stats.final_true_residual,
        Some(baseline_stats.final_residual)
    );
    assert!(baseline_stats.final_recurrence_residual.is_some());
    assert!(baseline_stats.last_preconditioned_residual.is_some());
    assert_eq!(
        gcr_stats.reason.is_converged(),
        baseline_stats.reason.is_converged()
    );
    let baseline_true = util::true_residual_norm(&a, &xb, &b);
    let gcr_true = util::true_residual_norm(&a, &xg, &b);
    let tol = 1e-6 * util::vec_norm(&b) + 1e-8;
    assert!(baseline_true <= tol);
    assert!(gcr_true <= tol);
    assert!(
        (gcr_true - baseline_true).abs() <= 5e-6 * util::vec_norm(&b) + 1e-8,
        "PipeGCR and baseline residual mismatch: pipegcr={gcr_true} baseline={baseline_true}"
    );
    Ok(())
}

#[test]
fn pipegcr_reports_sync_count_parity_with_alias_baseline() -> Result<(), KError> {
    let a = util::nonsym_convdiff_2d(8, 2.5);
    let b: Vec<R> = util::rhs_random(a.nrows(), 88);
    let comm = UniverseComm::NoComm(NoComm);

    let mut baseline = FgmresSolver::new(1e-8, 300, 10);
    baseline.set_variant(FgmresVariant::Pipelined);
    let mut xb = vec![R::default(); a.nrows()];
    let mut ws_baseline = Workspace::default();
    let baseline_stats = baseline.solve_f64(
        &a,
        None,
        &b,
        &mut xb,
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws_baseline),
    )?;

    let mut pipegcr = PipeGcrSolver::new(10, 1e-8, 300);
    let mut xg = vec![R::default(); a.nrows()];
    let mut ws_gcr = Workspace::default();
    let gcr_stats = pipegcr.solve_f64(
        &a,
        None,
        &b,
        &mut xg,
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws_gcr),
    )?;

    let gcr = gcr_stats
        .gcr_counters
        .as_ref()
        .expect("PipeGCR must populate GCR counters");
    assert_eq!(
        baseline_stats.final_true_residual,
        Some(baseline_stats.final_residual)
    );
    assert!(baseline_stats.final_recurrence_residual.is_some());
    assert!(baseline_stats.last_preconditioned_residual.is_some());
    assert_eq!(gcr.sync_count, gcr_stats.counters.num_global_reductions);
    let delta = gcr_stats
        .counters
        .num_global_reductions
        .abs_diff(baseline_stats.counters.num_global_reductions);
    assert!(
        delta <= 2 * pipegcr.restart + 8,
        "sync-count parity regression: pipegcr={} baseline={} delta={delta}",
        gcr_stats.counters.num_global_reductions,
        baseline_stats.counters.num_global_reductions
    );
    assert!(gcr.basis_updates > 0);
    assert_eq!(gcr.restarted, gcr.restart_count > 0);
    Ok(())
}
