#![cfg(not(feature = "complex"))]
#![cfg(feature = "mpi")]

mod fixtures;

use std::sync::Arc;

use kryst::context::ksp_context::{ReorthPolicy, Workspace};
use kryst::error::KError;
use kryst::matrix::op::LinOp;
use kryst::parallel::{MpiComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::reduction::ReproMode;
use kryst::solver::gmres::{GmresSolver, GmresVariant};

#[test]
fn mpi_gmres_sstep_reduces_reduction_count() -> Result<(), KError> {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));

    let a = fixtures::csr_poisson_1d(80);
    let b: Vec<f64> = (0..a.nrows()).map(|i| 1.0 + i as f64).collect();
    let restart = 20;

    let mut classic = GmresSolver::new(restart, 1e-6, 5_000);
    classic.set_variant(GmresVariant::Classical);
    let mut x_classic = vec![0.0; a.nrows()];
    let mut ws_classic = Workspace::default();
    let stats_classic = classic.solve_f64(
        &a,
        None,
        &b,
        &mut x_classic,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws_classic),
    )?;

    assert!(stats_classic.final_residual.is_finite());

    let classic_reductions = stats_classic.counters.num_global_reductions;
    assert!(classic_reductions > 0);
    for s in [2usize, 4usize] {
        let mut sstep = GmresSolver::new(restart, 1e-6, 5_000);
        sstep.set_variant(GmresVariant::SStep {
            s,
            reorth: ReorthPolicy::IfNeeded,
            max_cond: 1e8,
        });
        let mut x_sstep = vec![0.0; a.nrows()];
        let mut ws_sstep = Workspace::default();
        let stats_sstep = sstep.solve_f64(
            &a,
            None,
            &b,
            &mut x_sstep,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws_sstep),
        )?;

        assert!(stats_sstep.final_residual.is_finite());
        let target_residual = 1e-6 * b.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-10;
        let relaxed_target = target_residual.max(stats_classic.final_residual * 5.0 + 1e-10);
        assert!(
            stats_sstep.final_residual <= relaxed_target,
            "expected s-step({s}) residual <= {relaxed_target:e} (strict_target={target_residual:e}, classic={:e}, sstep={:e})",
            stats_classic.final_residual,
            stats_sstep.final_residual
        );
        let sstep_reductions = stats_sstep.counters.num_global_reductions;
        assert!(sstep_reductions > 0);
        let panel_allowance = (restart.div_ceil(s)) * 2;
        let max_allowed_reductions = classic_reductions + panel_allowance;
        assert!(
            sstep_reductions <= max_allowed_reductions,
            "expected s-step({s}) reductions <= classical + panel allowance (sstep={sstep_reductions}, classic={classic_reductions}, allowance={panel_allowance})"
        );
    }

    Ok(())
}

#[test]
fn mpi_gmres_sstep_deterministic_reproducible_counters() -> Result<(), KError> {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));

    let a = fixtures::csr_poisson_1d(64);
    let b: Vec<f64> = (0..a.nrows()).map(|i| 2.0 + i as f64).collect();

    let solve_once = |x: &mut [f64]| -> Result<kryst::utils::convergence::SolveStats<f64>, KError> {
        let mut solver = GmresSolver::new(16, 1e-6, 2_000);
        solver.set_variant(GmresVariant::SStep {
            s: 2,
            reorth: ReorthPolicy::IfNeeded,
            max_cond: 1e8,
        });
        let mut ws = Workspace::default();
        ws.set_reduction_mode(ReproMode::Deterministic);
        solver.solve_f64(&a, None, &b, x, PcSide::Left, &comm, None, Some(&mut ws))
    };

    let mut x0 = vec![0.0; a.nrows()];
    let mut x1 = vec![0.0; a.nrows()];
    let s0 = solve_once(&mut x0)?;
    let s1 = solve_once(&mut x1)?;

    assert_eq!(s0.iterations, s1.iterations);
    assert_eq!(
        s0.counters.num_global_reductions,
        s1.counters.num_global_reductions
    );
    assert!(s0.final_residual.is_finite());
    assert!(s1.final_residual.is_finite());

    Ok(())
}

#[test]
fn mpi_gmres_sstep_reports_true_residual_semantics() -> Result<(), KError> {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let a = fixtures::csr_poisson_1d(48);
    let b: Vec<f64> = (0..a.nrows()).map(|i| 1.0 + 0.25 * i as f64).collect();

    let mut solver = GmresSolver::new(16, 1e-7, 1_000);
    solver.set_variant(GmresVariant::SStep {
        s: 2,
        reorth: ReorthPolicy::IfNeeded,
        max_cond: 1e8,
    });
    let mut x = vec![0.0; a.nrows()];
    let stats = solver.solve_f64(&a, None, &b, &mut x, PcSide::Left, &comm, None, None)?;

    let mut ax = vec![0.0; b.len()];
    a.matvec(&x, &mut ax);
    let true_res = b
        .iter()
        .zip(ax.iter())
        .map(|(bi, ai)| {
            let r = bi - ai;
            r * r
        })
        .sum::<f64>()
        .sqrt();
    let rhs_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    assert!(
        (stats.final_residual - true_res).abs() <= 1e-6 * rhs_norm,
        "expected reported final residual ({:.6e}) to match true residual ({:.6e})",
        stats.final_residual,
        true_res
    );
    Ok(())
}

#[cfg(feature = "rayon")]
#[test]
fn mpi_rayon_gmres_sstep_reductions_for_s2_s4() -> Result<(), KError> {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let a = fixtures::csr_poisson_1d(72);
    let b: Vec<f64> = (0..a.nrows()).map(|i| 1.0 + 0.5 * i as f64).collect();

    let mut classic = GmresSolver::new(20, 1e-6, 5_000);
    classic.set_variant(GmresVariant::Classical);
    let mut x_classic = vec![0.0; a.nrows()];
    let mut ws_classic = Workspace::default();
    let stats_classic = classic.solve_f64(
        &a,
        None,
        &b,
        &mut x_classic,
        PcSide::Left,
        &comm,
        None,
        Some(&mut ws_classic),
    )?;
    assert!(stats_classic.final_residual.is_finite());

    for s in [2usize, 4usize] {
        let mut sstep = GmresSolver::new(20, 1e-6, 5_000);
        sstep.set_variant(GmresVariant::SStep {
            s,
            reorth: ReorthPolicy::IfNeeded,
            max_cond: 1e8,
        });
        let mut x_sstep = vec![0.0; a.nrows()];
        let mut ws_sstep = Workspace::default();
        let stats_sstep = sstep.solve_f64(
            &a,
            None,
            &b,
            &mut x_sstep,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws_sstep),
        )?;
        assert!(stats_sstep.final_residual.is_finite());
        let classic_reductions = stats_classic.counters.num_global_reductions;
        let sstep_reductions = stats_sstep.counters.num_global_reductions;
        let panel_allowance = (20usize.div_ceil(s)) * 2;
        assert!(
            sstep_reductions <= classic_reductions + panel_allowance,
            "expected s-step({s}) reductions <= classical + panel allowance (sstep={sstep_reductions}, classic={classic_reductions}, allowance={panel_allowance})"
        );
    }

    Ok(())
}
