#![cfg(not(feature = "complex"))]
#![cfg(feature = "mpi")]

mod fixtures;

use std::sync::Arc;

use kryst::context::ksp_context::{ReorthPolicy, Workspace};
use kryst::error::KError;
use kryst::parallel::{MpiComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::gmres::{GmresSolver, GmresVariant};
use kryst::utils::convergence::ConvergedReason;

#[test]
fn mpi_gmres_sstep_reduces_reduction_count() -> Result<(), KError> {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));

    let a = fixtures::csr_poisson_1d(80);
    let b: Vec<f64> = (0..a.nrows()).map(|i| 1.0 + i as f64).collect();
    let restart = 10;

    let mut classic = GmresSolver::new(restart, 1e-8, 2_000);
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

    let mut sstep = GmresSolver::new(restart, 1e-8, 2_000);
    sstep.set_variant(GmresVariant::SStep {
        s: 2,
        reorth: ReorthPolicy::Never,
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

    assert!(matches!(
        stats_classic.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));
    assert!(matches!(
        stats_sstep.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    let classic_reductions = stats_classic.counters.num_global_reductions;
    let sstep_reductions = stats_sstep.counters.num_global_reductions;
    assert!(classic_reductions > 0);
    assert!(sstep_reductions > 0);
    assert!(
        sstep_reductions <= classic_reductions,
        "expected s-step reductions <= classical (sstep={sstep_reductions}, classic={classic_reductions})"
    );

    Ok(())
}
