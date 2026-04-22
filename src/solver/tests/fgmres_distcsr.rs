use super::util;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::dist_csr::{DistributedPlanMetrics, choose_distributed_plan};
use crate::matrix::op::{DistLayout, LinOp};
use crate::matrix::{DistCsrOp, sparse::CsrMatrix};
#[cfg(feature = "rayon")]
use crate::parallel::RayonComm;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::LinearSolver;
use crate::solver::fgmres::{FgmresSolver, FgmresVariant, PipelinePolicy, ResidualCheckPolicy};
use approx::assert_abs_diff_eq;
use std::any::Any;

fn dist_fixture_2x2() -> Result<(DistCsrOp, Vec<f64>, Vec<f64>), KError> {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, 1.0, 2.0, 3.0],
    );
    let x_true = vec![1.0, -2.0];
    let mut b = vec![0.0; 2];
    csr.spmv(&x_true, &mut b);
    let comm = UniverseComm::NoComm(NoComm);
    let op = DistCsrOp::from_local_rows(2, 0, &csr, &[0, 2], comm)?;
    Ok((op, b, x_true))
}

struct HiddenDistOp {
    inner: DistCsrOp,
}

impl LinOp for HiddenDistOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        self.inner.dims()
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        self.inner.matvec(x, y);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn comm(&self) -> UniverseComm {
        self.inner.comm()
    }

    fn dist_layout(&self) -> Option<&DistLayout> {
        self.inner.dist_layout()
    }

    fn format(&self) -> crate::matrix::format::OpFormat {
        self.inner.format()
    }
}

#[test]
fn fgmres_distcsr_and_generic_routes_match_numerics() -> Result<(), KError> {
    let (dist, b, x_true) = dist_fixture_2x2()?;
    let comm = UniverseComm::NoComm(NoComm);

    let mut solver_dist = FgmresSolver::new(1e-12, 64, 8);
    let mut ws_dist = Workspace::new(2);
    solver_dist.setup_workspace(&mut ws_dist);
    let mut x_dist = vec![0.0; 2];
    let stats_dist = solver_dist.solve_f64(
        &dist,
        None,
        &b,
        &mut x_dist,
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws_dist),
    )?;
    assert!(stats_dist.reason.is_converged());

    let (dist_hidden, _, _) = dist_fixture_2x2()?;
    let hidden = HiddenDistOp { inner: dist_hidden };
    let mut solver_generic = FgmresSolver::new(1e-12, 64, 8);
    let mut ws_generic = Workspace::new(2);
    solver_generic.setup_workspace(&mut ws_generic);
    let mut x_generic = vec![0.0; 2];
    let stats_generic = solver_generic.solve_f64(
        &hidden,
        None,
        &b,
        &mut x_generic,
        PcSide::Right,
        &comm,
        None,
        Some(&mut ws_generic),
    )?;
    assert!(stats_generic.reason.is_converged());

    for i in 0..2 {
        assert_abs_diff_eq!(x_dist[i], x_true[i], epsilon = 1e-10);
        assert_abs_diff_eq!(x_generic[i], x_true[i], epsilon = 1e-10);
        assert_abs_diff_eq!(x_dist[i], x_generic[i], epsilon = 1e-12);
    }

    let r_dist = util::true_residual_norm(&dist, &x_dist, &b);
    let r_generic = util::true_residual_norm(&hidden, &x_generic, &b);
    assert!(r_dist <= 1e-10);
    assert_abs_diff_eq!(r_dist, r_generic, epsilon = 1e-12);
    Ok(())
}

#[cfg(feature = "rayon")]
#[test]
fn fgmres_distcsr_route_rejects_noncongruent_comm() {
    let (dist, b, _) = dist_fixture_2x2().expect("fixture");

    let mut solver = FgmresSolver::new(1e-12, 64, 8);
    let mut ws = Workspace::new(2);
    solver.setup_workspace(&mut ws);
    let mut x = vec![0.0; 2];
    let bad_comm = UniverseComm::Rayon(RayonComm::new());

    let err = solver
        .solve_f64(
            &dist,
            None,
            &b,
            &mut x,
            PcSide::Right,
            &bad_comm,
            None,
            Some(&mut ws),
        )
        .expect_err("non-congruent comm should fail in DistCSR route");

    match err {
        KError::InvalidInput(msg) => assert!(msg.contains("DistCSR route")),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn fgmres_distcsr_policy_selector_prefers_low_sync_for_comm_heavy_setup() {
    let solver = FgmresSolver::new(1e-10, 200, 64);
    let heavy_diag = choose_distributed_plan(
        &DistributedPlanMetrics {
            n_local_rows: 20_000,
            local_nnz: 80_000,
            local_diag_nnz: 28_000,
            ghost_nnz: 52_000,
            local_only_rows: 7_500,
            border_rows: 12_500,
            halo_recv_volume: 14_000,
            halo_send_volume: 13_500,
        },
        None,
    );
    let decision = solver.select_distcsr_policy(&heavy_diag, 8);
    assert_eq!(decision.variant, FgmresVariant::Pipelined);
    assert!(matches!(
        decision.pipeline_policy,
        PipelinePolicy::FallbackToClassicalOnStagnation
            | PipelinePolicy::PeriodicResidualReplacement
    ));
    assert_eq!(
        decision.residual_check_policy,
        ResidualCheckPolicy::RestartOnly
    );
    assert!(decision.restart <= solver.restart);
}

#[test]
fn fgmres_distcsr_policy_selector_prefers_classical_for_compute_heavy_setup() {
    let solver = FgmresSolver::new(1e-10, 200, 12);
    let compute_diag = choose_distributed_plan(
        &DistributedPlanMetrics {
            n_local_rows: 20_000,
            local_nnz: 90_000,
            local_diag_nnz: 86_000,
            ghost_nnz: 4_000,
            local_only_rows: 18_500,
            border_rows: 1_500,
            halo_recv_volume: 400,
            halo_send_volume: 450,
        },
        None,
    );
    let decision = solver.select_distcsr_policy(&compute_diag, 8);
    assert_eq!(decision.variant, FgmresVariant::Classical);
    assert_eq!(decision.pipeline_policy, PipelinePolicy::Strict);
    assert!(matches!(
        decision.residual_check_policy,
        ResidualCheckPolicy::OnConvergence | ResidualCheckPolicy::EveryIteration
    ));
    assert!(decision.restart >= solver.restart);
}
