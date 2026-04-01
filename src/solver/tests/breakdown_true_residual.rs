#![cfg(not(feature = "complex"))]

use crate::context::ksp_context::Workspace;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::kpc::KPreconditioner;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::bicgstab::BiCgStabSolver;
use crate::solver::gmres::GmresSolver;
use crate::utils::convergence::ConvergedReason;
use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ScriptedScaleOp {
    scales: Vec<f64>,
    calls: AtomicUsize,
}

impl ScriptedScaleOp {
    fn new(scales: Vec<f64>) -> Self {
        Self {
            scales,
            calls: AtomicUsize::new(0),
        }
    }
}

impl LinOpF64 for ScriptedScaleOp {
    fn dims(&self) -> (usize, usize) {
        (1, 1)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let idx = self.calls.fetch_add(1, Ordering::Relaxed);
        let scale = self
            .scales
            .get(idx)
            .copied()
            .unwrap_or_else(|| *self.scales.last().unwrap_or(&1.0));
        y[0] = scale * x[0];
    }
}

impl LinOp for ScriptedScaleOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (1, 1)
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        <Self as LinOpF64>::matvec(self, x, y);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn bicgstab_breakdown_reclassified_when_true_residual_meets_tol() {
    let op = ScriptedScaleOp::new(vec![0.0, 0.0, 1.0]);
    let b = vec![1.0];
    let mut x = vec![1.0];
    let mut solver = BiCgStabSolver::new(1e-12, 5);
    solver.atol = 1e-12;
    let comm = UniverseComm::NoComm(NoComm);
    let mut ws = Workspace::default();

    let stats = solver
        .solve(
            &op,
            None::<&dyn KPreconditioner<Scalar = f64>>,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect("bicgstab should return stats");

    assert_eq!(stats.reason, ConvergedReason::ConvergedAtol);
    assert!(stats.final_residual <= solver.atol);
    assert_eq!(stats.iterations, 0);
}

#[test]
fn gmres_breakdown_mismatch_reclassified_when_true_residual_meets_tol() {
    let op = ScriptedScaleOp::new(vec![2.0, 1.0]);
    let b = vec![1.0];
    let mut x = vec![0.0];
    let mut solver = GmresSolver::new(1, 0.6, 4);
    solver.conv.atol = 0.0;
    let comm = UniverseComm::NoComm(NoComm);
    let mut ws = Workspace::default();

    let stats = solver
        .solve(
            &op,
            None::<&dyn KPreconditioner<Scalar = f64>>,
            &b,
            &mut x,
            PcSide::Right,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect("gmres should return stats");

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));
    assert!(stats.final_residual <= solver.conv.rtol * 1.0);
}
