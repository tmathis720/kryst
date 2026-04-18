#![cfg(not(feature = "complex"))]

use crate::context::ksp_context::Workspace;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::kpc::KPreconditioner;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::bicgstab::BiCgStabSolver;
use crate::solver::fgmres::{FgmresSolver, FgmresVariant};
use crate::solver::gmres::GmresSolver;
use crate::utils::convergence::{AcceptanceStatus, ConvergedReason};
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ScriptedScaleOp {
    scales: Vec<f64>,
    calls: AtomicUsize,
}

struct ScriptedMat2Op {
    mats: Vec<[[f64; 2]; 2]>,
    calls: AtomicUsize,
}

impl ScriptedMat2Op {
    fn new(mats: Vec<[[f64; 2]; 2]>) -> Self {
        Self {
            mats,
            calls: AtomicUsize::new(0),
        }
    }

    fn next_mat(&self) -> [[f64; 2]; 2] {
        let idx = self.calls.fetch_add(1, Ordering::Relaxed);
        self.mats
            .get(idx)
            .copied()
            .unwrap_or_else(|| *self.mats.last().unwrap_or(&[[1.0, 0.0], [0.0, 1.0]]))
    }
}

impl LinOpF64 for ScriptedMat2Op {
    fn dims(&self) -> (usize, usize) {
        (2, 2)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let a = self.next_mat();
        y[0] = a[0][0] * x[0] + a[0][1] * x[1];
        y[1] = a[1][0] * x[0] + a[1][1] * x[1];
    }
}

impl LinOp for ScriptedMat2Op {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (2, 2)
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        <Self as LinOpF64>::matvec(self, x, y);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
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

    assert_eq!(stats.reason, ConvergedReason::ConvergedHappyBreakdown);
    assert_eq!(stats.acceptance_status, AcceptanceStatus::OkWithWarning);
    assert_eq!(
        stats.breakdown_reason,
        Some(ConvergedReason::DivergedBreakdownBiCG)
    );
    assert!(
        stats
            .residual_override_note
            .as_deref()
            .unwrap_or_default()
            .contains("BiCGStab breakdown")
    );
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

#[test]
fn bicgstab_omega_breakdown_reclassified_when_true_residual_meets_tol() {
    let op = ScriptedMat2Op::new(vec![
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, -1.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ]);
    let b = vec![1.0, 0.0];
    let mut x = vec![1.0, 0.0];
    let mut solver = BiCgStabSolver::new(1e-12, 5);
    solver.atol = 1e-12;
    solver.set_variant(crate::solver::bicgstab::BiCgStabVariant::FewerChecks);
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

    assert_eq!(stats.reason, ConvergedReason::ConvergedHappyBreakdown);
    assert_eq!(stats.acceptance_status, AcceptanceStatus::OkWithWarning);
    assert_eq!(
        stats.breakdown_reason,
        Some(ConvergedReason::DivergedBreakdownBiCG)
    );
    assert!(
        stats
            .residual_override_note
            .as_deref()
            .unwrap_or_default()
            .contains("omega near-zero/non-finite")
    );
    assert!(stats.final_residual <= solver.atol);
}

#[test]
fn bicgstab_omega_breakdown_keeps_hard_breakdown_above_tol() {
    let op = ScriptedMat2Op::new(vec![
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, -1.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ]);
    let b = vec![1.0, 0.0];
    let mut x = vec![1.0, 0.0];
    let mut solver = BiCgStabSolver::new(1e-12, 5);
    solver.atol = 1e-12;
    solver.set_variant(crate::solver::bicgstab::BiCgStabVariant::FewerChecks);
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

    assert_eq!(stats.reason, ConvergedReason::DivergedBreakdownBiCG);
    assert_eq!(stats.acceptance_status, AcceptanceStatus::Failed);
    assert_eq!(stats.breakdown_reason, None);
    assert!(stats.residual_override_note.is_none());
    assert!(stats.final_residual > solver.atol);
}

#[test]
fn fgmres_near_zero_subdiag_happy_breakdown_enabled() {
    let op = ScriptedScaleOp::new(vec![1.0]);
    let b = vec![1.0];
    let mut x = vec![0.0];
    let mut solver = FgmresSolver::new(1e-12, 6, 2);
    solver.atol = 0.0;
    solver.haptol = 1e-14;
    solver.set_variant(FgmresVariant::Classical);
    solver.set_happy_breakdown(true);
    let restarts = Arc::new(AtomicUsize::new(0));
    let restarts_clone = Arc::clone(&restarts);
    solver.on_restart = Some(Box::new(move |_, _| {
        restarts_clone.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }));
    let comm = UniverseComm::NoComm(NoComm);
    let mut ws = Workspace::default();

    let stats = solver
        .solve_f64(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect("fgmres should return stats");

    assert_eq!(stats.reason, ConvergedReason::ConvergedHappyBreakdown);
    assert!(stats.final_residual <= solver.rtol * 1.0);
    assert_eq!(stats.iterations, 1);
    assert_eq!(restarts.load(Ordering::Relaxed), 0);
}

#[test]
fn fgmres_near_zero_subdiag_happy_breakdown_disabled() {
    let op = ScriptedScaleOp::new(vec![1.0]);
    let b = vec![1.0];
    let mut x = vec![0.0];
    let mut solver = FgmresSolver::new(1e-12, 6, 2);
    solver.atol = 0.0;
    solver.haptol = 1e-14;
    solver.set_variant(FgmresVariant::Classical);
    solver.set_happy_breakdown(false);
    let restarts = Arc::new(AtomicUsize::new(0));
    let restarts_clone = Arc::clone(&restarts);
    solver.on_restart = Some(Box::new(move |_, _| {
        restarts_clone.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }));
    let comm = UniverseComm::NoComm(NoComm);
    let mut ws = Workspace::default();

    let stats = solver
        .solve_f64(
            &op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .expect("fgmres should return stats");

    assert_ne!(stats.reason, ConvergedReason::ConvergedHappyBreakdown);
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));
    assert!(stats.final_residual <= solver.rtol * 1.0);
    assert_eq!(stats.iterations, 1);
    assert_eq!(restarts.load(Ordering::Relaxed), 0);
}
