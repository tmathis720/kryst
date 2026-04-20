#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use kryst::context::ksp_context::Workspace;
use kryst::error::KError;
use kryst::matrix::op::LinOp;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::solver::FgmresSolver;
use kryst::utils::convergence::ConvergedReason;

struct MutableCountPc {
    hits: Arc<AtomicUsize>,
}

struct ScalePc {
    scale: f64,
}

impl Preconditioner for MutableCountPc {
    fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, _side: PcSide, _x: &[f64], _y: &mut [f64]) -> Result<(), KError> {
        panic!("FGMRES should not call immutable apply");
    }
    fn apply_mut(&mut self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        self.hits.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
        Ok(())
    }
}

impl Preconditioner for ScalePc {
    fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        for (dst, &src) in y.iter_mut().zip(x) {
            *dst = self.scale * src;
        }
        Ok(())
    }
    fn apply_mut(&mut self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        self.apply(side, x, y)
    }
}

#[test]
fn fgmres_uses_apply_mut() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut pc = MutableCountPc { hits: hits.clone() };

    let a = faer::Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let mut solver = FgmresSolver::new(1e-6, 10, 2);
    let b = [1.0f64, 2.0];
    let mut x = [0.0f64; 2];
    let mut work = Workspace::new(2);
    let _ = solver
        .solve_f64(
            &a,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Right,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            Some(&mut work),
        )
        .unwrap();

    assert!(
        hits.load(Ordering::Relaxed) > 0,
        "apply_mut was not invoked"
    );
}

#[test]
fn fgmres_solver_rejects_non_right_pc_side() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut pc = MutableCountPc { hits };

    let a = faer::Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let mut solver = FgmresSolver::new(1e-6, 10, 2);
    let b = [1.0f64, 2.0];
    let mut x = [0.0f64; 2];
    let mut work = Workspace::new(2);
    let err = solver
        .solve_f64(
            &a,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            Some(&mut work),
        )
        .expect_err("FGMRES must reject non-right PC side");
    match err {
        KError::InvalidInput(msg) => {
            assert!(msg.contains("FGMRES"));
            assert!(msg.to_lowercase().contains("right preconditioning"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn fgmres_solver_rejects_symmetric_pc_side() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut pc = MutableCountPc { hits };

    let a = faer::Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let mut solver = FgmresSolver::new(1e-6, 10, 2);
    let b = [1.0f64, 2.0];
    let mut x = [0.0f64; 2];
    let mut work = Workspace::new(2);
    let err = solver
        .solve_f64(
            &a,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Symmetric,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            Some(&mut work),
        )
        .expect_err("FGMRES must reject symmetric PC side");
    match err {
        KError::InvalidInput(msg) => {
            assert!(msg.contains("FGMRES"));
            assert!(msg.to_lowercase().contains("right preconditioning"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn fgmres_one_step_convergence_identity() {
    let a = faer::Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let mut solver = FgmresSolver::new(1e-12, 10, 4);
    solver.set_happy_breakdown(false);
    let b = [1.25f64, -0.75];
    let mut x = [0.0f64; 2];
    let mut work = Workspace::new(2);

    let stats = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Right,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            Some(&mut work),
        )
        .expect("FGMRES should converge in one step on identity");

    assert_eq!(stats.iterations, 1);
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));
    assert!((x[0] - b[0]).abs() <= 1e-12);
    assert!((x[1] - b[1]).abs() <= 1e-12);
}

#[test]
fn fgmres_exact_pc_happy_breakdown() {
    let a = faer::Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.0 });
    let mut solver = FgmresSolver::new(1e-12, 10, 4);
    solver.set_happy_breakdown(true);
    let mut pc = ScalePc { scale: 0.5 };
    let b = [2.0f64, 4.0];
    let mut x = [0.0f64; 2];
    let mut work = Workspace::new(2);

    let stats = solver
        .solve_f64(
            &a,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Right,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            Some(&mut work),
        )
        .expect("FGMRES should happy-breakdown with exact right preconditioner");

    assert_eq!(stats.reason, ConvergedReason::ConvergedHappyBreakdown);
    assert_eq!(stats.iterations, 1);
    assert!((x[0] - 1.0).abs() <= 1e-12);
    assert!((x[1] - 2.0).abs() <= 1e-12);
}

#[test]
fn fgmres_reports_singular_reduced_system_on_zero_operator() {
    let a = faer::Mat::<f64>::from_fn(1, 1, |_i, _j| 0.0);
    let mut solver = FgmresSolver::new(1e-12, 4, 2);
    solver.set_happy_breakdown(true);
    let b = [1.0f64];
    let mut x = [0.0f64; 1];
    let mut work = Workspace::new(1);

    let stats = solver
        .solve_f64(
            &a,
            None,
            &b,
            &mut x,
            PcSide::Right,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            Some(&mut work),
        )
        .expect("FGMRES returns stats even on reduced-system singularity");

    assert_eq!(stats.reason, ConvergedReason::DivergedReducedSystemSingular);
}
