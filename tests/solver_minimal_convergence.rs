#![cfg(not(feature = "complex"))]

use std::any::Any;
use std::sync::Arc;

use kryst::config::options::KspOptions;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::{LinOp, StructureId, ValuesId};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::utils::convergence::ConvergedReason;

#[derive(Clone)]
struct TinyDiagOp {
    diag: [f64; 2],
}

impl LinOp for TinyDiagOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (2, 2)
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        y[0] = self.diag[0] * x[0];
        y[1] = self.diag[1] * x[1];
    }

    fn supports_transpose(&self) -> bool {
        true
    }

    fn t_matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        self.matvec(x, y);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn structure_id(&self) -> StructureId {
        StructureId(1)
    }

    fn values_id(&self) -> ValuesId {
        ValuesId(1)
    }

    fn comm(&self) -> UniverseComm {
        UniverseComm::NoComm(NoComm)
    }
}

fn solve_with_rhs(solver_type: SolverType, opts: KspOptions, b: Vec<f64>) -> ConvergedReason {
    let op = Arc::new(TinyDiagOp { diag: [1.0, 2.0] });
    let mut x = vec![0.0; 2];

    let mut ksp = KspContext::new();
    ksp.set_type(solver_type).unwrap();
    ksp.set_operators(op, None);
    ksp.set_from_options(&opts).unwrap();
    ksp.setup().unwrap();
    let stats = ksp.solve(&b, &mut x).unwrap();
    stats.reason
}

fn solve_with(solver_type: SolverType, opts: KspOptions) -> ConvergedReason {
    solve_with_rhs(solver_type, opts, vec![1.0, 4.0])
}

fn assert_converged(reason: ConvergedReason) {
    assert!(
        matches!(
            reason,
            ConvergedReason::ConvergedAtol | ConvergedReason::ConvergedRtol
        ),
        "unexpected convergence reason: {reason:?}"
    );
}

#[test]
fn chebyshev_converges_minimally() {
    let reason = solve_with(
        SolverType::Chebyshev,
        KspOptions {
            chebyshev_omega: Some(0.5),
            ..Default::default()
        },
    );
    assert_converged(reason);
}

#[test]
fn cr_converges_minimally() {
    let reason = solve_with(SolverType::Cr, KspOptions::default());
    assert_converged(reason);
}

#[test]
fn tcqmr_converges_minimally() {
    let reason = solve_with_rhs(
        SolverType::Tcqmr,
        KspOptions {
            rtol: Some(1e-2),
            ..Default::default()
        },
        vec![0.0, 0.0],
    );
    assert_converged(reason);
}

#[test]
fn gcr_converges_minimally() {
    let reason = solve_with(
        SolverType::Gcr,
        KspOptions {
            gcr_restart: Some(5),
            rtol: Some(1e-8),
            ..Default::default()
        },
    );
    assert_converged(reason);
}
