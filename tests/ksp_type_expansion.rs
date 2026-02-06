#![cfg(not(feature = "complex"))]

use std::any::Any;
use std::sync::Arc;

use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::{LinOp, StructureId, ValuesId};
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
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

#[test]
fn parses_new_solver_types() {
    assert_eq!(
        "richardson".parse::<SolverType>().unwrap(),
        SolverType::Richardson
    );
    assert_eq!(
        "chebyshev".parse::<SolverType>().unwrap(),
        SolverType::Chebyshev
    );
    assert_eq!("cr".parse::<SolverType>().unwrap(), SolverType::Cr);
    assert_eq!("tcqmr".parse::<SolverType>().unwrap(), SolverType::Tcqmr);
    assert_eq!("gcr".parse::<SolverType>().unwrap(), SolverType::Gcr);
}

#[test]
fn left_side_constraint_for_chebyshev_and_cr() {
    let mut ksp = KspContext::new();
    ksp.try_set_pc_side(PcSide::Right).unwrap();
    assert!(ksp.set_type(SolverType::Chebyshev).is_err());

    let mut ksp2 = KspContext::new();
    ksp2.try_set_pc_side(PcSide::Right).unwrap();
    assert!(ksp2.set_type(SolverType::Cr).is_err());
}

#[test]
fn richardson_converges_and_reason_is_converged() {
    let op = Arc::new(TinyDiagOp { diag: [2.0, 4.0] });
    let b = vec![2.0, 8.0];
    let mut x = vec![0.0; 2];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Richardson).unwrap();
    ksp.set_operators(op, None);
    ksp.set_from_options(&kryst::config::options::KspOptions {
        richardson_omega: Some(0.25),
        ..Default::default()
    })
    .unwrap();
    ksp.setup().unwrap();
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedAtol | ConvergedReason::ConvergedRtol
    ));
}

#[test]
fn option_guards_enforced() {
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    let err = ksp
        .set_from_options(&kryst::config::options::KspOptions {
            chebyshev_omega: Some(0.8),
            ..Default::default()
        })
        .unwrap_err();
    assert!(format!("{err}").contains("ksp_chebyshev_omega"));
}
