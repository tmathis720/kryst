#![cfg(not(feature = "complex"))]

use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::jacobi::Jacobi;
use crate::solver::LinearSolver;
use crate::solver::gmres::GmresSolver;
use crate::solver::pcg::{PcgSolver, PcgVariant};

use super::util;

#[test]
fn pipelined_cg_reports_residual_replacements() -> Result<(), KError> {
    let a = util::rotated_anisotropy_2d(12, 0.3, 1e-2);
    let b = util::rhs_random(a.nrows(), 21);
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    let mut solver = PcgSolver::new(1e-8, 5_000);
    solver.set_variant(PcgVariant::Pipelined { replace_every: 5 });
    let mut ws = Workspace::default();
    let mut pc = Jacobi::new();
    pc.setup(op)?;
    let mut x: Vec<R> = vec![R::default(); a.nrows()];
    let comm = UniverseComm::NoComm(NoComm);
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
    assert!(stats.counters.residual_replacements > 0);
    Ok(())
}

#[test]
fn gmres_reorth_policy_toggle() {
    let mut solver = GmresSolver::new(10, 1e-8, 200);
    solver.set_reorth_policy(crate::context::ksp_context::ReorthPolicy::Always);
    assert!(matches!(
        solver.reorth_policy(),
        crate::context::ksp_context::ReorthPolicy::Always
    ));
    solver.set_reorth_policy(crate::context::ksp_context::ReorthPolicy::Never);
    assert!(matches!(
        solver.reorth_policy(),
        crate::context::ksp_context::ReorthPolicy::Never
    ));
}
