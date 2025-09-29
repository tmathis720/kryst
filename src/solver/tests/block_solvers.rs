use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::LinearSolver;
use crate::solver::block::{BlockKrylovOptions, gmres::BlockGmresSolver};

#[test]
fn block_gmres_reports_not_implemented() {
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 4;
    opts.restart_blocks = 2;
    let mut solver = BlockGmresSolver::new(opts);
    let a = super::util::spd_poisson2d(4);
    let b = vec![0.0; a.nrows()];
    let mut x = vec![0.0; a.nrows()];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    let err = solver
        .solve(
            op,
            None,
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap_err();
    assert!(matches!(err, KError::NotImplemented(_)));
}
