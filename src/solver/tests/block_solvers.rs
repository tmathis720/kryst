use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::jacobi::Jacobi;
use crate::solver::LinearSolver;
use crate::solver::block::{
    BlockKrylovOptions, bicgstab::BlockBicgstabSolver, gmres::BlockGmresSolver,
};
use crate::utils::convergence::ConvergedReason;

#[cfg(feature = "backend-faer")]
#[test]
fn block_gmres_converges_on_block_rhs() {
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 2;
    opts.restart_blocks = 4;
    opts.max_iters = 50;
    let mut solver = BlockGmresSolver::new(opts);
    let a = super::util::spd_poisson2d(4);
    let n = a.nrows();
    let mut b: Vec<R> = vec![R::default(); n * 2];
    for col in 0..2 {
        for row in 0..n {
            b[col * n + row] = R::from(1.0 + (col + row) as f64);
        }
    }
    let mut x: Vec<R> = vec![R::default(); n * 2];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    let stats = solver
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
        .unwrap();
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedAtol | ConvergedReason::ConvergedRtol
    ));
}

#[test]
fn block_bicgstab_converges_on_block_rhs() {
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 2;
    opts.max_iters = 100;
    let mut solver = BlockBicgstabSolver::new(opts);
    let a = super::util::spd_poisson2d(4);
    let n = a.nrows();
    let mut b: Vec<R> = vec![R::default(); n * 2];
    for col in 0..2 {
        for row in 0..n {
            b[col * n + row] = R::from(1.0 + (col + row) as f64);
        }
    }
    let mut x: Vec<R> = vec![R::default(); n * 2];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    let stats = solver
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
        .unwrap();
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedAtol | ConvergedReason::ConvergedRtol
    ));
}

#[test]
fn block_bicgstab_supports_left_preconditioned_runs() {
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 2;
    opts.max_iters = 100;
    let mut solver = BlockBicgstabSolver::new(opts);
    let a = super::util::spd_poisson2d(4);
    let n = a.nrows();
    let mut b: Vec<R> = vec![R::default(); n * 2];
    for col in 0..2 {
        for row in 0..n {
            b[col * n + row] = R::from(1.0 + (col + row) as f64);
        }
    }
    let mut x: Vec<R> = vec![R::default(); n * 2];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    let mut pc = Jacobi::new();
    pc.setup(op).unwrap();
    let stats = solver
        .solve(
            op,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap();
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedAtol | ConvergedReason::ConvergedRtol
    ));
}

#[cfg(feature = "backend-faer")]
#[test]
fn block_gmres_supports_left_preconditioned_runs() {
    let mut opts = BlockKrylovOptions::default();
    opts.block_size = 2;
    opts.restart_blocks = 4;
    opts.max_iters = 50;
    let mut solver = BlockGmresSolver::new(opts);
    let a = super::util::spd_poisson2d(4);
    let n = a.nrows();
    let mut b: Vec<R> = vec![R::default(); n * 2];
    for col in 0..2 {
        for row in 0..n {
            b[col * n + row] = R::from(1.0 + (col + row) as f64);
        }
    }
    let mut x: Vec<R> = vec![R::default(); n * 2];
    let mut ws = Workspace::default();
    let comm = UniverseComm::NoComm(NoComm);
    let op: &dyn crate::matrix::op::LinOp<S = f64> = &a;
    let mut pc = Jacobi::new();
    pc.setup(op).unwrap();
    let stats = solver
        .solve(
            op,
            Some(&mut pc),
            &b,
            &mut x,
            PcSide::Left,
            &comm,
            None,
            Some(&mut ws),
        )
        .unwrap();
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedAtol | ConvergedReason::ConvergedRtol
    ));
}
