use std::sync::Arc;

use crate::matrix::op::DenseOp;
use crate::parallel::{NoComm, UniverseComm};
use crate::preconditioner::PcSide;
use crate::solver::tests::util;
use crate::solver::{IdrsBuilder, IdrsSolver, LinearSolver};
use faer::Mat;

#[test]
fn idrs_solves_nonsymmetric_system() {
    let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => 4.0,
        (0, 1) => -1.0,
        (1, 0) => 2.0,
        (1, 1) => 3.0,
        (1, 2) => -1.0,
        (2, 1) => -2.0,
        (2, 2) => 5.0,
        _ => 0.0,
    });
    let op = DenseOp::new(Arc::new(a));
    let b = vec![3.0, 7.0, 4.0];
    let mut x = vec![0.0; 3];

    let mut solver: IdrsSolver = IdrsBuilder::new().s(2).tol(1e-10).maxit(200).build();

    let comm = UniverseComm::NoComm(NoComm);
    let stats = solver
        .solve(&op, None, &b, &mut x, PcSide::Left, &comm, None, None)
        .expect("IDR(s) solve should succeed");

    let true_res = util::true_residual_norm(&op, &x, &b);
    assert!(true_res < 1e-8, "true residual too large: {}", true_res);
    assert!(stats.final_residual < 1e-8, "final residual too large");
}
