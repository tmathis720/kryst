use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::matrix::op::LinOp;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use kryst::solver::{LinearSolver, QmrSolver};
use std::sync::Arc;

#[test]
fn qmr_solves_simple_nonsymmetric() {
    // A = [[2,1],[0,3]]; b = A * [1,2] = [4,6]
    let mut a_mat = Mat::<R>::zeros(2, 2);
    a_mat[(0, 0)] = R::from(2.0);
    a_mat[(0, 1)] = R::from(1.0);
    a_mat[(1, 0)] = R::default();
    a_mat[(1, 1)] = R::from(3.0);
    let a: Arc<dyn LinOp<S = f64>> = Arc::new(a_mat);
    let b = [R::from(4.0), R::from(6.0)];
    let mut x = [R::default(); 2];
    let mut solver = QmrSolver::new(1e-12, 100);
    let stats = solver
        .solve(
            a.as_ref(),
            None,
            &b,
            &mut x,
            PcSide::Left,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            None,
        )
        .expect("solve");
    assert!(
        (x[0] - R::from(1.0)).abs() < R::from(1e-8) && (x[1] - R::from(2.0)).abs() < R::from(1e-8),
        "x = {:?}",
        x
    );
    assert!(stats.final_residual <= R::from(1e-10));
}
