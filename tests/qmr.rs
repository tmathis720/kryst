use faer::Mat;
use kryst::matrix::op::LinOp;
use kryst::parallel::UniverseComm;
use kryst::solver::{LinearSolver, QmrSolver};
use std::sync::Arc;

#[test]
fn qmr_solves_simple_nonsymmetric() {
    // A = [[2,1],[0,3]]; b = A * [1,2] = [4,6]
    let mut a_mat = Mat::<f64>::zeros(2, 2);
    a_mat[(0, 0)] = 2.0;
    a_mat[(0, 1)] = 1.0;
    a_mat[(1, 0)] = 0.0;
    a_mat[(1, 1)] = 3.0;
    let a: Arc<dyn LinOp<S = f64>> = Arc::new(a_mat);
    let b = [4.0, 6.0];
    let mut x = [0.0, 0.0];
    let mut solver = QmrSolver::new(1e-12, 100);
    let stats = solver
        .solve(
            a.as_ref(),
            None,
            &b,
            &mut x,
            &UniverseComm::NoComm(kryst::parallel::NoComm),
            None,
            None,
        )
        .expect("solve");
    assert!(
        (x[0] - 1.0).abs() < 1e-8 && (x[1] - 2.0).abs() < 1e-8,
        "x = {:?}",
        x
    );
    assert!(stats.final_residual <= 1e-10);
}
