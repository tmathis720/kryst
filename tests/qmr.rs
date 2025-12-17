#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
use kryst::parallel::UniverseComm;
use kryst::preconditioner::PcSide;
use kryst::solver::QmrSolver;

#[test]
fn qmr_solves_simple_nonsymmetric() {
    // A = [[2,1],[0,3]]; b = A * [1,2] = [4,6]
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 2.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 0.0;
    a[(1, 1)] = 3.0;
    let b = [4.0f64, 6.0];
    let mut x = [0.0f64; 2];
    let mut solver = QmrSolver::new(1e-12, 100);
    let stats = solver
        .solve_f64(
            &a,
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
        (x[0] - 1.0).abs() < 1e-8 && (x[1] - 2.0).abs() < 1e-8,
        "x = {:?}",
        x
    );
    assert!(stats.final_residual <= 1e-10);
}
