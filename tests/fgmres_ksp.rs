use std::sync::Arc;

use faer::Mat;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::preconditioner::PcSide;

#[test]
fn fgmres_solves_dd_nonsym() {
    // Non-symmetric, diagonally-dominant 5x5 matrix
    let data = [
        [10.0, 2.0, 0.0, 0.0, 0.0],
        [3.0, 15.0, 4.0, 0.0, 0.0],
        [0.0, -2.0, 8.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 7.0, 3.0],
        [0.0, 0.0, 0.0, 2.0, 12.0],
    ];
    let mut a = Mat::<f64>::zeros(5, 5);
    for i in 0..5 {
        for j in 0..5 {
            a[(i, j)] = data[i][j];
        }
    }

    let amat: Arc<dyn kryst::matrix::op::LinOp<S = f64>> = Arc::new(a.clone());
    let pmat = amat.clone();

    // x_true = [1,2,3,4,5]; b = A x_true
    let x_true = [1., 2., 3., 4., 5.];
    let mut b = [0.0; 5];
    amat.matvec(&x_true, &mut b);

    let mut x = [0.0; 5];
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Fgmres).unwrap();
    ksp.set_pc_type(PcType::Jacobi, None).unwrap();
    ksp.set_operators(amat.clone(), Some(pmat));
    ksp.set_pc_side(PcSide::Right);

    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(
        stats.final_residual <= 1e-6,
        "res={:.3e}",
        stats.final_residual
    );
    for (xi, xt) in x.iter().zip(x_true.iter()) {
        assert!((xi - xt).abs() <= 1e-4, "xi={:.6}, expected {:.6}", xi, xt);
    }
}
