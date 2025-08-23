use std::sync::Arc;

use faer::Mat;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::LinOp;

#[test]
fn cgnr_solves_simple_ls() {
    let mut a = Mat::<f64>::zeros(3, 2);
    a[(0, 0)] = 1.0;
    a[(1, 1)] = 1.0;
    a[(2, 0)] = 1.0;
    a[(2, 1)] = 1.0;

    let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a);
    let pmat = amat.clone();

    let b = [1.0, 2.0, 3.0];
    let mut x = [0.0; 2];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cgnr).unwrap();
    ksp.set_operators(amat, Some(pmat));
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert!((x[0] - 1.0).abs() < 1e-8 && (x[1] - 2.0).abs() < 1e-8, "x = {:?}", x);
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
}

