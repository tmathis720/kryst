use faer::Mat;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use std::sync::Arc;

#[test]
fn pcg_solves_spd() {
    // A = [[4,1],[1,3]]
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 3.0;

    // Wrap A as LinOp
    let amat: Arc<dyn kryst::matrix::op::LinOp<S = f64>> = Arc::new(a.clone());
    let pmat = amat.clone();

    let b = [1.0, 2.0];
    let mut x = [0.0, 0.0];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Pcg).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.set_operators(amat, Some(pmat));
    let stats = ksp.solve(&b, &mut x).unwrap();

    let expected = [0.09090909090909091, 0.6363636363636364];
    assert!((x[0] - expected[0]).abs() < 1e-5);
    assert!((x[1] - expected[1]).abs() < 1e-5);
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
}
