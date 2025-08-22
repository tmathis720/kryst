use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use faer::Mat;
use std::sync::Arc;
use kryst::matrix::op::LinOp;

#[test]
fn preonly_runs_with_lu_pc() {
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 3.0;
    let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());
    let pmat: Arc<dyn LinOp<S = f64>> = Arc::new(a);

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Preonly).unwrap();
    ksp.set_pc_type(PcType::Lu, None).unwrap();
    ksp.set_operators(amat, Some(pmat));

    let b = [1.0, 2.0];
    let mut x = [0.0, 0.0];
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert_eq!(stats.iterations, 1);
    assert!((x[0] - 0.0909090909).abs() < 1e-8);
    assert!((x[1] - 0.6363636363).abs() < 1e-8);
}
