use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use faer::Mat;
use std::sync::Arc;

#[test]
fn preonly_runs_with_lu_pc() {
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 4.0;
    a[(1, 1)] = 3.0;
    let amat = Arc::new(a.clone()) as Arc<dyn kryst::matrix::op::LinOp<S = f64>>;
    let pmat = Arc::new(a) as Arc<dyn kryst::matrix::op::LinOp<S = f64>>;

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Preonly).unwrap();
    ksp.set_pc_type(PcType::Lu, None).unwrap();
    ksp.set_operators(amat, Some(pmat));

    let b = [1.0, 2.0];
    let mut x = [0.0, 0.0];
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert_eq!(stats.iterations, 1);
}
