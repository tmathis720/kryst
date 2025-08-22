use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use faer::Mat;
use std::sync::Arc;

#[test]
fn lu_preonly_succeeds() {
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 2.0;
    a[(1, 1)] = 3.0;
    let amat = Arc::new(a) as Arc<dyn kryst::matrix::op::LinOp<S = f64>>;

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Preonly).unwrap();
    ksp.set_pc_type(PcType::Lu, None).unwrap();
    ksp.set_operators(amat.clone(), None);

    let b = [2.0, 6.0];
    let mut x = [0.0, 0.0];
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert_eq!(x, [1.0, 2.0]);
    assert_eq!(stats.iterations, 1);
}

#[test]
fn qr_preonly_succeeds() {
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 4.0;
    a[(1, 1)] = 5.0;
    let amat = Arc::new(a) as Arc<dyn kryst::matrix::op::LinOp<S = f64>>;

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Preonly).unwrap();
    ksp.set_pc_type(PcType::Qr, None).unwrap();
    ksp.set_operators(amat.clone(), None);

    let b = [4.0, 10.0];
    let mut x = [0.0, 0.0];
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert_eq!(x, [1.0, 2.0]);
    assert_eq!(stats.iterations, 1);
}
