use std::sync::Arc;

use faer::Mat;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;

#[test]
fn cg_solves_spd_2x2() {
    let mut a = Mat::<f64>::zeros(2, 2);
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 3.0;

    let amat: Arc<dyn kryst::matrix::op::LinOp<S = f64>> = Arc::new(a.clone());
    let pmat = amat.clone();

    let x_true = [1.0, 2.0];
    let mut b = [0.0, 0.0];
    amat.matvec(&x_true, &mut b);

    let mut x = [0.0, 0.0];
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.set_tolerances(1e-12, 0.0, 1e20, 1000);
    ksp.set_operators(amat.clone(), Some(pmat));
    let stats = ksp.solve(&b, &mut x).unwrap();

    let mut r = [0.0, 0.0];
    amat.matvec(&x, &mut r);
    for i in 0..2 {
        r[i] = b[i] - r[i];
    }
    let res = (r[0] * r[0] + r[1] * r[1]).sqrt();
    assert!(res <= 1e-04, "res too large: {}", res);
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
}

#[test]
fn cg_with_jacobi_pc() {
    let mut a = Mat::<f64>::zeros(3, 3);
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 3.0;
    a[(1, 2)] = 1.0;
    a[(2, 1)] = 1.0;
    a[(2, 2)] = 2.0;

    let amat: Arc<dyn kryst::matrix::op::LinOp<S = f64>> = Arc::new(a.clone());
    let pmat = amat.clone();

    let x_true = [1.0, 2.0, 3.0];
    let mut b = [0.0; 3];
    amat.matvec(&x_true, &mut b);

    let mut x = [0.0; 3];
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::Jacobi, None).unwrap();
    ksp.set_tolerances(1e-12, 0.0, 1e20, 1000);
    ksp.set_operators(amat.clone(), Some(pmat));

    let _stats = ksp.solve(&b, &mut x).unwrap();

    let mut r = [0.0; 3];
    amat.matvec(&x, &mut r);
    for i in 0..3 {
        r[i] = b[i] - r[i];
    }
    let res = (r.iter().map(|v| v * v).sum::<f64>()).sqrt();
    assert!(res <= 1e-04);
}
