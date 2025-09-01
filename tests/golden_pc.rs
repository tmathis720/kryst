use std::sync::Arc;

use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::op::CsrOp;
use kryst::preconditioner::{Preconditioner, PcSide};

mod fixtures;
use fixtures::*;

#[test]
fn jacobi_golden_apply() {
    let a = csr_poisson_1d(4);
    // Expected Jacobi inverse of diag([2,2,2,2]) is 0.5 I
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];

    let mut jac = kryst::preconditioner::jacobi::Jacobi::new();
    jac.setup(&a).unwrap();
    jac.apply(PcSide::Left, &x, &mut y).unwrap();

    for (yi, xi) in y.iter().zip(x.iter()) {
        assert!((*yi - 0.5 * xi).abs() < 1e-12);
    }
}

#[test]
fn cg_on_spd_converges_with_tight_tol() {
    let n = 32;
    let a = csr_poisson_1d(n);
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];

    let aop = CsrOp::new(Arc::new(a.clone()));
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Pcg)
        .unwrap()
        .set_pc_type(PcType::Jacobi, None)
        .unwrap()
        .set_tolerances(1e-8, 1e-14, 1e6, 2000);
    ksp.try_set_pc_side(kryst::preconditioner::PcSide::Left).unwrap();

    ksp.set_operators(Arc::new(aop), None);
    ksp.setup().unwrap();
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert!(stats.final_residual < 1e-5);
    assert!(stats.iterations < 400);
}
