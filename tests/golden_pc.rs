#![cfg(not(feature = "complex"))]
use std::sync::Arc;

use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::op::CsrOp;
use kryst::preconditioner::{PcSide, Preconditioner};

mod fixtures;
use fixtures::*;

#[test]
fn jacobi_golden_apply() {
    let a = csr_poisson_1d(4);
    // Expected Jacobi inverse of diag([2,2,2,2]) is 0.5 I
    let x = vec![R::from(1.0), R::from(2.0), R::from(3.0), R::from(4.0)];
    let mut y = vec![R::default(); 4];

    let mut jac = kryst::preconditioner::jacobi::Jacobi::new();
    jac.setup(&a).unwrap();
    jac.apply(PcSide::Left, &x, &mut y).unwrap();

    let expected: Vec<R> = x.iter().map(|xi| *xi * R::from(0.5)).collect();
    let expected_s: Vec<S> = expected.iter().copied().map(S::from_real).collect();
    let y_s: Vec<S> = y.iter().copied().map(S::from_real).collect();
    assert_vec_close!("jacobi golden apply", &y_s, &expected_s);
}

#[test]
fn cg_on_spd_converges_with_tight_tol() {
    let n = 32;
    let a = csr_poisson_1d(n);
    let b = vec![R::from(1.0); n];
    let mut x = vec![R::default(); n];

    let aop = CsrOp::new(Arc::new(a.clone()));
    let mut ksp = KspContext::new();
    ksp.set_tolerances(1e-8, 1e-14, 1e6, 2000)
        .set_type(SolverType::Pcg)
        .unwrap()
        .set_pc_type(PcType::Jacobi, None)
        .unwrap();
    ksp.try_set_pc_side(kryst::preconditioner::PcSide::Left)
        .unwrap();

    ksp.set_operators(Arc::new(aop), None);
    ksp.setup().unwrap();
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert!(stats.final_residual < 1e-5);
    assert!(stats.iterations < 400);
}
