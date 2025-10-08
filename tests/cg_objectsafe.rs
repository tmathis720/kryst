use std::sync::Arc;

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;

#[test]
fn cg_solves_spd_2x2() {
    let mut a = Mat::<R>::zeros(2, 2);
    a[(0, 0)] = R::from(4.0);
    a[(0, 1)] = R::from(1.0);
    a[(1, 0)] = R::from(1.0);
    a[(1, 1)] = R::from(3.0);

    let amat: Arc<dyn kryst::matrix::op::LinOp<S = f64>> = Arc::new(a.clone());
    let pmat = amat.clone();

    let x_true = [R::from(1.0), R::from(2.0)];
    let mut b = [R::default(); 2];
    amat.matvec(&x_true, &mut b);

    let mut x = [R::default(); 2];
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.set_tolerances(R::from(1e-12), R::default(), R::from(1e20), 1000);
    ksp.set_operators(amat.clone(), Some(pmat));
    let stats = ksp.solve(&b, &mut x).unwrap();

    let x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
    let x_true_s: Vec<S> = x_true.iter().copied().map(S::from_real).collect();
    assert_vec_close!("cg solves spd 2x2", &x_s, &x_true_s);
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
}

#[test]
fn cg_with_jacobi_pc() {
    let mut a = Mat::<R>::zeros(3, 3);
    a[(0, 0)] = R::from(4.0);
    a[(0, 1)] = R::from(1.0);
    a[(1, 0)] = R::from(1.0);
    a[(1, 1)] = R::from(3.0);
    a[(1, 2)] = R::from(1.0);
    a[(2, 1)] = R::from(1.0);
    a[(2, 2)] = R::from(2.0);

    let amat: Arc<dyn kryst::matrix::op::LinOp<S = f64>> = Arc::new(a.clone());
    let pmat = amat.clone();

    let x_true = [R::from(1.0), R::from(2.0), R::from(3.0)];
    let mut b = [R::default(); 3];
    amat.matvec(&x_true, &mut b);

    let mut x = [R::default(); 3];
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::Jacobi, None).unwrap();
    ksp.set_tolerances(R::from(1e-12), R::default(), R::from(1e20), 1000);
    ksp.set_operators(amat.clone(), Some(pmat));

    let _stats = ksp.solve(&b, &mut x).unwrap();

    let x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
    let x_true_s: Vec<S> = x_true.iter().copied().map(S::from_real).collect();
    assert_vec_close!("cg with jacobi pc", &x_s, &x_true_s);
}
