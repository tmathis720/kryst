#![cfg(feature = "backend-faer")]
use std::sync::Arc;

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::preconditioner::PcSide;

#[test]
fn fgmres_solves_dd_nonsym() {
    // Non-symmetric, diagonally-dominant 5x5 matrix
    let data = [
        [
            S::from_real(10.0).real(),
            S::from_real(2.0).real(),
            R::default(),
            R::default(),
            R::default(),
        ],
        [
            S::from_real(3.0).real(),
            S::from_real(15.0).real(),
            S::from_real(4.0).real(),
            R::default(),
            R::default(),
        ],
        [
            R::default(),
            S::from_real(-2.0).real(),
            S::from_real(8.0).real(),
            S::from_real(1.0).real(),
            R::default(),
        ],
        [
            R::default(),
            R::default(),
            S::from_real(1.0).real(),
            S::from_real(7.0).real(),
            S::from_real(3.0).real(),
        ],
        [
            R::default(),
            R::default(),
            R::default(),
            S::from_real(2.0).real(),
            S::from_real(12.0).real(),
        ],
    ];
    let mut a = Mat::<R>::zeros(5, 5);
    for i in 0..5 {
        for j in 0..5 {
            a[(i, j)] = data[i][j];
        }
    }

    let amat: Arc<dyn kryst::matrix::op::LinOp<S = f64>> = Arc::new(a.clone());
    let pmat = amat.clone();

    // x_true = [1,2,3,4,5]; b = A x_true
    let x_true = [
        S::from_real(1.0).real(),
        S::from_real(2.0).real(),
        S::from_real(3.0).real(),
        S::from_real(4.0).real(),
        S::from_real(5.0).real(),
    ];
    let mut b = [R::default(); 5];
    amat.matvec(&x_true, &mut b);

    let mut x = [R::default(); 5];
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
    let x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
    let expected_s: Vec<S> = x_true.iter().copied().map(S::from_real).collect();
    assert_vec_close!("fgmres solves dd nonsym", &x_s, &expected_s);
}
