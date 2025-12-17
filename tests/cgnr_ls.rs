#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::LinOp;

#[test]
fn cgnr_solves_simple_ls() {
    let mut a = Mat::<R>::zeros(3, 2);
    a[(0, 0)] = R::from(1.0);
    a[(1, 1)] = R::from(1.0);
    a[(2, 0)] = R::from(1.0);
    a[(2, 1)] = R::from(1.0);

    let amat: Arc<dyn LinOp<S = f64>> = Arc::new(a);
    let pmat = amat.clone();

    let b = [R::from(1.0), R::from(2.0), R::from(3.0)];
    let mut x = [R::default(); 2];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cgnr).unwrap();
    ksp.set_operators(amat, Some(pmat));
    let stats = ksp.solve(&b, &mut x).unwrap();

    assert!(
        (x[0] - R::from(1.0)).abs() < R::from(1e-8) && (x[1] - R::from(2.0)).abs() < R::from(1e-8),
        "x = {:?}",
        x
    );
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
}
