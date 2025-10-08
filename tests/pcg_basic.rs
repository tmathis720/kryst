use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use std::sync::Arc;

#[test]
fn pcg_solves_spd() {
    // A = [[4,1],[1,3]]
    let mut a = Mat::<R>::zeros(2, 2);
    a[(0, 0)] = R::from(4.0);
    a[(0, 1)] = R::from(1.0);
    a[(1, 0)] = R::from(1.0);
    a[(1, 1)] = R::from(3.0);

    // Wrap A as LinOp
    let amat: Arc<dyn kryst::matrix::op::LinOp<S = f64>> = Arc::new(a.clone());
    let pmat = amat.clone();

    let b = [R::from(1.0), R::from(2.0)];
    let mut x = [R::default(), R::default()];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Pcg).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.set_operators(amat, Some(pmat));
    let stats = ksp.solve(&b, &mut x).unwrap();

    let expected = [
        R::from(0.090_909_090_909_090_91),
        R::from(0.636_363_636_363_636_4),
    ];
    let x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
    let expected_s: Vec<S> = expected.iter().copied().map(S::from_real).collect();
    assert_vec_close!("pcg solves spd", &x_s, &expected_s);
    assert!(matches!(
        stats.reason,
        kryst::utils::convergence::ConvergedReason::ConvergedRtol
            | kryst::utils::convergence::ConvergedReason::ConvergedAtol
    ));
}
