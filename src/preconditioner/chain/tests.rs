#![cfg(feature = "backend-faer")]

use super::*;
#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::{PcSide, Preconditioner};
#[cfg(not(feature = "complex"))]
use faer::Mat;
#[cfg(not(feature = "complex"))]
use std::sync::Arc;

#[test]
#[cfg(not(feature = "complex"))]
fn pc_chain_applies_in_sequence() {
    struct AddOne;
    impl Preconditioner for AddOne {
        fn setup(&mut self, _a: &dyn crate::matrix::op::LinOp<S = S>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = *xi + S::from_real(1.0);
            }
            Ok(())
        }
    }
    struct ScaleTwo;
    impl Preconditioner for ScaleTwo {
        fn setup(&mut self, _a: &dyn crate::matrix::op::LinOp<S = S>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = S::from_real(2.0) * *xi;
            }
            Ok(())
        }
    }

    let mut chain = PcChain::new(vec![Box::new(AddOne), Box::new(ScaleTwo), Box::new(AddOne)]);

    let a = Mat::<f64>::from_fn(1, 1, |_, _| 1.0);
    chain.setup(&a).unwrap();

    let x = [S::from_real(3.0), S::from_real(-1.0)];
    let mut y = [S::zero(), S::zero()];
    chain.apply(PcSide::Left, &x, &mut y).unwrap();
    assert_eq!(
        y,
        [
            (S::from_real(3.0) + S::from_real(1.0)) * S::from_real(2.0) + S::from_real(1.0),
            (S::from_real(-1.0) + S::from_real(1.0)) * S::from_real(2.0)
                + S::from_real(1.0),
        ]
    );
}

#[test]
#[cfg(not(feature = "complex"))]
fn deferred_chain_constructs_in_setup() {
    use crate::context::ksp_context::KspContext;
    use crate::context::pc_context::PcFactory;
    use crate::matrix::op::LinOp;

    let specs = PcFactory::create_pc_chain_from_str("jacobi->jacobi", None).unwrap();
    let mut ksp = KspContext::new();
    ksp.pending_chain = Some(specs);

    let a = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let aop: Arc<dyn LinOp<S = S>> = Arc::new(a);
    ksp.set_operators(aop.clone(), None);

    ksp.setup().unwrap();
    assert!(ksp.is_setup());
}

#[cfg(all(test, feature = "complex"))]
#[test]
fn apply_s_matches_real_chain() {
    struct AddOne;
    impl Preconditioner for AddOne {
        fn setup(&mut self, _a: &dyn crate::matrix::op::LinOp<S = S>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = *xi + S::from_real(1.0);
            }
            Ok(())
        }
    }
    struct ScaleTwo;
    impl Preconditioner for ScaleTwo {
        fn setup(&mut self, _a: &dyn crate::matrix::op::LinOp<S = S>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = S::from_real(2.0) * *xi;
            }
            Ok(())
        }
    }

    let mut chain = PcChain::new(vec![Box::new(AddOne), Box::new(ScaleTwo), Box::new(AddOne)]);

    // let a = Mat::<f64>::from_fn(1, 1, |_, _| 1.0);
    // chain.setup(&a).unwrap();

    let x_real = [S::from_real(3.0), S::from_real(-1.0)];
    let mut y_real = [S::zero(), S::zero()];
    chain
        .apply(PcSide::Left, &x_real, &mut y_real)
        .expect("chain apply real");

    let x_s: Vec<S> = x_real.to_vec();
    let mut y_s = vec![S::zero(); x_s.len()];
    let mut scratch = BridgeScratch::default();
    chain
        .apply_s(PcSide::Left, &x_s, &mut y_s, &mut scratch)
        .expect("chain apply_s");

    for (ys, yr) in y_s.iter().zip(y_real.iter()) {
        assert!((ys.real() - yr.real()).abs() < 1e-12);
    }
}
