use super::*;
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;
use std::sync::Arc;

#[test]
fn pc_chain_applies_in_sequence() {
    struct AddOne;
    impl Preconditioner for AddOne {
        fn setup(&mut self, _a: &dyn crate::matrix::op::LinOp<S = f64>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = xi + 1.0;
            }
            Ok(())
        }
    }
    struct ScaleTwo;
    impl Preconditioner for ScaleTwo {
        fn setup(&mut self, _a: &dyn crate::matrix::op::LinOp<S = f64>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = 2.0 * xi;
            }
            Ok(())
        }
    }

    let mut chain = PcChain::new(vec![Box::new(AddOne), Box::new(ScaleTwo), Box::new(AddOne)]);

    let a = Mat::<f64>::from_fn(1, 1, |_, _| 1.0);
    chain.setup(&a).unwrap();

    let x = [3.0, -1.0];
    let mut y = [0.0, 0.0];
    chain.apply(PcSide::Left, &x, &mut y).unwrap();
    assert_eq!(y, [(3.0 + 1.0) * 2.0 + 1.0, (-1.0 + 1.0) * 2.0 + 1.0]);
}

#[test]
fn deferred_chain_constructs_in_setup() {
    use crate::context::ksp_context::KspContext;
    use crate::context::pc_context::PcFactory;
    use crate::matrix::op::LinOp;

    let specs = PcFactory::create_pc_chain_from_str("jacobi->jacobi", None).unwrap();
    let mut ksp = KspContext::new();
    ksp.pending_chain = Some(specs);

    let a = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let aop: Arc<dyn LinOp<S = f64>> = Arc::new(a);
    ksp.set_operators(aop.clone(), None);

    ksp.setup().unwrap();
    assert!(ksp.is_setup());
}
