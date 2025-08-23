use super::*;
use faer::Mat;

struct Dummy;

impl Preconditioner for Dummy {
    fn setup(&mut self, _pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        z.copy_from_slice(r);
        Ok(())
    }
}

#[test]
fn default_direct_solve_errors() {
    let mut pc = Dummy;
    let a = Mat::<f64>::zeros(1, 1);
    let mut x = [0.0];
    let err = pc.direct_solve(&a, &[1.0], &mut x).unwrap_err();
    match err {
        KError::SolveError(msg) => assert!(msg.contains("direct_solve not supported")),
        _ => panic!("unexpected error variant"),
    }
}

#[test]
fn default_apply_mut_forwards_to_apply() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct CountPc {
        calls: AtomicUsize,
    }
    impl Preconditioner for CountPc {
        fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, _x: &[f64], _y: &mut [f64]) -> Result<(), KError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
    let mut pc = CountPc {
        calls: AtomicUsize::new(0),
    };
    let x = [0.0; 2];
    let mut y = [0.0; 2];
    pc.apply_mut(PcSide::Left, &x, &mut y).unwrap();
    assert_eq!(pc.calls.load(Ordering::Relaxed), 1);
}

mod legacy_bridge;
