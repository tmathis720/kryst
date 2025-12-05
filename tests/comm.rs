#![cfg(feature = "backend-faer")]
use faer::Mat;
use kryst::matrix::op::LinOp;
use kryst::parallel::{NoComm, UniverseComm};
#[cfg(not(any(feature = "mpi", feature = "rayon")))]
use std::sync::Arc;

#[test]
fn dense_linop_default_comm_is_no_comm() {
    let a = Mat::<f64>::zeros(4, 4);
    let op: &dyn LinOp<S = f64> = &a;
    assert_eq!(op.comm(), UniverseComm::NoComm(NoComm));
}

#[cfg(not(any(feature = "mpi", feature = "rayon")))]
#[test]
#[should_panic(expected = "communicator mismatch")]
fn set_operators_panics_on_comm_mismatch() {
    use kryst::context::ksp_context::KspContext;

    struct SerialOp;
    impl LinOp for SerialOp {
        type S = f64;
        fn dims(&self) -> (usize, usize) {
            (1, 1)
        }
        fn matvec(&self, _x: &[f64], _y: &mut [f64]) {}
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn comm(&self) -> UniverseComm {
            UniverseComm::Serial
        }
    }

    let a = Arc::new(Mat::<f64>::zeros(1, 1));
    let p = Arc::new(SerialOp);
    let mut ksp = KspContext::new();
    ksp.set_operators(a, Some(p));
}
