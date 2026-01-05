#![cfg(feature = "backend-faer")]

use super::*;
use crate::algebra::prelude::*;
#[cfg(not(feature = "complex"))]
use faer::Mat;

struct Dummy;

impl Preconditioner for Dummy {
    fn setup(&mut self, _pmat: &dyn LinOp<S = S>) -> Result<(), KError> {
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &[S], z: &mut [S]) -> Result<(), KError> {
        z.copy_from_slice(r);
        Ok(())
    }
}

#[test]
#[cfg(not(feature = "complex"))]
fn default_direct_solve_errors() {
    let mut pc = Dummy;
    let a = Mat::<f64>::zeros(1, 1);
    let mut x = [S::zero()];
    let rhs = [S::from_real(1.0)];
    let err = pc.direct_solve(&a, &rhs, &mut x).unwrap_err();
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
        fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
            Ok(())
        }
        fn apply(&self, _side: PcSide, _x: &[S], _y: &mut [S]) -> Result<(), KError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
    let mut pc = CountPc {
        calls: AtomicUsize::new(0),
    };
    let x = [S::zero(); 2];
    let mut y = [S::zero(); 2];
    pc.apply_mut(PcSide::Left, &x, &mut y).unwrap();
    assert_eq!(pc.calls.load(Ordering::Relaxed), 1);
}

mod asm_amg;
mod block_jacobi;
mod classical;
mod coarsen;
mod deflation;
mod direct_apply;
mod ilu_csr;
mod ilu_history;
mod jacobi;
mod legacy_bridge;
mod near_nullspace;
mod nodal;
mod post_interp;
mod spd;
