#![cfg(not(feature = "complex"))]
use kryst::algebra::bridge::BridgeScratch;
use kryst::algebra::prelude::*;
use kryst::assert_s_close;
use kryst::error::KError;
use kryst::matrix::op::LinOp;
use kryst::matrix::op_bridge::matvec_s;
use kryst::preconditioner::bridge::apply_pc_s;
use kryst::preconditioner::{PcSide, Preconditioner};
use std::any::Any;

struct Scale2Op;

impl LinOp for Scale2Op {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (0, 0)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi = 2.0 * xi;
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct IdentityPc;

impl Preconditioner for IdentityPc {
    fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        y.copy_from_slice(x);
        Ok(())
    }
}

#[test]
fn bridge_roundtrip() {
    let n = 4;
    let a = Scale2Op;
    let pc = IdentityPc;
    let mut scratch = BridgeScratch::default();

    let x: Vec<S> = (0..n).map(|i| S::from_real(i as f64)).collect();
    let mut y = vec![S::zero(); n];

    matvec_s(&a, &x, &mut y, &mut scratch);
    for (i, &yi) in y.iter().enumerate() {
        assert_s_close!("bridge matvec", yi, S::from_real(2.0 * i as f64));
        #[cfg(feature = "complex")]
        {
            assert_eq!(yi.conj(), yi);
        }
    }

    apply_pc_s(&pc, PcSide::Left, &x, &mut y, &mut scratch).unwrap();
    for (i, &yi) in y.iter().enumerate() {
        assert_s_close!("bridge pc", yi, S::from_real(i as f64));
    }
}
