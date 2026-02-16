#![cfg(feature = "complex")]

use std::sync::Arc;

use kryst::algebra::prelude::*;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::amg::AMG;
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn amg_complex_setup_apply_small() {
    let csr = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(1.0)]);
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();

    let rhs = vec![S::from_parts(1.0, -2.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    assert!(out[0].real().is_finite());
    assert!(out[0].imag().is_finite());
}

#[test]
fn amg_complex_transfer_override_plumbing() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.5),
            S::from_parts(-1.0, -0.5),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMG::default();

    let p = CsrMatrix::from_csr(
        2,
        1,
        vec![0, 1, 2],
        vec![0, 0],
        vec![S::from_parts(1.0, 0.2), S::from_parts(1.0, -0.2)],
    );
    let r = CsrMatrix::from_csr(
        1,
        2,
        vec![0, 2],
        vec![0, 1],
        vec![S::from_real(0.5), S::from_real(0.5)],
    );
    amg.set_level_transfer_operators(
        0,
        kryst::preconditioner::amg::AmgTransferOperators {
            prolongation: p,
            restriction: r,
        },
    );

    amg.setup(&op).unwrap();
    let rhs = vec![S::from_parts(1.0, 0.5), S::from_parts(-0.5, 1.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    assert!(
        out.iter()
            .all(|v| v.real().is_finite() && v.imag().is_finite())
    );
}
