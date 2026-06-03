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
fn amg_complex_diagonal_uses_complex_inverse() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_parts(2.0, 1.0), S::from_parts(-1.0, 3.0)],
    );
    let op = CsrOp::new(Arc::new(csr.clone()));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();

    let rhs = vec![S::from_parts(3.0, -4.0), S::from_parts(2.0, 5.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let expected = vec![S::from_parts(0.4, -2.2), S::from_parts(1.3, -1.1)];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1e-12, "got={got:?}, expected={exp:?}");
    }
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

#[test]
fn amg_complex_apply_residual_acceptance() {
    let csr = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            S::from_parts(4.0, 0.0),
            S::from_parts(-1.0, 0.2),
            S::from_parts(0.0, -0.1),
            S::from_parts(-1.0, -0.2),
            S::from_parts(4.2, 0.0),
            S::from_parts(-1.1, 0.1),
            S::from_parts(0.0, 0.1),
            S::from_parts(-1.1, -0.1),
            S::from_parts(3.8, 0.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr.clone()));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();

    let rhs = vec![
        S::from_parts(1.0, -0.3),
        S::from_parts(-0.5, 0.7),
        S::from_parts(0.25, 0.4),
    ];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let mut a_out = vec![S::zero(); rhs.len()];
    kryst::core::traits::MatVec::matvec(&csr, &out, &mut a_out);
    let r_inf = a_out
        .iter()
        .zip(rhs.iter())
        .map(|(l, r)| (*l - *r).abs())
        .fold(0.0f64, f64::max);
    assert!(
        r_inf.is_finite() && r_inf < 2.5,
        "residual too large: {r_inf}"
    );
}

#[test]
fn amg_complex_residual_reason_code_guard() {
    let csr = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(2.0)]);
    let op = CsrOp::new(Arc::new(csr.clone()));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();

    let rhs = vec![S::from_parts(1.0, 1.0)];
    let mut out = vec![S::zero(); 1];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let mut ax = vec![S::zero(); 1];
    kryst::core::traits::MatVec::matvec(&csr, &out, &mut ax);
    let r = (ax[0] - rhs[0]).abs();
    assert!(r < 1e-10, "unexpected residual guard trip: {r}");
}
