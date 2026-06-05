#![cfg(feature = "complex")]

use std::sync::Arc;

use kryst::algebra::prelude::*;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::amg::{AMG, AMGBuilder, CoarseSolve};
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn amg_complex_setup_apply_small() {
    let csr = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(1.0)]);
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "native_diagonal");

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
    assert_eq!(amg.complex_setup_mode_label(), "native_diagonal");

    let rhs = vec![S::from_parts(3.0, -4.0), S::from_parts(2.0, 5.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let expected = vec![S::from_parts(0.4, -2.2), S::from_parts(1.3, -1.1)];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1e-12, "got={got:?}, expected={exp:?}");
    }
}

#[test]
fn amg_complex_rejects_coarse_ilu_until_native_hierarchy_support() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.25),
            S::from_parts(-1.0, -0.25),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .coarse_solve(CoarseSolve::ILU)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    let err = amg
        .setup(&op)
        .expect_err("complex coarse ILU should be rejected");
    assert!(
        err.to_string().contains("coarse_solve=ILU"),
        "unexpected error: {err}"
    );
}

#[test]
fn amg_complex_native_hierarchy_required_rejects_imaginary_coupling() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.25),
            S::from_parts(-1.0, -0.25),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    let err = amg
        .setup(&op)
        .expect_err("native complex hierarchy should be required");
    assert!(
        err.to_string().contains("native complex hierarchy"),
        "unexpected error: {err}"
    );
}

#[test]
fn amg_complex_native_hierarchy_required_allows_real_valued_complex_matrix() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    amg.setup(&op)
        .expect("real-valued complex operator can use real hierarchy");
}

#[test]
fn amg_complex_native_hierarchy_required_allows_complex_diagonal_fast_path() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_parts(2.0, 1.0), S::from_parts(-1.0, 3.0)],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    amg.setup(&op)
        .expect("complex diagonal fast path is native scalar algebra");

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
        vec![S::from_real(1.0), S::from_real(1.0)],
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
fn amg_complex_rejects_imaginary_transfer_override_until_native_hierarchy_support() {
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
        vec![S::from_parts(1.0, 0.2), S::from_real(1.0)],
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

    let err = amg
        .setup(&op)
        .expect_err("imaginary complex transfer override should be rejected");
    assert!(
        err.to_string().contains("imaginary prolongation"),
        "unexpected error: {err}"
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
    let mut amg = AMGBuilder::new()
        .logging_level(1)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");
    amg.setup(&op).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "projected_real_hierarchy");
    assert_eq!(
        amg.stats().expect("stats").complex_setup_mode.as_str(),
        "projected_real_hierarchy"
    );

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
