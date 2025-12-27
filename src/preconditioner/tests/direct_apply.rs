#![cfg(all(not(feature = "complex"), feature = "backend-faer"))]

use crate::algebra::prelude::*;
#[cfg(any(feature = "dense-direct", feature = "superlu_dist"))]
use crate::error::KError;
use crate::matrix::op::CsrOp;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::PcSide;

#[cfg(any(feature = "dense-direct", feature = "superlu_dist"))]
use crate::preconditioner::Preconditioner;
use crate::preconditioner::builders as b;
#[cfg(feature = "superlu_dist")]
use crate::preconditioner::direct::SuperLuDistPc;
#[cfg(feature = "dense-direct")]
use crate::preconditioner::direct::{LuPc, QrPc};
use std::sync::Arc;

#[cfg(feature = "dense-direct")]
#[test]
#[cfg(not(feature = "complex"))]
fn direct_pc_apply_is_not_identity() {
    // LU apply should return a clear Unsupported error (PREONLY-only)
    let mut pc = LuPc::new();
    let two = S::from_real(2.0);
    let zero = S::zero();
    let a = faer::Mat::<f64>::from_fn(3, 3, |i, j| if i == j { two.real() } else { 0.0 });
    pc.setup(&a as &dyn LinOp<S = S>).unwrap();
    let one = S::from_real(1.0);
    let x = vec![one; 3];
    let mut y = vec![zero; 3];
    let err = pc.apply(PcSide::Left, &x, &mut y).unwrap_err();
    match err {
        KError::Unsupported(msg) => assert!(msg.to_lowercase().contains("preonly")),
        _ => panic!("expected Unsupported error for LuPc::apply"),
    }

    // QR apply should also be PREONLY-only
    let mut pc = QrPc::new();
    pc.setup(&a as &dyn LinOp<S = S>).unwrap();
    let err = pc.apply(PcSide::Left, &x, &mut y).unwrap_err();
    match err {
        KError::Unsupported(msg) => assert!(msg.to_lowercase().contains("preonly")),
        _ => panic!("expected Unsupported error for QrPc::apply"),
    }
}

#[test]
#[cfg(not(feature = "complex"))]
fn builders_sor_and_chebyshev_object_safe() {
    // Identity CSR as operator
    let csr = CsrMatrix::identity(5);
    let op = CsrOp::new(Arc::new(csr));
    let one = S::from_real(1.0);
    let zero = S::zero();

    // SOR
    let mut sor = b::build_sor(one, 1, crate::preconditioner::sor::MatSorType::APPLY_LOWER)
        .expect("build_sor should succeed");
    sor.setup(&op as &dyn LinOp<S = S>).unwrap();
    let x = vec![one; 5];
    let mut y = vec![zero; 5];
    sor.apply(PcSide::Left, &x, &mut y).unwrap();
    crate::assert_vec_close!("sor apply matches identity", &x, &y);

    // Chebyshev
    let half = R::from(0.5);
    let three_halves = R::from(1.5);
    let mut cheb =
        b::build_chebyshev(2, half, three_halves).expect("build_chebyshev should succeed");
    cheb.setup(&op as &dyn LinOp<S = S>).unwrap();
    let mut z = vec![zero; 5];
    cheb.apply(PcSide::Left, &x, &mut z).unwrap();
    assert!(z.iter().copied().all(|v| v.is_finite()));
}

#[test]
#[cfg(not(feature = "complex"))]
fn ilu_right_side_errors() {
    use crate::matrix::op::CsrOp;
    let csr = CsrMatrix::identity(3);
    let op = CsrOp::new(Arc::new(csr));
    let mut pc = b::build_ilu0().expect("build_ilu0 should succeed");
    pc.setup(&op as &dyn LinOp<S = S>).unwrap();
    let one = S::from_real(1.0);
    let zero = S::zero();
    let x = vec![one; 3];
    let mut y = vec![zero; 3];
    let err = pc.apply(PcSide::Right, &x, &mut y).unwrap_err();
    match err {
        crate::error::KError::InvalidInput(msg) => {
            assert!(msg.to_lowercase().contains("left only"))
        }
        _ => panic!("expected InvalidInput error"),
    }
}

#[cfg(all(feature = "dense-direct", feature = "complex"))]
#[test]
#[cfg(not(feature = "complex"))]
fn direct_pc_apply_s_matches_real_error() {
    let mut lu = LuPc::new();
    let mut qr = QrPc::new();
    let two = S::from_real(2.0).real();
    let zero = R::default();
    let a = faer::Mat::<f64>::from_fn(3, 3, |i, j| if i == j { two } else { zero });

    lu.setup(&a as &dyn LinOp<S = S>).unwrap();
    qr.setup(&a as &dyn LinOp<S = S>).unwrap();

    let rhs_s = vec![S::from_real(1.0); 3];
    let mut out_s = vec![S::zero(); 3];
    let mut scratch = BridgeScratch::default();

    let err = lu
        .apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
        .expect_err("LuPc::apply_s should surface PREONLY error");
    match err {
        KError::Unsupported(msg) => assert!(msg.to_lowercase().contains("preonly")),
        other => panic!("expected Unsupported error for LuPc::apply_s, got {other:?}"),
    }

    let err = qr
        .apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
        .expect_err("QrPc::apply_s should surface PREONLY error");
    match err {
        KError::Unsupported(msg) => assert!(msg.to_lowercase().contains("preonly")),
        other => panic!("expected Unsupported error for QrPc::apply_s, got {other:?}"),
    }
}

#[cfg(all(feature = "superlu_dist", feature = "complex"))]
#[test]
#[cfg(not(feature = "complex"))]
fn superlu_dist_apply_s_matches_real_error() {
    use crate::matrix::op::CsrOp;
    use crate::matrix::sparse::CsrMatrix;

    let csr = Arc::new(CsrMatrix::identity(3));
    let op = CsrOp::new(csr);

    let mut pc = SuperLuDistPc::new();
    pc.setup(&op as &dyn LinOp<S = S>)
        .expect("setup should accept CSR input under superlu_dist");

    let one = S::from_real(1.0);
    let zero = S::zero();
    let x_real = vec![one; 3];
    let mut y_real = vec![zero; 3];
    let err_real = pc
        .apply(PcSide::Left, &x_real, &mut y_real)
        .expect_err("SuperLuDistPc::apply should remain PREONLY-only");
    match err_real {
        KError::Unsupported(msg) => assert!(msg.to_lowercase().contains("preonly")),
        other => panic!("expected Unsupported error for SuperLuDistPc::apply, got {other:?}"),
    }

    let rhs_s = vec![S::from_real(1.0); 3];
    let mut out_s = vec![S::zero(); 3];
    let mut scratch = BridgeScratch::default();
    let err_s = pc
        .apply_s(PcSide::Left, &rhs_s, &mut out_s, &mut scratch)
        .expect_err("SuperLuDistPc::apply_s should surface PREONLY error");
    match err_s {
        KError::Unsupported(msg) => assert!(msg.to_lowercase().contains("preonly")),
        other => panic!("expected Unsupported error for SuperLuDistPc::apply_s, got {other:?}"),
    }
}
