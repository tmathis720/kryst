#[cfg(all(
    feature = "complex",
    any(feature = "dense-direct", feature = "superlu_dist")
))]
use crate::algebra::bridge::BridgeScratch;
#[cfg(all(
    feature = "complex",
    any(feature = "dense-direct", feature = "superlu_dist")
))]
use crate::algebra::prelude::*;
#[cfg(any(feature = "dense-direct", feature = "superlu_dist"))]
use crate::error::KError;
use crate::matrix::op::CsrOp;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
#[cfg(all(
    feature = "complex",
    any(feature = "dense-direct", feature = "superlu_dist")
))]
use crate::ops::kpc::KPreconditioner;
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
fn direct_pc_apply_is_not_identity() {
    // LU apply should return a clear Unsupported error (PREONLY-only)
    let mut pc = LuPc::new();
    let a = faer::Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 0.0 });
    pc.setup(&a as &dyn LinOp<S = f64>).unwrap();
    let x = vec![1.0; 3];
    let mut y = vec![0.0; 3];
    let err = pc.apply(PcSide::Left, &x, &mut y).unwrap_err();
    match err {
        KError::Unsupported(msg) => assert!(msg.to_lowercase().contains("preonly")),
        _ => panic!("expected Unsupported error for LuPc::apply"),
    }

    // QR apply should also be PREONLY-only
    let mut pc = QrPc::new();
    pc.setup(&a as &dyn LinOp<S = f64>).unwrap();
    let err = pc.apply(PcSide::Left, &x, &mut y).unwrap_err();
    match err {
        KError::Unsupported(msg) => assert!(msg.to_lowercase().contains("preonly")),
        _ => panic!("expected Unsupported error for QrPc::apply"),
    }
}

#[test]
fn builders_sor_and_chebyshev_object_safe() {
    // Identity CSR as operator
    let csr = CsrMatrix::identity(5);
    let op = CsrOp::new(Arc::new(csr));

    // SOR
    let mut sor = b::build_sor(
        1.0,
        1,
        crate::preconditioner::sor::MatSorType::APPLY_LOWER,
        false,
    )
    .expect("build_sor should succeed");
    sor.setup(&op as &dyn LinOp<S = f64>).unwrap();
    let x = vec![1.0; 5];
    let mut y = vec![0.0; 5];
    sor.apply(PcSide::Left, &x, &mut y).unwrap();
    assert_eq!(x, y);

    // Chebyshev
    let mut cheb = b::build_chebyshev(2, 0.5, 1.5).expect("build_chebyshev should succeed");
    cheb.setup(&op as &dyn LinOp<S = f64>).unwrap();
    let mut z = vec![0.0; 5];
    cheb.apply(PcSide::Left, &x, &mut z).unwrap();
    assert!(z.iter().copied().all(|v| v.is_finite()));
}

#[test]
fn ilu_right_side_errors() {
    use crate::matrix::op::CsrOp;
    let csr = CsrMatrix::identity(3);
    let op = CsrOp::new(Arc::new(csr));
    let mut pc = b::build_ilu0().expect("build_ilu0 should succeed");
    pc.setup(&op as &dyn LinOp<S = f64>).unwrap();
    let x = vec![1.0; 3];
    let mut y = vec![0.0; 3];
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
fn direct_pc_apply_s_matches_real_error() {
    let mut lu = LuPc::new();
    let mut qr = QrPc::new();
    let a = faer::Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 0.0 });

    lu.setup(&a as &dyn LinOp<S = f64>).unwrap();
    qr.setup(&a as &dyn LinOp<S = f64>).unwrap();

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
fn superlu_dist_apply_s_matches_real_error() {
    use crate::matrix::op::CsrOp;
    use crate::matrix::sparse::CsrMatrix;

    let csr = Arc::new(CsrMatrix::identity(3));
    let op = CsrOp::new(csr);

    let mut pc = SuperLuDistPc::new();
    pc.setup(&op as &dyn LinOp<S = f64>)
        .expect("setup should accept CSR input under superlu_dist");

    let x_real = vec![1.0; 3];
    let mut y_real = vec![0.0; 3];
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
