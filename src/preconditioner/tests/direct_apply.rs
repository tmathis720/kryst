#[cfg(feature = "dense-direct")]
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::op::CsrOp;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::builders as b;
#[cfg(feature = "dense-direct")]
use crate::preconditioner::direct::{LuPc, QrPc};
use crate::preconditioner::{PcSide, Preconditioner};
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
    let mut sor = b::build_sor(1.0, 1, crate::preconditioner::sor::MatSorType::APPLY_LOWER, false)
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
    assert!(z.iter().all(|v| v.is_finite()));
}
