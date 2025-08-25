use std::sync::Arc;

use faer::Mat;
use kryst::matrix::csc::CscMatrix;
use kryst::matrix::format::AsFormat;
use kryst::matrix::op::LinOp;
use kryst::matrix::sparse::CsrMatrix;

#[test]
fn csc_identity_matvec() {
    let n = 4;
    let col_ptr = vec![0, 1, 2, 3, 4];
    let row_idx = vec![0, 1, 2, 3];
    let vals = vec![1.0; 4];
    let csc = CscMatrix::from_csc(n, n, col_ptr, row_idx, vals);
    let mut y = vec![0.0; n];
    let x = vec![2.0, 3.0, 5.0, 7.0];
    csc.matvec(&x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn csc_dense_roundtrip_cache() {
    let a = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 0.0 });
    let c1 = a.to_csc_cached(0.0);
    let c2 = a.to_csc_cached(0.0);
    assert!(Arc::ptr_eq(&c1, &c2));
}

#[test]
fn csr_to_csc_and_back() {
    let csr = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let csc = csr.to_csc_cached(0.0);
    let csr2 = csc.to_csr_cached(0.0);
    assert_eq!(csr.row_ptr(), csr2.row_ptr());
    assert_eq!(csr.col_idx(), csr2.col_idx());
    assert_eq!(csr.values(), csr2.values());
}

#[test]
fn csc_linop_t_matvec() {
    let csr = CsrMatrix::from_csr(2, 3, vec![0, 1, 3], vec![0, 0, 1], vec![1.0, 2.0, 3.0]);
    let csc = csr.to_csc_cached(0.0);
    let x = vec![10.0, 100.0];
    let mut y = vec![0.0; 3];
    csc.t_matvec(&x, &mut y);
    assert_eq!(y, vec![210.0, 300.0, 0.0]);
}
