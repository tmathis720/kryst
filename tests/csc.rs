#![cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::assert_vec_close;
use kryst::matrix::backend::DefaultBackend;
use kryst::matrix::csc::CscMatrix;
use kryst::matrix::format::AsFormat;
use kryst::matrix::op::LinOp;
use kryst::matrix::sparse::CsrMatrix;

#[test]
fn csc_identity_matvec() {
    let n = 4;
    let col_ptr = vec![0, 1, 2, 3, 4];
    let row_idx = vec![0, 1, 2, 3];
    let vals = vec![R::from(1.0); 4];
    let csc = CscMatrix::from_csc(n, n, col_ptr, row_idx, vals);
    let mut y = vec![R::default(); n];
    let x = vec![R::from(2.0), R::from(3.0), R::from(5.0), R::from(7.0)];
    csc.matvec(&x, &mut y);
    let y_s: Vec<S> = y.iter().copied().map(S::from_real).collect();
    let x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
    assert_vec_close!("csc identity matvec", &y_s, &x_s);
}

#[test]
fn csc_dense_roundtrip_cache() {
    let a = Mat::<R>::from_fn(
        3,
        3,
        |i, j| if i == j { R::from(2.0) } else { R::default() },
    );
    let c1 = <Mat<R> as AsFormat<f64, DefaultBackend>>::to_csc_cached(&a, R::default());
    let c2 = <Mat<R> as AsFormat<f64, DefaultBackend>>::to_csc_cached(&a, R::default());
    assert!(Arc::ptr_eq(&c1, &c2));
}

#[test]
fn csr_to_csc_and_back() {
    let csr = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![R::from(1.0), R::from(2.0), R::from(3.0), R::from(4.0)],
    );
    let csc = <CsrMatrix<R> as AsFormat<f64, DefaultBackend>>::to_csc_cached(&csr, R::default());
    let csr2 = <CscMatrix<R> as AsFormat<f64, DefaultBackend>>::to_csr_cached(&csc, R::default());
    assert_eq!(csr.row_ptr(), csr2.row_ptr());
    assert_eq!(csr.col_idx(), csr2.col_idx());
    assert_eq!(csr.values(), csr2.values());
}

#[test]
fn csc_linop_t_matvec() {
    let csr = CsrMatrix::from_csr(
        2,
        3,
        vec![0, 1, 3],
        vec![0, 0, 1],
        vec![R::from(1.0), R::from(2.0), R::from(3.0)],
    );
    let csc = <CsrMatrix<R> as AsFormat<f64, DefaultBackend>>::to_csc_cached(&csr, R::default());
    let x = vec![R::from(10.0), R::from(100.0)];
    let mut y = vec![R::default(); 3];
    csc.t_matvec(&x, &mut y);
    let y_s: Vec<S> = y.iter().copied().map(S::from_real).collect();
    let expected = vec![S::from_real(210.0), S::from_real(300.0), S::from_real(0.0)];
    assert_vec_close!("csc transpose matvec", &y_s, &expected);
}
