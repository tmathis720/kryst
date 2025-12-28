#![cfg(all(feature = "backend-faer", feature = "complex"))]

use std::sync::Arc;

use faer::Mat;
use kryst::matrix::op::{DenseOp, LinOp};
use num_complex::Complex64;

#[test]
fn denseop_complex_matvec_and_adjoint() {
    let mat = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => Complex64::new(1.0, 1.0),
        (0, 1) => Complex64::new(2.0, 0.0),
        (1, 0) => Complex64::new(3.0, -1.0),
        (1, 1) => Complex64::new(4.0, 2.0),
        _ => Complex64::new(0.0, 0.0),
    });
    let op = DenseOp::<Complex64>::new(Arc::new(mat));

    let x = vec![Complex64::new(1.0, -1.0), Complex64::new(2.0, 1.0)];
    let mut y = vec![Complex64::new(0.0, 0.0); 2];
    op.matvec(&x, &mut y);

    let expected = vec![Complex64::new(6.0, 2.0), Complex64::new(8.0, 4.0)];
    assert_eq!(y, expected);

    let mut yt = vec![Complex64::new(0.0, 0.0); 2];
    op.t_matvec(&x, &mut yt);
    let expected_t = vec![Complex64::new(5.0, 3.0), Complex64::new(12.0, -2.0)];
    assert_eq!(yt, expected_t);
}
