#![cfg(not(feature = "complex"))]
use kryst::algebra::prelude::*;

#[test]
fn prelude_basic_ops_compile() {
    let a: S = S::from_real(3.0);
    let b: S = S::from_real(4.0);
    let _sum: S = a + b;
    let _prod: S = a * b;
    let _neg: S = -a;
    let _conj: S = a.conj();
    let _abs: R = a.abs();
    let _inv: S = a.inv();
    assert!(a.is_finite());
}
