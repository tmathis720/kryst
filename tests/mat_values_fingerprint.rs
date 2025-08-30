#![cfg(feature = "mat-values-fingerprint")]

use faer::Mat;
use kryst::matrix::format::AsFormat;
use std::sync::Arc;

#[test]
fn raw_mat_csc_cache_invalidation_with_fingerprint() {
    let mut m = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let c1 = <Mat<f64> as AsFormat>::to_csc_cached(&m, 0.0);
    // mutate values in-place
    m[(0, 0)] = 2.0;
    let c2 = <Mat<f64> as AsFormat>::to_csc_cached(&m, 0.0);
    let p1 = Arc::as_ptr(&c1) as usize;
    let p2 = Arc::as_ptr(&c2) as usize;
    assert_ne!(p1, p2, "CSC cache should invalidate when raw Mat values change under fingerprint feature");
}

