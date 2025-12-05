#![cfg(feature = "backend-faer")]
#![cfg(feature = "mat-values-fingerprint")]

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::matrix::backend::DefaultBackend;
use kryst::matrix::format::AsFormat;
use std::sync::Arc;

#[test]
fn raw_mat_csc_cache_invalidation_with_fingerprint() {
    let mut m = Mat::<R>::from_fn(
        2,
        2,
        |i, j| if i == j { R::from(1.0) } else { R::default() },
    );
    let c1 = <Mat<R> as AsFormat<f64, DefaultBackend>>::to_csc_cached(&m, R::default());
    // mutate values in-place
    m[(0, 0)] = R::from(2.0);
    let c2 = <Mat<R> as AsFormat<f64, DefaultBackend>>::to_csc_cached(&m, R::default());
    let p1 = Arc::as_ptr(&c1) as usize;
    let p2 = Arc::as_ptr(&c2) as usize;
    assert_ne!(
        p1, p2,
        "CSC cache should invalidate when raw Mat values change under fingerprint feature"
    );
}
