use std::sync::Arc;

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::matrix::format::AsFormat;
use kryst::matrix::op::DenseOp;

#[test]
fn denseop_csc_cache_includes_values_id() {
    // 2x2 dense with trivial values
    let m = Mat::<R>::from_fn(2, 2, |i, j| {
        if i == j {
            S::one().real()
        } else {
            R::default()
        }
    });
    let dop = DenseOp::new(Arc::new(m));
    // First conversion
    let c1 = <DenseOp as AsFormat<f64>>::to_csc_cached(&dop, R::default());
    // Mark numeric change (simulate in-place edits)
    dop.mark_values_changed();
    // Second conversion should not hit the same cache entry
    let c2 = <DenseOp as AsFormat<f64>>::to_csc_cached(&dop, R::default());

    // The arcs should be distinct instances because key includes ValuesId
    let p1 = Arc::as_ptr(&c1) as usize;
    let p2 = Arc::as_ptr(&c2) as usize;
    assert_ne!(p1, p2, "CSC cache entry not invalidated on values change");
}

#[cfg(not(feature = "mat-values-fingerprint"))]
#[test]
fn raw_mat_csc_cache_is_stable_without_values_tracking() {
    // Raw Mat has values_id == 0 -> same key if structure and drop_tol stay the same
    let m = Mat::<R>::from_fn(2, 2, |i, j| {
        if i == j {
            S::one().real()
        } else {
            R::default()
        }
    });
    let c1 = <Mat<R> as AsFormat<f64>>::to_csc_cached(&m, R::default());
    let c2 = <Mat<R> as AsFormat<f64>>::to_csc_cached(&m, R::default());
    let p1 = Arc::as_ptr(&c1) as usize;
    let p2 = Arc::as_ptr(&c2) as usize;
    assert_eq!(
        p1, p2,
        "Raw Mat should reuse the same CSC when values_id is unknown (0)"
    );
}
