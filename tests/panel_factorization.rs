use kryst::algebra::prelude::*;
#[cfg(feature = "superlu_dist")]
use kryst::solver::superlu_dist::{Panel, PivotingStrategy};

#[test]
#[cfg(feature = "superlu_dist")]
fn tiny_pivot_replacement() {
    let mut panel = Panel {
        width: 2,
        height: 2,
        data: vec![R::from(1e-12), R::from(1.0), R::default(), R::from(1.0)],
        row_indices: vec![0, 1],
        col_start: 0,
    };
    let factor = panel
        .factorize_lu(1e-6, PivotingStrategy::ThresholdWithFallback)
        .expect("factorization");
    assert!(factor.tiny_pivots_replaced > 0);
    assert!(panel.data.iter().all(|v| !v.is_nan()));
}
