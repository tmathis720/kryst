use kryst::solver::superlu_dist::{Panel, PivotingStrategy};

#[test]
fn tiny_pivot_replacement() {
    let mut panel = Panel {
        width: 2,
        height: 2,
        data: vec![1e-12, 1.0, 0.0, 1.0],
        row_indices: vec![0, 1],
        col_start: 0,
    };
    let factor = panel
        .factorize_lu(1e-6, PivotingStrategy::ThresholdWithFallback)
        .expect("factorization");
    assert!(factor.tiny_pivots_replaced > 0);
    assert!(panel.data.iter().all(|v| !v.is_nan()));
}
