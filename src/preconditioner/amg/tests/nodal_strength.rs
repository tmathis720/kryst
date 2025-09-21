use super::DofLayout;
use super::coarsen::{
    AggAlgo, AggOpts, build_aggregates, build_aggregates_nodal, lift_node_aggregates_to_dofs,
};
use super::strength_nodal;
use crate::matrix::sparse::CsrMatrix;

fn make_sample_matrix() -> CsrMatrix<f64> {
    let row_ptr = vec![0, 2, 5, 8, 10];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let vals = vec![
        4.0, -1.0, // row 0
        -1.0, 4.0, -2.0, // row 1
        -2.0, 5.0, -3.0, // row 2
        -3.0, 6.0, // row 3
    ];
    CsrMatrix::from_csr(4, 4, row_ptr, col_idx, vals)
}

#[test]
fn nodal_strength_frobenius_thresholding() {
    let a = make_sample_matrix();
    let nodal = strength_nodal::strength_nodal_from_csr(&a, 2, 0.25, false);
    assert_eq!(nodal.row_ptr, vec![0, 1, 2]);
    assert_eq!(nodal.col_idx, vec![1, 0]);

    // Raising the threshold above 1.0 prunes the only neighbor in the unnormalized graph
    let nodal_strict = strength_nodal::strength_nodal_from_csr(&a, 2, 1.1, false);
    assert_eq!(nodal_strict.row_ptr, vec![0, 0, 0]);
    assert!(nodal_strict.col_idx.is_empty());
}

#[test]
fn nodal_strength_normalized_scaling() {
    let a = make_sample_matrix();
    let nodal = strength_nodal::strength_nodal_from_csr(&a, 2, 0.03, true);
    assert_eq!(nodal.row_ptr, vec![0, 1, 2]);
    assert_eq!(nodal.col_idx, vec![1, 0]);

    // Slightly higher threshold should cull the connection once normalized
    let nodal_high = strength_nodal::strength_nodal_from_csr(&a, 2, 0.04, true);
    assert_eq!(nodal_high.row_ptr, vec![0, 0, 0]);
    assert!(nodal_high.col_idx.is_empty());
}

#[test]
fn nodal_aggregation_matches_legacy() {
    let a = make_sample_matrix();
    let layout = DofLayout::new(a.nrows(), 2);
    let theta = 0.2;
    let legacy_strength = strength_nodal::strength_nodal(&a, &layout, theta, false);
    let opts = AggOpts {
        mis_k: 1,
        cap_per_row: None,
    };
    let (agg_legacy_nodes, seeds_legacy_nodes) =
        build_aggregates(&legacy_strength, AggAlgo::PMIS, &opts);

    let nodal_strength = strength_nodal::strength_nodal_from_csr(&a, 2, theta, false);
    let (agg_nodes, seeds_nodes) = build_aggregates_nodal(&nodal_strength, AggAlgo::PMIS, &opts);
    assert_eq!(agg_nodes.len(), layout.n_nodes);
    assert_eq!(seeds_nodes.len(), layout.n_nodes);

    let (agg_from_nodes, seeds_from_nodes) =
        lift_node_aggregates_to_dofs(&agg_nodes, &seeds_nodes, &layout);
    assert_eq!(agg_from_nodes, vec![0, 0, 0, 0]);
    assert_eq!(seeds_from_nodes, vec![true, true, false, false]);

    let (agg_legacy_dof, seeds_legacy_dof) =
        lift_node_aggregates_to_dofs(&agg_legacy_nodes, &seeds_legacy_nodes, &layout);
    assert_eq!(agg_legacy_dof, agg_from_nodes);
    assert_eq!(seeds_legacy_dof, seeds_from_nodes);
}
