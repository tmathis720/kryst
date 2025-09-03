use super::*;
use crate::preconditioner::amg::util::DofLayout;
use crate::preconditioner::amg::strength_nodal::strength_nodal;
use crate::preconditioner::amg::coarsen::{build_aggregates, AggAlgo, AggOpts, lift_node_aggregates_to_dofs};
use crate::matrix::sparse::CsrMatrix;

#[test]
fn nodal_aggregates_group_dofs() {
    // 4 DOFs, block_size=2 -> 2 nodes
    let a = CsrMatrix::identity(4);
    let layout = DofLayout::new(4, 2);
    let s = strength_nodal(&a, &layout, 0.0, true);
    let (agg_node, is_c_node) = build_aggregates(&s, AggAlgo::RSGreedy, &AggOpts { mis_k: 1, cap_per_row: None });
    let (agg, is_c) = lift_node_aggregates_to_dofs(&agg_node, &is_c_node, &layout);
    assert_eq!(agg, vec![0,0,1,1]);
    assert_eq!(is_c, vec![true,true,true,true]);
}
