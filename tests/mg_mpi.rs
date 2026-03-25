#![cfg(all(feature = "mpi", not(feature = "complex")))]

mod fixtures;

use std::sync::Arc;

use kryst::matrix::dist_csr::DistCsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, MpiComm, UniverseComm};
use kryst::preconditioner::mg::MgPc;
use kryst::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};

fn local_rows_from_global(
    global: &CsrMatrix<f64>,
    row_start: usize,
    n_local: usize,
) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n_local + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n_local {
        let (cols, vals) = global.row(row_start + i);
        col_idx.extend_from_slice(cols);
        values.extend_from_slice(vals);
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n_local, global.ncols(), row_ptr, col_idx, values)
}

fn make_dist_poisson(comm: &UniverseComm, n_per: usize) -> DistCsrOp {
    let rank = comm.rank();
    let size = comm.size();
    let n_global = n_per * size;
    let row_start = rank * n_per;
    let global = fixtures::csr_poisson_1d(n_global);
    let local = local_rows_from_global(&global, row_start, n_per);
    let part_prefix: Vec<usize> = (0..=size).map(|p| p * n_per).collect();
    DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .expect("dist csr")
}

#[test]
fn mpi_mg_distributed_hierarchy_supports_cycle_variants_and_diagnostics() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    comm.set_reproducible(true);
    let dist = make_dist_poisson(&comm, 6);

    for cycle in ["v", "w", "f"] {
        let mut mg = MgPc::new(
            3,
            Some(cycle.to_string()),
            Some("jacobi".to_string()),
            Some(1),
            Some("linear".to_string()),
            Some("linear".to_string()),
            Some("full_weighting".to_string()),
            Some("ilu0".to_string()),
            None,
            None,
            None,
        );
        mg.setup(&dist).expect("mg setup");

        assert_eq!(mg.distributed_support(), PcDistributedSupport::Distributed);
        let rhs = vec![1.0; dist.local_nrows()];
        let mut y = vec![0.0; rhs.len()];
        mg.apply(PcSide::Left, &rhs, &mut y).expect("mg apply");
        assert!(y.iter().all(|v| v.is_finite()));

        let diagnostics = mg.diagnostics();
        assert!(diagnostics.len() >= 2, "expected multilevel diagnostics");
        assert!(
            diagnostics
                .iter()
                .all(|d| d.grid_complexity.is_finite() && d.operator_complexity.is_finite()),
            "diagnostics should report per-level complexity"
        );
    }
}
