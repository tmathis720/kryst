#![cfg(all(feature = "backend-faer", not(feature = "complex")))]

use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use std::sync::Arc;

fn nonsym_matrix(n: usize) -> CsrMatrix<f64> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(3 * n);
    let mut values = Vec::with_capacity(3 * n);
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(-1.0);
        }
        col_idx.push(i);
        values.push(4.0);
        if i + 1 < n {
            col_idx.push(i + 1);
            values.push(-2.0);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}

#[test]
fn gmres_reduction_counts_within_bounds() {
    let comm = UniverseComm::NoComm(NoComm);
    let a = nonsym_matrix(64);
    let op = CsrOp::new(Arc::new(a)).with_comm(comm.clone());
    let rhs = vec![1.0f64; 64];
    let mut x = vec![0.0f64; 64];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.try_set_pc_side(PcSide::Left).unwrap();
    ksp.set_restart(12);
    ksp.set_tolerances(1e-10, 0.0, 1e6, 200);
    ksp.set_operators(Arc::new(op), None);

    let stats = ksp.solve(&rhs, &mut x).unwrap();
    let reported = stats.counters.num_global_reductions;
    if reported > 0 {
        assert!(
            reported >= stats.iterations,
            "reported reductions {reported} < iterations {}",
            stats.iterations
        );
        let upper_bound = 2 * stats.iterations + ksp.restart + 8;
        assert!(
            reported <= upper_bound,
            "reported reductions {reported} exceeds upper bound {upper_bound} (iters={}, restart={})",
            stats.iterations,
            ksp.restart
        );
    }
}
