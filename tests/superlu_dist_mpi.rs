#[cfg(feature = "mpi")]
use kryst::matrix::dist_csr::DistCsrOp;
#[cfg(feature = "mpi")]
use kryst::matrix::op::LinOp;
#[cfg(feature = "mpi")]
use kryst::matrix::parcsr::builder::partition_rows;
#[cfg(feature = "mpi")]
use kryst::matrix::sparse::CsrMatrix;
#[cfg(feature = "mpi")]
use kryst::parallel::{MpiComm, UniverseComm};
#[cfg(feature = "mpi")]
use kryst::solver::api::Solver;
#[cfg(feature = "mpi")]
use kryst::solver::superlu_dist::SuperLuDistSolver;
#[cfg(feature = "mpi")]
use std::env;
#[cfg(feature = "mpi")]
use std::sync::Arc;

#[cfg(feature = "mpi")]
fn mpi_comm() -> Option<UniverseComm> {
    if env::var("KRYST_ENABLE_MPI_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping MPI tests: KRYST_ENABLE_MPI_TESTS not set");
        return None;
    }
    let Some(comm) = MpiComm::try_new() else {
        eprintln!("skipping MPI tests: MPI init failed");
        return None;
    };
    Some(UniverseComm::Mpi(Arc::new(comm)))
}

#[test]
#[cfg(feature = "mpi")]
fn superlu_dist_solver_matches_diagonal_solution() {
    let Some(comm) = mpi_comm() else {
        return;
    };
    if comm.size() < 2 {
        eprintln!("skipping MPI test: need at least 2 ranks");
        return;
    }

    let n = comm.size() * 2;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    row_ptr.push(0);
    for i in 0..n {
        col_idx.push(i);
        vals.push(2.0);
        row_ptr.push(col_idx.len());
    }
    let a = CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals);

    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let mut x = vec![0.0; n];
    let mut solver = SuperLuDistSolver::new();
    solver.setup(&a, &comm).unwrap();
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x, &comm).unwrap();

    for (i, &xi) in x.iter().enumerate() {
        let expected = b[i] / 2.0;
        assert!(
            (xi - expected).abs() < 1e-10,
            "rank {}: x[{i}] = {xi}, expected {expected}",
            comm.rank()
        );
    }
}

#[test]
#[cfg(feature = "mpi")]
fn dist_csr_halo_exchange_matches_tridiagonal() {
    let Some(comm) = mpi_comm() else {
        return;
    };
    if comm.size() < 2 {
        eprintln!("skipping MPI test: need at least 2 ranks");
        return;
    }

    let n_global = comm.size() * 4;
    let part_prefix_u64 = partition_rows(n_global as u64, &comm);
    let part_prefix: Vec<usize> = part_prefix_u64.iter().map(|&v| v as usize).collect();
    let row_start = part_prefix[comm.rank()];
    let row_end = part_prefix[comm.rank() + 1];
    let n_local = row_end - row_start;

    let mut row_ptr = Vec::with_capacity(n_local + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for global_row in row_start..row_end {
        if global_row > 0 {
            col_idx.push(global_row - 1);
            vals.push(-1.0);
        }
        col_idx.push(global_row);
        vals.push(2.0);
        if global_row + 1 < n_global {
            col_idx.push(global_row + 1);
            vals.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    let local_rows = CsrMatrix::from_csr(n_local, n_global, row_ptr, col_idx, vals);
    let dist_op =
        DistCsrOp::from_local_rows(n_global, row_start, &local_rows, &part_prefix, comm.clone())
            .unwrap();

    let x_global: Vec<f64> = (0..n_global).map(|i| (i + 1) as f64).collect();
    let x_local = x_global[row_start..row_end].to_vec();
    let mut y_local = vec![0.0; n_local];
    dist_op.matvec(&x_local, &mut y_local);

    for (local_row, &yi) in y_local.iter().enumerate() {
        let global_row = row_start + local_row;
        let mut expected = 2.0 * x_global[global_row];
        if global_row > 0 {
            expected -= x_global[global_row - 1];
        }
        if global_row + 1 < n_global {
            expected -= x_global[global_row + 1];
        }
        assert!(
            (yi - expected).abs() < 1e-10,
            "rank {}: y[{global_row}] = {yi}, expected {expected}",
            comm.rank()
        );
    }
}
