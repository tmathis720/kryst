#![cfg(not(feature = "complex"))]
#![cfg(feature = "mpi")]

mod fixtures;

use std::sync::Arc;

use kryst::parallel::{Comm, MpiComm, UniverseComm};

#[test]
fn mpi_rhs_norm_matches_serial_on_same_matrix() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));

    let n_per_rank = 7usize;
    let n_global = n_per_rank * comm.size();
    let matrix = fixtures::csr_poisson_1d(n_global);

    let rhs_global: Vec<f64> = (0..matrix.nrows())
        .map(|i| 1.0 + 0.125 * i as f64)
        .collect();
    let serial_norm = rhs_global.iter().map(|v| v * v).sum::<f64>().sqrt();

    let rank = comm.rank();
    let row_start = rank * n_per_rank;
    let row_end = row_start + n_per_rank;
    let rhs_local = &rhs_global[row_start..row_end];
    let rhs_norm2_local = rhs_local.iter().map(|v| v * v).sum::<f64>();
    let mpi_norm = comm.all_reduce_f64(rhs_norm2_local).sqrt();

    let tol = 1e-12 * serial_norm.max(1.0);
    assert!(
        (mpi_norm - serial_norm).abs() <= tol,
        "MPI-reduced rhs norm {mpi_norm:.16e} differs from serial {serial_norm:.16e} (tol={tol:.3e})"
    );
}
