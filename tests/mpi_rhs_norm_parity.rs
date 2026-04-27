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

#[test]
fn mpi_rhs_norm_report_parity_on_qc324_style_partition() {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let n_global = 324usize;
    let size = comm.size();
    let rank = comm.rank();

    let base = n_global / size;
    let rem = n_global % size;
    let local_n = base + usize::from(rank < rem);
    let row_start = rank * base + rank.min(rem);

    let rhs_global: Vec<f64> = (0..n_global).map(|i| 1.0 + (i as f64) * 1e-2).collect();
    let serial_norm = rhs_global.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rhs_local = &rhs_global[row_start..row_start + local_n];
    let mpi_norm = comm
        .all_reduce_f64(rhs_local.iter().map(|v| v * v).sum::<f64>())
        .sqrt();

    let serial_report = format!("rhs_norm_l2={serial_norm:.12e}");
    let mpi_report = format!("rhs_norm_l2={mpi_norm:.12e}");
    assert_eq!(
        serial_report, mpi_report,
        "reported norm text should stay stable across serial and MPI paths"
    );
}
