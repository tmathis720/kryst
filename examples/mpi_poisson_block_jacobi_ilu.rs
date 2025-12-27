//! Distributed MPI block-Jacobi with local ILU(0) on a 1D Poisson grid.
//!
//! Each MPI rank builds its own subblock of the global 1D Poisson matrix, constructs
//! a local ILU(0) preconditioner, wraps it with a distributed block-Jacobi helper, and
//! applies the block preconditioner to a dummy right-hand side vector. The example
//! prints per-rank row ownership and the global norm of the preconditioned vector.
//!
//! Requires the `backend-faer`, `mpi`, and no `complex` feature. Run with:
//! ```bash
//! cargo mpirun -n 4 --example mpi_poisson_block_jacobi_ilu --features backend-faer,mpi
//! ```
#[cfg(feature = "complex")]
fn main() {
    eprintln!("mpi_poisson_block_jacobi_ilu.rs is unavailable when built with --features complex");
}


#[cfg(not(all(feature = "backend-faer", feature = "mpi", not(feature = "complex"))))]
#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "mpi_poisson_block_jacobi_ilu requires backend-faer + mpi + not(feature = \"complex\")"
    );
    Ok(())
}

#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use kryst::error::KError;
#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use kryst::matrix::CsrMatrix;
#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use kryst::matrix::dist_csr::DistCsrOp;
#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use kryst::parallel::{Comm, MpiComm, UniverseComm};
#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use kryst::preconditioner::PcSide;
#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use kryst::preconditioner::dist::{
    DistVec, GlobalPcKind, LocalPcKind, MpiPcOptions, build_block_jacobi_pc,
};
#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use kryst::preconditioner::ilu::{IluConfig, IluType};
#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use std::error::Error;
#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
use std::sync::Arc;

#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
#[cfg(not(feature = "complex"))]
fn build_local_poisson_block(n_global: usize, row_start: usize, row_end: usize) -> CsrMatrix<f64> {
    let n_local = row_end - row_start;
    let mut row_ptr = Vec::with_capacity(n_local + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for global_row in row_start..row_end {
        if global_row > 0 {
            col_idx.push(global_row - 1);
            values.push(-1.0);
        }
        col_idx.push(global_row);
        values.push(2.0);
        if global_row + 1 < n_global {
            col_idx.push(global_row + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }

    CsrMatrix::from_csr(n_local, n_global, row_ptr, col_idx, values)
}

#[cfg(all(feature = "backend-faer", feature = "mpi", not(feature = "complex")))]
#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn Error>> {
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::new()));
    let rank = comm.rank();
    let size = comm.size();

    let n_global = 64.max(size * 4);
    let base_rows = n_global / size;
    let extra = n_global % size;
    let mut part_prefix = vec![0usize; size + 1];
    for r in 0..size {
        let add = base_rows + if r < extra { 1 } else { 0 };
        part_prefix[r + 1] = part_prefix[r] + add;
    }

    let row_start = part_prefix[rank];
    let row_end = part_prefix[rank + 1];
    let local_csr = build_local_poisson_block(n_global, row_start, row_end);
    let dist_op =
        DistCsrOp::from_local_rows(n_global, row_start, &local_csr, &part_prefix, comm.clone())?;

    let mut mpi_opts = MpiPcOptions::default();
    mpi_opts.global_pc = GlobalPcKind::BlockJacobi;
    mpi_opts.local_pc = LocalPcKind::Ilu;
    let mut ilu_cfg = IluConfig::default();
    ilu_cfg.ilu_type = IluType::ILU0;
    mpi_opts.ilu_config = ilu_cfg;

    let block_pc = build_block_jacobi_pc(&dist_op, &mpi_opts)?
        .expect("block Jacobi wrapper should be configured");

    let mut dist_rhs = DistVec::new(
        comm.clone(),
        row_start,
        n_global,
        vec![1.0; row_end - row_start],
    );

    block_pc.apply_global(PcSide::Left, &mut dist_rhs)?;
    let local_norm_sq: f64 = dist_rhs.local_view().iter().copied().map(|v| v * v).sum();
    let global_norm_sq = comm.allreduce_sum_real(local_norm_sq);

    println!(
        "Rank {} owns rows {}..{} and preconditioned {} entries",
        rank,
        row_start,
        row_end,
        row_end - row_start
    );

    if rank == 0 {
        println!(
            "Global preconditioned vector norm: {:.6e}",
            global_norm_sq.sqrt()
        );
    }

    Ok(())
}
