#![cfg(all(feature = "cuda", feature = "mpi"))]

use kryst::algebra::prelude::*;
use kryst::context::ksp_context::SolverType;
use kryst::context::pc_context::PcType;
use kryst::cuda::{
    CudaDistCsrOp, CudaKspContext, CudaMpiTransport, CudaOptions, CudaRuntime, CudaVector,
};
use kryst::matrix::sparse::CsrMatrix;
use kryst::parallel::{Comm, MpiComm, UniverseComm};
use std::sync::Arc;

#[test]
#[ignore = "requires mpirun with one CUDA device per local rank"]
fn distributed_cuda_pcg_exchanges_halos_and_reduces_scalars() {
    let Some(mpi) = MpiComm::try_new() else {
        return;
    };
    let comm = UniverseComm::Mpi(Arc::new(mpi));
    if comm.size() < 2 {
        return;
    }
    let direct = std::env::var_os("KRYST_TEST_CUDA_AWARE_MPI").is_some();
    let runtime = CudaRuntime::for_local_rank(CudaOptions {
        mpi_transport: if direct {
            CudaMpiTransport::DeviceDirect
        } else {
            CudaMpiTransport::Staged
        },
        ..CudaOptions::default()
    })
    .expect("rank CUDA runtime");
    let n_local = 3usize;
    let n_global = n_local * comm.size();
    let part: Vec<usize> = (0..=comm.size()).map(|rank| rank * n_local).collect();
    let row_start = part[comm.rank()];

    let mut row_ptr = Vec::with_capacity(n_local + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for local_row in 0..n_local {
        let global_row = row_start + local_row;
        if global_row > 0 {
            col_idx.push(global_row - 1);
            values.push(S::from_real(-1.0));
        }
        col_idx.push(global_row);
        values.push(S::from_real(2.0));
        if global_row + 1 < n_global {
            col_idx.push(global_row + 1);
            values.push(S::from_real(-1.0));
        }
        row_ptr.push(col_idx.len());
    }
    let local_rows = CsrMatrix::from_csr(n_local, n_global, row_ptr, col_idx, values);
    let operator = Arc::new(
        CudaDistCsrOp::from_local_rows(
            runtime.clone(),
            n_global,
            row_start,
            &local_rows,
            &part,
            comm.clone(),
        )
        .unwrap(),
    );
    assert_eq!(
        operator.transport(),
        if direct {
            CudaMpiTransport::DeviceDirect
        } else {
            CudaMpiTransport::Staged
        }
    );

    let rhs_host: Vec<S> = (0..n_local)
        .map(|local_row| {
            let global_row = row_start + local_row;
            S::from_real(if global_row == 0 || global_row + 1 == n_global {
                1.0
            } else {
                0.0
            })
        })
        .collect();
    let rhs = CudaVector::from_host(runtime.clone(), &rhs_host).unwrap();
    let mut solution = CudaVector::zeros(runtime.clone(), n_local).unwrap();
    let mut ksp = CudaKspContext::new(runtime);
    ksp.set_type(SolverType::Pcg).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    ksp.set_tolerances(1e-11, 1e-13, 1e8, n_global * 3).unwrap();
    ksp.set_operators(operator, None).unwrap();
    let stats = ksp.solve(&rhs, &mut solution).unwrap();
    assert!(stats.reason.is_converged(), "{stats:?}");
    for value in solution.to_host().unwrap() {
        assert!((value - S::one()).abs() < 1e-9, "{value:?}");
    }
}
