#[cfg(feature = "mpi")]
use std::sync::Arc;

use kryst::matrix::LinOp;
use kryst::algebra::prelude::*;
use kryst::algebra::scalar::S;
use kryst::matrix::{CsrMatrix, DistCsrOp};
#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;
use kryst::parallel::{Comm, NoComm, UniverseComm};

#[test]
fn dist_csr_spmv_matches_serial() {
    #[cfg(feature = "mpi")]
    let comm = UniverseComm::Mpi(Arc::new(MpiComm::try_new().unwrap()));
    #[cfg(not(feature = "mpi"))]
    let comm = UniverseComm::NoComm(NoComm);

    let rank = comm.rank();
    let size = comm.size();
    let n_per = 4;
    let n_global = n_per * size;
    let row_start = rank * n_per;

    let mut row_ptr = Vec::with_capacity(n_per + 1);
    row_ptr.push(0);
    let mut col_idx = Vec::new();
    let mut values: Vec<S> = Vec::new();
    for i in 0..n_per {
        let g = row_start + i;
        if g > 0 {
            col_idx.push(g - 1);
            values.push(S::from_real(-1.0));
        }
        col_idx.push(g);
        values.push(S::from_real(2.0));
        if g + 1 < n_global {
            col_idx.push(g + 1);
            values.push(S::from_real(-1.0));
        }
        row_ptr.push(col_idx.len());
    }
    let local = CsrMatrix::from_csr(n_per, n_global, row_ptr, col_idx, values);
    let part_prefix: Vec<usize> = (0..=size).map(|p| p * n_per).collect();
    let op = DistCsrOp::from_local_rows(n_global, row_start, &local, &part_prefix, comm.clone())
        .unwrap();

    let x_global: Vec<S> = (0..n_global).map(|i| S::from_real(i as f64)).collect();
    let x_local = x_global[row_start..row_start + n_per].to_vec();
    let mut y_local = vec![S::zero(); n_per];
    op.matvec(&x_local, &mut y_local);

    let packed_local = pack_scalars(&y_local);
    let mut packed_global = Vec::new();
    comm.gather(&packed_local, &mut packed_global, 0);
    if rank == 0 {
        let y_global = unpack_scalars(&packed_global);
        let mut y_ref = vec![S::zero(); n_global];
        for i in 0..n_global {
            let mut v = S::from_real(2.0) * x_global[i];
            if i > 0 {
                v = v - x_global[i - 1];
            }
            if i + 1 < n_global {
                v = v - x_global[i + 1];
            }
            y_ref[i] = v;
        }
        assert_eq!(y_global, y_ref);
    }
}

#[test]
fn dist_csr_numeric_update_changes_values_id() {
    let comm = UniverseComm::NoComm(NoComm);
    let part_prefix = vec![0, 1];
    let local = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(1.0)]);
    let mut op = DistCsrOp::from_local_rows(1, 0, &local, &part_prefix, comm).unwrap();
    let sid = op.structure_id();
    let vid = op.values_id();
    op.update_numeric(&[S::from_real(2.0)]).unwrap();
    assert_eq!(op.structure_id(), sid);
    assert_ne!(op.values_id(), vid);
}

#[cfg(feature = "complex")]
fn pack_scalars(values: &[S]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.push(v.real());
        out.push(v.imag());
    }
    out
}

#[cfg(not(feature = "complex"))]
fn pack_scalars(values: &[S]) -> Vec<f64> {
    values.iter().map(|&v| v.real()).collect()
}

#[cfg(feature = "complex")]
fn unpack_scalars(data: &[f64]) -> Vec<S> {
    data.chunks_exact(2)
        .map(|chunk| S::from_parts(chunk[0], chunk[1]))
        .collect()
}

#[cfg(not(feature = "complex"))]
fn unpack_scalars(data: &[f64]) -> Vec<S> {
    data.iter().map(|&v| S::from_real(v)).collect()
}
