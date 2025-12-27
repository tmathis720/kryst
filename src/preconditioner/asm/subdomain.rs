#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::algebra::prelude::*;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::error::KError;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::sparse::CsrMatrix;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::parallel::UniverseComm;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use std::collections::HashMap;

#[cfg(all(feature = "mpi", not(feature = "complex")))]
use super::comm_plan::alltoallv_u64;

#[cfg(all(feature = "mpi", not(feature = "complex")))]
#[derive(Debug, Clone)]
pub struct RemoteRow {
    pub cols: Vec<usize>,
    pub vals: Vec<R>,
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
pub fn request_remote_rows(
    comm: &UniverseComm,
    ownership: &[(usize, usize)],
    row_start: usize,
    row_end: usize,
    local: &CsrMatrix<f64>,
    requests: &[usize],
) -> Result<HashMap<usize, RemoteRow>, KError> {
    if requests.is_empty() {
        return Ok(HashMap::new());
    }

    let size = comm.size();
    let mut send = vec![Vec::<u64>::new(); size];
    for &row in requests {
        let owner = owner_of(row, ownership);
        if owner == comm.rank() {
            continue;
        }
        send[owner].push(row as u64);
    }

    let recv = alltoallv_u64(comm, &send)?;

    let mut responses = vec![Vec::<u64>::new(); size];
    for (peer, reqs) in recv.iter().enumerate() {
        if peer == comm.rank() {
            continue;
        }
        let mut buf = Vec::new();
        for &row_u64 in reqs {
            let row = row_u64 as usize;
            if row < row_start || row >= row_end {
                return Err(KError::InvalidInput(
                    "remote row request not owned by this rank".into(),
                ));
            }
            let local_row = row - row_start;
            let start = local.row_ptr()[local_row];
            let end = local.row_ptr()[local_row + 1];
            let cols = &local.col_idx()[start..end];
            let vals = &local.values()[start..end];
            pack_row(&mut buf, row, cols, vals);
        }
        responses[peer] = buf;
    }

    let recv_rows = alltoallv_u64(comm, &responses)?;
    let mut out = HashMap::new();
    for buf in recv_rows.into_iter() {
        if buf.is_empty() {
            continue;
        }
        unpack_rows(&buf, &mut out)?;
    }

    Ok(out)
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
pub fn build_subdomain_csr(
    subdofs: &[usize],
    row_start: usize,
    row_end: usize,
    local: &CsrMatrix<f64>,
    remote_rows: &HashMap<usize, RemoteRow>,
) -> Result<CsrMatrix<f64>, KError> {
    let n = subdofs.len();
    let mut map = HashMap::with_capacity(n * 2);
    for (i, &g) in subdofs.iter().enumerate() {
        map.insert(g, i);
    }

    let mut rowptr = Vec::with_capacity(n + 1);
    let mut colind = Vec::new();
    let mut values = Vec::new();
    rowptr.push(0);

    for &g in subdofs {
        if g >= row_start && g < row_end {
            let local_row = g - row_start;
            let start = local.row_ptr()[local_row];
            let end = local.row_ptr()[local_row + 1];
            for idx in start..end {
                let col = local.col_idx()[idx];
                if let Some(&lc) = map.get(&col) {
                    colind.push(lc);
                    values.push(local.values()[idx]);
                }
            }
        } else {
            let row = remote_rows.get(&g).ok_or_else(|| {
                KError::InvalidInput("missing remote row data for subdomain".into())
            })?;
            for (&col, &val) in row.cols.iter().zip(row.vals.iter()) {
                if let Some(&lc) = map.get(&col) {
                    colind.push(lc);
                    values.push(val);
                }
            }
        }
        rowptr.push(colind.len());
    }

    Ok(CsrMatrix::from_csr(n, n, rowptr, colind, values))
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
fn pack_row(dst: &mut Vec<u64>, row: usize, cols: &[usize], vals: &[R]) {
    dst.push(row as u64);
    dst.push(cols.len() as u64);
    for &col in cols {
        dst.push(col as u64);
    }
    for &val in vals {
        dst.push(val.to_bits());
    }
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
fn unpack_rows(buf: &[u64], out: &mut HashMap<usize, RemoteRow>) -> Result<(), KError> {
    let mut idx = 0;
    while idx < buf.len() {
        if idx + 2 > buf.len() {
            return Err(KError::InvalidInput("corrupt packed row buffer".into()));
        }
        let row = buf[idx] as usize;
        let nnz = buf[idx + 1] as usize;
        idx += 2;
        if idx + nnz > buf.len() {
            return Err(KError::InvalidInput("corrupt packed row buffer".into()));
        }
        let cols = buf[idx..idx + nnz]
            .iter()
            .map(|&v| v as usize)
            .collect::<Vec<_>>();
        idx += nnz;
        if idx + nnz > buf.len() {
            return Err(KError::InvalidInput("corrupt packed row buffer".into()));
        }
        let vals = buf[idx..idx + nnz]
            .iter()
            .map(|&v| f64::from_bits(v))
            .collect::<Vec<_>>();
        idx += nnz;
        out.insert(row, RemoteRow { cols, vals });
    }
    Ok(())
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
fn owner_of(g: usize, ownership: &[(usize, usize)]) -> usize {
    let mut lo = 0usize;
    let mut hi = ownership.len().saturating_sub(1);
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let (start, end) = ownership[mid];
        if g < start {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else if g >= end {
            lo = mid + 1;
        } else {
            return mid;
        }
    }
    lo.min(ownership.len().saturating_sub(1))
}
