#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::algebra::prelude::*;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::error::KError;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::parallel::{Comm, UniverseComm};

#[cfg(all(feature = "mpi", not(feature = "complex")))]
#[derive(Debug, Clone)]
pub struct CommPlan {
    pub imports: Vec<Vec<usize>>,
    pub exports: Vec<Vec<usize>>,
    pub import_locs: Vec<Vec<usize>>,
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
impl CommPlan {
    pub fn exchange_values(
        &self,
        comm: &UniverseComm,
        row_start: usize,
        local: &[R],
    ) -> Result<Vec<Vec<R>>, KError> {
        let mut send = Vec::with_capacity(self.exports.len());
        for export in &self.exports {
            let mut buf = Vec::with_capacity(export.len());
            for &g in export {
                let li = g - row_start;
                buf.push(local[li]);
            }
            send.push(buf);
        }
        alltoallv_scalar(comm, &send)
    }
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
pub fn alltoallv_u64(comm: &UniverseComm, send: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, KError> {
    let size = comm.size();
    if send.len() != size {
        return Err(KError::InvalidInput(
            "alltoallv_u64: send buffer length must match communicator size".into(),
        ));
    }
    let rank = comm.rank();

    let mut recv_counts = vec![0u64; size];
    let mut send_counts = vec![0u64; size];
    for (slot, buf) in send_counts.iter_mut().zip(send.iter()) {
        *slot = buf.len() as u64;
    }

    let mut recv_count_bufs = vec![[0u64; 1]; size];
    let count_bufs: Vec<[u64; 1]> = send_counts.iter().map(|&count| [count]).collect();
    let mut reqs = Vec::new();
    for peer in 0..size {
        if peer == rank {
            recv_counts[peer] = send_counts[peer];
            continue;
        }
        let buf = unsafe { &mut *recv_count_bufs.as_mut_ptr().add(peer) };
        reqs.push(comm.irecv_from_u64(buf, peer as i32));
    }
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        reqs.push(comm.isend_to_u64(&count_bufs[peer], peer as i32));
    }
    comm.wait_all(&mut reqs);
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        recv_counts[peer] = recv_count_bufs[peer][0];
    }

    let mut recv = vec![Vec::new(); size];
    let mut reqs = Vec::new();
    for peer in 0..size {
        if peer == rank {
            recv[peer] = send[peer].clone();
            continue;
        }
        let count = recv_counts[peer] as usize;
        recv[peer] = vec![0u64; count];
    }
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        let buf = unsafe { &mut *recv.as_mut_ptr().add(peer) };
        reqs.push(comm.irecv_from_u64(buf, peer as i32));
    }
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        reqs.push(comm.isend_to_u64(&send[peer], peer as i32));
    }
    comm.wait_all(&mut reqs);

    Ok(recv)
}

#[cfg(all(feature = "mpi", not(feature = "complex")))]
pub fn alltoallv_scalar(comm: &UniverseComm, send: &[Vec<R>]) -> Result<Vec<Vec<R>>, KError> {
    let size = comm.size();
    if send.len() != size {
        return Err(KError::InvalidInput(
            "alltoallv_scalar: send buffer length must match communicator size".into(),
        ));
    }
    let rank = comm.rank();

    let mut recv_counts = vec![0u64; size];
    let mut send_counts = vec![0u64; size];
    for (slot, buf) in send_counts.iter_mut().zip(send.iter()) {
        *slot = buf.len() as u64;
    }

    let mut recv_count_bufs = vec![[0u64; 1]; size];
    let count_bufs: Vec<[u64; 1]> = send_counts.iter().map(|&count| [count]).collect();
    let mut reqs = Vec::new();
    for peer in 0..size {
        if peer == rank {
            recv_counts[peer] = send_counts[peer];
            continue;
        }
        let buf = unsafe { &mut *recv_count_bufs.as_mut_ptr().add(peer) };
        reqs.push(comm.irecv_from_u64(buf, peer as i32));
    }
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        reqs.push(comm.isend_to_u64(&count_bufs[peer], peer as i32));
    }
    comm.wait_all(&mut reqs);
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        recv_counts[peer] = recv_count_bufs[peer][0];
    }

    let mut recv = vec![Vec::new(); size];
    let mut reqs = Vec::new();
    for peer in 0..size {
        if peer == rank {
            recv[peer] = send[peer].clone();
            continue;
        }
        let count = recv_counts[peer] as usize;
        recv[peer] = vec![R::zero(); count];
    }
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        let buf = unsafe { &mut *recv.as_mut_ptr().add(peer) };
        reqs.push(comm.irecv_from(buf, peer as i32));
    }
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        reqs.push(comm.isend_to(&send[peer], peer as i32));
    }
    comm.wait_all(&mut reqs);

    Ok(recv)
}
