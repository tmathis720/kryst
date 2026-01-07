#[cfg(feature = "mpi")]
use crate::algebra::prelude::*;
#[cfg(feature = "mpi")]
use crate::error::KError;
#[cfg(feature = "mpi")]
use crate::parallel::{Comm, UniverseComm};

#[cfg(feature = "mpi")]
#[derive(Debug, Clone)]
pub struct CommPlan {
    pub imports: Vec<Vec<usize>>,
    pub exports: Vec<Vec<usize>>,
    pub import_locs: Vec<Vec<usize>>,
}

#[cfg(feature = "mpi")]
impl CommPlan {
    pub fn exchange_values(
        &self,
        comm: &UniverseComm,
        row_start: usize,
        local: &[S],
    ) -> Result<Vec<Vec<S>>, KError> {
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

#[cfg(feature = "mpi")]
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

#[cfg(feature = "mpi")]
pub fn alltoallv_scalar(comm: &UniverseComm, send: &[Vec<S>]) -> Result<Vec<Vec<S>>, KError> {
    let size = comm.size();
    if send.len() != size {
        return Err(KError::InvalidInput(
            "alltoallv_scalar: send buffer length must match communicator size".into(),
        ));
    }
    let packed = pack_scalar_sends(send);
    let recv_packed = alltoallv_u64(comm, &packed)?;
    unpack_scalar_recvs(&recv_packed)
}

#[cfg(feature = "mpi")]
fn scalar_words() -> usize {
    #[cfg(feature = "complex")]
    {
        2
    }
    #[cfg(not(feature = "complex"))]
    {
        1
    }
}

#[cfg(feature = "mpi")]
fn pack_scalar_sends(send: &[Vec<S>]) -> Vec<Vec<u64>> {
    let words = scalar_words();
    send.iter()
        .map(|buf| {
            let mut packed = Vec::with_capacity(buf.len() * words);
            for &value in buf {
                pack_scalar(value, &mut packed);
            }
            packed
        })
        .collect()
}

#[cfg(feature = "mpi")]
fn unpack_scalar_recvs(recv: &[Vec<u64>]) -> Result<Vec<Vec<S>>, KError> {
    let words = scalar_words();
    let mut out = Vec::with_capacity(recv.len());
    for buf in recv {
        if buf.len() % words != 0 {
            return Err(KError::InvalidInput(
                "alltoallv_scalar: corrupt packed scalar buffer".into(),
            ));
        }
        let n = buf.len() / words;
        let mut scalars = Vec::with_capacity(n);
        for chunk in buf.chunks_exact(words) {
            scalars.push(unpack_scalar(chunk)?);
        }
        out.push(scalars);
    }
    Ok(out)
}

#[cfg(feature = "mpi")]
fn pack_scalar(value: S, dst: &mut Vec<u64>) {
    dst.push(value.real().to_bits());
    #[cfg(feature = "complex")]
    dst.push(value.imag().to_bits());
}

#[cfg(feature = "mpi")]
fn unpack_scalar(words: &[u64]) -> Result<S, KError> {
    #[cfg(feature = "complex")]
    {
        if words.len() != 2 {
            return Err(KError::InvalidInput(
                "packed scalar buffer length mismatch".into(),
            ));
        }
        Ok(S::from_parts(
            f64::from_bits(words[0]),
            f64::from_bits(words[1]),
        ))
    }
    #[cfg(not(feature = "complex"))]
    {
        if words.len() != 1 {
            return Err(KError::InvalidInput(
                "packed scalar buffer length mismatch".into(),
            ));
        }
        Ok(S::from_real(f64::from_bits(words[0])))
    }
}
