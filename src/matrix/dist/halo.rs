use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, RwLock};

use crate::algebra::scalar::{KrystScalar, R, S};
use crate::error::KError;
use crate::parallel::{Comm, UniverseComm};

pub struct HaloReq<'a> {
    pub recv_reqs: Vec<<UniverseComm as Comm>::Request<'a>>,
    pub send_reqs: Vec<<UniverseComm as Comm>::Request<'a>>,
}

pub struct HaloIndexPlan {
    pub comm: UniverseComm,
    pub rank: usize,
    pub size: usize,
    pub row_part: Arc<Vec<usize>>,
    pub row_start: usize,
    pub row_end: usize,
    pub n_local: usize,
    pub recv_map: BTreeMap<usize, Vec<usize>>,
    pub send_map: BTreeMap<usize, Vec<usize>>,
    pub send_local_idx: BTreeMap<usize, Vec<usize>>,
    pub ghost_index_of: BTreeMap<usize, usize>,
    pub ghost_ranges: BTreeMap<usize, Range<usize>>,
    pub n_ghost: usize,
}

impl HaloIndexPlan {
    pub fn new(
        comm: UniverseComm,
        row_part: Arc<Vec<usize>>,
        row_start: usize,
        row_end: usize,
        mut recv_map: BTreeMap<usize, Vec<usize>>,
    ) -> Result<Self, KError> {
        let rank = comm.rank();
        let size = comm.size();
        let n_local = row_end - row_start;

        // Clean up and deduplicate the recv map before the handshake.
        recv_map.retain(|&_nbr, cols| {
            if cols.is_empty() {
                return false;
            }
            cols.sort_unstable();
            cols.dedup();
            true
        });

        let mut counts_out = vec![0u64; size];
        for (&nbr, cols) in recv_map.iter() {
            if nbr >= size {
                return Err(KError::InvalidInput(format!(
                    "neighbor rank {nbr} out of bounds for size {size}"
                )));
            }
            if nbr == rank {
                return Err(KError::InvalidInput(
                    "recv_map contains the local rank".to_string(),
                ));
            }
            counts_out[nbr] = cols.len() as u64;
        }

        // Exchange the list lengths with every other rank.
        let mut counts_in = vec![0u64; size];
        let peers: Vec<usize> = (0..size).filter(|&r| r != rank).collect();
        if !peers.is_empty() {
            let mut reqs: Vec<<UniverseComm as Comm>::Request<'_>> = Vec::new();
            let mut counts_in_buf = vec![0u64; peers.len()];
            {
                let mut tail: &mut [u64] = counts_in_buf.as_mut_slice();
                for &r in &peers {
                    let (chunk, rest) = tail.split_at_mut(1);
                    reqs.push(comm.irecv_from_u64(chunk, r as i32));
                    tail = rest;
                }
            }
            for &r in &peers {
                reqs.push(comm.isend_to_u64(std::slice::from_ref(&counts_out[r]), r as i32));
            }
            comm.wait_all(&mut reqs);
            for (i, &r) in peers.iter().enumerate() {
                counts_in[r] = counts_in_buf[i];
            }
        }

        // Receive the explicit column lists from neighbors and send ours once.
        let mut send_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        if size > 1 {
            let neighbors: Vec<usize> = (0..size)
                .filter(|&r| r != rank && (counts_out[r] > 0 || counts_in[r] > 0))
                .collect();

            // Prepare buffers for incoming requests.
            let mut their_needs: Vec<Vec<u64>> = neighbors
                .iter()
                .map(|&r| vec![0u64; counts_in[r] as usize])
                .collect();

            let mut reqs: Vec<<UniverseComm as Comm>::Request<'_>> = Vec::new();
            for (buf, &r) in their_needs.iter_mut().zip(neighbors.iter()) {
                if !buf.is_empty() {
                    reqs.push(comm.irecv_from_u64(buf.as_mut_slice(), r as i32));
                }
            }

            // Send our requirements to neighbors (keep temporary storage until completion).
            let mut tmp_sends: Vec<Vec<u64>> = Vec::with_capacity(neighbors.len());
            for &r in &neighbors {
                let cols = recv_map.get(&r).map(|v| v.as_slice()).unwrap_or(&[]);
                if cols.is_empty() {
                    tmp_sends.push(Vec::new());
                } else {
                    tmp_sends.push(cols.iter().map(|&c| c as u64).collect());
                }
            }
            for (buf, &r) in tmp_sends.iter().zip(neighbors.iter()) {
                if !buf.is_empty() {
                    reqs.push(comm.isend_to_u64(buf.as_slice(), r as i32));
                }
            }
            comm.wait_all(&mut reqs);

            for (k, &r) in neighbors.iter().enumerate() {
                let mut list: Vec<usize> = their_needs[k].iter().map(|&c| c as usize).collect();
                list.sort_unstable();
                list.dedup();
                if !list.is_empty() {
                    send_map.insert(r, list);
                }
            }
        }

        let mut send_local_idx: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (&nbr, cols) in &send_map {
            let mut local_idx = Vec::with_capacity(cols.len());
            for &g in cols {
                if g < row_start || g >= row_end {
                    return Err(KError::InvalidInput(format!(
                        "neighbor {nbr} requested global column {g} not owned by rank {rank}"
                    )));
                }
                local_idx.push(g - row_start);
            }
            send_local_idx.insert(nbr, local_idx);
        }

        let mut ghost_index_of: BTreeMap<usize, usize> = BTreeMap::new();
        let mut ghost_ranges: BTreeMap<usize, Range<usize>> = BTreeMap::new();
        let mut n_ghost = 0;
        for (&nbr, cols) in &recv_map {
            if cols.is_empty() {
                continue;
            }
            let start = n_ghost;
            for &g in cols {
                ghost_index_of.insert(g, n_ghost);
                n_ghost += 1;
            }
            let end = n_ghost;
            ghost_ranges.insert(nbr, start..end);
        }

        Ok(Self {
            comm,
            rank,
            size,
            row_part,
            row_start,
            row_end,
            n_local,
            recv_map,
            send_map,
            send_local_idx,
            ghost_index_of,
            ghost_ranges,
            n_ghost,
        })
    }
}

pub struct HaloBuffers {
    pub send_buf: BTreeMap<usize, UnsafeCell<Vec<S>>>,
    pub recv_buf: BTreeMap<usize, UnsafeCell<Vec<S>>>,
    pub ghost_flat: RwLock<Vec<S>>,
}

impl HaloBuffers {
    pub fn new(plan: &HaloIndexPlan) -> Self {
        let mut send_buf = BTreeMap::new();
        for (&nbr, cols) in &plan.send_map {
            let mut buf = Vec::with_capacity(cols.len());
            buf.resize(cols.len(), S::zero());
            send_buf.insert(nbr, UnsafeCell::new(buf));
        }

        let mut recv_buf = BTreeMap::new();
        for (&nbr, cols) in &plan.recv_map {
            let mut buf = Vec::with_capacity(cols.len());
            buf.resize(cols.len(), S::zero());
            recv_buf.insert(nbr, UnsafeCell::new(buf));
        }

        let ghost_flat = RwLock::new(vec![S::zero(); plan.n_ghost]);

        Self {
            send_buf,
            recv_buf,
            ghost_flat,
        }
    }
}

/// Halo exchange plan for distributed CSR matvecs.
///
/// # Thread-safety
/// - `HaloPlan` is `Sync` but assumes `post_halo` / `complete_halo` are invoked
///   sequentially on a single thread per matvec; they must not be called from
///   multiple threads concurrently.
/// - The buffers guarded by this plan are never accessed from Rayon callbacks
///   while a matvec is running.
pub struct HaloPlan {
    pub index: Arc<HaloIndexPlan>,
    buffers: HaloBuffers,
}

// SAFETY: HaloPlan is shared immutably and all mutation occurs through the
// sequential `post_halo` / `complete_halo` calls on a single matvec. Rayon
// parallel regions never touch `send_buf`/`recv_buf`/`ghost_flat`, so the plan
// can safely be shared while still requiring external synchronization.
unsafe impl Sync for HaloPlan {}

impl HaloPlan {
    pub fn new(
        comm: UniverseComm,
        row_part: Arc<Vec<usize>>,
        row_start: usize,
        row_end: usize,
        recv_map: BTreeMap<usize, Vec<usize>>,
    ) -> Result<Self, KError> {
        let index = Arc::new(HaloIndexPlan::new(
            comm, row_part, row_start, row_end, recv_map,
        )?);
        let buffers = HaloBuffers::new(&index);
        Ok(Self { index, buffers })
    }

    pub fn ghost_slice_ref(&self) -> std::sync::RwLockReadGuard<'_, Vec<S>> {
        self.buffers.ghost_flat.read().unwrap()
    }

    pub fn post_halo<'a>(&'a self, x_local: &[S]) -> HaloReq<'a> {
        let mut recv_reqs = Vec::new();
        for (&nbr, cols) in &self.index.recv_map {
            if cols.is_empty() {
                continue;
            }
            if let Some(buf_lock) = self.buffers.recv_buf.get(&nbr) {
                let buf = unsafe { &mut *buf_lock.get() };
                let slice = halo_slice_mut(buf);
                let req = self.index.comm.irecv_from(slice, nbr as i32);
        recv_reqs.push(req);
            }
        }

        let mut send_reqs = Vec::new();
        for (&nbr, buf_lock) in &self.buffers.send_buf {
            if let Some(idxs) = self.index.send_local_idx.get(&nbr) {
                let buf = unsafe { &mut *buf_lock.get() };
                if buf.is_empty() {
                    continue;
                }
                for (dst, &idx_val) in buf.iter_mut().zip(idxs.iter()) {
                    *dst = x_local[idx_val];
                }
                let slice = halo_slice(buf);
                let req = self.index.comm.isend_to(slice, nbr as i32);
                send_reqs.push(req);
            }
        }

        HaloReq {
            recv_reqs,
            send_reqs,
        }
    }

    pub fn complete_halo(&self, mut req: HaloReq<'_>) {
        self.index.comm.wait_all(&mut req.recv_reqs);
        self.index.comm.wait_all(&mut req.send_reqs);

        if self.index.n_ghost > 0 {
            let mut ghost = self.buffers.ghost_flat.write().unwrap();
            for (&nbr, range) in &self.index.ghost_ranges {
                if range.is_empty() {
                    continue;
                }
                if let Some(buf_lock) = self.buffers.recv_buf.get(&nbr) {
                    let src = unsafe { &*buf_lock.get() };
                    ghost[range.clone()].copy_from_slice(src);
                }
            }
        }
    }
}

fn halo_slice(buf: &Vec<S>) -> &[R] {
    #[cfg(feature = "complex")]
    {
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const R, buf.len() * 2) }
    }
    #[cfg(not(feature = "complex"))]
    {
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const R, buf.len()) }
    }
}

fn halo_slice_mut(buf: &mut Vec<S>) -> &mut [R] {
    #[cfg(feature = "complex")]
    {
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut R, buf.len() * 2) }
    }
    #[cfg(not(feature = "complex"))]
    {
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut R, buf.len()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::{NoComm, UniverseComm};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn halo_plan_rejects_local_neighbor() {
        let comm = UniverseComm::NoComm(NoComm);
        let row_part = Arc::new(vec![0usize, 4usize]);
        let mut recv_map = BTreeMap::new();
        recv_map.insert(0, vec![1, 2]);
        let res = HaloPlan::new(comm, row_part, 0, 4, recv_map);
        assert!(matches!(res, Err(KError::InvalidInput(_))));
        if let Err(KError::InvalidInput(msg)) = res {
            assert!(msg.contains("local rank"));
        }
    }

    #[test]
    fn halo_plan_rejects_out_of_bounds_neighbor() {
        let comm = UniverseComm::NoComm(NoComm);
        let row_part = Arc::new(vec![0usize, 4usize]);
        let mut recv_map = BTreeMap::new();
        recv_map.insert(5, vec![8]);
        let res = HaloPlan::new(comm, row_part, 0, 4, recv_map);
        assert!(matches!(res, Err(KError::InvalidInput(_))));
        if let Err(KError::InvalidInput(msg)) = res {
            assert!(msg.contains("neighbor rank 5 out of bounds"))
        }
    }
}
