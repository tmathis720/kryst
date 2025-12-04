use crate::algebra::prelude::*;
use crate::matrix::dist::halo::HaloIndexPlan;
use crate::parallel::{Comm, UniverseComm};

/// Communication plan for halo exchanges in distributed matrices.
#[derive(Debug, Clone)]
pub struct HaloPlan {
    /// Ranks we communicate with. Order is consistent for send/recv.
    pub neighbors: Vec<i32>,
    /// CSR-style pointer into `send_idx` for each neighbor.
    pub send_ptr: Vec<usize>,
    /// Local indices of owned entries that need to be sent.
    pub send_idx: Vec<u64>,
    /// CSR-style pointer into `recv_idx` for each neighbor.
    pub recv_ptr: Vec<usize>,
    /// Positions in the ghost slice where received values should be unpacked.
    pub recv_idx: Vec<u64>,
}

impl Default for HaloPlan {
    fn default() -> Self {
        Self {
            neighbors: Vec::new(),
            send_ptr: vec![0],
            send_idx: Vec::new(),
            recv_ptr: vec![0],
            recv_idx: Vec::new(),
        }
    }
}

impl HaloPlan {
    /// Start nonblocking halo exchange.
    ///
    /// `x_owned` holds the vector owned by the current rank. `send_buf` and
    /// `recv_buf` must have length matching `send_idx` and `recv_idx`.
    pub fn begin_exchange<'a>(
        &'a self,
        comm: &'a UniverseComm,
        x_owned: &[R],
        send_buf: &'a mut [R],
        recv_buf: &'a mut [R],
    ) -> Vec<<UniverseComm as Comm>::Request<'a>> {
        assert_eq!(send_buf.len(), self.send_idx.len());
        assert_eq!(recv_buf.len(), self.recv_idx.len());

        let mut reqs: Vec<<UniverseComm as Comm>::Request<'a>> = Vec::new();

        // Post receives into disjoint slices of recv_buf
        let mut tail: &mut [R] = recv_buf;
        for (k, &nb) in self.neighbors.iter().enumerate() {
            let off = self.recv_ptr[k];
            let cnt = self.recv_ptr[k + 1] - off;
            if cnt > 0 {
                let (chunk, rest) = tail.split_at_mut(cnt);
                reqs.push(comm.irecv_from(chunk, nb));
                tail = rest;
            }
        }

        // Pack and send owned entries needed by neighbors
        for (p, &idx) in self.send_idx.iter().enumerate() {
            send_buf[p] = x_owned[idx as usize];
        }
        for (k, &nb) in self.neighbors.iter().enumerate() {
            let off = self.send_ptr[k];
            let cnt = self.send_ptr[k + 1] - off;
            if cnt > 0 {
                reqs.push(comm.isend_to(&send_buf[off..off + cnt], nb));
            }
        }

        reqs
    }

    /// Scatter the received buffer into the ghost slice.
    pub fn unpack(&self, recv_buf: &[R], x_ghost: &mut [R]) {
        assert_eq!(recv_buf.len(), self.recv_idx.len());
        for (p, &idx) in self.recv_idx.iter().enumerate() {
            x_ghost[idx as usize] = recv_buf[p];
        }
    }
}

impl From<&HaloIndexPlan> for HaloPlan {
    fn from(plan: &HaloIndexPlan) -> Self {
        let mut neighbors: Vec<i32> = plan
            .send_local_idx
            .keys()
            .chain(plan.recv_map.keys())
            .map(|&r| r as i32)
            .collect();
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors.retain(|&r| r != plan.rank as i32);

        let mut send_ptr = Vec::with_capacity(neighbors.len() + 1);
        let mut send_idx = Vec::new();
        send_ptr.push(0);
        for &nbr in &neighbors {
            if let Some(local_idxs) = plan.send_local_idx.get(&(nbr as usize)) {
                for &idx in local_idxs {
                    send_idx.push(idx as u64);
                }
            }
            send_ptr.push(send_idx.len());
        }

        let mut recv_ptr = Vec::with_capacity(neighbors.len() + 1);
        let mut recv_idx = Vec::new();
        recv_ptr.push(0);
        for &nbr in &neighbors {
            if let Some(cols) = plan.recv_map.get(&(nbr as usize)) {
                for &gcol in cols {
                    let ghost_pos = *plan
                        .ghost_index_of
                        .get(&gcol)
                        .expect("ghost_index_of must cover recv_map");
                    recv_idx.push(ghost_pos as u64);
                }
            }
            recv_ptr.push(recv_idx.len());
        }

        HaloPlan {
            neighbors,
            send_ptr,
            send_idx,
            recv_ptr,
            recv_idx,
        }
    }
}
