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
        x_owned: &[S],
        send_buf: &'a mut [S],
        recv_buf: &'a mut [S],
    ) -> Vec<<UniverseComm as Comm>::Request<'a>> {
        assert_eq!(send_buf.len(), self.send_idx.len());
        assert_eq!(recv_buf.len(), self.recv_idx.len());

        let mut reqs: Vec<<UniverseComm as Comm>::Request<'a>> = Vec::new();

        // Post receives into disjoint slices of recv_buf
        let mut tail: &mut [S] = recv_buf;
        for (k, &nb) in self.neighbors.iter().enumerate() {
            let off = self.recv_ptr[k];
            let cnt = self.recv_ptr[k + 1] - off;
            if cnt > 0 {
                let (chunk, rest) = tail.split_at_mut(cnt);
                reqs.push(comm.irecv_from(halo_slice_mut(chunk), nb));
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
                reqs.push(comm.isend_to(halo_slice(&send_buf[off..off + cnt]), nb));
            }
        }

        reqs
    }

    /// Scatter the received buffer into the ghost slice.
    pub fn unpack(&self, recv_buf: &[S], x_ghost: &mut [S]) {
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

fn halo_slice(buf: &[S]) -> &[R] {
    #[cfg(feature = "complex")]
    {
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const R, buf.len() * 2) }
    }
    #[cfg(not(feature = "complex"))]
    {
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const R, buf.len()) }
    }
}

fn halo_slice_mut(buf: &mut [S]) -> &mut [R] {
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

    #[cfg(feature = "complex")]
    #[test]
    fn halo_pack_unpack_preserves_complex_values() {
        let halo = HaloPlan {
            neighbors: vec![1],
            send_ptr: vec![0, 2],
            send_idx: vec![0, 2],
            recv_ptr: vec![0, 2],
            recv_idx: vec![1, 0],
        };
        let x_owned = vec![
            S::from_parts(1.0, -1.0),
            S::from_parts(2.0, 0.5),
            S::from_parts(-3.0, 4.0),
        ];
        let mut send_buf = vec![S::zero(); 2];
        let mut recv_buf = vec![S::zero(); 2];
        let comm = UniverseComm::NoComm(NoComm);

        let _reqs = halo.begin_exchange(&comm, &x_owned, &mut send_buf, &mut recv_buf);
        assert_eq!(send_buf, vec![x_owned[0], x_owned[2]]);

        let recv_buf = vec![S::from_parts(5.0, -2.0), S::from_parts(-6.5, 1.25)];
        let mut x_ghost = vec![S::zero(); 2];
        halo.unpack(&recv_buf, &mut x_ghost);
        assert_eq!(x_ghost, vec![recv_buf[1], recv_buf[0]]);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn halo_slice_handles_complex_stride() {
        let buf = vec![S::from_parts(1.5, -2.0), S::from_parts(3.25, 4.5)];
        let slice = super::halo_slice(&buf);
        assert_eq!(slice, &[1.5, -2.0, 3.25, 4.5]);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn halo_slice_mut_writes_complex_stride() {
        let mut buf = vec![S::zero(); 2];
        {
            let slice = super::halo_slice_mut(&mut buf);
            slice.copy_from_slice(&[1.0, -1.0, 2.0, 3.0]);
        }
        assert_eq!(buf, vec![S::from_parts(1.0, -1.0), S::from_parts(2.0, 3.0)]);
    }
}
