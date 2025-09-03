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
        x_owned: &[f64],
        send_buf: &'a mut [f64],
        recv_buf: &'a mut [f64],
    ) -> Vec<<UniverseComm as Comm>::Request<'a>> {
        assert_eq!(send_buf.len(), self.send_idx.len());
        assert_eq!(recv_buf.len(), self.recv_idx.len());

        let mut reqs: Vec<<UniverseComm as Comm>::Request<'a>> = Vec::new();

        // Post receives into disjoint slices of recv_buf
        let mut tail: &mut [f64] = recv_buf;
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
    pub fn unpack(&self, recv_buf: &[f64], x_ghost: &mut [f64]) {
        assert_eq!(recv_buf.len(), self.recv_idx.len());
        for (p, &idx) in self.recv_idx.iter().enumerate() {
            x_ghost[idx as usize] = recv_buf[p];
        }
    }
}
