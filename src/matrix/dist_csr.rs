use std::any::Any;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::KError;
use crate::matrix::{
    op::{ChangeIds, LinOp, StructureId, ValuesId},
    sparse::CsrMatrix,
};
use crate::parallel::{Comm, UniverseComm};

/// Distributed CSR operator with halo exchange
pub struct DistCsrOp {
    pub n_global: usize,
    pub row_start: usize,
    pub row_end: usize,
    pub a_on: CsrMatrix<f64>,
    pub a_off: CsrMatrix<f64>,
    pub n_local: usize,
    pub n_halo: usize,
    pub g2l: HashMap<usize, usize>,
    pub neighbors: Vec<i32>,
    pub recv_idx: Vec<usize>,
    pub recv_disp: Vec<usize>,
    pub send_idx: Vec<usize>,
    pub send_disp: Vec<usize>,
    pub x_halo: Mutex<Vec<f64>>,
    pub send_buf: Mutex<Vec<f64>>,
    pub recv_buf: Mutex<Vec<f64>>,
    ids: ChangeIds,
    comm: UniverseComm,
}

fn owner_of_row(i: usize, part_prefix: &[usize]) -> usize {
    let mut lo = 0usize;
    let mut hi = part_prefix.len() - 2; // last interval index
    while lo <= hi {
        let mid = (lo + hi) / 2;
        if i < part_prefix[mid + 1] {
            if i >= part_prefix[mid] {
                return mid;
            }
            if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        } else {
            lo = mid + 1;
        }
    }
    lo
}

impl DistCsrOp {
    /// Build from local row block of the global matrix
    pub fn from_local_rows(
        n_global: usize,
        row_start: usize,
        local_rows: &CsrMatrix<f64>,
        part_prefix: &[usize],
        comm: UniverseComm,
    ) -> Result<Self, KError> {
        let row_end = row_start + local_rows.nrows();
        let n_local = local_rows.nrows();
        let my_rank = comm.rank();
        let size = comm.size();

        let rp = local_rows.row_ptr();
        let ci = local_rows.col_idx();
        let vv = local_rows.values();

        // First pass: classify entries and record needed remote columns
        let mut local_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_local];
        let mut remote_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_local];
        let mut need_from: Vec<Vec<usize>> = vec![Vec::new(); size];
        for i in 0..n_local {
            for idx in rp[i]..rp[i + 1] {
                let gcol = ci[idx];
                let owner = owner_of_row(gcol, part_prefix);
                let val = vv[idx];
                if owner == my_rank {
                    local_entries[i].push((gcol, val));
                } else {
                    remote_entries[i].push((gcol, val));
                    need_from[owner].push(gcol);
                }
            }
        }
        // deduplicate recv lists
        for cols in need_from.iter_mut() {
            cols.sort_unstable();
            cols.dedup();
        }
        // ---- Phase A: counts exchange with all ranks ----
        let mut counts_out: Vec<u64> = vec![0; size];
        for r in 0..size {
            if r != my_rank {
                counts_out[r] = need_from[r].len() as u64;
            }
        }
        let mut counts_in: Vec<u64> = vec![0; size];
        let peers: Vec<usize> = (0..size).filter(|&r| r != my_rank).collect();
        {
            let mut reqs_counts: Vec<<UniverseComm as Comm>::Request<'_>> = Vec::new();
            let mut counts_in_buf: Vec<u64> = vec![0; peers.len()];
            // Post receives into disjoint 1-word slices using split_at_mut
            {
                let mut tail: &mut [u64] = counts_in_buf.as_mut_slice();
                for &r in &peers {
                    let (chunk, rest) = tail.split_at_mut(1);
                    reqs_counts.push(comm.irecv_from_u64(chunk, r as i32));
                    tail = rest;
                }
            }
            // Send our counts to peers
            for &r in &peers {
                reqs_counts.push(comm.isend_to_u64(std::slice::from_ref(&counts_out[r]), r as i32));
            }
            comm.wait_all(&mut reqs_counts);
            for (i, &r) in peers.iter().enumerate() {
                counts_in[r] = counts_in_buf[i];
            }
        }

        // Union neighbors: ranks we need from OR ranks that need from us
        let mut neighbors: Vec<i32> = Vec::new();
        for r in 0..size {
            if r == my_rank {
                continue;
            }
            if counts_out[r] > 0 || counts_in[r] > 0 {
                neighbors.push(r as i32);
            }
        }
        // assign halo indices and build g2l mapping
        let mut g2l: HashMap<usize, usize> = HashMap::new();
        for j in row_start..row_end {
            g2l.insert(j, j - row_start);
        }
        let mut recv_idx = Vec::new();
        let mut recv_disp = Vec::with_capacity(neighbors.len() + 1);
        recv_disp.push(0);
        let mut halo = n_local;
        for &nb in &neighbors {
            let cols = &need_from[nb as usize];
            for &gcol in cols {
                g2l.insert(gcol, halo);
                halo += 1;
                recv_idx.push(gcol);
            }
            recv_disp.push(recv_idx.len());
        }
        let n_halo = halo - n_local;

        // ---- Phase B: exchange the actual index vectors with neighbors ----
        // Receive lists of columns that neighbors need from us
        let sizes: Vec<usize> = neighbors
            .iter()
            .map(|&nb| counts_in[nb as usize] as usize)
            .collect();
        let mut recv_their_needs: Vec<Vec<u64>> = sizes.iter().map(|&n| vec![0u64; n]).collect();
        let mut reqs: Vec<<UniverseComm as Comm>::Request<'_>> = Vec::new();
        for (buf, &nb) in recv_their_needs.iter_mut().zip(neighbors.iter()) {
            if !buf.is_empty() {
                reqs.push(comm.irecv_from_u64(buf.as_mut_slice(), nb));
            }
        }
        // Send our needs to neighbors; keep buffers alive until completion
        let mut tmp_sends: Vec<Vec<u64>> = Vec::with_capacity(neighbors.len());
        for &nb in &neighbors {
            let cols = &need_from[nb as usize];
            if cols.is_empty() {
                tmp_sends.push(Vec::new());
            } else {
                tmp_sends.push(cols.iter().map(|&c| c as u64).collect());
            }
        }
        for (k, &nb) in neighbors.iter().enumerate() {
            let t = &tmp_sends[k];
            if !t.is_empty() {
                reqs.push(comm.isend_to_u64(t.as_slice(), nb));
            }
        }
        comm.wait_all(&mut reqs);
        reqs.clear();

        // Build send_idx/send_disp from the indices neighbors requested from us
        let mut send_idx: Vec<usize> = Vec::new();
        let mut send_disp: Vec<usize> = Vec::with_capacity(neighbors.len() + 1);
        send_disp.push(0);
        for (k, &nb) in neighbors.iter().enumerate() {
            let mut v = std::mem::take(&mut recv_their_needs[k]);
            v.sort_unstable();
            v.dedup();
            for g in &v {
                debug_assert!(
                    (*g as usize) >= row_start && (*g as usize) < row_end,
                    "Neighbor {} requested column {} not owned by rank {} [{}, {})",
                    nb,
                    g,
                    my_rank,
                    row_start,
                    row_end
                );
            }
            send_idx.extend(v.into_iter().map(|z| z as usize));
            send_disp.push(send_idx.len());
        }

        // Minimal runtime checks
        for &g in &send_idx {
            debug_assert!(
                g >= row_start && g < row_end,
                "send_idx contains nonlocal col {}",
                g
            );
        }
        for &g in &recv_idx {
            debug_assert!(
                owner_of_row(g, part_prefix) != my_rank,
                "recv_idx contains local col {}",
                g
            );
        }

        // Build CSR blocks
        let mut row_ptr_on = Vec::with_capacity(n_local + 1);
        row_ptr_on.push(0);
        let mut col_idx_on = Vec::new();
        let mut val_on = Vec::new();
        let mut row_ptr_off = Vec::with_capacity(n_local + 1);
        row_ptr_off.push(0);
        let mut col_idx_off = Vec::new();
        let mut val_off = Vec::new();
        for i in 0..n_local {
            for &(gcol, val) in &local_entries[i] {
                let j = g2l[&gcol];
                col_idx_on.push(j);
                val_on.push(val);
            }
            row_ptr_on.push(col_idx_on.len());
            for &(gcol, val) in &remote_entries[i] {
                let j = g2l[&gcol] - n_local;
                col_idx_off.push(j);
                val_off.push(val);
            }
            row_ptr_off.push(col_idx_off.len());
        }
        let a_on = CsrMatrix::from_csr(n_local, n_local, row_ptr_on, col_idx_on, val_on);
        let a_off = CsrMatrix::from_csr(n_local, n_halo, row_ptr_off, col_idx_off, val_off);

        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();
        let x_halo = vec![0.0; n_halo];
        let recv_buf = vec![0.0; n_halo];
        Ok(Self {
            n_global,
            row_start,
            row_end,
            a_on,
            a_off,
            n_local,
            n_halo,
            g2l,
            neighbors,
            recv_idx,
            recv_disp,
            send_idx,
            send_disp,
            x_halo: Mutex::new(x_halo),
            send_buf: Mutex::new(Vec::new()),
            recv_buf: Mutex::new(recv_buf),
            ids,
            comm,
        })
    }

    pub fn update_numeric(&mut self, a_on_vals: &[f64], a_off_vals: &[f64]) {
        self.a_on.values_mut().copy_from_slice(a_on_vals);
        self.a_off.values_mut().copy_from_slice(a_off_vals);
        self.ids.bump_values();
    }

    pub fn spmv_dist_impl(&self, x_local: &[f64], y_local: &mut [f64]) -> Result<(), KError> {
        if x_local.len() != self.n_local || y_local.len() != self.n_local {
            return Err(KError::InvalidInput("dimension mismatch".into()));
        }
        let mut recv_buf = self.recv_buf.lock().unwrap();
        let mut send_buf = self.send_buf.lock().unwrap();
        let mut x_halo = self.x_halo.lock().unwrap();
        let mut reqs: Vec<<UniverseComm as Comm>::Request<'_>> = Vec::new();
        let comm = &self.comm;
        // Post nonblocking receives into disjoint, increasing slices using split_at_mut
        let mut tail: &mut [f64] = &mut recv_buf[..];
        let mut running_off = 0usize;
        for (k, &nb) in self.neighbors.iter().enumerate() {
            let off = self.recv_disp[k];
            let cnt = self.recv_disp[k + 1] - off;
            if cnt > 0 {
                debug_assert_eq!(off, running_off);
                let (chunk, rest) = tail.split_at_mut(cnt);
                reqs.push(comm.irecv_from(chunk, nb));
                tail = rest;
                running_off += cnt;
            }
        }
        send_buf.resize(self.send_idx.len(), 0.0);
        for (p, &gcol) in self.send_idx.iter().enumerate() {
            let j = gcol - self.row_start;
            send_buf[p] = x_local[j];
        }
        for (k, &nb) in self.neighbors.iter().enumerate() {
            let off = self.send_disp[k];
            let cnt = self.send_disp[k + 1] - off;
            if cnt > 0 {
                reqs.push(comm.isend_to(&send_buf[off..off + cnt], nb));
            }
        }
        y_local.fill(0.0);
        self.a_on.spmv_scaled(1.0, x_local, 1.0, y_local)?;
        comm.wait_all(&mut reqs);
        // Drop requests to release borrows before reading recv_buf
        drop(reqs);
        x_halo.copy_from_slice(&recv_buf[..self.n_halo]);
        self.a_off.spmv_scaled(1.0, &x_halo, 1.0, y_local)?;
        Ok(())
    }
}

impl LinOp for DistCsrOp {
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.n_global, self.n_global)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        if let Err(e) = self.spmv_dist_impl(x, y) {
            panic!("DistCsrOp::matvec: {}", e);
        }
    }

    fn try_matvec(&self, x: &[f64], y: &mut [f64]) -> Result<(), crate::error::KError> {
        self.spmv_dist_impl(x, y)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn structure_id(&self) -> StructureId {
        self.ids.structure_id()
    }
    fn values_id(&self) -> ValuesId {
        self.ids.values_id()
    }

    fn comm(&self) -> UniverseComm {
        self.comm.clone()
    }
}
