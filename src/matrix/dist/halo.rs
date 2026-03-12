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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborOrder {
    RankAscending,
    SendVolumeDesc,
    RecvVolumeDesc,
}

#[derive(Debug, Clone)]
pub struct HaloTuning {
    pub coalesce_pack_runs: bool,
    pub min_run_len: usize,
    pub neighbor_order: NeighborOrder,
}

impl Default for HaloTuning {
    fn default() -> Self {
        Self {
            coalesce_pack_runs: true,
            min_run_len: 4,
            neighbor_order: NeighborOrder::RankAscending,
        }
    }
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

#[derive(Clone)]
struct RecvSchedule {
    nbr: usize,
}

#[derive(Clone)]
enum PackOp {
    Scatter {
        src: usize,
        dst: usize,
    },
    Run {
        src_start: usize,
        dst_start: usize,
        len: usize,
    },
}

#[derive(Clone)]
struct SendSchedule {
    nbr: usize,
    ops: Vec<PackOp>,
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
    recv_schedule: Vec<RecvSchedule>,
    send_schedule: Vec<SendSchedule>,
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
        Self::new_with_tuning(
            comm,
            row_part,
            row_start,
            row_end,
            recv_map,
            HaloTuning::default(),
        )
    }

    pub fn new_with_tuning(
        comm: UniverseComm,
        row_part: Arc<Vec<usize>>,
        row_start: usize,
        row_end: usize,
        recv_map: BTreeMap<usize, Vec<usize>>,
        tuning: HaloTuning,
    ) -> Result<Self, KError> {
        let index = Arc::new(HaloIndexPlan::new(
            comm, row_part, row_start, row_end, recv_map,
        )?);
        let buffers = HaloBuffers::new(&index);
        let recv_schedule = build_recv_schedule(&index, tuning.neighbor_order);
        let send_schedule = build_send_schedule(&index, &tuning, tuning.neighbor_order);
        Ok(Self {
            index,
            buffers,
            recv_schedule,
            send_schedule,
        })
    }

    pub fn ghost_slice_ref(&self) -> std::sync::RwLockReadGuard<'_, Vec<S>> {
        self.buffers.ghost_flat.read().unwrap()
    }

    pub fn post_halo<'a>(&'a self, x_local: &[S]) -> HaloReq<'a> {
        let mut recv_reqs = Vec::new();
        for item in &self.recv_schedule {
            if let Some(buf_lock) = self.buffers.recv_buf.get(&item.nbr) {
                let buf = unsafe { &mut *buf_lock.get() };
                if buf.is_empty() {
                    continue;
                }
                let slice = halo_slice_mut(buf);
                let req = self.index.comm.irecv_from(slice, item.nbr as i32);
                recv_reqs.push(req);
            }
        }

        let mut send_reqs = Vec::new();
        for item in &self.send_schedule {
            if let Some(buf_lock) = self.buffers.send_buf.get(&item.nbr) {
                let buf = unsafe { &mut *buf_lock.get() };
                if buf.is_empty() {
                    continue;
                }
                for op in &item.ops {
                    match *op {
                        PackOp::Scatter { src, dst } => buf[dst] = x_local[src],
                        PackOp::Run {
                            src_start,
                            dst_start,
                            len,
                        } => {
                            buf[dst_start..dst_start + len]
                                .copy_from_slice(&x_local[src_start..src_start + len]);
                        }
                    }
                }
                let slice = halo_slice(buf);
                let req = self.index.comm.isend_to(slice, item.nbr as i32);
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

fn build_recv_schedule(index: &HaloIndexPlan, neighbor_order: NeighborOrder) -> Vec<RecvSchedule> {
    let mut neighbors: Vec<usize> = index
        .recv_map
        .iter()
        .filter_map(|(&nbr, cols)| (!cols.is_empty()).then_some(nbr))
        .collect();
    order_neighbors(&mut neighbors, index, neighbor_order);
    neighbors
        .into_iter()
        .map(|nbr| RecvSchedule { nbr })
        .collect()
}

fn build_send_schedule(
    index: &HaloIndexPlan,
    tuning: &HaloTuning,
    neighbor_order: NeighborOrder,
) -> Vec<SendSchedule> {
    let mut neighbors: Vec<usize> = index
        .send_local_idx
        .iter()
        .filter_map(|(&nbr, idxs)| (!idxs.is_empty()).then_some(nbr))
        .collect();
    order_neighbors(&mut neighbors, index, neighbor_order);

    neighbors
        .into_iter()
        .map(|nbr| {
            let idxs = index
                .send_local_idx
                .get(&nbr)
                .expect("send schedule mismatch");
            let mut ops = Vec::new();
            let mut dst = 0usize;
            while dst < idxs.len() {
                if tuning.coalesce_pack_runs {
                    let mut len = 1usize;
                    while dst + len < idxs.len() && idxs[dst + len] == idxs[dst] + len {
                        len += 1;
                    }
                    if len >= tuning.min_run_len {
                        ops.push(PackOp::Run {
                            src_start: idxs[dst],
                            dst_start: dst,
                            len,
                        });
                        dst += len;
                        continue;
                    }
                }
                ops.push(PackOp::Scatter {
                    src: idxs[dst],
                    dst,
                });
                dst += 1;
            }
            SendSchedule { nbr, ops }
        })
        .collect()
}

fn order_neighbors(neighbors: &mut [usize], index: &HaloIndexPlan, neighbor_order: NeighborOrder) {
    match neighbor_order {
        NeighborOrder::RankAscending => neighbors.sort_unstable(),
        NeighborOrder::SendVolumeDesc => neighbors.sort_unstable_by(|a, b| {
            let av = index.send_local_idx.get(a).map_or(0, Vec::len);
            let bv = index.send_local_idx.get(b).map_or(0, Vec::len);
            bv.cmp(&av).then_with(|| a.cmp(b))
        }),
        NeighborOrder::RecvVolumeDesc => neighbors.sort_unstable_by(|a, b| {
            let av = index.recv_map.get(a).map_or(0, Vec::len);
            let bv = index.recv_map.get(b).map_or(0, Vec::len);
            bv.cmp(&av).then_with(|| a.cmp(b))
        }),
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

    fn build_index_for_schedule_tests() -> HaloIndexPlan {
        let comm = UniverseComm::NoComm(NoComm);
        let row_part = Arc::new(vec![0usize, 16usize]);
        let mut recv_map = BTreeMap::new();
        recv_map.insert(1, vec![100, 101]);
        recv_map.insert(2, vec![200, 201, 202, 203]);
        recv_map.insert(3, vec![300]);
        HaloIndexPlan {
            comm,
            rank: 0,
            size: 4,
            row_part,
            row_start: 0,
            row_end: 16,
            n_local: 16,
            recv_map,
            send_map: BTreeMap::from([
                (1usize, vec![0, 1]),
                (2usize, vec![4, 5, 6, 7]),
                (3usize, vec![9]),
            ]),
            send_local_idx: BTreeMap::from([
                (1usize, vec![0, 1]),
                (2usize, vec![4, 5, 6, 7]),
                (3usize, vec![9]),
            ]),
            ghost_index_of: BTreeMap::new(),
            ghost_ranges: BTreeMap::new(),
            n_ghost: 0,
        }
    }

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

    #[test]
    fn halo_send_schedule_coalesces_contiguous_runs() {
        let index = build_index_for_schedule_tests();
        let tuning = HaloTuning {
            coalesce_pack_runs: true,
            min_run_len: 3,
            neighbor_order: NeighborOrder::RankAscending,
        };
        let schedule = build_send_schedule(&index, &tuning, NeighborOrder::RankAscending);
        let nbr2 = schedule.iter().find(|item| item.nbr == 2).unwrap();
        assert!(matches!(
            nbr2.ops.as_slice(),
            [PackOp::Run {
                src_start: 4,
                dst_start: 0,
                len: 4
            }]
        ));
    }

    #[test]
    fn halo_recv_schedule_can_be_ordered_by_volume() {
        let index = build_index_for_schedule_tests();
        let recv = build_recv_schedule(&index, NeighborOrder::RecvVolumeDesc);
        let order: Vec<usize> = recv.into_iter().map(|item| item.nbr).collect();
        assert_eq!(order, vec![2, 1, 3]);
    }
}
