//! One-device-per-rank distributed CSR operator.
//!
//! The portable path packs only requested halo entries on the GPU, transfers
//! those compact buffers through page-locked host memory, exchanges them with
//! the existing communicator, and uploads the received ghost vector. Local
//! and off-rank CSR blocks remain device resident.

use super::CudaCsrOp;
use super::operator::{CudaLinOp, CudaOperation};
use super::runtime::{CudaMpiTransport, CudaRuntime, map_driver, map_driver_kind};
use super::vector::{CudaVector, DeviceScalar};
use crate::algebra::prelude::*;
use crate::error::{CudaErrorKind, KError};
use crate::matrix::dist::halo::HaloIndexPlan;
use crate::matrix::op::{StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
use crate::parallel::{Comm, UniverseComm};
use cudarc::driver::{CudaSlice, PinnedHostSlice};
#[cfg(feature = "mpi")]
use cudarc::driver::{DevicePtr, DevicePtrMut};
use std::any::Any;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
struct NeighborRange {
    rank: usize,
    range: Range<usize>,
}

struct DistWorkspace {
    packed_send: CudaVector,
    ghosts: CudaVector,
    offdiag_product: CudaVector,
    host_send: Option<PinnedHostSlice<DeviceScalar>>,
    host_recv: Option<PinnedHostSlice<DeviceScalar>>,
    #[cfg(feature = "mpi")]
    device_requests: Vec<crate::parallel::CudaMpiRequest>,
}

/// Distributed row-block CSR operator for one CUDA device per process/rank.
///
/// `x` and `y` passed to [`CudaLinOp::apply`] are the rank-local portions. The
/// operator exchanges only off-rank entries referenced by local rows.
pub struct CudaDistCsrOp {
    runtime: Arc<CudaRuntime>,
    comm: UniverseComm,
    n_global: usize,
    row_start: usize,
    row_end: usize,
    diagonal: CudaCsrOp,
    offdiagonal: Option<CudaCsrOp>,
    send_indices: Option<CudaSlice<u64>>,
    send_neighbors: Vec<NeighborRange>,
    recv_neighbors: Vec<NeighborRange>,
    workspace: Mutex<DistWorkspace>,
    transport: CudaMpiTransport,
}

impl std::fmt::Debug for CudaDistCsrOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDistCsrOp")
            .field("rank", &self.comm.rank())
            .field("size", &self.comm.size())
            .field("n_global", &self.n_global)
            .field("owned_rows", &(self.row_start..self.row_end))
            .field("transport", &self.transport)
            .finish_non_exhaustive()
    }
}

impl CudaDistCsrOp {
    #[allow(clippy::too_many_arguments)]
    pub fn from_local_rows(
        runtime: Arc<CudaRuntime>,
        n_global: usize,
        row_start: usize,
        local_rows: &CsrMatrix<S>,
        part_prefix: &[usize],
        comm: UniverseComm,
    ) -> Result<Self, KError> {
        validate_partition(n_global, row_start, local_rows.nrows(), part_prefix, &comm)?;
        if comm.size() > 1 {
            #[cfg(feature = "mpi")]
            if !matches!(&comm, UniverseComm::Mpi(_)) {
                return Err(KError::Unsupported(
                    "multi-rank CudaDistCsrOp requires an MPI communicator",
                ));
            }
            #[cfg(not(feature = "mpi"))]
            return Err(KError::Unsupported(
                "multi-rank CudaDistCsrOp requires the mpi feature",
            ));
        }
        let transport = match runtime.options().mpi_transport {
            // There is no portable, side-effect-free CUDA-aware MPI capability
            // query. Auto therefore retains the staged path; DeviceDirect is an
            // explicit assertion by the caller that its MPI supports pointers
            // in the selected CUDA context.
            CudaMpiTransport::Auto | CudaMpiTransport::Staged => CudaMpiTransport::Staged,
            CudaMpiTransport::DeviceDirect => CudaMpiTransport::DeviceDirect,
        };
        if local_rows.ncols() != n_global {
            return Err(KError::InvalidInput(format!(
                "distributed CUDA local CSR must have {n_global} global columns, got {}",
                local_rows.ncols()
            )));
        }

        let row_end = row_start + local_rows.nrows();
        let rank = comm.rank();
        let mut recv_map = BTreeMap::<usize, Vec<usize>>::new();
        for &column in local_rows.col_idx() {
            let owner = owner_of(column, part_prefix)?;
            if owner != rank {
                recv_map.entry(owner).or_default().push(column);
            }
        }
        let halo = HaloIndexPlan::new(
            comm.clone(),
            Arc::new(part_prefix.to_vec()),
            row_start,
            row_end,
            recv_map,
        )?;

        let n_local = local_rows.nrows();
        let mut diagonal_rows = Vec::with_capacity(n_local + 1);
        let mut diagonal_cols = Vec::new();
        let mut diagonal_values = Vec::new();
        let mut offdiagonal_rows = Vec::with_capacity(n_local + 1);
        let mut offdiagonal_cols = Vec::new();
        let mut offdiagonal_values = Vec::new();
        diagonal_rows.push(0);
        offdiagonal_rows.push(0);
        for row in 0..n_local {
            for entry in local_rows.row_ptr()[row]..local_rows.row_ptr()[row + 1] {
                let column = local_rows.col_idx()[entry];
                let value = local_rows.values()[entry];
                if (row_start..row_end).contains(&column) {
                    diagonal_cols.push(column - row_start);
                    diagonal_values.push(value);
                } else {
                    let ghost = halo.ghost_index_of.get(&column).ok_or_else(|| {
                        KError::InvalidInput(format!(
                            "global column {column} is absent from the CUDA halo plan"
                        ))
                    })?;
                    offdiagonal_cols.push(*ghost);
                    offdiagonal_values.push(value);
                }
            }
            diagonal_rows.push(diagonal_cols.len());
            offdiagonal_rows.push(offdiagonal_cols.len());
        }

        let diagonal = CudaCsrOp::from_csr_parts(
            runtime.clone(),
            n_local,
            n_local,
            &diagonal_rows,
            &diagonal_cols,
            &diagonal_values,
        )?;
        let offdiagonal = if halo.n_ghost == 0 {
            None
        } else {
            Some(CudaCsrOp::from_csr_parts(
                runtime.clone(),
                n_local,
                halo.n_ghost,
                &offdiagonal_rows,
                &offdiagonal_cols,
                &offdiagonal_values,
            )?)
        };

        let (send_indices_host, send_neighbors) = flatten_send_plan(&halo);
        let recv_neighbors = flatten_recv_plan(&halo);
        let send_indices = if send_indices_host.is_empty() {
            None
        } else {
            let buffer = runtime
                .stream()
                .clone_htod(&send_indices_host)
                .map_err(|error| map_driver("upload CUDA halo gather indices", error))?;
            let bytes = send_indices_host.len() * std::mem::size_of::<u64>();
            runtime.diagnostics_ref().allocation(bytes);
            runtime.diagnostics_ref().htod(bytes);
            Some(buffer)
        };

        let packed_send = CudaVector::zeros(runtime.clone(), send_indices_host.len())?;
        let ghosts = CudaVector::zeros(runtime.clone(), halo.n_ghost)?;
        let offdiag_product = CudaVector::zeros(runtime.clone(), n_local)?;
        let (host_send, host_recv) = if transport == CudaMpiTransport::Staged {
            (
                allocate_pinned(&runtime, send_indices_host.len(), "CUDA halo send")?,
                allocate_pinned(&runtime, halo.n_ghost, "CUDA halo receive")?,
            )
        } else {
            (None, None)
        };
        #[cfg(feature = "mpi")]
        let device_request_capacity = send_neighbors.len().saturating_add(recv_neighbors.len());

        Ok(Self {
            runtime,
            comm,
            n_global,
            row_start,
            row_end,
            diagonal,
            offdiagonal,
            send_indices,
            send_neighbors,
            recv_neighbors,
            workspace: Mutex::new(DistWorkspace {
                packed_send,
                ghosts,
                offdiag_product,
                host_send,
                host_recv,
                #[cfg(feature = "mpi")]
                device_requests: Vec::with_capacity(device_request_capacity),
            }),
            transport,
        })
    }

    pub fn global_size(&self) -> usize {
        self.n_global
    }

    pub fn owned_range(&self) -> Range<usize> {
        self.row_start..self.row_end
    }

    pub fn communicator(&self) -> &UniverseComm {
        &self.comm
    }

    pub fn transport(&self) -> CudaMpiTransport {
        self.transport
    }

    pub fn halo_send_volume(&self) -> usize {
        self.send_neighbors
            .iter()
            .map(|item| item.range.len())
            .sum()
    }

    pub fn halo_recv_volume(&self) -> usize {
        self.recv_neighbors
            .iter()
            .map(|item| item.range.len())
            .sum()
    }

    pub(crate) fn diagonal_block(&self) -> &CudaCsrOp {
        &self.diagonal
    }

    fn exchange_staged(
        &self,
        x: &CudaVector,
        y: &mut CudaVector,
        workspace: &mut DistWorkspace,
    ) -> Result<(), KError> {
        #[cfg(not(feature = "mpi"))]
        {
            let _ = (x, y, workspace);
            return Err(KError::Unsupported(
                "multi-rank staged CUDA halo exchange requires the mpi feature",
            ));
        }

        #[cfg(feature = "mpi")]
        {
            if let Some(indices) = self.send_indices.as_ref() {
                self.runtime
                    .gather(x.buffer(), indices, workspace.packed_send.buffer_mut())?;
                let send_host = workspace.host_send.as_mut().ok_or_else(|| {
                    KError::SolveError("CUDA halo send staging buffer is absent".into())
                })?;
                self.runtime
                    .stream()
                    .memcpy_dtoh(workspace.packed_send.buffer(), send_host)
                    .map_err(|error| map_driver("download packed CUDA halo", error))?;
                self.runtime
                    .diagnostics_ref()
                    .dtoh(workspace.packed_send.buffer().num_bytes());
            }

            let recv_base = if let Some(recv_host) = workspace.host_recv.as_mut() {
                let recv = recv_host.as_mut_slice().map_err(|error| {
                    map_driver_kind(
                        CudaErrorKind::Synchronization,
                        "access CUDA halo receive staging",
                        error,
                    )
                })?;
                self.runtime.diagnostics_ref().synchronization();
                let recv = device_scalars_as_reals_mut(recv);
                Some(recv.as_mut_ptr())
            } else {
                None
            };

            let send_base = if let Some(send_host) = workspace.host_send.as_ref() {
                let send = send_host.as_slice().map_err(|error| {
                    map_driver_kind(
                        CudaErrorKind::Synchronization,
                        "access CUDA halo send staging",
                        error,
                    )
                })?;
                self.runtime.diagnostics_ref().synchronization();
                let send = device_scalars_as_reals(send);
                Some(send.as_ptr())
            } else {
                None
            };

            let requests = &mut workspace.device_requests;
            requests.clear();
            let mut post_error = None;
            if let Some(recv_base) = recv_base {
                for item in &self.recv_neighbors {
                    let rank = match i32::try_from(item.rank) {
                        Ok(rank) => rank,
                        Err(_) => {
                            post_error = Some(cuda_mpi_error(
                                "post staged CUDA halo receive",
                                "neighbor rank exceeds i32",
                            ));
                            break;
                        }
                    };
                    let start = scalar_real_len(item.range.start);
                    let count = scalar_real_len(item.range.len());
                    // SAFETY: recv_base points into the live, uniquely borrowed
                    // pinned receive allocation. Neighbor ranges are validated by
                    // HaloIndexPlan and the allocation remains alive through wait.
                    let request =
                        unsafe { self.comm.raw_irecv_f64(recv_base.add(start), count, rank) };
                    match request {
                        Ok(request) => requests.push(request),
                        Err(error) => {
                            post_error =
                                Some(cuda_mpi_error("post staged CUDA halo receive", error));
                            break;
                        }
                    }
                }
            }
            if post_error.is_none()
                && let Some(send_base) = send_base
            {
                for item in &self.send_neighbors {
                    let start = scalar_real_len(item.range.start);
                    let count = scalar_real_len(item.range.len());
                    let rank = match i32::try_from(item.rank) {
                        Ok(rank) => rank,
                        Err(_) => {
                            post_error = Some(cuda_mpi_error(
                                "post staged CUDA halo send",
                                "neighbor rank exceeds i32",
                            ));
                            break;
                        }
                    };
                    // SAFETY: send_base points into the live pinned send
                    // allocation and is not modified until all requests complete.
                    let request =
                        unsafe { self.comm.raw_isend_f64(send_base.add(start), count, rank) };
                    match request {
                        Ok(request) => requests.push(request),
                        Err(error) => {
                            post_error = Some(cuda_mpi_error("post staged CUDA halo send", error));
                            break;
                        }
                    }
                }
            }
            if let Some(error) = post_error {
                let _ = self.comm.wait_cuda_requests(requests);
                requests.clear();
                return Err(error);
            }
            // Launch the independent diagonal block while halo messages are in
            // flight. Always complete the posted requests before returning a
            // possible CUDA error from this local multiply.
            let diagonal_result = self.diagonal.apply(CudaOperation::NonTranspose, x, y);
            let wait_result = self
                .comm
                .wait_cuda_requests(requests)
                .map_err(|error| cuda_mpi_error("wait for staged CUDA halo exchange", error));
            requests.clear();
            diagonal_result?;
            wait_result?;

            if let Some(recv_host) = workspace.host_recv.as_ref() {
                self.runtime
                    .stream()
                    .memcpy_htod(recv_host, workspace.ghosts.buffer_mut())
                    .map_err(|error| map_driver("upload received CUDA halo", error))?;
                self.runtime
                    .diagnostics_ref()
                    .htod(workspace.ghosts.buffer().num_bytes());
            }
            Ok(())
        }
    }

    fn exchange_device_direct(
        &self,
        x: &CudaVector,
        y: &mut CudaVector,
        workspace: &mut DistWorkspace,
    ) -> Result<(), KError> {
        #[cfg(not(feature = "mpi"))]
        {
            let _ = (x, y, workspace);
            return Err(KError::Unsupported(
                "CUDA-aware device-pointer halo exchange requires the mpi feature",
            ));
        }

        #[cfg(feature = "mpi")]
        {
            if let Some(indices) = self.send_indices.as_ref() {
                self.runtime
                    .gather(x.buffer(), indices, workspace.packed_send.buffer_mut())?;
            }

            // MPI is external to cudarc's stream dependency tracker. Complete
            // the gather before MPI reads packed_send; holding the pointer
            // guards until MPI_Wait then records the read/write completion for
            // subsequent work on this context's stream.
            self.runtime.synchronize()?;
            let stream = self.runtime.stream();
            let DistWorkspace {
                packed_send,
                ghosts,
                device_requests: requests,
                ..
            } = workspace;
            requests.clear();
            let (recv_base, recv_guard) = ghosts.buffer_mut().device_ptr_mut(stream);
            let (send_base, send_guard) = packed_send.buffer().device_ptr(stream);

            let mut post_error = None;
            for item in &self.recv_neighbors {
                let rank = match i32::try_from(item.rank) {
                    Ok(rank) => rank,
                    Err(_) => {
                        post_error = Some(cuda_mpi_error(
                            "post CUDA halo receive",
                            "neighbor rank exceeds i32",
                        ));
                        break;
                    }
                };
                let Some(offset) = item
                    .range
                    .start
                    .checked_mul(std::mem::size_of::<DeviceScalar>())
                    .and_then(|offset| recv_base.checked_add(offset as u64))
                else {
                    post_error = Some(cuda_mpi_error(
                        "post CUDA halo receive",
                        "device pointer offset overflow",
                    ));
                    break;
                };
                // SAFETY: ghosts is a live, uniquely borrowed CUDA allocation;
                // the pointer guard remains alive until all requests complete.
                match unsafe {
                    self.comm
                        .cuda_irecv_f64(offset, scalar_real_len(item.range.len()), rank)
                } {
                    Ok(request) => requests.push(request),
                    Err(error) => {
                        post_error = Some(cuda_mpi_error("post CUDA halo receive", error));
                        break;
                    }
                }
            }
            if post_error.is_none() {
                for item in &self.send_neighbors {
                    let rank = match i32::try_from(item.rank) {
                        Ok(rank) => rank,
                        Err(_) => {
                            post_error = Some(cuda_mpi_error(
                                "post CUDA halo send",
                                "neighbor rank exceeds i32",
                            ));
                            break;
                        }
                    };
                    let Some(offset) = item
                        .range
                        .start
                        .checked_mul(std::mem::size_of::<DeviceScalar>())
                        .and_then(|offset| send_base.checked_add(offset as u64))
                    else {
                        post_error = Some(cuda_mpi_error(
                            "post CUDA halo send",
                            "device pointer offset overflow",
                        ));
                        break;
                    };
                    // SAFETY: packed_send is a live CUDA allocation and is not
                    // modified until the send requests have completed.
                    match unsafe {
                        self.comm
                            .cuda_isend_f64(offset, scalar_real_len(item.range.len()), rank)
                    } {
                        Ok(request) => requests.push(request),
                        Err(error) => {
                            post_error = Some(cuda_mpi_error("post CUDA halo send", error));
                            break;
                        }
                    }
                }
            }
            if let Some(error) = post_error {
                let _ = self.comm.wait_cuda_requests(requests);
                requests.clear();
                drop(send_guard);
                drop(recv_guard);
                return Err(error);
            }
            let diagonal_result = self.diagonal.apply(CudaOperation::NonTranspose, x, y);
            let wait_result = self
                .comm
                .wait_cuda_requests(requests)
                .map_err(|error| cuda_mpi_error("wait for CUDA halo exchange", error));
            requests.clear();
            drop(send_guard);
            drop(recv_guard);
            diagonal_result?;
            wait_result?;
            Ok(())
        }
    }
}

impl CudaLinOp for CudaDistCsrOp {
    fn dims(&self) -> (usize, usize) {
        let n_local = self.row_end - self.row_start;
        (n_local, n_local)
    }

    fn apply(
        &self,
        operation: CudaOperation,
        x: &CudaVector,
        y: &mut CudaVector,
    ) -> Result<(), KError> {
        if operation != CudaOperation::NonTranspose {
            return Err(KError::Unsupported(
                "distributed CUDA transpose SpMV requires a reverse halo plan",
            ));
        }
        let n_local = self.row_end - self.row_start;
        if x.len() != n_local || y.len() != n_local {
            return Err(KError::InvalidInput(format!(
                "distributed CUDA SpMV requires local vectors of length {n_local}"
            )));
        }
        x.ensure_compatible(y)?;
        if x.device_ordinal() != self.runtime.device_ordinal() {
            return Err(super::runtime::cuda_error(
                CudaErrorKind::DeviceMismatch,
                "distributed CUDA SpMV",
                "operator and vector use different devices",
            ));
        }

        let mut workspace = self.workspace.lock().map_err(|_| {
            KError::SolveError("distributed CUDA workspace mutex was poisoned".into())
        })?;
        if let Some(offdiagonal) = self.offdiagonal.as_ref() {
            match self.transport {
                CudaMpiTransport::DeviceDirect => {
                    self.exchange_device_direct(x, y, &mut workspace)?
                }
                CudaMpiTransport::Auto | CudaMpiTransport::Staged => {
                    self.exchange_staged(x, y, &mut workspace)?
                }
            }
            let DistWorkspace {
                ghosts,
                offdiag_product,
                ..
            } = &mut *workspace;
            offdiagonal.apply(CudaOperation::NonTranspose, ghosts, offdiag_product)?;
            self.runtime
                .axpby(S::one(), offdiag_product.buffer(), S::one(), y.buffer_mut())?;
        } else {
            self.diagonal.apply(CudaOperation::NonTranspose, x, y)?;
        }
        Ok(())
    }

    fn prepare(&self) -> Result<(), KError> {
        self.diagonal.prepare()?;
        if let Some(offdiagonal) = self.offdiagonal.as_ref() {
            offdiagonal.prepare()?;
        }
        Ok(())
    }

    fn device_ordinal(&self) -> usize {
        self.runtime.device_ordinal()
    }

    fn communicator(&self) -> Option<&UniverseComm> {
        Some(&self.comm)
    }

    fn structure_id(&self) -> StructureId {
        self.diagonal.structure_id()
    }

    fn values_id(&self) -> ValuesId {
        self.diagonal.values_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "mpi")]
fn cuda_mpi_error(operation: &'static str, message: impl std::fmt::Display) -> KError {
    KError::Cuda {
        kind: CudaErrorKind::Library,
        operation,
        message: message.to_string(),
    }
}

fn validate_partition(
    n_global: usize,
    row_start: usize,
    n_local: usize,
    part_prefix: &[usize],
    comm: &UniverseComm,
) -> Result<(), KError> {
    if part_prefix.len() != comm.size() + 1
        || part_prefix.first() != Some(&0)
        || part_prefix.last() != Some(&n_global)
        || part_prefix.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(KError::InvalidInput(
            "distributed CUDA partition must be monotone, span the global size, and contain size+1 entries"
                .into(),
        ));
    }
    let rank = comm.rank();
    if row_start != part_prefix[rank] || row_start + n_local != part_prefix[rank + 1] {
        return Err(KError::InvalidInput(format!(
            "rank {rank} local rows do not match its distributed CUDA partition"
        )));
    }
    Ok(())
}

fn owner_of(column: usize, part_prefix: &[usize]) -> Result<usize, KError> {
    if column >= *part_prefix.last().unwrap_or(&0) {
        return Err(KError::InvalidInput(format!(
            "distributed CUDA column {column} exceeds the global dimension"
        )));
    }
    let upper = part_prefix.partition_point(|&start| start <= column);
    Ok(upper.saturating_sub(1).min(part_prefix.len() - 2))
}

fn flatten_send_plan(halo: &HaloIndexPlan) -> (Vec<u64>, Vec<NeighborRange>) {
    let mut indices = Vec::new();
    let mut neighbors = Vec::new();
    for (&rank, local_indices) in &halo.send_local_idx {
        let start = indices.len();
        indices.extend(local_indices.iter().map(|&index| index as u64));
        neighbors.push(NeighborRange {
            rank,
            range: start..indices.len(),
        });
    }
    (indices, neighbors)
}

fn flatten_recv_plan(halo: &HaloIndexPlan) -> Vec<NeighborRange> {
    halo.ghost_ranges
        .iter()
        .map(|(&rank, range)| NeighborRange {
            rank,
            range: range.clone(),
        })
        .collect()
}

fn allocate_pinned(
    runtime: &Arc<CudaRuntime>,
    len: usize,
    operation: &'static str,
) -> Result<Option<PinnedHostSlice<DeviceScalar>>, KError> {
    if len == 0 {
        return Ok(None);
    }
    // SAFETY: DeviceScalar is plain CUDA-compatible storage and MPI overwrites
    // receives before they are read. Send buffers are initialized by D2H copy.
    let buffer = unsafe { runtime.context().alloc_pinned::<DeviceScalar>(len) }
        .map_err(|error| map_driver_kind(CudaErrorKind::Allocation, operation, error))?;
    runtime
        .diagnostics_ref()
        .allocation(len * std::mem::size_of::<DeviceScalar>());
    Ok(Some(buffer))
}

#[inline]
fn scalar_real_len(len: usize) -> usize {
    #[cfg(feature = "complex")]
    {
        len.saturating_mul(2)
    }
    #[cfg(not(feature = "complex"))]
    {
        len
    }
}

fn device_scalars_as_reals(values: &[DeviceScalar]) -> &[R] {
    // SAFETY: DeviceScalar is f64 in real builds and a repr(C), two-f64 value
    // in complex builds, as checked by vector layout tests.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<R>(), scalar_real_len(values.len()))
    }
}

fn device_scalars_as_reals_mut(values: &mut [DeviceScalar]) -> &mut [R] {
    // SAFETY: same layout argument as `device_scalars_as_reals`; the returned
    // slice is tied to the unique mutable input borrow.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<R>(),
            scalar_real_len(values.len()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::NoComm;

    #[test]
    fn owner_lookup_handles_uneven_partitions() {
        let part = [0, 2, 5, 9];
        assert_eq!(owner_of(0, &part).unwrap(), 0);
        assert_eq!(owner_of(2, &part).unwrap(), 1);
        assert_eq!(owner_of(8, &part).unwrap(), 2);
        assert!(owner_of(9, &part).is_err());
    }

    #[test]
    fn validates_rank_local_partition() {
        let comm = UniverseComm::NoComm(NoComm);
        assert!(validate_partition(3, 0, 3, &[0, 3], &comm).is_ok());
        assert!(validate_partition(3, 1, 2, &[0, 3], &comm).is_err());
    }
}
