use std::sync::Mutex;
use std::sync::mpsc::Receiver;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::parallel::Comm;
use crate::reduction::{CommDeterministic, Packet, ReproMode};
#[cfg(feature = "mpi")]
use mpi::raw::AsRaw;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static WAIT_PAIR_COUNT: AtomicUsize = AtomicUsize::new(0);
static WAIT_VEC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Test-only counters that track how many global reductions were launched.
#[derive(Clone, Debug, Default)]
pub struct ReductionCounters {
    /// Number of distinct allreduce operations.
    pub allreduces: usize,
    /// Total number of scalar entries reduced across all operations.
    pub reduced_scalars: usize,
}

static TEST_COUNTER_ENABLED: AtomicBool = AtomicBool::new(false);
static TEST_COUNTERS: Lazy<Mutex<ReductionCounters>> =
    Lazy::new(|| Mutex::new(ReductionCounters::default()));

fn record_reduction(len: usize) {
    if TEST_COUNTER_ENABLED.load(Ordering::Relaxed) {
        let mut guard = TEST_COUNTERS.lock().unwrap();
        guard.allreduces += 1;
        guard.reduced_scalars += len;
    }
}

/// Enable or disable the global reduction counter used in tests.
pub fn install_test_counter(enable: bool) {
    TEST_COUNTER_ENABLED.store(enable, Ordering::SeqCst);
    if !enable {
        let mut guard = TEST_COUNTERS.lock().unwrap();
        *guard = ReductionCounters::default();
    }
}

/// Take the current reduction counters, resetting the stored values to zero.
pub fn take_test_counter() -> ReductionCounters {
    let mut guard = TEST_COUNTERS.lock().unwrap();
    std::mem::take(&mut *guard)
}

#[inline]
fn record_wait_pair() {
    WAIT_PAIR_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
fn record_wait_vec() {
    WAIT_VEC_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub mod test_hooks {
    use super::*;

    /// Reset the asynchronous reduction completion counters. Intended for tests.
    pub fn reset_wait_counters() {
        WAIT_PAIR_COUNT.store(0, Ordering::Relaxed);
        WAIT_VEC_COUNT.store(0, Ordering::Relaxed);
    }

    /// Return the number of completed pair and vector reductions recorded so far.
    pub fn wait_counters() -> (usize, usize) {
        (
            WAIT_PAIR_COUNT.load(Ordering::Relaxed),
            WAIT_VEC_COUNT.load(Ordering::Relaxed),
        )
    }
}

/// Reduction execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductMode {
    /// Fast, implementation-defined reduction path.
    Fast,
    /// Deterministic tree reductions.
    Deterministic,
    /// Deterministic reductions with extended-precision accumulation.
    DeterministicAccurate,
}

#[inline]
fn repro_mode_from_reduct(mode: ReductMode) -> ReproMode {
    match mode {
        ReductMode::Fast => ReproMode::Fast,
        ReductMode::Deterministic => ReproMode::Deterministic,
        ReductMode::DeterministicAccurate => ReproMode::DeterministicAccurate,
    }
}

/// Options that control asynchronous reductions.
#[derive(Debug, Clone)]
pub struct ReductOptions {
    pub mode: ReductMode,
    pub max_inflight: usize,
}

impl Default for ReductOptions {
    fn default() -> Self {
        Self {
            mode: ReductMode::Fast,
            max_inflight: 4,
        }
    }
}

/// State machine for deterministic reductions.
pub trait DeterministicState<T>: Send {
    /// Progress the reduction state. Returns `true` when the result is ready.
    fn progress(&mut self) -> bool;
    /// Take the final result.
    fn take(self: Box<Self>) -> T;
}

/// Handle for a nonblocking allreduce operation.
pub enum AllreduceHandle<T> {
    /// Result already available.
    Ready(T),
    /// MPI nonblocking reduction.
    #[cfg(feature = "mpi")]
    Mpi {
        req: mpi::ffi::MPI_Request,
        buf: Vec<R>,
        convert: fn(&[R]) -> T,
    },
    /// Shared-memory emulation backed by a channel.
    Rayon { rx: Receiver<T> },
    /// Deterministic engine state machine.
    Deterministic {
        state: Box<dyn DeterministicState<T>>,
    },
}

impl<T> AllreduceHandle<T> {
    fn new_ready(value: T) -> Self {
        AllreduceHandle::Ready(value)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AllreduceHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllreduceHandle::Ready(val) => f.debug_tuple("Ready").field(val).finish(),
            #[cfg(feature = "mpi")]
            AllreduceHandle::Mpi { req, buf, .. } => f
                .debug_struct("Mpi")
                .field("request", req)
                .field("buf", buf)
                .finish(),
            AllreduceHandle::Rayon { .. } => f.debug_struct("Rayon").finish(),
            AllreduceHandle::Deterministic { .. } => f.debug_struct("Deterministic").finish(),
        }
    }
}

/// Trait implemented by communicators that support the asynchronous reductions used by solvers.
pub trait AllreduceOps {
    /// Launch a nonblocking reduction of two scalars and return the handle and local contributions.
    fn allreduce2_async(
        &self,
        a: R,
        b: R,
        opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<(R, R)>, (R, R)), KError>;

    /// Launch a nonblocking reduction of an arbitrary-length vector of scalars.
    fn allreduce_n_async(
        &self,
        data: Vec<R>,
        opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<Vec<R>>, Vec<R>), KError>;

    /// Poll a pair reduction and return the result if ready.
    fn test_pair(h: &mut AllreduceHandle<(R, R)>) -> Option<(R, R)>;

    /// Poll a vector reduction and return the result if ready.
    fn test_vec(h: &mut AllreduceHandle<Vec<R>>) -> Option<Vec<R>>;

    /// Block until the pair reduction completes.
    fn wait_pair(h: AllreduceHandle<(R, R)>) -> (R, R);

    /// Block until the vector reduction completes.
    fn wait_vec(h: AllreduceHandle<Vec<R>>) -> Vec<R>;
}

/// Trait alias for communicators that support asynchronous reductions.
pub trait AsyncComm: Comm + AllreduceOps {}

impl<T> AsyncComm for T where T: Comm + AllreduceOps + ?Sized {}

/// Helper used by polling implementations to convert a completed handle into the ready state.
fn finalize_handle_pair(handle: &mut AllreduceHandle<(R, R)>, result: (R, R)) -> (R, R) {
    *handle = AllreduceHandle::Ready(result);
    if let AllreduceHandle::Ready(val) = handle {
        *val
    } else {
        unreachable!()
    }
}

fn finalize_handle_vec(handle: &mut AllreduceHandle<Vec<R>>, result: Vec<R>) -> Vec<R> {
    *handle = AllreduceHandle::Ready(result);
    if let AllreduceHandle::Ready(val) = handle {
        val.clone()
    } else {
        unreachable!()
    }
}

/// Convert a raw buffer into a pair.
fn convert_pair(buf: &[R]) -> (R, R) {
    debug_assert_eq!(buf.len(), 2);
    (buf[0], buf[1])
}

fn deterministic_reduce_vec<C>(comm: &C, data: &[R], mode: ReproMode) -> Vec<R>
where
    C: Comm + CommDeterministic,
{
    if data.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(data.len());
    let mut offset = 0;
    while offset < data.len() {
        let remaining = data.len() - offset;
        let width = remaining.min(4);
        match width {
            4 => {
                let mut chunk = [0.0f64; 4];
                chunk.copy_from_slice(&data[offset..offset + 4]);
                let packet = Packet::<4> { v: chunk };
                let reduced = comm.allreduce_det(&packet, mode);
                out.extend_from_slice(&reduced.v);
            }
            3 => {
                let mut chunk = [0.0f64; 3];
                chunk.copy_from_slice(&data[offset..offset + 3]);
                let packet = Packet::<3> { v: chunk };
                let reduced = comm.allreduce_det(&packet, mode);
                out.extend_from_slice(&reduced.v);
            }
            2 => {
                let mut chunk = [0.0f64; 2];
                chunk.copy_from_slice(&data[offset..offset + 2]);
                let packet = Packet::<2> { v: chunk };
                let reduced = comm.allreduce_det(&packet, mode);
                out.extend_from_slice(&reduced.v);
            }
            _ => {
                let packet = Packet::<1> { v: [data[offset]] };
                let reduced = comm.allreduce_det(&packet, mode);
                out.push(reduced.v[0]);
            }
        }
        offset += width;
    }
    out
}

#[cfg(feature = "mpi")]
fn mpi_test_request(req: &mut mpi::ffi::MPI_Request) -> bool {
    let mut flag = 0;
    let rc = unsafe { mpi::ffi::MPI_Test(req, &mut flag, mpi::ffi::RSMPI_STATUS_IGNORE) };
    debug_assert_eq!(rc, 0);
    flag != 0
}

#[cfg(feature = "mpi")]
fn mpi_wait_request(mut req: mpi::ffi::MPI_Request) {
    let rc = unsafe { mpi::ffi::MPI_Wait(&mut req, mpi::ffi::RSMPI_STATUS_IGNORE) };
    debug_assert_eq!(rc, 0);
}

impl AllreduceOps for crate::parallel::NoComm {
    fn allreduce2_async(
        &self,
        a: R,
        b: R,
        _opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<(R, R)>, (R, R)), KError> {
        record_reduction(2);
        let sum = (a, b);
        Ok((AllreduceHandle::new_ready(sum), sum))
    }

    fn allreduce_n_async(
        &self,
        data: Vec<R>,
        _opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<Vec<R>>, Vec<R>), KError> {
        record_reduction(data.len());
        Ok((AllreduceHandle::new_ready(data.clone()), data))
    }

    fn test_pair(h: &mut AllreduceHandle<(R, R)>) -> Option<(R, R)> {
        match h {
            AllreduceHandle::Ready(val) => Some(*val),
            _ => None,
        }
    }

    fn test_vec(h: &mut AllreduceHandle<Vec<R>>) -> Option<Vec<R>> {
        match h {
            AllreduceHandle::Ready(val) => Some(val.clone()),
            _ => None,
        }
    }

    fn wait_pair(h: AllreduceHandle<(R, R)>) -> (R, R) {
        record_wait_pair();
        match h {
            AllreduceHandle::Ready(val) => val,
            _ => unreachable!(),
        }
    }

    fn wait_vec(h: AllreduceHandle<Vec<R>>) -> Vec<R> {
        record_wait_vec();
        match h {
            AllreduceHandle::Ready(val) => val,
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "rayon")]
impl AllreduceOps for crate::parallel::rayon_comm::RayonComm {
    fn allreduce2_async(
        &self,
        a: R,
        b: R,
        _opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<(R, R)>, (R, R)), KError> {
        record_reduction(2);
        let (tx, rx) = std::sync::mpsc::channel();
        let local = (a, b);
        rayon::spawn_fifo(move || {
            let _ = tx.send(local);
        });
        Ok((AllreduceHandle::Rayon { rx }, local))
    }

    fn allreduce_n_async(
        &self,
        data: Vec<R>,
        _opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<Vec<R>>, Vec<R>), KError> {
        record_reduction(data.len());
        let (tx, rx) = std::sync::mpsc::channel();
        let local = data.clone();
        rayon::spawn_fifo(move || {
            let _ = tx.send(data);
        });
        Ok((AllreduceHandle::Rayon { rx }, local))
    }

    fn test_pair(h: &mut AllreduceHandle<(R, R)>) -> Option<(R, R)> {
        match h {
            AllreduceHandle::Ready(val) => Some(*val),
            AllreduceHandle::Rayon { rx } => rx.try_recv().ok().map(|v| finalize_handle_pair(h, v)),
            _ => None,
        }
    }

    fn test_vec(h: &mut AllreduceHandle<Vec<R>>) -> Option<Vec<R>> {
        match h {
            AllreduceHandle::Ready(val) => Some(val.clone()),
            AllreduceHandle::Rayon { rx } => rx.try_recv().ok().map(|v| finalize_handle_vec(h, v)),
            _ => None,
        }
    }

    fn wait_pair(h: AllreduceHandle<(R, R)>) -> (R, R) {
        record_wait_pair();
        match h {
            AllreduceHandle::Ready(val) => val,
            AllreduceHandle::Rayon { rx } => rx.recv().unwrap(),
            _ => unreachable!(),
        }
    }

    fn wait_vec(h: AllreduceHandle<Vec<R>>) -> Vec<R> {
        record_wait_vec();
        match h {
            AllreduceHandle::Ready(val) => val,
            AllreduceHandle::Rayon { rx } => rx.recv().unwrap(),
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "mpi")]
impl AllreduceOps for crate::parallel::mpi_comm::MpiComm {
    fn allreduce2_async(
        &self,
        a: R,
        b: R,
        opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<(R, R)>, (R, R)), KError> {
        if let ReductMode::Deterministic | ReductMode::DeterministicAccurate = opt.mode {
            record_reduction(2);
            let packet = Packet::<2> { v: [a, b] };
            let reduced = self.allreduce_det(&packet, repro_mode_from_reduct(opt.mode));
            let result = (reduced.v[0], reduced.v[1]);
            return Ok((AllreduceHandle::new_ready(result), (a, b)));
        }
        record_reduction(2);
        let mut buf = vec![a, b];
        let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            mpi::ffi::MPI_Iallreduce(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                2,
                mpi::ffi::RSMPI_DOUBLE,
                mpi::ffi::RSMPI_SUM,
                self.world.as_raw(),
                &mut req,
            )
        };
        if rc != 0 {
            return Err(KError::SolveError(format!("MPI_Iallreduce failed: {rc}")));
        }
        Ok((
            AllreduceHandle::Mpi {
                req,
                buf,
                convert: convert_pair,
            },
            (a, b),
        ))
    }

    fn allreduce_n_async(
        &self,
        data: Vec<R>,
        opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<Vec<R>>, Vec<R>), KError> {
        if let ReductMode::Deterministic | ReductMode::DeterministicAccurate = opt.mode {
            record_reduction(data.len());
            let reduced = deterministic_reduce_vec(self, &data, repro_mode_from_reduct(opt.mode));
            return Ok((AllreduceHandle::new_ready(reduced.clone()), reduced));
        }
        record_reduction(data.len());
        let mut buf = data.clone();
        let count = buf.len();
        let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            mpi::ffi::MPI_Iallreduce(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                count as i32,
                mpi::ffi::RSMPI_DOUBLE,
                mpi::ffi::RSMPI_SUM,
                self.world.as_raw(),
                &mut req,
            )
        };
        if rc != 0 {
            return Err(KError::SolveError(format!("MPI_Iallreduce failed: {rc}")));
        }
        Ok((
            AllreduceHandle::Mpi {
                req,
                buf,
                convert: |slice| slice.to_vec(),
            },
            data,
        ))
    }

    fn test_pair(h: &mut AllreduceHandle<(R, R)>) -> Option<(R, R)> {
        match h {
            AllreduceHandle::Ready(val) => Some(*val),
            AllreduceHandle::Mpi { req, buf, convert } => {
                if mpi_test_request(req) {
                    let result = convert(buf);
                    Some(finalize_handle_pair(h, result))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn test_vec(h: &mut AllreduceHandle<Vec<R>>) -> Option<Vec<R>> {
        match h {
            AllreduceHandle::Ready(val) => Some(val.clone()),
            AllreduceHandle::Mpi { req, buf, convert } => {
                if mpi_test_request(req) {
                    let result = convert(buf);
                    Some(finalize_handle_vec(h, result))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn wait_pair(h: AllreduceHandle<(R, R)>) -> (R, R) {
        record_wait_pair();
        match h {
            AllreduceHandle::Ready(val) => val,
            AllreduceHandle::Mpi { req, buf, convert } => {
                mpi_wait_request(req);
                convert(&buf)
            }
            _ => unreachable!(),
        }
    }

    fn wait_vec(h: AllreduceHandle<Vec<R>>) -> Vec<R> {
        record_wait_vec();
        match h {
            AllreduceHandle::Ready(val) => val,
            AllreduceHandle::Mpi { req, buf, convert } => {
                mpi_wait_request(req);
                convert(&buf)
            }
            _ => unreachable!(),
        }
    }
}

impl AllreduceOps for crate::parallel::UniverseComm {
    fn allreduce2_async(
        &self,
        a: R,
        b: R,
        opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<(R, R)>, (R, R)), KError> {
        match self {
            crate::parallel::UniverseComm::NoComm(comm) => comm.allreduce2_async(a, b, opt),
            #[cfg(feature = "mpi")]
            crate::parallel::UniverseComm::Mpi(comm) => comm.allreduce2_async(a, b, opt),
            #[cfg(feature = "rayon")]
            crate::parallel::UniverseComm::Rayon(comm) => comm.allreduce2_async(a, b, opt),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            crate::parallel::UniverseComm::Serial => {
                crate::parallel::NoComm.allreduce2_async(a, b, opt)
            }
        }
    }

    fn allreduce_n_async(
        &self,
        data: Vec<R>,
        opt: &ReductOptions,
    ) -> Result<(AllreduceHandle<Vec<R>>, Vec<R>), KError> {
        match self {
            crate::parallel::UniverseComm::NoComm(comm) => comm.allreduce_n_async(data, opt),
            #[cfg(feature = "mpi")]
            crate::parallel::UniverseComm::Mpi(comm) => comm.allreduce_n_async(data, opt),
            #[cfg(feature = "rayon")]
            crate::parallel::UniverseComm::Rayon(comm) => comm.allreduce_n_async(data, opt),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            crate::parallel::UniverseComm::Serial => {
                crate::parallel::NoComm.allreduce_n_async(data, opt)
            }
        }
    }

    fn test_pair(h: &mut AllreduceHandle<(R, R)>) -> Option<(R, R)> {
        match h {
            AllreduceHandle::Ready(val) => Some(*val),
            #[cfg(feature = "mpi")]
            AllreduceHandle::Mpi { .. } => {
                <crate::parallel::mpi_comm::MpiComm as AllreduceOps>::test_pair(h)
            }
            #[cfg(feature = "rayon")]
            AllreduceHandle::Rayon { .. } => {
                <crate::parallel::rayon_comm::RayonComm as AllreduceOps>::test_pair(h)
            }
            #[cfg(not(feature = "rayon"))]
            AllreduceHandle::Rayon { .. } => None,
            AllreduceHandle::Deterministic { .. } => None,
        }
    }

    fn test_vec(h: &mut AllreduceHandle<Vec<R>>) -> Option<Vec<R>> {
        match h {
            AllreduceHandle::Ready(val) => Some(val.clone()),
            #[cfg(feature = "mpi")]
            AllreduceHandle::Mpi { .. } => {
                <crate::parallel::mpi_comm::MpiComm as AllreduceOps>::test_vec(h)
            }
            #[cfg(feature = "rayon")]
            AllreduceHandle::Rayon { .. } => {
                <crate::parallel::rayon_comm::RayonComm as AllreduceOps>::test_vec(h)
            }
            #[cfg(not(feature = "rayon"))]
            AllreduceHandle::Rayon { .. } => None,
            AllreduceHandle::Deterministic { .. } => None,
        }
    }

    fn wait_pair(h: AllreduceHandle<(R, R)>) -> (R, R) {
        record_wait_pair();
        match h {
            AllreduceHandle::Ready(val) => val,
            #[cfg(feature = "mpi")]
            AllreduceHandle::Mpi { .. } => {
                <crate::parallel::mpi_comm::MpiComm as AllreduceOps>::wait_pair(h)
            }
            #[cfg(feature = "rayon")]
            AllreduceHandle::Rayon { .. } => {
                <crate::parallel::rayon_comm::RayonComm as AllreduceOps>::wait_pair(h)
            }
            #[cfg(not(feature = "rayon"))]
            AllreduceHandle::Rayon { .. } => unreachable!("rayon backend disabled"),
            AllreduceHandle::Deterministic { .. } => {
                panic!("deterministic reductions not implemented")
            }
        }
    }

    fn wait_vec(h: AllreduceHandle<Vec<R>>) -> Vec<R> {
        record_wait_vec();
        match h {
            AllreduceHandle::Ready(val) => val,
            #[cfg(feature = "mpi")]
            AllreduceHandle::Mpi { .. } => {
                <crate::parallel::mpi_comm::MpiComm as AllreduceOps>::wait_vec(h)
            }
            #[cfg(feature = "rayon")]
            AllreduceHandle::Rayon { .. } => {
                <crate::parallel::rayon_comm::RayonComm as AllreduceOps>::wait_vec(h)
            }
            #[cfg(not(feature = "rayon"))]
            AllreduceHandle::Rayon { .. } => unreachable!("rayon backend disabled"),
            AllreduceHandle::Deterministic { .. } => {
                panic!("deterministic reductions not implemented")
            }
        }
    }
}
