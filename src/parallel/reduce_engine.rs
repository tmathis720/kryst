use crate::algebra::parallel::{dot_conj_local_with_mode, sum_abs2_local_with_mode};
use crate::algebra::prelude::*;
use crate::parallel::{Comm, UniverseComm};
use crate::reduction::{CommDeterministic, Packet, ReproMode};
use crate::utils::reduction::{ReductExec, ReductOptions, record_reduction};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use mpi::raw::AsRaw;

#[cfg(feature = "rayon")]
use std::sync::mpsc::Receiver;

/// Reduction engine that selects synchronous or asynchronous collectives.
pub trait ReductionEngine: Send + Sync + std::fmt::Debug {
    fn supports_async(&self) -> bool;

    fn allreduce_sum_r(&self, x: R) -> R;
    fn allreduce_sum_s(&self, x: S) -> S;
    fn norm2_s(&self, x: &[S]) -> R;
    fn dot_s(&self, x: &[S], y: &[S]) -> S;

    fn sum_vec_r(&self, buf: Vec<R>) -> Vec<R> {
        self.iallreduce_sum_vec_r(buf).wait()
    }

    fn iallreduce_sum_r(&self, x: R) -> ReduceHandle<R>;
    fn iallreduce_sum_s(&self, x: S) -> ReduceHandle<S>;
    fn iallreduce_sum_vec_r(&self, buf: Vec<R>) -> ReduceHandle<Vec<R>>;
}

#[derive(Debug)]
pub struct ReduceHandle<T> {
    inner: ReduceHandleInner<T>,
}

#[derive(Debug)]
enum ReduceHandleInner<T> {
    Ready(T),
    #[cfg(feature = "mpi")]
    MpiScalar(MpiScalarState<T>),
    #[cfg(feature = "mpi")]
    MpiVec(MpiVecState<T>),
    #[cfg(feature = "rayon")]
    Rayon {
        rx: Receiver<T>,
    },
}

#[cfg(feature = "mpi")]
#[derive(Debug)]
struct MpiScalarState<T> {
    req: mpi::ffi::MPI_Request,
    buf: [R; 2],
    len: usize,
    convert: fn(&[R]) -> T,
    complete: bool,
}

#[cfg(feature = "mpi")]
#[derive(Debug)]
struct MpiVecState<T> {
    req: mpi::ffi::MPI_Request,
    buf: Vec<R>,
    convert: fn(Vec<R>) -> T,
    complete: bool,
}

impl<T> ReduceHandle<T> {
    fn ready(value: T) -> Self {
        Self {
            inner: ReduceHandleInner::Ready(value),
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.inner, ReduceHandleInner::Ready(_))
    }

    pub fn wait(self) -> T {
        match self.inner {
            ReduceHandleInner::Ready(val) => val,
            #[cfg(feature = "mpi")]
            ReduceHandleInner::MpiScalar(mut state) => {
                mpi_wait_request(&mut state.req);
                state.complete = true;
                (state.convert)(&state.buf[..state.len])
            }
            #[cfg(feature = "mpi")]
            ReduceHandleInner::MpiVec(mut state) => {
                mpi_wait_request(&mut state.req);
                state.complete = true;
                let buf = std::mem::take(&mut state.buf);
                (state.convert)(buf)
            }
            #[cfg(feature = "rayon")]
            ReduceHandleInner::Rayon { rx } => rx.recv().unwrap(),
        }
    }
}

impl<T: Copy> ReduceHandle<T> {
    pub fn test(&mut self) -> Option<T> {
        match &mut self.inner {
            ReduceHandleInner::Ready(val) => Some(*val),
            #[cfg(feature = "mpi")]
            ReduceHandleInner::MpiScalar(state) => {
                if mpi_test_request(&mut state.req) {
                    state.complete = true;
                    let result = (state.convert)(&state.buf[..state.len]);
                    self.inner = ReduceHandleInner::Ready(result);
                    Some(result)
                } else {
                    None
                }
            }
            #[cfg(feature = "mpi")]
            ReduceHandleInner::MpiVec(_) => None,
            #[cfg(feature = "rayon")]
            ReduceHandleInner::Rayon { rx } => rx.try_recv().ok().map(|v| {
                self.inner = ReduceHandleInner::Ready(v);
                v
            }),
        }
    }
}

#[cfg(feature = "mpi")]
impl<T> Drop for MpiScalarState<T> {
    fn drop(&mut self) {
        debug_assert!(
            self.complete,
            "ReduceHandle dropped before MPI reduction completed"
        );
    }
}

#[cfg(feature = "mpi")]
impl<T> Drop for MpiVecState<T> {
    fn drop(&mut self) {
        debug_assert!(
            self.complete,
            "ReduceHandle dropped before MPI reduction completed"
        );
    }
}

#[derive(Debug, Clone)]
pub struct CommReductionEngine {
    comm: UniverseComm,
    opts: ReductOptions,
}

impl CommReductionEngine {
    pub fn new(comm: UniverseComm, opts: ReductOptions) -> Self {
        Self { comm, opts }
    }

    fn effective_mode(&self) -> ReproMode {
        self.opts.effective_mode()
    }

    fn async_enabled(&self) -> bool {
        if matches!(self.opts.exec, ReductExec::Sync) {
            return false;
        }
        if self.opts.reproducible {
            return false;
        }
        if self.comm.size() <= 1 {
            return false;
        }
        if matches!(
            self.opts.mode,
            ReproMode::Deterministic | ReproMode::DeterministicAccurate
        ) {
            return false;
        }
        self.supports_async()
    }

    fn reduce_vec_in_place(&self, buf: &mut [R]) {
        let mode = self.effective_mode();
        if buf.is_empty() {
            return;
        }
        match mode {
            ReproMode::Fast => match &self.comm {
                UniverseComm::NoComm(_) => {}
                #[cfg(feature = "mpi")]
                UniverseComm::Mpi(comm) => {
                    let rc = unsafe {
                        mpi::ffi::MPI_Allreduce(
                            buf.as_ptr() as *const std::ffi::c_void,
                            buf.as_mut_ptr() as *mut std::ffi::c_void,
                            buf.len() as i32,
                            mpi::ffi::RSMPI_DOUBLE,
                            mpi::ffi::RSMPI_SUM,
                            comm.world.as_raw(),
                        )
                    };
                    debug_assert_eq!(rc, 0);
                }
                #[cfg(feature = "rayon")]
                UniverseComm::Rayon(_) => {}
                #[cfg(not(any(feature = "mpi", feature = "rayon")))]
                UniverseComm::Serial => {}
            },
            ReproMode::Deterministic | ReproMode::DeterministicAccurate => {
                for slot in buf.iter_mut() {
                    *slot = self.comm.reduce_sum_real_repro(*slot);
                }
            }
        }
    }
}

impl ReductionEngine for CommReductionEngine {
    fn supports_async(&self) -> bool {
        match &self.comm {
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(_) => true,
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_) => true,
            _ => false,
        }
    }

    fn allreduce_sum_r(&self, x: R) -> R {
        let mode = self.effective_mode();
        match mode {
            ReproMode::Fast => self.comm.allreduce_sum_real(x),
            _ => {
                let packet = Packet::<1> { v: [x] };
                self.comm.allreduce_det(&packet, mode).v[0]
            }
        }
    }

    fn allreduce_sum_s(&self, x: S) -> S {
        let mode = self.effective_mode();
        match mode {
            ReproMode::Fast => self.comm.allreduce_sum_scalar(x),
            _ => {
                #[cfg(feature = "complex")]
                {
                    let packet = Packet::<2> {
                        v: [x.real(), x.imag()],
                    };
                    let reduced = self.comm.allreduce_det(&packet, mode);
                    S::from_parts(reduced.v[0], reduced.v[1])
                }
                #[cfg(not(feature = "complex"))]
                {
                    let packet = Packet::<1> { v: [x.real()] };
                    let reduced = self.comm.allreduce_det(&packet, mode);
                    S::from_real(reduced.v[0])
                }
            }
        }
    }

    fn norm2_s(&self, x: &[S]) -> R {
        let mode = self.effective_mode();
        let local = sum_abs2_local_with_mode(x, mode);
        let global = self.allreduce_sum_r(local);
        let clamped = if global >= 0.0 { global } else { 0.0 };
        clamped.sqrt()
    }

    fn dot_s(&self, x: &[S], y: &[S]) -> S {
        debug_assert_eq!(x.len(), y.len());
        let mode = self.effective_mode();
        let local = dot_conj_local_with_mode(x, y, mode);
        self.allreduce_sum_s(local)
    }

    fn sum_vec_r(&self, buf: Vec<R>) -> Vec<R> {
        record_reduction(buf.len());
        let mut out = buf;
        self.reduce_vec_in_place(&mut out);
        out
    }

    fn iallreduce_sum_r(&self, x: R) -> ReduceHandle<R> {
        record_reduction(1);
        if !self.async_enabled() {
            return ReduceHandle::ready(self.allreduce_sum_r(x));
        }
        match &self.comm {
            UniverseComm::NoComm(_) => ReduceHandle::ready(x),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => {
                let mut buf = [x, R::default()];
                let req = mpi_iallreduce_in_place(&mut buf[..1], &comm.world);
                ReduceHandle {
                    inner: ReduceHandleInner::MpiScalar(MpiScalarState {
                        req,
                        buf,
                        len: 1,
                        convert: |slice| slice[0],
                        complete: false,
                    }),
                }
            }
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_) => {
                let (tx, rx) = std::sync::mpsc::channel();
                let local = x;
                rayon::spawn_fifo(move || {
                    let _ = tx.send(local);
                });
                ReduceHandle {
                    inner: ReduceHandleInner::Rayon { rx },
                }
            }
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => ReduceHandle::ready(x),
        }
    }

    fn iallreduce_sum_s(&self, x: S) -> ReduceHandle<S> {
        #[cfg(feature = "complex")]
        record_reduction(2);
        #[cfg(not(feature = "complex"))]
        record_reduction(1);
        if !self.async_enabled() {
            return ReduceHandle::ready(self.allreduce_sum_s(x));
        }
        match &self.comm {
            UniverseComm::NoComm(_) => ReduceHandle::ready(x),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => {
                #[cfg(feature = "complex")]
                let mut buf = [x.real(), x.imag()];
                #[cfg(not(feature = "complex"))]
                let mut buf = [x.real(), R::default()];
                #[cfg(feature = "complex")]
                let len = 2usize;
                #[cfg(not(feature = "complex"))]
                let len = 1usize;
                let req = mpi_iallreduce_in_place(&mut buf[..len], &comm.world);
                ReduceHandle {
                    inner: ReduceHandleInner::MpiScalar(MpiScalarState {
                        req,
                        buf,
                        len,
                        convert: |slice| {
                            #[cfg(feature = "complex")]
                            {
                                S::from_parts(slice[0], slice[1])
                            }
                            #[cfg(not(feature = "complex"))]
                            {
                                S::from_real(slice[0])
                            }
                        },
                        complete: false,
                    }),
                }
            }
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_) => {
                let (tx, rx) = std::sync::mpsc::channel();
                let local = x;
                rayon::spawn_fifo(move || {
                    let _ = tx.send(local);
                });
                ReduceHandle {
                    inner: ReduceHandleInner::Rayon { rx },
                }
            }
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => ReduceHandle::ready(x),
        }
    }

    fn iallreduce_sum_vec_r(&self, buf: Vec<R>) -> ReduceHandle<Vec<R>> {
        record_reduction(buf.len());
        if !self.async_enabled() {
            let mut out = buf;
            self.reduce_vec_in_place(&mut out);
            return ReduceHandle::ready(out);
        }
        match &self.comm {
            UniverseComm::NoComm(_) => ReduceHandle::ready(buf),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => {
                let mut buf = buf;
                let req = mpi_iallreduce_in_place(buf.as_mut_slice(), &comm.world);
                ReduceHandle {
                    inner: ReduceHandleInner::MpiVec(MpiVecState {
                        req,
                        buf,
                        convert: |v| v,
                        complete: false,
                    }),
                }
            }
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_) => {
                let (tx, rx) = std::sync::mpsc::channel();
                rayon::spawn_fifo(move || {
                    let _ = tx.send(buf);
                });
                ReduceHandle {
                    inner: ReduceHandleInner::Rayon { rx },
                }
            }
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => ReduceHandle::ready(buf),
        }
    }
}

impl UniverseComm {
    pub fn reduction_engine(&self, opts: &ReductOptions) -> Arc<dyn ReductionEngine> {
        Arc::new(CommReductionEngine::new(self.clone(), opts.clone()))
    }
}

#[cfg(feature = "mpi")]
fn mpi_iallreduce_in_place(
    buf: &mut [R],
    comm: &mpi::topology::SimpleCommunicator,
) -> mpi::ffi::MPI_Request {
    if buf.is_empty() {
        return unsafe { mpi::ffi::RSMPI_REQUEST_NULL };
    }
    let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        mpi::ffi::MPI_Iallreduce(
            buf.as_ptr() as *const std::ffi::c_void,
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            buf.len() as i32,
            mpi::ffi::RSMPI_DOUBLE,
            mpi::ffi::RSMPI_SUM,
            comm.as_raw(),
            &mut req,
        )
    };
    debug_assert_eq!(rc, 0);
    req
}

#[cfg(feature = "mpi")]
fn mpi_test_request(req: &mut mpi::ffi::MPI_Request) -> bool {
    let mut flag = 0;
    let rc = unsafe { mpi::ffi::MPI_Test(req, &mut flag, mpi::ffi::RSMPI_STATUS_IGNORE) };
    debug_assert_eq!(rc, 0);
    flag != 0
}

#[cfg(feature = "mpi")]
fn mpi_wait_request(req: &mut mpi::ffi::MPI_Request) {
    let rc = unsafe { mpi::ffi::MPI_Wait(req, mpi::ffi::RSMPI_STATUS_IGNORE) };
    debug_assert_eq!(rc, 0);
}

#[cfg(test)]
mod tests {
    use crate::parallel::{NoComm, UniverseComm};
    use crate::utils::reduction::{ReductExec, ReductOptions};

    #[test]
    fn no_comm_reduction_is_ready() {
        let comm = UniverseComm::NoComm(NoComm);
        let engine = comm.reduction_engine(&ReductOptions::default());
        let handle = engine.iallreduce_sum_r(1.0);
        assert!(handle.is_ready());
        assert_eq!(handle.wait(), 1.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn reproducible_forces_sync_for_rayon() {
        let comm = UniverseComm::Rayon(crate::parallel::rayon_comm::RayonComm::new());
        let mut opts = ReductOptions::default();
        opts.exec = ReductExec::Async;
        opts.reproducible = true;
        let engine = comm.reduction_engine(&opts);
        let handle = engine.iallreduce_sum_r(2.0);
        assert!(handle.is_ready());
        assert_eq!(handle.wait(), 2.0);
    }
}
