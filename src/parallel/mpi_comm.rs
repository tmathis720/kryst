//! MPI-based parallel communication module.
//!
//! This module provides an implementation of the `Comm` trait using the MPI (Message Passing Interface)
//! backend for distributed-memory parallelism. It enables communication and collective operations
//! between processes in a parallel environment, such as scatter, gather, barrier synchronization,
//! and all-reduce.
//!
//! # Example
//! ```no_run
//! use kryst::parallel::{Comm, MpiComm};
//!
//! let comm = MpiComm::new();
//! println!("Rank: {} / {}", comm.rank(), comm.size());
//! comm.barrier();
//! ```

use crate::algebra::prelude::*;
use mpi::collective::SystemOperation;
use mpi::raw::AsRaw;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

pub struct OwnedMpiRequest {
    pub(crate) handle: mpi::ffi::MPI_Request,
}

impl OwnedMpiRequest {
    pub(crate) fn new(handle: mpi::ffi::MPI_Request) -> Self {
        Self { handle }
    }

    pub(crate) fn wait(&mut self) {
        let rc = unsafe { mpi::ffi::MPI_Wait(&mut self.handle, mpi::ffi::RSMPI_STATUS_IGNORE) };
        debug_assert_eq!(rc, 0);
    }
}

/// MPI communicator wrapper for distributed parallelism.
///
/// Holds the MPI world communicator, the rank of the current process, and the total
/// number of processes. MPI itself is initialized exactly once for the entire
/// program via a global [`OnceLock`].
pub struct MpiComm {
    /// The MPI world communicator (all processes in the job).
    pub world: SimpleCommunicator,
    /// The rank (ID) of this process within the communicator.
    pub rank: usize,
    /// The total number of processes in the communicator.
    pub size: usize,
    /// Flag indicating whether deterministic reductions are requested.
    pub reproducible: AtomicBool,
}

unsafe impl Send for MpiComm {}
unsafe impl Sync for MpiComm {}

impl Default for MpiComm {
    fn default() -> Self {
        Self::new()
    }
}

// --- One-time MPI universe holder ---
static MPI_UNIVERSE: OnceLock<mpi::environment::Universe> = OnceLock::new();

fn universe() -> &'static mpi::environment::Universe {
    MPI_UNIVERSE.get_or_init(|| mpi::initialize().expect("MPI initialization failed"))
}

impl MpiComm {
    /// Initializes MPI once and constructs a new [`MpiComm`].
    pub fn new() -> Self {
        let world = universe().world().duplicate();
        let rank = world.rank() as usize;
        let size = world.size() as usize;

        #[cfg(feature = "rayon")]
        {
            crate::parallel::threads::init_global_rayon_pool(size);
        }

        MpiComm {
            world,
            rank,
            size,
            reproducible: AtomicBool::new(false),
        }
    }

    /// Best-effort constructor that returns `None` if initialization fails.
    pub fn try_new() -> Option<Self> {
        std::panic::catch_unwind(Self::new).ok()
    }

    #[inline]
    pub fn allreduce_sum_real(&self, v: R) -> R {
        use mpi::collective::SystemOperation;
        let mut acc = v;
        self.world
            .all_reduce_into(&v, &mut acc, SystemOperation::sum());
        acc
    }

    #[inline]
    pub fn allreduce_sum_scalar(&self, v: S) -> S {
        #[cfg(feature = "complex")]
        {
            let re = self.allreduce_sum_real(v.real());
            let im = self.allreduce_sum_real(v.imag());
            S::from_parts(re, im)
        }
        #[cfg(not(feature = "complex"))]
        {
            S::from_real(self.allreduce_sum_real(v.real()))
        }
    }

    pub fn allreduce_sum_scalars(&self, buf: &mut [S]) {
        if buf.is_empty() {
            return;
        }

        use mpi::collective::SystemOperation;

        #[cfg(feature = "complex")]
        {
            let n = buf.len();
            let mut re = vec![0.0f64; n];
            let mut im = vec![0.0f64; n];
            for (i, &z) in buf.iter().enumerate() {
                re[i] = z.real();
                im[i] = z.imag();
            }
            let mut re_sum = vec![0.0f64; n];
            let mut im_sum = vec![0.0f64; n];
            self.world
                .all_reduce_into(&re[..], &mut re_sum[..], SystemOperation::sum());
            self.world
                .all_reduce_into(&im[..], &mut im_sum[..], SystemOperation::sum());
            for (slot, (&r, &i)) in buf.iter_mut().zip(re_sum.iter().zip(im_sum.iter())) {
                *slot = S::from_parts(r, i);
            }
        }

        #[cfg(not(feature = "complex"))]
        {
            let mut tmp = vec![0.0f64; buf.len()];
            for (dst, &z) in tmp.iter_mut().zip(buf.iter()) {
                *dst = z.real();
            }
            let mut sum = vec![0.0f64; tmp.len()];
            self.world
                .all_reduce_into(&tmp[..], &mut sum[..], SystemOperation::sum());
            for (slot, &value) in buf.iter_mut().zip(sum.iter()) {
                *slot = S::from_real(value);
            }
        }
    }

    pub(crate) fn blocking_allreduce_sum_in_place(&self, buf: &mut [R]) {
        if buf.is_empty() {
            return;
        }

        let mut recv = vec![0.0f64; buf.len()];
        self.world
            .all_reduce_into(buf, &mut recv[..], SystemOperation::sum());
        buf.copy_from_slice(&recv);
    }

    pub(crate) fn immediate_allreduce_sum(&self, send: &[R], recv: &mut [R]) -> OwnedMpiRequest {
        if send.is_empty() {
            let handle = unsafe { mpi::ffi::RSMPI_REQUEST_NULL };
            return OwnedMpiRequest::new(handle);
        }

        debug_assert_eq!(send.len(), recv.len());

        let mut handle = MaybeUninit::<mpi::ffi::MPI_Request>::uninit();
        let rc = unsafe {
            mpi::ffi::MPI_Iallreduce(
                send.as_ptr() as *const c_void,
                recv.as_mut_ptr() as *mut c_void,
                send.len() as i32,
                mpi::ffi::RSMPI_DOUBLE,
                mpi::ffi::RSMPI_SUM,
                self.world.as_raw(),
                handle.as_mut_ptr(),
            )
        };
        debug_assert_eq!(rc, 0);
        let handle = unsafe { handle.assume_init() };
        OwnedMpiRequest::new(handle)
    }
}

impl super::Comm for MpiComm {
    type Vec = Vec<f64>;
    type Request<'a> = super::MpiRequest<'a>;

    /// Returns the rank (ID) of this process.
    fn rank(&self) -> usize {
        self.rank
    }
    /// Returns the total number of processes in the communicator.
    fn size(&self) -> usize {
        self.size
    }
    /// Synchronizes all processes at a barrier.
    fn barrier(&self) {
        self.world.barrier();
    }

    /// Distributes slices of a global array to all processes (scatter operation).
    ///
    /// - `global`: The full array to scatter (only used on root process).
    /// - `out`: The buffer to receive the scattered chunk (on each process).
    /// - `root`: The rank of the root process performing the scatter.
    fn scatter<T: Clone + mpi::datatype::Equivalence>(
        &self,
        global: &[T],
        out: &mut [T],
        root: usize,
    ) {
        let proc = self.world.process_at_rank(root as i32);
        if self.rank == root {
            proc.scatter_into_root(global, out);
        } else {
            proc.scatter_into(out);
        }
    }

    /// Gathers arrays from all processes to the root process (gather operation).
    ///
    /// - `local`: The local array to send from each process.
    /// - `out`: The buffer to receive the gathered data (only used on root process).
    /// - `root`: The rank of the root process collecting the data.
    fn gather<T: Clone + mpi::datatype::Equivalence>(
        &self,
        local: &[T],
        out: &mut Vec<T>,
        root: usize,
    ) {
        let proc = self.world.process_at_rank(root as i32);
        if self.rank == root {
            let mut recv = vec![local[0].clone(); local.len() * self.size];
            proc.gather_into_root(local, &mut recv);
            *out = recv;
        } else {
            proc.gather_into(local);
            out.clear();
        }
    }

    /// Performs an all-reduce sum operation across all processes.
    ///
    /// - `x`: The local value to be reduced.
    ///
    /// Returns the sum of `x` across all processes.
    fn all_reduce(&self, x: f64) -> f64 {
        use mpi::collective::SystemOperation;
        let mut y = x;
        self.world
            .all_reduce_into(&x, &mut y, SystemOperation::sum());
        y
    }

    /// All‐reduce a scalar (sum) across ranks - new trait method
    fn all_reduce_f64(&self, local: f64) -> f64 {
        self.all_reduce(local)
    }

    fn allreduce_sum2(&self, a: f64, b: f64) -> (f64, f64) {
        use mpi::collective::SystemOperation;
        let send = [a, b];
        let mut recv = [0.0f64; 2];
        self.world
            .all_reduce_into(&send, &mut recv, SystemOperation::sum());
        (recv[0], recv[1])
    }

    fn allreduce_sum_slice(&self, v: &mut [f64]) {
        use mpi::collective::SystemOperation;
        let mut out = vec![0.0f64; v.len()];
        self.world
            .all_reduce_into(v, &mut out[..], SystemOperation::sum());
        v.copy_from_slice(&out);
    }

    /// Split this communicator into sub‐colors
    fn split(&self, color: i32, key: i32) -> super::UniverseComm {
        use mpi::topology::Color;
        let sub = self
            .world
            .split_by_color_with_key(Color::with_value(color), key)
            .expect("MPI split failed");
        let rank = sub.rank() as usize;
        let size = sub.size() as usize;
        let repro = self.reproducible.load(Ordering::Relaxed);
        super::UniverseComm::Mpi(Arc::new(MpiComm {
            world: sub,
            rank,
            size,
            reproducible: AtomicBool::new(repro),
        }))
    }

    fn irecv_from<'a>(&'a self, buf: &'a mut [f64], src: i32) -> Self::Request<'a> {
        // Nonblocking receive via raw MPI
        let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
        let count = buf.len() as i32;
        let comm_raw = self.world.as_raw();
        let rc = unsafe {
            mpi::ffi::MPI_Irecv(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                count,
                mpi::ffi::RSMPI_DOUBLE,
                src,
                0,
                comm_raw,
                &mut req,
            )
        };
        debug_assert_eq!(rc, 0);
        super::MpiRequest {
            handle: req,
            _marker: std::marker::PhantomData,
        }
    }
    fn isend_to<'a>(&'a self, buf: &'a [f64], dest: i32) -> Self::Request<'a> {
        // Nonblocking send via raw MPI
        let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
        let count = buf.len() as i32;
        let comm_raw = self.world.as_raw();
        let rc = unsafe {
            mpi::ffi::MPI_Isend(
                buf.as_ptr() as *const std::ffi::c_void,
                count,
                mpi::ffi::RSMPI_DOUBLE,
                dest,
                0,
                comm_raw,
                &mut req,
            )
        };
        debug_assert_eq!(rc, 0);
        super::MpiRequest {
            handle: req,
            _marker: std::marker::PhantomData,
        }
    }
    fn irecv_from_u64<'a>(&'a self, buf: &'a mut [u64], src: i32) -> Self::Request<'a> {
        // Nonblocking receive of u64 via raw MPI
        let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
        let count = buf.len() as i32;
        let comm_raw = self.world.as_raw();
        let rc = unsafe {
            mpi::ffi::MPI_Irecv(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                count,
                mpi::ffi::RSMPI_UINT64_T,
                src,
                0,
                comm_raw,
                &mut req,
            )
        };
        debug_assert_eq!(rc, 0);
        super::MpiRequest {
            handle: req,
            _marker: std::marker::PhantomData,
        }
    }
    fn isend_to_u64<'a>(&'a self, buf: &'a [u64], dest: i32) -> Self::Request<'a> {
        // Nonblocking send of u64 via raw MPI
        let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
        let count = buf.len() as i32;
        let comm_raw = self.world.as_raw();
        let rc = unsafe {
            mpi::ffi::MPI_Isend(
                buf.as_ptr() as *const std::ffi::c_void,
                count,
                mpi::ffi::RSMPI_UINT64_T,
                dest,
                0,
                comm_raw,
                &mut req,
            )
        };
        debug_assert_eq!(rc, 0);
        super::MpiRequest {
            handle: req,
            _marker: std::marker::PhantomData,
        }
    }
    fn wait_all<'a>(&self, reqs: &mut [Self::Request<'a>]) {
        for rq in reqs {
            let rc = unsafe { mpi::ffi::MPI_Wait(&mut rq.handle, mpi::ffi::RSMPI_STATUS_IGNORE) };
            debug_assert_eq!(rc, 0);
        }
    }
}
