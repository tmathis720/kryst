//! MPI-based parallel communication module.
//!
//! This module provides an implementation of the `Comm` trait using the MPI (Message Passing Interface)
//! backend for distributed-memory parallelism. It enables communication and collective operations
//! between processes in a parallel environment, such as scatter, gather, barrier synchronization,
//! and all-reduce.
//!
//! # Example
//! ```no_run
//! let comm = MpiComm::new();
//! println!("Rank: {} / {}", comm.rank(), comm.size());
//! comm.barrier();
//! ```

use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use std::sync::{Arc, OnceLock};

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
}

unsafe impl Send for MpiComm {}
unsafe impl Sync for MpiComm {}

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

        MpiComm { world, rank, size }
    }

    /// Best-effort constructor that returns `None` if initialization fails.
    pub fn try_new() -> Option<Self> {
        std::panic::catch_unwind(|| Self::new()).ok()
    }
}

impl super::Comm for MpiComm {
    type Vec = Vec<f64>;
    type Request<'a> = ();

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
            .all_reduce_into(&x, &mut y, &SystemOperation::sum());
        y
    }

    /// All‐reduce a scalar (sum) across ranks - new trait method
    fn all_reduce_f64(&self, local: f64) -> f64 {
        self.all_reduce(local)
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
        super::UniverseComm::Mpi(Arc::new(MpiComm {
            world: sub,
            rank,
            size,
        }))
    }

    /// Parallel matrix-vector multiplication (currently serial, placeholder for distributed version).
    ///
    /// - `a`: The matrix (all rows available on all processes).
    /// - `x`: The input vector.
    /// - `y`: The output vector (to be filled with the result).
    ///
    /// # Note
    /// This implementation is currently serial and does not partition the matrix or vector by rank.
    /// In a true distributed setting, the matrix and vectors should be partitioned and communication
    /// performed as needed.
    fn parallel_mat_vec(&self, a: &faer::Mat<f64>, x: &[f64], y: &mut [f64]) {
        // For now, just serial mat-vec. TODO: partition by rank for distributed mat-vec.
        assert_eq!(a.ncols(), x.len());
        assert_eq!(a.nrows(), y.len());
        for i in 0..a.nrows() {
            y[i] = 0.0;
            for j in 0..a.ncols() {
                y[i] += a[(i, j)] * x[j];
            }
        }
    }

    fn irecv_from<'a>(&'a self, buf: &'a mut [f64], src: i32) -> Self::Request<'a> {
        self.world.process_at_rank(src).receive_into(buf);
        ()
    }
    fn isend_to<'a>(&'a self, buf: &'a [f64], dest: i32) -> Self::Request<'a> {
        self.world.process_at_rank(dest).send(buf);
        ()
    }
    fn wait_all<'a>(&self, _reqs: &mut [Self::Request<'a>]) {}
}
