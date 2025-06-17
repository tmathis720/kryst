/// MPI-based parallel communication module.
///
/// This module provides an implementation of the `Comm` trait using the MPI (Message Passing Interface)
/// backend for distributed-memory parallelism. It enables communication and collective operations
/// between processes in a parallel environment, such as scatter, gather, barrier synchronization,
/// and all-reduce. The implementation is only available when the `mpi` feature is enabled.
///
/// # Usage
///
/// - The `MpiComm` struct wraps an MPI communicator and exposes methods for parallel operations.
/// - The `Comm` trait is implemented for `MpiComm`, allowing it to be used as a drop-in replacement
///   for serial or other parallel communication backends.
///
/// # References
/// - [MPI Standard](https://www.mpi-forum.org/)
///
/// # Example
/// ```no_run
/// #[cfg(feature = "mpi")]
/// let comm = MpiComm::new();
/// println!("Rank: {} / {}", comm.rank(), comm.size());
/// comm.barrier();
/// ```

#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;

/// MPI communicator wrapper for distributed parallelism.
///
/// Holds the MPI world communicator, the rank of the current process, and the total number of processes.
#[cfg(feature = "mpi")]
pub struct MpiComm {
    /// The MPI world communicator (all processes in the job).
    pub world: SimpleCommunicator,
    /// The rank (ID) of this process within the communicator.
    pub rank: usize,
    /// The total number of processes in the communicator.
    pub size: usize,
}

#[cfg(feature = "mpi")]
impl MpiComm {
    /// Initializes MPI and constructs a new `MpiComm` instance.
    ///
    /// # Panics
    /// Panics if MPI initialization fails.
    pub fn new() -> Self {
        let universe = mpi::initialize().unwrap();
        let world    = universe.world();
        let rank     = world.rank() as usize;
        let size     = world.size() as usize;
        MpiComm { world, rank, size }
    }
}

#[cfg(feature = "mpi")]
impl super::Comm for MpiComm {
    type Vec = Vec<f64>;

    /// Returns the rank (ID) of this process.
    fn rank(&self) -> usize { self.rank }
    /// Returns the total number of processes in the communicator.
    fn size(&self) -> usize { self.size }
    /// Synchronizes all processes at a barrier.
    fn barrier(&self) { self.world.barrier(); }

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
        // Only the root process provides the global array; others provide an empty slice.
        self.world
            .process_at_rank(root as i32)
            .scatter_into_root(global, out);
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
        // Only the root process allocates the receive buffer; others use an empty Vec.
        let mut recvbuf = if self.rank == root {
            vec![local[0].clone(); local.len() * self.size]
        } else {
            Vec::new()
        };
        self.world
            .process_at_rank(root as i32)
            .gather_into_root(local, &mut recvbuf);
        if self.rank == root {
            *out = recvbuf;
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
        self.world.all_reduce_into(&x, &mut y, &SystemOperation::sum());
        y
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
}
