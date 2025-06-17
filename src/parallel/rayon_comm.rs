//! Rayon-based parallel communication implementation for shared-memory environments.
//!
//! This module provides a `RayonComm` struct that implements the `Comm` trait for parallel
//! operations using the Rayon thread pool. It is intended for use in shared-memory settings
//! where distributed communication is not required. The implementation provides parallel
//! matrix-vector multiplication and no-op collective operations, mimicking MPI-like interfaces.
//!
//! # Usage
//! Use `RayonComm` as the communicator in solver contexts to enable parallel computation
//! on a single node using all available CPU cores.
//!
//! # References
//! - [Rayon documentation](https://docs.rs/rayon)
//! - [num_cpus documentation](https://docs.rs/num_cpus)

use rayon::prelude::*;

/// Shared-memory communicator using Rayon for parallelism.
///
/// Implements the `Comm` trait for shared-memory parallelism. All collective operations
/// are no-ops or local, as there is no inter-process communication.
pub struct RayonComm;

impl RayonComm {
    /// Creates a new `RayonComm` and initializes the global Rayon thread pool
    /// to use all available CPU cores.
    ///
    /// If the global thread pool is already initialized, this is a no-op.
    pub fn new() -> Self {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build_global()
            .ok();
        RayonComm
    }
}

impl super::Comm for RayonComm {
    type Vec = Vec<f64>;

    /// Returns the rank of the current process (always 0 in shared memory).
    fn rank(&self) -> usize { 0 }

    /// Returns the number of parallel workers (number of CPU cores).
    fn size(&self) -> usize { num_cpus::get() }

    /// Synchronization barrier (no-op in shared memory, but uses a Rayon scope for API compatibility).
    fn barrier(&self) { rayon::scope(|_| {}); }

    /// Mimics scatter operation by copying a chunk of the global array to the output buffer.
    ///
    /// # Arguments
    /// * `global` - The global data array.
    /// * `out` - The output buffer to fill.
    /// * `root` - The root rank (used as an offset multiplier).
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], root: usize) {
        let n = out.len();
        let start = root * n;
        out.clone_from_slice(&global[start..start + n]);
    }

    /// Mimics gather operation by copying the local buffer into the output vector.
    ///
    /// # Arguments
    /// * `local` - The local data buffer.
    /// * `out` - The output vector to fill.
    /// * `_root` - The root rank (unused).
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        out.clear();
        out.extend_from_slice(local);
    }

    /// All-reduce operation (no-op, returns input value).
    ///
    /// In shared memory, all-reduce is unnecessary, so this just returns the input.
    fn all_reduce(&self, x: f64) -> f64 {
        x // No-op for shared memory
    }

    /// Parallel matrix-vector multiplication using Rayon.
    ///
    /// # Arguments
    /// * `a` - Matrix (faer::Mat<f64>).
    /// * `x` - Input vector.
    /// * `y` - Output vector (will be overwritten).
    fn parallel_mat_vec(&self, a: &faer::Mat<f64>, x: &[f64], y: &mut [f64]) {
        assert_eq!(a.ncols(), x.len(), "Matrix columns must match input vector length");
        assert_eq!(a.nrows(), y.len(), "Matrix rows must match output vector length");
        // Compute y = A * x in parallel over rows
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = (0..a.ncols()).map(|j| a[(i, j)] * x[j]).sum();
        });
    }
}
