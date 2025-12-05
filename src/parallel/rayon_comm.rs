//! Rayon-based parallel communication implementation for shared-memory environments.
//!
//! This module provides a `RayonComm` struct that implements the `Comm` trait for shared-memory
//! environments using the Rayon thread pool. Collective operations are local, reflecting the fact
//! that there is no inter-process communication in this mode.
//!
//! # Usage
//! Use `RayonComm` as the communicator in solver contexts to enable parallel computation
//! on a single node using all available CPU cores.
//!
//! # References
//! - [Rayon documentation](https://docs.rs/rayon)
//! - [num_cpus documentation](https://docs.rs/num_cpus)

use rayon::scope;

/// Shared-memory communicator using Rayon for parallelism.
///
/// Implements the `Comm` trait for shared-memory parallelism. All collective operations
/// are no-ops or local, as there is no inter-process communication.
#[derive(Clone)]
pub struct RayonComm;

impl Default for RayonComm {
    fn default() -> Self {
        Self::new()
    }
}

impl RayonComm {
    /// Creates a new `RayonComm` and initializes the global Rayon thread pool
    /// to use all available CPU cores.
    ///
    /// If the global thread pool is already initialized, this is a no-op.
    pub fn new() -> Self {
        #[cfg(feature = "rayon")]
        {
            crate::parallel::threads::init_global_rayon_pool(1);
        }
        RayonComm
    }
}

impl super::Comm for RayonComm {
    type Vec = Vec<f64>;
    type Request<'a> = ();

    /// Returns the rank of the current process (always 0 in shared memory).
    fn rank(&self) -> usize {
        0
    }

    /// Returns the number of parallel workers (number of CPU cores).
    fn size(&self) -> usize {
        crate::parallel::threads::current_rayon_threads()
    }

    /// Synchronization barrier (no-op in shared memory, but uses a Rayon scope for API compatibility).
    fn barrier(&self) {
        scope(|_| {});
    }

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

    /// All‐reduce a scalar (sum) across ranks - new trait method
    fn all_reduce_f64(&self, local: f64) -> f64 {
        local // No-op for shared memory
    }

    /// Split this communicator into sub‐colors
    fn split(&self, _color: i32, _key: i32) -> super::UniverseComm {
        super::UniverseComm::Rayon(RayonComm::new()) // For shared memory, just return a new instance
    }

    fn irecv_from<'a>(&'a self, _buf: &'a mut [f64], _src: i32) -> Self::Request<'a> {}
    fn isend_to<'a>(&'a self, _buf: &'a [f64], _dest: i32) -> Self::Request<'a> {}
    fn irecv_from_u64<'a>(&'a self, _buf: &'a mut [u64], _src: i32) -> Self::Request<'a> {}
    fn isend_to_u64<'a>(&'a self, _buf: &'a [u64], _dest: i32) -> Self::Request<'a> {}
    fn wait_all<'a>(&self, _reqs: &mut [Self::Request<'a>]) {}
}
