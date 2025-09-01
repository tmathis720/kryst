//! Thread-pool sizing and tuning for shared-memory parallelism (Rayon).
//!
//! # Overview
//! When the crate is compiled with the `rayon` feature, Kryst builds a single
//! global Rayon pool and reuses it across calls. The effective number of threads
//! is chosen as follows (once per process):
//!
//! 1. If `KRYST_THREADS` is set, use that value.
//! 2. Else if `RAYON_NUM_THREADS` is set, use that value.
//! 3. Else use `num_cpus::get()`.
//!
//! If running under MPI, we size the pool per rank as
//! `max(1, total_threads / mpi_size)` to avoid oversubscription.
//!
//! # Environment variables
//! - `KRYST_THREADS`: total Rayon threads (preferred; overrides Rayon default).
//! - `RAYON_NUM_THREADS`: standard Rayon override (used if `KRYST_THREADS` unset).
//! - `KRYST_PAR_CUTOFF`: row-count threshold (default `DEFAULT_PAR_CUTOFF`) used by
//!   [`CsrOp::matvec`](crate::matrix::op::CsrOp) to decide when to use the
//!   parallel SpMV path.
//!
//! # Examples
//! ```no_run
//! // Single-node tuning
//! std::env::set_var("KRYST_THREADS", "32");      // prefer a bigger pool
//! std::env::set_var("KRYST_PAR_CUTOFF", "8192"); // only parallelize big SpMVs
//!
//! // Under MPI (e.g., 4 ranks), each rank gets floor(32/4) = 8 threads.
//! ```
use std::sync::OnceLock;

#[cfg(feature = "rayon")]
use rayon::ThreadPoolBuilder;

/// Default row-count cutoff for enabling parallel SpMV in `CsrOp::matvec`.
pub const DEFAULT_PAR_CUTOFF: usize = 4096;

/// Helper to read an environment variable as usize.
/// Falls back to `default` if the variable is not set or invalid.
pub fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

/// One-time computed number of Rayon worker threads we actually use.
static EFFECTIVE_THREADS: OnceLock<usize> = OnceLock::new();

/// Decide and initialize the global Rayon thread pool exactly once.
/// - `mpi_size`: number of MPI ranks in the current communicator (>= 1)
/// - If env `KRYST_THREADS` is set, it overrides the global CPU count.
/// - If not set, we fall back to `RAYON_NUM_THREADS`, then `num_cpus::get()`.
/// - Per-rank threads = floor(total / mpi_size), clamped to >= 1.
///
/// Returns the number of threads actually used for Rayon.
pub fn init_global_rayon_pool(mpi_size: usize) -> usize {
    #[cfg(not(feature = "rayon"))]
    {
        let _ = mpi_size; // silence warning
        return 1;
    }

    #[cfg(feature = "rayon")]
    {
        *EFFECTIVE_THREADS.get_or_init(|| {
            let total = std::env::var("KRYST_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .or_else(|| {
                    std::env::var("RAYON_NUM_THREADS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or_else(num_cpus::get);

            let threads = std::cmp::max(1, total / std::cmp::max(1, mpi_size));
            // Build the global pool once. If someone built it earlier, this is a no-op.
            let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();
            threads
        })
    }
}

/// Returns how many Rayon threads we are running with (after init).
pub fn current_rayon_threads() -> usize {
    #[cfg(feature = "rayon")]
    {
        // If no pool yet, Rayon falls back to a default; prefer our recorded number if any.
        EFFECTIVE_THREADS
            .get()
            .copied()
            .unwrap_or_else(rayon::current_num_threads)
    }
    #[cfg(not(feature = "rayon"))]
    {
        1
    }
}

/// A light "guard" for clarity: constructing this ensures the pool is initialized.
/// Note: The global pool cannot be destroyed; this is just an explicit init point.
pub struct ThreadPoolGuard {
    threads: usize,
}

impl ThreadPoolGuard {
    /// Initialize the global pool for a communicator of size `mpi_size`.
    pub fn new_per_rank(mpi_size: usize) -> Self {
        let threads = init_global_rayon_pool(mpi_size);
        Self { threads }
    }

    /// Number of threads being used.
    pub fn threads(&self) -> usize {
        self.threads
    }
}
