//! Utility modules for logging, convergence checks, graph coloring, reordering, and profiling.

pub mod coloring;
pub mod convergence;
pub mod matrix_market;
pub mod monitor;
pub mod profiling;
pub mod reordering;
pub mod permutation;
#[cfg(feature = "tuning")]
pub mod tuning;
pub mod partition;
