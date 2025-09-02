//! Utility modules for logging, convergence checks, graph coloring, reordering, and profiling.

pub mod coloring;
pub mod convergence;
pub mod matrix_market;
pub mod merge;
pub mod metrics;
pub mod monitor;
pub mod partition;
pub mod permutation;
pub mod profiling;
pub mod reordering;
#[cfg(feature = "tuning")]
pub mod tuning;
