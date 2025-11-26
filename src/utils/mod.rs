//! Utility modules for logging, convergence checks, graph coloring, reordering, and profiling.

pub mod buffer_pool;
pub mod coloring;
pub mod convergence;
#[cfg(feature = "backend-faer")]
pub mod matrix_market;
pub mod merge;
pub mod metrics;
pub mod monitor;
pub mod partition;
pub mod permutation;
pub mod profiling;
pub mod reduction;
#[cfg(feature = "backend-faer")]
pub mod reordering;
#[cfg(feature = "tuning")]
pub mod tuning;

pub use monitor::{Event, Monitor, NullMonitor, TextMonitor};
