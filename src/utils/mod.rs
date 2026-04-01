//! Utility modules for logging, convergence checks, graph coloring, reordering, and profiling.

pub mod buffer_pool;
pub mod coloring;
pub mod conditioning;
pub mod convergence;
pub mod diagnostics;
pub mod invariants;
#[cfg(feature = "backend-faer")]
pub mod matrix_market;
pub mod matrix_screening;
pub mod merge;
pub mod metrics;
pub mod monitor;
pub mod partition;
pub mod permutation;
pub mod profiling;
pub mod reduction;
#[cfg(feature = "backend-faer")]
pub mod reordering;
#[cfg(all(feature = "tuning", not(feature = "complex")))]
pub mod tuning;

pub use metrics::true_residual_norm;
pub use monitor::{Event, Monitor, NullMonitor, TextMonitor};
