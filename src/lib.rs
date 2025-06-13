//! kryst: PETSc-style PC/KSP interface over Faer

#[cfg(feature = "parallel")]
pub mod parallel;

pub mod config;
pub mod context;
pub mod core;
pub mod error;
pub mod matrix;
pub mod preconditioner;
pub mod solver;
pub mod utils;

// Re-exports for convenience
pub use config::*;
pub use context::*;
pub use core::*;
pub use error::*;
pub use matrix::*;
pub use preconditioner::*;
pub use solver::*;
pub use utils::*;

// Re-export SolveStats at the crate root for convenience
pub use utils::convergence::SolveStats;
