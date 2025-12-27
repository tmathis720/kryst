//! Basic numeric traits and operations used throughout the crate.

pub mod blas;
pub mod bridge;
#[cfg(feature = "backend-faer")]
pub mod dense;
pub mod parallel;
pub mod parallel_cfg;
pub mod prelude;
pub mod scalar;

pub use parallel_cfg::{ParallelTune, parallel_tune, set_parallel_tune};
pub use scalar::{KrystScalar, R, S};
