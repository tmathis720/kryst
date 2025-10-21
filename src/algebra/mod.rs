//! Basic numeric traits and operations used throughout the crate.

pub mod blas;
pub mod bridge;
#[cfg(any(feature = "backend-faer"))]
pub mod dense;
pub mod parallel;
pub mod parallel_cfg;
pub mod prelude;
pub mod scalar;

pub use scalar::{KrystScalar, R, S};
