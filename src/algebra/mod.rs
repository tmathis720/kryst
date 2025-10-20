//! Basic numeric traits and operations used throughout the crate.

pub mod blas;
pub mod bridge;
pub mod parallel;
pub mod parallel_cfg;
pub mod prelude;
pub mod scalar;

pub use scalar::{KrystScalar, R, S};
