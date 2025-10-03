//! Basic numeric traits and operations used throughout the crate.

pub mod blas;
pub mod bridge;
pub mod prelude;
pub mod scalar;

pub use scalar::{KrystScalar, R, S};
