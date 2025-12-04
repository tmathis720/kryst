#![cfg(feature = "backend-faer")]

// Re-export backend-specific AsFormat implementations.
#[allow(unused_imports)]
pub use crate::matrix::backend::faer::format::*;
