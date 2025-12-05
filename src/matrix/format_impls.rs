//! Backend-specific `AsFormat` implementations for real-valued matrices.
//!
//! These re-exports are intended for AMG-oriented helpers and assume `S = f64`
//! on the underlying storage types.
#![cfg(feature = "backend-faer")]

// Re-export backend-specific AsFormat implementations.
#[allow(unused_imports)]
pub(crate) use crate::matrix::backend::faer::format::*;
