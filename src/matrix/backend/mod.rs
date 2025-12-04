//! Backend implementations for matrix abstractions.

#[cfg(feature = "backend-faer")]
pub mod faer;

#[cfg(feature = "backend-faer")]
pub use faer::{DefaultCscMat, DefaultCsrMat, DefaultDenseMat, FaerBackend};
