//! Backend implementations for matrix abstractions.

use crate::algebra::prelude::*;

/// Describes the dense and sparse storage types a backend exposes along with
/// conversion hooks between them.
pub trait SparseBackend<S: KrystScalar> {
    /// CSR matrix type for this backend.
    type Csr: Send + Sync + 'static;
    /// CSC matrix type for this backend.
    type Csc: Send + Sync + 'static;
    /// Dense matrix type for this backend.
    type Dense: Send + Sync + 'static;

    /// Convert a dense matrix into CSR (with drop tolerance).
    fn csr_from_dense(dense: &Self::Dense, drop_tol: S::Real) -> Self::Csr;

    /// Convert CSR → CSC without densifying.
    fn csc_from_csr(csr: &Self::Csr, drop_tol: S::Real) -> Self::Csc;

    /// Convert CSC → CSR without densifying.
    fn csr_from_csc(csc: &Self::Csc, drop_tol: S::Real) -> Self::Csr;

    /// Convert CSR → dense.
    fn dense_from_csr(csr: &Self::Csr) -> Self::Dense;

    /// Convert CSC → dense.
    fn dense_from_csc(csc: &Self::Csc) -> Self::Dense;
}

#[cfg(feature = "backend-faer")]
pub mod faer;

#[cfg(feature = "backend-faer")]
pub use faer::{DefaultCscMat, DefaultCsrMat, DefaultDenseMat, FaerBackend};

/// Alias for the default backend selected by current feature flags.
#[cfg(feature = "backend-faer")]
pub type DefaultBackend = crate::matrix::backend::faer::FaerBackend;
