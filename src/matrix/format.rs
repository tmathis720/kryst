//! Real-only format conversion trait used by AMG helpers.
//!
//! While the `AsFormat` trait is generic, current implementations target
//! `CsrMatrix<f64>`, `CscMatrix<f64>`, and dense `Mat<f64>` to support
//! real-valued operators in AMG/factorization workflows.
use crate::algebra::scalar::KrystScalar;
use crate::matrix::sparse_api::{CscMatRef, CsrMatRef};
use std::sync::Arc;

/// High-level format hints that preconditioners can request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatHint {
    Csr,
    Dense,
    Csc,
}

/// Trait for converting matrices into specific formats in a backend-agnostic way.
pub trait AsFormat<S: KrystScalar> {
    /// Backend CSR type.
    type Csr: CsrMatRef<S> + 'static;
    /// Backend CSC type.
    type Csc: CscMatRef<S> + 'static;

    /// Borrow as CSR if already in that format.
    fn as_csr(&self) -> Option<&Self::Csr> {
        None
    }

    /// Convert to CSR and cache the result.
    fn to_csr_cached(&self, drop_tol: S::Real) -> Arc<Self::Csr>;

    /// Borrow as CSC if already in that format.
    fn as_csc(&self) -> Option<&Self::Csc> {
        None
    }

    /// Convert to CSC and cache the result.
    fn to_csc_cached(&self, drop_tol: S::Real) -> Arc<Self::Csc>;
}
