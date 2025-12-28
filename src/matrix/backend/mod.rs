//! Backend implementations for matrix abstractions.
//!
//! Each backend declares its format coverage via [`SparseBackend::FORMAT_SUPPORT`]
//! so new backends must explicitly state which of Dense/CSR/CSC/BlockCSR are
//! supported before materialization attempts are routed.

use std::sync::Arc;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::format::{BackendFormatSupport, OpFormat};
use crate::matrix::op::LinOp;

/// Describes the dense and sparse storage types a backend exposes along with
/// conversion hooks between them.
pub trait SparseBackend<S: KrystScalar> {
    /// Format coverage for this backend. Keep in sync with materialization code.
    const FORMAT_SUPPORT: BackendFormatSupport;
    /// CSR matrix type for this backend.
    type Csr: Send + Sync + 'static;
    /// CSC matrix type for this backend.
    type Csc: Send + Sync + 'static;
    /// Dense matrix type for this backend.
    type Dense: Send + Sync + 'static;

    /// Convert a dense matrix into CSR (with drop tolerance).
    fn csr_from_dense(dense: &Self::Dense, drop_tol: S::Real) -> Result<Self::Csr, KError>;

    /// Convert CSR → CSC without densifying.
    fn csc_from_csr(csr: &Self::Csr, drop_tol: S::Real) -> Self::Csc;

    /// Convert CSC → CSR without densifying.
    fn csr_from_csc(csc: &Self::Csc, drop_tol: S::Real) -> Self::Csr;

    /// Convert CSR → dense.
    fn dense_from_csr(csr: &Self::Csr) -> Result<Self::Dense, KError>;

    /// Convert CSC → dense.
    fn dense_from_csc(csc: &Self::Csc) -> Result<Self::Dense, KError>;

    /// Validate that an [`OpFormat`] is supported by this backend.
    #[inline]
    fn ensure_supports(format: OpFormat) -> Result<(), KError> {
        if Self::FORMAT_SUPPORT.supports(format) {
            Ok(())
        } else {
            Err(unsupported_format_err(format))
        }
    }
}

fn unsupported_format_err(want: OpFormat) -> KError {
    match want {
        OpFormat::Dense => KError::Unsupported(
            "materialize: cannot produce Dense; enable backend-faer or backend-nalgebra",
        ),
        OpFormat::Csr => KError::Unsupported(
            "materialize: cannot produce Csr; enable backend-faer or another sparse backend",
        ),
        OpFormat::Csc => KError::Unsupported(
            "materialize: cannot produce Csc; enable backend-faer or another sparse backend",
        ),
        OpFormat::BlockCsr => {
            KError::Unsupported("materialize: BlockCsr is not supported yet")
        }
        OpFormat::Any => KError::Unsupported("materialize: OpFormat::Any requires no conversion"),
    }
}

#[inline]
fn backend_format_support() -> BackendFormatSupport {
    #[cfg(feature = "backend-faer")]
    {
        return <crate::matrix::backend::faer::FaerBackend as SparseBackend<S>>::FORMAT_SUPPORT;
    }
    #[cfg(feature = "backend-nalgebra")]
    {
        return <crate::matrix::backend::nalgebra::NalgebraBackend as SparseBackend<S>>::FORMAT_SUPPORT;
    }
    #[cfg(feature = "backend-naive")]
    {
        return <crate::matrix::backend::naive::NaiveBackend as SparseBackend<S>>::FORMAT_SUPPORT;
    }
    BackendFormatSupport::new(false, false, false, false)
}

/// Central entry point used by KSP/PCs to request a specific operator format.
pub fn materialize(
    op: Arc<dyn LinOp<S = S>>,
    want: OpFormat,
    drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    if want.is_any() || op.format() == want {
        return Ok(op);
    }

    if !backend_format_support().supports(want) {
        return Err(unsupported_format_err(want));
    }

    #[cfg(feature = "backend-faer")]
    if let Ok(m) = crate::matrix::backend::faer::try_materialize(op.clone(), want, drop_tol) {
        return Ok(m);
    }

    #[cfg(feature = "backend-nalgebra")]
    if let Ok(m) = crate::matrix::backend::nalgebra::try_materialize(op.clone(), want, drop_tol) {
        return Ok(m);
    }

    Err(unsupported_format_err(want))
}

/// Materialize when only a `&dyn LinOp` is available (e.g., inside PcChain).
pub fn materialize_ref(
    op: &dyn LinOp<S = S>,
    want: OpFormat,
    drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    if want.is_any() {
        return Err(unsupported_format_err(want));
    }

    if !backend_format_support().supports(want) {
        return Err(unsupported_format_err(want));
    }

    #[cfg(feature = "backend-faer")]
    if let Ok(m) = crate::matrix::backend::faer::try_materialize_ref(op, want, drop_tol) {
        return Ok(m);
    }

    #[cfg(feature = "backend-nalgebra")]
    if let Ok(m) = crate::matrix::backend::nalgebra::try_materialize_ref(op, want, drop_tol) {
        return Ok(m);
    }

    Err(unsupported_format_err(want))
}

#[cfg(feature = "backend-faer")]
pub mod faer;

#[cfg(feature = "backend-faer")]
pub use faer::{DefaultCscMat, DefaultCsrMat, DefaultDenseMat, FaerBackend};

/// Alias for the default backend selected by current feature flags.
#[cfg(feature = "backend-faer")]
pub type DefaultBackend = crate::matrix::backend::faer::FaerBackend;

#[cfg(feature = "backend-nalgebra")]
pub mod nalgebra;

#[cfg(feature = "backend-naive")]
pub mod naive;
