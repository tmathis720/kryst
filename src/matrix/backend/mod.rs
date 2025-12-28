//! Backend implementations for matrix abstractions.

use std::sync::Arc;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::format::OpFormat;
use crate::matrix::op::LinOp;

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
    fn csr_from_dense(dense: &Self::Dense, drop_tol: S::Real) -> Result<Self::Csr, KError>;

    /// Convert CSR → CSC without densifying.
    fn csc_from_csr(csr: &Self::Csr, drop_tol: S::Real) -> Self::Csc;

    /// Convert CSC → CSR without densifying.
    fn csr_from_csc(csc: &Self::Csc, drop_tol: S::Real) -> Self::Csr;

    /// Convert CSR → dense.
    fn dense_from_csr(csr: &Self::Csr) -> Result<Self::Dense, KError>;

    /// Convert CSC → dense.
    fn dense_from_csc(csc: &Self::Csc) -> Result<Self::Dense, KError>;
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

/// Central entry point used by KSP/PCs to request a specific operator format.
pub fn materialize(
    op: Arc<dyn LinOp<S = S>>,
    want: OpFormat,
    drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    if want.is_any() || op.format() == want {
        return Ok(op);
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
