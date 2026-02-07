#![cfg(feature = "backend-naive")]
//! Minimal backend stub used to exercise feature-gated backend wiring.
//!
//! This backend is intentionally sparse-agnostic. It exists as an integration
//! point for downstream experimentation or alternative storage types.

use std::sync::Arc;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::backend::SparseBackend;
use crate::matrix::format::{BackendFormatSupport, OpFormat};
use crate::matrix::op::LinOp;

/// Marker type for the naive backend.
pub struct NaiveBackend;

impl<S> SparseBackend<S> for NaiveBackend
where
    S: KrystScalar<Real = f64>,
{
    const FORMAT_SUPPORT: BackendFormatSupport =
        BackendFormatSupport::new(false, false, false, false);

    type Csr = ();
    type Csc = ();
    type Dense = ();

    fn csr_from_dense(_dense: &Self::Dense, _drop_tol: S::Real) -> Result<Self::Csr, KError> {
        Err(KError::Unsupported(
            "naive backend is a stub and does not support format conversions",
        ))
    }

    fn csc_from_csr(_csr: &Self::Csr, _drop_tol: S::Real) -> Self::Csc {
        unreachable!("naive backend is a stub and does not support format conversions")
    }

    fn csr_from_csc(_csc: &Self::Csc, _drop_tol: S::Real) -> Self::Csr {
        unreachable!("naive backend is a stub and does not support format conversions")
    }

    fn dense_from_csr(_csr: &Self::Csr) -> Result<Self::Dense, KError> {
        Err(KError::Unsupported(
            "naive backend is a stub and does not support format conversions",
        ))
    }

    fn dense_from_csc(_csc: &Self::Csc) -> Result<Self::Dense, KError> {
        Err(KError::Unsupported(
            "naive backend is a stub and does not support format conversions",
        ))
    }
}

pub fn try_materialize(
    _op: Arc<dyn LinOp<S = S>>,
    want: OpFormat,
    _drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    Err(KError::Unsupported(match want {
        OpFormat::Dense => "naive backend cannot materialize Dense",
        OpFormat::Csr => "naive backend cannot materialize Csr",
        OpFormat::Csc => "naive backend cannot materialize Csc",
        OpFormat::BlockCsr => "naive backend cannot materialize BlockCsr",
        OpFormat::Any => "naive backend cannot materialize OpFormat::Any",
    }))
}

pub fn try_materialize_ref(
    _op: &dyn LinOp<S = S>,
    want: OpFormat,
    _drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    Err(KError::Unsupported(match want {
        OpFormat::Dense => "naive backend cannot materialize Dense",
        OpFormat::Csr => "naive backend cannot materialize Csr",
        OpFormat::Csc => "naive backend cannot materialize Csc",
        OpFormat::BlockCsr => "naive backend cannot materialize BlockCsr",
        OpFormat::Any => "naive backend cannot materialize OpFormat::Any",
    }))
}
