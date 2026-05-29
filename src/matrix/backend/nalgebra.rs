#![cfg(feature = "backend-nalgebra")]

use std::sync::Arc;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::backend::SparseBackend;
use crate::matrix::format::{BackendFormatSupport, OpFormat};
use crate::matrix::op::LinOp;
use crate::parallel::Comm;

/// Marker type for the nalgebra backend.
pub struct NalgebraBackend;

impl<S> SparseBackend<S> for NalgebraBackend
where
    S: KrystScalar<Real = f64>,
{
    const FORMAT_SUPPORT: BackendFormatSupport =
        BackendFormatSupport::new(true, false, false, false);

    type Csr = ();
    type Csc = ();
    type Dense = nalgebra::DMatrix<S::Real>;

    fn csr_from_dense(_dense: &Self::Dense, _drop_tol: S::Real) -> Result<Self::Csr, KError> {
        Err(KError::Unsupported(
            "nalgebra backend does not support sparse formats",
        ))
    }

    fn csc_from_csr(_csr: &Self::Csr, _drop_tol: S::Real) -> Self::Csc {
        unreachable!("nalgebra backend does not support sparse formats")
    }

    fn csr_from_csc(_csc: &Self::Csc, _drop_tol: S::Real) -> Self::Csr {
        unreachable!("nalgebra backend does not support sparse formats")
    }

    fn dense_from_csr(_csr: &Self::Csr) -> Result<Self::Dense, KError> {
        Err(KError::Unsupported(
            "nalgebra backend does not support sparse formats",
        ))
    }

    fn dense_from_csc(_csc: &Self::Csc) -> Result<Self::Dense, KError> {
        Err(KError::Unsupported(
            "nalgebra backend does not support sparse formats",
        ))
    }
}

pub fn try_materialize(
    op: Arc<dyn LinOp<S = S>>,
    want: OpFormat,
    _drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    match want {
        OpFormat::Dense => {
            if op
                .as_any()
                .is::<crate::matrix::op_nalgebra::NalgebraDenseOp>()
            {
                return Ok(op);
            }

            let comm = op.comm();
            if !comm.is_trivial() && comm.size() > 1 {
                return Err(KError::Unsupported(
                    "nalgebra dense materialization requires a trivial communicator",
                ));
            }

            #[cfg(feature = "nalgebra-explicit-dense")]
            {
                return crate::matrix::op_nalgebra::explicit_dense_from_linop(op);
            }

            Err(KError::Unsupported(
                "nalgebra dense materialization requires NalgebraDenseOp (or enable nalgebra-explicit-dense)",
            ))
        }
        OpFormat::Any => Ok(op),
        _ => Err(KError::Unsupported(
            "nalgebra backend does not support the requested format",
        )),
    }
}

pub fn try_materialize_ref(
    op: &dyn LinOp<S = S>,
    want: OpFormat,
    _drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    match want {
        OpFormat::Dense => {
            if let Some(dense) = op
                .as_any()
                .downcast_ref::<crate::matrix::op_nalgebra::NalgebraDenseOp>()
            {
                return Ok(Arc::new(dense.clone()));
            }

            Err(KError::Unsupported(
                "nalgebra dense materialization requires NalgebraDenseOp",
            ))
        }
        _ => Err(KError::Unsupported(
            "nalgebra backend does not support the requested format",
        )),
    }
}
