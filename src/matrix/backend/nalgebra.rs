#![cfg(feature = "backend-nalgebra")]

use std::sync::Arc;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::format::OpFormat;
use crate::matrix::op::LinOp;

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
