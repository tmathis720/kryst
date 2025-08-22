use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};

#[cfg(feature = "superlu_dist")]
use crate::matrix::sparse::CsrMatrix;

pub struct SuperLuDistPc;

impl SuperLuDistPc {
    pub fn new() -> Self {
        Self
    }
}

impl Preconditioner for SuperLuDistPc {
    fn setup(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        #[cfg(feature = "superlu_dist")]
        {
            pmat.as_any()
                .downcast_ref::<CsrMatrix<f64>>()
                .ok_or_else(|| {
                    KError::InvalidInput("SuperLU_DIST PC requires CSR matrix".into())
                })?;
            Ok(())
        }

        #[cfg(not(feature = "superlu_dist"))]
        {
            let _ = pmat; // silence unused warning
            Err(KError::SolveError(
                "superlu_dist feature not enabled".into(),
            ))
        }
    }

    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        z.copy_from_slice(r);
        Ok(())
    }
}
