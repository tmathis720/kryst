use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::parallel::UniverseComm;

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

    fn direct_solve(
        &mut self,
        pmat: &dyn LinOp<S = f64>,
        b: &[f64],
        x: &mut [f64],
        comm: &UniverseComm,
    ) -> Result<(), KError> {
        #[cfg(feature = "superlu_dist")]
        {
            let a = pmat
                .as_any()
                .downcast_ref::<CsrMatrix<f64>>()
                .ok_or_else(|| KError::InvalidInput("SuperLU_DIST PC requires CSR matrix".into()))?;
            crate::solver::superlu_dist::solve(a, b, x, comm)
        }
        #[cfg(not(feature = "superlu_dist"))]
        {
            let _ = (pmat, b, x, comm);
            Err(KError::SolveError("superlu_dist feature not enabled".into()))
        }
    }
}
