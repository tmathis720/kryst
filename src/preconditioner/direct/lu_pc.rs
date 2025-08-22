use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::parallel::UniverseComm;
use faer::Mat;

pub struct LuPc;

impl LuPc {
    pub fn new() -> Self {
        Self
    }
}

impl Preconditioner for LuPc {
    fn setup(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        pmat.as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("LU PC requires faer::Mat<f64>".into()))?;
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        // minimal: identity; later, cache factors and solve M z = r
        z.copy_from_slice(r);
        Ok(())
    }

    fn direct_solve(
        &mut self,
        pmat: &dyn LinOp<S = f64>,
        b: &[f64],
        x: &mut [f64],
        _comm: &UniverseComm,
    ) -> Result<(), KError> {
        let a = pmat
            .as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("LU PC requires faer::Mat<f64>".into()))?;
        crate::solver::dense_lu::solve(a, b, x)
    }
}
