use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
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
}
