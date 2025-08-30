use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;

pub struct QrPc;

impl QrPc {
    pub fn new() -> Self {
        Self
    }
}

impl Preconditioner for QrPc {
    fn setup(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        pmat.as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("QR PC requires faer::Mat<f64>".into()))?;
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        let _ = (r, z);
        Err(KError::Unsupported(
            "QrPc is PREONLY-only; use SolverType::Preonly or call direct_solve",
        ))
    }

    fn direct_solve(
        &mut self,
        pmat: &dyn LinOp<S = f64>,
        b: &[f64],
        x: &mut [f64],
    ) -> Result<(), KError> {
        let a = pmat
            .as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("QR PC requires faer::Mat<f64>".into()))?;
        crate::solver::dense_qr::solve(a, b, x)
    }

    fn required_format(&self) -> crate::matrix::format::FormatHint {
        crate::matrix::format::FormatHint::Dense
    }
}
