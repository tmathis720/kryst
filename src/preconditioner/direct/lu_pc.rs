use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::bridge::apply_pc_s;
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
        let _ = (r, z);
        Err(KError::Unsupported(
            "LuPc::apply is PREONLY-only; use SolverType::Preonly or call direct_solve",
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
            .ok_or_else(|| KError::InvalidInput("LU PC requires faer::Mat<f64>".into()))?;
        crate::solver::dense_lu::solve(a, b, x)
    }

    fn required_format(&self) -> crate::matrix::format::FormatHint {
        crate::matrix::format::FormatHint::Dense
    }
}

impl KPreconditioner for LuPc {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        <Self as Preconditioner>::dims(self)
    }

    #[inline]
    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        apply_pc_s(self, side, x, y, scratch)
    }
}
