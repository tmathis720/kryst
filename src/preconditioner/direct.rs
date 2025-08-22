use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::legacy::LinearSolver;
use crate::solver::LuSolver;
use faer::Mat;

/// Minimal LU-based preconditioner that supports [`Preconditioner::direct_solve`].
pub struct LuPc {
    ready: bool,
}

impl LuPc {
    pub fn new() -> Self {
        Self { ready: false }
    }
}

impl Preconditioner for LuPc {
    fn setup(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let _m: &Mat<f64> = pmat
            .as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("LU PC requires faer::Mat<f64>".into()))?;
        self.ready = true;
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        z.copy_from_slice(r);
        Ok(())
    }

    fn direct_solve(
        &mut self,
        op: &dyn LinOp<S = f64>,
        b: &[f64],
        x: &mut [f64],
    ) -> Result<bool, KError> {
        let m: &Mat<f64> = op.as_any().downcast_ref::<Mat<f64>>().ok_or_else(|| {
            KError::InvalidInput("LU direct_solve requires faer::Mat<f64>".into())
        })?;

        let mut lu = LuSolver::new();
        let b_vec = b.to_vec();
        let mut x_vec = vec![0.0; x.len()];
        lu.solve(
            m,
            None,
            &b_vec,
            &mut x_vec,
            &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
            None,
            None,
        )?;
        x.copy_from_slice(&x_vec);
        Ok(true)
    }
}


