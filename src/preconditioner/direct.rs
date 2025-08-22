use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::legacy::LinearSolver;
use crate::solver::{LuSolver, QrSolver};
#[cfg(feature = "superlu_dist")]
use crate::solver::superlu_dist::SuperLuDistSolver;
use faer::Mat;
use crate::matrix::sparse::CsrMatrix;

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

/// Minimal QR-based preconditioner wrapper.
pub struct QrPc;

impl QrPc {
    pub fn new() -> Self { Self }
}

impl Preconditioner for QrPc {
    fn setup(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let _m: &Mat<f64> = pmat
            .as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("QR PC requires faer::Mat<f64>".into()))?;
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
            KError::InvalidInput("QR direct_solve requires faer::Mat<f64>".into())
        })?;

        let mut qr = QrSolver::new();
        let b_vec = b.to_vec();
        let mut x_vec = vec![0.0; x.len()];
        qr.solve(
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

/// Minimal SuperLU_DIST-based preconditioner wrapper operating on CSR matrices.
#[cfg_attr(not(feature = "superlu_dist"), allow(dead_code))]
pub struct SuperLuDistPc;

impl SuperLuDistPc {
    pub fn new() -> Self { Self }
}

impl Preconditioner for SuperLuDistPc {
    fn setup(&mut self, pmat: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let _a: &CsrMatrix<f64> = pmat
            .as_any()
            .downcast_ref::<CsrMatrix<f64>>()
            .ok_or_else(|| {
                KError::InvalidInput("SuperLU_DIST PC requires CSR matrix".into())
            })?;
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
        #[cfg(not(feature = "superlu_dist"))]
        {
            return Err(KError::SolveError(
                "superlu_dist feature not enabled".into(),
            ));
        }

        #[cfg(feature = "superlu_dist")]
        {
            let a: &CsrMatrix<f64> = op.as_any().downcast_ref::<CsrMatrix<f64>>().ok_or_else(|| {
                KError::InvalidInput("SuperLU_DIST expects CSR".into())
            })?;

            let mut slu = SuperLuDistSolver::new();
            let b_vec = b.to_vec();
            let mut x_vec = vec![0.0; x.len()];
            slu.solve(
                a,
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
}


