use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
#[cfg(feature = "complex")]
use crate::preconditioner::bridge::apply_pc_s;

#[cfg(feature = "superlu_dist")]
use crate::matrix::sparse::CsrMatrix;

#[cfg_attr(docsrs, doc(cfg(feature = "superlu_dist")))]
pub struct SuperLuDistPc {
    #[allow(dead_code)]
    comm: Option<UniverseComm>,
}

impl Default for SuperLuDistPc {
    fn default() -> Self {
        Self::new()
    }
}

impl SuperLuDistPc {
    pub fn new() -> Self {
        Self { comm: None }
    }
}

impl Preconditioner for SuperLuDistPc {
    fn setup(&mut self, pmat: &dyn LinOp<S = S>) -> Result<(), KError> {
        #[cfg(feature = "superlu_dist")]
        {
            self.comm = Some(pmat.comm());
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

    fn apply(&self, _side: PcSide, r: &[S], z: &mut [S]) -> Result<(), KError> {
        let _ = (r, z);
        Err(KError::Unsupported(
            "SuperLuDistPc::apply is PREONLY-only; use SolverType::Preonly or call direct_solve",
        ))
    }

    fn direct_solve(
        &mut self,
        pmat: &dyn LinOp<S = S>,
        b: &[S],
        x: &mut [S],
    ) -> Result<(), KError> {
        #[cfg(feature = "superlu_dist")]
        {
            let a = pmat
                .as_any()
                .downcast_ref::<CsrMatrix<f64>>()
                .ok_or_else(|| {
                    KError::InvalidInput("SuperLU_DIST PC requires CSR matrix".into())
                })?;
            let comm = self.comm.clone().unwrap_or_else(|| pmat.comm());
            #[cfg(not(feature = "complex"))]
            {
                crate::solver::superlu_dist::solve(a, b, x, &comm)
            }
            #[cfg(feature = "complex")]
            {
                let mut b_real = vec![0.0; b.len()];
                let mut x_real = vec![0.0; x.len()];
                crate::algebra::scalar::copy_scalar_to_real_in(b, &mut b_real);
                crate::solver::superlu_dist::solve(a, &b_real, &mut x_real, &comm)?;
                crate::algebra::scalar::copy_real_to_scalar_in(&x_real, x);
                Ok(())
            }
        }
        #[cfg(not(feature = "superlu_dist"))]
        {
            let _ = (pmat, b, x);
            Err(KError::SolveError(
                "superlu_dist feature not enabled".into(),
            ))
        }
    }

    fn required_format(&self) -> crate::matrix::format::FormatHint {
        crate::matrix::format::FormatHint::Csr
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for SuperLuDistPc {
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
