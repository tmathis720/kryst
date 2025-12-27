use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
#[cfg(feature = "complex")]
use crate::preconditioner::bridge::apply_pc_s;

pub struct QrPc;

impl QrPc {
    pub fn new() -> Self {
        Self
    }
}

impl Preconditioner for QrPc {
    fn setup(&mut self, pmat: &dyn LinOp<S = S>) -> Result<(), KError> {
        pmat.as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("QR PC requires faer::Mat<f64>".into()))?;
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &[S], z: &mut [S]) -> Result<(), KError> {
        let _ = (r, z);
        Err(KError::Unsupported(
            "QrPc is PREONLY-only; use SolverType::Preonly or call direct_solve",
        ))
    }

    fn direct_solve(
        &mut self,
        pmat: &dyn LinOp<S = S>,
        b: &[S],
        x: &mut [S],
    ) -> Result<(), KError> {
        let a = pmat
            .as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("QR PC requires faer::Mat<f64>".into()))?;
        #[cfg(not(feature = "complex"))]
        {
            crate::solver::dense_qr::solve(a, b, x)
        }
        #[cfg(feature = "complex")]
        {
            let n = b.len();
            let mut b_re = vec![0.0; n];
            let mut b_im = vec![0.0; n];
            for (i, bi) in b.iter().enumerate() {
                b_re[i] = bi.real();
                b_im[i] = bi.imag();
            }
            let mut x_re = vec![0.0; x.len()];
            let mut x_im = vec![0.0; x.len()];
            crate::solver::dense_qr::solve(a, &b_re, &mut x_re)?;
            crate::solver::dense_qr::solve(a, &b_im, &mut x_im)?;
            for (xi, (&re, &im)) in x.iter_mut().zip(x_re.iter().zip(x_im.iter())) {
                *xi = S::from_parts(re, im);
            }
            Ok(())
        }
    }

    fn required_format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Dense
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for QrPc {
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
