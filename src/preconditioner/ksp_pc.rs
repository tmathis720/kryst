use crate::algebra::scalar::{KrystScalar, R, S};
use crate::config::options::PcOptions;
use crate::context::pc_context::{PcFactory, PcType};
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};

pub struct KspAsPc {
    inner: Box<dyn Preconditioner>,
    maxits: usize,
    rtol: R,
}

impl KspAsPc {
    pub fn new(
        inner_pc_type: Option<String>,
        maxits: usize,
        rtol: R,
        opts: PcOptions,
    ) -> Result<Self, KError> {
        let pct = inner_pc_type
            .as_deref()
            .map(PcType::from_str)
            .transpose()?
            .unwrap_or(PcType::Jacobi);
        let inner = PcFactory::create_preconditioner(pct, Some(&opts))?;
        Ok(Self {
            inner,
            maxits: maxits.max(1),
            rtol,
        })
    }
}

impl Preconditioner for KspAsPc {
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.inner.setup(a)
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(
                "ksp-pc input/output length mismatch".into(),
            ));
        }
        y.copy_from_slice(x);
        let mut tmp = vec![S::zero(); y.len()];
        for _ in 0..self.maxits {
            self.inner.apply(side, y, &mut tmp)?;
            y.copy_from_slice(&tmp);
            let norm_sq = y.iter().map(|v| v.abs() * v.abs()).sum::<R>();
            if norm_sq.sqrt() <= self.rtol {
                break;
            }
        }
        Ok(())
    }
}

use std::str::FromStr;
