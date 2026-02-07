use crate::algebra::scalar::{KrystScalar, R, S};
use crate::config::options::{KspOptions, PcOptions};
use crate::context::ksp_context::KspContext;
use crate::context::pc_context::{PcFactory, PcType};
use crate::error::KError;
use crate::matrix::backend::materialize_ref;
use crate::matrix::op::LinOp;
use crate::preconditioner::{PcSide, Preconditioner};
use std::sync::Mutex;

pub struct KspAsPc {
    inner: Box<dyn Preconditioner>,
    inner_ksp_type: Option<String>,
    ksp_options: Option<KspOptions>,
    maxits: usize,
    rtol: R,
    inner_pc_opts: PcOptions,
    nested_ksp: Mutex<Option<KspContext>>,
}

impl KspAsPc {
    pub fn new(
        inner_pc_type: Option<String>,
        inner_ksp_type: Option<String>,
        maxits: usize,
        rtol: R,
        ksp_options: Option<KspOptions>,
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
            inner_ksp_type,
            ksp_options,
            maxits: maxits.max(1),
            rtol,
            inner_pc_opts: opts,
            nested_ksp: Mutex::new(None),
        })
    }
}

impl Preconditioner for KspAsPc {
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        if let Some(ref ksp) = self.inner_ksp_type {
            crate::context::ksp_context::SolverType::from_str(ksp)?;
        }
        if self.ksp_options.is_some() || self.inner_ksp_type.is_some() {
            if let Some(ksp) = self.try_build_ksp_context(a)? {
                *self.nested_ksp.lock().expect("ksp-pc nested lock") = Some(ksp);
                return Ok(());
            }
        }
        self.inner.setup(a)?;
        *self.nested_ksp.lock().expect("ksp-pc nested lock") = None;
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if let Some(ksp) = self.nested_ksp.lock().expect("ksp-pc nested lock").as_mut() {
            if x.len() != y.len() {
                return Err(KError::InvalidInput(
                    "ksp-pc input/output length mismatch".into(),
                ));
            }
            y.fill(S::zero());
            let _ = ksp.solve(x, y)?;
            return Ok(());
        }
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

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if let Some(ksp) = self.nested_ksp.lock().expect("ksp-pc nested lock").as_mut() {
            if x.len() != y.len() {
                return Err(KError::InvalidInput(
                    "ksp-pc input/output length mismatch".into(),
                ));
            }
            y.fill(S::zero());
            let _ = ksp.solve(x, y)?;
            return Ok(());
        }
        self.apply(side, x, y)
    }
}

use std::str::FromStr;

impl KspAsPc {
    fn try_build_ksp_context(&self, a: &dyn LinOp<S = S>) -> Result<Option<KspContext>, KError> {
        let want = a.format();
        if want.is_any() {
            return Ok(None);
        }
        let drop_tol = self.inner_pc_opts.drop_tol.unwrap_or_default();
        let amat = materialize_ref(a, want, drop_tol).ok();
        let Some(amat) = amat else {
            return Ok(None);
        };

        let mut ksp_opts = self.ksp_options.clone().unwrap_or_default();
        if ksp_opts.ksp_type.is_none() {
            ksp_opts.ksp_type = self.inner_ksp_type.clone();
        }
        if ksp_opts.maxits.is_none() {
            ksp_opts.maxits = Some(self.maxits);
        }
        if ksp_opts.rtol.is_none() {
            ksp_opts.rtol = Some(self.rtol);
        }

        let mut ksp = KspContext::new();
        ksp.set_from_all_options(&ksp_opts, &self.inner_pc_opts)?;
        ksp.set_operators(amat.clone(), None);
        ksp.setup()?;
        Ok(Some(ksp))
    }
}
