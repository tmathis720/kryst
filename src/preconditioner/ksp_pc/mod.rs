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
    inner_pc_type: Option<String>,
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
            inner_pc_type,
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
        self.teardown_nested_ksp();
        let mut ksp = KspContext::new();
        if self.configure_nested_ksp(&mut ksp, a)? {
            let mut guard = self.nested_ksp.lock().expect("ksp-pc nested lock");
            *guard = Some(ksp);
            return Ok(());
        }
        self.inner.setup(a)?;
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(
                "ksp-pc input/output length mismatch".into(),
            ));
        }
        if let Some(ksp) = self.nested_ksp.lock().expect("ksp-pc nested lock").as_mut() {
            y.fill(S::zero());
            let _ = ksp.solve(x, y)?;
            return Ok(());
        }
        self.inner.apply(side, x, y)
    }

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply(side, x, y)
    }
}

use std::str::FromStr;

impl KspAsPc {
    fn teardown_nested_ksp(&self) {
        let mut guard = self.nested_ksp.lock().expect("ksp-pc nested lock");
        *guard = None;
    }

    fn effective_ksp_options(&self) -> KspOptions {
        let mut ksp_opts = self.ksp_options.clone().unwrap_or_default();
        if ksp_opts.ksp_type.is_none() {
            ksp_opts.ksp_type = self
                .inner_ksp_type
                .clone()
                .or_else(|| Some("richardson".to_string()));
        }
        if ksp_opts.maxits.is_none() {
            ksp_opts.maxits = Some(self.maxits);
        }
        if ksp_opts.rtol.is_none() {
            ksp_opts.rtol = Some(self.rtol);
        }
        ksp_opts
    }

    fn effective_pc_options(&self) -> PcOptions {
        let mut pc_opts = self.inner_pc_opts.clone();
        if pc_opts.pc_type.is_none() {
            pc_opts.pc_type = self
                .inner_pc_type
                .clone()
                .or_else(|| Some("jacobi".to_string()));
        }
        pc_opts
    }

    fn configure_nested_ksp(
        &self,
        ksp: &mut KspContext,
        a: &dyn LinOp<S = S>,
    ) -> Result<bool, KError> {
        let want = a.format();
        if want.is_any() {
            return Ok(false);
        }
        let drop_tol = self.inner_pc_opts.drop_tol.unwrap_or_default();
        let Ok(amat) = materialize_ref(a, want, drop_tol) else {
            return Ok(false);
        };

        let ksp_opts = self.effective_ksp_options();
        let pc_opts = self.effective_pc_options();
        ksp.set_from_all_options(&ksp_opts, &pc_opts)?;
        ksp.try_set_operators_with_comm(amat, None, a.comm())?;
        ksp.setup()?;
        Ok(true)
    }
}
