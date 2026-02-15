use crate::algebra::scalar::{KrystScalar, R, S};
use crate::config::options::{KspOptions, PcOptions};
use crate::context::ksp_context::KspContext;
use crate::context::pc_context::{PcFactory, PcType};
use crate::error::KError;
use crate::matrix::backend::materialize_ref;
use crate::matrix::op::LinOp;
use crate::parallel::Comm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::utils::convergence::NestedPcFailure;
use std::str::FromStr;
use std::sync::Mutex;

struct InnerKspContext {
    ksp: KspContext,
    ksp_options: KspOptions,
    pc_options: PcOptions,
}

pub struct KspAsPc {
    inner: Box<dyn Preconditioner>,
    inner_ksp_type: Option<String>,
    inner_pc_type: Option<String>,
    ksp_options: Option<KspOptions>,
    maxits: usize,
    rtol: R,
    inner_pc_opts: PcOptions,
    inner_ctx: Mutex<Option<InnerKspContext>>,
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
            inner_ctx: Mutex::new(None),
        })
    }

    fn teardown_nested_ksp(&self) {
        let mut guard = self.inner_ctx.lock().expect("ksp-pc nested lock");
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

    fn check_nested_compatibility(
        &self,
        ksp_opts: &KspOptions,
        a: &dyn LinOp<S = S>,
    ) -> Result<(), KError> {
        let comm = a.comm();
        if matches!(ksp_opts.threads_mode.as_deref(), Some("global")) {
            return Err(KError::InvalidInput(
                "pc_type=ksp does not allow inner ksp_threads_mode=global; use context/serial"
                    .into(),
            ));
        }

        if comm.size() > 1
            && ksp_opts.threads.unwrap_or(1) > 1
            && matches!(ksp_opts.threads_mode.as_deref(), Some("context") | None)
        {
            return Err(KError::InvalidInput(
                "nested pc_type=ksp with MPI requires explicit inner serial threading; set pc_ksp_ksp_options.threads_mode=serial"
                    .into(),
            ));
        }
        Ok(())
    }

    fn configure_nested_ksp(&self, a: &dyn LinOp<S = S>) -> Result<bool, KError> {
        let want = a.format();
        if want.is_any() {
            return Ok(false);
        }
        let drop_tol = self.inner_pc_opts.drop_tol.unwrap_or_default();
        let Ok(amat) = materialize_ref(a, want, drop_tol) else {
            return Ok(false);
        };

        let ksp_opts = self.effective_ksp_options();
        self.check_nested_compatibility(&ksp_opts, a)?;
        let pc_opts = self.effective_pc_options();

        let mut ksp = KspContext::new();
        ksp.set_from_all_options(&ksp_opts, &pc_opts)?;
        ksp.try_set_operators_with_comm(amat, None, a.comm())?;
        ksp.setup()?;

        let mut guard = self.inner_ctx.lock().expect("ksp-pc nested lock");
        *guard = Some(InnerKspContext {
            ksp,
            ksp_options: ksp_opts,
            pc_options: pc_opts,
        });
        Ok(true)
    }
}

impl Preconditioner for KspAsPc {
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        if let Some(ref ksp) = self.inner_ksp_type {
            crate::context::ksp_context::SolverType::from_str(ksp)?;
        }
        self.teardown_nested_ksp();
        if self.configure_nested_ksp(a)? {
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

        if let Some(inner) = self.inner_ctx.lock().expect("ksp-pc nested lock").as_mut() {
            y.fill(S::zero());
            let stats = inner.ksp.solve(x, y)?;
            if stats.reason.is_diverged() {
                return Err(KError::NestedPcFailed(NestedPcFailure {
                    component: "pc_ksp",
                    reason: stats.reason,
                    iterations: stats.iterations,
                    detail: format!(
                        "inner_ksp={:?} inner_pc={:?}",
                        inner.ksp_options.ksp_type, inner.pc_options.pc_type
                    ),
                }));
            }
            return Ok(());
        }
        self.inner.apply(side, x, y)
    }

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply(side, x, y)
    }
}

#[cfg(all(test, feature = "backend-faer", not(feature = "complex")))]
mod tests {
    use super::*;
    use crate::config::options::KspOptions;
    use crate::context::ksp_context::{KspContext, SolverType};
    use crate::matrix::op::DenseOp;
    use faer::Mat;
    use std::sync::Arc;

    fn tri_diag(n: usize) -> Arc<DenseOp<f64>> {
        let m = Mat::<f64>::from_fn(n, n, |i, j| {
            if i == j {
                4.0
            } else if (i as isize - j as isize).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        });
        Arc::new(DenseOp::new(Arc::new(m)))
    }

    fn run_nested_case(outer: SolverType, inner: &str, side: &str) {
        let a = tri_diag(12);
        let n = a.dims().0;
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut ksp = KspContext::new();
        ksp.set_type(outer).unwrap();
        let ksp_opts = KspOptions {
            pc_side: Some(side.into()),
            maxits: Some(60),
            rtol: Some(1e-8),
            ..Default::default()
        };
        let pc_opts = PcOptions {
            pc_type: Some("ksp".into()),
            pc_ksp_ksp_options: Some(KspOptions {
                ksp_type: Some(inner.into()),
                maxits: Some(4),
                rtol: Some(1e-2),
                ..Default::default()
            }),
            pc_ksp_pc_options: Some(Box::new(PcOptions {
                pc_type: Some("jacobi".into()),
                ..Default::default()
            })),
            ..Default::default()
        };
        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
        ksp.set_operators(a, None);
        let stats = ksp.solve(&b, &mut x).unwrap();
        assert!(
            stats.reason.is_converged()
                || matches!(
                    stats.reason,
                    crate::utils::convergence::ConvergedReason::DivergedMaxIts
                )
        );
    }

    #[test]
    fn nested_ksp_pc_outer_gmres_multiple_inner_and_sides() {
        run_nested_case(SolverType::Gmres, "richardson", "left");
        run_nested_case(SolverType::Gmres, "gmres", "right");
    }

    #[test]
    fn nested_ksp_pc_outer_fgmres_multiple_inner_and_sides() {
        run_nested_case(SolverType::Fgmres, "richardson", "left");
        run_nested_case(SolverType::Fgmres, "gmres", "right");
    }

    #[test]
    fn nested_ksp_failure_surfaces_inner_reason() {
        let a = tri_diag(8);
        let n = a.dims().0;
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).unwrap();
        let ksp_opts = KspOptions {
            maxits: Some(20),
            rtol: Some(1e-10),
            ..Default::default()
        };
        let pc_opts = PcOptions {
            pc_type: Some("ksp".into()),
            pc_ksp_ksp_options: Some(KspOptions {
                ksp_type: Some("richardson".into()),
                maxits: Some(1),
                rtol: Some(1e-16),
                ..Default::default()
            }),
            pc_ksp_pc_options: Some(Box::new(PcOptions {
                pc_type: Some("none".into()),
                ..Default::default()
            })),
            ..Default::default()
        };
        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
        ksp.set_operators(a, None);
        let stats = ksp.solve(&b, &mut x).unwrap();
        assert!(stats.reason.is_diverged());
        let nested = stats
            .nested_pc_failure
            .expect("missing nested failure details");
        assert_eq!(nested.component, "pc_ksp");
        assert!(nested.reason.is_diverged());
    }
}
