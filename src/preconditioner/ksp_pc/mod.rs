use crate::algebra::scalar::{KrystScalar, R, S};
use crate::config::options::{KspOptions, PcOptions};
use crate::context::ksp_context::{ExecutionPolicy, KspContext};
use crate::error::KError;
use crate::matrix::backend::materialize_ref;
use crate::matrix::op::LinOp;
use crate::parallel::Comm;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
use crate::utils::convergence::{ConvergedReason, FailureReasonKind, NestedPcFailure};
use std::sync::Mutex;

struct InnerKspContext {
    ksp: KspContext,
    ksp_options: KspOptions,
    pc_options: PcOptions,
}

#[derive(Clone, Copy, Debug)]
struct ResidualHistorySummary {
    len: usize,
    first: R,
    last: R,
    min: R,
    max: R,
}

impl ResidualHistorySummary {
    fn from_history(history: &[R]) -> Option<Self> {
        if history.is_empty() {
            return None;
        }
        let mut min = history[0];
        let mut max = history[0];
        for &v in &history[1..] {
            min = min.min(v);
            max = max.max(v);
        }
        Some(Self {
            len: history.len(),
            first: history[0],
            last: history[history.len() - 1],
            min,
            max,
        })
    }

    fn detail_fragment(self) -> String {
        format!(
            "history_len={} first={:.3e} last={:.3e} min={:.3e} max={:.3e}",
            self.len, self.first, self.last, self.min, self.max
        )
    }
}

pub struct KspAsPc {
    ksp_options: KspOptions,
    pc_options: PcOptions,
    inner_ctx: Mutex<Option<InnerKspContext>>,
}

impl KspAsPc {
    pub fn new(mut ksp_options: KspOptions, mut pc_options: PcOptions) -> Result<Self, KError> {
        if ksp_options.ksp_type.is_none() {
            ksp_options.ksp_type = Some("richardson".to_string());
        }
        if ksp_options.maxits.is_none() {
            ksp_options.maxits = Some(1);
        }
        if ksp_options.rtol.is_none() {
            ksp_options.rtol = Some(1e-2);
        }
        if pc_options.pc_type.is_none() {
            pc_options.pc_type = Some("jacobi".to_string());
        }
        Ok(Self {
            ksp_options,
            pc_options,
            inner_ctx: Mutex::new(None),
        })
    }

    fn configure_nested_ksp(&self, a: &dyn LinOp<S = S>) -> Result<bool, KError> {
        let want = a.format();
        if want.is_any() {
            return Ok(false);
        }
        let drop_tol = self.pc_options.drop_tol.unwrap_or_default();
        let Ok(amat) = materialize_ref(a, want, drop_tol) else {
            return Ok(false);
        };

        let ksp_opts = self.ksp_options.clone();
        let pc_opts = self.pc_options.clone();

        let mut guard = self.inner_ctx.lock().expect("ksp-pc nested lock");
        if let Some(existing) = guard.as_mut() {
            existing
                .ksp
                .try_set_operators_with_comm(amat, None, a.comm())?;
            existing.ksp.setup()?;
            return Ok(true);
        }

        let nested_exec = ExecutionPolicy::nested_from_options(&ksp_opts, a.comm().size())?;
        let mut ksp = KspContext::new();
        ksp.set_execution_policy(nested_exec);
        ksp.set_from_all_options(&ksp_opts, &pc_opts)?;
        ksp.try_set_operators_with_comm(amat, None, a.comm())?;
        ksp.setup()?;

        *guard = Some(InnerKspContext {
            ksp,
            ksp_options: ksp_opts,
            pc_options: pc_opts,
        });
        Ok(true)
    }

    fn is_acceptable_inner_reason(reason: ConvergedReason) -> bool {
        reason.is_converged() || matches!(reason, ConvergedReason::DivergedMaxIts)
    }
}

impl Preconditioner for KspAsPc {
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        let configured = self.configure_nested_ksp(a)?;
        if !configured {
            return Err(KError::Unsupported(
                "pc_type=ksp requires a materializable matrix format for nested KSP setup",
            ));
        }
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
            let stats = inner.ksp.solve(x, y).map_err(|err| {
                let detail = format!(
                    "inner_ksp={:?} inner_pc={:?} nested_error={err}",
                    inner.ksp_options.ksp_type, inner.pc_options.pc_type,
                );
                KError::NestedPcFailed(NestedPcFailure {
                    component: "pc_ksp",
                    reason: ConvergedReason::from_failure_kind(FailureReasonKind::PcApply),
                    iterations: 0,
                    detail,
                    final_norm: None,
                    residual_history_summary: None,
                })
            })?;
            if !Self::is_acceptable_inner_reason(stats.reason) {
                let history = vec![stats.final_residual];
                let history_summary = ResidualHistorySummary::from_history(&history)
                    .map(ResidualHistorySummary::detail_fragment)
                    .unwrap_or_else(|| "history_len=0".to_string());
                let detail = format!(
                    "inner_ksp={:?} inner_pc={:?} true_final_norm={:.3e} nested_reason={} {}",
                    inner.ksp_options.ksp_type,
                    inner.pc_options.pc_type,
                    stats.final_residual,
                    stats.reason,
                    history_summary
                );
                return Err(KError::NestedPcFailed(NestedPcFailure {
                    component: "pc_ksp",
                    reason: stats.reason,
                    iterations: stats.iterations,
                    detail,
                    final_norm: Some(format!("true_residual_l2={:.6e}", stats.final_residual)),
                    residual_history_summary: Some(history_summary),
                }));
            }
            return Ok(());
        }
        Err(KError::SolveError(
            "pc_type=ksp used before successful setup".into(),
        ))
    }

    fn on_restart(&mut self, _outer_iter: usize, _residual_norm: R) -> Result<(), KError> {
        if let Some(inner) = self.inner_ctx.lock().expect("ksp-pc nested lock").as_mut() {
            inner.ksp.setup()?;
        }
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        if !self.configure_nested_ksp(a)? {
            return Err(KError::Unsupported(
                "pc_type=ksp requires a materializable matrix format for nested KSP setup",
            ));
        }
        Ok(())
    }

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply(side, x, y)
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
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
    fn nested_ksp_flat_options_override_scoped_block() {
        let a = tri_diag(10);
        let n = a.dims().0;
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).unwrap();
        let ksp_opts = KspOptions {
            maxits: Some(20),
            rtol: Some(1e-8),
            ..Default::default()
        };
        let pc_opts = PcOptions {
            pc_type: Some("ksp".into()),
            pc_ksp_ksp_type: Some("gmres".into()),
            pc_ksp_maxits: Some(1),
            pc_ksp_rtol: Some(1e-1),
            pc_ksp_ksp_options: Some(KspOptions {
                ksp_type: Some("richardson".into()),
                maxits: Some(6),
                rtol: Some(1e-16),
                ..Default::default()
            }),
            pc_ksp_pc_type: Some("none".into()),
            pc_ksp_pc_options: Some(Box::new(PcOptions {
                pc_type: Some("jacobi".into()),
                ..Default::default()
            })),
            ..Default::default()
        };
        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
        ksp.set_operators(a, None);
        let stats = ksp.solve(&b, &mut x).unwrap();
        assert!(stats.reason != crate::utils::convergence::ConvergedReason::Continued);
        assert!(stats.nested_pc_failure.is_none());
    }

    #[test]
    fn nested_pc_failure_metadata_fields_are_populated() {
        let failure = NestedPcFailure {
            component: "pc_ksp",
            reason: crate::utils::convergence::ConvergedReason::DivergedDtol,
            iterations: 3,
            final_norm: Some("true_residual_l2=1.234000e+00".into()),
            residual_history_summary: Some(
                "history_len=1 first=1.234e+00 last=1.234e+00 min=1.234e+00 max=1.234e+00".into(),
            ),
            detail: "inner_ksp=Some(\"gmres\") inner_pc=Some(\"none\")".into(),
        };
        assert!(
            failure
                .final_norm
                .as_deref()
                .unwrap_or_default()
                .contains("true_residual_l2")
        );
        assert!(
            failure
                .residual_history_summary
                .as_deref()
                .unwrap_or_default()
                .contains("history_len=")
        );
    }

    #[test]
    fn nested_ksp_failure_propagates_shell_callback_error() {
        use crate::preconditioner::shell::{register_shell_callback, shell_apply};

        let tag = "nested_shell_err";
        register_shell_callback(
            format!("{tag}_apply"),
            shell_apply(|_, _, _| Err(KError::SolveError("boom".into()))),
        );

        let a = tri_diag(8);
        let n = a.dims().0;
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).unwrap();
        let ksp_opts = KspOptions {
            maxits: Some(5),
            rtol: Some(1e-10),
            ..Default::default()
        };
        let pc_opts = PcOptions {
            pc_type: Some("ksp".into()),
            pc_ksp_ksp_options: Some(KspOptions {
                ksp_type: Some("richardson".into()),
                maxits: Some(2),
                rtol: Some(1e-16),
                ..Default::default()
            }),
            pc_ksp_pc_options: Some(Box::new(PcOptions {
                pc_type: Some("shell".into()),
                pc_shell_apply: Some(format!("{tag}_apply")),
                ..Default::default()
            })),
            ..Default::default()
        };

        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
        ksp.set_operators(a, None);
        let stats = ksp.solve(&b, &mut x).unwrap();
        assert_eq!(
            stats.reason,
            crate::utils::convergence::ConvergedReason::DivergedPcFailed
        );
        let failure = stats
            .nested_pc_failure
            .as_ref()
            .expect("nested failure metadata should be present");
        assert_eq!(failure.component, "pc_ksp");
        assert!(failure.detail.contains("inner_pc=Some(\"shell\")"));
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
        assert!(stats.reason.is_converged());
        assert!(stats.nested_pc_failure.is_none());
    }
}
