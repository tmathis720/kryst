//! Conjugate Gradient Squared (CGS).
//!
//! Expert method: CGS often exhibits volatile, non-monotone residuals and can
//! amplify round-off. Prefer (F)GMRES/BiCGStab for robustness. Use CGS when
//! you specifically want short recurrences and can handle breakdowns.
//!
//! - Preconditioning: currently not applied (API accepts `pc` but it is ignored).
//! - Monitors report the true residual `||r||_2`.
//! - Parallel safety: all inner products/norms use `UniverseComm`.

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::recompute_true_residual_norm;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
pub struct CgsSolver {
    pub(crate) conv: Convergence<f64>,
}

/// Relative threshold for CGS breakdown detection.
/// Trigger when |rho| or |sigma| is smaller than BRK_REL * scale.
const BRK_REL: f64 = 1e-12;
/// Absolute floor to guard subnormals.
const BRK_ABS: f64 = 1e-300;

impl CgsSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self { conv: Convergence { rtol, atol: 1e-12, dtol: 1e3, max_iters: maxits } }
    }

    #[inline]
    fn dot(x: &[f64], y: &[f64], comm: &UniverseComm) -> f64 {
        comm.dot(x, y)
    }
    #[inline]
    fn nrm2(x: &[f64], comm: &UniverseComm) -> f64 {
        Self::dot(x, x, comm).sqrt()
    }

    #[inline]
    fn take_or_resize(buf: &mut Vec<f64>, n: usize) {
        if buf.len() != n {
            buf.resize(n, 0.0);
        }
    }

    /// Acquire all CGS work vectors from `Workspace` (no steady-state allocs).
    /// We use:
    ///   tmp1 = r, tmp2 = v
    ///   q[0] = u, q[1] = p, q[2] = q, q[3] = upq, q[4] = w (A*(u+q))
    fn acquire<'a>(
        n: usize,
        work: &'a mut Workspace,
    ) -> (
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
    ) {
        Self::take_or_resize(&mut work.tmp1, n); // r
        Self::take_or_resize(&mut work.tmp2, n); // v
        while work.q.len() < 5 { work.q.push(Vec::new()); }
        for k in 0..5 { Self::take_or_resize(&mut work.q[k], n); }
        let r = &mut work.tmp1[..];
        let v = &mut work.tmp2[..];
        let (u, p, q, upq, w) = {
            let (q0, rest) = work.q.split_at_mut(1);
            let (q1, rest) = rest.split_at_mut(1);
            let (q2, rest) = rest.split_at_mut(1);
            let (q3, q4) = rest.split_at_mut(1);
            (&mut q0[0][..], &mut q1[0][..], &mut q2[0][..], &mut q3[0][..], &mut q4[0][..])
        };
        (r, v, u, p, q, upq, w)
    }
}

impl LinearSolver for CgsSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q.len() < 5 {
            work.q.resize(5, Vec::new());
        }
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let pc: Option<&dyn Preconditioner> = pc.as_deref();
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CGS");

        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput(
                "CGS requires a square operator".into(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput("CGS: vector length mismatch".into()));
        }

        // Require a Workspace to avoid heap leaks and repeated allocs.
        let work = work.ok_or_else(|| {
            KError::InvalidInput("CGS requires a Workspace; use KSP or Workspace::new(n)".into())
        })?;
        // Zero-length fast path
        if b.is_empty() {
            return Ok(SolveStats {
                iterations: 0,
                final_residual: 0.0,
                reason: ConvergedReason::ConvergedAtol,
            });
        }

        let (r, v, u, p, q, upq, w) = Self::acquire(n, work);
        let mut r_tld = vec![0.0; n]; // shadow residual (fixed)

        let _ = pc; // unused for now
        let _ = pc_side;

        // r = b - A x
        if x.iter().any(|&xi| xi != 0.0) {
            a.matvec(x, v);
            for i in 0..n {
                r[i] = b[i] - v[i];
            }
        } else {
            r.copy_from_slice(b);
        }
        r_tld.copy_from_slice(r);

        // Norm of shadow residual used to scale breakdown thresholds
        let rtld_norm = Self::nrm2(&r_tld, comm);

        // initial values
        let mut rnorm = Self::nrm2(r, comm);
        let res0_reported = rnorm;

        if let Some(ms) = monitors {
            for m in ms {
                m(0, rnorm);
            }
        }
        // quick exit via convergence policy against res0_reported baseline
        let (reason0, s0) = self.conv.check(rnorm, res0_reported, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            // ensure final_residual is true residual (already computed as rnorm)
            return Ok(SolveStats { iterations: 0, final_residual: rnorm, reason: s0.reason });
        }

        // CGS parameters
        let mut rho = Self::dot(&r_tld, r, comm); // (r~, r)
        // Robust breakdown check for rho
        let r_norm = Self::nrm2(r, comm);
        let rho_abs = rho.abs();
        let rho_thr = BRK_ABS.max(BRK_REL * rtld_norm * r_norm);
        if rho_abs <= rho_thr {
            return Err(KError::IndefiniteMatrix); // classic breakdown
        }

        // First iter: u = r, p = u
        u.copy_from_slice(r);
        p.copy_from_slice(u);

        let mut rho_old: f64;
        let mut iters = 0usize;
        for k in 1..=self.conv.max_iters {
            iters = k;

            // v = A p
            a.matvec(p, v);

            let sigma = Self::dot(&r_tld, v, comm); // (r~, v)
            // Robust breakdown check for sigma
            let v_norm = Self::nrm2(v, comm);
            let sigma_abs = sigma.abs();
            let sigma_thr = BRK_ABS.max(BRK_REL * rtld_norm * v_norm);
            if sigma_abs <= sigma_thr {
                return Err(KError::IndefiniteMatrix); // breakdown
            }
            let alpha = rho / sigma;

            // q = u - alpha v
            for i in 0..n {
                q[i] = u[i] - alpha * v[i];
            }

            // x += alpha * (u + q)
            for i in 0..n {
                x[i] += alpha * (u[i] + q[i]);
            }

            // r -= alpha * A (u + q)
            for i in 0..n {
                upq[i] = u[i] + q[i];
            }
            a.matvec(upq, w);
            for i in 0..n {
                r[i] -= alpha * w[i];
            }

            rnorm = Self::nrm2(r, comm);
            if let Some(ms) = monitors {
                for m in ms {
                    m(k, rnorm);
                }
            }

            // convergence / divergence tests against res0_reported
            let (reason, s) = self.conv.check(rnorm, res0_reported, k);
            if !matches!(reason, ConvergedReason::Continued) {
                return Ok(SolveStats { iterations: k, final_residual: rnorm, reason: s.reason });
            }

            // rho, beta updates
            rho_old = rho;
            rho = Self::dot(&r_tld, r, comm);
            // Robust breakdown check for rho update
            let r_norm = Self::nrm2(r, comm);
            let rho_abs = rho.abs();
            let rho_thr = BRK_ABS.max(BRK_REL * rtld_norm * r_norm);
            if rho_abs <= rho_thr {
                return Err(KError::IndefiniteMatrix); // breakdown
            }
            let beta = rho / rho_old;

            // u = r + beta q
            for i in 0..n {
                u[i] = r[i] + beta * q[i];
            }
            // p = u + beta (q + beta p)
            for i in 0..n {
                p[i] = u[i] + beta * (q[i] + beta * p[i]);
            }
        }

        // Max-its: recompute true residual and report divergence
        let mut tmp = vec![0.0; n];
        let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
        Ok(SolveStats { iterations: iters, final_residual: true_res, reason: ConvergedReason::DivergedMaxIts })
    }
}
