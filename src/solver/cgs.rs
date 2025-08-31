use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, SolveStats};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
pub struct CgsSolver {
    rtol: f64,
    atol: f64,
    dtol: f64,
    maxits: usize,
}

impl CgsSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            rtol,
            atol: 1e-12,
            dtol: 1e3,
            maxits,
        }
    }

    #[inline]
    fn dot(x: &[f64], y: &[f64], _comm: &UniverseComm) -> f64 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
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
        work: Option<&'a mut Workspace>,
    ) -> (
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
    ) {
        if let Some(wk) = work {
            Self::take_or_resize(&mut wk.tmp1, n); // r
            Self::take_or_resize(&mut wk.tmp2, n); // v
            while wk.q.len() < 5 {
                wk.q.push(Vec::new());
            }
            for k in 0..5 {
                Self::take_or_resize(&mut wk.q[k], n);
            }
            let r = &mut wk.tmp1[..];
            let v = &mut wk.tmp2[..];
            let (u, p, q, upq, w) = {
                let (q0, rest) = wk.q.split_at_mut(1);
                let (q1, rest) = rest.split_at_mut(1);
                let (q2, rest) = rest.split_at_mut(1);
                let (q3, q4) = rest.split_at_mut(1);
                (
                    &mut q0[0][..],
                    &mut q1[0][..],
                    &mut q2[0][..],
                    &mut q3[0][..],
                    &mut q4[0][..],
                )
            };
            (r, v, u, p, q, upq, w)
        } else {
            // Fallback for unit tests when no Workspace supplied
            let mk = |n| -> &'static mut [f64] { Box::leak(vec![0.0; n].into_boxed_slice()) };
            (mk(n), mk(n), mk(n), mk(n), mk(n), mk(n), mk(n))
        }
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

        // initial values
        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let mut rnorm = Self::nrm2(r, comm);

        if let Some(ms) = monitors {
            for m in ms {
                m(0, rnorm);
            }
        }
        // quick exit
        let thr = self.atol.max(self.rtol * bnorm);
        if rnorm <= thr {
            return Ok(SolveStats {
                iterations: 0,
                final_residual: rnorm,
                reason: if rnorm <= self.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                },
            });
        }

        // CGS parameters
        let mut rho = Self::dot(&r_tld, r, comm); // (r~, r)
        if rho.abs() <= f64::EPSILON {
            return Err(KError::IndefiniteMatrix); // classic breakdown
        }

        // First iter: u = r, p = u
        u.copy_from_slice(r);
        p.copy_from_slice(u);

        let mut rho_old: f64;
        let mut iters = 0usize;
        for k in 1..=self.maxits {
            iters = k;

            // v = A p
            a.matvec(p, v);

            let sigma = Self::dot(&r_tld, v, comm); // (r~, v)
            if sigma.abs() <= f64::EPSILON {
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

            // convergence / divergence tests
            if rnorm <= thr {
                return Ok(SolveStats {
                    iterations: k,
                    final_residual: rnorm,
                    reason: if rnorm <= self.atol {
                        ConvergedReason::ConvergedAtol
                    } else {
                        ConvergedReason::ConvergedRtol
                    },
                });
            }
            if !rnorm.is_finite() || rnorm >= self.dtol {
                return Ok(SolveStats {
                    iterations: k,
                    final_residual: rnorm,
                    reason: ConvergedReason::DivergedDtol,
                });
            }

            // rho, beta updates
            rho_old = rho;
            rho = Self::dot(&r_tld, r, comm);
            if rho.abs() <= f64::EPSILON {
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

        Ok(SolveStats {
            iterations: iters,
            final_residual: rnorm,
            reason: ConvergedReason::DivergedMaxIts,
        })
    }
}
