use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, SolveStats};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;

pub struct CgnrSolver {
    rtol: f64,
    atol: f64,
    dtol: f64,
    maxits: usize,
}

impl CgnrSolver {
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

    fn take_or_resize(buf: &mut Vec<f64>, n: usize) {
        if buf.len() != n {
            buf.resize(n, 0.0);
        }
    }

    fn acquire<'a>(
        n: usize,
        work: Option<&'a mut Workspace>,
    ) -> (
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
    ) {
        if let Some(wk) = work {
            Self::take_or_resize(&mut wk.tmp1, n);
            Self::take_or_resize(&mut wk.tmp2, n);
            while wk.q.len() < 3 {
                wk.q.push(Vec::new());
            }
            for k in 0..3 {
                Self::take_or_resize(&mut wk.q[k], n);
            }
            let (p_slice, rest) = wk.q.split_at_mut(1);
            let (ap_slice, rest) = rest.split_at_mut(1);
            let (atap_slice, _) = rest.split_at_mut(1);
            let r = &mut wk.tmp1[..];
            let z = &mut wk.tmp2[..];
            let p = &mut p_slice[0][..];
            let ap = &mut ap_slice[0][..];
            let atap = &mut atap_slice[0][..];
            (r, z, p, ap, atap)
        } else {
            let mk = |n| -> &'static mut [f64] { Box::leak(vec![0.0; n].into_boxed_slice()) };
            (mk(n), mk(n), mk(n), mk(n), mk(n))
        }
    }
}

impl LinearSolver for CgnrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q.len() < 3 {
            work.q.resize(3, Vec::new());
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
        let _guard = StageGuard::new("CGNR");

        let (m, ncols) = a.dims();
        if b.len() != m {
            return Err(KError::InvalidInput("CGNR: b has wrong length".into()));
        }
        if x.len() != ncols {
            return Err(KError::InvalidInput("CGNR: x has wrong length".into()));
        }

        if !a.supports_transpose() {
            return Err(KError::InvalidInput(
                "CGNR requires t_matvec; provide an operator that implements A^T·x".into(),
            ));
        }

        let (r, z, p, ap, _atap) = Self::acquire(ncols.max(m), work);
        let (r, z, p, ap, _atap) = (
            &mut r[..m],
            &mut z[..ncols],
            &mut p[..ncols],
            &mut ap[..m],
            &mut _atap[..ncols],
        );

        if x.iter().any(|&xi| xi != 0.0) {
            a.matvec(x, ap);
            for i in 0..m {
                r[i] = b[i] - ap[i];
            }
        } else {
            r.copy_from_slice(b);
        }

        a.t_matvec(r, z);

        let mut zhat_buf: Vec<f64> = if pc.is_some() {
            vec![0.0; ncols]
        } else {
            Vec::new()
        };
        if let Some(pc) = pc {
            pc.apply(pc_side, z, &mut zhat_buf)?;
        }
        let zhat_slice: &[f64] = if pc.is_some() { &zhat_buf[..] } else { &z[..] };

        p.copy_from_slice(zhat_slice);

        let mut rz = Self::dot(&z[..], zhat_slice, comm);
        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let mut rnow = Self::nrm2(r, comm);

        if let Some(ms) = monitors {
            for m in ms {
                m(0, rnow);
            }
        }

        let thr = self.atol.max(self.rtol * bnorm);
        if rnow <= thr {
            return Ok(SolveStats {
                iterations: 0,
                final_residual: rnow,
                reason: if rnow <= self.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                },
            });
        }

        let mut iters = 0usize;
        for k in 1..=self.maxits {
            iters = k;

            a.matvec(p, ap);

            let denom = Self::dot(ap, ap, comm);
            if denom <= 0.0 || !denom.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }
            let alpha = rz / denom;

            for i in 0..ncols {
                x[i] += alpha * p[i];
            }
            for i in 0..m {
                r[i] -= alpha * ap[i];
            }

            a.t_matvec(r, z);
            if let Some(pc) = pc {
                pc.apply(pc_side, z, &mut zhat_buf)?;
            }
            let zhat_slice: &[f64] = if pc.is_some() { &zhat_buf[..] } else { &z[..] };

            let rz_new = Self::dot(&z[..], zhat_slice, comm);
            rnow = Self::nrm2(r, comm);

            if let Some(ms) = monitors {
                for m in ms {
                    m(k, rnow);
                }
            }

            if rnow <= thr {
                return Ok(SolveStats {
                    iterations: k,
                    final_residual: rnow,
                    reason: if rnow <= self.atol {
                        ConvergedReason::ConvergedAtol
                    } else {
                        ConvergedReason::ConvergedRtol
                    },
                });
            }
            if rnow >= self.dtol || !rnow.is_finite() {
                return Ok(SolveStats {
                    iterations: k,
                    final_residual: rnow,
                    reason: ConvergedReason::DivergedDtol,
                });
            }

            let beta = rz_new / rz;
            for i in 0..ncols {
                p[i] = zhat_slice[i] + beta * p[i];
            }
            rz = rz_new;
        }

        Ok(SolveStats {
            iterations: iters,
            final_residual: rnow,
            reason: ConvergedReason::DivergedMaxIts,
        })
    }
}
