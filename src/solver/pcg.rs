use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, SolveStats};

#[derive(Debug, Clone, Copy)]
pub enum CgNormType {
    Preconditioned,
    Unpreconditioned,
    Natural,
    None,
}

pub struct PcgSolver {
    rtol: f64,
    atol: f64,
    dtol: f64,
    maxits: usize,
    norm_type: CgNormType,
}

impl PcgSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            rtol,
            atol: 1e-12,
            dtol: 1e3,
            maxits,
            norm_type: CgNormType::Unpreconditioned,
        }
    }

    pub fn with_norm(mut self, norm_type: CgNormType) -> Self {
        self.norm_type = norm_type;
        self
    }

    #[inline]
    fn dot(u: &[f64], v: &[f64], comm: &UniverseComm) -> f64 {
        comm.dot(u, v)
    }

    #[inline]
    fn nrm2(u: &[f64], comm: &UniverseComm) -> f64 {
        Self::dot(u, u, comm).sqrt()
    }

    fn take_or_resize(buf: &mut Vec<f64>, n: usize) {
        if buf.len() != n {
            buf.resize(n, 0.0);
        }
    }
}

impl LinearSolver for PcgSolver {
    type Error = KError;

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q.len() < 2 {
            work.q.resize(2, Vec::new());
        }
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let n = b.len();
        if x.len() != n {
            return Err(KError::InvalidInput("dimension mismatch: x,b".into()));
        }

        // Acquire buffers either from workspace or allocate temporaries
        let mut r_store = Vec::new();
        let mut z_store = Vec::new();
        let mut p_store = Vec::new();
        let mut w_store = Vec::new();

        let mut r: &mut [f64] = &mut [];
        let mut z: &mut [f64] = &mut [];
        let mut p: &mut [f64] = &mut [];
        let mut w: &mut [f64] = &mut [];

        if let Some(wk) = work {
            Self::take_or_resize(&mut wk.tmp1, n);
            Self::take_or_resize(&mut wk.tmp2, n);
            while wk.q.len() < 2 {
                wk.q.push(vec![0.0; n]);
            }
            for v in &mut wk.q[0..2] {
                Self::take_or_resize(v, n);
            }
            r = wk.tmp1.as_mut_slice();
            z = wk.tmp2.as_mut_slice();
            let (pbuf, wbuf) = wk.q.split_at_mut(1);
            p = pbuf[0].as_mut_slice();
            w = wbuf[0].as_mut_slice();
        } else {
            r_store.resize(n, 0.0);
            z_store.resize(n, 0.0);
            p_store.resize(n, 0.0);
            w_store.resize(n, 0.0);
            r = r_store.as_mut_slice();
            z = z_store.as_mut_slice();
            p = p_store.as_mut_slice();
            w = w_store.as_mut_slice();
        }

        // r = b - A x
        let zero_guess = x.iter().all(|&xi| xi == 0.0);
        if zero_guess {
            r.copy_from_slice(b);
        } else {
            let mut ax = vec![0.0; n];
            a.matvec(x, &mut ax);
            for i in 0..n {
                r[i] = b[i] - ax[i];
            }
        }

        // z = M^{-1} r
        if let Some(pc) = pc {
            pc.apply(PcSide::Left, r, z)?;
        } else {
            z.copy_from_slice(r);
        }

        let mut res = match self.norm_type {
            CgNormType::Preconditioned => Self::dot(r, z, comm).sqrt(),
            CgNormType::Unpreconditioned => Self::nrm2(r, comm),
            CgNormType::Natural => Self::dot(r, z, comm),
            CgNormType::None => 0.0,
        };
        let bnorm = if matches!(self.norm_type, CgNormType::None) {
            1.0
        } else {
            Self::nrm2(b, comm).max(1e-32)
        };

        if let Some(ms) = monitors {
            for m in ms {
                m(0, res);
            }
        }

        p.copy_from_slice(z);
        let mut rz = Self::dot(r, z, comm);

        if res <= self.atol.max(self.rtol * bnorm) {
            return Ok(SolveStats {
                iterations: 0,
                final_residual: res,
                reason: ConvergedReason::ConvergedAtol,
            });
        }

        let mut iters = 0usize;
        for k in 0..self.maxits {
            iters = k + 1;

            a.matvec(p, w);
            let pw = Self::dot(p, w, comm);
            if pw <= 0.0 || !pw.is_finite() {
                return Ok(SolveStats {
                    iterations: k,
                    final_residual: res,
                    reason: ConvergedReason::DivergedDtol,
                });
            }
            let alpha = rz / pw;

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * w[i];
            }

            if let Some(pc) = pc {
                pc.apply(PcSide::Left, r, z)?;
            } else {
                z.copy_from_slice(r);
            }

            let rz_new = Self::dot(r, z, comm);
            if rz_new < 0.0 || !rz_new.is_finite() {
                return Ok(SolveStats {
                    iterations: k + 1,
                    final_residual: res,
                    reason: ConvergedReason::DivergedDtol,
                });
            }

            res = match self.norm_type {
                CgNormType::Preconditioned => rz_new.sqrt(),
                CgNormType::Unpreconditioned => Self::nrm2(r, comm),
                CgNormType::Natural => rz_new,
                CgNormType::None => res,
            };

            if let Some(ms) = monitors {
                for m in ms {
                    m(k + 1, res);
                }
            }

            if res <= self.atol.max(self.rtol * bnorm) || res >= self.dtol {
                let reason = if res <= self.atol.max(self.rtol * bnorm) {
                    if res <= self.atol {
                        ConvergedReason::ConvergedAtol
                    } else {
                        ConvergedReason::ConvergedRtol
                    }
                } else {
                    ConvergedReason::DivergedDtol
                };
                return Ok(SolveStats {
                    iterations: k + 1,
                    final_residual: res,
                    reason,
                });
            }

            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }

        Ok(SolveStats {
            iterations: iters,
            final_residual: res,
            reason: ConvergedReason::DivergedMaxIts,
        })
    }
}
