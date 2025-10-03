//! # QMR side semantics
//!
//! Accepts [`PcSide::Left`] or [`PcSide::Right`]; monitors report the true `||r||`.

#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};

pub struct QmrSolver {
    pub conv: Convergence,
}

impl QmrSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            conv: Convergence {
                rtol,
                atol: 1e-12,
                dtol: 1e3,
                max_iters: maxits,
            },
        }
    }

    #[inline]
    fn dot(x: &[f64], y: &[f64], comm: &UniverseComm) -> f64 {
        comm.dot(x, y)
    }

    #[inline]
    fn nrm2(x: &[f64], comm: &UniverseComm) -> f64 {
        Self::dot(x, x, comm).sqrt()
    }

    fn ensure_workspace(work: &mut Workspace, n: usize) {
        let need = 6; // r_tld, p, p_tld, v, v_tld, s
        if work.tmp1.len() != n {
            work.tmp1.resize(n, 0.0);
        }
        if work.tmp2.len() != n {
            work.tmp2.resize(n, 0.0);
        }
        while work.q.len() < need {
            work.q.push(Vec::new());
        }
        for q in &mut work.q[..need] {
            if q.len() != n {
                q.resize(n, 0.0);
            }
        }
    }
}

impl LinearSolver for QmrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        let n = work.tmp1.len();
        Self::ensure_workspace(work, n);
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        _pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        _comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let (m, ncols) = a.dims();
        if m != ncols {
            return Err(KError::InvalidInput("QMR requires square A".into()));
        }
        if b.len() != m || x.len() != ncols {
            return Err(KError::InvalidInput("QMR: size mismatch".into()));
        }
        if !a.supports_transpose() {
            return Err(KError::InvalidInput("QMR requires A^T".into()));
        }

        if matches!(pc_side, PcSide::Symmetric) {
            return Err(KError::InvalidInput(
                "QMR: symmetric preconditioning not supported".into(),
            ));
        }
        let _pc: Option<&dyn Preconditioner> = _pc.as_deref();
        let _ = _pc;
        let _ = pc_side;

        let mons = monitors.unwrap_or(&[]);
        let mut local_work;
        let w = match work {
            Some(w) => w,
            None => {
                local_work = Workspace::new(ncols);
                Self::ensure_workspace(&mut local_work, ncols);
                &mut local_work
            }
        };
        Self::ensure_workspace(w, ncols);

        let (r, t) = (&mut w.tmp1, &mut w.tmp2);
        let (r_tld, p, p_tld, v, v_tld, s) = {
            let (a, rest) = w.q.split_at_mut(1);
            let (b, rest) = rest.split_at_mut(1);
            let (c, rest) = rest.split_at_mut(1);
            let (d, rest) = rest.split_at_mut(1);
            let (e, rest) = rest.split_at_mut(1);
            let (f, _) = rest.split_at_mut(1);
            (
                &mut a[0], &mut b[0], &mut c[0], &mut d[0], &mut e[0], &mut f[0],
            )
        };

        // r = b - A x
        a.matvec(x, r);
        for i in 0..ncols {
            r[i] = b[i] - r[i];
        }
        r_tld.copy_from_slice(r);

        let mut res = Self::nrm2(r, _comm);
        let bnorm = Self::nrm2(b, _comm).max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);
        if !mons.is_empty() {
            for m in mons {
                m(0, res);
            }
        }
        if res <= thr {
            let reason = if res <= self.conv.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            return Ok(SolveStats::new(0, res, reason));
        }

        let eps = 1e-300_f64;
        let mut rho = Self::dot(r_tld, r, _comm);
        if rho.abs() <= eps {
            return Err(KError::IndefiniteMatrix);
        }

        for k in 0..self.conv.max_iters {
            if k == 0 {
                p.copy_from_slice(r);
                p_tld.copy_from_slice(r_tld);
            } else {
                let rho_new = Self::dot(r_tld, r, _comm);
                if rho_new.abs() <= eps {
                    return Err(KError::IndefiniteMatrix);
                }
                let beta = rho_new / rho;
                for i in 0..ncols {
                    p[i] = r[i] + beta * p[i];
                    p_tld[i] = r_tld[i] + beta * p_tld[i];
                }
                rho = rho_new;
            }

            a.matvec(p, v);
            a.t_matvec(p_tld, v_tld);

            let sigma = Self::dot(p_tld, v, _comm);
            if sigma.abs() <= eps {
                return Err(KError::IndefiniteMatrix);
            }
            let alpha = rho / sigma;

            for i in 0..ncols {
                s[i] = r[i] - alpha * v[i];
            }
            a.matvec(s, t);

            let tt = Self::dot(t, t, _comm);
            if tt <= eps || !tt.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }
            let ts = Self::dot(t, s, _comm);
            let omega = ts / tt;

            for i in 0..ncols {
                x[i] += alpha * p[i] + omega * s[i];
            }
            for i in 0..ncols {
                r[i] = s[i] - omega * t[i];
            }

            // true residual
            a.matvec(x, t);
            for i in 0..ncols {
                t[i] = b[i] - t[i];
            }
            res = Self::nrm2(t, _comm);

            if !mons.is_empty() {
                for m in mons {
                    m(k + 1, res);
                }
            }

            let (reason, mut st) = self.conv.check(res, bnorm, k + 1);
            if matches!(
                reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ) {
                st.final_residual = res;
                return Ok(st);
            }
        }

        Ok(SolveStats::new(
            self.conv.max_iters,
            res,
            ConvergedReason::DivergedMaxIts,
        ))
    }
}
