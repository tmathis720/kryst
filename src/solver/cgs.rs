//! Conjugate Gradient Squared (CGS).
//!
//! Expert method: CGS often exhibits volatile, non-monotone residuals and can
//! amplify round-off. Prefer (F)GMRES/BiCGStab for robustness. Use CGS when
//! you specifically want short recurrences and can handle breakdowns.
//!
//! - Preconditioning: currently not applied (API accepts `pc` but it is ignored).
//! - Monitors report the true residual `||r||_2`.
//! - Parallel safety: all inner products/norms use `UniverseComm`.

use crate::algebra::blas::dot_conj;
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::common::{recompute_true_residual_norm_s, take_or_resize};
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
pub struct CgsSolver {
    pub(crate) conv: Convergence,
}

/// Relative threshold for CGS breakdown detection.
/// Trigger when |rho| or |sigma| is smaller than BRK_REL * scale.
const BRK_REL: R = 1e-12;
/// Absolute floor to guard subnormals.
const BRK_ABS: R = 1e-300;

#[cfg(feature = "complex")]
use num_complex::Complex64;

fn reduce_real<C: Comm>(comm: &C, value: R) -> R {
    if comm.size() == 1 {
        value
    } else {
        comm.allreduce_sum(value)
    }
}

fn reduce_scalar<C: Comm>(comm: &C, value: S) -> S {
    if comm.size() == 1 {
        value
    } else {
        #[cfg(feature = "complex")]
        {
            let local: Complex64 = value;
            let (re, im) = comm.allreduce_sum2(local.re, local.im);
            Complex64::new(re, im)
        }
        #[cfg(not(feature = "complex"))]
        {
            S::from_real(comm.allreduce_sum(value.real()))
        }
    }
}

fn dot<C: Comm>(x: &[S], y: &[S], comm: &C) -> S {
    let local = dot_conj(x, y);
    reduce_scalar(comm, local)
}

fn norm2<C: Comm>(x: &[S], comm: &C) -> R {
    let local = dot_conj(x, x).real();
    reduce_real(comm, local).sqrt()
}

struct CgsWorkspace<'a> {
    r: &'a mut [S],
    v: &'a mut [S],
    u: &'a mut [S],
    p: &'a mut [S],
    q: &'a mut [S],
    upq: &'a mut [S],
    w: &'a mut [S],
    scratch: &'a mut BridgeScratch,
}

impl<'a> CgsWorkspace<'a> {
    fn acquire(n: usize, work: &'a mut Workspace) -> Self {
        take_or_resize(&mut work.tmp1, n);
        take_or_resize(&mut work.tmp2, n);
        while work.q_s.len() < 5 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..5] {
            take_or_resize(buf, n);
        }
        let (q0, rest) = work.q_s.split_at_mut(1);
        let (q1, rest) = rest.split_at_mut(1);
        let (q2, rest) = rest.split_at_mut(1);
        let (q3, q4) = rest.split_at_mut(1);
        Self {
            r: &mut work.tmp1[..n],
            v: &mut work.tmp2[..n],
            u: &mut q0[0][..n],
            p: &mut q1[0][..n],
            q: &mut q2[0][..n],
            upq: &mut q3[0][..n],
            w: &mut q4[0][..n],
            scratch: &mut work.bridge,
        }
    }
}

impl CgsSolver {
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

    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_comm<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        let _ = pc;
        let _ = pc_side;
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

        let work = work.ok_or_else(|| {
            KError::InvalidInput("CGS requires a Workspace; use KSP or Workspace::new(n)".into())
        })?;

        if b.is_empty() {
            return Ok(SolveStats::new(0, 0.0, ConvergedReason::ConvergedAtol));
        }

        let buffers = CgsWorkspace::acquire(n, work);
        let CgsWorkspace {
            r,
            v,
            u,
            p,
            q,
            upq,
            w,
            scratch,
        } = buffers;
        let monitors = monitors.unwrap_or(&[]);

        let mut r_tld = vec![S::zero(); n];

        if x.iter().any(|&xi| xi != S::zero()) {
            a.matvec_s(x, &mut *v, &mut *scratch);
            for i in 0..n {
                r[i] = b[i] - v[i];
            }
        } else {
            r.copy_from_slice(b);
        }
        r_tld.copy_from_slice(r);

        let rtld_norm = norm2(&r_tld, comm);

        let mut rnorm = norm2(r, comm);
        let res0_reported = rnorm;

        for m in monitors {
            m(0, rnorm);
        }

        let (reason0, s0) = self.conv.check(rnorm, res0_reported, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            return Ok(SolveStats::new(0, rnorm, s0.reason));
        }

        let mut rho = dot(&r_tld, r, comm);
        let mut r_norm = norm2(r, comm);
        let mut rho_abs = rho.abs();
        let mut rho_thr = BRK_ABS.max(BRK_REL * rtld_norm * r_norm);
        if rho_abs <= rho_thr {
            return Err(KError::IndefiniteMatrix);
        }

        u.copy_from_slice(r);
        p.copy_from_slice(u);

        let mut iters = 0usize;
        for k in 1..=self.conv.max_iters {
            iters = k;

            a.matvec_s(p, &mut *v, &mut *scratch);

            let sigma = dot(&r_tld, v, comm);
            let sigma_abs = sigma.abs();
            let v_norm = norm2(v, comm);
            let sigma_thr = BRK_ABS.max(BRK_REL * rtld_norm * v_norm);
            if sigma_abs <= sigma_thr {
                return Err(KError::IndefiniteMatrix);
            }
            let alpha = rho / sigma;

            for i in 0..n {
                q[i] = u[i] - alpha * v[i];
            }

            for i in 0..n {
                let sum = u[i] + q[i];
                x[i] += alpha * sum;
                upq[i] = sum;
            }

            a.matvec_s(upq, &mut *w, &mut *scratch);
            for i in 0..n {
                r[i] -= alpha * w[i];
            }

            rnorm = norm2(r, comm);
            for m in monitors {
                m(k, rnorm);
            }

            let (reason, s) = self.conv.check(rnorm, res0_reported, k);
            if !matches!(reason, ConvergedReason::Continued) {
                return Ok(SolveStats::new(k, rnorm, s.reason));
            }

            let rho_old = rho;
            rho = dot(&r_tld, r, comm);
            r_norm = norm2(r, comm);
            rho_abs = rho.abs();
            rho_thr = BRK_ABS.max(BRK_REL * rtld_norm * r_norm);
            if rho_abs <= rho_thr {
                return Err(KError::IndefiniteMatrix);
            }
            let beta = rho / rho_old;

            for i in 0..n {
                u[i] = r[i] + beta * q[i];
            }
            for i in 0..n {
                p[i] = u[i] + beta * (q[i] + beta * p[i]);
            }
        }

        let true_res = recompute_true_residual_norm_s(a, b, x, comm, &mut *w, &mut *scratch);
        Ok(SolveStats::new(
            iters,
            true_res,
            ConvergedReason::DivergedMaxIts,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        self.solve_with_comm(a, pc, b, x, pc_side, comm, monitors, work)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_f64<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn PreconditionerF64>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError>
    where
        A: LinOpF64 + LinOp<S = f64> + Send + Sync + ?Sized,
    {
        let op = as_s_op(a);
        let pc_wrapper = pc.map(as_s_pc);
        let pc_ref = pc_wrapper
            .as_ref()
            .map(|w| w as &dyn KPreconditioner<Scalar = S>);

        #[cfg(not(feature = "complex"))]
        {
            let b_s: &[S] = unsafe { &*(b as *const [f64] as *const [S]) };
            let x_s: &mut [S] = unsafe { &mut *(x as *mut [f64] as *mut [S]) };
            self.solve(&op, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
        }
        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result = self.solve(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work);
            if result.is_ok() {
                for (dst, src) in x.iter_mut().zip(x_s.iter()) {
                    *dst = src.real();
                }
            }
            result
        }
    }
}

impl LinearSolver for CgsSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q_s.len() < 5 {
            work.q_s.resize(5, Vec::new());
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
        self.solve_f64(a, pc.as_deref(), b, x, pc_side, comm, monitors, work)
    }
}
