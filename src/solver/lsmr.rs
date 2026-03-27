//! LSMR solver for least-squares and rank-deficient systems.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::UniverseComm;
use crate::preconditioner::{self, PcSide, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::MonitorCallback;
use crate::solver::common::{ReductCtx, recompute_true_residual_norm_s, take_or_resize};
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use crate::utils::monitor::{ResidualSnapshot, log_residuals};
use std::any::Any;

pub struct LsmrSolver {
    pub conv: Convergence,
}

impl LsmrSolver {
    pub fn new(rtol: f64, max_iters: usize) -> Self {
        Self {
            conv: Convergence {
                rtol,
                atol: 1e-12,
                dtol: 1e3,
                max_iters,
            },
        }
    }
}

#[inline]
fn sym_ortho(a: R, b: R) -> (R, R, R) {
    if b == R::default() {
        (R::from(1.0), R::default(), a)
    } else if a == R::default() {
        (R::default(), R::from(1.0), b)
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s = R::from(1.0) / (R::from(1.0) + tau * tau).sqrt();
        let c = s * tau;
        (c, s, b / s)
    } else {
        let tau = b / a;
        let c = R::from(1.0) / (R::from(1.0) + tau * tau).sqrt();
        let s = c * tau;
        (c, s, a / c)
    }
}

struct LsmrWorkspace<'a> {
    u: &'a mut [S],
    v: &'a mut [S],
    h: &'a mut [S],
    hbar: &'a mut [S],
    at_u: &'a mut [S],
    av: &'a mut [S],
    tmp_true: &'a mut [S],
    scratch: &'a mut BridgeScratch,
}

impl<'a> LsmrWorkspace<'a> {
    fn acquire(work: &'a mut Workspace, m: usize, n: usize) -> Self {
        take_or_resize(&mut work.tmp1, m);
        take_or_resize(&mut work.tmp2, n);
        if work.bridge_tmp.len() != m {
            work.bridge_tmp.resize(m, S::zero());
        }
        while work.q_s.len() < 3 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..3] {
            take_or_resize(buf, n);
        }
        if work.z_s.is_empty() {
            work.z_s.push(Vec::new());
        }
        take_or_resize(&mut work.z_s[0], m);

        let (h_slice, rest) = work.q_s.split_at_mut(1);
        let (hbar_slice, rest) = rest.split_at_mut(1);
        let (at_u_slice, _) = rest.split_at_mut(1);
        Self {
            u: &mut work.tmp1[..m],
            v: &mut work.tmp2[..n],
            h: &mut h_slice[0][..n],
            hbar: &mut hbar_slice[0][..n],
            at_u: &mut at_u_slice[0][..n],
            av: &mut work.z_s[0][..m],
            tmp_true: &mut work.bridge_tmp[..m],
            scratch: &mut work.bridge,
        }
    }
}

impl LsmrSolver {
    #[allow(clippy::too_many_arguments)]
    fn solve_internal<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        let (m, n) = a.dims();
        if b.len() != m {
            return Err(KError::InvalidInput("LSMR: b has wrong length".into()));
        }
        if x.len() != n {
            return Err(KError::InvalidInput("LSMR: x has wrong length".into()));
        }
        if !a.supports_t_matvec_s() {
            return Err(KError::InvalidInput(
                "LSMR requires t_matvec; provide an operator that implements A^T·x".into(),
            ));
        }
        if pc.is_some() {
            return Err(KError::Unsupported(
                "LSMR preconditioning is not yet supported",
            ));
        }
        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "LSMR currently supports Left preconditioning only".into(),
            ));
        }

        let mut owned;
        let work = if let Some(work) = work {
            work
        } else {
            owned = Workspace::new(m.max(n));
            &mut owned
        };
        let red = ReductCtx::new(comm, Some(&*work));
        let mut buffers = LsmrWorkspace::acquire(work, m, n);
        let LsmrWorkspace {
            u,
            v,
            h,
            hbar,
            at_u,
            av,
            tmp_true,
            scratch,
        } = &mut buffers;

        let monitors = monitors.unwrap_or(&[]);

        if x.iter().any(|&xi| xi.abs() > R::default()) {
            a.matvec_s(x, av, scratch);
            for (ui, (&bi, &avi)) in u.iter_mut().zip(b.iter().zip(av.iter())) {
                *ui = bi - avi;
            }
        } else {
            u.copy_from_slice(b);
        }

        let mut beta = red.norm2(&u[..m]);
        if beta == R::default() {
            return Ok(SolveStats::new(
                0,
                R::default(),
                ConvergedReason::ConvergedAtol,
            ));
        }
        let beta_s = S::from_real(beta);
        for ui in &mut u[..m] {
            *ui /= beta_s;
        }

        a.t_matvec_s(u, v, scratch);
        let mut alpha = red.norm2(&v[..n]);
        if alpha == R::default() {
            return Ok(SolveStats::new(0, beta, ConvergedReason::ConvergedAtol));
        }
        let alpha_s = S::from_real(alpha);
        for vi in &mut v[..n] {
            *vi /= alpha_s;
        }

        let mut zetabar = alpha * beta;
        let mut alphabar = alpha;
        let mut rho = R::from(1.0);
        let mut rhobar = R::from(1.0);
        let mut cbar = R::from(1.0);
        let mut sbar = R::default();

        h.copy_from_slice(v);
        hbar.fill(S::zero());

        let mut betadd = beta;
        let mut betad = R::default();
        let mut rhodold = R::from(1.0);
        let mut tautildeold = R::default();
        let mut thetatilde = R::default();
        let mut zeta = R::default();
        let mut d = R::default();

        let mut normr = beta;
        let res0 = beta;
        for m in monitors {
            let _ = m(0, normr, 0);
        }
        log_residuals(
            0,
            "LSMR",
            ResidualSnapshot {
                true_residual: beta,
                preconditioned_residual: beta,
                recurrence_residual: Some(normr),
            },
        );

        let mut final_reason = ConvergedReason::Continued;
        let mut iters = 0usize;

        for k in 1..=self.conv.max_iters {
            iters = k;

            a.matvec_s(v, av, scratch);
            let alpha_s = S::from_real(alpha);
            for i in 0..m {
                u[i] = av[i] - alpha_s * u[i];
            }
            beta = red.norm2(&u[..m]);
            if beta == R::default() {
                final_reason = ConvergedReason::ConvergedAtol;
                break;
            }
            let beta_s = S::from_real(beta);
            for ui in &mut u[..m] {
                *ui /= beta_s;
            }
            a.t_matvec_s(u, at_u, scratch);
            for i in 0..n {
                v[i] = at_u[i] - beta_s * v[i];
            }
            alpha = red.norm2(&v[..n]);
            if alpha == R::default() {
                final_reason = ConvergedReason::ConvergedAtol;
                break;
            }
            let alpha_s = S::from_real(alpha);
            for vi in &mut v[..n] {
                *vi /= alpha_s;
            }

            let (chat, shat, alphahat) = sym_ortho(alphabar, R::default());
            let rhoold = rho;
            let (c, s, new_rho) = sym_ortho(alphahat, beta);
            rho = new_rho;
            let thetanew = s * alpha;
            alphabar = c * alpha;

            let rhobarold = rhobar;
            let zetaold = zeta;
            let thetabar = sbar * rho;
            let (new_cbar, new_sbar, new_rhobar) = sym_ortho(cbar * rho, thetanew);
            cbar = new_cbar;
            sbar = new_sbar;
            rhobar = new_rhobar;
            zeta = cbar * zetabar;
            zetabar *= -sbar;

            let h_scale = -(thetabar * rho) / (rhoold * rhobarold);
            let h_scale_s = S::from_real(h_scale);
            for i in 0..n {
                hbar[i] = hbar[i] * h_scale_s + h[i];
            }
            let x_scale = zeta / (rho * rhobar);
            let x_scale_s = S::from_real(x_scale);
            for i in 0..n {
                x[i] += x_scale_s * hbar[i];
            }
            let h_update_scale = -(thetanew / rho);
            let h_update_s = S::from_real(h_update_scale);
            for i in 0..n {
                h[i] = h_update_s * h[i] + v[i];
            }

            let betaacute = chat * betadd;
            let betacheck = -shat * betadd;
            let betahat = c * betaacute;
            betadd = -s * betaacute;

            let thetatildeold = thetatilde;
            let (ctildeold, stildeold, rhotildeold) = sym_ortho(rhodold, thetabar);
            thetatilde = stildeold * rhobar;
            rhodold = ctildeold * rhobar;
            betad = -stildeold * betad + ctildeold * betahat;

            tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
            let taud = (zeta - thetatilde * tautildeold) / rhodold;
            d += betacheck * betacheck;
            normr = (d + (betad - taud) * (betad - taud) + betadd * betadd).sqrt();

            for m in monitors {
                let _ = m(k, normr, 0);
            }

            a.matvec_s(x, av, scratch);
            for i in 0..m {
                tmp_true[i] = b[i] - av[i];
            }
            let true_res = red.norm2(&tmp_true[..m]);
            log_residuals(
                k,
                "LSMR",
                ResidualSnapshot {
                    true_residual: true_res,
                    preconditioned_residual: true_res,
                    recurrence_residual: Some(normr),
                },
            );

            let (reason, _) = self.conv.check(normr, res0, k);
            if matches!(
                reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ) {
                let true_res_check =
                    recompute_true_residual_norm_s(a, b, x, comm, red.engine(), tmp_true, scratch);
                let (reason_true, _) = self.conv.check(true_res_check, res0, k);
                if matches!(
                    reason_true,
                    ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
                ) {
                    final_reason = reason_true;
                    break;
                }
            }
        }

        let true_res =
            recompute_true_residual_norm_s(a, b, x, comm, red.engine(), tmp_true, scratch);
        let bnorm = res0.max(R::from(1e-32));
        let rel = true_res / bnorm;
        let mut reason = if true_res <= self.conv.atol {
            ConvergedReason::ConvergedAtol
        } else if rel <= self.conv.rtol * R::from(10.0) {
            ConvergedReason::ConvergedRtol
        } else if iters >= self.conv.max_iters {
            ConvergedReason::DivergedMaxIts
        } else {
            ConvergedReason::Continued
        };
        if matches!(
            final_reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ) {
            reason = final_reason;
        }

        Ok(SolveStats::new(iters, true_res, reason))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_k<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        self.solve_internal(a, pc, b, x, pc_side, comm, monitors, work)
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
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
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
            self.solve_internal(&op, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
        }

        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result =
                self.solve_internal(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work);
            if result.is_ok() {
                for (dst, src) in x.iter_mut().zip(x_s.iter()) {
                    *dst = src.real();
                }
            }
            result
        }
    }
}

impl LinearSolver for LsmrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        if w.q_s.len() < 3 {
            w.q_s.resize(3, Vec::new());
        }
        if w.z_s.is_empty() {
            w.z_s.resize(1, Vec::new());
        }
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn preconditioner::Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let pc = pc.map(|m| m as &dyn PreconditionerF64);
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}
