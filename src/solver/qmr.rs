//! # QMR side semantics
//!
//! Accepts [`PcSide::Left`] or [`PcSide::Right`]; monitors report the true `||r||`.

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
use std::any::Any;

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

    #[allow(clippy::too_many_arguments)]
    fn solve_internal<A>(
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
        if matches!(pc_side, PcSide::Symmetric) {
            return Err(KError::InvalidInput(
                "QMR: symmetric preconditioning not supported".to_string(),
            ));
        }
        let _ = pc; // Preconditioning support is not wired yet.

        let (m, ncols) = a.dims();
        if m != ncols {
            return Err(KError::InvalidInput(
                "QMR requires a square operator".to_string(),
            ));
        }
        if b.len() != m || x.len() != ncols {
            return Err(KError::InvalidInput(
                "QMR: vector size mismatch".to_string(),
            ));
        }
        if !a.supports_t_matvec_s() {
            return Err(KError::InvalidInput(
                "QMR requires t_matvec; provide an operator that implements A^T·x".to_string(),
            ));
        }

        let monitors = monitors.unwrap_or(&[]);

        let mut owned_workspace: Workspace;
        let work: &mut Workspace = match work {
            Some(ws) => ws,
            None => {
                owned_workspace = Workspace::new(ncols);
                &mut owned_workspace
            }
        };
        let buffers = QmrWorkspace::acquire(work, ncols);
        let QmrWorkspace {
            r,
            t,
            r_tld,
            p,
            p_tld,
            v,
            v_tld,
            s,
            tmp_true,
            scratch,
        } = buffers;

        if x.iter().any(|&xi| xi != S::zero()) {
            a.matvec_s(x, r, scratch);
            for (ri, &bi) in r.iter_mut().zip(b.iter()) {
                let ai = *ri;
                *ri = bi - ai;
            }
        } else {
            r.copy_from_slice(b);
        }
        r_tld.copy_from_slice(r);

        let mut res = norm2(r, comm);
        let bnorm = norm2(b, comm).max(1e-32);

        for m in monitors {
            m(0, res);
        }

        let (reason0, mut stats0) = self.conv.check(res, bnorm, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let true_res = recompute_true_residual_norm_s(a, b, x, comm, tmp_true, scratch);
            stats0.final_residual = true_res;
            return Ok(stats0);
        }

        let eps = 1e-300;
        let mut rho = dot(r_tld, r, comm);
        if rho.abs() <= eps {
            return Err(KError::IndefiniteMatrix);
        }

        for k in 0..self.conv.max_iters {
            if k == 0 {
                p.copy_from_slice(r);
                p_tld.copy_from_slice(r_tld);
            } else {
                let rho_new = dot(r_tld, r, comm);
                if rho_new.abs() <= eps {
                    return Err(KError::IndefiniteMatrix);
                }
                let beta = rho_new / rho;
                for i in 0..ncols {
                    let ri = r[i];
                    let old_p = p[i];
                    p[i] = ri + beta * old_p;

                    let ri_tld = r_tld[i];
                    let old_pt = p_tld[i];
                    p_tld[i] = ri_tld + beta * old_pt;
                }
                rho = rho_new;
            }

            a.matvec_s(p, v, scratch);
            a.t_matvec_s(p_tld, v_tld, scratch);

            let sigma = dot(p_tld, v, comm);
            if sigma.abs() <= eps {
                return Err(KError::IndefiniteMatrix);
            }
            let alpha = rho / sigma;

            for i in 0..ncols {
                s[i] = r[i] - alpha * v[i];
            }
            a.matvec_s(s, t, scratch);

            let tt = dot(t, t, comm).real();
            if tt <= eps || !tt.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }
            let ts = dot(t, s, comm);
            let omega = ts / S::from_real(tt);

            for i in 0..ncols {
                x[i] += alpha * p[i] + omega * s[i];
            }
            for i in 0..ncols {
                let si = s[i];
                let ti = t[i];
                r[i] = si - omega * ti;
                r_tld[i] = si - omega.conj() * ti;
            }

            res = norm2(r, comm);
            for m in monitors {
                m(k + 1, res);
            }

            let (reason, mut stats) = self.conv.check(res, bnorm, k + 1);
            if !matches!(reason, ConvergedReason::Continued) {
                let true_res = recompute_true_residual_norm_s(a, b, x, comm, tmp_true, scratch);
                stats.final_residual = true_res;
                return Ok(stats);
            }
        }

        let true_res = recompute_true_residual_norm_s(a, b, x, comm, tmp_true, scratch);
        Ok(SolveStats::new(
            self.conv.max_iters,
            true_res,
            ConvergedReason::DivergedMaxIts,
        ))
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
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
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

    #[allow(clippy::too_many_arguments)]
    pub fn solve<A>(
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
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

impl LinearSolver for QmrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q_s.len() < 6 {
            work.q_s.resize(6, Vec::new());
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
        let pc = pc.map(|m| m as &dyn PreconditionerF64);
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

fn dot<C: Comm>(x: &[S], y: &[S], comm: &C) -> S {
    comm.allreduce_sum_scalar(crate::algebra::blas::dot_conj(x, y))
}

fn norm2<C: Comm>(x: &[S], comm: &C) -> R {
    let global = comm.allreduce_sum_scalar(crate::algebra::blas::dot_conj(x, x));
    global.real().sqrt()
}

struct QmrWorkspace<'a> {
    r: &'a mut [S],
    t: &'a mut [S],
    r_tld: &'a mut [S],
    p: &'a mut [S],
    p_tld: &'a mut [S],
    v: &'a mut [S],
    v_tld: &'a mut [S],
    s: &'a mut [S],
    tmp_true: &'a mut [S],
    scratch: &'a mut BridgeScratch,
}

impl<'a> QmrWorkspace<'a> {
    fn acquire(work: &'a mut Workspace, n: usize) -> Self {
        take_or_resize(&mut work.tmp1, n);
        take_or_resize(&mut work.tmp2, n);
        if work.bridge_tmp.len() != n {
            work.bridge_tmp.resize(n, S::zero());
        }
        while work.q_s.len() < 6 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..6] {
            take_or_resize(buf, n);
        }

        let (r_tld_slice, rest) = work.q_s.split_at_mut(1);
        let (p_slice, rest) = rest.split_at_mut(1);
        let (p_tld_slice, rest) = rest.split_at_mut(1);
        let (v_slice, rest) = rest.split_at_mut(1);
        let (v_tld_slice, rest) = rest.split_at_mut(1);
        let (s_slice, _) = rest.split_at_mut(1);

        Self {
            r: &mut work.tmp1[..n],
            t: &mut work.tmp2[..n],
            r_tld: &mut r_tld_slice[0][..n],
            p: &mut p_slice[0][..n],
            p_tld: &mut p_tld_slice[0][..n],
            v: &mut v_slice[0][..n],
            v_tld: &mut v_tld_slice[0][..n],
            s: &mut s_slice[0][..n],
            tmp_true: &mut work.bridge_tmp[..n],
            scratch: &mut work.bridge,
        }
    }
}
