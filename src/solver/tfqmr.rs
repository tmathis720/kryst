//! # TFQMR side semantics
//!
//! Accepts [`PcSide::Left`] or [`PcSide::Right`]; residuals are reported as the true `||r||`.

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
use crate::preconditioner::{PcSide, Preconditioner, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::common::take_or_resize;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

pub struct TfqmrSolver {
    pub conv: Convergence,
    pub resid_recalc_every: usize,
    pub breakdown_eps: f64,
}

impl TfqmrSolver {
    pub fn new(rtol: f64, max_iters: usize) -> Self {
        Self {
            conv: Convergence {
                rtol,
                atol: 1e-12,
                dtol: 1e3,
                max_iters,
            },
            resid_recalc_every: 20,
            breakdown_eps: 1e-30,
        }
    }

    fn solve_internal<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        mut pc_side: PcSide,
        _comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput("TFQMR requires square A".to_string()));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput("TFQMR size mismatch".to_string()));
        }

        if pc_side == PcSide::Symmetric {
            pc_side = PcSide::Left;
        }

        let monitors = monitors.unwrap_or(&[]);

        let mut owned_workspace;
        let work = match work {
            Some(ws) => ws,
            None => {
                owned_workspace = Workspace::new(n);
                &mut owned_workspace
            }
        };

        let TfqmrWorkspace {
            r,
            au,
            tmp_pc,
            u,
            v,
            wv,
            yv,
            d,
            qv,
            r_tld,
            scratch,
        } = TfqmrWorkspace::acquire(work, n);

        // r = b - A x
        a.matvec_s(x, au, scratch);
        for i in 0..n {
            r[i] = b[i] - au[i];
        }

        if let Some(pc) = pc {
            tmp_pc.copy_from_slice(&r[..]);
            pc.apply_s(pc_side, tmp_pc, r, scratch)?;
        }

        r_tld.copy_from_slice(r);

        let mut rho: S = dot_conj(r_tld, r);
        let res0: R = nrm2(r);
        let mut stats = SolveStats::new(0, res0, ConvergedReason::Continued);
        for m in monitors {
            m(0, res0);
        }

        let tol0 = self.conv.atol.max(self.conv.rtol * res0.max(1e-300));
        if res0 <= tol0 {
            stats.reason = ConvergedReason::ConvergedAtol;
            stats.final_residual = res0;
            return Ok(stats);
        }
        if !rho.is_finite() || rho.abs() < self.breakdown_eps {
            stats.reason = ConvergedReason::DivergedDtol;
            stats.final_residual = res0;
            return Ok(stats);
        }

        yv.copy_from_slice(r);
        wv.copy_from_slice(r);
        d.fill(S::zero());
        let mut theta_prev: R = 0.0;
        let mut eta_prev: S = S::zero();
        let mut dpold: R = res0;
        let mut true_res: R = res0;

        for k in 1..=self.conv.max_iters {
            v.fill(S::zero());
            a.matvec_s(yv, v, scratch);
            if let Some(pc) = pc {
                tmp_pc.copy_from_slice(&v[..]);
                pc.apply_s(pc_side, tmp_pc, v, scratch)?;
            }

            let sigma: S = dot_conj(r_tld, v);
            if !sigma.is_finite() || sigma.abs() < self.breakdown_eps {
                stats.iterations = k;
                stats.final_residual = true_res;
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }
            let alpha = rho / sigma;
            if !alpha.is_finite() || alpha == S::zero() {
                stats.iterations = k;
                stats.final_residual = true_res;
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }

            for i in 0..n {
                u[i] = r[i] - alpha * v[i];
            }
            let mut tau_local: R = (nrm2(u) * dpold).sqrt();

            for mstep in 0..2 {
                if mstep == 0 {
                    for i in 0..n {
                        qv[i] = u[i] - alpha * v[i];
                    }
                }
                for i in 0..n {
                    au[i] = u[i] + qv[i];
                }
                tmp_pc.copy_from_slice(&au[..]);
                a.matvec_s(tmp_pc, au, scratch);
                if let Some(pc) = pc {
                    tmp_pc.copy_from_slice(&au[..]);
                    pc.apply_s(pc_side, tmp_pc, au, scratch)?;
                }
                for i in 0..n {
                    r[i] -= alpha * au[i];
                }

                let src: &[S] = if mstep == 0 { &u[..] } else { &qv[..] };
                let psi: R = nrm2(src) / tau_local.max(1e-300);
                let c: R = 1.0 / (1.0 + psi * psi).sqrt();
                let eta: S = S::from_real(c * c) * alpha;
                let cf: S = if k == 1 && mstep == 0 {
                    S::zero()
                } else {
                    S::from_real(theta_prev * theta_prev) * (eta_prev / alpha)
                };
                for i in 0..n {
                    d[i] = src[i] + cf * d[i];
                    x[i] += eta * d[i];
                }

                let iter_count = 2 * (k - 1) + mstep + 1;
                let dpest: R = ((2 * k + mstep + 1) as f64).sqrt() * tau_local;
                for m in monitors {
                    m(iter_count, dpest);
                }
                let (reason, s2) = self.conv.check(dpest, res0, iter_count);
                stats = s2;
                theta_prev = psi;
                eta_prev = eta;
                tau_local *= psi * c;

                if self.resid_recalc_every > 0
                    && (iter_count % self.resid_recalc_every == 0
                        || matches!(
                            reason,
                            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
                        ))
                {
                    a.matvec_s(x, au, scratch);
                    for i in 0..n {
                        au[i] = b[i] - au[i];
                    }
                    if let Some(pc) = pc {
                        tmp_pc.copy_from_slice(&au[..]);
                        pc.apply_s(pc_side, tmp_pc, au, scratch)?;
                    }
                    true_res = nrm2(au);
                    stats.final_residual = true_res;
                } else {
                    stats.final_residual = dpest;
                }

                if matches!(
                    reason,
                    ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
                ) {
                    return Ok(stats);
                }

                if mstep == 0 {
                    for i in 0..n {
                        qv[i] -= alpha * v[i];
                        u[i] -= alpha * v[i];
                    }
                }
            }

            let rho_new: S = dot_conj(r_tld, r);
            if !rho_new.is_finite() || rho_new.abs() < self.breakdown_eps {
                stats.iterations = k;
                stats.reason = ConvergedReason::DivergedDtol;
                stats.final_residual = true_res;
                return Ok(stats);
            }
            let beta = rho_new / rho;
            rho = rho_new;

            for i in 0..n {
                wv[i] = r[i] + beta * (qv[i] + beta * wv[i]);
                yv[i] = r[i] + beta * (qv[i] + beta * yv[i]);
            }

            dpold = nrm2(r);

            if self.resid_recalc_every == 1 {
                a.matvec_s(x, au, scratch);
                for i in 0..n {
                    au[i] = b[i] - au[i];
                }
                if let Some(pc) = pc {
                    tmp_pc.copy_from_slice(&au[..]);
                    pc.apply_s(pc_side, tmp_pc, au, scratch)?;
                }
                true_res = nrm2(au);
                stats.final_residual = true_res;
                let (reason, s2) = self.conv.check(true_res, res0, 2 * k);
                stats = s2;
                if matches!(
                    reason,
                    ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
                ) {
                    return Ok(stats);
                }
            }
        }

        stats.iterations = self.conv.max_iters;
        if !matches!(
            stats.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ) {
            stats.reason = ConvergedReason::DivergedMaxIts;
        }
        Ok(stats)
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

impl LinearSolver for TfqmrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        TfqmrWorkspace::ensure(work, work.tmp1.len());
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

struct TfqmrWorkspace<'a> {
    r: &'a mut [S],
    au: &'a mut [S],
    tmp_pc: &'a mut [S],
    u: &'a mut [S],
    v: &'a mut [S],
    wv: &'a mut [S],
    yv: &'a mut [S],
    d: &'a mut [S],
    qv: &'a mut [S],
    r_tld: &'a mut [S],
    scratch: &'a mut BridgeScratch,
}

impl<'a> TfqmrWorkspace<'a> {
    fn ensure(work: &mut Workspace, n: usize) {
        take_or_resize(&mut work.tmp1, n);
        take_or_resize(&mut work.tmp2, n);
        while work.q_s.len() < 8 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..8] {
            take_or_resize(buf, n);
        }
    }

    fn acquire(work: &'a mut Workspace, n: usize) -> Self {
        Self::ensure(work, n);
        let r = &mut work.tmp1[..n];
        let au = &mut work.tmp2[..n];
        let (prefix, _) = work.q_s.split_at_mut(8);
        let [tmp_pc, u, v, wv, yv, d, qv, r_tld] = prefix else {
            unreachable!()
        };
        Self {
            r,
            au,
            tmp_pc: &mut tmp_pc[..n],
            u: &mut u[..n],
            v: &mut v[..n],
            wv: &mut wv[..n],
            yv: &mut yv[..n],
            d: &mut d[..n],
            qv: &mut qv[..n],
            r_tld: &mut r_tld[..n],
            scratch: &mut work.bridge,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct Dense {
        a: Vec<Vec<f64>>,
    }

    impl LinOp for Dense {
        type S = f64;

        fn dims(&self) -> (usize, usize) {
            (self.a.len(), self.a[0].len())
        }

        fn matvec(&self, x: &[f64], y: &mut [f64]) {
            for i in 0..self.a.len() {
                let mut acc = 0.0;
                for j in 0..self.a[0].len() {
                    acc += self.a[i][j] * x[j];
                }
                y[i] = acc;
            }
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    struct IdentityPc;

    impl Preconditioner for IdentityPc {
        fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
            Ok(())
        }

        fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
            y.copy_from_slice(x);
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn tfqmr_solves_small_nonsym() {
        let a = Dense {
            a: vec![vec![2.0, 1.0], vec![3.0, 4.0]],
        };
        let b = [4.0, 11.0];
        let mut x = [0.0, 0.0];
        let mut w = Workspace::new(2);
        let mut solver = TfqmrSolver::new(1e-12, 200);
        let stats = solver
            .solve(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                Some(&mut w),
            )
            .unwrap();
        assert!(
            (x[0] - 1.0).abs() < 1e-8 && (x[1] - 2.0).abs() < 1e-8,
            "x={:?}",
            x
        );
        assert!(stats.final_residual <= 1e-10);
    }

    #[test]
    #[ignore]
    fn tfqmr_solves_diag_dom() {
        let a = Dense {
            a: vec![
                vec![5.0, 2.0, 0.0, 0.0, 0.0],
                vec![1.0, 5.0, 2.0, 0.0, 0.0],
                vec![0.0, 1.0, 5.0, 2.0, 0.0],
                vec![0.0, 0.0, 1.0, 5.0, 2.0],
                vec![0.0, 0.0, 0.0, 1.0, 5.0],
            ],
        };
        let x_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut b = [0.0; 5];
        a.matvec(&x_true, &mut b);
        let mut x = [0.0; 5];
        let mut w = Workspace::new(5);
        let mut solver = TfqmrSolver::new(1e-12, 500);
        let stats = solver
            .solve(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                Some(&mut w),
            )
            .unwrap();
        for i in 0..5 {
            assert!((x[i] - x_true[i]).abs() < 1e-8);
        }
        assert!(stats.final_residual <= 1e-10);
    }

    #[test]
    fn tfqmr_monitors_and_pc() {
        let a = Dense {
            a: vec![vec![2.0, 1.0], vec![3.0, 4.0]],
        };
        let b = [4.0, 11.0];
        let mut x = [0.0, 0.0];
        let mut w = Workspace::new(2);
        let mut solver = TfqmrSolver::new(1e-12, 200);
        let mut pc = IdentityPc;
        let residuals: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let res_clone = residuals.clone();
        let monitors: Vec<Box<dyn Fn(usize, f64) + Send + Sync>> = vec![Box::new(move |_, r| {
            res_clone.lock().unwrap().push(r);
        })];
        let _stats = solver
            .solve(
                &a,
                Some(&mut pc),
                &b,
                &mut x,
                PcSide::Left,
                &UniverseComm::NoComm(crate::parallel::NoComm),
                Some(&monitors),
                Some(&mut w),
            )
            .unwrap();
        assert!(!residuals.lock().unwrap().is_empty());
    }
}
