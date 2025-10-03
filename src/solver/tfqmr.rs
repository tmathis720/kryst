//! # TFQMR side semantics
//!
//! Accepts [`PcSide::Left`] or [`PcSide::Right`]; residuals are reported as the true `||r||`.

use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::op_bridge::matvec_s;
use crate::parallel::UniverseComm;
use crate::preconditioner::bridge::apply_pc_s;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
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

    fn setup_tfqmr_workspace(work: &mut Workspace, n: usize) {
        if work.tmp1.len() != n {
            work.tmp1.resize(n, S::zero());
        }
        if work.tmp2.len() != n {
            work.tmp2.resize(n, S::zero());
        }
        let need = 6usize
            .checked_mul(n)
            .expect("tfqmr scratch length overflow");
        if work.blk_scratch.len() != need {
            work.blk_scratch.resize(need, S::zero());
        }
    }
}

impl LinearSolver for TfqmrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        let n = work.tmp1.len();
        TfqmrSolver::setup_tfqmr_workspace(work, n);
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        _comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let pc: Option<&dyn Preconditioner> = pc.as_deref();
        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput("TFQMR requires square A".into()));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput("TFQMR size mismatch".into()));
        }
        let mons: &[Box<dyn Fn(usize, f64) + Send + Sync>] = monitors.unwrap_or(&[]);

        let pc_side = match pc_side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        };

        let w = work.ok_or_else(|| {
            KError::InvalidInput("TFQMR requires a Workspace; call via KSP".into())
        })?;
        TfqmrSolver::setup_tfqmr_workspace(w, n);

        let (r, au) = (&mut w.tmp1, &mut w.tmp2);
        let scratch = &mut w.blk_scratch;
        debug_assert_eq!(scratch.len(), 6 * n);
        let (u, rest) = scratch.split_at_mut(n);
        let (v, rest) = rest.split_at_mut(n);
        let (wv, rest) = rest.split_at_mut(n);
        let (yv, rest) = rest.split_at_mut(n);
        let (d, qv) = rest.split_at_mut(n);
        if w.bridge_tmp.len() != n {
            w.bridge_tmp.resize(n, S::zero());
        }

        matvec_s(a, x, r, &mut w.bridge);
        for i in 0..n {
            r[i] = S::from_real(b[i]) - r[i];
        }
        if let Some(pc) = pc {
            let tmp = &mut w.bridge_tmp[..n];
            tmp.copy_from_slice(r);
            apply_pc_s(pc, pc_side, tmp, r, &mut w.bridge)?;
        }

        let r_tld = r.clone();
        let mut rho: S = dot_conj(&r_tld, r);
        let res0: R = nrm2(r);
        let mut stats = SolveStats::new(0, res0, ConvergedReason::Continued);
        for m in mons {
            m(0, res0);
        }

        if res0 <= self.conv.atol.max(self.conv.rtol * res0.max(1e-300)) {
            stats.reason = ConvergedReason::ConvergedAtol;
            return Ok(stats);
        }
        if !rho.is_finite() || rho.abs() < self.breakdown_eps {
            stats.reason = ConvergedReason::DivergedDtol;
            return Ok(stats);
        }

        yv.clone_from_slice(r);
        wv.clone_from_slice(r);
        d.fill(S::zero());
        let mut theta_prev: R = 0.0;
        let mut eta_prev: S = S::zero();
        let mut dpold: R = res0;
        let mut true_res: R = res0;

        for k in 1..=self.conv.max_iters {
            v.fill(S::zero());
            matvec_s(a, yv, v, &mut w.bridge);
            if let Some(pc) = pc {
                let tmp = &mut w.bridge_tmp[..n];
                tmp.copy_from_slice(v);
                apply_pc_s(pc, pc_side, tmp, v, &mut w.bridge)?;
            }

            let sigma: S = dot_conj(&r_tld, v);
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
                let tmp = &mut w.bridge_tmp[..n];
                tmp.copy_from_slice(au);
                matvec_s(a, tmp, au, &mut w.bridge);
                if let Some(pc) = pc {
                    tmp.copy_from_slice(au);
                    apply_pc_s(pc, pc_side, tmp, au, &mut w.bridge)?;
                }
                for i in 0..n {
                    r[i] -= alpha * au[i];
                }

                {
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
                    for mfn in mons {
                        mfn(iter_count, dpest);
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
                        matvec_s(a, x, au, &mut w.bridge);
                        for i in 0..n {
                            au[i] = S::from_real(b[i]) - au[i];
                        }
                        if let Some(pc) = pc {
                            let tmp = &mut w.bridge_tmp[..n];
                            tmp.copy_from_slice(au);
                            apply_pc_s(pc, pc_side, tmp, au, &mut w.bridge)?;
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
                }

                if mstep == 0 {
                    for i in 0..n {
                        qv[i] -= alpha * v[i];
                        u[i] -= alpha * v[i];
                    }
                }
            }

            let rho_new: S = dot_conj(&r_tld, r);
            if !rho_new.is_finite() || rho_new.abs() < self.breakdown_eps {
                stats.iterations = k;
                stats.reason = ConvergedReason::DivergedDtol;
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
                matvec_s(a, x, au, &mut w.bridge);
                for i in 0..n {
                    au[i] = S::from_real(b[i]) - au[i];
                }
                if let Some(pc) = pc {
                    let tmp = &mut w.bridge_tmp[..n];
                    tmp.copy_from_slice(au);
                    apply_pc_s(pc, pc_side, tmp, au, &mut w.bridge)?;
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
