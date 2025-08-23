use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

pub struct TfqmrSolver {
    pub conv: Convergence<f64>,
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
            work.tmp1.resize(n, 0.0);
        }
        if work.tmp2.len() != n {
            work.tmp2.resize(n, 0.0);
        }
        while work.q.len() < 6 {
            work.q.push(vec![0.0; n]);
        }
        for v in &mut work.q[..6] {
            if v.len() != n {
                v.resize(n, 0.0);
            }
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
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        _comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
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

        let (r, Au) = (&mut w.tmp1, &mut w.tmp2);
        let (u_slice, rest) = w.q.split_at_mut(1);
        let (v_slice, rest) = rest.split_at_mut(1);
        let (wv_slice, rest) = rest.split_at_mut(1);
        let (yv_slice, rest) = rest.split_at_mut(1);
        let (d_slice, rest) = rest.split_at_mut(1);
        let (qv_slice, _) = rest.split_at_mut(1);
        let (u, v, wv, yv, d, qv) = (
            &mut u_slice[0][..],
            &mut v_slice[0][..],
            &mut wv_slice[0][..],
            &mut yv_slice[0][..],
            &mut d_slice[0][..],
            &mut qv_slice[0][..],
        );

        a.matvec(x, r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }
        if let Some(pc) = pc {
            let rin = r.clone();
            pc.apply(pc_side, &rin, r)?;
        }

        let r_tld = r.clone();
        let mut rho = dot(&r_tld, r);
        let res0 = norm2(r);
        let mut stats = SolveStats {
            iterations: 0,
            final_residual: res0,
            reason: ConvergedReason::Continued,
        };
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
        d.fill(0.0);
        let mut theta_prev = 0.0;
        let mut eta_prev = 0.0;
        let mut dpold = res0;
        let mut true_res = res0;

        for k in 1..=self.conv.max_iters {
            v.fill(0.0);
            a.matvec(yv, v);
            if let Some(pc) = pc {
                let vin = v.to_vec();
                pc.apply(pc_side, &vin, v)?;
            }

            let sigma = dot(&r_tld, v);
            if !sigma.is_finite() || sigma.abs() < self.breakdown_eps {
                stats.iterations = k;
                stats.final_residual = true_res;
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }
            let alpha = rho / sigma;
            if !alpha.is_finite() || alpha == 0.0 {
                stats.iterations = k;
                stats.final_residual = true_res;
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }

            for i in 0..n {
                u[i] = r[i] - alpha * v[i];
            }
            let mut tau_local = (norm2(u) * dpold).sqrt();

            for mstep in 0..2 {
                if mstep == 0 {
                    for i in 0..n {
                        qv[i] = u[i] - alpha * v[i];
                    }
                }
                for i in 0..n {
                    Au[i] = u[i] + qv[i];
                }
                let tmp_in = Au.to_vec();
                a.matvec(&tmp_in, Au);
                if let Some(pc) = pc {
                    let tmp2 = Au.to_vec();
                    pc.apply(pc_side, &tmp2, Au)?;
                }
                for i in 0..n {
                    r[i] -= alpha * Au[i];
                }

                {
                    let src: &[f64] = if mstep == 0 { &u[..] } else { &qv[..] };
                    let psi = norm2(src) / tau_local.max(1e-300);
                    let c = 1.0 / (1.0 + psi * psi).sqrt();
                    let eta = c * c * alpha;
                    let cf = if k == 1 && mstep == 0 {
                        0.0
                    } else {
                        theta_prev * theta_prev * (eta_prev / alpha)
                    };
                    for i in 0..n {
                        d[i] = src[i] + cf * d[i];
                        x[i] += eta * d[i];
                    }

                    let iter_count = 2 * (k - 1) + mstep + 1;
                    let dpest = ((2 * k + mstep + 1) as f64).sqrt() * tau_local;
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
                        a.matvec(x, Au);
                        for i in 0..n {
                            Au[i] = b[i] - Au[i];
                        }
                        if let Some(pc) = pc {
                            let tmp = Au.to_vec();
                            pc.apply(pc_side, &tmp, Au)?;
                        }
                        true_res = norm2(Au);
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

            let rho_new = dot(&r_tld, r);
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

            dpold = norm2(r);

            if self.resid_recalc_every == 1 {
                a.matvec(x, Au);
                for i in 0..n {
                    Au[i] = b[i] - Au[i];
                }
                if let Some(pc) = pc {
                    let tmp = Au.clone();
                    pc.apply(pc_side, &tmp, Au)?;
                }
                true_res = norm2(Au);
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

#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

#[inline]
fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
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
        let pc = IdentityPc;
        let residuals: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let res_clone = residuals.clone();
        let monitors: Vec<Box<dyn Fn(usize, f64) + Send + Sync>> = vec![Box::new(move |_, r| {
            res_clone.lock().unwrap().push(r);
        })];
        let _stats = solver
            .solve(
                &a,
                Some(&pc),
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
