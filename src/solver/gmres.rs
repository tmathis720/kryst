//! # GMRES side semantics
//!
//! - **Left**: Arnoldi on `M^{-1} A`
//!   - `v1 = M^{-1} r0 / ||M^{-1} r0||`
//!   - Krylov matvec: `w = M^{-1} A v_k`
//!   - Update: `x += V y`
//!   - Iteration monitors should report `||M^{-1} r||`; final stats report true `||r||`.
//!
//! - **Right**: Arnoldi on `A M^{-1}`
//!   - `v1 = r0 / ||r0||`, `u = M^{-1} v_k`, `w = A u`
//!   - Store `Z_k = u` and update via `x += Z y`
//!   - Monitors report true `||r||`.
//!
//! Implementation detail: `Workspace.z` holds the `Z_k` basis for Right and FGMRES.

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::recompute_true_residual_norm;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GmresOrthog {
    Mgs,
    Cgs,
}

pub struct GmresSolver {
    pub restart: usize,
    pub conv: Convergence<f64>,
    /// Happy breakdown tolerance
    pub haptol: f64,
    /// Orthogonalization flavor (currently not altering algorithmic path)
    pub orthog: GmresOrthog,
    /// Whether to perform a second orthogonalization pass
    pub reorthog: bool,
    /// Whether to treat near-zero residual as a happy breakdown
    pub happy_breakdown: bool,
}

impl GmresSolver {
    pub fn new(restart: usize, rtol: f64, maxits: usize) -> Self {
        Self {
            restart: restart.max(1),
            conv: Convergence {
                rtol,
                atol: 1e-12,
                dtol: 1e3,
                max_iters: maxits,
            },
            haptol: 1e-12,
            orthog: GmresOrthog::Mgs,
            reorthog: false,
            happy_breakdown: true,
        }
    }

    #[inline]
    fn dot(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
    }
    #[inline]
    fn nrm2(x: &[f64]) -> f64 {
        Self::dot(x, x).sqrt()
    }

    /// y -= a * x
    #[inline]
    fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi -= a * xi;
        }
    }

    fn ensure_workspace(&self, w: &mut Workspace, n: usize, need_z: bool) {
        let m = self.restart;
        if w.tmp1.len() != n {
            w.tmp1.resize(n, 0.0);
        }
        if w.tmp2.len() != n {
            w.tmp2.resize(n, 0.0);
        }
        w.ensure_gmres_slabs(n, m, need_z);
        if w.h.len() < m + 1 {
            w.h.resize(m + 1, Vec::new());
        }
        for row in &mut w.h[..m + 1] {
            if row.len() != m {
                row.resize(m, 0.0);
            }
        }
        if w.cs.len() < m {
            w.cs.resize(m, 0.0);
        }
        if w.sn.len() < m {
            w.sn.resize(m, 0.0);
        }
        if w.g.len() < m + 1 {
            w.g.resize(m + 1, 0.0);
        }
    }

    fn apply_precond(
        &self,
        pc: Option<&dyn Preconditioner>,
        side: PcSide,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), KError> {
        if let Some(p) = pc {
            p.apply(side, x, y)
        } else {
            y.copy_from_slice(x);
            Ok(())
        }
    }

    fn apply_givens_and_update(
        h: &mut [Vec<f64>],
        cs: &mut [f64],
        sn: &mut [f64],
        g: &mut [f64],
        k: usize,
    ) {
        for i in 0..k {
            let (hij, hij1) = {
                let (top, bottom) = h.split_at_mut(i + 1);
                (&mut top[i][k], &mut bottom[0][k])
            };
            let c = cs[i];
            let s = sn[i];
            let t = c * *hij + s * *hij1;
            *hij1 = -s * *hij + c * *hij1;
            *hij = t;
        }

        let (hkk, hk1k) = {
            let (top, bottom) = h.split_at_mut(k + 1);
            (&mut top[k][k], &mut bottom[0][k])
        };
        let (c, s) = if *hk1k == 0.0 {
            (1.0, 0.0)
        } else {
            let r = (*hkk * *hkk + *hk1k * *hk1k).sqrt();
            (*hkk / r, *hk1k / r)
        };
        cs[k] = c;
        sn[k] = s;
        {
            let (top, bottom) = h.split_at_mut(k + 1);
            let hjk = &mut top[k][k];
            let hj1k = &mut bottom[0][k];
            let t = c * *hjk + s * *hj1k;
            *hj1k = -s * *hjk + c * *hj1k;
            *hjk = t;
        }
        let t = c * g[k] + s * g[k + 1];
        g[k + 1] = -s * g[k] + c * g[k + 1];
        g[k] = t;
    }

    fn backsolve(h: &[Vec<f64>], g: &[f64], k: usize) -> Vec<f64> {
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for l in (i + 1)..k {
                sum -= h[i][l] * y[l];
            }
            y[i] = sum / h[i][i];
        }
        y
    }

}

impl LinearSolver for GmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        // Workspace slabs are sized during `solve` when the operator dimensions are known.
        let _ = w;
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<SolveStats<f64>, Self::Error> {
        let pc: Option<&dyn Preconditioner> = pc.as_deref();
        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "GMRES: dimension mismatch or non-square operator".into(),
            ));
        }

        let pc_side = match pc_side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        };

        let mut owned_ws;
        let ws = if let Some(w) = work {
            w
        } else {
            owned_ws = Workspace::new(n);
            &mut owned_ws
        };
        let need_z = matches!(pc_side, PcSide::Right);
        self.ensure_workspace(ws, n, need_z);

        // r0 = b - A x
        a.matvec(x, &mut ws.tmp1);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }

        ws.h.iter_mut().for_each(|r| r.fill(0.0));
        ws.cs.fill(0.0);
        ws.sn.fill(0.0);
        ws.g.fill(0.0);

        let beta = match pc_side {
            PcSide::Left => {
                self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                let beta = Self::nrm2(&ws.tmp2);
                let v0 = ws.vcol_mut(n, 0);
                if beta > 0.0 {
                    for i in 0..n {
                        v0[i] = ws.tmp2[i] / beta;
                    }
                } else {
                    v0.fill(0.0);
                }
                beta
            }
            PcSide::Right => {
                let beta = Self::nrm2(&ws.tmp1);
                let v0 = ws.vcol_mut(n, 0);
                if beta > 0.0 {
                    for i in 0..n {
                        v0[i] = ws.tmp1[i] / beta;
                    }
                } else {
                    v0.fill(0.0);
                }
                beta
            }
            PcSide::Symmetric => unreachable!(),
        };

        ws.g[0] = beta;
        let bnorm = Self::nrm2(b).max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);

        let mut total_iters = 0usize;
        let mut res = beta;
        let mut stats = SolveStats {
            iterations: 0,
            final_residual: res,
            reason: ConvergedReason::Continued,
        };

        if let Some(ms) = monitors {
            for m in ms {
                m(0, res);
            }
        }
        if res <= thr {
            stats.reason = if res <= self.conv.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            stats.final_residual = res;
            return Ok(stats);
        }

        'outer: loop {
            let mut k_steps = 0usize;
            for k in 0..self.restart {
                match pc_side {
                    PcSide::Left => {
                        a.matvec(ws.vcol(n, k), &mut ws.tmp1);
                        self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                        for i in 0..=k {
                            let hij = Self::dot(&ws.tmp2, ws.vcol(n, i));
                            ws.h[i][k] = hij;
                            Self::axpy(&mut ws.tmp2, hij, ws.vcol(n, i));
                        }
                        let hkp1k = Self::nrm2(&ws.tmp2);
                        ws.h[k + 1][k] = hkp1k;
                        let vnext = ws.vcol_mut(n, k + 1);
                        if hkp1k > 0.0 {
                            for i in 0..n {
                                vnext[i] = ws.tmp2[i] / hkp1k;
                            }
                        } else {
                            vnext.fill(0.0);
                        }
                    }
                    PcSide::Right => {
                        self.apply_precond(pc, PcSide::Right, ws.vcol(n, k), &mut ws.tmp2)?;
                        {
                            let zk = ws.zcol_mut(n, k);
                            zk.copy_from_slice(&ws.tmp2);
                        }
                        a.matvec(ws.zcol(n, k), &mut ws.tmp1);
                        for i in 0..=k {
                            let hij = Self::dot(&ws.tmp1, ws.vcol(n, i));
                            ws.h[i][k] = hij;
                            Self::axpy(&mut ws.tmp1, hij, ws.vcol(n, i));
                        }
                        let hkp1k = Self::nrm2(&ws.tmp1);
                        ws.h[k + 1][k] = hkp1k;
                        let vnext = ws.vcol_mut(n, k + 1);
                        if hkp1k > 0.0 {
                            for i in 0..n {
                                vnext[i] = ws.tmp1[i] / hkp1k;
                            }
                        } else {
                            vnext.fill(0.0);
                        }
                    }
                    PcSide::Symmetric => unreachable!(),
                }

                Self::apply_givens_and_update(&mut ws.h, &mut ws.cs, &mut ws.sn, &mut ws.g, k);

                res = ws.g[k + 1].abs();
                total_iters += 1;
                k_steps = k + 1;

                if let Some(ms) = monitors {
                    for m in ms {
                        m(total_iters, res);
                    }
                }

                if res <= thr {
                    break;
                }

                if total_iters >= self.conv.max_iters {
                    break;
                }
            }

            let y = Self::backsolve(&ws.h, &ws.g, k_steps);
            match pc_side {
                PcSide::Left => {
                    for (j, yj) in y.iter().enumerate() {
                        let vj = ws.vcol(n, j);
                        for (xi, &vj_i) in x.iter_mut().zip(vj) {
                            *xi += yj * vj_i;
                        }
                    }
                }
                PcSide::Right => {
                    for (j, yj) in y.iter().enumerate() {
                        let zj = ws.zcol(n, j);
                        for (xi, &zj_i) in x.iter_mut().zip(zj) {
                            *xi += yj * zj_i;
                        }
                    }
                }
                PcSide::Symmetric => unreachable!(),
            }

            // Recompute residual
            a.matvec(x, &mut ws.tmp1);
            for i in 0..n {
                ws.tmp1[i] = b[i] - ws.tmp1[i];
            }
            res = match pc_side {
                PcSide::Left => {
                    self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                    Self::nrm2(&ws.tmp2)
                }
                PcSide::Right => Self::nrm2(&ws.tmp1),
                PcSide::Symmetric => unreachable!(),
            };

            if res <= thr || total_iters >= self.conv.max_iters {
                break 'outer;
            }

            // Prepare next cycle
            ws.h.iter_mut().for_each(|r| r.fill(0.0));
            ws.cs.fill(0.0);
            ws.sn.fill(0.0);
            ws.g.fill(0.0);

            match pc_side {
                PcSide::Left => {
                    self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                    let beta = Self::nrm2(&ws.tmp2);
                    let v0 = ws.vcol_mut(n, 0);
                    if beta > 0.0 {
                        for i in 0..n {
                            v0[i] = ws.tmp2[i] / beta;
                        }
                    } else {
                        v0.fill(0.0);
                    }
                    ws.g[0] = beta;
                }
                PcSide::Right => {
                    let beta = Self::nrm2(&ws.tmp1);
                    let v0 = ws.vcol_mut(n, 0);
                    if beta > 0.0 {
                        for i in 0..n {
                            v0[i] = ws.tmp1[i] / beta;
                        }
                    } else {
                        v0.fill(0.0);
                    }
                    ws.g[0] = beta;
                }
                PcSide::Symmetric => unreachable!(),
            }
        }

        // Compute true residual for reporting using the communicator
        let true_res = recompute_true_residual_norm(a, b, x, _comm, &mut ws.tmp1);
        let (mut reason, mut s) = self.conv.check(true_res, bnorm, total_iters);
        s.final_residual = true_res;
        if matches!(reason, ConvergedReason::Continued) {
            // Normalize reason based on absolute/relative thresholds
            reason = if true_res <= self.conv.atol {
                ConvergedReason::ConvergedAtol
            } else if true_res <= self.conv.rtol * bnorm {
                ConvergedReason::ConvergedRtol
            } else {
                ConvergedReason::DivergedMaxIts
            };
            s.reason = reason;
        }
        Ok(s)
    }
}

impl GmresSolver {
    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart.max(1);
    }
    pub fn set_orthog(&mut self, o: GmresOrthog) {
        self.orthog = o;
    }
    pub fn set_reorthog(&mut self, flag: bool) {
        self.reorthog = flag;
    }
    pub fn set_happy_breakdown(&mut self, flag: bool) {
        self.happy_breakdown = flag;
    }

    #[cfg(test)]
    pub fn debug_config(&self) -> (usize, GmresOrthog, bool, bool) {
        (self.restart, self.orthog, self.reorthog, self.happy_breakdown)
    }
}
