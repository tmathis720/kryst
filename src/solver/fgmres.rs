//! Flexible GMRES (FGMRES) over &dyn LinOp<f64>, right-preconditioned, object-safe.

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, SolveStats};
use crate::solver::common::recompute_true_residual_norm;
use std::any::Any;

/// Orthogonalization flavor
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orthog {
    Classical,
    Modified,
}

pub struct FgmresSolver {
    pub rtol: f64,
    pub atol: f64,
    pub dtol: f64,
    pub maxits: usize,
    pub restart: usize,
    pub orthog: Orthog,
    pub haptol: f64,
    /// If true, size basis/H for maxits; otherwise per-restart sizing.
    pub preallocate: bool,
    /// Optional hook called once per outer iteration block (after backsolve) to allow user to tweak PC.
    #[deprecated(note = "Use Preconditioner::on_restart(iter, res) instead")]
    pub modify_pc_on_restart:
        Option<Box<dyn FnMut(usize, f64) -> Result<(), KError> + Send + Sync>>,
    /// Whether to treat near-zero residual as a happy breakdown
    pub happy_breakdown: bool,
}

impl FgmresSolver {
    pub fn new(rtol: f64, maxits: usize, restart: usize) -> Self {
        Self {
            rtol,
            atol: 1e-12,
            dtol: 1e3,
            maxits,
            restart: restart.max(1),
            orthog: Orthog::Classical,
            haptol: 1e-12,
            preallocate: false,
            modify_pc_on_restart: None,
            happy_breakdown: true,
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

    fn ensure_workspace(&self, w: &mut Workspace, n: usize, m: usize) {
        // Need tmp1/tmp2, V basis q[0..=m], Z basis z[0..m-1]
        if w.tmp1.len() != n {
            w.tmp1.resize(n, 0.0);
        }
        if w.tmp2.len() != n {
            w.tmp2.resize(n, 0.0);
        }
        if w.q.len() < m + 1 {
            w.q.resize(m + 1, Vec::new());
        }
        for q in &mut w.q[..m + 1] {
            if q.len() != n {
                q.resize(n, 0.0);
            }
        }
        if w.z.len() < m {
            w.z.resize(m, Vec::new());
        }
        for z in &mut w.z[..m] {
            if z.len() != n {
                z.resize(n, 0.0);
            }
        }
        if w.h.len() < m + 1 {
            w.h.resize(m + 1, Vec::new());
        }
        for r in &mut w.h[..m + 1] {
            if r.len() < m {
                r.resize(m, 0.0);
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

    fn apply_givens(hij: &mut f64, hij1: &mut f64, c: f64, s: f64) {
        let t = c * (*hij) + s * (*hij1);
        *hij1 = -s * (*hij) + c * (*hij1);
        *hij = t;
    }

    fn givens(a: f64, b: f64) -> (f64, f64) {
        if b == 0.0 {
            (1.0, 0.0)
        } else {
            let r = (a * a + b * b).sqrt();
            (a / r, b / r)
        }
    }

    pub fn solve_flexible(
        &mut self,
        a: &dyn LinOp<S = f64>,
        mut pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput(
                "FGMRES requires a square operator".into(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput("FGMRES: vector size mismatch".into()));
        }

        let pc_side = match pc_side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        };

        let block_m = if self.preallocate {
            self.restart.min(self.maxits)
        } else {
            self.restart
        };

        let mut owned_ws;
        let had_ws = work.is_some();
        let ws = if let Some(ws) = work {
            ws
        } else {
            owned_ws = Workspace {
                tmp1: vec![0.0; n],
                tmp2: vec![0.0; n],
                q: vec![vec![0.0; n]; block_m + 1],
                h: vec![vec![0.0; block_m]; block_m + 1],
                cs: vec![0.0; block_m],
                sn: vec![0.0; block_m],
                g: vec![0.0; block_m + 1],
                z: vec![vec![0.0; n]; block_m],
            };
            &mut owned_ws
        };
        self.ensure_workspace(ws, n, block_m);

        a.matvec(x, &mut ws.tmp1);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }
        let mut beta0 = Self::nrm2(&ws.tmp1, comm);
        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let thr = self.atol.max(self.rtol * bnorm);

        if beta0 > 0.0 {
            let v0 = &mut ws.q[0][..];
            for i in 0..n {
                v0[i] = ws.tmp1[i] / beta0;
            }
        }

        ws.g.fill(0.0);
        ws.g[0] = beta0;

        let mut total_iters = 0usize;
        let mut res = beta0;
        let mut stats = SolveStats {
            iterations: 0,
            final_residual: res,
            reason: ConvergedReason::Continued,
        };

        if let Some(mons) = monitors {
            for m in mons {
                m(0, res);
            }
        }
        if res <= thr {
            stats.final_residual = res;
            stats.reason = if res <= self.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            return Ok(stats);
        }

        while total_iters < self.maxits {
            let m_this = if self.preallocate {
                block_m.min(self.maxits - total_iters)
            } else {
                self.restart.min(self.maxits - total_iters)
            };

            let mut arnoldi_steps = 0usize;
            let mut converged = false;

            for j in 0..m_this {
                {
                    let vj = &ws.q[j][..];
                    let zj = &mut ws.z[j][..];
                    if let Some(pc_) = pc.as_deref_mut() {
                        pc_.apply_mut(pc_side, vj, zj)?;
                    } else {
                        zj.copy_from_slice(vj);
                    }
                }
                {
                    let zj = &ws.z[j][..];
                    a.matvec(zj, &mut ws.tmp2);
                }

                for i in 0..=j {
                    let vi = &ws.q[i][..];
                    let hij = Self::dot(&ws.tmp2, vi, comm);
                    ws.h[i][j] = hij;
                    for (w_i, &vi_val) in ws.tmp2.iter_mut().zip(vi) {
                        *w_i -= hij * vi_val;
                    }
                }

                if matches!(self.orthog, Orthog::Modified) {
                    for i in 0..=j {
                        let vi = &ws.q[i][..];
                        let corr = Self::dot(&ws.tmp2, vi, comm);
                        if corr.abs() > 1e-12 {
                            ws.h[i][j] += corr;
                            for (w_i, &vi_val) in ws.tmp2.iter_mut().zip(vi) {
                                *w_i -= corr * vi_val;
                            }
                        }
                    }
                }

                let hij1 = Self::nrm2(&ws.tmp2, comm);
                ws.h[j + 1][j] = hij1;

                {
                    let v_next = &mut ws.q[j + 1][..];
                    if hij1 > 0.0 {
                        for i in 0..n {
                            v_next[i] = ws.tmp2[i] / hij1;
                        }
                    } else {
                        v_next.fill(0.0);
                    }
                }

                for i in 0..j {
                    let (top, bottom) = ws.h.split_at_mut(i + 1);
                    let hij = &mut top[i][j];
                    let hij1 = &mut bottom[0][j];
                    Self::apply_givens(hij, hij1, ws.cs[i], ws.sn[i]);
                }
                let (c, s) = {
                    let (top, bottom) = ws.h.split_at_mut(j + 1);
                    let hjj = top[j][j];
                    let hj1j = bottom[0][j];
                    Self::givens(hjj, hj1j)
                };
                ws.cs[j] = c;
                ws.sn[j] = s;
                {
                    let (top, bottom) = ws.h.split_at_mut(j + 1);
                    let hjj = &mut top[j][j];
                    let hj1j = &mut bottom[0][j];
                    Self::apply_givens(hjj, hj1j, c, s);
                }
                let t = c * ws.g[j] + s * ws.g[j + 1];
                ws.g[j + 1] = -s * ws.g[j] + c * ws.g[j + 1];
                ws.g[j] = t;

                res = ws.g[j + 1].abs();
                total_iters += 1;
                arnoldi_steps = j + 1;

                if let Some(mons) = monitors {
                    for m in mons {
                        m(total_iters, res);
                    }
                }

                let res0 = beta0;
                let (reason, sstats) = crate::utils::convergence::Convergence {
                    rtol: self.rtol,
                    atol: self.atol,
                    dtol: self.dtol,
                    max_iters: self.maxits,
                }
                .check(res, res0, total_iters);
                stats = sstats;
                if matches!(
                    reason,
                    ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
                ) {
                    stats.final_residual = res;
                    stats.iterations = total_iters;
                    converged = true;
                    break;
                }
            }

            let k = arnoldi_steps;
            let mut y = vec![0.0; k];
            for i in (0..k).rev() {
                let mut sum = ws.g[i];
                for l in (i + 1)..k {
                    sum -= ws.h[i][l] * y[l];
                }
                y[i] = sum / ws.h[i][i];
            }

            for i in 0..k {
                let zi = &ws.z[i][..];
                for (xj, &zij) in x.iter_mut().zip(zi) {
                    *xj += y[i] * zij;
                }
            }

            if converged {
                stats.reason = if res <= self.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                };
                stats.final_residual = res;
                break;
            }

            if total_iters >= self.maxits {
                break;
            }

            a.matvec(x, &mut ws.tmp1);
            for i in 0..n {
                ws.tmp1[i] = b[i] - ws.tmp1[i];
            }
            beta0 = Self::nrm2(&ws.tmp1, comm);
            ws.g.fill(0.0);
            ws.g[0] = beta0;
            let v0 = &mut ws.q[0][..];
            if beta0 > 0.0 {
                for i in 0..n {
                    v0[i] = ws.tmp1[i] / beta0;
                }
            } else {
                v0.fill(0.0);
            }
            // Allow both legacy hook and the new Preconditioner::on_restart to adjust PC
            if let Some(hook) = self.modify_pc_on_restart.as_mut() {
                hook(total_iters, beta0)?;
            }
            if let Some(pc_) = pc.as_deref_mut() {
                pc_.on_restart(total_iters, beta0)?;
            }
        }

        stats.iterations = total_iters;

        // Compute true residual at exit for reporting
        let true_res = if had_ws {
            // `ws.tmp1` has length n; safe to reuse for recompute
            recompute_true_residual_norm(a, b, x, comm, &mut ws.tmp1)
        } else {
            let mut tmp = vec![0.0; n];
            recompute_true_residual_norm(a, b, x, comm, &mut tmp)
        };
        stats.final_residual = true_res;

        if matches!(stats.reason, ConvergedReason::Continued) {
            stats.reason = if true_res <= self.atol {
                ConvergedReason::ConvergedAtol
            } else if true_res <= self.rtol * bnorm {
                ConvergedReason::ConvergedRtol
            } else {
                ConvergedReason::DivergedMaxIts
            };
        }
        Ok(stats)
    }
}

impl LinearSolver for FgmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        let m = self.restart;
        if w.q.len() < m + 1 {
            w.q.resize(m + 1, Vec::new());
        }
        if w.z.len() < m {
            w.z.resize(m, Vec::new());
        }
        if w.h.len() < m + 1 {
            w.h.resize(m + 1, Vec::new());
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
        // Delegate directly to the flexible path with mutable PC support.
        self.solve_flexible(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

impl FgmresSolver {
    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart.max(1);
    }
    pub fn set_orthog(&mut self, o: Orthog) {
        self.orthog = o;
    }
    pub fn set_reorthog(&mut self, flag: bool) {
        self.orthog = if flag { Orthog::Modified } else { Orthog::Classical };
    }
    pub fn set_happy_breakdown(&mut self, flag: bool) {
        self.happy_breakdown = flag;
    }

    #[cfg(test)]
    pub fn debug_config(&self) -> (usize, Orthog, bool, bool) {
        (
            self.restart,
            self.orthog,
            matches!(self.orthog, Orthog::Modified),
            self.happy_breakdown,
        )
    }
}
