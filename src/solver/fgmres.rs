//! Flexible GMRES (FGMRES) over &dyn LinOp<f64>, right-preconditioned, object-safe.

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, SolveStats};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::trace;
use std::any::Any;

/// Orthogonalization flavor
#[derive(Clone, Copy)]
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
    pub modify_pc_on_restart:
        Option<Box<dyn FnMut(usize, f64) -> Result<(), KError> + Send + Sync>>,
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
        // We need: tmp1, tmp2 sized; q vectors for V(0..m) and Z(0..m-1): total 2m+1.
        if w.tmp1.len() != n {
            w.tmp1.resize(n, 0.0);
        }
        if w.tmp2.len() != n {
            w.tmp2.resize(n, 0.0);
        }
        let need_q = 2 * m + 1;
        if w.q.len() < need_q {
            w.q.resize(need_q, Vec::new());
        }
        for q in &mut w.q {
            if q.len() != n {
                q.resize(n, 0.0);
            }
        }
        // H: (m+1) x m
        if w.h.len() < m + 1 {
            w.h.resize(m + 1, Vec::new());
        }
        for r in &mut w.h {
            if r.len() < m {
                r.resize(m, 0.0);
            }
        }
        // Givens + RHS
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

        let block_m = if self.preallocate {
            self.restart.min(self.maxits)
        } else {
            self.restart
        };

        let mut owned_ws;
        let ws = if let Some(ws) = work {
            ws
        } else {
            owned_ws = Workspace {
                tmp1: vec![0.0; n],
                tmp2: vec![0.0; n],
                q: vec![vec![0.0; n]; 2 * block_m + 1],
                h: vec![vec![0.0; block_m]; block_m + 1],
                cs: vec![0.0; block_m],
                sn: vec![0.0; block_m],
                g: vec![0.0; block_m + 1],
            };
            &mut owned_ws
        };
        self.ensure_workspace(ws, n, block_m);

        let v_off = 0usize;
        let z_off = block_m + 1;

        a.matvec(x, &mut ws.tmp1);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }
        let mut beta0 = Self::nrm2(&ws.tmp1, comm);
        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let thr = self.atol.max(self.rtol * bnorm);

        if beta0 > 0.0 {
            let v0 = &mut ws.q[v_off + 0][..];
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
                let (v_part, z_part) = ws.q.split_at_mut(z_off);
                {
                    let vj = &v_part[v_off + j][..];
                    let zj = &mut z_part[j][..];
                    if let Some(pc_) = pc.as_deref_mut() {
                        pc_.apply_mut(pc_side, vj, zj)?;
                    } else {
                        zj.copy_from_slice(vj);
                    }
                }
                {
                    let zj = &z_part[j][..];
                    a.matvec(zj, &mut ws.tmp2);
                }

                for i in 0..=j {
                    let vi = &v_part[v_off + i][..];
                    let hij = Self::dot(&ws.tmp2, vi, comm);
                    ws.h[i][j] = hij;
                    for (w_i, &vi_val) in ws.tmp2.iter_mut().zip(vi) {
                        *w_i -= hij * vi_val;
                    }
                }

                if matches!(self.orthog, Orthog::Modified) {
                    for i in 0..=j {
                        let vi = &v_part[v_off + i][..];
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
                    let v_next = &mut v_part[v_off + j + 1][..];
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
                let zi = &ws.q[z_off + i][..];
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
            let v0 = &mut ws.q[v_off + 0][..];
            if beta0 > 0.0 {
                for i in 0..n {
                    v0[i] = ws.tmp1[i] / beta0;
                }
            } else {
                v0.fill(0.0);
            }
            if let Some(hook) = self.modify_pc_on_restart.as_mut() {
                hook(total_iters, beta0)?;
            }
        }

        stats.iterations = total_iters;
        stats.final_residual = res;
        if matches!(stats.reason, ConvergedReason::Continued) {
            stats.reason = if res <= thr {
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
        // Reserve for worst-case restart used at runtime; n is filled later in solve.
        let m = self.restart;
        if w.q.len() < 2 * m + 1 {
            w.q.resize(2 * m + 1, Vec::new());
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
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("FGMRES");

        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput(
                "FGMRES requires a square operator".into(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput("FGMRES: vector size mismatch".into()));
        }

        // Choose restart block size
        let block_m = if self.preallocate {
            self.restart.min(self.maxits)
        } else {
            self.restart
        };

        // Acquire/size workspace
        let mut owned_ws;
        let ws = if let Some(ws) = work {
            ws
        } else {
            owned_ws = Workspace {
                tmp1: vec![0.0; n],
                tmp2: vec![0.0; n],
                q: vec![vec![0.0; n]; 2 * block_m + 1],
                h: vec![vec![0.0; block_m]; block_m + 1],
                cs: vec![0.0; block_m],
                sn: vec![0.0; block_m],
                g: vec![0.0; block_m + 1],
            };
            &mut owned_ws
        };
        self.ensure_workspace(ws, n, block_m);

        // Layout in Workspace.q:
        //   V-basis: q[0..=m] (len m+1)
        //   Z-basis: q[m+1 .. m+m+1] (len m)
        let v_off = 0usize;
        let z_off = block_m + 1;

        // r = b - A x
        a.matvec(x, &mut ws.tmp1);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }
        let mut beta0 = Self::nrm2(&ws.tmp1, comm);
        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let thr = self.atol.max(self.rtol * bnorm);

        // v0 = r / beta
        if beta0 > 0.0 {
            let v0 = &mut ws.q[v_off + 0][..];
            for i in 0..n {
                v0[i] = ws.tmp1[i] / beta0;
            }
        }

        // g = [beta, 0, 0, ...]
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

        // Outer cycles
        while total_iters < self.maxits {
            let m_this = if self.preallocate {
                block_m.min(self.maxits - total_iters)
            } else {
                self.restart.min(self.maxits - total_iters)
            };

            let mut arnoldi_steps = 0usize;
            let mut converged = false;

            for j in 0..m_this {
                let (v_part, z_part) = ws.q.split_at_mut(z_off);
                // z_j = M^{-1} v_j
                {
                    let vj = &v_part[v_off + j][..];
                    let zj = &mut z_part[j][..];
                    if let Some(pc_) = pc {
                        pc_.apply(pc_side, vj, zj)?;
                    } else {
                        zj.copy_from_slice(vj);
                    }
                }
                // w = A z_j -> tmp2
                {
                    let zj = &z_part[j][..];
                    a.matvec(zj, &mut ws.tmp2);
                }

                // h[0..j] = v_i^T w; w -= sum h_ij v_i
                for i in 0..=j {
                    let vi = &v_part[v_off + i][..];
                    let hij = Self::dot(&ws.tmp2, vi, comm);
                    ws.h[i][j] = hij;
                    for (w_i, &vi_val) in ws.tmp2.iter_mut().zip(vi) {
                        *w_i -= hij * vi_val;
                    }
                }

                if matches!(self.orthog, Orthog::Modified) {
                    for i in 0..=j {
                        let vi = &v_part[v_off + i][..];
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
                    let v_next = &mut v_part[v_off + j + 1][..];
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

            // Back-substitute y
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
                let zi = &ws.q[z_off + i][..];
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
            let v0 = &mut ws.q[v_off + 0][..];
            if beta0 > 0.0 {
                for i in 0..n {
                    v0[i] = ws.tmp1[i] / beta0;
                }
            } else {
                v0.fill(0.0);
            }
            if let Some(hook) = self.modify_pc_on_restart.as_mut() {
                hook(total_iters, beta0)?;
            }
        }

        stats.iterations = total_iters;
        stats.final_residual = res;
        if matches!(stats.reason, ConvergedReason::Continued) {
            stats.reason = if res <= thr {
                ConvergedReason::ConvergedRtol
            } else {
                ConvergedReason::DivergedMaxIts
            };
        }

        #[cfg(feature = "logging")]
        trace!(
            "FGMRES done: iters={}, resid={:.3e}",
            stats.iterations, stats.final_residual
        );
        Ok(stats)
    }
}
