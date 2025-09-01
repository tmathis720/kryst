//! Flexible GMRES (FGMRES) over &dyn LinOp<f64>, right-preconditioned, object-safe.

use crate::context::ksp_context::{Workspace, GmresSpec};
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
    /// Optional hook called once per restart (after backsolve) so caller can adapt the PC.
    pub on_restart: Option<Box<dyn FnMut(usize, f64) -> Result<(), KError> + Send + Sync>>,
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
            on_restart: None,
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
        w.acquire_gmres(GmresSpec {
            n,
            m,
            need_z: true,
            block_s: 0,
        });
    }

    // legacy helpers for in-place Givens rotations removed; Workspace now handles
    // orthogonalization and updating of the Hessenberg system.

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

        let ws = work.ok_or_else(|| {
            KError::InvalidInput("FGMRES requires caller-provided Workspace".into())
        })?;
        self.ensure_workspace(ws, n, block_m);

        a.matvec(x, &mut ws.tmp1);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }
        let mut beta0 = Self::nrm2(&ws.tmp1, comm);
        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let thr = self.atol.max(self.rtol * bnorm);

        if beta0 > 0.0 {
            for i in 0..n {
                ws.tmp2[i] = ws.tmp1[i] / beta0;
            }
            ws.copy_tmp2_into_vcol(0);
        } else {
            ws.v_col(0).fill(0.0);
        }

        ws.h_mem.fill(0.0);
        ws.cs.fill(0.0);
        ws.sn.fill(0.0);
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
                    let (vj, zj) = ws.v_and_z_mut(j);
                    if let Some(pc_) = pc.as_deref_mut() {
                        pc_.apply_mut(pc_side, vj, zj)?;
                    } else {
                        zj.copy_from_slice(vj);
                    }
                }
                {
                    let (zj, tmp2) = ws.z_and_tmp2_mut(j);
                    a.matvec(zj, tmp2);
                }

                for i in 0..=j {
                    let hij = {
                        let vi = &ws.v_mem[i * n..(i + 1) * n];
                        let hij = Self::dot(&ws.tmp2, vi, comm);
                        for (w_i, &vi_val) in ws.tmp2.iter_mut().zip(vi) {
                            *w_i -= hij * vi_val;
                        }
                        hij
                    };
                    *ws.h_at_mut(i, j) = hij;
                }

                if matches!(self.orthog, Orthog::Modified) {
                    for i in 0..=j {
                        let corr = {
                            let vi = &ws.v_mem[i * n..(i + 1) * n];
                            let corr = Self::dot(&ws.tmp2, vi, comm);
                            if corr.abs() > 1e-12 {
                                for (w_i, &vi_val) in ws.tmp2.iter_mut().zip(vi) {
                                    *w_i -= corr * vi_val;
                                }
                            }
                            corr
                        };
                        if corr.abs() > 1e-12 {
                            *ws.h_at_mut(i, j) += corr;
                        }
                    }
                }

                let hij1 = Self::nrm2(&ws.tmp2, comm);
                *ws.h_at_mut(j + 1, j) = hij1;

                if hij1 > 0.0 {
                    for i in 0..n {
                        ws.tmp2[i] /= hij1;
                    }
                    ws.copy_tmp2_into_vcol(j + 1);
                } else {
                    ws.v_col(j + 1).fill(0.0);
                }

                ws.apply_prev_givens_to_col(j, j);
                ws.apply_final_givens_and_update_g(j);

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
                    sum -= ws.h_at(i, l) * y[l];
                }
                y[i] = sum / ws.h_at(i, i);
            }

            for i in 0..k {
                let zi = &ws.z_mem[i * n..(i + 1) * n];
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
            ws.h_mem.fill(0.0);
            ws.cs.fill(0.0);
            ws.sn.fill(0.0);
            ws.g.fill(0.0);
            ws.g[0] = beta0;
            if beta0 > 0.0 {
                for i in 0..n {
                    ws.tmp2[i] = ws.tmp1[i] / beta0;
                }
                ws.copy_tmp2_into_vcol(0);
            } else {
                ws.v_col(0).fill(0.0);
            }

            // Allow both legacy hook and the new Preconditioner::on_restart to adjust PC
            if let Some(hook) = self.on_restart.as_mut() {
                hook(total_iters, beta0)?;
            }
            if let Some(pc_) = pc.as_deref_mut() {
                pc_.on_restart(total_iters, beta0)?;
            }
        }

        stats.iterations = total_iters;

        // Compute true residual at exit for reporting
        let true_res = recompute_true_residual_norm(a, b, x, comm, &mut ws.tmp1);
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
        // Sizing is performed in solve_flexible() once n is known.
        let _ = w;
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
