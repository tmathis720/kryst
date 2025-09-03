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
//! Implementation detail: `Workspace.z_mem` holds the `Z_k` basis for Right and FGMRES
//! in column-major form. The legacy `Workspace.z` (Vec<Vec<_>>) is not used by
//! GMRES/FGMRES.

use crate::context::ksp_context::{GmresSpec, Workspace};
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

    fn ensure_workspace(&self, w: &mut Workspace, n: usize, side: PcSide) {
        let spec = GmresSpec {
            n,
            m: self.restart,
            need_z: matches!(side, PcSide::Right),
            block_s: 0,
        };
        w.acquire_gmres(spec);
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

    fn backsolve(h: &[f64], g: &[f64], k: usize) -> Vec<f64> {
        let ld = g.len();
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for l in (i + 1)..k {
                sum -= h[l * ld + i] * y[l];
            }
            y[i] = sum / h[i * ld + i];
        }
        y
    }

    fn axpy_update_vcols(x: &mut [f64], ws: &Workspace, k: usize, y: &[f64]) {
        let n = ws.n();
        for j in 0..k {
            let yj = y[j];
            let v = &ws.v_mem[j * n..(j + 1) * n];
            for (xi, &vj) in x.iter_mut().zip(v) {
                *xi += yj * vj;
            }
        }
    }
    fn axpy_update_zcols(x: &mut [f64], ws: &Workspace, k: usize, y: &[f64]) {
        let n = ws.n();
        for j in 0..k {
            let yj = y[j];
            let z = &ws.z_mem[j * n..(j + 1) * n];
            for (xi, &zj) in x.iter_mut().zip(z) {
                *xi += yj * zj;
            }
        }
    }
}

impl LinearSolver for GmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        // Defer exact sizing to solve(), once n and side are known.
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
        self.ensure_workspace(ws, n, pc_side);
        // r0 = b - A x
        a.matvec(x, &mut ws.tmp1);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }

        ws.h_mem.fill(0.0);
        ws.cs.fill(0.0);
        ws.sn.fill(0.0);
        ws.g.fill(0.0);

        let beta = match pc_side {
            PcSide::Left => {
                self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                let beta = Self::nrm2(&ws.tmp2);
                if beta > 0.0 {
                    for i in 0..n {
                        ws.tmp2[i] /= beta;
                    }
                    ws.copy_tmp2_into_vcol(0);
                } else {
                    ws.v_col(0).fill(0.0);
                }
                beta
            }
            PcSide::Right => {
                let beta = Self::nrm2(&ws.tmp1);
                if beta > 0.0 {
                    for i in 0..n {
                        ws.tmp2[i] = ws.tmp1[i] / beta;
                    }
                    ws.copy_tmp2_into_vcol(0);
                } else {
                    ws.v_col(0).fill(0.0);
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
                        let vk = &ws.v_mem[k * n..(k + 1) * n];
                        a.matvec(vk, &mut ws.tmp1);
                        self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                        for i in 0..=k {
                            let hij;
                            {
                                let vi = &ws.v_mem[i * n..(i + 1) * n];
                                hij = Self::dot(&ws.tmp2, vi);
                                for (w, &vi_j) in ws.tmp2.iter_mut().zip(vi) {
                                    *w -= hij * vi_j;
                                }
                            }
                            *ws.h_at_mut(i, k) = hij;
                        }
                        let hnext = Self::nrm2(&ws.tmp2);
                        *ws.h_at_mut(k + 1, k) = hnext;
                        if hnext > 0.0 {
                            for i in 0..n {
                                ws.tmp2[i] /= hnext;
                            }
                            ws.copy_tmp2_into_vcol(k + 1);
                        } else {
                            ws.v_col(k + 1).fill(0.0);
                        }
                    }
                    PcSide::Right => {
                        let vk = &ws.v_mem[k * n..(k + 1) * n];
                        self.apply_precond(pc, PcSide::Right, vk, &mut ws.tmp2)?;
                        {
                            let zk = &mut ws.z_mem[k * n..(k + 1) * n];
                            zk.copy_from_slice(&ws.tmp2[..n]);
                        }
                        a.matvec(&ws.tmp2, &mut ws.tmp1);
                        for i in 0..=k {
                            let hij;
                            {
                                let vi = &ws.v_mem[i * n..(i + 1) * n];
                                hij = Self::dot(&ws.tmp1, vi);
                                for (w, &vi_j) in ws.tmp1.iter_mut().zip(vi) {
                                    *w -= hij * vi_j;
                                }
                            }
                            *ws.h_at_mut(i, k) = hij;
                        }
                        let hnext = Self::nrm2(&ws.tmp1);
                        *ws.h_at_mut(k + 1, k) = hnext;
                        if hnext > 0.0 {
                            for i in 0..n {
                                ws.tmp1[i] /= hnext;
                            }
                            ws.copy_tmp1_into_vcol(k + 1);
                        } else {
                            ws.v_col(k + 1).fill(0.0);
                        }
                    }
                    PcSide::Symmetric => unreachable!(),
                }

                ws.apply_prev_givens_to_col(k, k);
                ws.apply_final_givens_and_update_g(k);

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

            let y = Self::backsolve(&ws.h_mem, &ws.g, k_steps);
            match pc_side {
                PcSide::Left => Self::axpy_update_vcols(x, ws, k_steps, &y),
                PcSide::Right => Self::axpy_update_zcols(x, ws, k_steps, &y),
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
            ws.h_mem.fill(0.0);
            ws.cs.fill(0.0);
            ws.sn.fill(0.0);
            ws.g.fill(0.0);

            match pc_side {
                PcSide::Left => {
                    self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                    let beta = Self::nrm2(&ws.tmp2);
                    if beta > 0.0 {
                        for i in 0..n {
                            ws.tmp2[i] /= beta;
                        }
                        ws.copy_tmp2_into_vcol(0);
                    } else {
                        ws.v_col(0).fill(0.0);
                    }
                    ws.g[0] = beta;
                }
                PcSide::Right => {
                    let beta = Self::nrm2(&ws.tmp1);
                    if beta > 0.0 {
                        for i in 0..n {
                            ws.tmp2[i] = ws.tmp1[i] / beta;
                        }
                        ws.copy_tmp2_into_vcol(0);
                    } else {
                        ws.v_col(0).fill(0.0);
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
        (
            self.restart,
            self.orthog,
            self.reorthog,
            self.happy_breakdown,
        )
    }
}
