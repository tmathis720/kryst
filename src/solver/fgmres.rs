//! Flexible GMRES (FGMRES) over the scalar-generic [`KLinOp`] interface with optional
//! mutable preconditioning support.

use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::algebra::scalar::{copy_real_to_scalar_in, copy_scalar_to_real_in};
use crate::context::ksp_context::{GmresSpec, ReorthPolicy, Workspace};
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc_mut};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::recompute_true_residual_norm_s;
use crate::utils::convergence::{ConvergedReason, SolveStats};
use std::any::Any;

/// Orthogonalization flavor
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orthog {
    Classical,
    Modified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FgmresVariant {
    Classical,
    Pipelined,
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
    pub variant: FgmresVariant,
    /// Strategy for the second orthogonalization pass
    pub reorth: ReorthPolicy,
    /// Threshold used by the "if-needed" reorthogonalization strategy
    pub reorth_tol: f64,
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
            variant: FgmresVariant::Classical,
            reorth: ReorthPolicy::IfNeeded,
            reorth_tol: 0.7,
        }
    }

    #[inline]
    fn dot(x: &[S], y: &[S]) -> S {
        dot_conj(x, y)
    }
    #[inline]
    fn nrm2(x: &[S]) -> R {
        nrm2(x)
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

    #[allow(clippy::too_many_arguments)]
    pub fn solve_k<A>(
        &mut self,
        a: &A,
        mut pc: Option<&mut dyn KPreconditioner<Scalar = S>>,
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
        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput(
                "FGMRES requires a square operator".to_string(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "FGMRES: vector size mismatch".to_string(),
            ));
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
        let ws = if let Some(w) = work {
            w
        } else {
            owned_ws = Workspace::new(n);
            &mut owned_ws
        };
        self.ensure_workspace(ws, n, block_m);

        let mons = monitors.unwrap_or(&[]);

        a.matvec_s(x, &mut ws.tmp1[..n], &mut ws.bridge);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }

        let mut beta0 = Self::nrm2(&ws.tmp1[..n]);
        let bnorm = Self::nrm2(b).max(1e-32);
        let thr = self.atol.max(self.rtol * bnorm);

        ws.h_mem.fill(S::zero());
        ws.cs.fill(0.0);
        ws.sn.fill(S::zero());
        ws.g.fill(S::zero());
        ws.g[0] = S::from_real(beta0);

        if beta0 > 0.0 {
            let inv = S::from_real(1.0 / beta0);
            for (dst, &src) in ws.tmp2[..n].iter_mut().zip(&ws.tmp1[..n]) {
                *dst = src * inv;
            }
            ws.copy_tmp2_into_vcol(0);
        } else {
            ws.v_col(0).fill(S::zero());
        }

        let mut total_iters = 0usize;
        let mut res = beta0;
        let mut stats = SolveStats::new(0, res, ConvergedReason::Continued);
        let start_reduct = crate::utils::reduction::test_hooks::wait_counters();

        for m in mons {
            m(0, res);
        }
        if res <= thr {
            stats.final_residual = res;
            stats.reason = if res <= self.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            let true_res =
                recompute_true_residual_norm_s(a, b, x, comm, &mut ws.tmp1[..n], &mut ws.bridge);
            stats.final_residual = true_res;
            let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
            let reductions = end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1;
            let counters = crate::utils::convergence::SolverCounters {
                num_global_reductions: reductions,
                residual_replacements: 0,
            };
            return Ok(stats.with_counters(counters));
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
                match self.variant {
                    FgmresVariant::Classical => {
                        let base = j * n;
                        ws.tmp1[..n].copy_from_slice(&ws.v_mem[base..base + n]);
                        if let Some(pc_ref) = pc.as_deref_mut() {
                            pc_ref.apply_mut_s(
                                pc_side,
                                &ws.tmp1[..n],
                                &mut ws.tmp2[..n],
                                &mut ws.bridge,
                            )?;
                            ws.z_mem[base..base + n].copy_from_slice(&ws.tmp2[..n]);
                        } else {
                            ws.z_mem[base..base + n].copy_from_slice(&ws.tmp1[..n]);
                        }

                        ws.tmp1[..n].copy_from_slice(&ws.z_mem[base..base + n]);
                        a.matvec_s(&ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge);

                        let mut hvals = vec![S::zero(); j + 1];
                        {
                            let tmp2 = &mut ws.tmp2[..n];
                            for i in 0..=j {
                                let vi = &ws.v_mem[i * n..(i + 1) * n];
                                let hij = Self::dot(vi, tmp2);
                                hvals[i] = hij;
                                for (w_i, &vi_val) in tmp2.iter_mut().zip(vi) {
                                    *w_i -= hij * vi_val;
                                }
                            }
                            if matches!(self.orthog, Orthog::Modified) {
                                for i in 0..=j {
                                    let vi = &ws.v_mem[i * n..(i + 1) * n];
                                    let corr = Self::dot(vi, tmp2);
                                    if corr.abs() > 1e-12 {
                                        for (w_i, &vi_val) in tmp2.iter_mut().zip(vi) {
                                            *w_i -= corr * vi_val;
                                        }
                                        hvals[i] += corr;
                                    }
                                }
                            }
                        }
                        for i in 0..=j {
                            *ws.h_at_mut(i, j) = hvals[i];
                        }

                        let hij1 = nrm2(&ws.tmp2[..n]);
                        *ws.h_at_mut(j + 1, j) = S::from_real(hij1);

                        if hij1 > 0.0 {
                            let inv = S::from_real(1.0 / hij1);
                            for val in &mut ws.tmp2[..n] {
                                *val *= inv;
                            }
                            ws.copy_tmp2_into_vcol(j + 1);
                        } else {
                            ws.v_col(j + 1).fill(S::zero());
                        }
                    }
                    FgmresVariant::Pipelined => {
                        #[cfg(feature = "complex")]
                        {
                            return Err(KError::NotImplemented(
                                "Pipelined FGMRES is not yet implemented for complex scalars"
                                    .to_string(),
                            ));
                        }
                        #[cfg(not(feature = "complex"))]
                        {
                            let base = j * n;
                            ws.tmp1[..n].copy_from_slice(&ws.v_mem[base..base + n]);
                            if let Some(pc_ref) = pc.as_deref_mut() {
                                pc_ref.apply_mut_s(
                                    pc_side,
                                    &ws.tmp1[..n],
                                    &mut ws.tmp2[..n],
                                    &mut ws.bridge,
                                )?;
                                ws.z_mem[base..base + n].copy_from_slice(&ws.tmp2[..n]);
                            } else {
                                ws.z_mem[base..base + n].copy_from_slice(&ws.tmp1[..n]);
                            }

                            ws.tmp1[..n].copy_from_slice(&ws.z_mem[base..base + n]);
                            a.matvec_s(&ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge);

                            ws.pipelined_w[..n].copy_from_slice(&ws.tmp2[..n]);
                            ws.pipelined_arnoldi_step(j, n, comm, self.reorth, self.reorth_tol)?;
                        }
                    }
                }

                ws.apply_prev_givens_to_col(j, j);
                ws.apply_final_givens_and_update_g(j);

                res = ws.g[j + 1].abs();
                total_iters += 1;
                arnoldi_steps = j + 1;

                for m in mons {
                    m(total_iters, res);
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
            let mut y = vec![S::zero(); k];
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

            a.matvec_s(x, &mut ws.tmp1[..n], &mut ws.bridge);
            for i in 0..n {
                ws.tmp1[i] = b[i] - ws.tmp1[i];
            }
            beta0 = Self::nrm2(&ws.tmp1[..n]);

            ws.h_mem.fill(S::zero());
            ws.cs.fill(0.0);
            ws.sn.fill(S::zero());
            ws.g.fill(S::zero());
            ws.g[0] = S::from_real(beta0);
            if beta0 > 0.0 {
                let inv = S::from_real(1.0 / beta0);
                for (dst, &src) in ws.tmp2[..n].iter_mut().zip(&ws.tmp1[..n]) {
                    *dst = src * inv;
                }
                ws.copy_tmp2_into_vcol(0);
            } else {
                ws.v_col(0).fill(S::zero());
            }

            if let Some(hook) = self.on_restart.as_mut() {
                hook(total_iters, beta0)?;
            }
            if let Some(pc_ref) = pc.as_deref_mut() {
                pc_ref.on_restart_s(total_iters, beta0)?;
            }
        }

        stats.iterations = total_iters;

        let true_res =
            recompute_true_residual_norm_s(a, b, x, comm, &mut ws.tmp1[..n], &mut ws.bridge);
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
        let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
        let reductions = end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1;
        let counters = crate::utils::convergence::SolverCounters {
            num_global_reductions: reductions,
            residual_replacements: 0,
        };
        Ok(stats.with_counters(counters))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_f64(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let (_, n) = a.dims();
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "FGMRES: vector size mismatch".to_string(),
            ));
        }

        let mut x_s = vec![S::zero(); n];
        copy_real_to_scalar_in(x, &mut x_s);
        let mut b_s = vec![S::zero(); n];
        copy_real_to_scalar_in(b, &mut b_s);

        let op = as_s_op(a);
        let mut pc_storage = pc.map(as_s_pc_mut);
        let stats = self.solve_k(
            &op,
            pc_storage
                .as_mut()
                .map(|w| w as &mut dyn KPreconditioner<Scalar = S>),
            &b_s,
            &mut x_s,
            pc_side,
            comm,
            monitors,
            work,
        )?;

        copy_scalar_to_real_in(&x_s, x);
        Ok(stats)
    }
}

impl LinearSolver for FgmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        // Sizing is performed in solve_f64()/solve_k once n is known.
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
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
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
        self.reorth = if flag {
            ReorthPolicy::Always
        } else {
            ReorthPolicy::Never
        };
    }

    pub fn set_reorth_policy(&mut self, policy: ReorthPolicy) {
        self.reorth = policy;
    }

    pub fn set_reorth_tol(&mut self, tol: f64) {
        self.reorth_tol = tol.max(0.0);
    }
    pub fn set_happy_breakdown(&mut self, flag: bool) {
        self.happy_breakdown = flag;
    }
    pub fn set_variant(&mut self, variant: FgmresVariant) {
        self.variant = variant;
    }

    #[cfg(test)]
    pub fn debug_config(&self) -> (usize, Orthog, bool, bool) {
        (
            self.restart,
            self.orthog,
            !matches!(self.reorth, ReorthPolicy::Never),
            self.happy_breakdown,
        )
    }
}
