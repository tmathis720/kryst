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

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::{GmresSpec, ReorthPolicy, Workspace};
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::ReductCtx;
#[cfg(feature = "metrics")]
use crate::utils::convergence::SolveMetrics;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use crate::utils::monitor::{log_residuals, ResidualSnapshot};
use smallvec::SmallVec;
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GmresOrthog {
    Mgs,
    Cgs,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AugmentationPolicy {
    None,
    GmresDR { k: usize },
    Lgmres { ell: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GmresVariant {
    Classical,
    Pipelined,
    SStep {
        s: usize,
        reorth: ReorthPolicy,
        max_cond: f64,
    },
}

pub struct GmresSolver {
    pub restart: usize,
    pub conv: Convergence,
    /// Happy breakdown tolerance
    pub haptol: f64,
    /// Orthogonalization flavor (currently not altering algorithmic path)
    pub orthog: GmresOrthog,
    /// Strategy for the second orthogonalization pass
    pub reorth: ReorthPolicy,
    /// Threshold used by the "if-needed" reorthogonalization strategy
    pub reorth_tol: f64,
    /// Whether to treat near-zero residual as a happy breakdown
    pub happy_breakdown: bool,
    pub variant: GmresVariant,
    pub augmentation: AugmentationPolicy,
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
            reorth: ReorthPolicy::IfNeeded,
            reorth_tol: 0.7,
            happy_breakdown: true,
            variant: GmresVariant::Classical,
            augmentation: AugmentationPolicy::None,
        }
    }

    fn ensure_workspace(&self, w: &mut Workspace, n: usize, side: PcSide) {
        let block_s = match self.variant {
            GmresVariant::SStep { s, .. } => s,
            _ => 0,
        };
        let spec = GmresSpec {
            n,
            m: self.restart,
            need_z: matches!(side, PcSide::Right),
            block_s,
        };
        w.acquire_gmres(spec);
        let rmax = self.augmentation_dim();
        if rmax > 0 {
            w.gmres_recycle
                .configure(n, rmax, self.augmentation.clone());
        } else {
            w.gmres_recycle.configure(n, 0, AugmentationPolicy::None);
        }
    }

    pub fn reorth_policy(&self) -> ReorthPolicy {
        self.reorth
    }

    fn augmentation_dim(&self) -> usize {
        match self.augmentation {
            AugmentationPolicy::None => 0,
            AugmentationPolicy::GmresDR { k } => k.min(self.restart),
            AugmentationPolicy::Lgmres { ell } => ell.min(self.restart),
        }
    }

    fn backsolve(h: &[S], g: &[S], k: usize, ld: usize) -> Vec<S> {
        let mut y = vec![S::zero(); k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for l in (i + 1)..k {
                sum -= h[l * ld + i] * y[l];
            }
            y[i] = sum / h[i * ld + i];
        }
        y
    }

    fn axpy_update_vcols(x: &mut [S], ws: &Workspace, k: usize, y: &[S]) {
        let n = ws.n();
        for j in 0..k {
            let yj = y[j];
            let v = &ws.v_mem[j * n..(j + 1) * n];
            for (xi, &vj) in x.iter_mut().zip(v) {
                *xi += yj * vj;
            }
        }
    }
    fn axpy_update_zcols(x: &mut [S], ws: &Workspace, k: usize, y: &[S]) {
        let n = ws.n();
        for j in 0..k {
            let yj = y[j];
            let z = &ws.z_mem[j * n..(j + 1) * n];
            for (xi, &zj) in x.iter_mut().zip(z) {
                *xi += yj * zj;
            }
        }
    }

    fn true_residual_norm<A: KLinOp<Scalar = S> + ?Sized>(
        a: &A,
        b: &[S],
        x: &[S],
        red: &dyn crate::parallel::ReductionEngine,
        tmp: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> R {
        a.matvec_s(x, tmp, scratch);
        for i in 0..tmp.len() {
            tmp[i] = b[i] - tmp[i];
        }
        red.norm2_s(tmp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve<A, P>(
        &mut self,
        a: &A,
        pc: Option<&P>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
        P: KPreconditioner<Scalar = S> + ?Sized,
    {
        if matches!(self.variant, GmresVariant::SStep { .. }) {
            return self.solve_sstep(a, pc, b, x, pc_side, comm, monitors, work);
        }
        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "GMRES: dimension mismatch or non-square operator".into(),
            ));
        }

        let pc_apply_side = pc_side;
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
        let red = ReductCtx::new(comm, Some(&*ws));

        let mons = monitors.unwrap_or(&[]);
        #[cfg(feature = "metrics")]
        let mut metrics = SolveMetrics::default();

        #[cfg(feature = "metrics")]
        let matvec_start = std::time::Instant::now();
        a.matvec_s(x, &mut ws.tmp1, &mut ws.bridge);
        #[cfg(feature = "metrics")]
        {
            metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
        }
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }

        ws.h_mem.fill(S::zero());
        ws.cs.fill(R::default());
        ws.sn.fill(S::zero());
        ws.g.fill(S::zero());

        let mut reduction_count = 0usize;
        let mut res: R;
        let (beta, mut bnorm) = match pc_side {
            PcSide::Left => {
                if let Some(pc) = pc {
                    let tmp2 = &mut ws.tmp2[..n];
                    #[cfg(feature = "metrics")]
                    let pc_start = std::time::Instant::now();
                    pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                    }
                } else {
                    ws.tmp2[..n].copy_from_slice(&ws.tmp1[..n]);
                }
                let mut norms = [R::zero(); 2];
                red.norm2_many_into(&[&ws.tmp2[..n], b], &mut norms);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += 2 * std::mem::size_of::<R>();
                }
                let beta = norms[0];
                if beta > R::default() {
                    let inv = S::from_real(1.0 / beta);
                    for val in &mut ws.tmp2[..n] {
                        *val *= inv;
                    }
                    ws.copy_tmp2_into_vcol(0);
                } else {
                    ws.v_col(0).fill(S::zero());
                }
                (beta, norms[1])
            }
            PcSide::Right => {
                let mut norms = [R::zero(); 2];
                red.norm2_many_into(&[&ws.tmp1[..n], b], &mut norms);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += 2 * std::mem::size_of::<R>();
                }
                let beta = norms[0];
                if beta > R::default() {
                    let inv = S::from_real(1.0 / beta);
                    for (dst, &src) in ws.tmp2[..n].iter_mut().zip(&ws.tmp1[..n]) {
                        *dst = src * inv;
                    }
                    ws.copy_tmp2_into_vcol(0);
                } else {
                    ws.v_col(0).fill(S::zero());
                }
                (beta, norms[1])
            }
        };

        ws.g[0] = S::from_real(beta);
        bnorm = bnorm.max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);

        let mut total_iters = 0usize;
        res = beta;
        let mut stats = SolveStats::new(0, res, ConvergedReason::Continued);
        let start_reduct = crate::utils::reduction::test_hooks::wait_counters();

        for m in mons {
            m(0, res);
        }
        let true_res =
            Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
        reduction_count += 1;
        #[cfg(feature = "metrics")]
        {
            metrics.bytes_reduced += std::mem::size_of::<R>();
        }
        let precond_res = if let Some(pc) = pc {
            let tmp2 = &mut ws.tmp2[..n];
            #[cfg(feature = "metrics")]
            let pc_start = std::time::Instant::now();
            pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
            #[cfg(feature = "metrics")]
            {
                metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
            }
            let norm = red.norm2(&ws.tmp2[..n]);
            reduction_count += 1;
            #[cfg(feature = "metrics")]
            {
                metrics.bytes_reduced += std::mem::size_of::<R>();
            }
            norm
        } else {
            let norm = red.norm2(&ws.tmp1[..n]);
            reduction_count += 1;
            #[cfg(feature = "metrics")]
            {
                metrics.bytes_reduced += std::mem::size_of::<R>();
            }
            norm
        };
        log_residuals(
            0,
            "GMRES",
            ResidualSnapshot {
                true_residual: true_res,
                preconditioned_residual: precond_res,
                recurrence_residual: Some(res),
            },
        );
        if res <= thr {
            stats.reason = if res <= self.conv.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            stats.final_residual = res;
            let true_res =
                Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
            stats.final_residual = true_res;
            let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
            let async_reductions = end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1;
            let reductions = reduction_count + async_reductions;
            let counters = crate::utils::convergence::SolverCounters {
                num_global_reductions: reductions,
                residual_replacements: 0,
            };
            let mut stats = stats.with_counters(counters);
            #[cfg(feature = "metrics")]
            {
                metrics.reductions = reductions;
                stats.metrics = metrics;
            }
            return Ok(stats);
        }

        'outer: loop {
            let mut k_steps = 0usize;
            for k in 0..self.restart {
                match self.variant {
                    GmresVariant::Classical => match pc_side {
                        PcSide::Left => {
                            let vk = &ws.v_mem[k * n..(k + 1) * n];
                            #[cfg(feature = "metrics")]
                            let matvec_start = std::time::Instant::now();
                            a.matvec_s(vk, &mut ws.tmp1, &mut ws.bridge);
                            #[cfg(feature = "metrics")]
                            {
                                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
                            }
                            if let Some(pc) = pc {
                                let tmp2 = &mut ws.tmp2[..n];
                                #[cfg(feature = "metrics")]
                                let pc_start = std::time::Instant::now();
                                pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                                }
                            } else {
                                ws.tmp2[..n].copy_from_slice(&ws.tmp1[..n]);
                            }
                            let mut hvals: SmallVec<[S; 32]> = SmallVec::with_capacity(k + 1);
                            hvals.resize(k + 1, S::zero());
                            {
                                let tmp2_slice: &[S] = &ws.tmp2[..n];
                                let mut pairs: SmallVec<[(&[S], &[S]); 32]> =
                                    SmallVec::with_capacity(k + 1);
                                for i in 0..=k {
                                    let vi = &ws.v_mem[i * n..(i + 1) * n];
                                    pairs.push((vi, tmp2_slice));
                                }
                                red.dot_many_into(pairs.as_slice(), hvals.as_mut_slice());
                                reduction_count += 1;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.bytes_reduced += (k + 1) * std::mem::size_of::<R>();
                                }
                            }
                            {
                                let tmp2 = &mut ws.tmp2[..n];
                                for (i, hij) in hvals.iter().copied().enumerate() {
                                    let vi = &ws.v_mem[i * n..(i + 1) * n];
                                    for (w, &vi_j) in tmp2.iter_mut().zip(vi) {
                                        *w -= hij * vi_j;
                                    }
                                }
                            }
                            for i in 0..=k {
                                *ws.h_at_mut(i, k) = hvals[i];
                            }
                            let hnext = red.norm2(&ws.tmp2[..n]);
                            reduction_count += 1;
                            #[cfg(feature = "metrics")]
                            {
                                metrics.bytes_reduced += std::mem::size_of::<R>();
                            }
                            *ws.h_at_mut(k + 1, k) = S::from_real(hnext);
                            if hnext > R::default() {
                                let inv = S::from_real(1.0 / hnext);
                                for val in &mut ws.tmp2[..n] {
                                    *val *= inv;
                                }
                                ws.copy_tmp2_into_vcol(k + 1);
                            } else {
                                ws.v_col(k + 1).fill(S::zero());
                            }
                        }
                        PcSide::Right => {
                            let vk = &ws.v_mem[k * n..(k + 1) * n];
                            if let Some(pc) = pc {
                                let tmp2 = &mut ws.tmp2[..n];
                                #[cfg(feature = "metrics")]
                                let pc_start = std::time::Instant::now();
                                pc.apply_s(pc_apply_side, vk, tmp2, &mut ws.bridge)?;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                                }
                            } else {
                                ws.tmp2[..n].copy_from_slice(vk);
                            }
                            {
                                let (tmp2, zk) = ws.tmp2_and_z_mut(k);
                                zk.copy_from_slice(tmp2);
                            }
                            #[cfg(feature = "metrics")]
                            let matvec_start = std::time::Instant::now();
                            a.matvec_s(&ws.tmp2[..n], &mut ws.tmp1, &mut ws.bridge);
                            #[cfg(feature = "metrics")]
                            {
                                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
                            }
                            let mut hvals: SmallVec<[S; 32]> = SmallVec::with_capacity(k + 1);
                            hvals.resize(k + 1, S::zero());
                            {
                                let tmp1_slice: &[S] = &ws.tmp1[..n];
                                let mut pairs: SmallVec<[(&[S], &[S]); 32]> =
                                    SmallVec::with_capacity(k + 1);
                                for i in 0..=k {
                                    let vi = &ws.v_mem[i * n..(i + 1) * n];
                                    pairs.push((vi, tmp1_slice));
                                }
                                red.dot_many_into(pairs.as_slice(), hvals.as_mut_slice());
                                reduction_count += 1;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.bytes_reduced += (k + 1) * std::mem::size_of::<R>();
                                }
                            }
                            {
                                let tmp1 = &mut ws.tmp1[..n];
                                for (i, hij) in hvals.iter().copied().enumerate() {
                                    let vi = &ws.v_mem[i * n..(i + 1) * n];
                                    for (w, &vi_j) in tmp1.iter_mut().zip(vi) {
                                        *w -= hij * vi_j;
                                    }
                                }
                            }
                            for i in 0..=k {
                                *ws.h_at_mut(i, k) = hvals[i];
                            }
                            let hnext = red.norm2(&ws.tmp1[..n]);
                            reduction_count += 1;
                            #[cfg(feature = "metrics")]
                            {
                                metrics.bytes_reduced += std::mem::size_of::<R>();
                            }
                            *ws.h_at_mut(k + 1, k) = S::from_real(hnext);
                            if hnext > R::default() {
                                let inv = S::from_real(1.0 / hnext);
                                for val in &mut ws.tmp1[..n] {
                                    *val *= inv;
                                }
                                ws.copy_tmp1_into_vcol(k + 1);
                            } else {
                                ws.v_col(k + 1).fill(S::zero());
                            }
                        }                    },
                    GmresVariant::Pipelined => match pc_side {
                        PcSide::Left => {
                            let vk = &ws.v_mem[k * n..(k + 1) * n];
                            #[cfg(feature = "metrics")]
                            let matvec_start = std::time::Instant::now();
                            a.matvec_s(vk, &mut ws.tmp1, &mut ws.bridge);
                            #[cfg(feature = "metrics")]
                            {
                                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
                            }
                            if let Some(pc) = pc {
                                #[cfg(feature = "metrics")]
                                let pc_start = std::time::Instant::now();
                                pc.apply_s(
                                    PcSide::Left,
                                    &ws.tmp1[..n],
                                    ws.pipelined_w.as_mut_slice(),
                                    &mut ws.bridge,
                                )?;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                                }
                            } else {
                                ws.pipelined_w[..n].copy_from_slice(&ws.tmp1[..n]);
                            }
                            let pipe = ws.pipelined_arnoldi_step(
                                k,
                                n,
                                red.engine(),
                                self.reorth,
                                self.reorth_tol,
                            )?;
                            let reductions = match pipe {
                                crate::context::ksp_context::PipeReduct::Sync { reductions } => {
                                    reductions
                                }
                                crate::context::ksp_context::PipeReduct::Async { handle } => {
                                    #[cfg(feature = "metrics")]
                                    let wait_start = std::time::Instant::now();
                                    let glob = handle.wait();
                                    #[cfg(feature = "metrics")]
                                    {
                                        metrics.reduction_wait_nanos +=
                                            wait_start.elapsed().as_nanos() as u64;
                                    }
                                    ws.finish_pipelined_arnoldi(
                                        k,
                                        n,
                                        red.engine(),
                                        self.reorth,
                                        self.reorth_tol,
                                        glob,
                                    )?
                                }
                            };
                            reduction_count += reductions;
                            #[cfg(feature = "metrics")]
                            {
                                let payload_len = ws.pipelined_payload.len();
                                metrics.reductions += reductions;
                                metrics.bytes_reduced +=
                                    payload_len * std::mem::size_of::<R>() * reductions;
                            }
                        }
                        PcSide::Right => {
                            let vk = &ws.v_mem[k * n..(k + 1) * n];
                            if let Some(pc) = pc {
                                #[cfg(feature = "metrics")]
                                let pc_start = std::time::Instant::now();
                                pc.apply_s(pc_apply_side, vk, &mut ws.tmp2[..n], &mut ws.bridge)?;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                                }
                            } else {
                                ws.tmp2[..n].copy_from_slice(vk);
                            }
                            {
                                let (tmp2, zk) = ws.tmp2_and_z_mut(k);
                                zk.copy_from_slice(tmp2);
                            }
                            #[cfg(feature = "metrics")]
                            let matvec_start = std::time::Instant::now();
                            a.matvec_s(
                                &ws.tmp2[..n],
                                ws.pipelined_w.as_mut_slice(),
                                &mut ws.bridge,
                            );
                            #[cfg(feature = "metrics")]
                            {
                                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
                            }
                            let pipe = ws.pipelined_arnoldi_step(
                                k,
                                n,
                                red.engine(),
                                self.reorth,
                                self.reorth_tol,
                            )?;
                            let reductions = match pipe {
                                crate::context::ksp_context::PipeReduct::Sync { reductions } => {
                                    reductions
                                }
                                crate::context::ksp_context::PipeReduct::Async { handle } => {
                                    #[cfg(feature = "metrics")]
                                    let wait_start = std::time::Instant::now();
                                    let glob = handle.wait();
                                    #[cfg(feature = "metrics")]
                                    {
                                        metrics.reduction_wait_nanos +=
                                            wait_start.elapsed().as_nanos() as u64;
                                    }
                                    ws.finish_pipelined_arnoldi(
                                        k,
                                        n,
                                        red.engine(),
                                        self.reorth,
                                        self.reorth_tol,
                                        glob,
                                    )?
                                }
                            };
                            reduction_count += reductions;
                            #[cfg(feature = "metrics")]
                            {
                                let payload_len = ws.pipelined_payload.len();
                                metrics.reductions += reductions;
                                metrics.bytes_reduced +=
                                    payload_len * std::mem::size_of::<R>() * reductions;
                            }
                        }                    },
                    GmresVariant::SStep { .. } => {
                        unreachable!("s-step path should exit before iteration loop")
                    }
                }

                ws.apply_prev_givens_to_col(k, k);
                ws.apply_final_givens_and_update_g(k);

                res = ws.g[k + 1].abs();
                total_iters += 1;
                k_steps = k + 1;

                for m in mons {
                    m(total_iters, res);
                }
                let true_res =
                    Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += std::mem::size_of::<R>();
                }
                let precond_res = if let Some(pc) = pc {
                    let tmp2 = &mut ws.tmp2[..n];
                    #[cfg(feature = "metrics")]
                    let pc_start = std::time::Instant::now();
                    pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                    }
                    let norm = red.norm2(&ws.tmp2[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    norm
                } else {
                    let norm = red.norm2(&ws.tmp1[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    norm
                };
                log_residuals(
                    total_iters,
                    "GMRES",
                    ResidualSnapshot {
                        true_residual: true_res,
                        preconditioned_residual: precond_res,
                        recurrence_residual: Some(res),
                    },
                );

                if res <= thr || total_iters >= self.conv.max_iters {
                    break;
                }
            }

            if k_steps == 0 {
                break;
            }

            let y = Self::backsolve(&ws.h_mem, &ws.g, k_steps, ws.ld_h());
            match pc_side {
                PcSide::Left => Self::axpy_update_vcols(x, ws, k_steps, &y),
                PcSide::Right => Self::axpy_update_zcols(x, ws, k_steps, &y),            }

            #[cfg(feature = "metrics")]
            let matvec_start = std::time::Instant::now();
            a.matvec_s(x, &mut ws.tmp1, &mut ws.bridge);
            #[cfg(feature = "metrics")]
            {
                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
            }
            for i in 0..n {
                ws.tmp1[i] = b[i] - ws.tmp1[i];
            }
            res = match pc_side {
                PcSide::Left => {
                    if let Some(pc) = pc {
                        let tmp2 = &mut ws.tmp2[..n];
                        #[cfg(feature = "metrics")]
                        let pc_start = std::time::Instant::now();
                        pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                        #[cfg(feature = "metrics")]
                        {
                            metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                        }
                    } else {
                        ws.tmp2[..n].copy_from_slice(&ws.tmp1[..n]);
                    }
                    let res = red.norm2(&ws.tmp2[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    res
                }
                PcSide::Right => {
                    let res = red.norm2(&ws.tmp1[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    res
                };

            stats.iterations = total_iters;
            stats.final_residual = res;

            if res <= thr || total_iters >= self.conv.max_iters {
                break 'outer;
            }

            ws.h_mem.fill(S::zero());
            ws.cs.fill(R::default());
            ws.sn.fill(S::zero());
            ws.g.fill(S::zero());

            let beta = match pc_side {
                PcSide::Left => {
                    if let Some(pc) = pc {
                        let tmp2 = &mut ws.tmp2[..n];
                        #[cfg(feature = "metrics")]
                        let pc_start = std::time::Instant::now();
                        pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                        #[cfg(feature = "metrics")]
                        {
                            metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                        }
                    } else {
                        ws.tmp2[..n].copy_from_slice(&ws.tmp1[..n]);
                    }
                    let beta = red.norm2(&ws.tmp2[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    if beta > R::default() {
                        let inv = S::from_real(1.0 / beta);
                        for val in &mut ws.tmp2[..n] {
                            *val *= inv;
                        }
                        ws.copy_tmp2_into_vcol(0);
                    } else {
                        ws.v_col(0).fill(S::zero());
                    }
                    beta
                }
                PcSide::Right => {
                    let beta = red.norm2(&ws.tmp1[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    if beta > R::default() {
                        let inv = S::from_real(1.0 / beta);
                        for (dst, &src) in ws.tmp2[..n].iter_mut().zip(&ws.tmp1[..n]) {
                            *dst = src * inv;
                        }
                        ws.copy_tmp2_into_vcol(0);
                    } else {
                        ws.v_col(0).fill(S::zero());
                    }
                    beta
                }
            };

            ws.g[0] = S::from_real(beta);
            res = beta;

            for m in mons {
                m(total_iters, res);
            }
            let true_res =
                Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
            reduction_count += 1;
            #[cfg(feature = "metrics")]
            {
                metrics.bytes_reduced += std::mem::size_of::<R>();
            }
            let precond_res = if let Some(pc) = pc {
                let tmp2 = &mut ws.tmp2[..n];
                #[cfg(feature = "metrics")]
                let pc_start = std::time::Instant::now();
                pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                #[cfg(feature = "metrics")]
                {
                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                }
                let norm = red.norm2(&ws.tmp2[..n]);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += std::mem::size_of::<R>();
                }
                norm
            } else {
                let norm = red.norm2(&ws.tmp1[..n]);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += std::mem::size_of::<R>();
                }
                norm
            };
            log_residuals(
                total_iters,
                "GMRES",
                ResidualSnapshot {
                    true_residual: true_res,
                    preconditioned_residual: precond_res,
                    recurrence_residual: Some(res),
                },
            );
        }

        let (reason, _) = self.conv.check(res, bnorm, total_iters);
        stats.reason = reason;

        let true_res =
            Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
        stats.final_residual = true_res;

        let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
        let async_reductions = end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1;
        let reductions = reduction_count + async_reductions;
        let counters = crate::utils::convergence::SolverCounters {
            num_global_reductions: reductions,
            residual_replacements: 0,
        };
        let mut stats = stats.with_counters(counters);
        #[cfg(feature = "metrics")]
        {
            metrics.reductions = reductions;
            stats.metrics = metrics;
        }
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_f64<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError>
    where
        A: LinOpF64 + crate::matrix::op::LinOp<S = f64> + Send + Sync + ?Sized,
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
            self.solve(&op, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
        }
        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result = self.solve(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work);
            if let Ok(_) = result {
                for (dst, src) in x.iter_mut().zip(x_s.iter()) {
                    *dst = src.real();
                }
            }
            result
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_sstep<A, P>(
        &mut self,
        a: &A,
        pc: Option<&P>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
        P: KPreconditioner<Scalar = S> + ?Sized,
    {
        #[cfg(feature = "complex")]
        {
            return Err(KError::NotImplemented(
                "GMRES s-step is not yet implemented for complex scalars".into(),
            ));
        }

        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "GMRES: dimension mismatch or non-square operator".into(),
            ));
        }

        let (block_s, reorth_policy, max_cond) = match self.variant {
            GmresVariant::SStep {
                s,
                reorth,
                max_cond,
            } => (s.max(1), reorth, max_cond),
            _ => unreachable!("solve_sstep called for non s-step variant"),
        };
        if block_s > 1 {
            // Multi-step blocks are not yet numerically robust; fall back to classical GMRES.
            let prev_variant = self.variant;
            self.variant = GmresVariant::Classical;
            let result = self.solve(a, pc, b, x, pc_side, comm, monitors, work);
            self.variant = prev_variant;
            return result;
        }
        let reorth_tol = R::from(self.reorth_tol).max(R::zero());

        let pc_apply_side = pc_side;
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
        let red = ReductCtx::new(comm, Some(&*ws));

        let mons = monitors.unwrap_or(&[]);
        #[cfg(feature = "metrics")]
        let mut metrics = SolveMetrics::default();

        #[cfg(feature = "metrics")]
        let matvec_start = std::time::Instant::now();
        a.matvec_s(x, &mut ws.tmp1, &mut ws.bridge);
        #[cfg(feature = "metrics")]
        {
            metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
        }
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }

        ws.h_mem.fill(S::zero());
        ws.cs.fill(R::default());
        ws.sn.fill(S::zero());
        ws.g.fill(S::zero());

        let mut reduction_count = 0usize;
        let red_engine = ws
            .reduction_engine()
            .cloned()
            .unwrap_or_else(|| comm.reduction_engine(ws.reduction_options()));

        let mut res: R;
        let (beta, mut bnorm) = match pc_side {
            PcSide::Left => {
                if let Some(pc) = pc {
                    let tmp2 = &mut ws.tmp2[..n];
                    #[cfg(feature = "metrics")]
                    let pc_start = std::time::Instant::now();
                    pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                    }
                } else {
                    ws.tmp2[..n].copy_from_slice(&ws.tmp1[..n]);
                }
                let mut norms = [R::zero(); 2];
                red.norm2_many_into(&[&ws.tmp2[..n], b], &mut norms);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += 2 * std::mem::size_of::<R>();
                }
                let beta = norms[0];
                if beta > R::default() {
                    let inv = S::from_real(1.0 / beta);
                    for val in &mut ws.tmp2[..n] {
                        *val *= inv;
                    }
                    ws.copy_tmp2_into_vcol(0);
                } else {
                    ws.v_col(0).fill(S::zero());
                }
                (beta, norms[1])
            }
            PcSide::Right => {
                let mut norms = [R::zero(); 2];
                red.norm2_many_into(&[&ws.tmp1[..n], b], &mut norms);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += 2 * std::mem::size_of::<R>();
                }
                let beta = norms[0];
                if beta > R::default() {
                    let inv = S::from_real(1.0 / beta);
                    for (dst, &src) in ws.tmp2[..n].iter_mut().zip(&ws.tmp1[..n]) {
                        *dst = src * inv;
                    }
                    ws.copy_tmp2_into_vcol(0);
                } else {
                    ws.v_col(0).fill(S::zero());
                }
                (beta, norms[1])
            }
        };

        if matches!(pc_side, PcSide::Right) {
            let v0: &[S] = &ws.v_mem[0..n];
            let z0: &mut [S] = &mut ws.z_mem[0..n];
            if let Some(pc) = pc {
                #[cfg(feature = "metrics")]
                let pc_start = std::time::Instant::now();
                pc.apply_s(pc_apply_side, v0, z0, &mut ws.bridge)?;
                #[cfg(feature = "metrics")]
                {
                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                }
            } else {
                z0.copy_from_slice(v0);
            }
        }

        ws.g[0] = S::from_real(beta);
        bnorm = bnorm.max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);

        let mut total_iters = 0usize;
        res = beta;
        let mut stats = SolveStats::new(0, res, ConvergedReason::Continued);
        let start_reduct = crate::utils::reduction::test_hooks::wait_counters();

        for m in mons {
            m(0, res);
        }
        let true_res =
            Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
        reduction_count += 1;
        #[cfg(feature = "metrics")]
        {
            metrics.bytes_reduced += std::mem::size_of::<R>();
        }
        let precond_res = if let Some(pc) = pc {
            let tmp2 = &mut ws.tmp2[..n];
            #[cfg(feature = "metrics")]
            let pc_start = std::time::Instant::now();
            pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
            #[cfg(feature = "metrics")]
            {
                metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
            }
            let norm = red.norm2(&ws.tmp2[..n]);
            reduction_count += 1;
            #[cfg(feature = "metrics")]
            {
                metrics.bytes_reduced += std::mem::size_of::<R>();
            }
            norm
        } else {
            let norm = red.norm2(&ws.tmp1[..n]);
            reduction_count += 1;
            #[cfg(feature = "metrics")]
            {
                metrics.bytes_reduced += std::mem::size_of::<R>();
            }
            norm
        };
        log_residuals(
            0,
            "GMRES(s-step)",
            ResidualSnapshot {
                true_residual: true_res,
                preconditioned_residual: precond_res,
                recurrence_residual: Some(res),
            },
        );
        if res <= thr {
            stats.reason = if res <= self.conv.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            stats.final_residual = res;
            let true_res =
                Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
            stats.final_residual = true_res;
            let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
            let async_reductions = end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1;
            let reductions = reduction_count + async_reductions;
            let counters = crate::utils::convergence::SolverCounters {
                num_global_reductions: reductions,
                residual_replacements: 0,
            };
            let mut stats = stats.with_counters(counters);
            #[cfg(feature = "metrics")]
            {
                metrics.reductions = reductions;
                stats.metrics = metrics;
            }
            return Ok(stats);
        }

        // Local block workspace buffer (column-major n x block).
        // This avoids borrowing ws.sstep_mut() for the entire solve and fixes all borrow conflicts.
        let mut w_block: Vec<S> = Vec::new();

        'outer: loop {
            let mut k_steps = 0usize;
            let mut k = 0usize;
            while k < self.restart {
                let block = block_s.min(self.restart - k);
                if block == 0 {
                    break;
                }
                // Resize W (n x block), column-major.
                w_block.resize(n * block, S::zero());

                // Build W columns: w_j = (M^{-1} A)^{j+1} v_k   (Left)
                //              or  w_j = A (M^{-1} w_{j-1})     (Right)
                for j in 0..block {
                    // src is either v_k (j==0) or previous W column (j>0).
                    let (prev_cols, cur_and_rest) = w_block.split_at_mut(j * n);
                    let src: &[S] = if j == 0 {
                        &ws.v_mem[k * n..(k + 1) * n]
                    } else {
                        &prev_cols[(j - 1) * n..j * n]
                    };
                    let wj: &mut [S] = &mut cur_and_rest[0..n];

                    match pc_side {
                        PcSide::Left => {
                            #[cfg(feature = "metrics")]
                            let matvec_start = std::time::Instant::now();
                            a.matvec_s(src, &mut ws.tmp1, &mut ws.bridge);
                            #[cfg(feature = "metrics")]
                            {
                                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
                            }

                            if let Some(pc) = pc {
                                let tmp2 = &mut ws.tmp2[..n];
                                #[cfg(feature = "metrics")]
                                let pc_start = std::time::Instant::now();
                                pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                                }
                                wj.copy_from_slice(tmp2);
                            } else {
                                wj.copy_from_slice(&ws.tmp1[..n]);
                            }
                        }
                        PcSide::Right => {
                            if let Some(pc) = pc {
                                let tmp2 = &mut ws.tmp2[..n];
                                #[cfg(feature = "metrics")]
                                let pc_start = std::time::Instant::now();
                                pc.apply_s(pc_apply_side, src, tmp2, &mut ws.bridge)?;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                                }
                            } else {
                                ws.tmp2[..n].copy_from_slice(src);
                            }

                            #[cfg(feature = "metrics")]
                            let matvec_start = std::time::Instant::now();
                            a.matvec_s(&ws.tmp2[..n], &mut ws.tmp1, &mut ws.bridge);
                            #[cfg(feature = "metrics")]
                            {
                                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
                            }

                            wj.copy_from_slice(&ws.tmp1[..n]);
                        }
                    }
                }

                let mut pre_norms = Vec::new();
                if matches!(reorth_policy, ReorthPolicy::IfNeeded) {
                    pre_norms.resize(block, R::zero());
                    let mut cols: Vec<&[S]> = Vec::with_capacity(block);
                    for j in 0..block {
                        cols.push(&w_block[j * n..(j + 1) * n]);
                    }
                    red.norm2_many_into(&cols, &mut pre_norms);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += block * std::mem::size_of::<R>();
                    }
                }

                let mut cvals: Vec<S> = vec![S::zero(); (k + 1) * block];
                {
                    let mut pairs: SmallVec<[(&[S], &[S]); 64]> =
                        SmallVec::with_capacity((k + 1) * block);
                    for i in 0..=k {
                        let vi = &ws.v_mem[i * n..(i + 1) * n];
                        for j in 0..block {
                            let wj = &w_block[j * n..(j + 1) * n];
                            pairs.push((vi, wj));
                        }
                    }
                    red.dot_many_into(pairs.as_slice(), cvals.as_mut_slice());
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += (k + 1) * block * std::mem::size_of::<R>();
                    }
                } // pairs dropped here; we can safely mutate w_block next.
                // W <- W - V C
                for i in 0..=k {
                    let vi = &ws.v_mem[i * n..(i + 1) * n];
                    for j in 0..block {
                        let coeff = cvals[i * block + j];
                        let wj = &mut w_block[j * n..(j + 1) * n];
                        for (w, &vi_j) in wj.iter_mut().zip(vi) {
                            *w -= coeff * vi_j;
                        }
                    }
                }

                macro_rules! compute_gram {
                    ($w_block:expr) => {{
                        let mut gram: Vec<S> = vec![S::zero(); block * block];
                        let mut pairs: SmallVec<[(&[S], &[S]); 64]> =
                            SmallVec::with_capacity(block * block);
                        for i in 0..block {
                            let wi = &$w_block[i * n..(i + 1) * n];
                            for j in 0..block {
                                let wj = &$w_block[j * n..(j + 1) * n];
                                pairs.push((wi, wj));
                            }
                        }
                        red.dot_many_into(pairs.as_slice(), gram.as_mut_slice());
                        reduction_count += 1;
                        #[cfg(feature = "metrics")]
                        {
                            metrics.bytes_reduced += block * block * std::mem::size_of::<R>();
                        }
                        gram
                    };
                }

                let mut gram = match reorth_policy {
                    ReorthPolicy::Always => {
                        let mut c2: Vec<S> = vec![S::zero(); (k + 1) * block];
                        {
                            let mut pairs: SmallVec<[(&[S], &[S]); 64]> =
                                SmallVec::with_capacity((k + 1) * block);
                            for i in 0..=k {
                                let vi = &ws.v_mem[i * n..(i + 1) * n];
                                for j in 0..block {
                                    let wj = &w_block[j * n..(j + 1) * n];
                                    pairs.push((vi, wj));
                                }
                            }
                            red.dot_many_into(pairs.as_slice(), c2.as_mut_slice());
                            reduction_count += 1;
                            #[cfg(feature = "metrics")]
                            {
                                metrics.bytes_reduced +=
                                    (k + 1) * block * std::mem::size_of::<R>();
                            }
                        }
                        for i in 0..=k {
                            let vi = &ws.v_mem[i * n..(i + 1) * n];
                            for j in 0..block {
                                let coeff = c2[i * block + j];
                                cvals[i * block + j] += coeff;
                                let wj = &mut w_block[j * n..(j + 1) * n];
                                for (w, &vi_j) in wj.iter_mut().zip(vi) {
                                    *w -= coeff * vi_j;
                                }
                            }
                        }
                        compute_gram!(&w_block)
                    }
                    ReorthPolicy::Never => compute_gram!(&w_block),
                    ReorthPolicy::IfNeeded => {
                        let mut gram = compute_gram!(&w_block);
                        let mut trigger_reorth = false;
                        if reorth_tol > R::zero() {
                            for j in 0..block {
                                let pre = pre_norms[j];
                                if pre > R::zero() {
                                    let post_sq = gram[j * block + j].real();
                                    let post_sq = if post_sq > R::zero() {
                                        post_sq
                                    } else {
                                        R::zero()
                                    };
                                    let thresh = reorth_tol * pre;
                                    if post_sq <= thresh * thresh {
                                        trigger_reorth = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if trigger_reorth {
                            let mut c2: Vec<S> = vec![S::zero(); (k + 1) * block];
                            {
                                let mut pairs: SmallVec<[(&[S], &[S]); 64]> =
                                    SmallVec::with_capacity((k + 1) * block);
                                for i in 0..=k {
                                    let vi = &ws.v_mem[i * n..(i + 1) * n];
                                    for j in 0..block {
                                        let wj = &w_block[j * n..(j + 1) * n];
                                        pairs.push((vi, wj));
                                    }
                                }
                                red.dot_many_into(pairs.as_slice(), c2.as_mut_slice());
                                reduction_count += 1;
                                #[cfg(feature = "metrics")]
                                {
                                    metrics.bytes_reduced +=
                                        (k + 1) * block * std::mem::size_of::<R>();
                                }
                            }
                            for i in 0..=k {
                                let vi = &ws.v_mem[i * n..(i + 1) * n];
                                for j in 0..block {
                                    let coeff = c2[i * block + j];
                                    cvals[i * block + j] += coeff;
                                    let wj = &mut w_block[j * n..(j + 1) * n];
                                    for (w, &vi_j) in wj.iter_mut().zip(vi) {
                                        *w -= coeff * vi_j;
                                    }
                                }
                            }
                            gram = compute_gram!(&w_block);
                        }
                        gram
                    }
                };


                let mut r_block = vec![R::default(); block * block];
                for (dst, src) in r_block.iter_mut().zip(gram.iter()) {
                    *dst = src.real();
                }
                Self::chol_upper(&mut r_block, block)?;
                let cond = Self::estimate_triangular_condition(&r_block, block);
                if cond > max_cond {
                    return Err(KError::FactorError(
                        "s-step GMRES: block conditioning exceeds max_cond".into(),
                    ));
                }
                // W <- W * R^{-1}   (solve X R = W in-place for X)
                #[cfg(not(feature = "complex"))]
                {
                    let w_data = w_block.as_mut_slice();
                    let w_real: &mut [R] = unsafe {
                        std::slice::from_raw_parts_mut(w_data.as_mut_ptr() as *mut R, w_data.len())
                    };
                    Self::triangular_solve_right_upper(&r_block, block, w_real, n);
                }

                // Store new basis vectors V_{k+1..k+block} from the orthonormalized block W.
                for j in 0..block {
                    let dst = &mut ws.v_mem[(k + 1 + j) * n..(k + 2 + j) * n];
                    let src = &w_block[j * n..(j + 1) * n];
                    dst.copy_from_slice(src);
                }


                // For Right preconditioning, also build Z columns for these new V columns.
                if matches!(pc_side, PcSide::Right) {
                    for j in 0..block {
                        let vj: &[S] = &ws.v_mem[(k + 1 + j) * n..(k + 2 + j) * n];
                        let zj: &mut [S] = &mut ws.z_mem[(k + 1 + j) * n..(k + 2 + j) * n];
                        if let Some(pc) = pc {
                            #[cfg(feature = "metrics")]
                            let pc_start = std::time::Instant::now();
                            pc.apply_s(pc_apply_side, vj, zj, &mut ws.bridge)?;
                            #[cfg(feature = "metrics")]
                            {
                                metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                            }
                        } else {
                            zj.copy_from_slice(vj);
                        }
                    }
                }

                // Fill the Hessenberg block: upper part is C, lower part is R from Cholesky.
                for i in 0..=k {
                    for j in 0..block {
                        *ws.h_at_mut(i, k + j) = cvals[i * block + j];
                    }
                }
                for i in 0..block {
                    for j in i..block {
                        *ws.h_at_mut(k + 1 + i, k + j) = S::from_real(r_block[i * block + j]);
                    }
                }

                for j in 0..block {
                    let col = k + j;
                    ws.apply_prev_givens_to_col(col, col);
                    ws.apply_final_givens_and_update_g(col);

                    res = ws.g[col + 1].abs();
                    total_iters += 1;
                    k_steps = col + 1;

                    for m in mons {
                        m(total_iters, res);
                    }
                    let true_res =
                        Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    let precond_res = if let Some(pc) = pc {
                        let tmp2 = &mut ws.tmp2[..n];
                        #[cfg(feature = "metrics")]
                        let pc_start = std::time::Instant::now();
                        pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                        #[cfg(feature = "metrics")]
                        {
                            metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                        }
                        let norm = red.norm2(&ws.tmp2[..n]);
                        reduction_count += 1;
                        #[cfg(feature = "metrics")]
                        {
                            metrics.bytes_reduced += std::mem::size_of::<R>();
                        }
                        norm
                    } else {
                        let norm = red.norm2(&ws.tmp1[..n]);
                        reduction_count += 1;
                        #[cfg(feature = "metrics")]
                        {
                            metrics.bytes_reduced += std::mem::size_of::<R>();
                        }
                        norm
                    };
                    log_residuals(
                        total_iters,
                        "GMRES(s-step)",
                        ResidualSnapshot {
                            true_residual: true_res,
                            preconditioned_residual: precond_res,
                            recurrence_residual: Some(res),
                        },
                    );

                    if res <= thr || total_iters >= self.conv.max_iters {
                        break;
                    }
                }
                if res <= thr || total_iters >= self.conv.max_iters {
                    break;
                }

                k += block;
            }

            if k_steps == 0 {
                break;
            }

            let y = Self::backsolve(&ws.h_mem, &ws.g, k_steps, ws.ld_h());
            match pc_side {
                PcSide::Left => Self::axpy_update_vcols(x, ws, k_steps, &y),
                PcSide::Right => Self::axpy_update_zcols(x, ws, k_steps, &y),
            }

            #[cfg(feature = "metrics")]
            let matvec_start = std::time::Instant::now();
            a.matvec_s(x, &mut ws.tmp1, &mut ws.bridge);
            #[cfg(feature = "metrics")]
            {
                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
            }
            for i in 0..n {
                ws.tmp1[i] = b[i] - ws.tmp1[i];
            }
            res = match pc_side {
                PcSide::Left => {
                    if let Some(pc) = pc {
                        let tmp2 = &mut ws.tmp2[..n];
                        #[cfg(feature = "metrics")]
                        let pc_start = std::time::Instant::now();
                        pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                        #[cfg(feature = "metrics")]
                        {
                            metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                        }
                    } else {
                        ws.tmp2[..n].copy_from_slice(&ws.tmp1[..n]);
                    }
                    let res = red.norm2(&ws.tmp2[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    res
                }
                PcSide::Right => {
                    let res = red.norm2(&ws.tmp1[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    res
                };

            stats.iterations = total_iters;
            stats.final_residual = res;

            if res <= thr || total_iters >= self.conv.max_iters {
                break 'outer;
            }

            ws.h_mem.fill(S::zero());
            ws.cs.fill(R::default());
            ws.sn.fill(S::zero());
            ws.g.fill(S::zero());

            let beta = match pc_side {
                PcSide::Left => {
                    if let Some(pc) = pc {
                        let tmp2 = &mut ws.tmp2[..n];
                        #[cfg(feature = "metrics")]
                        let pc_start = std::time::Instant::now();
                        pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                        #[cfg(feature = "metrics")]
                        {
                            metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                        }
                    } else {
                        ws.tmp2[..n].copy_from_slice(&ws.tmp1[..n]);
                    }
                    let beta = red.norm2(&ws.tmp2[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    if beta > R::default() {
                        let inv = S::from_real(1.0 / beta);
                        for val in &mut ws.tmp2[..n] {
                            *val *= inv;
                        }
                        ws.copy_tmp2_into_vcol(0);
                    } else {
                        ws.v_col(0).fill(S::zero());
                    }
                    beta
                }
                PcSide::Right => {
                    let beta = red.norm2(&ws.tmp1[..n]);
                    reduction_count += 1;
                    #[cfg(feature = "metrics")]
                    {
                        metrics.bytes_reduced += std::mem::size_of::<R>();
                    }
                    if beta > R::default() {
                        let inv = S::from_real(1.0 / beta);
                        for (dst, &src) in ws.tmp2[..n].iter_mut().zip(&ws.tmp1[..n]) {
                            *dst = src * inv;
                        }
                        ws.copy_tmp2_into_vcol(0);
                    } else {
                        ws.v_col(0).fill(S::zero());
                    }
                    beta
                };

            ws.g[0] = S::from_real(beta);
            res = beta;

            for m in mons {
                m(total_iters, res);
            }
            let true_res =
                Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
            reduction_count += 1;
            #[cfg(feature = "metrics")]
            {
                metrics.bytes_reduced += std::mem::size_of::<R>();
            }
            let precond_res = if let Some(pc) = pc {
                let tmp2 = &mut ws.tmp2[..n];
                #[cfg(feature = "metrics")]
                let pc_start = std::time::Instant::now();
                pc.apply_s(pc_apply_side, &ws.tmp1[..n], tmp2, &mut ws.bridge)?;
                #[cfg(feature = "metrics")]
                {
                    metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
                }
                let norm = red.norm2(&ws.tmp2[..n]);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += std::mem::size_of::<R>();
                }
                norm
            } else {
                let norm = red.norm2(&ws.tmp1[..n]);
                reduction_count += 1;
                #[cfg(feature = "metrics")]
                {
                    metrics.bytes_reduced += std::mem::size_of::<R>();
                }
                norm
            };
            log_residuals(
                total_iters,
                "GMRES(s-step)",
                ResidualSnapshot {
                    true_residual: true_res,
                    preconditioned_residual: precond_res,
                    recurrence_residual: Some(res),
                },
            );
        }

        let (reason, _) = self.conv.check(res, bnorm, total_iters);
        stats.reason = reason;

        let true_res =
            Self::true_residual_norm(a, b, x, red.engine(), &mut ws.tmp1, &mut ws.bridge);
        stats.final_residual = true_res;

        let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
        let async_reductions = end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1;
        let reductions = reduction_count + async_reductions;
        let counters = crate::utils::convergence::SolverCounters {
            num_global_reductions: reductions,
            residual_replacements: 0,
        };
        let mut stats = stats.with_counters(counters);
        #[cfg(feature = "metrics")]
        {
            metrics.reductions = reductions;
            stats.metrics = metrics;
        }
        Ok(stats)
    }

    #[allow(dead_code)]
    #[inline]
    fn mat_idx(ld: usize, row: usize, col: usize) -> usize {
        col * ld + row
    }

    #[allow(dead_code)]
    fn chol_upper(mat: &mut [R], n: usize) -> Result<(), KError> {
        for j in 0..n {
            for i in 0..=j {
                let mut sum = mat[Self::mat_idx(n, i, j)];
                for k in 0..i {
                    let left = mat[Self::mat_idx(n, k, i)];
                    let right = mat[Self::mat_idx(n, k, j)];
                    sum -= left * right;
                }
                if i == j {
                    if sum <= R::default() || !sum.is_finite() {
                        return Err(KError::FactorError(
                            "s-step GMRES: Cholesky factorization failed".into(),
                        ));
                    }
                    mat[Self::mat_idx(n, i, j)] = sum.sqrt();
                } else {
                    let diag = mat[Self::mat_idx(n, i, i)];
                    if diag.abs() <= R::default() {
                        return Err(KError::FactorError(
                            "s-step GMRES: zero diagonal during Cholesky".into(),
                        ));
                    }
                    mat[Self::mat_idx(n, i, j)] = sum / diag;
                }
            }
            for i in (j + 1)..n {
                mat[Self::mat_idx(n, i, j)] = R::default();
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn triangular_solve_right_upper(r: &[R], block: usize, data: &mut [R], nrows: usize) {
        // Solve X R = data in-place for X, where data holds W' (nrows x block)
        // stored column-major. After the solve, data contains Q.
        for col in 0..block {
            let mut col_slice = vec![R::default(); nrows];
            // copy target column (W' column)
            for row in 0..nrows {
                col_slice[row] = data[col * nrows + row];
            }
            for k in 0..col {
                let r_kj = r[Self::mat_idx(block, k, col)];
                if r_kj.abs() > R::default() {
                    for row in 0..nrows {
                        col_slice[row] -= r_kj * data[k * nrows + row];
                    }
                }
            }
            let diag = r[Self::mat_idx(block, col, col)];
            if diag.abs() > R::default() {
                for row in 0..nrows {
                    data[col * nrows + row] = col_slice[row] / diag;
                }
            } else {
                for row in 0..nrows {
                    data[col * nrows + row] = R::default();
                }
            }
        }
    }

    #[allow(dead_code)]
    fn estimate_triangular_condition(r: &[R], block: usize) -> R {
        let mut max_diag = R::default();
        let mut min_diag = R::MAX;
        for j in 0..block {
            let diag = r[Self::mat_idx(block, j, j)].abs();
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }
        if min_diag <= R::default() {
            R::INFINITY
        } else if max_diag == R::default() {
            R::default()
        } else {
            max_diag / min_diag
        }
    }
}

impl LinearSolver for GmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        // Pre-size GMRES buffers during KSP setup using the workspace dimension.
        // If the side is unknown here, size conservatively based on whether a Z basis
        // was previously requested.
        let n = w.n();
        if n == 0 {
            return;
        }
        let side = if w.has_z() {
            PcSide::Right
        } else {
            PcSide::Left
        };
        self.ensure_workspace(w, n, side);
    }

    #[allow(clippy::too_many_arguments)]
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
        self.solve_f64(a, pc.as_deref(), b, x, pc_side, comm, monitors, work)
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
    pub fn set_variant(&mut self, variant: GmresVariant) {
        self.variant = variant;
    }

    #[cfg(test)]
    pub fn debug_config(&self) -> (usize, GmresOrthog, bool, bool) {
        (
            self.restart,
            self.orthog,
            !matches!(self.reorth, ReorthPolicy::Never),
            self.happy_breakdown,
        )
    }
}
