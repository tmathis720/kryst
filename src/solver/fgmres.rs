//! Flexible GMRES (FGMRES) over the scalar-generic [`KLinOp`] interface with optional
//! mutable preconditioning support.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::algebra::scalar::{copy_real_to_scalar_in, copy_scalar_to_real_in};
use crate::context::ksp_context::{GmresSpec, GmresWorkspaceLayout, ReorthPolicy, Workspace};
use crate::error::KError;
use crate::matrix::dist_csr::{DistributedPlanDiagnostics, HaloOverlapMode};
use crate::matrix::op::LinOp;
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc_mut};
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::MonitorCallback;
use crate::solver::common::exit_checks::true_residual_converged_reason;
use crate::solver::common::{ReductCtx, call_monitors, recompute_true_residual_norm_s};
#[cfg(feature = "metrics")]
use crate::utils::convergence::SolveMetrics;
use crate::utils::convergence::{ConvergedReason, FgmresCounters, ReductionModel, SolveStats};
use crate::utils::monitor::{
    ResidualSnapshot, log_krylov_stagnation, log_residuals, stagnation_detected,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use smallvec::SmallVec;
use std::any::Any;

/// Orthogonalization method
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrthogMethod {
    /// Classical Gram-Schmidt (CGS) projection.
    ClassicalGS,
    /// Modified Gram-Schmidt (MGS) projection.
    ModifiedGS,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgsRefinement {
    Never,
    IfNeeded,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FgmresVariant {
    Classical,
    Pipelined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelinePolicy {
    Strict,
    FallbackToClassicalOnStagnation,
    PeriodicResidualReplacement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FgmresStagnationPolicy {
    Disabled,
    PipelineFallbackOnly,
    RestartClassicalToo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualCheckPolicy {
    /// Recompute only at startup and restart boundaries (plus safety checks).
    RestartOnly,
    /// Recompute when recurrence indicates a convergence candidate.
    OnConvergence,
    /// Recompute every inner iteration.
    EveryIteration,
    /// Recompute every inner iteration and emit diagnostic residual logs.
    Debug,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModifyPcPolicy {
    Never,
    OnRestart,
    EachIteration,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DistCsrPolicyDecision {
    pub(crate) variant: FgmresVariant,
    pub(crate) pipeline_policy: PipelinePolicy,
    pub(crate) residual_check_policy: ResidualCheckPolicy,
    pub(crate) restart: usize,
    pub(crate) tag: &'static str,
    pub(crate) reason: &'static str,
}

pub type ModifyPcCallback = dyn FnMut(usize, usize, R, Option<R>, &mut dyn KPreconditioner<Scalar = S>) -> Result<(), KError>
    + Send
    + Sync;

pub struct FgmresSolver {
    pub rtol: f64,
    pub atol: f64,
    pub dtol: f64,
    pub maxits: usize,
    pub restart: usize,
    pub orthog: OrthogMethod,
    /// Refinement strategy for the ClassicalGS orthogonalization path.
    pub cgs_refinement: CgsRefinement,
    pub haptol: f64,
    /// Controls workspace provisioning strategy.
    ///
    /// - `false` (default): allocate restart-local basis/QR storage for `restart` columns.
    /// - `true`: preallocate once for `min(restart, maxits)` columns and reuse that capacity
    ///   across cycles to avoid repeated growth checks.
    pub preallocate: bool,
    /// Optional hook called once per restart (after backsolve) so caller can adapt the PC.
    pub on_restart: Option<Box<dyn FnMut(usize, f64) -> Result<(), KError> + Send + Sync>>,
    /// Policy controlling when mutable PC-adaptation callbacks are invoked.
    pub modify_pc_policy: ModifyPcPolicy,
    /// Optional mutable preconditioner adaptation callback.
    pub modify_pc: Option<Box<ModifyPcCallback>>,
    /// Whether to treat near-zero residual as a happy breakdown
    pub happy_breakdown: bool,
    pub variant: FgmresVariant,
    /// Strategy for the second orthogonalization pass
    pub reorth: ReorthPolicy,
    /// Threshold used by the "if-needed" reorthogonalization strategy
    pub reorth_tol: f64,
    /// Controls how often explicit true residual recomputation is performed.
    pub residual_check_policy: ResidualCheckPolicy,
    pub pipeline_policy: PipelinePolicy,
    pub stagnation_policy: FgmresStagnationPolicy,
    pub min_inner_before_fallback: usize,
}

impl FgmresSolver {
    #[cfg(feature = "logging")]
    #[inline]
    fn monitor_residual_semantics_tag(event: &'static str, true_residual: bool) -> &'static str {
        match (event, true_residual) {
            ("initial", true) => "monitor_event=initial monitor_residual=true_norm",
            ("inner", false) => "monitor_event=inner monitor_residual=recurrence_norm",
            ("restart", true) => "monitor_event=restart monitor_residual=true_norm",
            _ => "monitor_event=unknown monitor_residual=unknown",
        }
    }

    pub fn new(rtol: f64, maxits: usize, restart: usize) -> Self {
        Self {
            rtol,
            atol: 1e-12,
            dtol: 1e3,
            maxits,
            restart: restart.max(1),
            orthog: OrthogMethod::ClassicalGS,
            cgs_refinement: CgsRefinement::IfNeeded,
            haptol: 1e-12,
            preallocate: false,
            on_restart: None,
            modify_pc_policy: ModifyPcPolicy::OnRestart,
            modify_pc: None,
            happy_breakdown: true,
            variant: FgmresVariant::Classical,
            reorth: ReorthPolicy::IfNeeded,
            reorth_tol: 0.7,
            residual_check_policy: ResidualCheckPolicy::OnConvergence,
            pipeline_policy: PipelinePolicy::FallbackToClassicalOnStagnation,
            stagnation_policy: FgmresStagnationPolicy::RestartClassicalToo,
            min_inner_before_fallback: 4,
        }
    }

    #[inline]
    fn should_check_true_residual_every_iteration(&self) -> bool {
        matches!(
            self.residual_check_policy,
            ResidualCheckPolicy::EveryIteration | ResidualCheckPolicy::Debug
        )
    }

    #[inline]
    fn should_modify_pc_each_iteration(&self) -> bool {
        matches!(self.modify_pc_policy, ModifyPcPolicy::EachIteration)
    }

    #[inline]
    fn should_modify_pc_on_restart(&self) -> bool {
        matches!(self.modify_pc_policy, ModifyPcPolicy::OnRestart)
    }

    #[inline]
    fn call_modify_pc_callback(
        &mut self,
        global_iter: usize,
        cycle_local_iter: usize,
        recurrence_residual: R,
        true_residual: Option<R>,
        pc: &mut dyn KPreconditioner<Scalar = S>,
    ) -> Result<(), KError> {
        if let Some(cb) = self.modify_pc.as_mut() {
            cb(
                global_iter,
                cycle_local_iter,
                recurrence_residual,
                true_residual,
                pc,
            )?;
        }
        Ok(())
    }

    #[inline]
    fn is_debug_residual_policy(&self) -> bool {
        matches!(self.residual_check_policy, ResidualCheckPolicy::Debug)
    }

    fn reduction_model(&self) -> ReductionModel {
        match self.variant {
            FgmresVariant::Classical => ReductionModel {
                variant: "fgmres-classical",
                startup: 2,
                per_iteration: 2.0,
                tail: 1,
            },
            FgmresVariant::Pipelined => ReductionModel {
                variant: "fgmres-pipelined",
                startup: 2,
                per_iteration: 1.0,
                tail: 1,
            },
        }
    }

    #[inline]
    fn effective_variant_label(&self) -> &'static str {
        match self.variant {
            FgmresVariant::Classical => "classical",
            FgmresVariant::Pipelined => "pipelined",
        }
    }

    #[inline]
    fn effective_residual_check_policy_label(&self) -> &'static str {
        match self.residual_check_policy {
            ResidualCheckPolicy::RestartOnly => "restart-only",
            ResidualCheckPolicy::OnConvergence => "on-convergence",
            ResidualCheckPolicy::EveryIteration => "every-iteration",
            ResidualCheckPolicy::Debug => "debug",
        }
    }

    #[inline]
    fn workspace_cols(&self) -> usize {
        if self.preallocate {
            self.restart.min(self.maxits)
        } else {
            self.restart
        }
    }

    pub(crate) fn stagnation_action(&mut self, inner_iter: usize) -> (&'static str, bool) {
        if inner_iter < self.min_inner_before_fallback {
            return ("stagnation fallback gated by min-inner floor", false);
        }
        let mut should_restart = false;
        let action = match self.stagnation_policy {
            FgmresStagnationPolicy::Disabled => "stagnation policy disabled",
            FgmresStagnationPolicy::PipelineFallbackOnly => {
                match (self.variant, self.pipeline_policy) {
                    (FgmresVariant::Pipelined, PipelinePolicy::FallbackToClassicalOnStagnation) => {
                        self.variant = FgmresVariant::Classical;
                        should_restart = true;
                        "switching to classical restart"
                    }
                    (FgmresVariant::Pipelined, PipelinePolicy::PeriodicResidualReplacement) => {
                        should_restart = true;
                        "periodic residual replacement restart"
                    }
                    (FgmresVariant::Pipelined, PipelinePolicy::Strict) => {
                        "strict pipelined policy: no fallback"
                    }
                    _ => "pipeline-only policy: no classical restart",
                }
            }
            FgmresStagnationPolicy::RestartClassicalToo => {
                match (self.variant, self.pipeline_policy) {
                    (FgmresVariant::Pipelined, PipelinePolicy::FallbackToClassicalOnStagnation) => {
                        self.variant = FgmresVariant::Classical;
                        should_restart = true;
                        "switching to classical restart"
                    }
                    (FgmresVariant::Pipelined, PipelinePolicy::PeriodicResidualReplacement) => {
                        should_restart = true;
                        "periodic residual replacement restart"
                    }
                    (FgmresVariant::Pipelined, PipelinePolicy::Strict) => {
                        "strict pipelined policy: no fallback"
                    }
                    _ => {
                        should_restart = true;
                        "restarting FGMRES"
                    }
                }
            }
        };
        (action, should_restart)
    }

    fn ensure_workspace(&self, w: &mut Workspace, n: usize) {
        w.acquire_gmres(GmresSpec {
            n,
            m: self.workspace_cols(),
            need_z: true,
            block_s: 0,
        });
    }

    /// Estimate bytes required by the FGMRES workspace sections for `(n, restart)`.
    ///
    /// This includes the formally sized GMRES/FGMRES sections (`V`, `Z`, `H`, Givens/QR,
    /// pipeline payload) plus solve-global temporary vectors (`tmp1/tmp2`, pipeline vectors).
    /// It excludes allocator overhead and optional communication arenas.
    pub fn memory_bytes(n: usize, restart: usize) -> usize {
        let m = restart.max(1);
        let layout = GmresWorkspaceLayout::from_spec(GmresSpec {
            n,
            m,
            need_z: true,
            block_s: 0,
        });
        let scalar_items = layout
            .v_len
            .saturating_add(layout.z_len)
            .saturating_add(layout.h_len)
            .saturating_add(m)
            .saturating_add(layout.g_len)
            .saturating_add(layout.givens_len)
            .saturating_add(layout.tmp_len.saturating_mul(2))
            .saturating_add(layout.pipelined_vec_len.saturating_mul(2));
        let real_items = m.saturating_add(layout.pipelined_payload_len);
        scalar_items
            .saturating_mul(std::mem::size_of::<S>())
            .saturating_add(real_items.saturating_mul(std::mem::size_of::<R>()))
    }

    #[inline]
    fn cgs_refinement_mode(&self) -> CgsRefinement {
        self.cgs_refinement
    }

    pub(crate) fn select_distcsr_policy(
        &self,
        diag: &DistributedPlanDiagnostics,
        comm_size: usize,
    ) -> DistCsrPolicyDecision {
        let halo_volume = (diag.halo_recv_volume + diag.halo_send_volume) as f64;
        let comm_pressure = diag.expected_communication_fraction;
        let compute_pressure = diag.expected_computation_fraction;
        let overlap_enabled = diag.overlap_mode == HaloOverlapMode::Interior;
        let communication_heavy =
            comm_size > 1 && (overlap_enabled || comm_pressure >= 0.55 || halo_volume >= 4096.0);

        if communication_heavy {
            let very_heavy = comm_pressure >= 0.70 || halo_volume >= 16_384.0;
            let restart = if very_heavy {
                self.restart.min(24).max(8)
            } else {
                self.restart.min(32).max(8)
            };
            DistCsrPolicyDecision {
                variant: FgmresVariant::Pipelined,
                pipeline_policy: if very_heavy {
                    PipelinePolicy::PeriodicResidualReplacement
                } else {
                    PipelinePolicy::FallbackToClassicalOnStagnation
                },
                residual_check_policy: ResidualCheckPolicy::RestartOnly,
                restart,
                tag: "distcsr_policy=comm_heavy",
                reason: "high halo/communication pressure detected",
            }
        } else {
            DistCsrPolicyDecision {
                variant: FgmresVariant::Classical,
                pipeline_policy: PipelinePolicy::Strict,
                residual_check_policy: if compute_pressure >= 0.70 {
                    ResidualCheckPolicy::OnConvergence
                } else {
                    ResidualCheckPolicy::EveryIteration
                },
                restart: self.restart.max(16).min(self.maxits.max(1)),
                tag: "distcsr_policy=compute_heavy",
                reason: "low halo pressure and compute-dominant local work",
            }
        }
    }

    fn apply_distcsr_policy_hook(&mut self, diag: &DistributedPlanDiagnostics, comm_size: usize) {
        let decision = self.select_distcsr_policy(diag, comm_size);
        self.variant = decision.variant;
        self.pipeline_policy = decision.pipeline_policy;
        self.residual_check_policy = decision.residual_check_policy;
        self.restart = decision.restart.max(1);

        #[cfg(feature = "logging")]
        if log::log_enabled!(log::Level::Info) {
            log::info!(
                "FGMRES DistCSR policy selected: {} reason={} variant={:?} pipeline_policy={:?} residual_check_policy={:?} restart={} comm_pressure={:.3} compute_pressure={:.3} overlap={:?}",
                decision.tag,
                decision.reason,
                decision.variant,
                decision.pipeline_policy,
                decision.residual_check_policy,
                decision.restart,
                diag.expected_communication_fraction,
                diag.expected_computation_fraction,
                diag.overlap_mode,
            );
        }
    }

    pub fn apply_distcsr_policy(&mut self, diag: &DistributedPlanDiagnostics, comm_size: usize) {
        self.apply_distcsr_policy_hook(diag, comm_size);
    }

    fn should_run_cgs_refinement(&self, wnorm0: R, hnext: R) -> bool {
        match self.cgs_refinement_mode() {
            CgsRefinement::Never => false,
            CgsRefinement::Always => true,
            CgsRefinement::IfNeeded => wnorm0 > R::zero() && hnext < self.reorth_tol * wnorm0,
        }
    }

    #[inline]
    const fn rayon_len_threshold() -> usize {
        2048
    }

    #[inline]
    fn residual_update_in_place(&self, residual: &mut [S], rhs: &[S]) {
        #[cfg(feature = "rayon")]
        {
            if residual.len() >= Self::rayon_len_threshold() {
                residual
                    .par_iter_mut()
                    .zip(rhs.par_iter())
                    .for_each(|(ri, &bi)| *ri = bi - *ri);
                return;
            }
        }
        #[cfg(not(feature = "rayon"))]
        let _ = rhs;
        for (ri, &bi) in residual.iter_mut().zip(rhs.iter()) {
            *ri = bi - *ri;
        }
    }

    #[inline]
    fn scaled_copy(&self, dst: &mut [S], src: &[S], alpha: S) {
        #[cfg(feature = "rayon")]
        {
            if dst.len() >= Self::rayon_len_threshold() {
                dst.par_iter_mut()
                    .zip(src.par_iter())
                    .for_each(|(d, &s)| *d = s * alpha);
                return;
            }
        }
        #[cfg(not(feature = "rayon"))]
        let _ = src;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s * alpha;
        }
    }

    #[inline]
    fn scale_in_place(&self, v: &mut [S], alpha: S) {
        #[cfg(feature = "rayon")]
        {
            if v.len() >= Self::rayon_len_threshold() {
                v.par_iter_mut().for_each(|vi| *vi *= alpha);
                return;
            }
        }
        for vi in v.iter_mut() {
            *vi *= alpha;
        }
    }

    #[inline]
    fn axpy_in_place(&self, y: &mut [S], x: &[S], alpha: S) {
        #[cfg(feature = "rayon")]
        {
            if y.len() >= Self::rayon_len_threshold() {
                y.par_iter_mut()
                    .zip(x.par_iter())
                    .for_each(|(yi, &xi)| *yi -= alpha * xi);
                return;
            }
        }
        #[cfg(not(feature = "rayon"))]
        let _ = x;
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi -= alpha * xi;
        }
    }

    fn orthogonalize_cgs(
        &self,
        ws: &mut Workspace,
        red: &ReductCtx,
        n: usize,
        j: usize,
    ) -> (R, bool) {
        let wnorm0 = if matches!(self.cgs_refinement_mode(), CgsRefinement::IfNeeded) {
            red.norm2(&ws.tmp2[..n])
        } else {
            R::zero()
        };
        let mut hvals: SmallVec<[S; 32]> = SmallVec::with_capacity(j + 1);
        hvals.resize(j + 1, S::zero());
        {
            let tmp2_slice: &[S] = &ws.tmp2[..n];
            let mut pairs: SmallVec<[(&[S], &[S]); 32]> = SmallVec::with_capacity(j + 1);
            for i in 0..=j {
                pairs.push((&ws.v_mem[i * n..(i + 1) * n], tmp2_slice));
            }
            red.dot_many_into(pairs.as_slice(), hvals.as_mut_slice());
        }

        for (i, hij) in hvals.iter().copied().enumerate() {
            let vi = &ws.v_mem[i * n..(i + 1) * n];
            self.axpy_in_place(&mut ws.tmp2[..n], vi, hij);
        }

        let mut hnext = red.norm2(&ws.tmp2[..n]);
        let ran_refinement = self.should_run_cgs_refinement(wnorm0, hnext);
        if ran_refinement {
            let mut corr: SmallVec<[S; 32]> = SmallVec::with_capacity(j + 1);
            corr.resize(j + 1, S::zero());
            {
                let tmp2_slice: &[S] = &ws.tmp2[..n];
                let mut pairs: SmallVec<[(&[S], &[S]); 32]> = SmallVec::with_capacity(j + 1);
                for i in 0..=j {
                    pairs.push((&ws.v_mem[i * n..(i + 1) * n], tmp2_slice));
                }
                red.dot_many_into(pairs.as_slice(), corr.as_mut_slice());
            }
            for (i, corr_val) in corr.into_iter().enumerate() {
                let vi = &ws.v_mem[i * n..(i + 1) * n];
                self.axpy_in_place(&mut ws.tmp2[..n], vi, corr_val);
                hvals[i] += corr_val;
            }
            hnext = red.norm2(&ws.tmp2[..n]);
        }

        for i in 0..=j {
            *ws.h_at_mut(i, j) = hvals[i];
        }
        (hnext, ran_refinement)
    }

    fn orthogonalize_mgs(
        &self,
        ws: &mut Workspace,
        red: &ReductCtx,
        n: usize,
        j: usize,
    ) -> (R, bool) {
        for i in 0..=j {
            let hij = red.dot(&ws.v_mem[i * n..(i + 1) * n], &ws.tmp2[..n]);
            *ws.h_at_mut(i, j) = hij;
            let vi = &ws.v_mem[i * n..(i + 1) * n];
            self.axpy_in_place(&mut ws.tmp2[..n], vi, hij);
        }
        (red.norm2(&ws.tmp2[..n]), false)
    }

    // legacy helpers for in-place Givens rotations removed; Workspace now handles
    // orthogonalization and updating of the Hessenberg system.

    #[allow(clippy::too_many_arguments)]
    fn solve_classical_cycle<A>(
        &mut self,
        a: &A,
        pc: &mut Option<&mut dyn KPreconditioner<Scalar = S>>,
        ws: &mut Workspace,
        red: &ReductCtx,
        pc_side: PcSide,
        n: usize,
        j: usize,
        total_iters: usize,
        recurrence_residual: R,
        modify_pc_calls: &mut usize,
        orthogonalization_passes: &mut usize,
        max_orthogonality_loss_estimate: &mut R,
        #[cfg(feature = "metrics")] metrics: &mut SolveMetrics,
    ) -> Result<R, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        let base = j * n;
        ws.tmp1[..n].copy_from_slice(&ws.v_mem[base..base + n]);
        if let Some(pc_ref) = pc.as_deref_mut() {
            if self.should_modify_pc_each_iteration() {
                *modify_pc_calls += 1;
                self.call_modify_pc_callback(total_iters, j, recurrence_residual, None, pc_ref)?;
            }
            #[cfg(feature = "metrics")]
            let pc_start = std::time::Instant::now();
            pc_ref.apply_mut_s(pc_side, &ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge)?;
            #[cfg(feature = "metrics")]
            {
                metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
            }
            ws.z_mem[base..base + n].copy_from_slice(&ws.tmp2[..n]);
        } else {
            ws.z_mem[base..base + n].copy_from_slice(&ws.tmp1[..n]);
        }

        ws.tmp1[..n].copy_from_slice(&ws.z_mem[base..base + n]);
        #[cfg(feature = "metrics")]
        let matvec_start = std::time::Instant::now();
        a.matvec_s(&ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge);
        #[cfg(feature = "metrics")]
        {
            metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
        }

        let wnorm_before = red.norm2(&ws.tmp2[..n]);
        let arnoldi_norm_scale = wnorm_before.max(R::one());
        let (hij1, used_second_pass) = match self.orthog {
            OrthogMethod::ClassicalGS => self.orthogonalize_cgs(ws, red, n, j),
            OrthogMethod::ModifiedGS => self.orthogonalize_mgs(ws, red, n, j),
        };
        *orthogonalization_passes += 1 + usize::from(used_second_pass);
        #[cfg(feature = "metrics")]
        {
            let dot_reductions = match (self.orthog, self.cgs_refinement_mode()) {
                (OrthogMethod::ClassicalGS, CgsRefinement::Never) => j + 2,
                (OrthogMethod::ClassicalGS, CgsRefinement::IfNeeded) => j + 3,
                (OrthogMethod::ClassicalGS, CgsRefinement::Always) => j + 3,
                (OrthogMethod::ModifiedGS, _) => j + 2,
            };
            metrics.bytes_reduced += dot_reductions * std::mem::size_of::<R>();
        }
        *ws.h_at_mut(j + 1, j) = S::from_real(hij1);
        if wnorm_before > R::zero() {
            let ratio = (hij1 / wnorm_before).clamp(R::zero(), R::one());
            let loss_estimate = (R::one() - ratio).max(R::zero());
            if loss_estimate > *max_orthogonality_loss_estimate {
                *max_orthogonality_loss_estimate = loss_estimate;
            }
        }

        if hij1 > R::default() {
            let inv = S::from_real(1.0 / hij1);
            self.scale_in_place(&mut ws.tmp2[..n], inv);
            ws.copy_tmp2_into_vcol(j + 1);
        } else {
            ws.v_col(j + 1).fill(S::zero());
        }
        Ok(arnoldi_norm_scale)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_pipelined_cycle<A>(
        &mut self,
        a: &A,
        pc: &mut Option<&mut dyn KPreconditioner<Scalar = S>>,
        ws: &mut Workspace,
        red: &ReductCtx,
        red_engine: &dyn crate::parallel::ReductionEngine,
        pc_side: PcSide,
        n: usize,
        j: usize,
        total_iters: usize,
        recurrence_residual: R,
        pipeline_reductions: &mut usize,
        async_waits: &mut usize,
        deferred_pipeline_waits: &mut usize,
        immediate_pipeline_completions: &mut usize,
        modify_pc_calls: &mut usize,
        orthogonalization_passes: &mut usize,
        #[cfg(feature = "metrics")] metrics: &mut SolveMetrics,
    ) -> Result<R, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        let base = j * n;
        ws.tmp1[..n].copy_from_slice(&ws.v_mem[base..base + n]);
        if let Some(pc_ref) = pc.as_deref_mut() {
            if self.should_modify_pc_each_iteration() {
                *modify_pc_calls += 1;
                self.call_modify_pc_callback(total_iters, j, recurrence_residual, None, pc_ref)?;
            }
            #[cfg(feature = "metrics")]
            let pc_start = std::time::Instant::now();
            pc_ref.apply_mut_s(pc_side, &ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge)?;
            #[cfg(feature = "metrics")]
            {
                metrics.pc_apply_nanos += pc_start.elapsed().as_nanos() as u64;
            }
            ws.z_mem[base..base + n].copy_from_slice(&ws.tmp2[..n]);
        } else {
            ws.z_mem[base..base + n].copy_from_slice(&ws.tmp1[..n]);
        }

        ws.tmp1[..n].copy_from_slice(&ws.z_mem[base..base + n]);
        #[cfg(feature = "metrics")]
        let matvec_start = std::time::Instant::now();
        a.matvec_s(&ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge);
        #[cfg(feature = "metrics")]
        {
            metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
        }

        ws.pipelined_w[..n].copy_from_slice(&ws.tmp2[..n]);
        let arnoldi_norm_scale = red.norm2(&ws.pipelined_w[..n]).max(R::one());
        #[cfg(feature = "metrics")]
        let reduction_launch_start = std::time::Instant::now();
        let pipe = ws.launch_pipelined_arnoldi_reduction(j, n, red_engine)?;
        *orthogonalization_passes += match self.reorth {
            ReorthPolicy::Always => 2,
            _ => 1,
        };
        #[cfg(feature = "metrics")]
        let payload_len = Workspace::pipelined_payload_len_for_k(j);
        #[cfg(feature = "metrics")]
        let reduction_launched_elapsed = reduction_launch_start.elapsed();
        *deferred_pipeline_waits += 1;
        let reductions = match pipe {
            crate::context::ksp_context::PipeReduct::Sync { reductions } => reductions,
            crate::context::ksp_context::PipeReduct::Async { handle } => {
                if handle.is_ready() {
                    *immediate_pipeline_completions += 1;
                } else {
                    *async_waits += 1;
                }
                #[cfg(feature = "metrics")]
                {
                    metrics.reduction_overlap_nanos += reduction_launched_elapsed.as_nanos() as u64;
                    let wait_start = std::time::Instant::now();
                    let glob = handle.wait();
                    metrics.reduction_wait_nanos += wait_start.elapsed().as_nanos() as u64;
                    ws.finish_pipelined_arnoldi(
                        j,
                        n,
                        red_engine,
                        self.reorth,
                        self.reorth_tol,
                        glob,
                    )?
                }
                #[cfg(not(feature = "metrics"))]
                {
                    ws.finalize_pipelined_arnoldi(
                        crate::context::ksp_context::PipeReduct::Async { handle },
                        j,
                        n,
                        red_engine,
                        self.reorth,
                        self.reorth_tol,
                    )?
                }
            }
        };
        *pipeline_reductions += reductions;
        #[cfg(feature = "metrics")]
        {
            metrics.bytes_reduced += payload_len * std::mem::size_of::<R>() * reductions;
        }
        Ok(arnoldi_norm_scale)
    }

    #[inline]
    fn update_hessenberg_qr_and_residual(&self, ws: &mut Workspace, j: usize) -> R {
        ws.apply_prev_givens_to_col(j, j);
        ws.apply_final_givens_and_update_g(j);
        ws.g[j + 1].abs()
    }

    fn backsolve_least_squares(&self, ws: &Workspace, k: usize) -> Result<Vec<S>, KError> {
        let mut y = vec![S::zero(); k];
        for i in (0..k).rev() {
            let mut sum = ws.g[i];
            for l in (i + 1)..k {
                sum -= ws.h_at(i, l) * y[l];
            }
            let diag = ws.h_at(i, i);
            if !diag.real().is_finite() || diag.abs() <= self.haptol {
                return Err(KError::SolveError(
                    "FGMRES reduced system singular".to_string(),
                ));
            }
            y[i] = sum / diag;
        }
        Ok(y)
    }

    #[inline]
    fn apply_solution_correction(&self, ws: &Workspace, n: usize, y: &[S], x: &mut [S]) {
        #[cfg(feature = "rayon")]
        {
            if n >= Self::rayon_len_threshold() {
                x.par_iter_mut().enumerate().for_each(|(col, xj)| {
                    let mut accum = *xj;
                    for (i, yi) in y.iter().enumerate() {
                        accum += *yi * ws.z_mem[i * n + col];
                    }
                    *xj = accum;
                });
                return;
            }
        }
        for (i, yi) in y.iter().enumerate() {
            let zi = &ws.z_mem[i * n..(i + 1) * n];
            for (xj, &zij) in x.iter_mut().zip(zi) {
                *xj += *yi * zij;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_k<A>(
        &mut self,
        a: &A,
        pc: Option<&mut dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        self.solve_k_with_dist_policy(a, pc, b, x, pc_side, comm, monitors, work, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_k_with_dist_policy<A>(
        &mut self,
        a: &A,
        mut pc: Option<&mut dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
        dist_plan_diag: Option<&DistributedPlanDiagnostics>,
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

        if pc_side != PcSide::Right {
            return Err(KError::InvalidInput(format!(
                "FGMRES supports only right preconditioning; got {pc_side:?}"
            )));
        }

        if let Some(diag) = dist_plan_diag {
            self.apply_distcsr_policy_hook(diag, comm.size());
        }
        let workspace_cols = self.workspace_cols();

        let mut owned_ws;
        let ws = if let Some(w) = work {
            w
        } else {
            owned_ws = Workspace::new(n);
            &mut owned_ws
        };
        self.ensure_workspace(ws, n);
        let red = ReductCtx::new(comm, Some(&*ws));

        let mons = monitors.unwrap_or(&[]);
        #[cfg(feature = "metrics")]
        let mut metrics = SolveMetrics::default();

        #[cfg(feature = "metrics")]
        let matvec_start = std::time::Instant::now();
        a.matvec_s(x, &mut ws.tmp1[..n], &mut ws.bridge);
        #[cfg(feature = "metrics")]
        {
            metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
        }
        self.residual_update_in_place(&mut ws.tmp1[..n], b);

        let mut norms = [R::zero(); 2];
        red.norm2_many_into(&[&ws.tmp1[..n], b], &mut norms);
        #[cfg(feature = "metrics")]
        {
            metrics.bytes_reduced += 2 * std::mem::size_of::<R>();
        }
        let mut beta0 = norms[0];
        let bnorm = norms[1].max(1e-32);
        let thr = self.atol.max(self.rtol * bnorm);

        ws.clear_gmres_restart_state();
        ws.g[0] = S::from_real(beta0);

        if beta0 > R::default() {
            let inv = S::from_real(1.0 / beta0);
            self.scaled_copy(&mut ws.tmp2[..n], &ws.tmp1[..n], inv);
            ws.copy_tmp2_into_vcol(0);
        } else {
            ws.v_col(0).fill(S::zero());
        }

        let mut total_iters = 0usize;
        let mut res = beta0;
        let mut stats = SolveStats::new(0, res, ConvergedReason::Continued)
            .with_effective_runtime_policy(
                self.effective_variant_label(),
                self.restart,
                self.effective_residual_check_policy_label(),
            );
        stats.final_recurrence_residual = Some(res);
        let red_engine = ws
            .reduction_engine()
            .cloned()
            .unwrap_or_else(|| comm.reduction_engine(ws.reduction_options()));
        let mut pipeline_reductions = 0usize;
        let mut async_waits = 0usize;
        let mut deferred_pipeline_waits = 0usize;
        let mut immediate_pipeline_completions = 0usize;
        let mut orthogonalization_passes = 0usize;
        let mut orthogonalization_rank_loss = false;
        let mut max_orthogonality_loss_estimate = R::zero();
        let mut restart_count = 0usize;
        let mut inner_iterations_last_cycle = 0usize;
        let mut happy_breakdowns = 0usize;
        let mut explicit_residual_checks = 0usize;
        let mut pipeline_fallbacks = 0usize;
        let mut modify_pc_calls = 0usize;
        let start_reduct = crate::utils::reduction::test_hooks::wait_counters();

        #[cfg(feature = "logging")]
        if log::log_enabled!(log::Level::Info) {
            log::info!(
                "FGMRES monitor semantics: {}",
                Self::monitor_residual_semantics_tag("initial", true)
            );
        }

        // Iteration 0 monitor payload is the true residual norm ||b - A x0||_2.
        if call_monitors(mons, 0, res, pipeline_reductions) {
            explicit_residual_checks += 1;
            let true_res = recompute_true_residual_norm_s(
                a,
                b,
                x,
                comm,
                red.engine(),
                &mut ws.tmp1[..n],
                &mut ws.bridge,
            );
            let last_preconditioned_residual = if let Some(pc_ref) = pc.as_deref_mut() {
                pc_ref.apply_mut_s(pc_side, &ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge)?;
                Some(red.norm2(&ws.tmp2[..n]))
            } else {
                Some(red.norm2(&ws.tmp1[..n]))
            };
            let counters = crate::utils::convergence::SolverCounters {
                num_global_reductions: pipeline_reductions,
                overlap_global_reductions: async_waits,
                residual_replacements: async_waits,
            };
            let mut stats = SolveStats::new(0, true_res, ConvergedReason::StoppedByMonitor)
                .with_effective_runtime_policy(
                    self.effective_variant_label(),
                    self.restart,
                    self.effective_residual_check_policy_label(),
                )
                .with_counters(counters)
                .with_fgmres_counters(FgmresCounters {
                    restart_count,
                    inner_iterations_last_cycle,
                    orthog_passes: orthogonalization_passes,
                    happy_breakdowns,
                    explicit_residual_checks,
                    pipeline_fallbacks,
                    modify_pc_calls,
                    deferred_pipeline_waits,
                    immediate_pipeline_completions,
                })
                .with_orthogonalization_diagnostics(
                    orthogonalization_passes,
                    orthogonalization_rank_loss,
                    max_orthogonality_loss_estimate,
                );
            stats.final_recurrence_residual = Some(res);
            stats.final_true_residual = Some(true_res);
            stats.last_preconditioned_residual = last_preconditioned_residual;
            return Ok(stats);
        }
        explicit_residual_checks += 1;
        let true_res = recompute_true_residual_norm_s(
            a,
            b,
            x,
            comm,
            red.engine(),
            &mut ws.tmp1[..n],
            &mut ws.bridge,
        );
        let precond_res = if let Some(pc) = pc.as_deref_mut() {
            pc.apply_mut_s(pc_side, &ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge)?;
            red.norm2(&ws.tmp2[..n])
        } else {
            red.norm2(&ws.tmp1[..n])
        };
        stats.final_residual = true_res;
        stats.final_true_residual = Some(true_res);
        stats.last_preconditioned_residual = Some(precond_res);
        log_residuals(
            0,
            "FGMRES",
            ResidualSnapshot {
                true_residual: true_res,
                preconditioned_residual: precond_res,
                recurrence_residual: Some(res),
            },
        );
        if res <= thr {
            stats.reason = if res <= self.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            stats.final_residual = true_res;
            stats.final_true_residual = Some(true_res);
            stats.final_recurrence_residual = Some(res);
            stats.last_preconditioned_residual = Some(precond_res);
            let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
            let reductions =
                end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1 + pipeline_reductions;
            let counters = crate::utils::convergence::SolverCounters {
                num_global_reductions: reductions,
                overlap_global_reductions: async_waits,
                residual_replacements: async_waits,
            };
            let stats = stats.with_counters(counters);
            #[cfg(feature = "metrics")]
            {
                metrics.reductions = reductions;
                stats.metrics = metrics;
            }
            return Ok(stats
                .with_fgmres_counters(FgmresCounters {
                    restart_count,
                    inner_iterations_last_cycle,
                    orthog_passes: orthogonalization_passes,
                    happy_breakdowns,
                    explicit_residual_checks,
                    pipeline_fallbacks,
                    modify_pc_calls,
                    deferred_pipeline_waits,
                    immediate_pipeline_completions,
                })
                .with_orthogonalization_diagnostics(
                    orthogonalization_passes,
                    orthogonalization_rank_loss,
                    max_orthogonality_loss_estimate,
                ));
        }

        let mut stagnation_residuals: Vec<R> = Vec::with_capacity(6);
        let stagnation_threshold = S::from_real(0.95).real();

        while total_iters < self.maxits {
            let m_this = if self.preallocate {
                workspace_cols.min(self.maxits - total_iters)
            } else {
                self.restart.min(self.maxits - total_iters)
            };

            let mut arnoldi_steps = 0usize;
            let mut converged = false;
            let mut converged_reason: Option<ConvergedReason> = None;
            let mut hapend = false;

            for j in 0..m_this {
                let arnoldi_norm_scale = match self.variant {
                    FgmresVariant::Classical => self.solve_classical_cycle(
                        a,
                        &mut pc,
                        ws,
                        &red,
                        pc_side,
                        n,
                        j,
                        total_iters,
                        res,
                        &mut modify_pc_calls,
                        &mut orthogonalization_passes,
                        &mut max_orthogonality_loss_estimate,
                        #[cfg(feature = "metrics")]
                        &mut metrics,
                    )?,
                    FgmresVariant::Pipelined => self.solve_pipelined_cycle(
                        a,
                        &mut pc,
                        ws,
                        &red,
                        red_engine.as_ref(),
                        pc_side,
                        n,
                        j,
                        total_iters,
                        res,
                        &mut pipeline_reductions,
                        &mut async_waits,
                        &mut deferred_pipeline_waits,
                        &mut immediate_pipeline_completions,
                        &mut modify_pc_calls,
                        &mut orthogonalization_passes,
                        #[cfg(feature = "metrics")]
                        &mut metrics,
                    )?,
                };

                let h_subdiag = ws.h_at(j + 1, j).abs();
                let haptol_scaled = self.haptol.max(0.0) * arnoldi_norm_scale;
                let hap_event = h_subdiag <= haptol_scaled;
                if hap_event {
                    let loss_estimate = R::one();
                    if loss_estimate > max_orthogonality_loss_estimate {
                        max_orthogonality_loss_estimate = loss_estimate;
                    }
                }
                if hap_event {
                    *ws.h_at_mut(j + 1, j) = S::zero();
                    res = self.update_hessenberg_qr_and_residual(ws, j);
                    total_iters += 1;
                    arnoldi_steps = j + 1;
                    stats.final_recurrence_residual = Some(res);
                    hapend = true;
                    if self.happy_breakdown {
                        happy_breakdowns += 1;
                        converged = true;
                        converged_reason = Some(ConvergedReason::ConvergedHappyBreakdown);
                    } else {
                        orthogonalization_rank_loss = true;
                        explicit_residual_checks += 1;
                        let true_res = recompute_true_residual_norm_s(
                            a,
                            b,
                            x,
                            comm,
                            red.engine(),
                            &mut ws.tmp1[..n],
                            &mut ws.bridge,
                        );
                        stats = SolveStats::new(
                            total_iters,
                            true_res,
                            ConvergedReason::DivergedArnoldiRankLoss,
                        )
                        .with_orthogonalization_diagnostics(
                            orthogonalization_passes,
                            orthogonalization_rank_loss,
                            max_orthogonality_loss_estimate,
                        );
                        stats.final_true_residual = Some(true_res);
                        let precond_res = if let Some(pc_ref) = pc.as_deref_mut() {
                            pc_ref.apply_mut_s(
                                pc_side,
                                &ws.tmp1[..n],
                                &mut ws.tmp2[..n],
                                &mut ws.bridge,
                            )?;
                            red.norm2(&ws.tmp2[..n])
                        } else {
                            red.norm2(&ws.tmp1[..n])
                        };
                        stats.last_preconditioned_residual = Some(precond_res);
                        converged = true;
                        converged_reason = Some(ConvergedReason::DivergedArnoldiRankLoss);
                    }
                    break;
                }

                res = self.update_hessenberg_qr_and_residual(ws, j);
                total_iters += 1;
                arnoldi_steps = j + 1;

                #[cfg(feature = "logging")]
                if total_iters == 1 && log::log_enabled!(log::Level::Info) {
                    log::info!(
                        "FGMRES monitor semantics: {}",
                        Self::monitor_residual_semantics_tag("inner", false)
                    );
                }

                // Inner-iteration monitor payload is the Hessenberg recurrence residual |g[j+1]|.
                if call_monitors(mons, total_iters, res, pipeline_reductions) {
                    let counters = crate::utils::convergence::SolverCounters {
                        num_global_reductions: pipeline_reductions,
                        overlap_global_reductions: async_waits,
                        residual_replacements: async_waits,
                    };
                    explicit_residual_checks += 1;
                    let true_res = recompute_true_residual_norm_s(
                        a,
                        b,
                        x,
                        comm,
                        red.engine(),
                        &mut ws.tmp1[..n],
                        &mut ws.bridge,
                    );
                    let precond_res = if let Some(pc_ref) = pc.as_deref_mut() {
                        pc_ref.apply_mut_s(
                            pc_side,
                            &ws.tmp1[..n],
                            &mut ws.tmp2[..n],
                            &mut ws.bridge,
                        )?;
                        red.norm2(&ws.tmp2[..n])
                    } else {
                        red.norm2(&ws.tmp1[..n])
                    };
                    let mut stats =
                        SolveStats::new(total_iters, true_res, ConvergedReason::StoppedByMonitor)
                            .with_counters(counters)
                            .with_fgmres_counters(FgmresCounters {
                                restart_count,
                                inner_iterations_last_cycle,
                                orthog_passes: orthogonalization_passes,
                                happy_breakdowns,
                                explicit_residual_checks,
                                pipeline_fallbacks,
                                modify_pc_calls,
                                deferred_pipeline_waits,
                                immediate_pipeline_completions,
                            })
                            .with_orthogonalization_diagnostics(
                                orthogonalization_passes,
                                orthogonalization_rank_loss,
                                max_orthogonality_loss_estimate,
                            );
                    stats.final_recurrence_residual = Some(res);
                    stats.final_true_residual = Some(true_res);
                    stats.last_preconditioned_residual = Some(precond_res);
                    return Ok(stats);
                }

                if self.should_check_true_residual_every_iteration() {
                    explicit_residual_checks += 1;
                    let true_res = recompute_true_residual_norm_s(
                        a,
                        b,
                        x,
                        comm,
                        red.engine(),
                        &mut ws.tmp1[..n],
                        &mut ws.bridge,
                    );
                    if self.is_debug_residual_policy() {
                        log_residuals(
                            total_iters,
                            "FGMRES",
                            ResidualSnapshot {
                                true_residual: true_res,
                                preconditioned_residual: true_res,
                                recurrence_residual: Some(res),
                            },
                        );
                    }
                }

                if let Some(reason) = ConvergedReason::from_non_finite(res) {
                    explicit_residual_checks += 1;
                    let true_res = recompute_true_residual_norm_s(
                        a,
                        b,
                        x,
                        comm,
                        red.engine(),
                        &mut ws.tmp1[..n],
                        &mut ws.bridge,
                    );
                    stats = SolveStats::new(total_iters, true_res, reason);
                    stats.final_true_residual = Some(true_res);
                    stats.final_recurrence_residual = Some(res);
                    let precond_res = if let Some(pc_ref) = pc.as_deref_mut() {
                        pc_ref.apply_mut_s(
                            pc_side,
                            &ws.tmp1[..n],
                            &mut ws.tmp2[..n],
                            &mut ws.bridge,
                        )?;
                        red.norm2(&ws.tmp2[..n])
                    } else {
                        red.norm2(&ws.tmp1[..n])
                    };
                    stats.last_preconditioned_residual = Some(precond_res);
                    stats = stats.with_orthogonalization_diagnostics(
                        orthogonalization_passes,
                        orthogonalization_rank_loss,
                        max_orthogonality_loss_estimate,
                    );
                    converged = true;
                    break;
                }

                stagnation_residuals.push(res);
                if stagnation_residuals.len() > 6 {
                    stagnation_residuals.remove(0);
                }
                if stagnation_detected(&stagnation_residuals, stagnation_threshold) {
                    let (action, should_restart) = self.stagnation_action(j + 1);
                    log_krylov_stagnation("FGMRES", total_iters, res, action);
                    stagnation_residuals.clear();
                    if should_restart {
                        pipeline_fallbacks += 1;
                        break;
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
                    stats.iterations = total_iters;
                    converged = true;
                    converged_reason = Some(reason);
                    break;
                }
            }

            let k = arnoldi_steps;
            inner_iterations_last_cycle = k;
            let reduced_diag_threshold = self.haptol.max(0.0);
            let reduced_system_diagonal_ok = (0..k).all(|i| {
                let diag = ws.h_at(i, i);
                diag.real().is_finite() && diag.abs() > reduced_diag_threshold
            });
            if !reduced_system_diagonal_ok {
                explicit_residual_checks += 1;
                let true_res = recompute_true_residual_norm_s(
                    a,
                    b,
                    x,
                    comm,
                    red.engine(),
                    &mut ws.tmp1[..n],
                    &mut ws.bridge,
                );
                stats = SolveStats::new(
                    total_iters,
                    true_res,
                    ConvergedReason::DivergedReducedSystemSingular,
                );
                stats.final_true_residual = Some(true_res);
                stats.final_recurrence_residual = Some(res);
                let precond_res = if let Some(pc_ref) = pc.as_deref_mut() {
                    pc_ref.apply_mut_s(
                        pc_side,
                        &ws.tmp1[..n],
                        &mut ws.tmp2[..n],
                        &mut ws.bridge,
                    )?;
                    red.norm2(&ws.tmp2[..n])
                } else {
                    red.norm2(&ws.tmp1[..n])
                };
                stats.last_preconditioned_residual = Some(precond_res);
                let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
                let reductions = end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1
                    + pipeline_reductions;
                let counters = crate::utils::convergence::SolverCounters {
                    num_global_reductions: reductions,
                    overlap_global_reductions: async_waits,
                    residual_replacements: async_waits,
                };
                return Ok(stats
                    .with_counters(counters)
                    .with_fgmres_counters(FgmresCounters {
                        restart_count,
                        inner_iterations_last_cycle,
                        orthog_passes: orthogonalization_passes,
                        happy_breakdowns,
                        explicit_residual_checks,
                        pipeline_fallbacks,
                        modify_pc_calls,
                        deferred_pipeline_waits,
                        immediate_pipeline_completions,
                    })
                    .with_orthogonalization_diagnostics(
                        orthogonalization_passes,
                        orthogonalization_rank_loss,
                        max_orthogonality_loss_estimate,
                    ));
            }
            let y = match self.backsolve_least_squares(ws, k) {
                Ok(coeffs) => coeffs,
                Err(_) => {
                    explicit_residual_checks += 1;
                    let true_res = recompute_true_residual_norm_s(
                        a,
                        b,
                        x,
                        comm,
                        red.engine(),
                        &mut ws.tmp1[..n],
                        &mut ws.bridge,
                    );
                    stats = SolveStats::new(
                        total_iters,
                        true_res,
                        ConvergedReason::DivergedReducedSystemSingular,
                    );
                    stats.final_true_residual = Some(true_res);
                    stats.final_recurrence_residual = Some(res);
                    let precond_res = if let Some(pc_ref) = pc.as_deref_mut() {
                        pc_ref.apply_mut_s(
                            pc_side,
                            &ws.tmp1[..n],
                            &mut ws.tmp2[..n],
                            &mut ws.bridge,
                        )?;
                        red.norm2(&ws.tmp2[..n])
                    } else {
                        red.norm2(&ws.tmp1[..n])
                    };
                    stats.last_preconditioned_residual = Some(precond_res);
                    let end_reduct = crate::utils::reduction::test_hooks::wait_counters();
                    let reductions = end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1
                        + pipeline_reductions;
                    let counters = crate::utils::convergence::SolverCounters {
                        num_global_reductions: reductions,
                        overlap_global_reductions: async_waits,
                        residual_replacements: async_waits,
                    };
                    return Ok(stats
                        .with_counters(counters)
                        .with_fgmres_counters(FgmresCounters {
                            restart_count,
                            inner_iterations_last_cycle,
                            orthog_passes: orthogonalization_passes,
                            happy_breakdowns,
                            explicit_residual_checks,
                            pipeline_fallbacks,
                            modify_pc_calls,
                            deferred_pipeline_waits,
                            immediate_pipeline_completions,
                        })
                        .with_orthogonalization_diagnostics(
                            orthogonalization_passes,
                            orthogonalization_rank_loss,
                            max_orthogonality_loss_estimate,
                        ));
                }
            };

            self.apply_solution_correction(ws, n, &y, x);

            #[cfg(feature = "logging")]
            if log::log_enabled!(log::Level::Info) {
                log::info!(
                    "FGMRES monitor semantics: {}",
                    Self::monitor_residual_semantics_tag("restart", true)
                );
            }

            explicit_residual_checks += 1;
            let true_res = recompute_true_residual_norm_s(
                a,
                b,
                x,
                comm,
                red.engine(),
                &mut ws.tmp1[..n],
                &mut ws.bridge,
            );
            let precond_res = if let Some(pc_ref) = pc.as_deref_mut() {
                pc_ref.apply_mut_s(pc_side, &ws.tmp1[..n], &mut ws.tmp2[..n], &mut ws.bridge)?;
                red.norm2(&ws.tmp2[..n])
            } else {
                red.norm2(&ws.tmp1[..n])
            };
            stats.final_residual = true_res;
            stats.final_true_residual = Some(true_res);
            stats.final_recurrence_residual = Some(res);
            stats.last_preconditioned_residual = Some(precond_res);
            if let Some(reason) = converged_reason {
                stats.reason = reason;
            }
            if true_res <= thr {
                stats.reason = if matches!(stats.reason, ConvergedReason::ConvergedHappyBreakdown) {
                    ConvergedReason::ConvergedHappyBreakdown
                } else if true_res <= self.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                };
                break;
            }
            if !true_res.is_finite() {
                stats.reason = ConvergedReason::from_non_finite(true_res)
                    .unwrap_or(ConvergedReason::DivergedNan);
                break;
            }

            if converged {
                stats.reason = if hapend {
                    converged_reason
                } else {
                    converged_reason.or_else(|| {
                        true_residual_converged_reason(true_res, bnorm, self.atol, self.rtol)
                    })
                }
                .unwrap_or(ConvergedReason::DivergedBreakdown);
                break;
            }

            if total_iters >= self.maxits {
                break;
            }

            #[cfg(feature = "metrics")]
            let matvec_start = std::time::Instant::now();
            a.matvec_s(x, &mut ws.tmp1[..n], &mut ws.bridge);
            #[cfg(feature = "metrics")]
            {
                metrics.matvec_nanos += matvec_start.elapsed().as_nanos() as u64;
            }
            self.residual_update_in_place(&mut ws.tmp1[..n], b);
            beta0 = red.norm2(&ws.tmp1[..n]);
            #[cfg(feature = "metrics")]
            {
                metrics.bytes_reduced += std::mem::size_of::<R>();
            }

            ws.clear_gmres_restart_state();
            ws.g[0] = S::from_real(beta0);
            restart_count += 1;
            if beta0 > R::default() {
                let inv = S::from_real(1.0 / beta0);
                self.scaled_copy(&mut ws.tmp2[..n], &ws.tmp1[..n], inv);
                ws.copy_tmp2_into_vcol(0);
            } else {
                ws.v_col(0).fill(S::zero());
            }

            if let Some(hook) = self.on_restart.as_mut() {
                hook(total_iters, beta0)?;
            }
            if let Some(pc_ref) = pc.as_deref_mut() {
                if self.should_modify_pc_on_restart() {
                    modify_pc_calls += 1;
                    self.call_modify_pc_callback(total_iters, 0, beta0, Some(true_res), pc_ref)?;
                }
                pc_ref.on_restart_s(total_iters, beta0)?;
            }
        }

        stats.iterations = total_iters;
        let true_res = stats.final_true_residual.unwrap_or(stats.final_residual);
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
        let reductions =
            end_reduct.0 + end_reduct.1 - start_reduct.0 - start_reduct.1 + pipeline_reductions;
        let counters = crate::utils::convergence::SolverCounters {
            num_global_reductions: reductions,
            overlap_global_reductions: async_waits,
            residual_replacements: async_waits,
        };
        let stats = stats
            .with_counters(counters)
            .with_fgmres_counters(FgmresCounters {
                restart_count,
                inner_iterations_last_cycle,
                orthog_passes: orthogonalization_passes,
                happy_breakdowns,
                explicit_residual_checks,
                pipeline_fallbacks,
                modify_pc_calls,
                deferred_pipeline_waits,
                immediate_pipeline_completions,
            })
            .with_orthogonalization_diagnostics(
                orthogonalization_passes,
                orthogonalization_rank_loss,
                max_orthogonality_loss_estimate,
            );
        #[cfg(feature = "metrics")]
        {
            metrics.reductions = reductions;
            stats.metrics = metrics;
        }
        Ok(stats.with_reduction_model(self.reduction_model()))
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_dist_csr_inputs(
        a: &crate::matrix::DistCsrOp,
        comm: &UniverseComm,
        b_len: usize,
        x_len: usize,
    ) -> Result<(), KError> {
        let op_comm = a.comm();
        if !op_comm.congruent(comm) {
            return Err(KError::InvalidInput(format!(
                "FGMRES DistCSR route: communicator mismatch (A={:?}, solve={:?})",
                op_comm, comm
            )));
        }

        let layout = a.dist_layout().ok_or_else(|| {
            KError::InvalidInput("FGMRES DistCSR route: missing distributed layout".into())
        })?;
        let n_local = layout.row_end.saturating_sub(layout.row_start);
        if b_len != n_local || x_len != n_local {
            return Err(KError::InvalidInput(format!(
                "FGMRES DistCSR route: local vector lengths must match layout rows (b={}, x={}, local_rows={})",
                b_len, x_len, n_local
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_dist_csr(
        &mut self,
        a: &crate::matrix::DistCsrOp,
        pc: Option<&mut dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        self.solve_k_with_dist_policy(
            a,
            pc,
            b,
            x,
            pc_side,
            comm,
            monitors,
            work,
            Some(a.plan_diagnostics()),
        )
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
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let (_, n) = a.dims();
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "FGMRES: vector size mismatch".to_string(),
            ));
        }
        if let Some(dist) = a.as_any().downcast_ref::<crate::matrix::DistCsrOp>() {
            Self::validate_dist_csr_inputs(dist, comm, b.len(), x.len())?;
        }

        let mut x_s = vec![S::zero(); n];
        copy_real_to_scalar_in(x, &mut x_s);
        let mut b_s = vec![S::zero(); n];
        copy_real_to_scalar_in(b, &mut b_s);

        let mut pc_storage = pc.map(as_s_pc_mut);
        let pc_ref = pc_storage
            .as_mut()
            .map(|w| w as &mut dyn KPreconditioner<Scalar = S>);
        let stats = if let Some(dist) = a.as_any().downcast_ref::<crate::matrix::DistCsrOp>() {
            self.solve_dist_csr(dist, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work)?
        } else {
            let op = as_s_op(a);
            self.solve_k(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work)?
        };

        copy_scalar_to_real_in(&x_s, x);
        Ok(stats.with_reduction_model(self.reduction_model()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::bridge::BridgeScratch;
    use crate::algebra::scalar::KrystScalar;
    use crate::parallel::{NoComm, UniverseComm};
    use crate::utils::reduction::{ReductExec, ReductOptions};

    struct DiagOp {
        diag: Vec<S>,
    }

    impl KLinOp for DiagOp {
        type Scalar = S;

        fn dims(&self) -> (usize, usize) {
            (self.diag.len(), self.diag.len())
        }

        fn matvec_s(
            &self,
            x: &[Self::Scalar],
            y: &mut [Self::Scalar],
            _scratch: &mut BridgeScratch,
        ) {
            for ((yi, &ai), &xi) in y.iter_mut().zip(self.diag.iter()).zip(x.iter()) {
                *yi = ai * xi;
            }
        }
    }

    fn run_pipelined_with_options(comm: UniverseComm, exec: ReductExec) -> SolveStats<f64> {
        let a = DiagOp {
            diag: vec![
                S::from_real(4.0),
                S::from_real(3.0),
                S::from_real(2.0),
                S::from_real(1.5),
            ],
        };
        let b = vec![
            S::from_real(1.0),
            S::from_real(-2.0),
            S::from_real(3.0),
            S::from_real(-1.0),
        ];
        let mut x = vec![S::zero(); b.len()];
        let mut solver = FgmresSolver::new(1e-12, 60, 8);
        solver.set_variant(FgmresVariant::Pipelined);
        solver.set_pipeline_policy(PipelinePolicy::Strict);

        let mut ws = Workspace::new(b.len());
        ws.set_reduction_options(ReductOptions {
            exec,
            ..ReductOptions::default()
        });

        let stats = solver
            .solve_k(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Right,
                &comm,
                None,
                Some(&mut ws),
            )
            .expect("pipelined FGMRES solve should succeed");
        assert!(
            stats.final_residual < 1e-9,
            "expected tight convergence, got residual={}",
            stats.final_residual
        );
        stats
    }

    #[test]
    fn pipelined_sync_reduction_converges_stably() {
        let stats = run_pipelined_with_options(UniverseComm::NoComm(NoComm), ReductExec::Sync);
        assert!(stats.iterations > 0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn pipelined_async_and_sync_reductions_are_numerically_equivalent() {
        let sync_stats = run_pipelined_with_options(
            UniverseComm::Rayon(crate::parallel::rayon_comm::RayonComm::new()),
            ReductExec::Sync,
        );
        let async_stats = run_pipelined_with_options(
            UniverseComm::Rayon(crate::parallel::rayon_comm::RayonComm::new()),
            ReductExec::Async,
        );

        assert_eq!(sync_stats.reason, async_stats.reason);
        assert!(
            (sync_stats.final_residual - async_stats.final_residual).abs() < 1e-10,
            "sync={} async={}",
            sync_stats.final_residual,
            async_stats.final_residual
        );
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn mpi_smoke_pipelined_overlap_counters_are_stable() {
        let comm = UniverseComm::Mpi(std::sync::Arc::new(
            crate::parallel::mpi_comm::MpiComm::new(),
        ));
        let stats = run_pipelined_with_options(comm, ReductExec::Async);
        let counters = stats
            .fgmres_counters
            .expect("fgmres counters should be present");
        assert!(counters.deferred_pipeline_waits >= counters.immediate_pipeline_completions);
    }
}

impl LinearSolver for FgmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        // Pre-size the GMRES-family buffers during KSP setup using the workspace dimension.
        // FGMRES uses right-preconditioned Arnoldi; left/symmetric inputs map to right.
        let n = w.n();
        if n == 0 {
            return;
        }
        self.ensure_workspace(w, n);
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

impl FgmresSolver {
    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart.max(1);
    }
    pub fn set_orthog(&mut self, o: OrthogMethod) {
        self.orthog = o;
    }
    pub fn set_cgs_refinement(&mut self, refinement: CgsRefinement) {
        self.cgs_refinement = refinement;
    }
    pub fn set_reorthog(&mut self, flag: bool) {
        self.reorth = if flag {
            ReorthPolicy::Always
        } else {
            ReorthPolicy::Never
        };
        self.cgs_refinement = if flag {
            CgsRefinement::Always
        } else {
            CgsRefinement::Never
        };
    }

    pub fn set_reorth_policy(&mut self, policy: ReorthPolicy) {
        self.reorth = policy;
        self.cgs_refinement = match policy {
            ReorthPolicy::Never => CgsRefinement::Never,
            ReorthPolicy::IfNeeded => CgsRefinement::IfNeeded,
            ReorthPolicy::Always => CgsRefinement::Always,
        };
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

    pub fn set_residual_check_policy(&mut self, policy: ResidualCheckPolicy) {
        self.residual_check_policy = policy;
    }

    pub fn set_modify_pc_policy(&mut self, policy: ModifyPcPolicy) {
        self.modify_pc_policy = policy;
    }

    pub fn set_pipeline_policy(&mut self, policy: PipelinePolicy) {
        self.pipeline_policy = policy;
    }

    pub fn set_stagnation_policy(&mut self, policy: FgmresStagnationPolicy) {
        self.stagnation_policy = policy;
    }

    pub fn set_strict_true_residual_cadence(&mut self, flag: bool) {
        self.residual_check_policy = if flag {
            ResidualCheckPolicy::EveryIteration
        } else {
            ResidualCheckPolicy::OnConvergence
        };
    }

    #[cfg(test)]
    pub fn debug_config(&self) -> (usize, OrthogMethod, bool, bool) {
        (
            self.restart,
            self.orthog,
            !matches!(self.cgs_refinement, CgsRefinement::Never),
            self.happy_breakdown,
        )
    }
}
