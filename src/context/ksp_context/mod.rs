//! # KSP context
//!
//! ## Operator/PC lifecycle
//! 1. [`set_operators`] stores `A` and `P` (or `A` if `P` is `None`).
//! 2. Enforces communicator congruence via [`LinOp::comm()`]. Prefer
//!    [`try_set_operators`] in library code: it returns an error on mismatch, while
//!    [`set_operators`] panics for backward compatibility.
//! 3. [`setup`] resolves any deferred PC specs (including chains), then calls
//!    [`Preconditioner::setup`] followed by reuse logic:
//!    - If structure id changed → [`update_symbolic`]
//!    - Else if values id changed and numeric reuse allowed → [`update_numeric`]
//!    - Else unchanged.
//!
//! For efficient reuse across nonlinear iterations or time steps, wrap matrices in
//! [`DenseOp`](crate::matrix::op::DenseOp) or [`CsrOp`](crate::matrix::op::CsrOp) and call
//! [`mark_values_changed`](crate::matrix::op::DenseOp::mark_values_changed) or
//! [`mark_structure_changed`](crate::matrix::op::DenseOp::mark_structure_changed) after
//! in-place modifications. This ensures cache keys and reuse decisions reflect updates.
//!
//! ## Side policy
//! [`pc_side`](struct.KspContext.html#structfield.pc_side) is passed to solvers; PCs **do not** decide left vs right placement.
//!
//! ### Solver vs preconditioning side
//!
//! | Solver            | Allowed sides   | Notes                                                   |
//! |-------------------|-----------------|---------------------------------------------------------|
//! | `CG`, `PCG`       | Left only       | Requires HPD `A` and HPD left preconditioner `M`.       |
//! | `FGMRES`          | Right only      | Uses right-preconditioned Arnoldi; Left/Symmetric are rejected. |
//! | `PCA-GMRES`       | Left or Right   | Mapped to [`PcaPcMode`] during setup.                   |
//! | All other solvers | Left or Right   | `Symmetric` behaves like left but is passed to PCs.     |
//!
//! Incompatible combinations return [`KError::InvalidInput`] during configuration.
//!
//! ## Deferred PCs / Chaining
//! [`PcFactory::create_deferred_pc`] stores type+options without a matrix.
//! [`PcFactory::construct_deferred_preconditioner`] materializes it once `P` is known.
//! [`PcChain`] composes multiple PCs: `y = P_k(...P_1(x))`.
//!
//! ## Monitors
//! Iteration monitors receive `(iter, residual)` where the residual is solver-specific
//! (preconditioned norm for Left CG/GMRES, true norm for Right GMRES). Final stats
//! always include the true residual.
//!
//! For FGMRES specifically, inner monitor events report the Hessenberg recurrence
//! residual, while restart-boundary updates use an explicit true residual check.
//! Solver outputs expose this split through `final_recurrence_residual` vs
//! `final_true_residual`, and through `fgmres_counters.explicit_residual_checks`.
//!
//! ## PREONLY behavior
//! `Preonly` is a non-iterative mode: it invokes `Preconditioner::direct_solve` on the
//! selected preconditioner using the preconditioner operator (`P`, or `A` when `P` is `None`).
//! Use it with direct PCs such as `LU`, `QR`, or `SuperLU_DIST`.

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "rayon")]
use crate::algebra::parallel::set_rayon_threads;
use crate::algebra::parallel_cfg::{parallel_tune, set_parallel_tune};
use crate::algebra::prelude::*;
use crate::config::options::{CgVariant, KspOptions, KspType, PcOptions};
use crate::context::pc_context::{DeferredPcInfo, PcFactory, PcType};
use crate::error::KError;
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use crate::matrix::DistCsrOp;
use crate::matrix::backend::materialize;
use crate::matrix::op::{LinOp, StructureId, ValuesId, wrap_with_comm};
#[cfg(feature = "complex")]
use crate::ops::klinop::KLinOp;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::parallel::Comm;
use crate::parallel::threads::{KspExecStage, ScopedThreadPolicy};
#[cfg(feature = "mpi")]
use crate::preconditioner::PcDistributedSupport;
#[cfg(all(feature = "backend-faer", not(feature = "complex"), feature = "mpi"))]
use crate::preconditioner::asm::{AsmBlockSolver, AsmInnerPc, Weighting};
#[cfg(all(feature = "backend-faer", not(feature = "complex"), feature = "mpi"))]
use crate::preconditioner::dist::{
    DistCoarseStrategy, DistPcAdapter, DistPcBuilder, DistRouteDecisionReason,
    DistRouteResolveInput, GlobalPcKind, resolve_dist_route,
};
#[cfg(feature = "backend-faer")]
use crate::preconditioner::dist::{
    DistLocalApplyMode, DistRouteDecisionReport, DistRouteFallbackReason, DistRoutePolicy,
    DistRouteSelection, MpiPcOptions, validate_dist_route_policy_budget,
};
use crate::preconditioner::{PcReusePolicy, PcSide, Preconditioner};
use crate::reduction::ReproMode;
use crate::solver::gmres::StagnationPolicy;
use crate::solver::{
    BiCgStabSolver, BiCgStabVariant, CgSolver, CgnrSolver, CgsSolver, ChebyshevSolver, CrSolver,
    FgmresSolver, GcrSolver, GmresSolver, LinearSolver, MinresSolver, MonitorAction,
    MonitorCallback, PCG_PIPELINED_DEFAULT_REPLACE_EVERY, PcaGmresSolver, PcaPcMode, PcgSolver,
    PcgVariant, PipeGcrSolver, RichardsonSolver, TcqmrSolver,
};
#[cfg(feature = "complex")]
use crate::solver::{QmrSolver, TfqmrSolver};
use crate::utils::convergence::{
    AcceptanceStatus, ConvergedReason, FailureStage, ReasonDiagnosticsCounters, ReasonEmitter,
    SolveStats, classify_acceptance_status,
};
use crate::utils::diagnostics::{KspDiagnostics, PcDiagnostics};
use crate::utils::reduction::{ReductOptions, reduction_latency_estimate_us};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
#[cfg(feature = "backend-faer")]
mod distcsr_capability;
mod execution;
mod workspace;
pub use crate::core::block::BlockVec;
#[cfg(feature = "backend-faer")]
use distcsr_capability::{
    DistCsrCapabilityEntry, DistCsrCapabilityKey, build_dist_route_decision_report,
    resolve_distcsr_capability,
};
use execution::KrylovVariant;
pub use execution::{
    AdaptiveExecutionDecision, ExecutionPolicy, NestedPolicyContext, OverlapStrategy,
    ThreadingPolicy,
};
pub use workspace::{
    GmresSStepWorkspace, GmresSpec, GmresWorkspaceLayout, PipeReduct, ReorthPolicy, Workspace,
};

#[cfg(feature = "complex")]
struct LinOpAsK<'a> {
    inner: &'a dyn LinOp<S = S>,
}

#[cfg(feature = "complex")]
impl<'a> KLinOp for LinOpAsK<'a> {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        self.inner.dims()
    }

    #[inline]
    fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        self.inner.matvec(x, y);
    }

    #[inline]
    fn supports_t_matvec_s(&self) -> bool {
        self.inner.supports_transpose()
    }

    #[inline]
    fn t_matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        self.inner.t_matvec(x, y);
    }
}

#[cfg(feature = "complex")]
struct PcAsK<'a> {
    inner: &'a mut dyn Preconditioner,
}

#[cfg(feature = "complex")]
impl<'a> KPreconditioner for PcAsK<'a> {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        self.inner.dims()
    }

    #[inline]
    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        self.inner.apply(side, x, y)
    }

    #[inline]
    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        self.inner.apply_mut(side, x, y)
    }

    #[inline]
    fn on_restart_s(&mut self, outer_iter: usize, residual_norm: R) -> Result<(), KError> {
        self.inner.on_restart(outer_iter, residual_norm)
    }
}

/// Supported solver types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    Cg,
    Cgnr,
    Gmres,
    Fgmres,
    BiCgStab,
    Cgs,
    Pcg,
    Minres,
    Lsqr,
    Lsmr,
    PcaGmres,
    Qmr,
    Tfqmr,
    Tcqmr,
    Richardson,
    Chebyshev,
    Cr,
    Gcr,
    PipeGcr,
    Preonly,
}

impl SolverType {
    /// Return the preconditioning side required by this solver, if any.
    #[inline]
    pub fn required_pc_side(self) -> Option<PcSide> {
        match self {
            SolverType::Cg
            | SolverType::Pcg
            | SolverType::Minres
            | SolverType::Lsqr
            | SolverType::Lsmr
            | SolverType::Chebyshev
            | SolverType::Cr => Some(PcSide::Left),
            _ => None,
        }
    }

    #[inline]
    pub fn right_only_pc_side(self) -> bool {
        matches!(
            self,
            SolverType::Fgmres | SolverType::Gcr | SolverType::PipeGcr
        )
    }
}

impl FromStr for SolverType {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cg" => Ok(SolverType::Cg),
            "cgnr" => Ok(SolverType::Cgnr),
            "gmres" => Ok(SolverType::Gmres),
            "fgmres" => Ok(SolverType::Fgmres),
            "bicgstab" => Ok(SolverType::BiCgStab),
            "cgs" => Ok(SolverType::Cgs),
            "pcg" => Ok(SolverType::Pcg),
            "minres" => Ok(SolverType::Minres),
            "lsqr" => Ok(SolverType::Lsqr),
            "lsmr" => Ok(SolverType::Lsmr),
            "pca_gmres" | "pcagmres" => Ok(SolverType::PcaGmres),
            "qmr" => Ok(SolverType::Qmr),
            "tfqmr" => Ok(SolverType::Tfqmr),
            "tcqmr" => Ok(SolverType::Tcqmr),
            "richardson" => Ok(SolverType::Richardson),
            "chebyshev" => Ok(SolverType::Chebyshev),
            "cr" => Ok(SolverType::Cr),
            "gcr" => Ok(SolverType::Gcr),
            "gcr_pipe" | "pipegcr" => Ok(SolverType::PipeGcr),
            "preonly" => Ok(SolverType::Preonly),
            other => Err(KError::UnrecognizedSolverType(other.to_string())),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MonitorPolicy {
    AllRanks,
    Rank0Only,
}

#[derive(Clone, Debug)]
struct PcChainPlan {
    candidates: Vec<Vec<DeferredPcInfo>>,
    active: usize,
}

impl PcChainPlan {
    fn new(candidates: Vec<Vec<DeferredPcInfo>>) -> Result<Self, KError> {
        if candidates.is_empty() {
            return Err(KError::InvalidInput("empty PC chain".into()));
        }
        Ok(Self {
            candidates,
            active: 0,
        })
    }

    fn active_specs(&self) -> &[DeferredPcInfo] {
        &self.candidates[self.active]
    }

    fn advance(&mut self) -> bool {
        if self.active + 1 < self.candidates.len() {
            self.active += 1;
            true
        } else {
            false
        }
    }
}

/// Minimal KSP context holding solver, preconditioner, and operators.
pub struct KspContext {
    solver: Option<Box<dyn LinearSolver<Error = KError> + 'static>>,
    pc: Option<Box<dyn Preconditioner>>,
    pub(crate) pending_pc: Option<DeferredPcInfo>,
    pc_spec: Option<DeferredPcInfo>,
    pub(crate) pc_chain_plan: Option<PcChainPlan>,
    amat: Option<Arc<dyn LinOp<S = S>>>,
    pmat: Option<Arc<dyn LinOp<S = S>>>,
    bound_comm: Option<crate::parallel::UniverseComm>,
    work: Option<Workspace>,
    setup_called: bool,
    monitors: Vec<Box<MonitorCallback<R>>>,
    monitor_policy: MonitorPolicy,
    solver_type: Option<SolverType>,
    pub rtol: R,
    pub atol: R,
    pub dtol: R,
    pub maxits: usize,
    pub restart: usize,
    pub pc_side: PcSide,
    pc_side_explicit: bool,
    pc_reuse: PcReusePolicy,
    last_pc_sid: Option<StructureId>,
    last_pc_vid: Option<ValuesId>,
    reduction_opts: ReductOptions,
    reproducible: bool,
    exec: ExecutionPolicy,
    scoped_threads: ScopedThreadPolicy,
    #[cfg(feature = "backend-faer")]
    pending_mpi_pc: Option<PendingMpiPc>,
    #[cfg(feature = "backend-faer")]
    dist_route_diag: DistRouteDiagnosticsState,
    // Pending/staged solver-specific options to apply when solver type is set
    pending_gmres: PendingGmres,
    pending_fgmres: PendingFgmres,
    pending_pcg: PendingPcg,
    pending_bicgstab: PendingBiCgStab,
    last_converged_reason: Option<ConvergedReason>,
    reason_counters: ReasonDiagnosticsCounters,
    adaptive_exec: Option<AdaptiveExecutionDecision>,
}

impl fmt::Debug for KspContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_struct("KspContext");
        dbg.field("solver", &self.solver.as_ref().map(|_| "set"))
            .field("pc", &self.pc.as_ref().map(|_| "set"))
            .field("pending_pc", &self.pending_pc)
            .field("pc_spec", &self.pc_spec)
            .field("pc_chain_plan", &self.pc_chain_plan)
            .field("amat_set", &self.amat.is_some())
            .field("pmat_set", &self.pmat.is_some())
            .field("bound_comm", &self.bound_comm)
            .field("work", &self.work)
            .field("setup_called", &self.setup_called)
            .field("monitors_len", &self.monitors.len())
            .field("monitor_policy", &self.monitor_policy)
            .field("solver_type", &self.solver_type)
            .field("rtol", &self.rtol)
            .field("atol", &self.atol)
            .field("dtol", &self.dtol)
            .field("maxits", &self.maxits)
            .field("restart", &self.restart)
            .field("pc_side", &self.pc_side)
            .field("pc_side_explicit", &self.pc_side_explicit)
            .field("pc_reuse", &self.pc_reuse)
            .field("last_pc_sid", &self.last_pc_sid)
            .field("last_pc_vid", &self.last_pc_vid)
            .field("reduction_opts", &self.reduction_opts)
            .field("reproducible", &self.reproducible)
            .field("exec", &self.exec)
            .field("scoped_threads", &self.scoped_threads);
        #[cfg(feature = "backend-faer")]
        dbg.field("pending_mpi_pc", &self.pending_mpi_pc);
        dbg.field("pending_gmres", &self.pending_gmres)
            .field("pending_fgmres", &self.pending_fgmres)
            .field("pending_pcg", &self.pending_pcg)
            .field("pending_bicgstab", &self.pending_bicgstab)
            .field("last_converged_reason", &self.last_converged_reason)
            .field("reason_counters", &self.reason_counters)
            .field("adaptive_exec", &self.adaptive_exec);
        dbg.finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingGmresVariant {
    Classical,
    Pipelined,
    SStep,
}

#[derive(Clone, Debug, Default)]
struct PendingGmres {
    restart: Option<usize>,
    orthog: Option<crate::solver::gmres::GmresOrthog>,
    reorth: Option<ReorthPolicy>,
    reorth_tol: Option<R>,
    happy_breakdown: Option<bool>,
    stagnation_policy: Option<StagnationPolicy>,
    variant: Option<PendingGmresVariant>,
    sstep: Option<usize>,
    sstep_max_cond: Option<R>,
}

#[derive(Clone, Debug, Default)]
struct PendingFgmres {
    restart: Option<usize>,
    orthog: Option<crate::solver::fgmres::OrthogMethod>,
    reorth: Option<ReorthPolicy>,
    reorth_tol: Option<R>,
    happy_breakdown: Option<bool>,
    variant: Option<crate::solver::fgmres::FgmresVariant>,
}

#[derive(Clone, Debug, Default)]
struct PendingPcg {
    pipelined: Option<bool>,
    replace_every: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct PendingBiCgStab {
    variant: Option<BiCgStabVariant>,
    replace_every: Option<usize>,
}

#[derive(Clone, Debug)]
#[cfg(feature = "backend-faer")]
struct PendingMpiPc {
    mpi_opts: MpiPcOptions,
    pc_opts: PcOptions,
}

#[derive(Clone, Debug, Default)]
#[cfg(feature = "backend-faer")]
struct DistRouteDiagnosticsState {
    selected_route: Option<String>,
    capability_entry: Option<DistCsrCapabilityEntry>,
    decision_report: Option<DistRouteDecisionReport>,
    fallback_chain: Vec<String>,
    fallback_reason: Option<String>,
    fallback_counters: BTreeMap<String, usize>,
    preflight: Option<DistRoutePreflightState>,
    replay_tokens: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
#[cfg(feature = "backend-faer")]
struct DistRoutePreflightState {
    probe_key: String,
    outcome: String,
    reason_codes: Vec<String>,
    native_ready: bool,
    cached_hits: usize,
}

fn insert_value<T: Serialize>(map: &mut BTreeMap<String, Value>, key: &str, value: T) {
    if let Ok(value) = serde_json::to_value(value) {
        map.insert(key.to_string(), value);
    }
}

impl Default for KspContext {
    fn default() -> Self {
        Self::new()
    }
}

impl KspContext {
    #[cfg(feature = "backend-faer")]
    fn distributed_setting_warnings(
        route_policy: DistRoutePolicy,
        local_apply_mode: DistLocalApplyMode,
    ) -> Vec<&'static str> {
        let mut warnings = Vec::new();

        if matches!(
            route_policy,
            DistRoutePolicy::Adapted | DistRoutePolicy::RootGather
        ) && local_apply_mode.is_distributed_native()
        {
            warnings.push("native_local_apply_with_non_native_route_policy");
        }

        if route_policy == DistRoutePolicy::Native && !local_apply_mode.is_distributed_native() {
            warnings.push("native_route_policy_with_adapted_local_wrapper");
        }

        warnings
    }

    #[cfg(feature = "backend-faer")]
    fn distributed_mode_family(route: Option<&str>) -> Option<&'static str> {
        let route = route?;
        if route.starts_with("distcsr_native_block_jacobi")
            || route.starts_with("configured_global")
        {
            return Some("native_distributed");
        }
        if route.starts_with("local_adapter") || route.starts_with("root_gather") {
            return Some("adapter_distributed");
        }
        None
    }

    #[inline]
    fn effective_side_for_solver(side: PcSide, solver_type: SolverType) -> PcSide {
        match solver_type {
            SolverType::Fgmres | SolverType::Gcr | SolverType::PipeGcr => PcSide::Right,
            _ => match side {
                PcSide::Symmetric => PcSide::Left,
                s => s,
            },
        }
    }

    #[inline]
    fn effective_pc_side(&self) -> PcSide {
        self.solver_type
            .map(|st| Self::effective_side_for_solver(self.pc_side, st))
            .unwrap_or(self.pc_side)
    }

    fn parse_reorth_policy(label: &str) -> Result<ReorthPolicy, KError> {
        match label.to_lowercase().as_str() {
            "never" => Ok(ReorthPolicy::Never),
            "ifneeded" | "if-needed" => Ok(ReorthPolicy::IfNeeded),
            "always" => Ok(ReorthPolicy::Always),
            other => Err(KError::SolveError(format!(
                "Unrecognized reorth policy: {other} (expected 'never'|'ifneeded'|'always')"
            ))),
        }
    }

    fn parse_gmres_variant(label: &str) -> Result<(PendingGmresVariant, Option<usize>), KError> {
        let (variant, maybe_s) = match label.split_once(':') {
            Some((v, s)) => (v.trim(), Some(s.trim())),
            None => (label.trim(), None),
        };
        let parsed_variant = match variant {
            "classical" => PendingGmresVariant::Classical,
            "pipelined" => PendingGmresVariant::Pipelined,
            "sstep" => PendingGmresVariant::SStep,
            other => {
                return Err(KError::SolveError(format!(
                    "Unrecognized ksp_gmres_variant: {other} (expected 'classical'|'pipelined'|'sstep')"
                )));
            }
        };
        let sstep = if let Some(s) = maybe_s {
            let val: usize = s.parse().map_err(|_| {
                KError::SolveError(format!("Invalid ksp_gmres_variant s-step size: {s}"))
            })?;
            if val < 1 {
                return Err(KError::SolveError(
                    "ksp_gmres_variant s-step size must be >= 1".into(),
                ));
            }
            Some(val)
        } else {
            None
        };
        Ok((parsed_variant, sstep))
    }

    fn parse_reduction_mode(label: &str) -> Result<ReproMode, KError> {
        match label.to_lowercase().as_str() {
            "fast" => Ok(ReproMode::Fast),
            "deterministic" | "det" => Ok(ReproMode::Deterministic),
            "deterministic-accurate" | "deterministic_accurate" | "accurate" => {
                Ok(ReproMode::DeterministicAccurate)
            }
            other => Err(KError::SolveError(format!(
                "Unrecognized ksp_reduction mode: {other} (expected 'fast'|'deterministic'|'deterministic-accurate')"
            ))),
        }
    }

    /// Validate that `side` is compatible with `solver_type` (if set).
    /// Mirrors `configure_pc_side()` logic but used at set-time to fail fast.
    fn check_pc_side_now(&self, side: PcSide) -> Result<(), KError> {
        let normalized = if let Some(st) = self.solver_type {
            Self::effective_side_for_solver(side, st)
        } else {
            side
        };
        if let Some(st) = self.solver_type {
            if st.right_only_pc_side() && side != PcSide::Right {
                return Err(KError::InvalidInput(format!(
                    "{st:?} supports only right preconditioning; got {side:?}"
                )));
            }
            if let Some(required) = st.required_pc_side() {
                if normalized != required {
                    return Err(KError::InvalidInput(format!(
                        "{st:?} requires left preconditioning; got {side:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn bind_or_check_comm(&mut self, comm: &crate::parallel::UniverseComm) -> Result<(), KError> {
        if let Some(ref bound) = self.bound_comm {
            if !bound.congruent(comm) {
                return Err(KError::InvalidInput(format!(
                    "KspContext communicator mismatch: bound={}, new={}",
                    bound.id(),
                    comm.id()
                )));
            }
        } else {
            self.bound_comm = Some(comm.clone());
        }
        Ok(())
    }

    pub fn new() -> Self {
        Self {
            solver: None,
            pc: None,
            pending_pc: None,
            pc_spec: None,
            pc_chain_plan: None,
            amat: None,
            pmat: None,
            bound_comm: None,
            work: None,
            setup_called: false,
            monitors: Vec::new(),
            monitor_policy: MonitorPolicy::AllRanks,
            solver_type: None,
            rtol: 1e-5,
            atol: 1e-50,
            dtol: 1e5,
            maxits: 10_000,
            restart: 30,
            pc_side: PcSide::Left,
            pc_side_explicit: false,
            pc_reuse: PcReusePolicy::Auto,
            last_pc_sid: None,
            last_pc_vid: None,
            reduction_opts: ReductOptions::default(),
            reproducible: false,
            exec: ExecutionPolicy::default(),
            scoped_threads: ScopedThreadPolicy::default(),
            #[cfg(feature = "backend-faer")]
            pending_mpi_pc: None,
            #[cfg(feature = "backend-faer")]
            dist_route_diag: DistRouteDiagnosticsState::default(),
            pending_gmres: PendingGmres::default(),
            pending_fgmres: PendingFgmres::default(),
            pending_pcg: PendingPcg::default(),
            pending_bicgstab: PendingBiCgStab::default(),
            last_converged_reason: None,
            reason_counters: ReasonDiagnosticsCounters::default(),
            adaptive_exec: None,
        }
    }

    #[cfg(feature = "rayon")]
    pub fn with_thread_pool(mut self, pool: Arc<rayon::ThreadPool>) -> Self {
        self.exec.threading = ThreadingPolicy::Pool(pool);
        self
    }

    pub fn set_execution_policy(&mut self, policy: ExecutionPolicy) -> &mut Self {
        self.exec = policy;
        self
    }

    #[cfg(feature = "rayon")]
    pub fn set_threads(&mut self, n: usize) -> Result<&mut Self, KError> {
        self.exec = self.exec.clone().with_threads(n)?;
        Ok(self)
    }

    #[cfg(not(feature = "rayon"))]
    pub fn set_threads(&mut self, _n: usize) -> Result<&mut Self, KError> {
        Err(KError::Unsupported(
            "thread pool configuration requires feature=\"rayon\"".into(),
        ))
    }

    pub fn set_type(&mut self, solver_type: SolverType) -> Result<&mut Self, KError> {
        if solver_type.right_only_pc_side() {
            if self.pc_side_explicit && self.pc_side != PcSide::Right {
                return Err(KError::InvalidInput(format!(
                    "{solver_type:?} supports only right preconditioning; got {:?}",
                    self.pc_side
                )));
            }
            if !self.pc_side_explicit {
                self.pc_side = PcSide::Right;
            }
        }
        if let Some(required) = solver_type.required_pc_side() {
            let normalized = Self::effective_side_for_solver(self.pc_side, solver_type);
            if self.pc_side_explicit {
                if normalized != required {
                    return Err(KError::InvalidInput(format!(
                        "{solver_type:?} requires left preconditioning; got {:?}",
                        self.pc_side
                    )));
                }
            } else {
                self.pc_side = required;
            }
        }

        self.solver_type = Some(solver_type);
        let solver: Option<Box<dyn LinearSolver<Error = KError> + 'static>> = match solver_type {
            SolverType::Cg => Some(Box::new(
                CgSolver::new(self.rtol, self.maxits)
                    .with_norm(crate::solver::cg::CgNormType::Preconditioned),
            )),
            SolverType::Cgnr => Some(Box::new(CgnrSolver::new(self.rtol, self.maxits))),
            SolverType::Gmres => {
                let mut s = GmresSolver::new(self.restart, self.rtol, self.maxits);
                // Apply any staged GMRES parameters
                self.apply_gmres_pending_to(&mut s);
                Some(Box::new(s))
            }
            SolverType::Fgmres => {
                let mut s = FgmresSolver::new(self.rtol, self.maxits, self.restart);
                self.apply_fgmres_pending_to(&mut s);
                Some(Box::new(s))
            }
            SolverType::BiCgStab => Some(Box::new({
                let mut s = BiCgStabSolver::new(self.rtol, self.maxits);
                Self::apply_bicgstab_pending(&self.pending_bicgstab, &mut s);
                s
            })),
            SolverType::Cgs => Some(Box::new(CgsSolver::new(self.rtol, self.maxits))),
            SolverType::Pcg => Some(Box::new({
                let mut s = PcgSolver::new(self.rtol, self.maxits)
                    .with_norm(crate::solver::pcg::CgNormType::Preconditioned);
                self.apply_pcg_pending_to(&mut s);
                s
            })),
            SolverType::Minres => Some(Box::new(MinresSolver::new(self.rtol, self.maxits))),
            SolverType::Lsqr => Some(Box::new(crate::solver::LsqrSolver::new(
                self.rtol,
                self.maxits,
            ))),
            SolverType::Lsmr => Some(Box::new(crate::solver::LsmrSolver::new(
                self.rtol,
                self.maxits,
            ))),
            SolverType::PcaGmres => {
                let mut s = PcaGmresSolver::new(self.restart, 1, 1, self.rtol, self.maxits);
                s.pc_mode = crate::solver::PcaPcMode::Left;
                Some(Box::new(s))
            }
            SolverType::Qmr => Some(Box::new(crate::solver::QmrSolver::new(
                self.rtol,
                self.maxits,
            ))),
            SolverType::Tfqmr => Some(Box::new(crate::solver::TfqmrSolver::new(
                self.rtol,
                self.maxits,
            ))),
            SolverType::Tcqmr => Some(Box::new(TcqmrSolver::new(self.rtol, self.maxits))),
            SolverType::Richardson => Some(Box::new(RichardsonSolver::new(self.rtol, self.maxits))),
            SolverType::Chebyshev => {
                let mut s = ChebyshevSolver::new(self.rtol, self.maxits);
                s.set_omega(0.8);
                Some(Box::new(s))
            }
            SolverType::Cr => Some(Box::new(CrSolver::new(self.rtol, self.maxits))),
            SolverType::Gcr => Some(Box::new(GcrSolver::new(
                self.restart,
                self.rtol,
                self.maxits,
            ))),
            SolverType::PipeGcr => Some(Box::new(PipeGcrSolver::new(
                self.restart,
                self.rtol,
                self.maxits,
            ))),
            SolverType::Preonly => {
                // PREONLY is intentionally "no iterative solver".
                // We’ll dispatch to `pc.direct_solve()` in `solve()`.
                None
            }
        };
        self.solver = solver;
        // Fail fast if an explicit side was set and is incompatible with the selected solver
        if self.pc_side_explicit {
            self.check_pc_side_now(self.pc_side)?
        }
        self.invalidate_solver_setup();
        Ok(self)
    }

    pub fn set_type_from_str(&mut self, solver_type: &str) -> Result<&mut Self, KError> {
        let st = SolverType::from_str(solver_type)?;
        self.set_type(st)
    }

    pub fn set_pc_type(
        &mut self,
        pc_type: PcType,
        opts: Option<&PcOptions>,
    ) -> Result<&mut Self, KError> {
        let spec = PcFactory::create_deferred_pc(pc_type, opts.cloned())?;
        self.pc_spec = Some(spec.clone());
        match PcFactory::create_preconditioner(pc_type, opts) {
            Ok(pc) => {
                self.pc = Some(pc);
                self.pending_pc = None;
                self.pc_chain_plan = None;
            }
            Err(_) => {
                self.pc = None;
                self.pending_pc = Some(spec);
                self.pc_chain_plan = None;
            }
        }
        self.invalidate_pc_setup();
        Ok(self)
    }

    pub fn set_pc_type_from_str(&mut self, pc_type: &str) -> Result<&mut Self, KError> {
        let pct = PcType::from_str(pc_type)?;
        self.set_pc_type(pct, None)
    }

    fn set_pc_chain_candidates_from_specs(
        &mut self,
        candidates: Vec<Vec<DeferredPcInfo>>,
    ) -> Result<&mut Self, KError> {
        self.pc = None;
        self.pending_pc = None;
        self.pc_spec = None;
        self.pc_chain_plan = Some(PcChainPlan::new(candidates)?);
        self.invalidate_pc_setup();
        Ok(self)
    }

    /// Configure a preconditioner chain with fallback candidates.
    pub fn set_pc_chain_candidates_from_options(
        &mut self,
        candidates: Vec<Vec<PcOptions>>,
    ) -> Result<&mut Self, KError> {
        let specs = candidates
            .into_iter()
            .map(|chain| PcFactory::create_deferred_pc_chain_from_options(&chain))
            .collect::<Result<Vec<_>, _>>()?;
        self.set_pc_chain_candidates_from_specs(specs)
    }

    /// Convenience for PREONLY: set solver type and a direct PC in one call.
    pub fn set_preonly_with_pc(
        &mut self,
        pc_type: PcType,
        opts: Option<&PcOptions>,
    ) -> Result<&mut Self, KError> {
        self.set_type(SolverType::Preonly)?;
        self.set_pc_type(pc_type, opts)?;
        Ok(self)
    }

    /// Set the preconditioning side directly (panics if incompatible with the active solver).
    ///
    /// Prefer `try_set_pc_side` in library code to handle errors.
    pub fn set_pc_side(&mut self, side: PcSide) -> &mut Self {
        self.try_set_pc_side(side).unwrap()
    }

    /// Set the preconditioning side, failing early if incompatible with the current solver.
    pub fn try_set_pc_side(&mut self, side: PcSide) -> Result<&mut Self, KError> {
        self.check_pc_side_now(side)?;
        self.pc_side = side;
        self.pc_side_explicit = true;
        self.invalidate_solver_setup();
        Ok(self)
    }

    /// Set the preconditioning side from a string ("left", "right", or "symmetric").
    /// Fails fast if incompatible with the active solver.
    pub fn set_pc_side_from_str(&mut self, side: &str) -> Result<&mut Self, KError> {
        let ps = PcSide::from_str(side)?;
        self.try_set_pc_side(ps)
    }

    /// Configure the KSP context using parsed KSP options.
    pub fn set_from_options(&mut self, opts: &KspOptions) -> Result<&mut Self, KError> {
        #[cfg(feature = "rayon")]
        {
            self.scoped_threads.set_outer_threads(opts.threads);
            self.scoped_threads.set_inner_threads(opts.threads);
            if let Some(mode) = opts.threads_mode.as_deref() {
                match mode {
                    "context" => {}
                    "hybrid" => {}
                    "global" => {}
                    "serial" => {
                        self.exec.threading = ThreadingPolicy::Serial;
                        self.scoped_threads.set_outer_threads(Some(1));
                        self.scoped_threads.set_inner_threads(Some(1));
                    }
                    other => {
                        return Err(KError::InvalidInput(format!(
                            "unknown ksp_threads_mode: {other}"
                        )));
                    }
                }
            }

            if let Some(n) = opts.threads {
                match opts.threads_mode.as_deref().unwrap_or("context") {
                    "context" | "hybrid" => {
                        self.exec = self.exec.clone().with_threads(n)?;
                        self.scoped_threads.set_outer_threads(Some(n));
                    }
                    "serial" => {
                        self.exec.threading = ThreadingPolicy::Serial;
                        self.scoped_threads.set_outer_threads(Some(1));
                    }
                    "global" => {
                        set_rayon_threads(n);
                        self.scoped_threads.set_outer_threads(Some(n));
                    }
                    other => {
                        return Err(KError::InvalidInput(format!(
                            "unknown ksp_threads_mode: {other}"
                        )));
                    }
                }
            }
        }

        #[cfg(all(not(feature = "rayon"), feature = "logging"))]
        if opts.threads.is_some() || opts.threads_mode.is_some() {
            log::warn!("Ignoring ksp_threads: build without feature=\"rayon\"");
        }

        if opts.min_len_vec.is_some()
            || opts.min_rows_spmv.is_some()
            || opts.chunk_rows_spmv.is_some()
            || opts.min_work_spmm_dense.is_some()
            || opts.chunk_rows_spmm_dense.is_some()
            || opts.chunk_cols_spmm_dense.is_some()
        {
            let mut tune = parallel_tune();
            if let Some(v) = opts.min_len_vec {
                tune.min_len_vec = v;
            }
            if let Some(v) = opts.min_rows_spmv {
                tune.min_rows_spmv = v;
            }
            if let Some(v) = opts.chunk_rows_spmv {
                tune.chunk_rows_spmv = v;
            }
            if let Some(v) = opts.min_work_spmm_dense {
                tune.min_work_spmm_dense = v;
            }
            if let Some(v) = opts.chunk_rows_spmm_dense {
                tune.chunk_rows_spmm_dense = v;
            }
            if let Some(v) = opts.chunk_cols_spmm_dense {
                tune.chunk_cols_spmm_dense = v;
            }
            set_parallel_tune(tune);
        }

        let requested_solver_type = opts
            .ksp_type
            .as_deref()
            .map(SolverType::from_str)
            .transpose()?;
        let effective_solver_type = requested_solver_type.or(self.solver_type);

        if opts.richardson_omega.is_some()
            && !matches!(effective_solver_type, Some(SolverType::Richardson))
        {
            return Err(KError::InvalidInput(
                "-ksp_richardson_omega requires -ksp_type richardson".into(),
            ));
        }
        if opts.chebyshev_omega.is_some()
            && !matches!(effective_solver_type, Some(SolverType::Chebyshev))
        {
            return Err(KError::InvalidInput(
                "-ksp_chebyshev_omega requires -ksp_type chebyshev".into(),
            ));
        }
        if opts.gcr_restart.is_some()
            && !matches!(
                self.solver_type,
                Some(SolverType::Gcr | SolverType::PipeGcr)
            )
        {
            return Err(KError::InvalidInput(
                "-ksp_gcr_restart requires -ksp_type gcr or -ksp_type pipegcr".into(),
            ));
        }
        if let Some(omega) = opts.richardson_omega {
            if omega <= 0.0 {
                return Err(KError::InvalidInput(
                    "-ksp_richardson_omega must be > 0".into(),
                ));
            }
        }
        if let Some(omega) = opts.chebyshev_omega {
            if omega <= 0.0 {
                return Err(KError::InvalidInput(
                    "-ksp_chebyshev_omega must be > 0".into(),
                ));
            }
        }
        let mut scalar_solver_config_changed = false;
        if let Some(rtol) = opts.rtol {
            self.rtol = rtol;
            scalar_solver_config_changed = true;
        }
        if let Some(atol) = opts.atol {
            self.atol = atol;
            scalar_solver_config_changed = true;
        }
        if let Some(dtol) = opts.dtol {
            self.dtol = dtol;
            scalar_solver_config_changed = true;
        }
        if let Some(maxits) = opts.maxits {
            self.maxits = maxits;
            scalar_solver_config_changed = true;
        }
        if let Some(restart) = opts.restart {
            self.restart = restart;
        }
        if let (Some(SolverType::Fgmres), Some(side_label)) =
            (requested_solver_type, opts.pc_side.as_ref())
        {
            let parsed_side = PcSide::from_str(side_label)?;
            if parsed_side != PcSide::Right {
                return Err(KError::InvalidInput(format!(
                    "FGMRES supports only right preconditioning; got {parsed_side:?} from -ksp_pc_side={side_label}"
                )));
            }
        }
        if let Some(st) = requested_solver_type {
            self.set_type(st)?;
        } else if scalar_solver_config_changed {
            self.invalidate_solver_setup();
        }
        if let Some(ref side) = opts.pc_side {
            self.set_pc_side_from_str(side)?;
        }
        if let Some(ref mode) = opts.reduction {
            let parsed = Self::parse_reduction_mode(mode)?;
            self.reduction_opts.mode = parsed;
            if let Some(ref mut w) = self.work {
                w.set_reduction_mode(parsed);
            }
        }

        if let Some(rank0_only) = opts.ksp_monitor_rank0 {
            self.monitor_policy = if rank0_only {
                MonitorPolicy::Rank0Only
            } else {
                MonitorPolicy::AllRanks
            };
        }

        if let Some(flag) = opts.reproducible {
            self.reproducible = flag;
            self.reduction_opts.reproducible = flag;
            self.exec = self.exec.clone().with_reproducible(flag);
        }

        let requested_cg_variant = opts.cg_variant.or_else(|| {
            opts.cg_pipelined.map(|flag| {
                if flag {
                    CgVariant::Pipelined
                } else {
                    CgVariant::Classic
                }
            })
        });

        if let Some(r) = opts.gcr_restart {
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|b| b.as_any_mut().downcast_mut::<GcrSolver>())
            {
                *s = GcrSolver::new(r, self.rtol, self.maxits);
            }
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|b| b.as_any_mut().downcast_mut::<PipeGcrSolver>())
            {
                *s = PipeGcrSolver::new(r, self.rtol, self.maxits);
            }
        }

        if let Some(omega) = opts.richardson_omega {
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|b| b.as_any_mut().downcast_mut::<RichardsonSolver>())
            {
                s.set_omega(omega);
            }
        }

        if let Some(omega) = opts.chebyshev_omega {
            if matches!(self.solver_type, Some(SolverType::Chebyshev)) {
                if let Some(s) = self
                    .solver
                    .as_mut()
                    .and_then(|b| b.as_any_mut().downcast_mut::<RichardsonSolver>())
                {
                    s.set_omega(omega);
                }
            }
        }

        // --- GMRES options ---
        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<GmresSolver>())
        {
            if let Some(r) = opts.effective_restart_for(KspType::GMRES) {
                s.set_restart(r);
                self.restart = r;
                self.pending_gmres.restart = Some(r);
            }
            if let Some(ref orth) = opts.gmres_orthog {
                let o = match orth.as_str() {
                    "mgs" | "modified" => crate::solver::gmres::GmresOrthog::Mgs,
                    "cgs" | "classical" => crate::solver::gmres::GmresOrthog::Cgs,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_gmres_orthog: {other} (expected 'mgs'|'modified'|'cgs'|'classical')"
                        )));
                    }
                };
                s.set_orthog(o);
                self.pending_gmres.orthog = Some(o);
            }
            if let Some(ref mode) = opts.gmres_reorth {
                let policy = Self::parse_reorth_policy(mode)?;
                s.set_reorth_policy(policy);
                self.pending_gmres.reorth = Some(policy);
            } else if let Some(flag) = opts.gmres_reorthog {
                s.set_reorthog(flag);
                self.pending_gmres.reorth = Some(if flag {
                    ReorthPolicy::Always
                } else {
                    ReorthPolicy::Never
                });
            }
            if let Some(tol) = opts.gmres_reorth_tol {
                s.set_reorth_tol(tol);
                self.pending_gmres.reorth_tol = Some(tol);
            }
            if let Some(flag) = opts.gmres_happy_breakdown {
                s.set_happy_breakdown(flag);
                self.pending_gmres.happy_breakdown = Some(flag);
            }
            if let Some(ref variant) = opts.gmres_variant {
                let (pv, sstep) = Self::parse_gmres_variant(variant)?;
                self.pending_gmres.variant = Some(pv);
                if let Some(sstep) = sstep {
                    self.pending_gmres.sstep = Some(sstep);
                }
            }
            if let Some(sstep) = opts.gmres_sstep {
                self.pending_gmres.sstep = Some(sstep);
            }
            if let Some(cond) = opts.gmres_sstep_max_cond {
                self.pending_gmres.sstep_max_cond = Some(cond);
            }
        } else {
            if let Some(r) = opts.effective_restart_for(KspType::GMRES) {
                self.pending_gmres.restart = Some(r);
                self.restart = r;
            }
            if let Some(ref orth) = opts.gmres_orthog {
                self.pending_gmres.orthog = Some(match orth.as_str() {
                    "mgs" | "modified" => crate::solver::gmres::GmresOrthog::Mgs,
                    "cgs" | "classical" => crate::solver::gmres::GmresOrthog::Cgs,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_gmres_orthog: {other} (expected 'mgs'|'modified'|'cgs'|'classical')"
                        )));
                    }
                });
            }
            if let Some(ref mode) = opts.gmres_reorth {
                self.pending_gmres.reorth = Some(Self::parse_reorth_policy(mode)?);
            } else if let Some(flag) = opts.gmres_reorthog {
                self.pending_gmres.reorth = Some(if flag {
                    ReorthPolicy::Always
                } else {
                    ReorthPolicy::Never
                });
            }
            if let Some(tol) = opts.gmres_reorth_tol {
                self.pending_gmres.reorth_tol = Some(tol);
            }
            if let Some(flag) = opts.gmres_happy_breakdown {
                self.pending_gmres.happy_breakdown = Some(flag);
            }
            if let Some(ref variant) = opts.gmres_variant {
                let (pv, sstep) = Self::parse_gmres_variant(variant)?;
                self.pending_gmres.variant = Some(pv);
                if let Some(sstep) = sstep {
                    self.pending_gmres.sstep = Some(sstep);
                }
            }
            if let Some(sstep) = opts.gmres_sstep {
                self.pending_gmres.sstep = Some(sstep);
            }
            if let Some(cond) = opts.gmres_sstep_max_cond {
                self.pending_gmres.sstep_max_cond = Some(cond);
            }
        }

        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<GmresSolver>())
        {
            let snapshot = self.pending_gmres.clone();
            Self::apply_gmres_pending(&snapshot, s);
        }

        // --- FGMRES options ---
        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<FgmresSolver>())
        {
            if let Some(r) = opts.effective_restart_for(KspType::FGMRES) {
                s.set_restart(r);
                self.restart = r;
                self.pending_fgmres.restart = Some(r);
            }
            // Map FGMRES orthogonalization labels to implemented algorithms.
            if let Some(ref orth) = opts.fgmres_orthog {
                let o = match orth.as_str() {
                    "cgs" | "classical" => crate::solver::fgmres::OrthogMethod::ClassicalGS,
                    "mgs" | "modified" => crate::solver::fgmres::OrthogMethod::ModifiedGS,
                    "cgs_refined" | "cgs-refined" | "refined" => {
                        s.set_cgs_refinement(crate::solver::fgmres::CgsRefinement::Always);
                        crate::solver::fgmres::OrthogMethod::ClassicalGS
                    }
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_orthog: {other} (expected 'cgs'|'classical'|'mgs'|'modified'|'cgs_refined')"
                        )));
                    }
                };
                s.set_orthog(o);
                self.pending_fgmres.orthog = Some(o);
            }
            if let Some(ref mode) = opts.fgmres_reorth {
                let policy = Self::parse_reorth_policy(mode)?;
                s.set_reorth_policy(policy);
                self.pending_fgmres.reorth = Some(policy);
            } else if let Some(flag) = opts.fgmres_reorthog {
                s.set_reorthog(flag);
                self.pending_fgmres.reorth = Some(if flag {
                    ReorthPolicy::Always
                } else {
                    ReorthPolicy::Never
                });
            }
            if let Some(tol) = opts.fgmres_reorth_tol {
                s.set_reorth_tol(tol);
                self.pending_fgmres.reorth_tol = Some(tol);
            }
            if let Some(flag) = opts.fgmres_happy_breakdown {
                s.set_happy_breakdown(flag);
                self.pending_fgmres.happy_breakdown = Some(flag);
            }
            if let Some(ref variant) = opts.fgmres_variant {
                let v = match variant.as_str() {
                    "classical" => crate::solver::fgmres::FgmresVariant::Classical,
                    "pipelined" => crate::solver::fgmres::FgmresVariant::Pipelined,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_variant: {other} (expected 'classical'|'pipelined')"
                        )));
                    }
                };
                s.set_variant(v);
                self.pending_fgmres.variant = Some(v);
            }
        } else {
            if let Some(r) = opts.effective_restart_for(KspType::FGMRES) {
                self.pending_fgmres.restart = Some(r);
                self.restart = r;
            }
            if let Some(ref orth) = opts.fgmres_orthog {
                self.pending_fgmres.orthog = Some(match orth.as_str() {
                    "cgs" | "classical" => crate::solver::fgmres::OrthogMethod::ClassicalGS,
                    "mgs" | "modified" => crate::solver::fgmres::OrthogMethod::ModifiedGS,
                    "cgs_refined" | "cgs-refined" | "refined" => {
                        self.pending_fgmres.reorth = Some(ReorthPolicy::Always);
                        crate::solver::fgmres::OrthogMethod::ClassicalGS
                    }
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_orthog: {other} (expected 'cgs'|'classical'|'mgs'|'modified'|'cgs_refined')"
                        )));
                    }
                });
            }
            if let Some(ref mode) = opts.fgmres_reorth {
                self.pending_fgmres.reorth = Some(Self::parse_reorth_policy(mode)?);
            } else if let Some(flag) = opts.fgmres_reorthog {
                self.pending_fgmres.reorth = Some(if flag {
                    ReorthPolicy::Always
                } else {
                    ReorthPolicy::Never
                });
            }
            if let Some(tol) = opts.fgmres_reorth_tol {
                self.pending_fgmres.reorth_tol = Some(tol);
            }
            if let Some(flag) = opts.fgmres_happy_breakdown {
                self.pending_fgmres.happy_breakdown = Some(flag);
            }
            if let Some(ref variant) = opts.fgmres_variant {
                self.pending_fgmres.variant = Some(match variant.as_str() {
                    "classical" => crate::solver::fgmres::FgmresVariant::Classical,
                    "pipelined" => crate::solver::fgmres::FgmresVariant::Pipelined,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_variant: {other} (expected 'classical'|'pipelined')"
                        )));
                    }
                });
            }
        }

        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<CgSolver>())
        {
            if let Some(variant) = requested_cg_variant {
                s.set_variant(variant);
            }
            if let Some(ref norm) = opts.cg_norm {
                let n = match norm.as_str() {
                    "precond" => crate::solver::cg::CgNormType::Preconditioned,
                    "unprecond" => crate::solver::cg::CgNormType::Unpreconditioned,
                    "natural" => crate::solver::cg::CgNormType::Natural,
                    "none" => crate::solver::cg::CgNormType::None,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_cg_norm: {other}"
                        )));
                    }
                };
                s.set_norm(n);
            }
            if let Some(r) = opts.trust_region {
                s.set_trust_region(r);
            }
            if let Some(flag) = opts.cg_use_async {
                s.set_async_enabled(flag);
            }
            if let Some(min_n) = opts.cg_async_min_n {
                s.set_async_min_n(min_n);
            }
            if let Some(repl) = opts.cg_replace_every {
                s.set_pipelined_residual_refresh_every(Some(repl));
            }
        }
        let mut pcg_pending_updated = false;
        if let Some(variant) = requested_cg_variant {
            if matches!(variant, CgVariant::Pipelined)
                && Self::effective_side_for_solver(self.pc_side, SolverType::Pcg) != PcSide::Left
            {
                return Err(KError::InvalidInput(
                    "Pipelined PCG requires left preconditioning".into(),
                ));
            }
            self.pending_pcg.pipelined = Some(matches!(variant, CgVariant::Pipelined));
            pcg_pending_updated = true;
        }
        if let Some(repl) = opts.cg_replace_every {
            self.pending_pcg.replace_every = Some(repl);
            pcg_pending_updated = true;
        }
        if pcg_pending_updated {
            let snapshot = self.pending_pcg.clone();
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|b| b.as_any_mut().downcast_mut::<PcgSolver>())
            {
                Self::apply_pcg_pending(&snapshot, s);
            }
        }
        let mut bicg_pending_updated = false;
        if let Some(ref variant) = opts.bicgstab_variant {
            let parsed = match variant.as_str() {
                "classic" => BiCgStabVariant::Classic,
                "fewerchecks" | "fewer_checks" | "fewer-checks" | "lowsync" | "low_sync"
                | "low-sync" => BiCgStabVariant::FewerChecks,
                "reliable" => BiCgStabVariant::Reliable {
                    residual_replace_every: self
                        .pending_bicgstab
                        .replace_every
                        .unwrap_or(32)
                        .max(1),
                },
                other => {
                    return Err(KError::SolveError(format!(
                        "Unrecognized ksp_bicgstab_variant: {other} (expected 'classic'|'fewerchecks'|'reliable'; legacy aliases: 'lowsync'|'low_sync'|'low-sync')"
                    )));
                }
            };
            self.pending_bicgstab.variant = Some(parsed);
            bicg_pending_updated = true;
        }
        if let Some(repl) = opts.bicgstab_replace_every {
            self.pending_bicgstab.replace_every = Some(repl);
            if self.pending_bicgstab.variant.is_none() {
                self.pending_bicgstab.variant = Some(BiCgStabVariant::Reliable {
                    residual_replace_every: repl,
                });
            }
            bicg_pending_updated = true;
        }
        if bicg_pending_updated {
            let snapshot = self.pending_bicgstab.clone();
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|b| b.as_any_mut().downcast_mut::<BiCgStabSolver>())
            {
                Self::apply_bicgstab_pending(&snapshot, s);
            }
        }
        self.invalidate_solver_setup();
        Ok(self)
    }

    fn apply_gmres_pending(pending: &PendingGmres, s: &mut GmresSolver) {
        if let Some(r) = pending.restart {
            s.set_restart(r);
        }
        if let Some(o) = pending.orthog {
            s.set_orthog(o);
        }
        if let Some(p) = pending.reorth {
            s.set_reorth_policy(p);
        }
        if let Some(tol) = pending.reorth_tol {
            s.set_reorth_tol(tol);
        }
        if let Some(f) = pending.happy_breakdown {
            s.set_happy_breakdown(f);
        }
        if let Some(policy) = pending.stagnation_policy {
            s.set_stagnation_policy(policy);
        }
        let mut variant_kind = pending.variant;
        if variant_kind.is_none()
            && matches!(s.variant, crate::solver::gmres::GmresVariant::SStep { .. })
            && (pending.sstep.is_some() || pending.sstep_max_cond.is_some())
        {
            variant_kind = Some(PendingGmresVariant::SStep);
        }

        if let Some(kind) = variant_kind {
            match kind {
                PendingGmresVariant::Classical => {
                    s.set_variant(crate::solver::gmres::GmresVariant::Classical);
                }
                PendingGmresVariant::Pipelined => {
                    s.set_variant(crate::solver::gmres::GmresVariant::Pipelined);
                }
                PendingGmresVariant::SStep => {
                    let current = match s.variant {
                        crate::solver::gmres::GmresVariant::SStep {
                            s,
                            reorth,
                            max_cond,
                        } => Some((s, reorth, max_cond)),
                        _ => None,
                    };
                    let block_s = pending
                        .sstep
                        .or_else(|| current.map(|(s, _, _)| s))
                        .unwrap_or(2);
                    let max_cond = pending
                        .sstep_max_cond
                        .or_else(|| current.map(|(_, _, cond)| cond))
                        .unwrap_or(1e8);
                    let reorth = pending
                        .reorth
                        .or_else(|| current.map(|(_, r, _)| r))
                        .unwrap_or_else(|| s.reorth_policy());
                    s.set_variant(crate::solver::gmres::GmresVariant::SStep {
                        s: block_s,
                        reorth,
                        max_cond,
                    });
                }
            }
        }
    }

    fn apply_gmres_pending_to(&self, s: &mut GmresSolver) {
        Self::apply_gmres_pending(&self.pending_gmres, s);
    }

    fn apply_fgmres_pending_to(&self, s: &mut FgmresSolver) {
        if let Some(r) = self.pending_fgmres.restart {
            s.set_restart(r);
        }
        if let Some(o) = self.pending_fgmres.orthog {
            s.set_orthog(o);
        }
        if let Some(p) = self.pending_fgmres.reorth {
            s.set_reorth_policy(p);
        }
        if let Some(tol) = self.pending_fgmres.reorth_tol {
            s.set_reorth_tol(tol);
        }
        if let Some(f) = self.pending_fgmres.happy_breakdown {
            s.set_happy_breakdown(f);
        }
        if let Some(v) = self.pending_fgmres.variant {
            s.set_variant(v);
        }
    }

    fn apply_bicgstab_pending(pending: &PendingBiCgStab, s: &mut BiCgStabSolver) {
        if let Some(variant) = pending.variant {
            s.set_variant(match variant {
                BiCgStabVariant::Reliable {
                    residual_replace_every: _,
                } => {
                    let replace_every = pending.replace_every.unwrap_or(32).max(1);
                    BiCgStabVariant::Reliable {
                        residual_replace_every: replace_every,
                    }
                }
                other => other,
            });
        } else if let Some(replace_every) = pending.replace_every {
            s.set_variant(BiCgStabVariant::Reliable {
                residual_replace_every: replace_every.max(1),
            });
        }
    }

    fn apply_pcg_pending(pending: &PendingPcg, s: &mut PcgSolver) {
        if let Some(flag) = pending.pipelined {
            if flag {
                let replace_every = pending
                    .replace_every
                    .unwrap_or(PCG_PIPELINED_DEFAULT_REPLACE_EVERY);
                s.set_variant(PcgVariant::Pipelined { replace_every });
            } else {
                s.set_variant(PcgVariant::Classic);
            }
        } else if matches!(s.variant(), PcgVariant::Pipelined { .. })
            && let Some(replace_every) = pending.replace_every
        {
            s.set_variant(PcgVariant::Pipelined { replace_every });
        }
    }

    fn apply_pcg_pending_to(&self, s: &mut PcgSolver) {
        Self::apply_pcg_pending(&self.pending_pcg, s);
    }

    /// Configure both KSP and PC from their respective option sets.
    pub fn set_from_all_options(
        &mut self,
        ksp_opts: &KspOptions,
        pc_opts: &PcOptions,
    ) -> Result<&mut Self, KError> {
        self.set_from_options(ksp_opts)?;
        if let Some(ref pct) = pc_opts.pc_type {
            let pct = PcType::from_str(pct)?;
            self.set_pc_type(pct, Some(pc_opts))?;
        }
        if let Some(ref pol) = pc_opts.reuse_policy {
            let pol = match pol.as_str() {
                "never" => PcReusePolicy::Never,
                "reuse_numeric" => PcReusePolicy::ReuseNumeric,
                _ => PcReusePolicy::Auto,
            };
            self.set_pc_reuse_policy(pol);
        }
        if let Some(ref side) = ksp_opts.pc_side {
            self.set_pc_side_from_str(side)?;
        }
        if let Some(ref chain_opts) = pc_opts.chain {
            self.set_pc_chain_candidates_from_options(vec![chain_opts.clone()])?;
        } else if let Some(ref chain_str) = pc_opts.pc_chain {
            let candidates =
                PcFactory::create_pc_chain_candidates_from_str(chain_str, Some(pc_opts))?;
            self.set_pc_chain_candidates_from_specs(candidates)?;
        }
        #[cfg(feature = "backend-faer")]
        {
            self.pending_mpi_pc = Some(PendingMpiPc {
                mpi_opts: pc_opts.mpi_pc_options()?,
                pc_opts: pc_opts.clone(),
            });
            self.dist_route_diag = DistRouteDiagnosticsState::default();
        }
        let diagnostics = self.view();
        if ksp_opts.ksp_view.unwrap_or(false) {
            println!("{}", diagnostics.to_json_pretty());
        }
        if pc_opts.pc_view.unwrap_or(false) {
            println!("{}", diagnostics.pc_view().to_json_pretty());
        }
        Ok(self)
    }

    /// Return structured diagnostics for the active KSP/PC configuration.
    pub fn view(&self) -> KspDiagnostics {
        let mut solver_config = BTreeMap::new();
        insert_value(&mut solver_config, "rtol", self.rtol);
        insert_value(&mut solver_config, "atol", self.atol);
        insert_value(&mut solver_config, "dtol", self.dtol);
        insert_value(&mut solver_config, "maxits", self.maxits);
        insert_value(&mut solver_config, "restart", self.restart);
        let effective_pc_side = self.effective_pc_side();
        insert_value(
            &mut solver_config,
            "pc_side_requested",
            format!("{:?}", self.pc_side),
        );
        insert_value(
            &mut solver_config,
            "pc_side_effective",
            format!("{:?}", effective_pc_side),
        );
        if effective_pc_side != self.pc_side {
            insert_value(
                &mut solver_config,
                "pc_side_note",
                "normalized to solver-compatible side",
            );
        }
        insert_value(
            &mut solver_config,
            "pc_reuse",
            format!("{:?}", self.pc_reuse),
        );
        insert_value(
            &mut solver_config,
            "monitor_policy",
            format!("{:?}", self.monitor_policy),
        );
        insert_value(
            &mut solver_config,
            "reduction_mode",
            format!("{:?}", self.reduction_opts.mode),
        );
        insert_value(
            &mut solver_config,
            "reduction_exec",
            format!("{:?}", self.reduction_opts.exec),
        );
        insert_value(
            &mut solver_config,
            "reduction_max_inflight",
            self.reduction_opts.max_inflight,
        );
        insert_value(
            &mut solver_config,
            "reduction_reproducible",
            self.reduction_opts.reproducible,
        );
        insert_value(&mut solver_config, "reproducible", self.reproducible);
        insert_value(
            &mut solver_config,
            "execution_policy",
            format!("{:?}", self.exec),
        );
        insert_value(
            &mut solver_config,
            "execution_stage_threads",
            self.scoped_threads.diagnostics(),
        );
        insert_value(
            &mut solver_config,
            "execution_stage_threads_effective",
            self.scoped_threads.diagnostics(),
        );
        if let Some(adaptive) = &self.adaptive_exec {
            insert_value(
                &mut solver_config,
                "execution_policy_auto",
                adaptive.auto_report.concise(),
            );
        }
        if let Some(adaptive) = &self.adaptive_exec {
            insert_value(&mut solver_config, "adaptive_threading", adaptive.threading);
            insert_value(
                &mut solver_config,
                "adaptive_recommended_threads",
                adaptive.recommended_threads,
            );
            insert_value(
                &mut solver_config,
                "adaptive_threading_reason",
                adaptive.threading_reason,
            );
            insert_value(
                &mut solver_config,
                "adaptive_reduction_selected",
                format!("{:?}", adaptive.selected_reduction),
            );
            insert_value(
                &mut solver_config,
                "adaptive_reduction_exec",
                format!("{:?}", adaptive.reduction_exec),
            );
            insert_value(
                &mut solver_config,
                "adaptive_overlap",
                format!("{:?}", adaptive.overlap),
            );
            insert_value(
                &mut solver_config,
                "adaptive_variant",
                format!("{:?}", adaptive.variant),
            );
            insert_value(
                &mut solver_config,
                "adaptive_reduction_latency_us",
                adaptive.reduction_latency_us,
            );
            insert_value(
                &mut solver_config,
                "adaptive_tuner_mode",
                format!("{:?}", adaptive.tune_decision.mode),
            );
            insert_value(
                &mut solver_config,
                "adaptive_tuner_rationale",
                adaptive.tune_decision.rationale,
            );
            insert_value(
                &mut solver_config,
                "adaptive_thresholds_baseline",
                serde_json::json!({
                    "min_len_vec": adaptive.tune_decision.baseline.min_len_vec,
                    "min_rows_spmv": adaptive.tune_decision.baseline.min_rows_spmv,
                    "min_rows_ilu_factorization": adaptive.tune_decision.baseline.min_rows_ilu_factorization,
                    "min_rows_ilu_triangular": adaptive.tune_decision.baseline.min_rows_ilu_triangular,
                    "min_rows_asm_apply": adaptive.tune_decision.baseline.min_rows_asm_apply,
                }),
            );
            insert_value(
                &mut solver_config,
                "adaptive_thresholds_selected",
                serde_json::json!({
                    "min_len_vec": adaptive.tune_decision.selected.min_len_vec,
                    "min_rows_spmv": adaptive.tune_decision.selected.min_rows_spmv,
                    "min_rows_ilu_factorization": adaptive.tune_decision.selected.min_rows_ilu_factorization,
                    "min_rows_ilu_triangular": adaptive.tune_decision.selected.min_rows_ilu_triangular,
                    "min_rows_asm_apply": adaptive.tune_decision.selected.min_rows_asm_apply,
                }),
            );
            insert_value(
                &mut solver_config,
                "adaptive_kernel_timing_ns_per_elem",
                serde_json::json!({
                    "serial": adaptive.tune_decision.kernel_timing.serial_ns_per_elem,
                    "parallel": adaptive.tune_decision.kernel_timing.parallel_ns_per_elem,
                    "serial_samples": adaptive.tune_decision.kernel_timing.serial_samples,
                    "parallel_samples": adaptive.tune_decision.kernel_timing.parallel_samples,
                }),
            );
            if let Some(s) = adaptive.sstep_block {
                insert_value(&mut solver_config, "adaptive_sstep_block", s);
            }
        }

        #[cfg(feature = "backend-faer")]
        if let Some(pending) = self.pending_mpi_pc.as_ref() {
            let warning_codes = Self::distributed_setting_warnings(
                pending.mpi_opts.route_policy,
                pending.mpi_opts.local_apply_mode,
            );
            let mut effective_dist_config = BTreeMap::new();
            insert_value(
                &mut effective_dist_config,
                "global_pc",
                format!("{:?}", pending.mpi_opts.global_pc),
            );
            insert_value(
                &mut effective_dist_config,
                "local_pc",
                format!("{:?}", pending.mpi_opts.local_pc),
            );
            insert_value(
                &mut effective_dist_config,
                "route_policy",
                format!("{:?}", pending.mpi_opts.route_policy),
            );
            insert_value(
                &mut effective_dist_config,
                "local_apply_mode",
                pending
                    .mpi_opts
                    .local_apply_mode
                    .communication_strategy_name(),
            );
            insert_value(
                &mut effective_dist_config,
                "local_apply_requires_native",
                pending.mpi_opts.local_apply_mode.requires_native(),
            );
            insert_value(
                &mut effective_dist_config,
                "local_apply_is_native",
                pending.mpi_opts.local_apply_mode.is_distributed_native(),
            );
            insert_value(
                &mut effective_dist_config,
                "warnings",
                warning_codes.clone(),
            );
            insert_value(
                &mut solver_config,
                "pc_dist_route_policy",
                format!("{:?}", pending.mpi_opts.route_policy),
            );
            insert_value(
                &mut solver_config,
                "pc_dist_local_apply_effective",
                pending
                    .mpi_opts
                    .local_apply_mode
                    .communication_strategy_name(),
            );
            insert_value(
                &mut solver_config,
                "pc_dist_option_warnings",
                warning_codes.clone(),
            );
            insert_value(
                &mut solver_config,
                "pc_dist_effective_config",
                effective_dist_config.clone(),
            );
            insert_value(
                &mut solver_config,
                "pc_dist_selected_route",
                self.dist_route_diag
                    .selected_route
                    .clone()
                    .unwrap_or_else(|| "unresolved".to_string()),
            );
            if let Some(report) = self.dist_route_diag.decision_report.as_ref() {
                insert_value(&mut solver_config, "pc_dist_route_decision_report", report);
            }
            if let Some(entry) = self.dist_route_diag.capability_entry.as_ref() {
                insert_value(&mut solver_config, "pc_dist_capability_entry", entry);
                insert_value(
                    &mut solver_config,
                    "pc_dist_native_distributed_supported",
                    entry.supports_native_distributed_mode,
                );
                insert_value(
                    &mut solver_config,
                    "pc_dist_adapter_distributed_supported",
                    entry.supports_adapter_distributed_mode,
                );
                insert_value(
                    &mut solver_config,
                    "pc_dist_requested_distributed_mode",
                    entry.requested_distributed_mode,
                );
            }
            if let Some(mode) =
                Self::distributed_mode_family(self.dist_route_diag.selected_route.as_deref())
            {
                insert_value(&mut solver_config, "pc_dist_negotiated_mode", mode);
            }
            insert_value(
                &mut solver_config,
                "pc_dist_fallback_chain",
                self.dist_route_diag.fallback_chain.clone(),
            );
            insert_value(
                &mut solver_config,
                "pc_dist_fallback_counters",
                self.dist_route_diag.fallback_counters.clone(),
            );
            if let Some(reason) = self.dist_route_diag.fallback_reason.clone() {
                insert_value(&mut solver_config, "pc_dist_fallback_reason", reason);
            }
            if let Some(preflight) = self.dist_route_diag.preflight.as_ref() {
                insert_value(
                    &mut solver_config,
                    "pc_dist_preflight_outcome",
                    preflight.outcome.clone(),
                );
                insert_value(
                    &mut solver_config,
                    "pc_dist_preflight_reason_codes",
                    preflight.reason_codes.clone(),
                );
                insert_value(
                    &mut solver_config,
                    "pc_dist_preflight_native_ready",
                    preflight.native_ready,
                );
                insert_value(
                    &mut solver_config,
                    "pc_dist_preflight_cached_hits",
                    preflight.cached_hits,
                );
            }
            if !self.dist_route_diag.replay_tokens.is_empty() {
                insert_value(
                    &mut solver_config,
                    "pc_dist_replay_tokens",
                    self.dist_route_diag.replay_tokens.clone(),
                );
            }
        }

        let pc = self
            .pc_spec
            .as_ref()
            .or(self.pending_pc.as_ref())
            .map(|spec| {
                let mut diag =
                    PcDiagnostics::from_options(Some(spec.pc_type), spec.options.as_ref());
                #[cfg(feature = "backend-faer")]
                if let Some(pending) = self.pending_mpi_pc.as_ref() {
                    let warning_codes = Self::distributed_setting_warnings(
                        pending.mpi_opts.route_policy,
                        pending.mpi_opts.local_apply_mode,
                    );
                    let mut effective_dist_config = BTreeMap::new();
                    insert_value(
                        &mut effective_dist_config,
                        "global_pc",
                        format!("{:?}", pending.mpi_opts.global_pc),
                    );
                    insert_value(
                        &mut effective_dist_config,
                        "local_pc",
                        format!("{:?}", pending.mpi_opts.local_pc),
                    );
                    insert_value(
                        &mut effective_dist_config,
                        "route_policy",
                        format!("{:?}", pending.mpi_opts.route_policy),
                    );
                    insert_value(
                        &mut effective_dist_config,
                        "local_apply_mode",
                        pending
                            .mpi_opts
                            .local_apply_mode
                            .communication_strategy_name(),
                    );
                    insert_value(
                        &mut effective_dist_config,
                        "warnings",
                        warning_codes.clone(),
                    );
                    insert_value(
                        &mut diag.config,
                        "pc_dist_selected_route",
                        self.dist_route_diag
                            .selected_route
                            .clone()
                            .unwrap_or_else(|| "unresolved".to_string()),
                    );
                    if let Some(report) = self.dist_route_diag.decision_report.as_ref() {
                        insert_value(&mut diag.config, "pc_dist_route_decision_report", report);
                    }
                    if let Some(entry) = self.dist_route_diag.capability_entry.as_ref() {
                        insert_value(&mut diag.config, "pc_dist_capability_entry", entry);
                        diag.native_distributed_supported =
                            Some(entry.supports_native_distributed_mode);
                        diag.adapter_distributed_supported =
                            Some(entry.supports_adapter_distributed_mode);
                    }
                    diag.distributed_mode = Self::distributed_mode_family(
                        self.dist_route_diag.selected_route.as_deref(),
                    )
                    .map(str::to_string);
                    insert_value(
                        &mut diag.config,
                        "pc_dist_fallback_chain",
                        self.dist_route_diag.fallback_chain.clone(),
                    );
                    insert_value(
                        &mut diag.config,
                        "pc_dist_fallback_counters",
                        self.dist_route_diag.fallback_counters.clone(),
                    );
                    insert_value(
                        &mut diag.config,
                        "pc_dist_local_apply_effective",
                        pending
                            .mpi_opts
                            .local_apply_mode
                            .communication_strategy_name(),
                    );
                    insert_value(
                        &mut diag.config,
                        "pc_dist_option_warnings",
                        warning_codes.clone(),
                    );
                    insert_value(
                        &mut diag.config,
                        "pc_dist_effective_config",
                        effective_dist_config.clone(),
                    );
                    if let Some(reason) = self.dist_route_diag.fallback_reason.clone() {
                        insert_value(&mut diag.config, "pc_dist_fallback_reason", reason);
                    }
                    if let Some(preflight) = self.dist_route_diag.preflight.as_ref() {
                        insert_value(
                            &mut diag.config,
                            "pc_dist_preflight_outcome",
                            preflight.outcome.clone(),
                        );
                        insert_value(
                            &mut diag.config,
                            "pc_dist_preflight_reason_codes",
                            preflight.reason_codes.clone(),
                        );
                        insert_value(
                            &mut diag.config,
                            "pc_dist_preflight_native_ready",
                            preflight.native_ready,
                        );
                        insert_value(
                            &mut diag.config,
                            "pc_dist_preflight_cached_hits",
                            preflight.cached_hits,
                        );
                    }
                }
                Box::new(diag)
            });
        let pc_chain = self
            .pc_chain_plan
            .as_ref()
            .map(|plan| {
                plan.active_specs()
                    .iter()
                    .map(|spec| {
                        let mut diag =
                            PcDiagnostics::from_options(Some(spec.pc_type), spec.options.as_ref());
                        #[cfg(feature = "backend-faer")]
                        if let Some(pending) = self.pending_mpi_pc.as_ref() {
                            let warning_codes = Self::distributed_setting_warnings(
                                pending.mpi_opts.route_policy,
                                pending.mpi_opts.local_apply_mode,
                            );
                            let mut effective_dist_config = BTreeMap::new();
                            insert_value(
                                &mut effective_dist_config,
                                "global_pc",
                                format!("{:?}", pending.mpi_opts.global_pc),
                            );
                            insert_value(
                                &mut effective_dist_config,
                                "local_pc",
                                format!("{:?}", pending.mpi_opts.local_pc),
                            );
                            insert_value(
                                &mut effective_dist_config,
                                "route_policy",
                                format!("{:?}", pending.mpi_opts.route_policy),
                            );
                            insert_value(
                                &mut effective_dist_config,
                                "local_apply_mode",
                                pending
                                    .mpi_opts
                                    .local_apply_mode
                                    .communication_strategy_name(),
                            );
                            insert_value(
                                &mut effective_dist_config,
                                "warnings",
                                warning_codes.clone(),
                            );
                            insert_value(
                                &mut diag.config,
                                "pc_dist_selected_route",
                                self.dist_route_diag
                                    .selected_route
                                    .clone()
                                    .unwrap_or_else(|| "unresolved".to_string()),
                            );
                            if let Some(report) = self.dist_route_diag.decision_report.as_ref() {
                                insert_value(
                                    &mut diag.config,
                                    "pc_dist_route_decision_report",
                                    report,
                                );
                            }
                            if let Some(entry) = self.dist_route_diag.capability_entry.as_ref() {
                                insert_value(&mut diag.config, "pc_dist_capability_entry", entry);
                                diag.native_distributed_supported =
                                    Some(entry.supports_native_distributed_mode);
                                diag.adapter_distributed_supported =
                                    Some(entry.supports_adapter_distributed_mode);
                            }
                            diag.distributed_mode = Self::distributed_mode_family(
                                self.dist_route_diag.selected_route.as_deref(),
                            )
                            .map(str::to_string);
                            insert_value(
                                &mut diag.config,
                                "pc_dist_fallback_chain",
                                self.dist_route_diag.fallback_chain.clone(),
                            );
                            insert_value(
                                &mut diag.config,
                                "pc_dist_fallback_counters",
                                self.dist_route_diag.fallback_counters.clone(),
                            );
                            insert_value(
                                &mut diag.config,
                                "pc_dist_local_apply_effective",
                                pending
                                    .mpi_opts
                                    .local_apply_mode
                                    .communication_strategy_name(),
                            );
                            insert_value(
                                &mut diag.config,
                                "pc_dist_option_warnings",
                                warning_codes.clone(),
                            );
                            insert_value(
                                &mut diag.config,
                                "pc_dist_effective_config",
                                effective_dist_config.clone(),
                            );
                            if let Some(reason) = self.dist_route_diag.fallback_reason.clone() {
                                insert_value(&mut diag.config, "pc_dist_fallback_reason", reason);
                            }
                            if let Some(preflight) = self.dist_route_diag.preflight.as_ref() {
                                insert_value(
                                    &mut diag.config,
                                    "pc_dist_preflight_outcome",
                                    preflight.outcome.clone(),
                                );
                                insert_value(
                                    &mut diag.config,
                                    "pc_dist_preflight_reason_codes",
                                    preflight.reason_codes.clone(),
                                );
                                insert_value(
                                    &mut diag.config,
                                    "pc_dist_preflight_native_ready",
                                    preflight.native_ready,
                                );
                                insert_value(
                                    &mut diag.config,
                                    "pc_dist_preflight_cached_hits",
                                    preflight.cached_hits,
                                );
                            }
                        }
                        diag
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|chain| !chain.is_empty());

        KspDiagnostics {
            solver_type: self.solver_type.map(|st| format!("{st:?}")),
            solver_config,
            pc,
            pc_chain,
            setup_called: self.setup_called,
            bound_comm_id: self.bound_comm.as_ref().map(|comm| comm.id()),
            last_converged_reason: self.last_converged_reason.map(|r| format!("{r:?}")),
            last_converged_reason_petsc: self
                .last_converged_reason
                .map(|r| r.petsc_reason().to_string()),
            reason_counters_breakdown: self.reason_counters.breakdown,
            reason_counters_nan: self.reason_counters.nan,
            reason_counters_inf: self.reason_counters.inf,
            reason_counters_pc_setup: self.reason_counters.pc_setup,
            reason_counters_pc_apply: self.reason_counters.pc_apply,
        }
    }

    /// Assign the system and preconditioner operators.
    ///
    /// Returns an error if the communicators of `A` and `P` are not congruent.
    /// `LinOp::comm()` is the single source of truth for parallel context;
    /// mismatches indicate a caller bug.
    ///
    /// ## Reuse contract
    /// - `StructureId` changes trigger symbolic rebuilds (pattern/dimensions).
    /// - `ValuesId` changes trigger numeric rebuilds when supported/allowed.
    /// - If both IDs are unchanged, `setup()` is a cheap no-op aside from
    ///   workspace sizing and bookkeeping.
    /// - `StructureId(0)` / `ValuesId(0)` mean \"unknown\" and force conservative
    ///   refreshes. Use wrappers like `CsrOp` / `DenseOp` and call
    ///   `mark_structure_changed` / `mark_values_changed` after in-place edits.
    ///
    /// ## Communicator binding
    /// `KspContext` becomes bound to the operator communicator once operators
    /// are set. To override or wrap operator communicators, use
    /// [`try_set_operators_with_comm`].
    ///
    /// On success, invalidates any prior setup (PC reuse and workspace).
    pub fn try_set_operators(
        &mut self,
        amat: Arc<dyn LinOp<S = S>>,
        pmat: Option<Arc<dyn LinOp<S = S>>>,
    ) -> Result<&mut Self, KError> {
        let pmat = pmat.unwrap_or_else(|| amat.clone());
        let ac = amat.comm();
        let pc = pmat.comm();
        if !ac.congruent(&pc) {
            return Err(KError::InvalidInput(format!(
                "Amat/Pmat communicator mismatch (not congruent): A={}, P={}",
                ac.id(),
                pc.id()
            )));
        }
        self.bind_or_check_comm(&ac)?;

        let a_dims = amat.dims();
        let p_dims = pmat.dims();
        if a_dims != p_dims {
            return Err(KError::InvalidInput(format!(
                "Amat/Pmat dimension mismatch: A={:?}, P={:?}",
                a_dims, p_dims
            )));
        }
        if let (Some(a_layout), Some(p_layout)) = (amat.dist_layout(), pmat.dist_layout()) {
            if a_layout.global_rows != p_layout.global_rows
                || a_layout.global_cols != p_layout.global_cols
            {
                return Err(KError::InvalidInput(format!(
                    "Amat/Pmat global dimension mismatch: A=({}x{}), P=({}x{})",
                    a_layout.global_rows,
                    a_layout.global_cols,
                    p_layout.global_rows,
                    p_layout.global_cols
                )));
            }
            if a_layout.row_start != p_layout.row_start || a_layout.row_end != p_layout.row_end {
                return Err(KError::InvalidInput(format!(
                    "Amat/Pmat ownership range mismatch: A=[{},{}), P=[{},{}).",
                    a_layout.row_start, a_layout.row_end, p_layout.row_start, p_layout.row_end
                )));
            }
        }
        #[cfg(feature = "invariants")]
        log::debug!(
            "set_operators: comm_id={} size={} rank={} A_dims={:?} P_dims={:?} A_ids=({:?},{:?}) P_ids=({:?},{:?})",
            ac.id(),
            ac.size(),
            ac.rank(),
            a_dims,
            p_dims,
            amat.structure_id(),
            amat.values_id(),
            pmat.structure_id(),
            pmat.values_id()
        );
        self.amat = Some(amat);
        self.pmat = Some(pmat);
        self.invalidate_pc_setup();
        Ok(self)
    }

    /// Like `try_set_operators`, but first wraps operators with an explicit communicator.
    pub fn try_set_operators_with_comm(
        &mut self,
        amat: Arc<dyn LinOp<S = S>>,
        pmat: Option<Arc<dyn LinOp<S = S>>>,
        comm: crate::parallel::UniverseComm,
    ) -> Result<&mut Self, KError> {
        self.bind_or_check_comm(&comm)?;

        let a_base = amat.comm();
        if !a_base.is_trivial() && !a_base.congruent(&comm) {
            return Err(KError::InvalidInput(format!(
                "Cannot override nontrivial Amat communicator: base={}, requested={}",
                a_base.id(),
                comm.id()
            )));
        }

        if let Some(ref p) = pmat {
            let p_base = p.comm();
            if !p_base.is_trivial() && !p_base.congruent(&comm) {
                return Err(KError::InvalidInput(format!(
                    "Cannot override nontrivial Pmat communicator: base={}, requested={}",
                    p_base.id(),
                    comm.id()
                )));
            }
        }

        let a_wrapped = wrap_with_comm(amat, comm.clone());
        let p_wrapped = pmat.map(|p| wrap_with_comm(p, comm.clone()));
        self.try_set_operators(a_wrapped, p_wrapped)
    }

    /// Assign the system and preconditioner operators.
    ///
    /// Panics if the communicators of `A` and `P` differ. Prefer
    /// [`KspContext::try_set_operators`] in libraries to handle errors.
    pub fn set_operators(
        &mut self,
        amat: Arc<dyn LinOp<S = S>>,
        pmat: Option<Arc<dyn LinOp<S = S>>>,
    ) -> &mut Self {
        self.try_set_operators(amat, pmat).unwrap()
    }

    /// Like `set_operators`, but first wraps operators with an explicit communicator.
    /// Panics on communicator mismatch. Prefer
    /// [`KspContext::try_set_operators_with_comm`].
    pub fn set_operators_with_comm(
        &mut self,
        amat: Arc<dyn LinOp<S = S>>,
        pmat: Option<Arc<dyn LinOp<S = S>>>,
        comm: crate::parallel::UniverseComm,
    ) -> &mut Self {
        self.try_set_operators_with_comm(amat, pmat, comm).unwrap()
    }

    pub fn set_pc_reuse_policy(&mut self, policy: PcReusePolicy) -> &mut Self {
        self.pc_reuse = policy;
        self.invalidate_solver_setup();
        self
    }

    fn reset_pc_ids(&mut self) {
        self.last_pc_sid = None;
        self.last_pc_vid = None;
    }

    fn tune_ilutp_options(opts: &mut PcOptions, attempt: usize) -> bool {
        let mut changed = false;
        let current_fill = opts.ilutp_max_fill.unwrap_or(10);
        let next_fill = (current_fill.saturating_mul(2)).min(200);
        if next_fill > current_fill {
            opts.ilutp_max_fill = Some(next_fill);
            changed = true;
        }

        let current_perm = opts.ilutp_perm_tol.unwrap_or(0.0);
        let next_perm = match attempt {
            1 if current_perm < 0.1 => 0.1,
            2 if current_perm < 0.2 => 0.2,
            _ if current_perm < 0.5 => 0.5,
            _ => current_perm,
        };
        if next_perm > current_perm {
            opts.ilutp_perm_tol = Some(next_perm);
            changed = true;
        }

        if opts.ilu_reordering.is_none() {
            opts.ilu_reordering = Some("rcm".to_string());
            changed = true;
        }

        changed
    }

    fn tune_ilutp_options_in_chain(specs: &mut [DeferredPcInfo], attempt: usize) -> bool {
        let mut changed = false;
        for spec in specs {
            if spec.pc_type == PcType::Ilutp {
                let opts = spec.options.get_or_insert_with(PcOptions::default);
                if Self::tune_ilutp_options(opts, attempt) {
                    changed = true;
                }
            }
        }
        changed
    }

    fn try_setup_chain_plan(
        &mut self,
        pmat: Arc<dyn LinOp<S = S>>,
        sid: StructureId,
        vid: ValuesId,
    ) -> Result<(), KError> {
        let Some(plan) = self.pc_chain_plan.as_mut() else {
            return Ok(());
        };

        let mut last_err = None;
        while plan.active < plan.candidates.len() {
            let mut specs = plan.active_specs().to_vec();
            let mut attempt = 0usize;
            loop {
                let mut chain =
                    match PcFactory::construct_deferred_pc_chain(specs.clone(), pmat.as_ref()) {
                        Ok(chain) => chain,
                        Err(err) => {
                            last_err = Some(err);
                            break;
                        }
                    };
                let want = chain.required_format();
                let tol = chain.preferred_drop_tol_for_format().unwrap_or_default();
                let pmat_view = materialize(pmat.clone(), want, tol)?;
                match chain.setup(pmat_view.as_ref()) {
                    Ok(()) => {
                        self.pc = Some(chain);
                        plan.candidates[plan.active] = specs;
                        self.last_pc_sid = Some(sid);
                        self.last_pc_vid = Some(vid);
                        return Ok(());
                    }
                    Err(err) => {
                        last_err = Some(err);
                        attempt += 1;
                        let mut tuned = specs.clone();
                        if attempt <= 3 && Self::tune_ilutp_options_in_chain(&mut tuned, attempt) {
                            log::warn!(
                                "PC chain setup failed; retrying with tuned ILUTP options (attempt {attempt})"
                            );
                            specs = tuned;
                            continue;
                        }
                        break;
                    }
                }
            }

            if !plan.advance() {
                break;
            }
            if let Some(ref err) = last_err {
                log::warn!("PC chain setup failed: {err}; falling back to next candidate");
            }
        }

        Err(last_err.unwrap_or_else(|| {
            KError::SolveError("PC chain setup failed: no viable candidates".into())
        }))
    }

    fn retry_ilutp_setup(
        &mut self,
        pmat: Arc<dyn LinOp<S = S>>,
        sid: StructureId,
        vid: ValuesId,
        err: KError,
    ) -> Result<(), KError> {
        let Some(mut spec) = self.pc_spec.clone() else {
            return Err(err);
        };
        if spec.pc_type != PcType::Ilutp {
            return Err(err);
        }

        for attempt in 1..=3 {
            let opts = spec.options.get_or_insert_with(PcOptions::default);
            if !Self::tune_ilutp_options(opts, attempt) {
                break;
            }
            log::warn!("ILUTP setup failed; retrying with tuned options (attempt {attempt})");
            let mut pc = PcFactory::construct_deferred_preconditioner(spec.clone(), pmat.as_ref())?;
            let want = pc.required_format();
            let tol = pc.preferred_drop_tol_for_format().unwrap_or_default();
            let pmat_view = materialize(pmat.clone(), want, tol)?;
            if pc.setup(pmat_view.as_ref()).is_ok() {
                self.pc = Some(pc);
                self.pc_spec = Some(spec);
                self.last_pc_sid = Some(sid);
                self.last_pc_vid = Some(vid);
                return Ok(());
            }
        }

        Err(err)
    }

    fn handle_pc_setup_failure(
        &mut self,
        err: KError,
        pmat: Arc<dyn LinOp<S = S>>,
        sid: StructureId,
        vid: ValuesId,
    ) -> Result<(), KError> {
        if self.pc_chain_plan.is_some() {
            self.pc = None;
            self.reset_pc_ids();
            return self
                .try_setup_chain_plan(pmat.clone(), sid, vid)
                .map_err(|fallback_err| {
                    KError::SolveError(format!(
                        "PC setup failed: {err}; fallback failed: {fallback_err}"
                    ))
                });
        }
        if self.retry_ilutp_setup(pmat, sid, vid, err.clone()).is_ok() {
            return Ok(());
        }
        Err(err)
    }

    pub fn last_pc_sid(&self) -> Option<StructureId> {
        self.last_pc_sid
    }
    pub fn last_pc_vid(&self) -> Option<ValuesId> {
        self.last_pc_vid
    }

    /// Prepare preconditioner and workspace.
    pub fn setup(&mut self) -> Result<(), KError> {
        let exec = self.exec.clone();
        let scoped = self.scoped_threads.clone();
        scoped.install_stage(KspExecStage::OuterSetup, || {
            exec.install(|| self.setup_impl())
        })
    }

    fn setup_impl(&mut self) -> Result<(), KError> {
        let pmat = self
            .pmat
            .as_ref()
            .cloned()
            .ok_or_else(|| KError::InvalidInput("Pmat not set".into()))?;
        let amat = self
            .amat
            .as_ref()
            .cloned()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;

        let (m, n) = amat.dims();
        let (pm, pn) = pmat.dims();
        let allow_rect = matches!(
            self.solver_type,
            Some(SolverType::Cgnr | SolverType::Lsqr | SolverType::Lsmr)
        );
        if !allow_rect && m != n {
            return Err(KError::InvalidInput(format!(
                "Amat must be square: got {}x{}",
                m, n
            )));
        }
        if m != pm || n != pn {
            return Err(KError::InvalidInput(format!(
                "Amat/Pmat dimension mismatch during setup: A=({m},{n}), P=({pm},{pn})"
            )));
        }
        #[cfg(feature = "invariants")]
        {
            let comm = amat.comm();
            log::debug!(
                "setup start: comm_id={} size={} rank={} dims=({},{}) pc_dims=({},{}) pc_reuse={:?} solver={:?} pc_side_requested={:?} pc_side_effective={:?} A_ids=({:?},{:?}) P_ids=({:?},{:?})",
                comm.id(),
                comm.size(),
                comm.rank(),
                m,
                n,
                pm,
                pn,
                self.pc_reuse,
                self.solver_type,
                self.pc_side,
                self.effective_pc_side(),
                amat.structure_id(),
                amat.values_id(),
                pmat.structure_id(),
                pmat.values_id()
            );
        }

        if self.pc.is_none() {
            #[cfg(all(feature = "backend-faer", not(feature = "complex"), feature = "mpi"))]
            {
                if self.pending_pc.is_none() && self.pc_chain_plan.is_none() {
                    if let Some(ref pending) = self.pending_mpi_pc
                        && pending.mpi_opts.global_pc != GlobalPcKind::None
                    {
                        let dist_op = pmat
                            .as_any()
                            .downcast_ref::<DistCsrOp>()
                            .or_else(|| amat.as_any().downcast_ref::<DistCsrOp>());
                        if let Some(dist_op) = dist_op
                            && dist_op.comm().size() > 1
                        {
                            let pc = self.build_mpi_global_pc(pending, dist_op)?;
                            self.pc = Some(pc);
                        }
                    }
                }
            }
            if self.pc_chain_plan.is_none() {
                if let Some(spec) = self.pending_pc.take() {
                    let pc = PcFactory::construct_deferred_preconditioner(spec, pmat.as_ref())?;
                    self.pc = Some(pc);
                }
            }
        }

        let sid = {
            let id = pmat.structure_id();
            if id.0 != 0 {
                id
            } else {
                StructureId(Arc::as_ptr(&pmat) as *const () as usize as u64)
            }
        };
        let vid = pmat.values_id();

        if self.pc.is_none() {
            // no factory hook here; assume pc set elsewhere
            self.last_pc_sid = None;
            self.last_pc_vid = None;
        }

        if self.pc.is_none() {
            if self.pc_chain_plan.is_some() {
                self.try_setup_chain_plan(pmat.clone(), sid, vid)?;
            }
        }

        if let Some(pc) = self.pc.as_mut() {
            // Pre-convert once to the PC's requested format, preserving communicator.
            let want = pc.required_format();
            let tol = pc.preferred_drop_tol_for_format().unwrap_or_default();
            let pmat_view = materialize(pmat.clone(), want, tol)?;

            match self.last_pc_sid {
                None => {
                    if let Err(err) = pc.setup(pmat_view.as_ref()) {
                        return self.handle_pc_setup_failure(err, pmat.clone(), sid, vid);
                    }
                    self.last_pc_sid = Some(sid);
                    self.last_pc_vid = Some(vid);
                }
                Some(old_sid) if old_sid != sid => {
                    if let Err(err) = pc.update_symbolic(pmat_view.as_ref()) {
                        return self.handle_pc_setup_failure(err, pmat.clone(), sid, vid);
                    }
                    self.last_pc_sid = Some(sid);
                    self.last_pc_vid = Some(vid);
                }
                Some(_old_sid) => {
                    let vid_known = vid.0 != 0;
                    let values_changed = self.last_pc_vid != Some(vid);
                    match self.pc_reuse {
                        PcReusePolicy::Never => {
                            if !vid_known || values_changed {
                                if let Err(err) = pc.update_symbolic(pmat_view.as_ref()) {
                                    return self.handle_pc_setup_failure(
                                        err,
                                        pmat.clone(),
                                        sid,
                                        vid,
                                    );
                                }
                                self.last_pc_vid = Some(vid);
                            }
                        }
                        PcReusePolicy::ReuseNumeric => {
                            if pc.supports_numeric_update() {
                                if !vid_known {
                                    log::debug!(
                                        "ValuesId unknown; conservatively refreshing numeric data. Wrap your matrix in DenseOp/CsrOp and call mark_values_changed() to enable exact reuse."
                                    );
                                }
                                pc.update_numeric(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            } else if !vid_known || values_changed {
                                if let Err(err) = pc.update_symbolic(pmat_view.as_ref()) {
                                    return self.handle_pc_setup_failure(
                                        err,
                                        pmat.clone(),
                                        sid,
                                        vid,
                                    );
                                }
                                self.last_pc_vid = Some(vid);
                            }
                        }
                        PcReusePolicy::Auto => {
                            if (!vid_known || values_changed)
                                && pc.supports_numeric_update()
                                && self.pc_reuse.allow_numeric()
                            {
                                if !vid_known {
                                    log::debug!(
                                        "ValuesId unknown; conservatively refreshing numeric data. Wrap your matrix in DenseOp/CsrOp and call mark_values_changed() to enable exact reuse."
                                    );
                                }
                                pc.update_numeric(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            } else if !vid_known || values_changed {
                                if let Err(err) = pc.update_symbolic(pmat_view.as_ref()) {
                                    return self.handle_pc_setup_failure(
                                        err,
                                        pmat.clone(),
                                        sid,
                                        vid,
                                    );
                                }
                                self.last_pc_vid = Some(vid);
                            }
                        }
                    }
                }
            }
        }

        #[cfg(all(not(feature = "complex"), feature = "mpi"))]
        {
            let _ = self.maybe_upgrade_local_pc(pmat.clone(), sid, vid)?;
        }

        let (m, _) = amat.dims();
        let local_work = self
            .work
            .as_ref()
            .map(|w| w.local_work_estimate())
            .unwrap_or(m.saturating_mul(self.restart.max(1)));
        let adaptive = AdaptiveExecutionDecision::decide(
            m,
            amat.comm().size(),
            self.restart,
            local_work,
            reduction_latency_estimate_us(),
            self.reproducible,
            matches!(self.monitor_policy, MonitorPolicy::AllRanks) && self.monitors.len() > 1,
            &self.reduction_opts,
        );
        self.reduction_opts.mode = adaptive.selected_reduction;
        self.reduction_opts.exec = adaptive.reduction_exec;
        if matches!(adaptive.threading, "serial") {
            self.exec.threading = ThreadingPolicy::Serial;
        }
        if let Some(solver) = self.solver.as_mut() {
            match adaptive.variant {
                KrylovVariant::Classical => {
                    if let Some(gmres) = solver.as_any_mut().downcast_mut::<GmresSolver>() {
                        gmres.set_variant(crate::solver::gmres::GmresVariant::Classical);
                    }
                    if let Some(fgmres) = solver.as_any_mut().downcast_mut::<FgmresSolver>() {
                        fgmres.set_variant(crate::solver::fgmres::FgmresVariant::Classical);
                    }
                    if let Some(pcg) = solver.as_any_mut().downcast_mut::<PcgSolver>() {
                        pcg.set_variant(crate::solver::pcg::PcgVariant::Classic);
                    }
                }
                KrylovVariant::Pipelined => {
                    if let Some(gmres) = solver.as_any_mut().downcast_mut::<GmresSolver>() {
                        gmres.set_variant(crate::solver::gmres::GmresVariant::Pipelined);
                    }
                    if let Some(fgmres) = solver.as_any_mut().downcast_mut::<FgmresSolver>() {
                        fgmres.set_variant(crate::solver::fgmres::FgmresVariant::Pipelined);
                    }
                    if let Some(pcg) = solver.as_any_mut().downcast_mut::<PcgSolver>() {
                        pcg.set_variant(crate::solver::pcg::PcgVariant::Pipelined {
                            replace_every: crate::solver::pcg::PCG_PIPELINED_DEFAULT_REPLACE_EVERY,
                        });
                    }
                }
                KrylovVariant::SStep => {
                    if let Some(gmres) = solver.as_any_mut().downcast_mut::<GmresSolver>() {
                        gmres.set_variant(crate::solver::gmres::GmresVariant::SStep {
                            s: adaptive.sstep_block.unwrap_or(2),
                            reorth: gmres.reorth_policy(),
                            max_cond: 1e8,
                        });
                    }
                }
            }
        }
        self.adaptive_exec = Some(adaptive);

        let needs_new = self
            .work
            .as_ref()
            .map(|w| w.tmp1.len() != m)
            .unwrap_or(true);

        if needs_new {
            self.work = Some(Workspace::new(m));
        }

        if let Some(w) = self.work.as_mut() {
            // Keep workspace metadata in sync even when reusing allocations.
            w.set_reduction_options(self.reduction_opts.clone());
            w.set_reduction_engine(amat.comm().reduction_engine(&self.reduction_opts));

            // Critical fix: re-run solver-specific workspace setup every setup().
            if let Some(solver) = self.solver.as_mut() {
                solver.setup_workspace(w);
            }
        }
        self.setup_called = true;
        Ok(())
    }

    /// Solve the linear system using stored operators.
    pub fn solve(&mut self, b: &[S], x: &mut [S]) -> Result<SolveStats<R>, KError> {
        let exec = self.exec.clone();
        let scoped = self.scoped_threads.clone();
        scoped.install_stage(KspExecStage::OuterApply, || {
            exec.install(|| self.solve_impl(b, x))
        })
    }

    fn solve_impl(&mut self, b: &[S], x: &mut [S]) -> Result<SolveStats<R>, KError> {
        if !self.setup_called {
            if let Err(err) = self.setup_impl() {
                if let Some(reason) = ReasonEmitter::from_error(&err, FailureStage::Setup) {
                    let amat = self.amat.clone();
                    let res = if let Some(amat) = amat.as_ref() {
                        self.residual_norm_for_stats(amat.as_ref(), b, x)
                            .unwrap_or_else(|_| R::default())
                    } else {
                        R::default()
                    };
                    let mut stats = SolveStats::new(0, res, reason);
                    if let Some(failure) =
                        ReasonEmitter::nested_pc_failure(&err, FailureStage::Setup)
                    {
                        stats = stats.with_nested_pc_failure(failure);
                    }
                    self.apply_residual_contract_classification(
                        &mut stats,
                        amat.as_deref(),
                        b,
                        Some(&err),
                    )?;
                    let stats = stats.finalize_reason_counters();
                    self.last_converged_reason = Some(reason);
                    self.reason_counters.record_reason(reason);
                    if let Some(inner) = stats.nested_pc_failure.as_ref() {
                        self.reason_counters.record_reason(inner.reason);
                    }
                    return Ok(stats);
                }
                return Err(err);
            }
        }
        let amat = self
            .amat
            .clone()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;
        let pmat = self
            .pmat
            .clone()
            .ok_or_else(|| KError::InvalidInput("Pmat not set".into()))?;
        let (m, n) = amat.dims();
        let (pm, pn) = pmat.dims();

        let allow_rect = matches!(
            self.solver_type,
            Some(SolverType::Cgnr | SolverType::Lsqr | SolverType::Lsmr)
        );
        if !allow_rect && m != n {
            return Err(KError::InvalidInput(format!(
                "Amat must be square: got {}x{}",
                m, n
            )));
        }
        if m != pm || n != pn {
            return Err(KError::InvalidInput(format!(
                "Amat/Pmat dimension mismatch: A=({m},{n}), P=({pm},{pn})"
            )));
        }
        if b.len() != m {
            return Err(KError::InvalidInput(format!(
                "rhs length {} does not match operator rows {}",
                b.len(),
                m
            )));
        }
        if x.len() != n {
            return Err(KError::InvalidInput(format!(
                "solution length {} does not match operator cols {}",
                x.len(),
                n
            )));
        }
        #[cfg(feature = "invariants")]
        {
            let comm = amat.comm();
            log::debug!(
                "solve start: comm_id={} size={} rank={} dims=({},{}) rhs_len={} x_len={} solver={:?} pc_side_requested={:?} pc_side_effective={:?} pc_reuse={:?} A_ids=({:?},{:?}) P_ids=({:?},{:?})",
                comm.id(),
                comm.size(),
                comm.rank(),
                m,
                n,
                b.len(),
                x.len(),
                self.solver_type,
                self.pc_side,
                self.effective_pc_side(),
                self.pc_reuse,
                amat.structure_id(),
                amat.values_id(),
                pmat.structure_id(),
                pmat.values_id()
            );
        }
        // Configure solver preconditioning side, validating compatibility along the way.
        self.configure_pc_side()?;

        if matches!(self.solver_type, Some(SolverType::Preonly)) {
            let pmat = pmat.clone();

            {
                let pc = self.pc.as_mut().ok_or_else(|| {
                    KError::SolveError("PREONLY requires a direct PC (LU/QR/SuperLU_DIST)".into())
                })?;
                if !pc.supports_numeric_update() {
                    // Not a reliable indicator of directness, but provides a hint if a user
                    // accidentally selects a non-direct PC like Jacobi/ILU.
                    log::debug!(
                        "PREONLY: selected PC may not be a direct solver; expecting LU/QR/SuperLU_DIST."
                    );
                }
                pc.direct_solve(pmat.as_ref(), b, x)?;
            }

            let mat_for_residual: &dyn LinOp<S = S> = amat.as_ref();

            let res = self.true_residual_norm_in_place(mat_for_residual, b, x)?;
            self.last_converged_reason = Some(ConvergedReason::ConvergedAtol);
            self.reason_counters
                .record_reason(ConvergedReason::ConvergedAtol);
            return Ok(
                SolveStats::new(0, res, ConvergedReason::ConvergedAtol).finalize_reason_counters()
            );
        }

        let amat_ref = amat.as_ref();

        let monitors = if self.monitors.is_empty() {
            None
        } else {
            Some(self.monitors.as_slice())
        };
        let comm = amat_ref.comm();
        #[cfg(not(feature = "complex"))]
        {
            let pc = self
                .pc
                .as_mut()
                .map(|b| b.as_mut() as &mut dyn Preconditioner);
            let solver = self
                .solver
                .as_mut()
                .ok_or_else(|| KError::SolveError("No solver".into()))?;
            let mut stats = match solver.solve(
                amat_ref,
                pc,
                b,
                x,
                self.pc_side,
                &comm,
                monitors,
                self.work.as_mut(),
            ) {
                Ok(stats) => stats,
                Err(err) => {
                    if let Some(reason) = ReasonEmitter::from_error(&err, FailureStage::Solve) {
                        if matches!(err, KError::PcFailed(_)) {
                            if let KError::PcFailed(ref msg) = err {
                                log::warn!("KSP diverged due to preconditioner failure: {msg}");
                                #[cfg(feature = "backend-faer")]
                                {
                                    let token = format!(
                                        "phase=apply;rank={};size={};solver={:?};message={}",
                                        comm.rank(),
                                        comm.size(),
                                        self.solver_type,
                                        msg.replace(';', ",")
                                    );
                                    self.set_dist_replay_token("pc_apply_failure", token);
                                }
                            }
                        }
                        let res = self.true_residual_norm_in_place(amat_ref, b, x)?;
                        let mut stats = SolveStats::new(0, res, reason);
                        if let Some(failure) =
                            ReasonEmitter::nested_pc_failure(&err, FailureStage::Solve)
                        {
                            stats = stats.with_nested_pc_failure(failure);
                        }
                        self.apply_residual_contract_classification(
                            &mut stats,
                            Some(amat_ref),
                            b,
                            Some(&err),
                        )?;
                        let stats = stats.finalize_reason_counters();
                        self.last_converged_reason = Some(reason);
                        self.reason_counters.record_reason(reason);
                        if let Some(inner) = stats.nested_pc_failure.as_ref() {
                            self.reason_counters.record_reason(inner.reason);
                        }
                        return Ok(stats);
                    }
                    return Err(err);
                }
            };

            let res = self.true_residual_norm_in_place(amat_ref, b, x)?;
            stats.final_residual = res;
            self.apply_residual_contract_classification(&mut stats, Some(amat_ref), b, None)?;
            let stats = stats.finalize_reason_counters();
            if let Some(inner) = stats.nested_pc_failure.as_ref() {
                log::warn!(
                    "nested_pc_failure component={} reason={} iters={} detail={} final_norm={:?} residual_history={:?}",
                    inner.component,
                    inner.reason,
                    inner.iterations,
                    inner.detail,
                    inner.final_norm,
                    inner.residual_history_summary
                );
            }
            self.last_converged_reason = Some(stats.reason);
            self.reason_counters.record_reason(stats.reason);
            if let Some(inner) = stats.nested_pc_failure.as_ref() {
                self.reason_counters.record_reason(inner.reason);
            }
            Ok(stats)
        }

        #[cfg(feature = "complex")]
        {
            let solver_type = self
                .solver_type
                .ok_or_else(|| KError::SolveError("No solver".into()))?;
            let pc_side = self.pc_side;
            let effective_pc_side = self.effective_pc_side();
            let solver = self
                .solver
                .as_mut()
                .ok_or_else(|| KError::SolveError("No solver".into()))?;
            let work = self.work.as_mut();

            let mut pc_adapter = self.pc.as_mut().map(|pc| PcAsK { inner: pc.as_mut() });
            let mut pc_k = pc_adapter
                .as_mut()
                .map(|pc| pc as &mut dyn KPreconditioner<Scalar = S>);

            let op = LinOpAsK { inner: amat_ref };

            let mut stats = match (|| -> Result<SolveStats<R>, KError> {
                Ok(match solver_type {
                    SolverType::Cg => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<CgSolver>()
                            .ok_or_else(|| KError::SolveError("CG solver missing".into()))?;
                        s.solve_with_comm(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Cgnr => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<CgnrSolver>()
                            .ok_or_else(|| KError::SolveError("CGNR solver missing".into()))?;
                        s.solve_k(&op, pc_k.as_deref(), b, x, pc_side, &comm, monitors, work)?
                    }
                    SolverType::Gmres => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<GmresSolver>()
                            .ok_or_else(|| KError::SolveError("GMRES solver missing".into()))?;
                        s.solve(&op, pc_k.as_deref(), b, x, pc_side, &comm, monitors, work)?
                    }
                    SolverType::Fgmres => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<FgmresSolver>()
                            .ok_or_else(|| KError::SolveError("FGMRES solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref_mut(),
                            b,
                            x,
                            effective_pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::BiCgStab => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<BiCgStabSolver>()
                            .ok_or_else(|| KError::SolveError("BiCGStab solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                        .with_reduction_model(s.reduction_model())
                    }
                    SolverType::Cgs => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<CgsSolver>()
                            .ok_or_else(|| KError::SolveError("CGS solver missing".into()))?;
                        s.solve(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Pcg => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<PcgSolver>()
                            .ok_or_else(|| KError::SolveError("PCG solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Minres => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<MinresSolver>()
                            .ok_or_else(|| KError::SolveError("MINRES solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Lsqr => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<crate::solver::LsqrSolver>()
                            .ok_or_else(|| KError::SolveError("LSQR solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Lsmr => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<crate::solver::LsmrSolver>()
                            .ok_or_else(|| KError::SolveError("LSMR solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::PcaGmres => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<PcaGmresSolver>()
                            .ok_or_else(|| KError::SolveError("PCA-GMRES solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Qmr => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<QmrSolver>()
                            .ok_or_else(|| KError::SolveError("QMR solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Tfqmr => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<TfqmrSolver>()
                            .ok_or_else(|| KError::SolveError("TFQMR solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Tcqmr | SolverType::Richardson => {
                        return Err(KError::Unsupported(
                            "Selected solver is not yet available for complex scalars".into(),
                        ));
                    }
                    SolverType::Chebyshev => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<ChebyshevSolver>()
                            .ok_or_else(|| KError::SolveError("Chebyshev solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Cr => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<CrSolver>()
                            .ok_or_else(|| KError::SolveError("CR solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Gcr => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<GcrSolver>()
                            .ok_or_else(|| KError::SolveError("GCR solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref_mut(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::PipeGcr => {
                        let s = solver
                            .as_any_mut()
                            .downcast_mut::<PipeGcrSolver>()
                            .ok_or_else(|| KError::SolveError("PipeGCR solver missing".into()))?;
                        s.solve_k(
                            &op,
                            pc_k.as_deref_mut(),
                            b,
                            x,
                            self.pc_side,
                            &comm,
                            monitors,
                            work,
                        )?
                    }
                    SolverType::Preonly => unreachable!("PREONLY handled earlier"),
                })
            })() {
                Ok(stats) => stats,
                Err(err) => {
                    if let Some(reason) = ReasonEmitter::from_error(&err, FailureStage::Solve) {
                        if let KError::PcFailed(ref msg) = err {
                            log::warn!("KSP diverged due to preconditioner failure: {msg}");
                        }
                        let res = self.true_residual_norm_in_place(amat_ref, b, x)?;
                        let mut stats = SolveStats::new(0, res, reason);
                        if let Some(failure) =
                            ReasonEmitter::nested_pc_failure(&err, FailureStage::Solve)
                        {
                            stats = stats.with_nested_pc_failure(failure);
                        }
                        self.apply_residual_contract_classification(
                            &mut stats,
                            Some(amat_ref),
                            b,
                            Some(&err),
                        )?;
                        let stats = stats.finalize_reason_counters();
                        self.last_converged_reason = Some(reason);
                        self.reason_counters.record_reason(reason);
                        if let Some(inner) = stats.nested_pc_failure.as_ref() {
                            self.reason_counters.record_reason(inner.reason);
                        }
                        return Ok(stats);
                    }
                    return Err(err);
                }
            };

            let res = self.true_residual_norm_in_place(amat_ref, b, x)?;
            stats.final_residual = res;
            self.apply_residual_contract_classification(&mut stats, Some(amat_ref), b, None)?;
            let stats = stats.finalize_reason_counters();
            if let Some(inner) = stats.nested_pc_failure.as_ref() {
                log::warn!(
                    "nested_pc_failure component={} reason={} iters={} detail={} final_norm={:?} residual_history={:?}",
                    inner.component,
                    inner.reason,
                    inner.iterations,
                    inner.detail,
                    inner.final_norm,
                    inner.residual_history_summary
                );
            }
            self.last_converged_reason = Some(stats.reason);
            self.reason_counters.record_reason(stats.reason);
            if let Some(inner) = stats.nested_pc_failure.as_ref() {
                self.reason_counters.record_reason(inner.reason);
            }
            Ok(stats)
        }
    }

    fn residual_norm_for_stats(
        &mut self,
        mat: &dyn LinOp<S = S>,
        b: &[S],
        x: &[S],
    ) -> Result<R, KError> {
        if self.work.is_some() {
            return self.true_residual_norm_in_place(mat, b, x);
        }
        let mut tmp = vec![S::zero(); b.len()];
        mat.try_matvec(x, &mut tmp)
            .map_err(|e| KError::SolveError(format!("residual matvec failed: {e}")))?;
        for (ri, bi) in tmp.iter_mut().zip(b.iter()) {
            *ri = *bi - *ri;
        }
        let comm = mat.comm();
        let red = comm.reduction_engine(&self.reduction_opts);
        Ok(red.norm2_s(&tmp))
    }

    fn true_residual_norm_in_place(
        &mut self,
        mat: &dyn LinOp<S = S>,
        b: &[S],
        x: &[S],
    ) -> Result<R, KError> {
        let w = self
            .work
            .as_mut()
            .ok_or_else(|| KError::SolveError("Workspace not initialized".into()))?;

        if w.tmp1.len() != b.len() {
            return Err(KError::InvalidInput(format!(
                "workspace.tmp1 len {} != b len {} (setup missing or dimension mismatch)",
                w.tmp1.len(),
                b.len()
            )));
        }

        mat.try_matvec(x, &mut w.tmp1)
            .map_err(|e| KError::SolveError(format!("residual matvec failed: {e}")))?;

        for (ri, bi) in w.tmp1.iter_mut().zip(b.iter()) {
            *ri = *bi - *ri;
        }

        let comm = mat.comm();
        let red = w
            .reduction_engine()
            .cloned()
            .unwrap_or_else(|| comm.reduction_engine(w.reduction_options()));
        Ok(red.norm2_s(&w.tmp1))
    }

    fn apply_residual_contract_classification(
        &mut self,
        stats: &mut SolveStats<R>,
        mat: Option<&dyn LinOp<S = S>>,
        b: &[S],
        source_err: Option<&KError>,
    ) -> Result<(), KError> {
        let bnorm = if let Some(mat) = mat {
            let comm = mat.comm();
            if let Some(work) = self.work.as_ref() {
                let red = work
                    .reduction_engine()
                    .cloned()
                    .unwrap_or_else(|| comm.reduction_engine(work.reduction_options()));
                red.norm2_s(b)
            } else {
                comm.reduction_engine(&self.reduction_opts).norm2_s(b)
            }
        } else {
            R::default()
        };
        let tol = self.atol.max(self.rtol * bnorm).real();
        let true_res = stats.final_residual.real();
        let acceptance = classify_acceptance_status(stats.reason, true_res, tol);
        stats.acceptance_status = acceptance;

        if stats.reason.is_diverged() || matches!(acceptance, AcceptanceStatus::OkWithWarning) {
            stats.breakdown_reason = Some(stats.reason);
        }
        if matches!(acceptance, AcceptanceStatus::OkWithWarning) {
            let mut note = format!(
                "true_residual={true_res:.6e} <= max(atol, rtol*||b||)={tol:.6e}; accepted_with_warning from {}",
                stats.reason.petsc_reason()
            );
            if let Some(err) = source_err {
                note.push_str(&format!("; source_error={err}"));
            }
            stats.residual_override_note = Some(note);
        }
        Ok(())
    }

    fn invalidate_solver_setup(&mut self) {
        self.setup_called = false;
    }

    fn invalidate_pc_setup(&mut self) {
        self.setup_called = false;
        self.reset_pc_ids();
    }

    #[cfg(feature = "backend-faer")]
    fn set_dist_route_selected(&mut self, route: impl Into<String>) {
        self.dist_route_diag.selected_route = Some(route.into());
    }

    #[cfg(feature = "backend-faer")]
    fn set_dist_route_capability_entry(&mut self, entry: DistCsrCapabilityEntry) {
        self.dist_route_diag.capability_entry = Some(entry);
    }

    #[cfg(feature = "backend-faer")]
    fn set_dist_route_decision_report(&mut self, report: DistRouteDecisionReport) {
        self.dist_route_diag.decision_report = Some(report);
    }

    #[cfg(feature = "backend-faer")]
    fn push_dist_route_fallback(&mut self, reason: DistRouteFallbackReason) {
        let key = reason.as_str().to_string();
        self.dist_route_diag.fallback_chain.push(key.clone());
        *self
            .dist_route_diag
            .fallback_counters
            .entry(key)
            .or_insert(0) += 1;
    }

    #[cfg(feature = "backend-faer")]
    fn set_dist_route_fallback_reason(&mut self, reason: impl Into<String>) {
        self.dist_route_diag.fallback_reason = Some(reason.into());
    }

    #[cfg(feature = "backend-faer")]
    fn refresh_and_validate_dist_route_report(
        &mut self,
        pending: &PendingMpiPc,
        selected: DistRouteSelection,
    ) -> Result<(), KError> {
        let report = build_dist_route_decision_report(
            &pending.mpi_opts,
            selected,
            self.dist_route_diag.fallback_reason.clone(),
            self.dist_route_diag.fallback_chain.len(),
        );
        validate_dist_route_policy_budget(&report, &self.dist_route_diag.fallback_chain)?;
        self.set_dist_route_decision_report(report);
        Ok(())
    }

    #[cfg(all(feature = "backend-faer", not(feature = "complex"), feature = "mpi"))]
    fn dist_preflight_reasons(
        &self,
        pmat: &Arc<dyn LinOp<S = S>>,
        pending: &PendingMpiPc,
        dist_op: Option<&DistCsrOp>,
        local_only_pc: bool,
    ) -> Vec<String> {
        let mut reasons = Vec::new();
        let comm = pmat.comm();
        let layout_ok = pmat.dist_layout().is_some() || dist_op.is_some();
        if !layout_ok {
            reasons.push("layout_incomplete".to_string());
        }
        if comm.size() <= 1 {
            reasons.push("communicator_not_distributed".to_string());
        }
        if pending.mpi_opts.local_apply_mode.is_distributed_native() && dist_op.is_none() {
            reasons.push("halo_not_ready".to_string());
        }
        if local_only_pc
            && pending.mpi_opts.global_pc == GlobalPcKind::None
            && pending.mpi_opts.route_policy == DistRoutePolicy::Adapted
        {
            reasons.push("pc_global_local_incompatible".to_string());
        }
        reasons
    }

    #[cfg(feature = "backend-faer")]
    fn maybe_reuse_dist_preflight(&mut self, probe_key: &str) -> Option<DistRoutePreflightState> {
        let state = self.dist_route_diag.preflight.as_mut()?;
        if state.probe_key != probe_key {
            return None;
        }
        state.cached_hits = state.cached_hits.saturating_add(1);
        Some(state.clone())
    }

    #[cfg(feature = "backend-faer")]
    fn store_dist_preflight(&mut self, state: DistRoutePreflightState) {
        self.dist_route_diag.preflight = Some(state);
    }

    #[cfg(feature = "backend-faer")]
    fn set_dist_replay_token(&mut self, key: &str, token: String) {
        self.dist_route_diag
            .replay_tokens
            .insert(key.to_string(), token);
    }

    #[cfg(all(feature = "backend-faer", not(feature = "complex"), feature = "mpi"))]
    fn build_mpi_global_pc(
        &self,
        pending: &PendingMpiPc,
        dist_op: &DistCsrOp,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        match pending.mpi_opts.global_pc {
            GlobalPcKind::BlockJacobi => {
                let builder = DistPcBuilder::BlockJacobi {
                    opts: pending.mpi_opts.clone(),
                };
                Ok(Box::new(DistPcAdapter::build(dist_op, builder)?))
            }
            GlobalPcKind::Asm => {
                let builder = Self::build_dist_asm_builder(pending, GlobalPcKind::Asm)?;
                Ok(Box::new(DistPcAdapter::build(dist_op, builder)?))
            }
            GlobalPcKind::Ras => {
                let builder = Self::build_dist_asm_builder(pending, GlobalPcKind::Ras)?;
                Ok(Box::new(DistPcAdapter::build(dist_op, builder)?))
            }
            GlobalPcKind::None => Err(KError::InvalidInput(
                "pc_global=none should not build a global PC".into(),
            )),
        }
    }

    #[cfg(all(feature = "backend-faer", not(feature = "complex"), feature = "mpi"))]
    fn build_dist_asm_builder(
        pending: &PendingMpiPc,
        global: GlobalPcKind,
    ) -> Result<DistPcBuilder, KError> {
        let overlap = pending.pc_opts.asm_overlap.unwrap_or(0);
        let subdomain_hint = pending.pc_opts.asm_subdomain_size;
        let block_solver = match pending.pc_opts.asm_block_solver.as_deref() {
            Some("csr") => AsmBlockSolver::Csr,
            Some("ludense") | None => AsmBlockSolver::LuDense,
            Some(other) => {
                return Err(KError::InvalidInput(format!(
                    "unknown pc_asm_block_solver: {other}"
                )));
            }
        };
        let inner_pc = match pending.pc_opts.asm_inner_pc.as_deref() {
            Some("jacobi") => AsmInnerPc::Jacobi,
            Some("ilut") => AsmInnerPc::Ilut {
                drop_tol: pending.pc_opts.ilut_drop_tol.unwrap_or(1e-4),
                max_fill: pending.pc_opts.ilut_max_fill.unwrap_or(20),
            },
            Some("ilutp") => AsmInnerPc::Ilutp {
                drop_tol: pending.pc_opts.ilutp_drop_tol.unwrap_or(1e-4),
                max_fill: pending.pc_opts.ilutp_max_fill.unwrap_or(10),
                perm_tol: pending.pc_opts.ilutp_perm_tol.unwrap_or(0.1),
            },
            Some("ilu") | Some("ilu0") | None => AsmInnerPc::Ilu0,
            Some(other) => {
                return Err(KError::InvalidInput(format!(
                    "unknown pc_asm_inner_pc: {other}"
                )));
            }
        };
        let weighting = match pending.pc_opts.asm_weighting.as_deref() {
            Some("uniform") => Weighting::Uniform,
            Some("none") | None => Weighting::None,
            Some(other) => {
                return Err(KError::InvalidInput(format!(
                    "unsupported distributed pc_asm_weighting: {other}"
                )));
            }
        };

        Ok(match global {
            GlobalPcKind::Asm => DistPcBuilder::Asm {
                overlap,
                subdomain_hint,
                block_solver,
                inner_pc,
                weighting,
                coarse_strategy: DistCoarseStrategy::None,
                local_apply_mode: pending.mpi_opts.local_apply_mode,
            },
            GlobalPcKind::Ras => DistPcBuilder::Ras {
                overlap,
                subdomain_hint,
                block_solver,
                inner_pc,
                weighting,
                coarse_strategy: DistCoarseStrategy::None,
                local_apply_mode: pending.mpi_opts.local_apply_mode,
            },
            _ => {
                return Err(KError::InvalidInput(
                    "ASM builder requested for non-ASM global kind".into(),
                ));
            }
        })
    }

    #[cfg(all(feature = "backend-faer", not(feature = "complex"), feature = "mpi"))]
    fn maybe_upgrade_local_pc(
        &mut self,
        pmat: Arc<dyn LinOp<S = S>>,
        sid: StructureId,
        vid: ValuesId,
    ) -> Result<bool, KError> {
        let local_only_pc = match self.pc.as_ref() {
            Some(pc) => pc.distributed_support() == PcDistributedSupport::LocalOnly,
            None => return Ok(false),
        };
        let comm = pmat.comm();
        let is_distributed = comm.size() > 1
            && (pmat.dist_layout().is_some()
                || pmat.as_any().downcast_ref::<DistCsrOp>().is_some());
        if !is_distributed {
            return Ok(false);
        }

        let Some(pending) = self.pending_mpi_pc.as_ref().cloned() else {
            return Err(KError::InvalidInput(
                "selected preconditioner is rank-local for a distributed operator; set -pc_global block_jacobi|asm|ras"
                    .into(),
            ));
        };

        let explicit_global = pending.mpi_opts.global_pc != GlobalPcKind::None;
        let strict_local_apply = pending.mpi_opts.local_apply_mode.requires_native();
        let dist_op = pmat.as_any().downcast_ref::<DistCsrOp>();
        let preflight_probe_key = format!(
            "comm={}#size={}#distcsr={}#layout={}#global={:?}#local={:?}#apply={}#route={:?}#local_only={}",
            pmat.comm().id(),
            pmat.comm().size(),
            dist_op.is_some(),
            pmat.dist_layout().is_some(),
            pending.mpi_opts.global_pc,
            pending.mpi_opts.local_pc,
            pending
                .mpi_opts
                .local_apply_mode
                .communication_strategy_name(),
            pending.mpi_opts.route_policy,
            local_only_pc,
        );
        let mut preflight_cached = false;
        let preflight = if let Some(cached) = self.maybe_reuse_dist_preflight(&preflight_probe_key)
        {
            preflight_cached = true;
            cached
        } else {
            let reason_codes = self.dist_preflight_reasons(&pmat, &pending, dist_op, local_only_pc);
            let native_ready = reason_codes.is_empty();
            let outcome = if native_ready { "passed" } else { "rejected" }.to_string();
            let state = DistRoutePreflightState {
                probe_key: preflight_probe_key.clone(),
                outcome,
                reason_codes,
                native_ready,
                cached_hits: 0,
            };
            self.store_dist_preflight(state.clone());
            state
        };

        let capability_entry = resolve_distcsr_capability(DistCsrCapabilityKey {
            solver_type: self.solver_type,
            global_pc: pending.mpi_opts.global_pc,
            local_pc: pending.mpi_opts.local_pc,
            apply_mode: pending.mpi_opts.local_apply_mode,
        });
        self.set_dist_route_capability_entry(capability_entry.clone());

        let decision = resolve_dist_route(DistRouteResolveInput {
            has_distcsr: dist_op.is_some(),
            explicit_global,
            native_global_candidate: capability_entry.native_global_candidate,
            route_policy: pending.mpi_opts.route_policy,
            local_only_pc,
            local_apply_mode: pending.mpi_opts.local_apply_mode,
        });
        let decision_reasons = {
            let accepted = decision
                .accepted
                .iter()
                .map(|r| r.as_str())
                .collect::<Vec<_>>()
                .join(",");
            let rejected = decision
                .rejected
                .iter()
                .map(|r| r.as_str())
                .collect::<Vec<_>>()
                .join(",");
            format!("accept={accepted};reject={rejected}")
        };

        if decision.selected == DistRouteSelection::DistCsrNativeBlockJacobi {
            if !preflight.native_ready {
                let reasons = if preflight.reason_codes.is_empty() {
                    "unknown".to_string()
                } else {
                    preflight.reason_codes.join(",")
                };
                self.set_dist_route_selected(format!(
                    "{}:preflight_rejected:{}:{}",
                    DistRouteSelection::DistCsrNativeBlockJacobi.as_str(),
                    reasons,
                    decision_reasons
                ));
                self.push_dist_route_fallback(DistRouteFallbackReason::NativeSetupFailed);
                self.set_dist_route_fallback_reason(format!(
                    "native preflight rejected: {reasons}"
                ));
                self.refresh_and_validate_dist_route_report(
                    &pending,
                    DistRouteSelection::DistCsrNativeBlockJacobi,
                )?;
                if strict_local_apply {
                    return Err(KError::InvalidInput(format!(
                        "pc_dist_local_apply=strict requires native DistCsr apply readiness; preflight rejected: {reasons}"
                    )));
                }
            } else {
                let dist_op = dist_op.expect("resolver selected native route without DistCsrOp");
                let mut native_pending = pending.clone();
                native_pending.mpi_opts.global_pc = GlobalPcKind::BlockJacobi;
                if !native_pending
                    .mpi_opts
                    .local_apply_mode
                    .is_distributed_native()
                {
                    return Err(KError::InvalidInput(
                        "native DistCsr route selected with non-native local apply mode".into(),
                    ));
                }
                self.set_dist_route_selected(format!(
                    "{}:{}:{}",
                    DistRouteSelection::DistCsrNativeBlockJacobi.as_str(),
                    native_pending
                        .mpi_opts
                        .local_apply_mode
                        .communication_strategy_name(),
                    decision_reasons
                ));
                let setup_token = format!(
                    "phase=setup;rank={};size={};row_start={};local_rows={};global_rows={};route={}",
                    dist_op.comm().rank(),
                    dist_op.comm().size(),
                    dist_op.local_row_offset(),
                    dist_op.local_nrows(),
                    dist_op.n_global,
                    DistRouteSelection::DistCsrNativeBlockJacobi.as_str()
                );
                self.set_dist_replay_token("native_setup", setup_token);
                if !explicit_global {
                    self.push_dist_route_fallback(DistRouteFallbackReason::AutoPromotedFromLocal);
                }
                self.refresh_and_validate_dist_route_report(
                    &pending,
                    DistRouteSelection::DistCsrNativeBlockJacobi,
                )?;
                let mut new_pc = self.build_mpi_global_pc(&native_pending, dist_op)?;
                let want = new_pc.required_format();
                let tol = new_pc.preferred_drop_tol_for_format().unwrap_or_default();
                let pmat_view = materialize(pmat.clone(), want, tol)?;
                match new_pc.setup(pmat_view.as_ref()) {
                    Ok(()) => {
                        self.pc = Some(new_pc);
                        self.last_pc_sid = Some(sid);
                        self.last_pc_vid = Some(vid);
                        return Ok(true);
                    }
                    Err(err) => {
                        self.push_dist_route_fallback(DistRouteFallbackReason::NativeSetupFailed);
                        self.set_dist_route_fallback_reason(err.to_string());
                        self.refresh_and_validate_dist_route_report(
                            &pending,
                            DistRouteSelection::DistCsrNativeBlockJacobi,
                        )?;
                        if strict_local_apply {
                            return Err(KError::InvalidInput(format!(
                                "pc_dist_local_apply=strict requires native DistCsr apply setup, but setup failed: {err}"
                            )));
                        }
                        log::warn!(
                            "Native distributed preconditioner setup failed, enabling fallback chain: {err}"
                        );
                    }
                }
            }
        }

        if decision.selected == DistRouteSelection::ConfiguredGlobal {
            let Some(dist_op) = dist_op else {
                self.push_dist_route_fallback(DistRouteFallbackReason::MissingDistCsrOperator);
                self.refresh_and_validate_dist_route_report(
                    &pending,
                    DistRouteSelection::ConfiguredGlobal,
                )?;
                return Err(KError::InvalidInput(
                    "-pc_global requires DistCsrOp when running distributed".into(),
                ));
            };
            let configured_label = match pending.mpi_opts.global_pc {
                GlobalPcKind::BlockJacobi => "distcsr_native_block_jacobi_configured",
                GlobalPcKind::Asm => "distcsr_native_asm",
                GlobalPcKind::Ras => "distcsr_native_ras",
                GlobalPcKind::None => "configured_global_none",
            };
            self.set_dist_route_selected(format!("{configured_label}:{decision_reasons}"));
            self.push_dist_route_fallback(DistRouteFallbackReason::ConfiguredGlobalFallback);
            self.refresh_and_validate_dist_route_report(
                &pending,
                DistRouteSelection::ConfiguredGlobal,
            )?;
            let setup_token = format!(
                "phase=setup;rank={};size={};row_start={};local_rows={};global_rows={};route={}",
                dist_op.comm().rank(),
                dist_op.comm().size(),
                dist_op.local_row_offset(),
                dist_op.local_nrows(),
                dist_op.n_global,
                DistRouteSelection::ConfiguredGlobal.as_str()
            );
            self.set_dist_replay_token("configured_setup", setup_token);
            let mut new_pc = self.build_mpi_global_pc(&pending, dist_op)?;
            let want = new_pc.required_format();
            let tol = new_pc.preferred_drop_tol_for_format().unwrap_or_default();
            let pmat_view = materialize(pmat.clone(), want, tol)?;
            if let Err(err) = new_pc.setup(pmat_view.as_ref()) {
                self.set_dist_route_fallback_reason(err.to_string());
                self.refresh_and_validate_dist_route_report(
                    &pending,
                    DistRouteSelection::ConfiguredGlobal,
                )?;
                self.handle_pc_setup_failure(err, pmat, sid, vid)?;
                return Ok(false);
            }
            self.pc = Some(new_pc);
            self.last_pc_sid = Some(sid);
            self.last_pc_vid = Some(vid);
            return Ok(true);
        }

        if preflight_cached {
            log::debug!(
                "Reused cached distributed preflight outcome={} reasons={:?}",
                preflight.outcome,
                preflight.reason_codes
            );
        }

        if !local_only_pc {
            return Ok(false);
        }

        if decision
            .rejected
            .contains(&DistRouteDecisionReason::RoutePolicyAdapted)
        {
            self.set_dist_route_selected(format!(
                "{}:{}",
                DistRouteSelection::LocalAdapter.as_str(),
                decision_reasons
            ));
            self.push_dist_route_fallback(DistRouteFallbackReason::AdapterOnlyPolicy);
            self.refresh_and_validate_dist_route_report(
                &pending,
                DistRouteSelection::LocalAdapter,
            )?;
            if strict_local_apply {
                return Err(KError::InvalidInput(
                    "pc_dist_local_apply=strict forbids local adapter fallback; use a native DistCsr route".into(),
                ));
            }
            return Ok(false);
        }

        if decision.selected == DistRouteSelection::RootGather {
            self.set_dist_route_selected(format!(
                "{}:{}",
                DistRouteSelection::RootGather.as_str(),
                decision_reasons
            ));
            self.push_dist_route_fallback(DistRouteFallbackReason::RootGatherPolicy);
            self.refresh_and_validate_dist_route_report(&pending, DistRouteSelection::RootGather)?;
            if strict_local_apply {
                return Err(KError::InvalidInput(
                    "pc_dist_local_apply=strict forbids root-gather fallback; use a native DistCsr route".into(),
                ));
            }
            return Ok(false);
        }

        Err(KError::InvalidInput(
            "distributed operator detected but no native DistCsr route available; set -pc_global block_jacobi|asm|ras or select -pc_dist_route adapted for explicit local adapter fallback"
                .into(),
        ))
    }

    /// Add an iteration monitor callback.
    pub fn add_monitor<F>(&mut self, f: F)
    where
        F: Fn(usize, R, usize) -> MonitorAction + Send + Sync + 'static,
    {
        self.monitors.push(Box::new(f));
    }

    /// Set how monitors are invoked across ranks.
    pub fn set_monitor_policy(&mut self, policy: MonitorPolicy) -> &mut Self {
        self.monitor_policy = policy;
        self
    }

    /// Add a monitor that only runs on rank 0 (when bound to an MPI communicator).
    pub fn add_monitor_rank0<F>(&mut self, f: F)
    where
        F: Fn(usize, R, usize) -> MonitorAction + Send + Sync + 'static,
    {
        self.monitors
            .push(Box::new(move |it, r, reds| f(it, r, reds)));
        self.monitor_policy = MonitorPolicy::Rank0Only;
    }

    /// Return the number of registered monitors.
    pub fn num_monitors(&self) -> usize {
        self.monitors.len()
    }

    /// Clear all registered monitors.
    pub fn clear_monitors(&mut self) {
        self.monitors.clear();
    }

    #[cfg(test)]
    pub fn set_preconditioner(&mut self, pc: Box<dyn Preconditioner>) {
        self.pc = Some(pc);
        self.invalidate_pc_setup();
    }

    /// Invoke all monitors with the provided iteration and residual.
    pub fn invoke_monitors(&self, iter: usize, residual: R, reductions: usize) {
        let do_call = match self.monitor_policy {
            MonitorPolicy::AllRanks => true,
            MonitorPolicy::Rank0Only => self
                .bound_comm
                .as_ref()
                .map(|c| c.rank() == 0)
                .unwrap_or(true),
        };
        if !do_call {
            return;
        }
        for m in &self.monitors {
            let _ = m(iter, residual, reductions);
        }
    }

    /// Set solver tolerances and maximum iterations.
    pub fn set_tolerances(&mut self, rtol: R, atol: R, dtol: R, maxits: usize) -> &mut Self {
        self.rtol = rtol;
        self.atol = atol;
        self.dtol = dtol;
        self.maxits = maxits;
        if let Some(st) = self.solver_type {
            if let Err(err) = self.set_type(st) {
                log::warn!("failed to rebuild solver after set_tolerances: {err}");
                self.invalidate_solver_setup();
            }
        }
        self.invalidate_solver_setup();
        self
    }

    /// Configure the underlying solver based on the requested preconditioning side.
    fn configure_pc_side(&mut self) -> Result<(), KError> {
        let side = self.effective_pc_side();

        if let Some(SolverType::PcaGmres) = self.solver_type {
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|s| s.as_any_mut().downcast_mut::<PcaGmresSolver>())
            {
                s.pc_mode = match side {
                    PcSide::Left | PcSide::Symmetric => PcaPcMode::Left,
                    PcSide::Right => PcaPcMode::Right,
                };
            }
        }

        if let Some(st) = self.solver_type {
            if st.right_only_pc_side() && side != PcSide::Right {
                return Err(KError::InvalidInput(format!(
                    "{st:?} supports only right preconditioning; got {side:?}"
                )));
            }
            if let Some(required) = st.required_pc_side() {
                if side != required {
                    return Err(KError::InvalidInput(format!(
                        "{st:?} requires left preconditioning; got {side:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Query whether setup has been performed.
    pub fn is_setup(&self) -> bool {
        self.setup_called
    }

    /// Set the GMRES restart parameter.
    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart;
        // Restart impacts GMRES-family solver workspaces and must update the active solver
        // immediately (and be staged for later solver construction).
        self.pending_gmres.restart = Some(restart);
        self.pending_fgmres.restart = Some(restart);
        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<GmresSolver>())
        {
            s.set_restart(restart);
        }
        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<FgmresSolver>())
        {
            s.set_restart(restart);
        }
        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<PcaGmresSolver>())
        {
            s.set_restart(restart);
        }
        self.invalidate_solver_setup();
    }

    pub fn set_gmres_stagnation_policy(&mut self, policy: StagnationPolicy) {
        self.pending_gmres.stagnation_policy = Some(policy);
        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<GmresSolver>())
        {
            s.set_stagnation_policy(policy);
        }
        self.invalidate_solver_setup();
    }
}

impl KspContext {
    /// Test-only: view current workspace (e.g., to inspect GMRES V/Z basis sizes).
    pub fn debug_workspace(&self) -> Option<&Workspace> {
        self.work.as_ref()
    }

    /// Test-only: inject a preconditioner for controlled testing.
    pub fn set_pc_box_for_tests(&mut self, pc: Box<dyn Preconditioner>) {
        self.pc = Some(pc);
        self.invalidate_pc_setup();
    }

    pub fn debug_gmres_runtime(
        &mut self,
    ) -> Option<(
        usize,
        usize,
        crate::solver::gmres::GmresVariant,
        ReorthPolicy,
        StagnationPolicy,
    )> {
        let s = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<GmresSolver>())?;
        Some((
            s.restart,
            s.conv.max_iters,
            s.variant,
            s.reorth,
            s.stagnation_policy(),
        ))
    }
}

#[cfg(all(test, feature = "backend-faer"))]
mod tests {
    use super::*;
    use crate::config::options::{KspOptions, PcOptions};
    use crate::context::pc_context::PcType;
    #[cfg(not(feature = "complex"))]
    use crate::matrix::op::{CsrOp, DenseOp};
    #[cfg(not(feature = "complex"))]
    use crate::matrix::utils::poisson_2d;
    use crate::preconditioner::PcSide;
    #[cfg(not(feature = "complex"))]
    use crate::utils::convergence::AcceptanceStatus;
    #[cfg(not(feature = "complex"))]
    use crate::utils::matrix_market::read_matrix_market;
    #[cfg(not(feature = "complex"))]
    use faer::Mat;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[cfg(any(feature = "mpi", feature = "rayon"))]
    use std::sync::{Mutex, MutexGuard, OnceLock};

    struct CountingPc {
        setup_ct: Arc<AtomicUsize>,
        sym_ct: Arc<AtomicUsize>,
        num_ct: Arc<AtomicUsize>,
        supports_num: bool,
    }

    #[cfg(not(feature = "complex"))]
    struct BreakdownApplyPc;

    #[cfg(not(feature = "complex"))]
    impl Preconditioner for BreakdownApplyPc {
        fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
            Ok(())
        }

        fn apply(&self, _side: PcSide, _x: &[S], _y: &mut [S]) -> Result<(), KError> {
            Err(KError::BreakdownOrIndefinite)
        }
    }

    impl Preconditioner for CountingPc {
        fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
            self.setup_ct.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
            y.copy_from_slice(x);
            Ok(())
        }

        fn supports_numeric_update(&self) -> bool {
            self.supports_num
        }

        fn update_symbolic(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
            self.sym_ct.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn update_numeric(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
            self.num_ct.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[cfg(feature = "mpi")]
    fn mpi_test_guard() -> MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        GUARD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("mpi_test_guard poisoned")
    }

    #[cfg(feature = "rayon")]
    fn rayon_test_guard() -> MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        GUARD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("rayon_test_guard poisoned")
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn setup_workspace_runs_on_solver_switch_same_dim_cgnr_to_cg() {
        let a = Mat::<R>::from_fn(4, 4, |i, j| if i == j { 2.0 } else { 0.5 });
        let a = Arc::new(a);

        let mut ksp = KspContext::new();
        ksp.set_operators(a, None);

        ksp.set_type(SolverType::Cgnr).unwrap();
        ksp.setup().unwrap();

        // CGNR sets up a small workspace footprint.
        let ws = ksp.debug_workspace().unwrap();
        assert_eq!(ws.q_s.len(), 2);

        // Switch to CG without changing matrix dims.
        ksp.set_type(SolverType::Cg).unwrap();
        ksp.setup().unwrap();

        // CG setup must have run and expanded workspace requirements.
        let ws = ksp.debug_workspace().unwrap();
        assert_eq!(ws.q_s.len(), 4);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn setup_workspace_runs_on_solver_switch_same_dim() {
        let a = Mat::<R>::from_fn(4, 4, |i, j| if i == j { 2.0 } else { 0.5 });
        let a = Arc::new(a);

        let mut ksp = KspContext::new();
        ksp.set_operators(a.clone(), None);

        ksp.set_type(SolverType::Cg).unwrap();
        ksp.setup().unwrap();

        // CG should not have allocated GMRES storage.
        let ws = ksp.debug_workspace().unwrap();
        assert_eq!(ws.v_mem.len(), 0);
        assert_eq!(ws.h_mem.len(), 0);

        // Switch to GMRES *without changing matrix dims*
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.set_restart(3);
        ksp.setup().unwrap();

        // GMRES setup must have configured basis + Hessenberg storage.
        let ws = ksp.debug_workspace().unwrap();
        assert_eq!(ws.v_mem.len(), (3 + 1) * 4);
        assert_eq!(ws.h_mem.len(), (3 + 1) * 3);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn setup_workspace_runs_on_restart_change() {
        let a = Mat::<R>::from_fn(4, 4, |i, j| if i == j { 2.0 } else { 0.5 });
        let a = Arc::new(a);

        let mut ksp = KspContext::new();
        ksp.set_operators(a, None);
        ksp.set_type(SolverType::Gmres).unwrap();

        ksp.set_restart(2);
        ksp.setup().unwrap();
        let ws = ksp.debug_workspace().unwrap();
        assert_eq!(ws.v_mem.len(), (2 + 1) * 4);
        assert_eq!(ws.h_mem.len(), (2 + 1) * 2);

        // Change restart, same dimension -> must resize GMRES buffers
        ksp.set_restart(5);
        ksp.setup().unwrap();
        let ws = ksp.debug_workspace().unwrap();
        assert_eq!(ws.v_mem.len(), (5 + 1) * 4);
        assert_eq!(ws.h_mem.len(), (5 + 1) * 5);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn setup_workspace_runs_on_cg_variant_change() {
        let a = Mat::<R>::from_fn(4, 4, |i, j| if i == j { 2.0 } else { 0.5 });
        let a = Arc::new(a);

        let mut ksp = KspContext::new();
        ksp.set_operators(a, None);
        ksp.set_type(SolverType::Cg).unwrap();

        // Default CG variant should require 4 q_s buffers.
        ksp.setup().unwrap();
        let ws = ksp.debug_workspace().unwrap();
        assert_eq!(ws.q_s.len(), 4);

        // Switch CG to pipelined via options; setup() should re-run solver workspace setup.
        let opts = KspOptions {
            cg_pipelined: Some(true),
            ..Default::default()
        };
        ksp.set_from_options(&opts).unwrap();
        ksp.setup().unwrap();

        // Pipelined CG requires one additional buffer.
        let ws = ksp.debug_workspace().unwrap();
        assert_eq!(ws.q_s.len(), 5);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    #[cfg(not(feature = "mat-values-fingerprint"))]
    fn auto_unknown_valuesid_refreshes_symbolic_when_no_numeric_update() {
        let setup_ct = Arc::new(AtomicUsize::new(0));
        let sym_ct = Arc::new(AtomicUsize::new(0));
        let num_ct = Arc::new(AtomicUsize::new(0));

        let pc = CountingPc {
            setup_ct: setup_ct.clone(),
            sym_ct: sym_ct.clone(),
            num_ct: num_ct.clone(),
            supports_num: false,
        };

        let a = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });
        let a = Arc::new(a);

        let mut ksp = KspContext::new();
        ksp.set_operators(a, None);
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.set_pc_box_for_tests(Box::new(pc));
        ksp.set_pc_reuse_policy(PcReusePolicy::Auto);

        ksp.setup().unwrap();
        assert_eq!(setup_ct.load(Ordering::Relaxed), 1);
        assert_eq!(sym_ct.load(Ordering::Relaxed), 0);
        assert_eq!(num_ct.load(Ordering::Relaxed), 0);

        ksp.setup().unwrap();
        assert_eq!(setup_ct.load(Ordering::Relaxed), 1);
        assert_eq!(sym_ct.load(Ordering::Relaxed), 1);
        assert_eq!(num_ct.load(Ordering::Relaxed), 0);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn restart_change_does_not_rebuild_pc() {
        let setup_ct = Arc::new(AtomicUsize::new(0));
        let sym_ct = Arc::new(AtomicUsize::new(0));
        let num_ct = Arc::new(AtomicUsize::new(0));

        let pc = CountingPc {
            setup_ct: setup_ct.clone(),
            sym_ct: sym_ct.clone(),
            num_ct: num_ct.clone(),
            supports_num: false,
        };

        let a = Arc::new(Mat::<R>::from_fn(
            4,
            4,
            |i, j| if i == j { 2.0 } else { 0.5 },
        ));
        let a = Arc::new(DenseOp::<f64>::new(a));

        let mut ksp = KspContext::new();
        ksp.set_operators(a, None);
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.set_pc_box_for_tests(Box::new(pc));

        ksp.setup().unwrap();
        assert_eq!(setup_ct.load(Ordering::Relaxed), 1);
        assert_eq!(sym_ct.load(Ordering::Relaxed), 0);
        assert_eq!(num_ct.load(Ordering::Relaxed), 0);

        ksp.set_restart(5);
        ksp.setup().unwrap();
        assert_eq!(setup_ct.load(Ordering::Relaxed), 1);
        assert_eq!(sym_ct.load(Ordering::Relaxed), 0);
        assert_eq!(num_ct.load(Ordering::Relaxed), 0);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn tolerances_change_does_not_rebuild_pc() {
        let setup_ct = Arc::new(AtomicUsize::new(0));
        let sym_ct = Arc::new(AtomicUsize::new(0));
        let num_ct = Arc::new(AtomicUsize::new(0));

        let pc = CountingPc {
            setup_ct: setup_ct.clone(),
            sym_ct: sym_ct.clone(),
            num_ct: num_ct.clone(),
            supports_num: true,
        };

        let a = Arc::new(Mat::<R>::from_fn(
            4,
            4,
            |i, j| if i == j { 2.0 } else { 0.5 },
        ));
        let a = Arc::new(DenseOp::<f64>::new(a));

        let mut ksp = KspContext::new();
        ksp.set_operators(a, None);
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.set_pc_box_for_tests(Box::new(pc));

        ksp.setup().unwrap();
        assert_eq!(setup_ct.load(Ordering::Relaxed), 1);
        assert_eq!(sym_ct.load(Ordering::Relaxed), 0);
        assert_eq!(num_ct.load(Ordering::Relaxed), 0);

        ksp.set_tolerances(1e-6, 1e-12, 1e8, 1234);
        ksp.setup().unwrap();
        assert_eq!(setup_ct.load(Ordering::Relaxed), 1);
        assert_eq!(sym_ct.load(Ordering::Relaxed), 0);
        assert_eq!(num_ct.load(Ordering::Relaxed), 0);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn operator_change_triggers_pc_refresh() {
        let setup_ct = Arc::new(AtomicUsize::new(0));
        let sym_ct = Arc::new(AtomicUsize::new(0));
        let num_ct = Arc::new(AtomicUsize::new(0));

        let pc = CountingPc {
            setup_ct: setup_ct.clone(),
            sym_ct: sym_ct.clone(),
            num_ct: num_ct.clone(),
            supports_num: true,
        };

        let a1 = Arc::new(Mat::<R>::from_fn(
            3,
            3,
            |i, j| if i == j { 2.0 } else { 0.25 },
        ));
        let a1 = Arc::new(DenseOp::<f64>::new(a1));

        let mut ksp = KspContext::new();
        ksp.set_operators(a1, None);
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.set_pc_box_for_tests(Box::new(pc));

        ksp.setup().unwrap();
        let total_before = setup_ct.load(Ordering::Relaxed)
            + sym_ct.load(Ordering::Relaxed)
            + num_ct.load(Ordering::Relaxed);

        let a2 = Arc::new(Mat::<R>::from_fn(
            3,
            3,
            |i, j| if i == j { 3.0 } else { 0.75 },
        ));
        let a2 = Arc::new(DenseOp::<f64>::new(a2));
        ksp.try_set_operators(a2, None).unwrap();

        ksp.setup().unwrap();
        let total_after = setup_ct.load(Ordering::Relaxed)
            + sym_ct.load(Ordering::Relaxed)
            + num_ct.load(Ordering::Relaxed);
        assert_eq!(total_after, total_before + 1);
    }

    #[cfg(feature = "dense-direct")]
    #[cfg(not(feature = "complex"))]
    #[test]
    fn preonly_with_lu_pc_solves() {
        // Simple 2x2 SPD: [2 1; 1 2]
        let a = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 1.0 });
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Preonly).unwrap();
        ksp.set_pc_type(PcType::Lu, None).unwrap();
        ksp.set_operators(Arc::new(a), None);

        let b: Vec<R> = vec![3.0, 3.0];
        let mut x = vec![0.0; 2];
        let stats = ksp.solve(&b, &mut x).unwrap();

        // Verify Ax ≈ b using the stored operator
        let amat = ksp.amat.as_ref().unwrap().clone();
        let mut ax = vec![0.0; 2];
        amat.matvec(&x, &mut ax);
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < R::from_real(1e-10));
        }
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.reason, ConvergedReason::ConvergedAtol);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn preonly_without_direct_pc_errors() {
        let a = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Preonly).unwrap();
        // Jacobi is not a direct solver
        ksp.set_pc_type(PcType::Jacobi, None).unwrap();
        ksp.set_operators(Arc::new(a), None);

        let b: Vec<R> = vec![1.0, 2.0];
        let mut x = vec![0.0; 2];

        let err = ksp.solve(&b, &mut x).unwrap_err();
        match err {
            KError::SolveError(msg) => {
                assert!(msg.to_lowercase().contains("direct"))
            }
            _ => panic!("unexpected error type: {:?}", err),
        }
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn try_set_operators_ok_same_comm() {
        let m = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        // NoComm for both => equal
        let a = Arc::new(DenseOp::<f64>::new(Arc::new(m)));
        let mut ksp = KspContext::new();
        ksp.try_set_operators(a.clone(), None).unwrap();
        assert_eq!(ksp.is_setup(), false); // setup is invalidated but not run
    }

    // Congruent communicators should be accepted (MPI_Comm_dup).
    #[cfg(feature = "mpi")]
    #[test]
    #[cfg(not(feature = "complex"))]
    fn try_set_operators_allows_congruent_dup_comm() {
        use crate::parallel::MpiComm;

        let _guard = mpi_test_guard();
        let m = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let op = Arc::new(DenseOp::<f64>::new(Arc::new(m)));

        let world = std::sync::Arc::new(MpiComm::new());
        let dup = std::sync::Arc::new(world.dup());

        let a = wrap_with_comm(op.clone(), crate::parallel::UniverseComm::Mpi(world));
        let p = wrap_with_comm(op, crate::parallel::UniverseComm::Mpi(dup));

        let mut ksp = KspContext::new();
        ksp.try_set_operators(a, Some(p)).unwrap();
    }

    // This one is only meaningful when MPI is enabled (distinct comms are possible).
    #[cfg(feature = "mpi")]
    #[test]
    #[cfg(not(feature = "complex"))]
    fn try_set_operators_err_mismatched_comm() {
        use crate::parallel::{Comm as _, MpiComm};

        let _guard = mpi_test_guard();
        // Build a small dense and wrap with different communicators
        let m = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let op = Arc::new(DenseOp::<f64>::new(Arc::new(m)));

        let world = std::sync::Arc::new(MpiComm::new());
        if world.size() < 2 {
            return;
        }
        let comm_a = crate::parallel::UniverseComm::Mpi(world.clone());
        let comm_b = world.split((world.rank() % 2) as i32, world.rank() as i32); // different group

        let a_comm = wrap_with_comm(op.clone(), comm_a.clone());
        let p_comm = wrap_with_comm(op.clone(), comm_b.clone());

        let mut ksp = KspContext::new();
        let err = match ksp.try_set_operators(a_comm, Some(p_comm)) {
            Err(e) => e,
            Ok(_) => panic!("expected communicator mismatch error"),
        };
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("communicator mismatch"))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }

    #[cfg(feature = "mpi")]
    #[test]
    #[cfg(not(feature = "complex"))]
    fn try_set_operators_with_comm_rejects_noncongruent_override() {
        use crate::parallel::{Comm as _, MpiComm, UniverseComm};

        let _guard = mpi_test_guard();
        let m = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let op = Arc::new(DenseOp::<f64>::new(Arc::new(m)));

        let world = std::sync::Arc::new(MpiComm::new());
        if world.size() < 2 {
            return;
        }
        let world_comm = UniverseComm::Mpi(world.clone());
        let sub = world.split((world.rank() % 2) as i32, world.rank() as i32);

        let op_world = wrap_with_comm(op, world_comm);

        let mut ksp = KspContext::new();
        let err = ksp
            .try_set_operators_with_comm(op_world, None, sub)
            .unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(msg.contains("override"));
    }

    #[cfg(feature = "mpi")]
    #[test]
    #[cfg(not(feature = "complex"))]
    fn try_set_operators_with_comm_allows_trivial_override() {
        use crate::parallel::{MpiComm, UniverseComm};

        let _guard = mpi_test_guard();
        let m = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let op = Arc::new(DenseOp::<f64>::new(Arc::new(m))); // NoComm by default

        let world = std::sync::Arc::new(MpiComm::new());
        let comm = UniverseComm::Mpi(world);

        let mut ksp = KspContext::new();
        ksp.try_set_operators_with_comm(op, None, comm).unwrap();
    }

    #[cfg(feature = "mpi")]
    #[test]
    #[cfg(not(feature = "complex"))]
    fn monitor_policy_rank0_only_invokes_on_root() {
        use crate::parallel::MpiComm;

        let _guard = mpi_test_guard();
        let world = std::sync::Arc::new(MpiComm::new());
        if world.size() < 2 {
            return;
        }
        let comm = crate::parallel::UniverseComm::Mpi(world.clone());

        let m = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let op = Arc::new(DenseOp::<f64>::new(Arc::new(m)));
        let op_comm = wrap_with_comm(op, comm);

        let mut ksp = KspContext::new();
        ksp.try_set_operators(op_comm, None).unwrap();

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_handle = Arc::clone(&counter);
        ksp.add_monitor_rank0(move |_, _, _| {
            counter_handle.fetch_add(1, Ordering::Relaxed);
            crate::solver::MonitorAction::Continue
        });

        ksp.invoke_monitors(0, 1.0, 0);

        if world.rank() == 0 {
            assert_eq!(counter.load(Ordering::Relaxed), 1);
        } else {
            assert_eq!(counter.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn fgmres_rejects_left_via_try_set_pc_side() {
        let mut ksp = KspContext::new();
        ksp.set_pc_side(PcSide::Right);
        ksp.set_type(SolverType::Fgmres).unwrap();
        let err = ksp
            .try_set_pc_side(PcSide::Left)
            .expect_err("FGMRES must reject left preconditioning");
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("fgmres"));
                assert!(msg.to_lowercase().contains("right preconditioning"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn fgmres_effective_side_tracks_requested_side() {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Fgmres).unwrap();

        ksp.try_set_pc_side(PcSide::Right).unwrap();
        assert_eq!(ksp.effective_pc_side(), PcSide::Right);
    }

    #[test]
    fn set_side_then_set_type_fails_fast() {
        let mut ksp = KspContext::new();
        // Start with a right side (illegal for most solvers)
        ksp.set_pc_side(PcSide::Right); // allowed until a solver constrains it
        // Now pick a left-only solver; we expect an error
        let err = match ksp.set_type(SolverType::Cg) {
            Ok(_) => panic!("expected CG to reject right preconditioning"),
            Err(e) => e,
        };
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("cg"));
                assert!(msg.to_lowercase().contains("left"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn gmres_accepts_both_sides() {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.try_set_pc_side(PcSide::Left).unwrap();
        ksp.try_set_pc_side(PcSide::Right).unwrap();
        // Symmetric is normalized to Left; should pass
        ksp.try_set_pc_side(PcSide::Symmetric).unwrap();
    }

    #[test]
    fn cg_requires_left_side_in_context() {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Cg).unwrap();
        let err = match ksp.try_set_pc_side(PcSide::Right) {
            Ok(_) => panic!("expected CG to reject right preconditioning"),
            Err(e) => e,
        };
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("cg"));
                assert!(msg.to_lowercase().contains("left"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn pcg_requires_left_side_in_context() {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Pcg).unwrap();
        let err = match ksp.try_set_pc_side(PcSide::Right) {
            Ok(_) => panic!("expected PCG to reject right preconditioning"),
            Err(e) => e,
        };
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("pcg"));
                assert!(msg.to_lowercase().contains("left"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn gmres_options_apply_immediately_and_when_staged() {
        use crate::context::ksp_context::ReorthPolicy;
        use crate::solver::gmres::{GmresOrthog, GmresSolver};
        let mut ksp = KspContext::new();

        // Stage opts before type
        let opts = KspOptions {
            gmres_restart: Some(47),
            gmres_orthog: Some("mgs".into()),
            gmres_reorth: Some("always".into()),
            gmres_reorth_tol: Some(0.55),
            gmres_happy_breakdown: Some(true),
            ..Default::default()
        };
        ksp.set_from_options(&opts).unwrap();
        ksp.set_type(SolverType::Gmres).unwrap();

        let s = ksp
            .solver
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<GmresSolver>()
            .unwrap();

        let (restart, orth, reo, hb) = s.debug_config();
        assert_eq!(restart, 47);
        assert_eq!(orth, GmresOrthog::Mgs);
        assert!(reo);
        assert!(hb);
        assert!(matches!(s.reorth, ReorthPolicy::Always));
        assert!((s.reorth_tol - 0.55).abs() < 1e-12);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn complex_accepts_pipelined_gmres_at_options_parse_time() {
        let mut ksp = KspContext::new();
        let opts = KspOptions {
            ksp_type: Some("gmres".into()),
            gmres_variant: Some("pipelined".into()),
            ..Default::default()
        };
        assert!(ksp.set_from_options(&opts).is_ok());
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn fgmres_options_apply() {
        use crate::context::ksp_context::ReorthPolicy;
        use crate::solver::fgmres::{FgmresSolver, FgmresVariant, OrthogMethod};
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Fgmres).unwrap();

        let opts = KspOptions {
            fgmres_restart: Some(25),
            fgmres_orthog: Some("cgs".into()),
            fgmres_reorth: Some("never".into()),
            fgmres_reorth_tol: Some(0.42),
            fgmres_happy_breakdown: Some(true),
            fgmres_variant: Some("pipelined".into()),
            ..Default::default()
        };
        ksp.set_from_options(&opts).unwrap();

        let s = ksp
            .solver
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<FgmresSolver>()
            .unwrap();
        let (restart, orth, reo, hb) = s.debug_config();
        assert_eq!(restart, 25);
        assert_eq!(orth, OrthogMethod::ClassicalGS);
        assert!(!reo);
        assert!(hb);
        assert_eq!(s.variant, FgmresVariant::Pipelined);
        assert!(matches!(s.reorth, ReorthPolicy::Never));
        assert!((s.reorth_tol - 0.42).abs() < 1e-12);
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn fgmres_orthog_cgs_refined_maps_cleanly() {
        use crate::solver::fgmres::{FgmresSolver, OrthogMethod};
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Fgmres).unwrap();
        let opts = KspOptions {
            fgmres_orthog: Some("cgs_refined".into()),
            ..Default::default()
        };
        ksp.set_from_options(&opts).unwrap();
        let s = ksp
            .solver
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<FgmresSolver>()
            .unwrap();
        let (_, orth, _, _) = s.debug_config();
        assert_eq!(orth, OrthogMethod::ClassicalGS);
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn fgmres_options_reject_non_right_pc_side() {
        let mut ksp = KspContext::new();
        let opts = KspOptions {
            ksp_type: Some("fgmres".into()),
            pc_side: Some("left".into()),
            ..Default::default()
        };
        let err = ksp
            .set_from_options(&opts)
            .expect_err("FGMRES must reject left side from options");
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.contains("FGMRES"));
                assert!(msg.to_lowercase().contains("right preconditioning"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn fgmres_options_apply_accepts_pipelined_for_complex() {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Fgmres).unwrap();

        let opts = KspOptions {
            fgmres_variant: Some("pipelined".into()),
            ..Default::default()
        };
        assert!(ksp.set_from_options(&opts).is_ok());
    }

    #[test]
    fn reduction_option_updates_workspace() {
        use crate::reduction::ReproMode;
        let mut ksp = KspContext::new();
        ksp.work = Some(Workspace::new(4));
        let opts = KspOptions {
            reduction: Some("deterministic".into()),
            ..Default::default()
        };
        ksp.set_from_options(&opts).unwrap();
        let ws = ksp.work.as_ref().unwrap();
        assert!(matches!(
            ws.reduction_options().mode,
            ReproMode::Deterministic
        ));

        let opts = KspOptions {
            reduction: Some("deterministic-accurate".into()),
            ..Default::default()
        };
        ksp.set_from_options(&opts).unwrap();
        let ws = ksp.work.as_ref().unwrap();
        assert!(matches!(
            ws.reduction_options().mode,
            ReproMode::DeterministicAccurate
        ));
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn reproducible_ksp_disables_overlap_reduction_counters() {
        fn run_once(ksp_type: &str, variant_key: &str, variant_value: &str) -> (Vec<f64>, usize) {
            let a = Arc::new(poisson_2d(6, 6));
            let op = Arc::new(CsrOp::<R>::new(a.clone()));

            let mut ksp = KspContext::new();
            ksp.set_operators(op, None);
            let solver_type = match ksp_type {
                "gmres" => SolverType::Gmres,
                "fgmres" => SolverType::Fgmres,
                "bicgstab" => SolverType::BiCgStab,
                _ => panic!("unexpected solver type"),
            };
            ksp.set_type(solver_type).expect("set type");

            let mut opts = KspOptions {
                reproducible: Some(true),
                reduction: Some("deterministic".into()),
                maxits: Some(20),
                rtol: Some(1e-10),
                ..Default::default()
            };
            match variant_key {
                "gmres_variant" => opts.gmres_variant = Some(variant_value.into()),
                "fgmres_variant" => opts.fgmres_variant = Some(variant_value.into()),
                "bicgstab_variant" => opts.bicgstab_variant = Some(variant_value.into()),
                _ => {}
            }
            ksp.set_from_options(&opts).expect("set options");
            ksp.setup().expect("setup");

            let n = a.nrows();
            let b = vec![S::from_real(1.0); n];
            let mut x = vec![S::zero(); n];
            let stats = ksp.solve(&b, &mut x).expect("solve");
            let real_x: Vec<f64> = x.iter().map(|v| v.real()).collect();
            (real_x, stats.counters.overlap_global_reductions)
        }

        for (solver, key, value) in [
            ("gmres", "gmres_variant", "pipelined"),
            ("fgmres", "fgmres_variant", "pipelined"),
            ("bicgstab", "bicgstab_variant", "fewerchecks"),
        ] {
            let (x1, overlap1) = run_once(solver, key, value);
            let (x2, overlap2) = run_once(solver, key, value);
            assert_eq!(overlap1, 0, "{solver} overlap counter must be zero");
            assert_eq!(overlap2, 0, "{solver} overlap counter must be zero");
            let bits1: Vec<u64> = x1.iter().map(|v| v.to_bits()).collect();
            let bits2: Vec<u64> = x2.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                bits1, bits2,
                "{solver} reproducible run must be bitwise stable"
            );
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn set_from_options_threads_context_does_not_touch_global_rayon() {
        let _guard = rayon_test_guard();
        crate::algebra::parallel::reset_global_rayon_config_calls();
        let mut ksp = KspContext::new();
        let opts = KspOptions {
            threads: Some(4),
            ..Default::default()
        };
        ksp.set_from_options(&opts).unwrap();
        assert_eq!(crate::algebra::parallel::global_rayon_config_calls(), 0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn set_from_options_threads_global_touches_global_rayon() {
        let _guard = rayon_test_guard();
        crate::algebra::parallel::reset_global_rayon_config_calls();
        let mut ksp = KspContext::new();
        let opts = KspOptions {
            threads: Some(2),
            threads_mode: Some("global".into()),
            ..Default::default()
        };
        ksp.set_from_options(&opts).unwrap();
        assert_eq!(crate::algebra::parallel::global_rayon_config_calls(), 1);
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn chain_allows_amg_then_jacobi() {
        let a = Arc::new(poisson_2d(8, 8));
        let op = Arc::new(CsrOp::<R>::new(a.clone()));

        let mut ksp = KspContext::new();
        ksp.set_operators(op.clone(), None);

        let ksp_opts = KspOptions {
            ksp_type: Some("gmres".into()),
            ..Default::default()
        };

        let pc_opts = PcOptions {
            chain: Some(vec![
                PcOptions {
                    pc_type: Some("amg".into()),
                    ..Default::default()
                },
                PcOptions {
                    pc_type: Some("jacobi".into()),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        };

        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
        ksp.setup().unwrap();

        let n = op.dims().0;
        let b = vec![S::from_real(1.0); n];
        let mut x = vec![S::zero(); n];
        match ksp.solve(&b, &mut x) {
            Ok(stats) => assert!(stats.final_residual.is_finite()),
            Err(KError::Unsupported(msg)) => {
                assert!(cfg!(feature = "complex"));
                assert!(msg.to_lowercase().contains("complex"));
            }
            Err(err) => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn asm_with_amg_block_solver_builds_and_runs() {
        let a = Arc::new(poisson_2d(8, 8));
        let op = Arc::new(CsrOp::<R>::new(a.clone()));

        let mut ksp = KspContext::new();
        ksp.set_operators(op.clone(), None);

        let ksp_opts = KspOptions {
            ksp_type: Some("gmres".into()),
            ..Default::default()
        };

        let pc_opts = PcOptions {
            pc_type: Some("asm".into()),
            asm_block_solver: Some("amg".into()),
            asm_overlap: Some(1),
            ..Default::default()
        };

        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
        ksp.setup().unwrap();

        let n = op.dims().0;
        let b = vec![S::from_real(1.0); n];
        let mut x = vec![S::zero(); n];
        match ksp.solve(&b, &mut x) {
            Ok(_) => {}
            Err(KError::Unsupported(msg)) => {
                assert!(cfg!(feature = "complex"));
                assert!(msg.to_lowercase().contains("complex"));
            }
            Err(err) => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn complex_chain_reports_stage_for_amg() {
        let a = Arc::new(poisson_2d(4, 4));
        let op = Arc::new(CsrOp::<R>::new(a.clone()));

        let mut ksp = KspContext::new();
        ksp.set_operators(op.clone(), None);

        let ksp_opts = KspOptions {
            ksp_type: Some("gmres".into()),
            ..Default::default()
        };

        let pc_opts = PcOptions {
            chain: Some(vec![PcOptions {
                pc_type: Some("amg".into()),
                ..Default::default()
            }]),
            ..Default::default()
        };

        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();

        match ksp.setup() {
            Ok(()) => {
                let n = op.dims().0;
                let b = vec![S::from_real(1.0); n];
                let mut x = vec![S::zero(); n];
                match ksp.solve(&b, &mut x) {
                    Ok(_) => {}
                    Err(KError::Unsupported(msg)) => {
                        assert!(msg.to_lowercase().contains("complex"));
                    }
                    Err(err) => panic!("unexpected error: {err:?}"),
                }
            }
            Err(err) => {
                let msg = err.to_string().to_lowercase();
                assert!(msg.contains("stage 0"));
                assert!(msg.contains("amg"));
            }
        }
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn solve_breakdown_reason_can_be_accepted_with_true_residual_override() {
        let a = Mat::<R>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.set_operators(Arc::new(a), None);
        ksp.set_pc_box_for_tests(Box::new(BreakdownApplyPc));
        ksp.setup().expect("setup");

        let b = vec![1.0, -2.0];
        let mut x = b.clone();
        let stats = ksp.solve(&b, &mut x).expect("solve");

        assert_eq!(stats.reason, ConvergedReason::DivergedBreakdown);
        assert_eq!(stats.acceptance_status, AcceptanceStatus::OkWithWarning);
        assert_eq!(
            stats.breakdown_reason,
            Some(ConvergedReason::DivergedBreakdown)
        );
        assert!(
            stats
                .residual_override_note
                .as_deref()
                .unwrap_or("")
                .contains("accepted_with_warning")
        );
    }

    #[test]
    #[cfg(not(feature = "complex"))]
    fn bicgstab_add20_breakdown_prone_case_reports_transparent_diagnostics() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/mtx");
        let matrix = read_matrix_market(root.join("add20.mtx")).expect("read add20 matrix");
        let rhs = read_matrix_market(root.join("add20_rhs1.mtx")).expect("read add20 rhs");
        let a = Arc::new(CsrOp::<R>::new(Arc::new(
            matrix.to_csr_matrix().expect("add20 to csr"),
        )));
        let b = rhs
            .values
            .iter()
            .copied()
            .map(S::from_real)
            .collect::<Vec<_>>();
        let mut x = vec![S::zero(); b.len()];

        let mut ksp = KspContext::new();
        ksp.set_operators(a, None);
        ksp.set_type(SolverType::BiCgStab).expect("set type");
        ksp.rtol = 1e-8;
        ksp.atol = 1e-12;
        ksp.maxits = 500;
        let opts = KspOptions {
            bicgstab_variant: Some("fewerchecks".into()),
            ..Default::default()
        };
        ksp.set_from_options(&opts).expect("set opts");
        ksp.setup().expect("setup");

        let stats = ksp.solve(&b, &mut x).expect("solve");
        assert!(stats.final_residual.is_finite());
        if stats.reason.is_diverged() {
            assert_eq!(stats.breakdown_reason, Some(stats.reason));
            if matches!(stats.acceptance_status, AcceptanceStatus::OkWithWarning) {
                assert!(stats.residual_override_note.is_some());
            }
        }
    }
}
