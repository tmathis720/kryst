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
//! | `FGMRES`          | Right only      | Flexible right-only pipeline.                           |
//! | `PCA-GMRES`       | Left or Right   | Mapped to [`PcaPcMode`] during setup.                   |
//! | All other solvers | Left or Right   | `Symmetric` is normalized to `Left` before dispatch.    |
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
use crate::preconditioner::dist::MpiPcOptions;
use crate::error::KError;
use crate::matrix::backend::materialize;
use crate::matrix::op::{LinOp, StructureId, ValuesId, wrap_with_comm};
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use crate::matrix::DistCsrOp;
#[cfg(feature = "complex")]
use crate::ops::klinop::KLinOp;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::parallel::Comm;
use crate::preconditioner::{PcReusePolicy, PcSide, Preconditioner};
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use crate::preconditioner::dist::{DistPcAdapter, DistPcBuilder, GlobalPcKind, MpiPcOptions};
use crate::reduction::ReproMode;
use crate::solver::{
    BiCgStabSolver, CgSolver, CgnrSolver, CgsSolver, FgmresSolver, GmresSolver, LinearSolver,
    MinresSolver, PCG_PIPELINED_DEFAULT_REPLACE_EVERY, PcaGmresSolver, PcaPcMode, PcgSolver,
    PcgVariant, QmrSolver, TfqmrSolver,
};
use crate::utils::convergence::{ConvergedReason, SolveStats};
use crate::utils::reduction::ReductOptions;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
mod execution;
mod workspace;
pub use crate::core::block::BlockVec;
pub use execution::{ExecutionPolicy, ThreadingPolicy};
pub use workspace::{GmresSStepWorkspace, GmresSpec, PipeReduct, ReorthPolicy, Workspace};

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
    PcaGmres,
    Qmr,
    Tfqmr,
    Preonly,
}

impl SolverType {
    /// Return the preconditioning side required by this solver, if any.
    #[inline]
    pub fn required_pc_side(self) -> Option<PcSide> {
        match self {
            SolverType::Cg | SolverType::Pcg => Some(PcSide::Left),
            _ => None,
        }
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
            "pca_gmres" | "pcagmres" => Ok(SolverType::PcaGmres),
            "qmr" => Ok(SolverType::Qmr),
            "tfqmr" => Ok(SolverType::Tfqmr),
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

/// Minimal KSP context holding solver, preconditioner, and operators.
pub struct KspContext {
    solver: Option<Box<dyn LinearSolver<Error = KError> + 'static>>,
    pc: Option<Box<dyn Preconditioner>>,
    pub(crate) pending_pc: Option<DeferredPcInfo>,
    pub(crate) pending_chain: Option<Vec<DeferredPcInfo>>,
    amat: Option<Arc<dyn LinOp<S = S>>>,
    pmat: Option<Arc<dyn LinOp<S = S>>>,
    bound_comm: Option<crate::parallel::UniverseComm>,
    work: Option<Workspace>,
    setup_called: bool,
    monitors: Vec<Box<dyn Fn(usize, R) + Send + Sync>>,
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
    pending_mpi_pc: Option<PendingMpiPc>,
    // Pending/staged solver-specific options to apply when solver type is set
    pending_gmres: PendingGmres,
    pending_fgmres: PendingFgmres,
    pending_pcg: PendingPcg,
}

impl fmt::Debug for KspContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KspContext")
            .field("solver", &self.solver.as_ref().map(|_| "set"))
            .field("pc", &self.pc.as_ref().map(|_| "set"))
            .field("pending_pc", &self.pending_pc)
            .field("pending_chain", &self.pending_chain)
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
            .field("pending_mpi_pc", &self.pending_mpi_pc)
            .field("pending_gmres", &self.pending_gmres)
            .field("pending_fgmres", &self.pending_fgmres)
            .field("pending_pcg", &self.pending_pcg)
            .finish()
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
    variant: Option<PendingGmresVariant>,
    sstep: Option<usize>,
    sstep_max_cond: Option<R>,
}

#[derive(Clone, Debug, Default)]
struct PendingFgmres {
    restart: Option<usize>,
    orthog: Option<crate::solver::fgmres::Orthog>,
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

#[derive(Clone, Debug)]
struct PendingMpiPc {
    mpi_opts: MpiPcOptions,
    pc_opts: PcOptions,
}

impl Default for KspContext {
    fn default() -> Self {
        Self::new()
    }
}

impl KspContext {
    #[inline]
    fn normalize_side(side: PcSide) -> PcSide {
        match side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        }
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
        let normalized = Self::normalize_side(side);
        if let Some(st) = self.solver_type {
            if let Some(required) = st.required_pc_side() {
                if normalized != required {
                    return Err(KError::InvalidInput(format!(
                        "{st:?} requires left preconditioning; got {side:?}"
                    )));
                }
            } else if matches!(st, SolverType::Fgmres) {
                if normalized != PcSide::Right {
                    return Err(KError::InvalidInput(
                        "FGMRES only supports right preconditioning".into(),
                    ));
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
            pending_chain: None,
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
            pending_mpi_pc: None,
            pending_gmres: PendingGmres::default(),
            pending_fgmres: PendingFgmres::default(),
            pending_pcg: PendingPcg::default(),
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
        if let Some(required) = solver_type.required_pc_side() {
            let normalized = Self::normalize_side(self.pc_side);
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
            SolverType::BiCgStab => Some(Box::new(BiCgStabSolver::new(self.rtol, self.maxits))),
            SolverType::Cgs => Some(Box::new(CgsSolver::new(self.rtol, self.maxits))),
            SolverType::Pcg => Some(Box::new({
                let mut s = PcgSolver::new(self.rtol, self.maxits)
                    .with_norm(crate::solver::pcg::CgNormType::Preconditioned);
                self.apply_pcg_pending_to(&mut s);
                s
            })),
            SolverType::Minres => Some(Box::new(MinresSolver::new(self.rtol, self.maxits))),
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
        match PcFactory::create_preconditioner(pc_type, opts) {
            Ok(pc) => {
                self.pc = Some(pc);
                self.pending_pc = None;
                self.pending_chain = None;
            }
            Err(_) => {
                let spec = PcFactory::create_deferred_pc(pc_type, opts.cloned())?;
                self.pc = None;
                self.pending_pc = Some(spec);
                self.pending_chain = None;
            }
        }
        self.invalidate_pc_setup();
        Ok(self)
    }

    pub fn set_pc_type_from_str(&mut self, pc_type: &str) -> Result<&mut Self, KError> {
        let pct = PcType::from_str(pc_type)?;
        self.set_pc_type(pct, None)
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
            if let Some(mode) = opts.threads_mode.as_deref() {
                match mode {
                    "context" => {}
                    "global" => {}
                    "serial" => {
                        self.exec.threading = ThreadingPolicy::Serial;
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
                    "context" => {
                        self.exec = self.exec.clone().with_threads(n)?;
                    }
                    "serial" => {
                        self.exec.threading = ThreadingPolicy::Serial;
                    }
                    "global" => {
                        set_rayon_threads(n);
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
            set_parallel_tune(tune);
        }

        if let Some(ref t) = opts.ksp_type {
            let st = SolverType::from_str(t)?;
            self.set_type(st)?;
        }
        if let Some(rtol) = opts.rtol {
            self.rtol = rtol;
        }
        if let Some(atol) = opts.atol {
            self.atol = atol;
        }
        if let Some(dtol) = opts.dtol {
            self.dtol = dtol;
        }
        if let Some(maxits) = opts.maxits {
            self.maxits = maxits;
        }
        if let Some(restart) = opts.restart {
            self.restart = restart;
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
                    "mgs" => crate::solver::gmres::GmresOrthog::Mgs,
                    "cgs" => crate::solver::gmres::GmresOrthog::Cgs,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_gmres_orthog: {other} (expected 'mgs'|'cgs')"
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
                    "mgs" => crate::solver::gmres::GmresOrthog::Mgs,
                    "cgs" => crate::solver::gmres::GmresOrthog::Cgs,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_gmres_orthog: {other} (expected 'mgs'|'cgs')"
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
            // Map "mgs"/"cgs" to Modified/Classical
            if let Some(ref orth) = opts.fgmres_orthog {
                let o = match orth.as_str() {
                    "mgs" => crate::solver::fgmres::Orthog::Modified,
                    "cgs" => crate::solver::fgmres::Orthog::Classical,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_orthog: {other} (expected 'mgs'|'cgs')"
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
                    "mgs" => crate::solver::fgmres::Orthog::Modified,
                    "cgs" => crate::solver::fgmres::Orthog::Classical,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_orthog: {other} (expected 'mgs'|'cgs')"
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
        }
        let mut pcg_pending_updated = false;
        if let Some(variant) = requested_cg_variant {
            if matches!(variant, CgVariant::Pipelined)
                && Self::normalize_side(self.pc_side) != PcSide::Left
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
            let specs = PcFactory::create_deferred_pc_chain_from_options(chain_opts)?;
            self.pc = None;
            self.pending_pc = None;
            self.pending_chain = Some(specs);
            self.invalidate_pc_setup();
        }
        self.pending_mpi_pc = Some(PendingMpiPc {
            mpi_opts: pc_opts.mpi_pc_options()?,
            pc_opts: pc_opts.clone(),
        });
        Ok(self)
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

    pub fn last_pc_sid(&self) -> Option<StructureId> {
        self.last_pc_sid
    }
    pub fn last_pc_vid(&self) -> Option<ValuesId> {
        self.last_pc_vid
    }

    /// Prepare preconditioner and workspace.
    pub fn setup(&mut self) -> Result<(), KError> {
        let exec = self.exec.clone();
        exec.install(|| self.setup_impl())
    }

    fn setup_impl(&mut self) -> Result<(), KError> {
        let pmat = self
            .pmat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Pmat not set".into()))?;
        let amat = self
            .amat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;

        let (m, n) = amat.dims();
        let (pm, pn) = pmat.dims();
        let allow_rect = matches!(self.solver_type, Some(SolverType::Cgnr));
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
                "setup start: comm_id={} size={} rank={} dims=({},{}) pc_dims=({},{}) pc_reuse={:?} solver={:?} pc_side={:?} A_ids=({:?},{:?}) P_ids=({:?},{:?})",
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
                amat.structure_id(),
                amat.values_id(),
                pmat.structure_id(),
                pmat.values_id()
            );
        }

        if self.pc.is_none() {
            #[cfg(all(not(feature = "complex"), feature = "mpi"))]
            {
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
                        self.pending_pc = None;
                        self.pending_chain = None;
                    }
                }
            }
            if let Some(specs) = self.pending_chain.take() {
                let chain = PcFactory::construct_deferred_pc_chain(specs, pmat.as_ref())?;
                self.pc = Some(chain);
            } else if let Some(spec) = self.pending_pc.take() {
                let pc = PcFactory::construct_deferred_preconditioner(spec, pmat.as_ref())?;
                self.pc = Some(pc);
            }
        }

        let sid = {
            let id = pmat.structure_id();
            if id.0 != 0 {
                id
            } else {
                StructureId(Arc::as_ptr(pmat) as *const () as usize as u64)
            }
        };
        let vid = pmat.values_id();

        if self.pc.is_none() {
            // no factory hook here; assume pc set elsewhere
            self.last_pc_sid = None;
            self.last_pc_vid = None;
        }

        if let Some(pc) = self.pc.as_mut() {
            // Pre-convert once to the PC's requested format, preserving communicator.
            let want = pc.required_format();
            let tol = pc.preferred_drop_tol_for_format().unwrap_or_default();
            let pmat_view = materialize(pmat.clone(), want, tol)?;

            match self.last_pc_sid {
                None => {
                    pc.setup(pmat_view.as_ref())?;
                    self.last_pc_sid = Some(sid);
                    self.last_pc_vid = Some(vid);
                }
                Some(old_sid) if old_sid != sid => {
                    pc.update_symbolic(pmat_view.as_ref())?;
                    self.last_pc_sid = Some(sid);
                    self.last_pc_vid = Some(vid);
                }
                Some(_old_sid) => {
                    let vid_known = vid.0 != 0;
                    let values_changed = self.last_pc_vid != Some(vid);
                    match self.pc_reuse {
                        PcReusePolicy::Never => {
                            if !vid_known || values_changed {
                                pc.update_symbolic(pmat_view.as_ref())?;
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
                                pc.update_symbolic(pmat_view.as_ref())?;
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
                                pc.update_symbolic(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            }
                        }
                    }
                }
            }
        }

        let (m, _) = amat.dims();
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
        exec.install(|| self.solve_impl(b, x))
    }

    fn solve_impl(&mut self, b: &[S], x: &mut [S]) -> Result<SolveStats<R>, KError> {
        if !self.setup_called {
            self.setup_impl()?;
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

        let allow_rect = matches!(self.solver_type, Some(SolverType::Cgnr));
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
                "solve start: comm_id={} size={} rank={} dims=({},{}) rhs_len={} x_len={} solver={:?} pc_side={:?} pc_reuse={:?} A_ids=({:?},{:?}) P_ids=({:?},{:?})",
                comm.id(),
                comm.size(),
                comm.rank(),
                m,
                n,
                b.len(),
                x.len(),
                self.solver_type,
                self.pc_side,
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
            return Ok(SolveStats::new(0, res, ConvergedReason::ConvergedAtol));
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
            let mut stats = solver.solve(
                amat_ref,
                pc,
                b,
                x,
                self.pc_side,
                &comm,
                monitors,
                self.work.as_mut(),
            )?;

            let res = self.true_residual_norm_in_place(amat_ref, b, x)?;
            stats.final_residual = res;
            Ok(stats)
        }

        #[cfg(feature = "complex")]
        {
            let solver_type = self
                .solver_type
                .ok_or_else(|| KError::SolveError("No solver".into()))?;
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

            let mut stats = match solver_type {
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
                        self.pc_side,
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
                SolverType::Gmres => {
                    let s = solver
                        .as_any_mut()
                        .downcast_mut::<GmresSolver>()
                        .ok_or_else(|| KError::SolveError("GMRES solver missing".into()))?;
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
                        self.pc_side,
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
                    return Err(KError::Unsupported(
                        "PCG is not yet available for complex scalars".into(),
                    ));
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
                SolverType::Preonly => unreachable!("PREONLY handled earlier"),
            };

            let res = self.true_residual_norm_in_place(amat_ref, b, x)?;
            stats.final_residual = res;
            Ok(stats)
        }
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

    fn invalidate_solver_setup(&mut self) {
        self.setup_called = false;
    }

    fn invalidate_pc_setup(&mut self) {
        self.setup_called = false;
        self.reset_pc_ids();
    }

    #[cfg(all(not(feature = "complex"), feature = "mpi"))]
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
                PcFactory::create_preconditioner(PcType::Asm, Some(&pending.pc_opts))
            }
            GlobalPcKind::None => Err(KError::InvalidInput(
                "pc_global=none should not build a global PC".into(),
            )),
        }
    }

    /// Add an iteration monitor callback.
    pub fn add_monitor<F>(&mut self, f: F)
    where
        F: Fn(usize, R) + Send + Sync + 'static,
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
        F: Fn(usize, R) + Send + Sync + 'static,
    {
        self.monitors.push(Box::new(move |it, r| f(it, r)));
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
    pub fn invoke_monitors(&self, iter: usize, residual: R) {
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
            m(iter, residual);
        }
    }

    /// Set solver tolerances and maximum iterations.
    pub fn set_tolerances(&mut self, rtol: R, atol: R, dtol: R, maxits: usize) -> &mut Self {
        self.rtol = rtol;
        self.atol = atol;
        self.dtol = dtol;
        self.maxits = maxits;
        self.invalidate_solver_setup();
        self
    }

    /// Configure the underlying solver based on the requested preconditioning side.
    fn configure_pc_side(&mut self) -> Result<(), KError> {
        // Treat symmetric as left; only specialized PCs interpret it differently.
        let side = match self.pc_side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        };

        if let Some(SolverType::PcaGmres) = self.solver_type {
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|s| s.as_any_mut().downcast_mut::<PcaGmresSolver>())
            {
                s.pc_mode = match side {
                    PcSide::Left => PcaPcMode::Left,
                    PcSide::Right => PcaPcMode::Right,
                    PcSide::Symmetric => unreachable!(),
                };
            }
        }

        if let Some(st) = self.solver_type {
            if let Some(required) = st.required_pc_side() {
                if side != required {
                    return Err(KError::InvalidInput(format!(
                        "{st:?} requires left preconditioning; got {side:?}"
                    )));
                }
            } else if matches!(st, SolverType::Fgmres) {
                if side != PcSide::Right {
                    return Err(KError::InvalidInput(
                        "FGMRES only supports right preconditioning".into(),
                    ));
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
        ksp.add_monitor_rank0(move |_, _| {
            counter_handle.fetch_add(1, Ordering::Relaxed);
        });

        ksp.invoke_monitors(0, 1.0);

        if world.rank() == 0 {
            assert_eq!(counter.load(Ordering::Relaxed), 1);
        } else {
            assert_eq!(counter.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn fgmres_rejects_left_early_via_try_set_pc_side() {
        let mut ksp = KspContext::new();
        // Ensure current side is compatible so set_type succeeds
        ksp.set_pc_side(PcSide::Right);
        ksp.set_type(SolverType::Fgmres).unwrap();
        match ksp.try_set_pc_side(PcSide::Left) {
            Err(KError::InvalidInput(msg)) => assert!(msg.to_lowercase().contains("fgmres")),
            Err(other) => panic!("unexpected error type: {:?}", other),
            Ok(_) => panic!("expected error for incompatible FGMRES side"),
        }
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
        use crate::solver::fgmres::{FgmresSolver, FgmresVariant, Orthog};
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
        assert_eq!(orth, Orthog::Classical);
        assert!(!reo);
        assert!(hb);
        assert_eq!(s.variant, FgmresVariant::Pipelined);
        assert!(matches!(s.reorth, ReorthPolicy::Never));
        assert!((s.reorth_tol - 0.42).abs() < 1e-12);
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
}
