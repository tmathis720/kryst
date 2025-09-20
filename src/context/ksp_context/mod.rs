//! # KSP context
//!
//! ## Operator/PC lifecycle
//! 1. [`set_operators`] stores `A` and `P` (or `A` if `P` is `None`).
//! 2. Enforces communicator equality via [`LinOp::comm()`]. Prefer
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

use crate::config::options::{KspOptions, KspType, PcOptions};
use crate::context::pc_context::{DeferredPcInfo, PcFactory, PcType};
use crate::error::KError;
use crate::matrix::convert::materialize_linop_with_hint;
use crate::matrix::op::{LinOp, StructureId, ValuesId, wrap_with_comm};
use crate::parallel::Comm;
use crate::preconditioner::{PcReusePolicy, PcSide, Preconditioner};
use crate::solver::{
    BiCgStabSolver, CgSolver, CgnrSolver, CgsSolver, FgmresSolver, GmresSolver, LinearSolver,
    MinresSolver, PcaGmresSolver, PcaPcMode, PcgSolver,
};
use crate::utils::convergence::{ConvergedReason, SolveStats};
use std::str::FromStr;
use std::sync::Arc;
mod workspace;
pub use workspace::{GmresSpec, Workspace};

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

/// Minimal KSP context holding solver, preconditioner, and operators.
pub struct KspContext {
    solver: Option<Box<dyn LinearSolver<Error = KError> + 'static>>,
    pc: Option<Box<dyn Preconditioner>>,
    pub(crate) pending_pc: Option<DeferredPcInfo>,
    pub(crate) pending_chain: Option<Vec<DeferredPcInfo>>,
    amat: Option<Arc<dyn LinOp<S = f64>>>,
    pmat: Option<Arc<dyn LinOp<S = f64>>>,
    work: Option<Workspace>,
    setup_called: bool,
    monitors: Vec<Box<dyn Fn(usize, f64) + Send + Sync>>,
    solver_type: Option<SolverType>,
    pub rtol: f64,
    pub atol: f64,
    pub dtol: f64,
    pub maxits: usize,
    pub restart: usize,
    pub pc_side: PcSide,
    pc_side_explicit: bool,
    pc_reuse: PcReusePolicy,
    last_pc_sid: Option<StructureId>,
    last_pc_vid: Option<ValuesId>,
    // Pending/staged solver-specific options to apply when solver type is set
    pending_gmres: PendingGmres,
    pending_fgmres: PendingFgmres,
}

#[derive(Clone, Debug, Default)]
struct PendingGmres {
    restart: Option<usize>,
    orthog: Option<crate::solver::gmres::GmresOrthog>,
    reorthog: Option<bool>,
    happy_breakdown: Option<bool>,
}

#[derive(Clone, Debug, Default)]
struct PendingFgmres {
    restart: Option<usize>,
    orthog: Option<crate::solver::fgmres::Orthog>,
    reorthog: Option<bool>,
    happy_breakdown: Option<bool>,
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

    /// Validate that `side` is compatible with `solver_type` (if set).
    /// Mirrors `configure_pc_side()` logic but used at set-time to fail fast.
    fn check_pc_side_now(&self, side: PcSide) -> Result<(), KError> {
        let side = Self::normalize_side(side);
        if let Some(st) = self.solver_type {
            match st {
                SolverType::Fgmres => {
                    if side != PcSide::Right {
                        return Err(KError::InvalidInput(
                            "FGMRES only supports right preconditioning".into(),
                        ));
                    }
                }
                SolverType::BiCgStab | SolverType::Gmres | SolverType::PcaGmres => {
                    // both left and right are fine for these
                }
                _ => {
                    if side == PcSide::Right {
                        return Err(KError::InvalidInput(
                            "Selected solver only supports left preconditioning".into(),
                        ));
                    }
                }
            }
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
            work: None,
            setup_called: false,
            monitors: Vec::new(),
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
            pending_gmres: PendingGmres::default(),
            pending_fgmres: PendingFgmres::default(),
        }
    }

    pub fn set_type(&mut self, solver_type: SolverType) -> Result<&mut Self, KError> {
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
            SolverType::Pcg => Some(Box::new(
                PcgSolver::new(self.rtol, self.maxits)
                    .with_norm(crate::solver::pcg::CgNormType::Preconditioned),
            )),
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
        self.invalidate_setup();
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
        self.invalidate_setup();
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
        self.invalidate_setup();
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
            if let Some(flag) = opts.gmres_reorthog {
                s.set_reorthog(flag);
                self.pending_gmres.reorthog = Some(flag);
            }
            if let Some(flag) = opts.gmres_happy_breakdown {
                s.set_happy_breakdown(flag);
                self.pending_gmres.happy_breakdown = Some(flag);
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
            if let Some(flag) = opts.gmres_reorthog {
                self.pending_gmres.reorthog = Some(flag);
            }
            if let Some(flag) = opts.gmres_happy_breakdown {
                self.pending_gmres.happy_breakdown = Some(flag);
            }
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
            if let Some(flag) = opts.fgmres_reorthog {
                s.set_reorthog(flag);
                self.pending_fgmres.reorthog = Some(flag);
            }
            if let Some(flag) = opts.fgmres_happy_breakdown {
                s.set_happy_breakdown(flag);
                self.pending_fgmres.happy_breakdown = Some(flag);
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
            if let Some(flag) = opts.fgmres_reorthog {
                self.pending_fgmres.reorthog = Some(flag);
            }
            if let Some(flag) = opts.fgmres_happy_breakdown {
                self.pending_fgmres.happy_breakdown = Some(flag);
            }
        }

        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<CgSolver>())
        {
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
        }
        self.invalidate_setup();
        Ok(self)
    }

    fn apply_gmres_pending_to(&self, s: &mut GmresSolver) {
        if let Some(r) = self.pending_gmres.restart {
            s.set_restart(r);
        }
        if let Some(o) = self.pending_gmres.orthog {
            s.set_orthog(o);
        }
        if let Some(f) = self.pending_gmres.reorthog {
            s.set_reorthog(f);
        }
        if let Some(f) = self.pending_gmres.happy_breakdown {
            s.set_happy_breakdown(f);
        }
    }

    fn apply_fgmres_pending_to(&self, s: &mut FgmresSolver) {
        if let Some(r) = self.pending_fgmres.restart {
            s.set_restart(r);
        }
        if let Some(o) = self.pending_fgmres.orthog {
            s.set_orthog(o);
        }
        if let Some(f) = self.pending_fgmres.reorthog {
            s.set_reorthog(f);
        }
        if let Some(f) = self.pending_fgmres.happy_breakdown {
            s.set_happy_breakdown(f);
        }
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
            self.invalidate_setup();
        }
        Ok(self)
    }

    /// Assign the system and preconditioner operators.
    ///
    /// Returns an error if the communicators of `A` and `P` differ.
    /// `LinOp::comm()` is the single source of truth for parallel context;
    /// mismatches indicate a caller bug.
    ///
    /// On success, invalidates any prior setup (PC reuse and workspace).
    pub fn try_set_operators(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
    ) -> Result<&mut Self, KError> {
        let pmat = pmat.unwrap_or_else(|| amat.clone());
        let ac = amat.comm();
        let pc = pmat.comm();
        if ac != pc {
            self.invalidate_setup();
            return Err(KError::InvalidInput(format!(
                "Amat/Pmat communicator mismatch: A={}, P={}",
                ac.id(),
                pc.id()
            )));
        }
        self.amat = Some(amat);
        self.pmat = Some(pmat);
        self.invalidate_setup();
        Ok(self)
    }

    /// Like `try_set_operators`, but first wraps operators with an explicit communicator.
    pub fn try_set_operators_with_comm(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
        comm: crate::parallel::UniverseComm,
    ) -> Result<&mut Self, KError> {
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
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
    ) -> &mut Self {
        self.try_set_operators(amat, pmat).unwrap()
    }

    /// Like `set_operators`, but first wraps operators with an explicit communicator.
    /// Panics on communicator mismatch. Prefer
    /// [`KspContext::try_set_operators_with_comm`].
    pub fn set_operators_with_comm(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
        comm: crate::parallel::UniverseComm,
    ) -> &mut Self {
        self.try_set_operators_with_comm(amat, pmat, comm).unwrap()
    }

    pub fn set_pc_reuse_policy(&mut self, policy: PcReusePolicy) -> &mut Self {
        self.pc_reuse = policy;
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
        let pmat = self
            .pmat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Pmat not set".into()))?;
        let amat = self
            .amat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;

        if self.pc.is_none() {
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
            let hint = pc.required_format();
            let tol = pc.preferred_drop_tol_for_format().unwrap_or(0.0);
            let pmat_view = materialize_linop_with_hint(pmat.as_ref(), hint, tol)?;

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
                            } else if values_changed {
                                pc.update_symbolic(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            }
                        }
                    }
                }
            }
        }

        let (m, _) = amat.dims();
        if self
            .work
            .as_ref()
            .map(|w| w.tmp1.len() != m)
            .unwrap_or(true)
        {
            self.work = Some(Workspace::new(m));
            if let Some(ref mut solver) = self.solver
                && let Some(ref mut w) = self.work
            {
                solver.setup_workspace(w);
            }
        }
        self.setup_called = true;
        Ok(())
    }

    /// Solve the linear system using stored operators.
    pub fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<SolveStats<f64>, KError> {
        if !self.setup_called {
            self.setup()?;
        }
        if matches!(self.solver_type, Some(SolverType::Preonly)) {
            let pmat = self
                .pmat
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("Pmat not set".into()))?;
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
            return Ok(SolveStats {
                iterations: 1,
                final_residual: 0.0,
                reason: ConvergedReason::ConvergedAtol,
            });
        }

        // Configure solver preconditioning side and validate compatibility
        self.configure_pc_side()?;

        let amat = self
            .amat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;

        let monitors = if self.monitors.is_empty() {
            None
        } else {
            Some(self.monitors.as_slice())
        };
        let comm = amat.comm();
        let pc = self
            .pc
            .as_mut()
            .map(|b| b.as_mut() as &mut dyn Preconditioner);
        let solver = self
            .solver
            .as_mut()
            .ok_or_else(|| KError::SolveError("No solver".into()))?;
        let mut stats = solver.solve(
            amat.as_ref(),
            pc,
            b,
            x,
            self.pc_side,
            &comm,
            monitors,
            self.work.as_mut(),
        )?;

        // Compute true residual r = b - A x and use its norm for reporting
        let mut residual = vec![0.0f64; b.len()];
        if let Err(e) = amat.try_matvec(x, &mut residual) {
            return Err(KError::SolveError(format!("residual matvec failed: {e}")));
        }
        for (ri, &bi) in residual.iter_mut().zip(b.iter()) {
            *ri = bi - *ri;
        }
        let res_sq = comm.dot(&residual, &residual);
        stats.final_residual = res_sq.sqrt();
        Ok(stats)
    }

    fn invalidate_setup(&mut self) {
        self.setup_called = false;
        self.reset_pc_ids();
    }

    /// Add an iteration monitor callback.
    pub fn add_monitor<F>(&mut self, f: F)
    where
        F: Fn(usize, f64) + Send + Sync + 'static,
    {
        self.monitors.push(Box::new(f));
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
    }

    /// Invoke all monitors with the provided iteration and residual.
    pub fn invoke_monitors(&self, iter: usize, residual: f64) {
        for m in &self.monitors {
            m(iter, residual);
        }
    }

    /// Set solver tolerances and maximum iterations.
    pub fn set_tolerances(&mut self, rtol: f64, atol: f64, dtol: f64, maxits: usize) -> &mut Self {
        self.rtol = rtol;
        self.atol = atol;
        self.dtol = dtol;
        self.maxits = maxits;
        self.invalidate_setup();
        self
    }

    /// Configure the underlying solver based on the requested preconditioning side.
    fn configure_pc_side(&mut self) -> Result<(), KError> {
        // Treat symmetric as left; only specialized PCs interpret it differently.
        let side = match self.pc_side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        };

        match self.solver_type {
            Some(SolverType::PcaGmres) => {
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
            Some(SolverType::Fgmres) => {
                if side != PcSide::Right {
                    return Err(KError::SolveError(
                        "FGMRES only supports right preconditioning".into(),
                    ));
                }
            }
            Some(SolverType::Gmres) => {}
            _ => {
                if side == PcSide::Right {
                    return Err(KError::SolveError(
                        "Selected solver only supports left preconditioning".into(),
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
        self.invalidate_setup();
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::pc_context::PcType;
    use crate::matrix::op::DenseOp;
    use crate::preconditioner::PcSide;
    use faer::Mat;
    use std::sync::Arc;

    #[cfg(feature = "dense-direct")]
    #[test]
    fn preonly_with_lu_pc_solves() {
        // Simple 2x2 SPD: [2 1; 1 2]
        let a = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 1.0 });
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Preonly).unwrap();
        ksp.set_pc_type(PcType::Lu, None).unwrap();
        ksp.set_operators(Arc::new(a), None);

        let b = vec![3.0, 3.0];
        let mut x = vec![0.0; 2];
        let stats = ksp.solve(&b, &mut x).unwrap();

        // Verify Ax ≈ b using the stored operator
        let amat = ksp.amat.as_ref().unwrap().clone();
        let mut ax = vec![0.0; 2];
        amat.matvec(&x, &mut ax);
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-10);
        }
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.reason, ConvergedReason::ConvergedAtol);
    }

    #[test]
    fn preonly_without_direct_pc_errors() {
        let a = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Preonly).unwrap();
        // Jacobi is not a direct solver
        ksp.set_pc_type(PcType::Jacobi, None).unwrap();
        ksp.set_operators(Arc::new(a), None);

        let b = vec![1.0, 2.0];
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
    fn try_set_operators_ok_same_comm() {
        let m = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        // NoComm for both => equal
        let a = Arc::new(DenseOp::new(Arc::new(m)));
        let mut ksp = KspContext::new();
        ksp.try_set_operators(a.clone(), None).unwrap();
        assert_eq!(ksp.is_setup(), false); // setup is invalidated but not run
    }

    // This one is only meaningful when MPI is enabled (distinct comms are possible).
    #[cfg(feature = "mpi")]
    #[test]
    fn try_set_operators_err_mismatched_comm() {
        use crate::parallel::{Comm as _, MpiComm};

        // Build a small dense and wrap with different communicators
        let m = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let op = Arc::new(DenseOp::new(Arc::new(m)));

        let world = std::sync::Arc::new(MpiComm::new());
        let comm_a = crate::parallel::UniverseComm::Mpi(world.clone());
        let comm_b = world.split(1, world.rank() as i32); // different underlying handle

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
        // Now pick a left-only solver; we expect an unwrap panic due to Err
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ksp.set_type(SolverType::Cg).unwrap();
        }));
        assert!(
            res.is_err(),
            "expected panic due to incompatible side for CG"
        );
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
    fn gmres_options_apply_immediately_and_when_staged() {
        use crate::solver::gmres::{GmresOrthog, GmresSolver};
        let mut ksp = KspContext::new();

        // Stage opts before type
        let opts = KspOptions {
            gmres_restart: Some(47),
            gmres_orthog: Some("mgs".into()),
            gmres_reorthog: Some(true),
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
    }

    #[test]
    fn fgmres_options_apply() {
        use crate::solver::fgmres::{FgmresSolver, Orthog};
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Fgmres).unwrap();

        let opts = KspOptions {
            fgmres_restart: Some(25),
            fgmres_orthog: Some("cgs".into()),
            fgmres_reorthog: Some(false),
            fgmres_happy_breakdown: Some(true),
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
    }
}
