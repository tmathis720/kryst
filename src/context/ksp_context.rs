//! # KSP context
//!
//! ## Operator/PC lifecycle
//! 1. [`set_operators`] stores `A` and `P` (or `A` if `P` is `None`).
//! 2. Enforces **communicator equality** via [`LinOp::comm()`]. Mismatch aborts early.
//! 3. [`setup`] resolves any deferred PC specs (including chains), then calls
//!    [`Preconditioner::setup`] followed by reuse logic:
//!    - If structure id changed → [`update_symbolic`]
//!    - Else if values id changed and numeric reuse allowed → [`update_numeric`]
//!    - Else unchanged.
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

use crate::config::options::{KspOptions, PcOptions};
use crate::context::pc_context::{DeferredPcInfo, PcFactory, PcType};
use crate::error::KError;
use crate::matrix::op::{LinOp, StructureId, ValuesId, wrap_with_comm};
use crate::parallel::Comm;
use crate::preconditioner::{PcReusePolicy, PcSide, Preconditioner};
use crate::solver::{
    BiCgStabSolver, CgSolver, CgnrSolver, CgsSolver, FgmresSolver, GmresSolver, LinearSolver,
    MatSolverAdapter, MinresSolver, PcaGmresSolver, PcaPcMode, PcgSolver,
};
use crate::utils::convergence::{ConvergedReason, SolveStats};
use std::str::FromStr;
use std::sync::Arc;

/// Workspace placeholder reused by solvers.
#[derive(Debug)]
pub struct Workspace {
    pub tmp1: Vec<f64>,
    pub tmp2: Vec<f64>,
    pub q: Vec<Vec<f64>>,
    pub h: Vec<Vec<f64>>,
    pub cs: Vec<f64>,
    pub sn: Vec<f64>,
    pub g: Vec<f64>,
    /// Preconditioned basis vectors (Z) used by right-preconditioned solvers
    /// or flexible methods like FGMRES. Left-preconditioned solvers leave this
    /// empty.
    pub z: Vec<Vec<f64>>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        Self {
            tmp1: vec![0.0; n],
            tmp2: vec![0.0; n],
            q: Vec::new(),
            h: Vec::new(),
            cs: Vec::new(),
            sn: Vec::new(),
            g: Vec::new(),
            z: Vec::new(),
        }
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
    solver: Option<Box<dyn LinearSolver<Error = KError>>>,
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
    pc_reuse: PcReusePolicy,
    last_pc_sid: Option<StructureId>,
    last_pc_vid: Option<ValuesId>,
}

impl KspContext {
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
            rtol: 1e-6,
            atol: 1e-12,
            dtol: 1e3,
            maxits: 1000,
            restart: 30,
            pc_side: PcSide::Left,
            pc_reuse: PcReusePolicy::Auto,
            last_pc_sid: None,
            last_pc_vid: None,
        }
    }

    pub fn set_type(&mut self, solver_type: SolverType) -> Result<&mut Self, KError> {
        self.solver_type = Some(solver_type);
        let solver: Box<dyn LinearSolver<Error = KError>> = match solver_type {
            SolverType::Cg => Box::new(
                CgSolver::new(self.rtol, self.maxits)
                    .with_norm(crate::solver::cg::CgNormType::Preconditioned),
            ),
            SolverType::Cgnr => Box::new(CgnrSolver::new(self.rtol, self.maxits)),
            SolverType::Gmres => Box::new(GmresSolver::new(self.restart, self.rtol, self.maxits)),
            SolverType::Fgmres => Box::new(FgmresSolver::new(self.rtol, self.maxits, self.restart)),
            SolverType::BiCgStab => Box::new(MatSolverAdapter::new(BiCgStabSolver::new(
                self.rtol,
                self.maxits,
            ))),
            SolverType::Cgs => Box::new(CgsSolver::new(self.rtol, self.maxits)),
            SolverType::Pcg => Box::new(
                PcgSolver::new(self.rtol, self.maxits)
                    .with_norm(crate::solver::pcg::CgNormType::Preconditioned),
            ),
            SolverType::Minres => Box::new(MatSolverAdapter::new(MinresSolver::new(
                self.rtol,
                self.maxits,
            ))),
            SolverType::PcaGmres => {
                let mut s = PcaGmresSolver::new(self.restart, 1, 1, self.rtol, self.maxits);
                s.pc_mode = crate::solver::PcaPcMode::Left;
                Box::new(s)
            }
            SolverType::Qmr => Box::new(crate::solver::QmrSolver::new(self.rtol, self.maxits)),
            SolverType::Tfqmr => Box::new(crate::solver::TfqmrSolver::new(self.rtol, self.maxits)),
            SolverType::Preonly => {
                return Err(KError::SolveError("Preonly solver not available".into()));
            }
        };
        self.solver = Some(solver);
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

    /// Set the preconditioning side directly.
    pub fn set_pc_side(&mut self, side: PcSide) -> &mut Self {
        self.pc_side = side;
        self.invalidate_setup();
        self
    }

    /// Set the preconditioning side from a string ("left", "right", or "symmetric").
    pub fn set_pc_side_from_str(&mut self, side: &str) -> Result<&mut Self, KError> {
        let ps = PcSide::from_str(side)?;
        Ok(self.set_pc_side(ps))
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
            self.pc_side = PcSide::from_str(side)?;
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
            if let Some(flag) = opts.cg_single_reduction {
                s.set_single_reduction(flag);
            }
            if let Some(r) = opts.trust_region {
                s.set_trust_region(r);
            }
        }
        self.invalidate_setup();
        Ok(self)
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
            self.pc_side = PcSide::from_str(side)?;
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
    /// # Panics
    /// Panics if the communicators of `A` and `P` differ. `LinOp::comm()` is the
    /// single source of truth for parallel context; mismatches indicate a bug.
    pub fn set_operators(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
    ) -> &mut Self {
        let pmat = pmat.unwrap_or_else(|| amat.clone());
        let ac = amat.comm();
        let pc = pmat.comm();
        if ac != pc {
            self.invalidate_setup();
            let msg = format!(
                "Amat/Pmat communicator mismatch: A={}, P={}",
                ac.id(),
                pc.id()
            );
            panic!("{}", msg);
        }
        self.amat = Some(amat.clone());
        self.pmat = Some(pmat);
        self.invalidate_setup();
        self
    }

    pub fn set_operators_with_comm(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
        comm: crate::parallel::UniverseComm,
    ) -> &mut Self {
        let a_wrapped = wrap_with_comm(amat, comm.clone());
        let p_wrapped = pmat.map(|p| wrap_with_comm(p, comm.clone()));
        self.set_operators(a_wrapped, p_wrapped)
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
                let m = pmat
                    .as_any()
                    .downcast_ref::<faer::Mat<f64>>()
                    .ok_or_else(|| {
                        KError::InvalidInput(
                            "expected faer::Mat<f64> for chain construction".into(),
                        )
                    })?;
                let chain = PcFactory::construct_deferred_pc_chain(specs, m)?;
                self.pc = Some(chain);
            } else if let Some(spec) = self.pending_pc.take() {
                let m = pmat
                    .as_any()
                    .downcast_ref::<faer::Mat<f64>>()
                    .ok_or_else(|| {
                        KError::InvalidInput("expected faer::Mat<f64> for PC construction".into())
                    })?;
                let pc = PcFactory::construct_deferred_preconditioner(spec, m)?;
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
            match self.last_pc_sid {
                None => {
                    pc.setup(pmat.as_ref())?;
                    self.last_pc_sid = Some(sid);
                    self.last_pc_vid = Some(vid);
                }
                Some(old_sid) if old_sid != sid => {
                    pc.update_symbolic(pmat.as_ref())?;
                    self.last_pc_sid = Some(sid);
                    self.last_pc_vid = Some(vid);
                }
                Some(_old_sid) => {
                    if self.last_pc_vid != Some(vid)
                        && self.pc_reuse.allow_numeric()
                        && pc.supports_numeric_update()
                    {
                        pc.update_numeric(pmat.as_ref())?;
                        self.last_pc_vid = Some(vid);
                    } else if self.last_pc_vid != Some(vid) {
                        pc.update_symbolic(pmat.as_ref())?;
                        self.last_pc_vid = Some(vid);
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
            if let Some(ref mut solver) = self.solver {
                if let Some(ref mut w) = self.work {
                    solver.setup_workspace(w);
                }
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
        let mut pc = self
            .pc
            .as_mut()
            .map(|b| b.as_mut() as &mut dyn Preconditioner);
        let solver = self
            .solver
            .as_mut()
            .ok_or_else(|| KError::SolveError("No solver".into()))?;

        // Some solvers (e.g. FGMRES) require a mutable preconditioner and have
        // a specialised entry point. Handle this before computing the final
        // residual below.
        let mut stats = if let Some(fgmres) = solver
            .as_any_mut()
            .downcast_mut::<crate::solver::FgmresSolver>()
        {
            fgmres.solve_flexible(
                amat.as_ref(),
                pc.as_deref_mut(),
                b,
                x,
                self.pc_side,
                &comm,
                monitors,
                self.work.as_mut(),
            )?
        } else {
            solver.solve(
                amat.as_ref(),
                pc.map(|p| p as &dyn Preconditioner),
                b,
                x,
                self.pc_side,
                &comm,
                monitors,
                self.work.as_mut(),
            )?
        };

        // Compute true residual r = b - A x and use its norm for reporting
        let mut residual = vec![0.0f64; b.len()];
        amat.matvec(x, &mut residual);
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

#[cfg(test)]
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
