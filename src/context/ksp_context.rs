use crate::config::options::PcOptions;
use crate::context::pc_context::{PcFactory, PcType};
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::{CgSolver, GmresSolver, LinearSolver, MatSolverAdapter};
use crate::utils::convergence::SolveStats;
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
        }
    }
}

/// Supported solver types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    Cg,
    Gmres,
    Preonly,
}

/// Minimal KSP context holding solver, preconditioner, and operators.
pub struct KspContext {
    solver: Option<Box<dyn LinearSolver<Error = KError>>>,
    pc: Option<Box<dyn Preconditioner>>,
    amat: Option<Arc<dyn LinOp<S = f64>>>,
    pmat: Option<Arc<dyn LinOp<S = f64>>>,
    work: Option<Workspace>,
    setup_called: bool,
    pub rtol: f64,
    pub atol: f64,
    pub dtol: f64,
    pub maxits: usize,
    pub restart: usize,
    pub pc_side: PcSide,
}

impl KspContext {
    pub fn new() -> Self {
        Self {
            solver: None,
            pc: None,
            amat: None,
            pmat: None,
            work: None,
            setup_called: false,
            rtol: 1e-6,
            atol: 1e-12,
            dtol: 1e3,
            maxits: 1000,
            restart: 30,
            pc_side: PcSide::Left,
        }
    }

    pub fn set_type(&mut self, solver_type: SolverType) -> Result<&mut Self, KError> {
        let solver: Box<dyn LinearSolver<Error = KError>> = match solver_type {
            SolverType::Cg => {
                Box::new(MatSolverAdapter::new(CgSolver::new(self.rtol, self.maxits)))
            }
            SolverType::Gmres => Box::new(MatSolverAdapter::new(GmresSolver::new(
                self.restart,
                self.rtol,
                self.maxits,
            ))),
            SolverType::Preonly => {
                return Err(KError::SolveError("Preonly solver not available".into()))
            }
        };
        self.solver = Some(solver);
        self.invalidate_setup();
        Ok(self)
    }

    pub fn set_pc_type(
        &mut self,
        pc_type: PcType,
        opts: Option<&PcOptions>,
    ) -> Result<&mut Self, KError> {
        self.pc = Some(PcFactory::create_preconditioner(pc_type, opts)?);
        self.invalidate_setup();
        Ok(self)
    }

    /// Assign the system and preconditioner operators.
    pub fn set_operators(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
    ) -> &mut Self {
        self.amat = Some(amat.clone());
        self.pmat = Some(pmat.unwrap_or(amat));
        self.invalidate_setup();
        self
    }

    /// Prepare preconditioner and workspace.
    pub fn setup(&mut self) -> Result<(), KError> {
        let amat = self
            .amat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;
        let pmat = self
            .pmat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Pmat not set".into()))?;

        if let Some(pc) = self.pc.as_mut() {
            pc.setup(pmat.as_ref())?;
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
        let amat = self
            .amat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;
        let solver = self
            .solver
            .as_mut()
            .ok_or_else(|| KError::SolveError("No solver".into()))?;
        solver.solve(
            amat.as_ref(),
            self.pc.as_deref(),
            b,
            x,
            &UniverseComm::NoComm(crate::parallel::NoComm),
            None,
            self.work.as_mut(),
        )
    }

    fn invalidate_setup(&mut self) {
        self.setup_called = false;
    }
}
