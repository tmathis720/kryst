use crate::config::options::PcOptions;
use crate::context::pc_context::{PcType, PcFactory};
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{Preconditioner, PcSide};
use crate::solver::{LinearSolver, MatSolverAdapter, CgSolver, GmresSolver};
use crate::parallel::UniverseComm;
use crate::utils::convergence::{SolveStats};

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
pub enum SolverType { Cg, Gmres, Preonly }

/// Minimal KSP context holding solver and preconditioner.
pub struct KspContext {
    solver: Option<Box<dyn LinearSolver<Error = KError>>>,
    pc: Option<Box<dyn Preconditioner>>,
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
            SolverType::Cg => Box::new(MatSolverAdapter::new(CgSolver::new(self.rtol, self.maxits))),
            SolverType::Gmres => Box::new(MatSolverAdapter::new(GmresSolver::new(self.restart, self.rtol, self.maxits))),
            SolverType::Preonly => return Err(KError::SolveError("Preonly solver not available".into())),
        };
        self.solver = Some(solver);
        Ok(self)
    }

    pub fn set_pc_type(&mut self, pc_type: PcType, opts: Option<&PcOptions>) -> Result<&mut Self, KError> {
        self.pc = Some(PcFactory::create_preconditioner(pc_type, opts)?);
        Ok(self)
    }

    pub fn solve(&mut self, a: &dyn LinOp<S = f64>, b: &[f64], x: &mut [f64]) -> Result<SolveStats<f64>, KError> {
        let solver = self.solver.as_mut().ok_or_else(|| KError::SolveError("No solver".into()))?;
        solver.solve(
            a,
            self.pc.as_deref(),
            b,
            x,
            &UniverseComm::NoComm(crate::parallel::NoComm),
            None,
            None,
        )
    }
}
