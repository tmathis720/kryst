use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::{FgmresSolver, FgmresVariant, LinearSolver, MonitorCallback};
use crate::utils::convergence::SolveStats;
use std::any::Any;

/// PETSc-compatible PipeGCR surface, backed by pipelined flexible GMRES.
pub struct PipeGcrSolver {
    inner: FgmresSolver,
}

impl PipeGcrSolver {
    pub fn new(restart: usize, rtol: f64, maxits: usize) -> Self {
        let mut inner = FgmresSolver::new(rtol, maxits, restart);
        inner.set_variant(FgmresVariant::Pipelined);
        Self { inner }
    }
}

impl LinearSolver for PipeGcrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        self.inner.setup_workspace(work);
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
        self.inner.solve(a, pc, b, x, pc_side, comm, monitors, work)
    }
}
