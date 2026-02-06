use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::{LinearSolver, MonitorCallback, TfqmrSolver};
use crate::utils::convergence::SolveStats;
use std::any::Any;

/// PETSc-compatible TCQMR surface; currently mapped to TFQMR kernel.
pub struct TcqmrSolver {
    inner: TfqmrSolver,
}

impl TcqmrSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            inner: TfqmrSolver::new(rtol, maxits),
        }
    }
}

impl LinearSolver for TcqmrSolver {
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
        self.inner
            .solve(a, pc.as_deref(), b, x, pc_side, comm, monitors, work)
    }
}
