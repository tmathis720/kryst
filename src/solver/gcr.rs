use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::{FgmresSolver, LinearSolver, MonitorCallback};
use crate::utils::convergence::SolveStats;
use std::any::Any;

/// PETSc-compatible GCR surface, backed by flexible GMRES.
pub struct GcrSolver {
    inner: FgmresSolver,
}

impl GcrSolver {
    pub fn new(restart: usize, rtol: f64, maxits: usize) -> Self {
        Self {
            inner: FgmresSolver::new(rtol, maxits, restart),
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
        self.inner
            .solve_k(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

impl LinearSolver for GcrSolver {
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
