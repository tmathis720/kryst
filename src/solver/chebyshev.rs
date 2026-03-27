use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::richardson::RichardsonSolver;
use crate::solver::{LinearSolver, MonitorCallback};
use crate::utils::convergence::SolveStats;
use std::any::Any;

pub struct ChebyshevSolver {
    inner: RichardsonSolver,
}

impl ChebyshevSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            inner: RichardsonSolver::new(rtol, maxits),
        }
    }

    pub fn set_omega(&mut self, omega: f64) {
        self.inner.set_omega(omega);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_k<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
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
        self.inner.solve(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

impl LinearSolver for ChebyshevSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        mut pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        self.inner.solve_f64(
            a,
            pc.map(|p| &*p),
            b,
            x,
            pc_side,
            comm,
            monitors,
            work,
        )
    }
}
