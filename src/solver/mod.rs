//! Solver traits and implementations.
//!
//! Most users configure solvers through [`KspContext`](crate::context::KspContext) and
//! [`SolverType`](crate::context::ksp_context::SolverType). This module exposes the
//! underlying solver types and traits for advanced use, custom pipelines, or testing.

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::utils::convergence::SolveStats;
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonitorAction {
    Continue,
    Stop,
}

pub type MonitorCallback<R> = dyn Fn(usize, R, usize) -> MonitorAction + Send + Sync;

pub mod api;
pub use api::Solver;
pub mod adapters;
pub use crate::ops::klinop::KLinOp;
pub use crate::ops::kpc::KPreconditioner;
pub use adapters::LegacyDirectAdapter;

pub mod block;

/// Object-safe linear solver operating on `f64` slices and [`LinOp`] operators.
pub trait LinearSolver: Send + Any {
    type Error;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Allow solver to configure workspace buffers.
    fn setup_workspace(&mut self, _work: &mut Workspace) {}

    /// Solve `a * x = b` optionally using a preconditioner.
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
    ) -> Result<SolveStats<f64>, Self::Error>;
}

/// Legacy generic solver trait retained for existing implementations.
pub mod legacy {
    use crate::algebra::prelude::KrystScalar;
    use crate::preconditioner::legacy::Preconditioner;
    use crate::solver::MonitorCallback;
    use crate::utils::convergence::SolveStats;

    pub trait LinearSolver<M: ?Sized, V> {
        type Error;
        type Scalar: KrystScalar;

        fn solve(
            &mut self,
            a: &M,
            pc: Option<&(dyn Preconditioner<M, V> + '_)>,
            b: &V,
            x: &mut V,
            pc_side: crate::preconditioner::PcSide,
            comm: &crate::parallel::UniverseComm,
            monitors: Option<&[Box<MonitorCallback<Self::Scalar>>]>,
            work: Option<&mut crate::context::ksp_context::Workspace>,
        ) -> Result<SolveStats<Self::Scalar>, Self::Error>;

        fn setup_workspace(&mut self, _work: &mut crate::context::ksp_context::Workspace) {}

        fn solve_simple(
            &mut self,
            a: &M,
            pc: Option<&(dyn Preconditioner<M, V> + '_)>,
            b: &V,
            x: &mut V,
            pc_side: crate::preconditioner::PcSide,
            comm: &crate::parallel::UniverseComm,
        ) -> Result<SolveStats<Self::Scalar>, Self::Error>
        where
            Self: Sized,
        {
            self.solve(a, pc, b, x, pc_side, comm, None, None)
        }

        fn solve_with_monitors(
            &mut self,
            a: &M,
            pc: Option<&(dyn Preconditioner<M, V> + '_)>,
            b: &V,
            x: &mut V,
            pc_side: crate::preconditioner::PcSide,
            comm: &crate::parallel::UniverseComm,
            monitors: &[Box<MonitorCallback<Self::Scalar>>],
        ) -> Result<SolveStats<Self::Scalar>, Self::Error>
        where
            Self: Sized,
        {
            self.solve(a, pc, b, x, pc_side, comm, Some(monitors), None)
        }

        fn solve_with_workspace(
            &mut self,
            a: &M,
            pc: Option<&(dyn Preconditioner<M, V> + '_)>,
            b: &V,
            x: &mut V,
            pc_side: crate::preconditioner::PcSide,
            comm: &crate::parallel::UniverseComm,
            work: &mut crate::context::ksp_context::Workspace,
        ) -> Result<SolveStats<Self::Scalar>, Self::Error>
        where
            Self: Sized,
        {
            self.solve(a, pc, b, x, pc_side, comm, None, Some(work))
        }
    }
}

/// Adapter allowing legacy generic solvers (MatVec/Vec) to be used with the
/// new object-safe [`LinearSolver`] trait over [`LinOp`].
pub struct OpSolverAdapter<S> {
    inner: S,
}

impl<S> OpSolverAdapter<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

#[cfg(not(feature = "complex"))]
struct OpPcAdapter<'p> {
    inner: &'p dyn Preconditioner,
}

#[cfg(not(feature = "complex"))]
impl<'p, 'm> crate::preconditioner::legacy::Preconditioner<dyn LinOp<S = f64> + 'm, Vec<f64>>
    for OpPcAdapter<'p>
{
    fn setup(&mut self, _a: &(dyn LinOp<S = f64> + 'm)) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        self.inner.apply(side, r.as_slice(), z.as_mut_slice())
    }
}

#[cfg(not(feature = "complex"))]
impl<S> LinearSolver for OpSolverAdapter<S>
where
    S: for<'a> legacy::LinearSolver<
            dyn LinOp<S = f64> + 'a,
            Vec<f64>,
            Scalar = f64,
            Error = KError,
        > + Send
        + 'static,
{
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
        let mut x_vec = x.to_vec();
        let b_vec = b.to_vec();
        // Coerce &mut dyn Preconditioner -> &dyn Preconditioner for the legacy adapter.
        let pc_adapter = pc.as_deref().map(|p| OpPcAdapter { inner: p });
        let pc_ref = pc_adapter.as_ref().map(|p| {
            p
                as &dyn crate::preconditioner::legacy::Preconditioner<
                    dyn LinOp<S = f64> + '_,
                    Vec<f64>,
                >
        });
        let stats = self
            .inner
            .solve(a, pc_ref, &b_vec, &mut x_vec, pc_side, comm, monitors, work)?;
        x.copy_from_slice(&x_vec);
        Ok(stats)
    }
}

// Re-export solver implementations
pub mod cg;
pub use cg::CgSolver;
pub mod cgnr;
pub use cgnr::CgnrSolver;

pub mod gmres;
pub use gmres::GmresSolver;
pub mod fgmres;
pub use fgmres::FgmresSolver;
pub mod bicgstab;
pub use bicgstab::{BiCgStabBreakdownPolicy, BiCgStabSolver, BiCgStabVariant};
pub mod idrs;
pub use idrs::{
    BreakdownRepair as IdrsBreakdownRepair, IdrsBuilder, IdrsOptions, IdrsSolver,
    Omega as IdrsOmega, ShadowP as IdrsShadowP,
};
pub mod cgs;
pub use cgs::CgsSolver;
pub mod richardson;
pub use richardson::RichardsonSolver;
pub mod chebyshev;
pub use chebyshev::ChebyshevSolver;
pub mod cr;
pub use cr::CrSolver;
pub mod pcg;
pub use pcg::{PCG_PIPELINED_DEFAULT_REPLACE_EVERY, PcgSolver, PcgVariant};
pub mod minres;
pub use minres::MinresSolver;
pub mod lsmr;
pub use lsmr::LsmrSolver;
pub mod lsqr;
pub use lsqr::LsqrSolver;
// Dense direct modules are gated; opt-in via `dense-direct`.
#[cfg(feature = "dense-direct")]
pub mod dense_lu;
#[cfg(feature = "dense-direct")]
pub mod dense_qr;
#[cfg(feature = "backend-faer")]
pub mod direct_lu;

#[cfg(feature = "backend-faer")]
pub use direct_lu::{LuSolver, QrSolver};
#[cfg(feature = "superlu_dist")]
pub mod superlu_dist;
#[cfg(feature = "superlu_dist")]
pub use superlu_dist::SuperLuDistSolver;

pub mod qmr;
pub use qmr::QmrSolver;
pub mod tfqmr;
pub use tfqmr::TfqmrSolver;
pub mod tcqmr;
pub use tcqmr::TcqmrSolver;

pub mod gcr;
pub use gcr::GcrSolver;
pub mod pipegcr;
pub use pipegcr::PipeGcrSolver;

pub mod pca_gmres;
pub use pca_gmres::{PcaGmresSolver, PcaPcMode};

pub mod common;

#[cfg(test)]
mod tests;
