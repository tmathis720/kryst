//! Solver traits and adapters.

use crate::matrix::op::LinOp;
use crate::preconditioner::{Preconditioner, PcSide};
use crate::utils::convergence::SolveStats;
use crate::context::ksp_context::Workspace;
use crate::parallel::UniverseComm;
use crate::error::KError;

/// Object-safe linear solver operating on `f64` slices and [`LinOp`] operators.
pub trait LinearSolver: Send {
    type Error;

    /// Allow solver to configure workspace buffers.
    fn setup_workspace(&mut self, _work: &mut Workspace) {}

    /// Solve `a * x = b` optionally using a preconditioner.
    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error>;
}

/// Legacy generic solver trait retained for existing implementations.
pub mod legacy {
    use crate::preconditioner::legacy::Preconditioner;
    use crate::utils::convergence::SolveStats;

    pub trait LinearSolver<M: ?Sized, V> {
        type Error;
        type Scalar: Copy + PartialOrd + From<f64>;

        fn solve(
            &mut self,
            a: &M,
            pc: Option<&(dyn Preconditioner<M, V> + '_)>,
            b: &V,
            x: &mut V,
            comm: &crate::parallel::UniverseComm,
            monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
            work: Option<&mut crate::context::ksp_context::Workspace>,
        ) -> Result<SolveStats<Self::Scalar>, Self::Error>;

        fn setup_workspace(&mut self, _work: &mut crate::context::ksp_context::Workspace) {}

        fn solve_simple(
            &mut self,
            a: &M,
            pc: Option<&(dyn Preconditioner<M, V> + '_)>,
            b: &V,
            x: &mut V,
            comm: &crate::parallel::UniverseComm,
        ) -> Result<SolveStats<Self::Scalar>, Self::Error>
        where
            Self: Sized,
        {
            self.solve(a, pc, b, x, comm, None, None)
        }

        fn solve_with_monitors(
            &mut self,
            a: &M,
            pc: Option<&(dyn Preconditioner<M, V> + '_)>,
            b: &V,
            x: &mut V,
            comm: &crate::parallel::UniverseComm,
            monitors: &[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>],
        ) -> Result<SolveStats<Self::Scalar>, Self::Error>
        where
            Self: Sized,
        {
            self.solve(a, pc, b, x, comm, Some(monitors), None)
        }

        fn solve_with_workspace(
            &mut self,
            a: &M,
            pc: Option<&(dyn Preconditioner<M, V> + '_)>,
            b: &V,
            x: &mut V,
            comm: &crate::parallel::UniverseComm,
            work: &mut crate::context::ksp_context::Workspace,
        ) -> Result<SolveStats<Self::Scalar>, Self::Error>
        where
            Self: Sized,
        {
            self.solve(a, pc, b, x, comm, None, Some(work))
        }
    }
}

/// Adapter allowing legacy matrix-based solvers to be used with the new
/// object-safe [`LinearSolver`] trait.
pub struct MatSolverAdapter<S> {
    inner: S,
}

impl<S> MatSolverAdapter<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

struct MatPcAdapter<'a> {
    inner: &'a dyn Preconditioner,
}

impl<'a> crate::preconditioner::legacy::Preconditioner<faer::Mat<f64>, Vec<f64>> for MatPcAdapter<'a> {
    fn setup(&mut self, _a: &faer::Mat<f64>) -> Result<(), KError> { Ok(()) }
    fn apply(&self, side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        self.inner.apply(side, r.as_slice(), z.as_mut_slice())
    }
}

impl<S> LinearSolver for MatSolverAdapter<S>
where
    S: legacy::LinearSolver<faer::Mat<f64>, Vec<f64>, Scalar = f64, Error = KError>,
{
    type Error = KError;

    fn setup_workspace(&mut self, work: &mut Workspace) {
        self.inner.setup_workspace(work);
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let mat = a
            .as_any()
            .downcast_ref::<faer::Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("solver requires faer::Mat<f64>".into()))?;
        let mut x_vec = x.to_vec();
        let b_vec = b.to_vec();
        let pc_adapter = pc.map(|p| MatPcAdapter { inner: p });
        let pc_ref = pc_adapter
            .as_ref()
            .map(|p| p as &dyn crate::preconditioner::legacy::Preconditioner<faer::Mat<f64>, Vec<f64>>);
        let stats = self
            .inner
            .solve(mat, pc_ref, &b_vec, &mut x_vec, comm, monitors, work)?;
        x.copy_from_slice(&x_vec);
        Ok(stats)
    }
}

// Re-export solver implementations
pub mod direct_lu;
pub use direct_lu::{LuSolver, QrSolver};

pub mod cg;
pub use cg::CgSolver;

pub mod gmres;
pub use gmres::GmresSolver;

pub mod bicgstab;
pub use bicgstab::BiCgStabSolver;

pub mod cgs;
pub use cgs::CgsSolver;

pub mod qmr;
pub use qmr::QmrSolver;

pub mod minres;
pub use minres::MinresSolver;

pub mod tfqmr;
pub use tfqmr::TfqmrSolver;

pub mod cgnr;
pub use cgnr::{CgnrSolver, CgneSolver};

pub mod pcg;
pub use self::pcg::PcgSolver;

pub mod fgmres;
pub use fgmres::FgmresSolver;

pub mod pca_gmres;
pub use pca_gmres::PcaGmresSolver;

pub mod superlu_dist;
pub use superlu_dist::SuperLuDistSolver;

