//! Krylov and direct solver interfaces and implementations.
//!
//! This module provides a unified interface for both direct and iterative linear solvers, as well as
//! re-exports for all supported solver types. The `LinearSolver` trait defines a common API for
//! solving linear systems Ax = b, optionally with a preconditioner. All solvers return convergence
//! statistics via `SolveStats`.
//!
//! # Usage
//! - Implementations include CG, GMRES, BiCGStab, MINRES, QMR, FGMRES, and direct LU/QR solvers.
//! - All solvers are accessible via their respective types (e.g., `CgSolver`, `GmresSolver`, etc.).
//! - The trait is generic over matrix and vector types, and supports optional preconditioning.
//!
//! # Example
//! ```rust,ignore
//! use krylovkit::solver::{LinearSolver, CgSolver};
//! // ...
//! let mut solver = CgSolver::new(1e-8, 100);
//! let stats = solver.solve(&a, None, &b, &mut x)?;
//! ```

use crate::utils::convergence::SolveStats;
use crate::preconditioner::Preconditioner;

/// Common interface for any direct or iterative linear solver.
///
/// # Type Parameters
/// * `M` - Matrix type
/// * `V` - Vector type
///
pub trait LinearSolver<M, V> {
    type Error;
    
    /// Scalar type used by the solver (e.g., f32, f64)
    type Scalar: Copy + PartialOrd + From<f64>;
    
    /// Solve the linear system A·x = b, optionally with preconditioner M⁻¹, writing result into `x`.
    ///
    /// This unified method handles all solve variants:
    /// - Monitors are called only if monitoring is enabled at runtime
    /// - Profiling is performed only if profiling is enabled at runtime
    /// - Workspace is used for efficiency when provided
    ///
    /// # Arguments
    /// * `a` - Matrix (system operator)
    /// * `pc` - Optional preconditioner
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    /// * `comm` - Communicator for parallel operations
    /// * `monitors` - Optional callbacks to invoke at each iteration with (iteration, residual_norm)
    /// * `work` - Optional pre-allocated workspace containing temporary vectors
    ///
    /// # Returns
    /// * `Ok(SolveStats)` with convergence information
    /// * `Err(Self::Error)` on failure
    fn solve(
        &mut self,
        a: &M,
        pc: Option<&dyn Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
        work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<Self::Scalar>, Self::Error>;
    
    /// Setup workspace for the solver.
    /// 
    /// This method allows solvers to configure workspace buffers they need
    /// and grab references to tmp1, tmp2, etc. from the unified workspace.
    fn setup_workspace(&mut self, _work: &mut crate::context::ksp_context::Workspace) {}
    
    /// Convenience method for solving without monitors or workspace.
    fn solve_simple(
        &mut self,
        a: &M,
        pc: Option<&dyn Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
    ) -> Result<SolveStats<Self::Scalar>, Self::Error> {
        self.solve(a, pc, b, x, comm, None, None)
    }
    
    /// Convenience method for solving with monitors but no workspace.
    fn solve_with_monitors(
        &mut self,
        a: &M,
        pc: Option<&dyn Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
        monitors: &[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>],
    ) -> Result<SolveStats<Self::Scalar>, Self::Error> {
        self.solve(a, pc, b, x, comm, Some(monitors), None)
    }
    
    /// Convenience method for solving with workspace but no monitors.
    fn solve_with_workspace(
        &mut self,
        a: &M,
        pc: Option<&dyn Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
        work: &mut crate::context::ksp_context::Workspace,
    ) -> Result<SolveStats<Self::Scalar>, Self::Error> {
        self.solve(a, pc, b, x, comm, None, Some(work))
    }
}

// Re-export all supported solver types for user convenience
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
