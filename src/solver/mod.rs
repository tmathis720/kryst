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
//! ```rust
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
    /// Solve the linear system A·x = b, optionally with preconditioner M⁻¹, writing result into `x`.
    ///
    /// # Arguments
    /// * `a` - Matrix (system operator)
    /// * `pc` - Optional preconditioner
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    ///
    /// # Returns
    /// * `Ok(SolveStats)` with convergence information
    /// * `Err(Self::Error)` on failure
    fn solve(
        &mut self,
        a: &M,
        pc: Option<&dyn Preconditioner<M, V>>,
        b: &V,
        x: &mut V
    ) -> Result<SolveStats<<Self as LinearSolver<M, V>>::Scalar>, Self::Error>;
    /// Scalar type used by the solver (e.g., f32, f64)
    type Scalar: Copy + PartialOrd + From<f64>;
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
