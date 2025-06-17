//! Factory for Krylov Subspace Methods (KSP).
//!
//! This module provides the `KspContext` struct, which acts as a factory and context holder for various Krylov subspace iterative solvers
//! (such as CG, GMRES, BiCGStab, MINRES, etc.) and their associated preconditioners. It allows users to select a solver kind,
//! configure solver parameters, and solve linear systems in a unified way.
//!
//! # Usage
//!
//! 1. Construct a `KspContext` with the desired solver kind, matrix, preconditioner, tolerance, and iteration limits.
//! 2. Call `solve_context` to solve a linear system `Ax = b`.
//!
//! # Supported Solvers
//! - CG, PCG, GMRES (left/right), FGMRES, BiCGStab, CGS, QMR, TFQMR, MINRES, CGNR
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.
//! - Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods, 2nd Edition (Barrett et al.)

use crate::solver::{CgSolver, GmresSolver, LinearSolver, PcgSolver, gmres::Preconditioning};
use crate::solver::fgmres::FgmresSolver;
use crate::preconditioner::Preconditioner;

/// Enum representing the available Krylov solver types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// Conjugate Gradient (CG) method (for SPD matrices)
    Cg,
    /// Preconditioned Conjugate Gradient (PCG)
    Pcg,
    /// GMRES with left preconditioning
    GmresLeft,
    /// GMRES with right preconditioning
    GmresRight,
    /// Flexible GMRES (FGMRES)
    Fgmres,
    /// BiConjugate Gradient Stabilized (BiCGStab)
    Bicgstab,
    /// Conjugate Gradient Squared (CGS)
    Cgs,
    /// Quasi-Minimal Residual (QMR)
    Qmr,
    /// Transpose-Free QMR (TFQMR)
    Tfqmr,
    /// Minimal Residual (MINRES)
    Minres,
    /// Conjugate Gradient on the Normal Equations (CGNR)
    Cgnr,
}

/// Context and configuration for a Krylov subspace solver.
///
/// Holds the matrix, preconditioner, solver kind, tolerance, and other parameters.
/// Use `solve_context` to solve a linear system with the configured solver.
pub struct KspContext<M, V, T> {
    /// The type of Krylov solver to use
    pub kind: SolverKind,
    /// The system matrix (must implement MatVec and MatTransVec)
    pub a: M,
    /// Optional preconditioner (for standard solvers)
    pub pc: Option<Box<dyn Preconditioner<M, V>>>,
    /// Optional flexible preconditioner (for FGMRES)
    pub flex_pc: Option<Box<dyn crate::preconditioner::FlexiblePreconditioner<M, V>>>,
    /// Convergence tolerance (relative or absolute, depending on solver)
    pub tol: T,
    /// Maximum number of iterations
    pub max_it: usize,
    /// Restart parameter (for GMRES/FGMRES)
    pub restart: usize, // for GMRES
}

impl<M, V, T> KspContext<M, V, T>
where
    M: 'static + crate::core::traits::MatVec<V> + crate::core::traits::MatTransVec<V> + std::fmt::Debug,
    (): crate::core::traits::InnerProduct<V, Scalar = T>,
    V: 'static + AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone + std::fmt::Debug + Send + Sync,
    T: 'static + num_traits::Float + Clone + From<f64> + std::fmt::Debug + std::ops::AddAssign + std::iter::Sum + num_traits::FromPrimitive + Send + Sync,
{
    /// Solve the linear system `Ax = b` using the configured solver and preconditioner.
    ///
    /// # Arguments
    /// * `b` - Right-hand side vector
    /// * `x` - Solution vector (will be overwritten with the result)
    /// * `comm` - Optional parallel communication context (not used by all solvers)
    ///
    /// # Returns
    /// * `Ok(SolveStats)` on success
    /// * `Err(KError)` on failure or breakdown
    pub fn solve_context<C: crate::parallel::Comm<Vec = Vec<T>>>(&mut self, b: &V, x: &mut V, comm: Option<&C>) -> Result<crate::utils::convergence::SolveStats<T>, crate::error::KError> {
        match self.kind {
            SolverKind::GmresLeft => {
                // GMRES with left preconditioning
                let mut solver = GmresSolver::new(self.restart, self.tol, self.max_it).with_preconditioning(Preconditioning::Left);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::GmresRight => {
                // GMRES with right preconditioning
                let mut solver = GmresSolver::new(self.restart, self.tol, self.max_it).with_preconditioning(Preconditioning::Right);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            // FGMRES supports flexible preconditioning
            SolverKind::Fgmres => {
                let mut solver = FgmresSolver::new(self.tol, self.max_it, self.restart);
                match self.flex_pc {
                    Some(ref mut flex_pc) => solver.solve_flex(&self.a, Some(flex_pc.as_mut()), b, x),
                    None => solver.solve_flex(&self.a, None, b, x),
                }
            }
            SolverKind::Cg => {
                // Conjugate Gradient (CG)
                let mut solver = CgSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Pcg => {
                // Preconditioned Conjugate Gradient (PCG)
                let mut solver = PcgSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Bicgstab => {
                // BiCGStab
                let mut solver = crate::solver::bicgstab::BiCgStabSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Cgs => {
                // CGS
                let mut solver = crate::solver::cgs::CgsSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Qmr => {
                // QMR (requires special handling for transpose)
                return self.solve_qmr(b, x);
            }
            SolverKind::Tfqmr => {
                // TFQMR
                let mut solver = crate::solver::tfqmr::TfqmrSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Minres => {
                // MINRES
                let mut solver = crate::solver::minres::MinresSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Cgnr => {
                // CGNR
                let mut solver = crate::solver::cgnr::CgnrSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
        }
    }

    /// Helper for QMR solver, which requires the matrix to implement MatTransVec.
    ///
    /// # Arguments
    /// * `b` - Right-hand side vector
    /// * `x` - Solution vector (will be overwritten)
    ///
    /// # Returns
    /// * `Ok(SolveStats)` on success
    /// * `Err(KError)` on failure
    #[allow(unused)]
    fn solve_qmr(&mut self, b: &V, x: &mut V) -> Result<crate::utils::convergence::SolveStats<T>, crate::error::KError>
    where
        M: crate::core::traits::MatTransVec<V>,
    {
        let mut solver = crate::solver::qmr::QmrSolver::new(self.tol, self.max_it);
        solver.solve(&self.a, self.pc.as_deref(), b, x)
    }
}
