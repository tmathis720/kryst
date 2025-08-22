//! Direct dense solvers using Faer: LU and QR factorizations.
//!
//! This module provides wrappers for direct dense linear solvers using the Faer library.
//! It includes LU (with full pivoting) and QR solvers for square or rectangular systems.
//! These solvers are suitable for small to medium-sized dense systems where direct methods are feasible.
//!
//! # Usage
//! - Use `LuSolver` for general square systems (may be faster, but less stable for rank-deficient matrices).
//! - Use `QrSolver` for square or rectangular systems (more stable for rank-deficient or nearly singular matrices).
//!
//! # References
//! - Faer documentation: https://github.com/sarah-ek/faer-rs
//! - Golub & Van Loan, Matrix Computations

use crate::error::KError;
use crate::solver::legacy::LinearSolver;
use faer::linalg::solvers::{FullPivLu, Qr, SolveCore};
use faer::{Mat, MatMut, Conj};
use faer::traits::{ComplexField, RealField};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;

/// LU solver using full pivoting from Faer.
///
/// Stores the LU factorization for reuse (if desired).
pub struct LuSolver<T> {
    /// Cached LU factorization (if computed)
    factor: Option<FullPivLu<T>>,
}

impl<T: ComplexField + RealField> LuSolver<T> {
    /// Create a new LU solver (no factorization yet).
    pub fn new() -> Self {
        LuSolver { factor: None }
    }

    /// Solve using the cached LU factorization.
    ///
    /// # Panics
    /// Panics if called before any factorization has been performed.
    ///
    /// # Arguments
    /// * `b` - Right-hand side vector
    /// * `x` - Output vector (solution)
    pub fn solve_cached(&self, b: &[T], x: &mut [T]) {
        if let Some(factor) = &self.factor {
            let n = b.len();
            x.clone_from_slice(b);
            let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
            factor.solve_in_place_with_conj(Conj::No, x_mat);
        } else {
            panic!("LuSolver: solve_cached called before factorization");
        }
    }
}

impl<T> LinearSolver<Mat<T>, Vec<T>> for LuSolver<T>
where
    T: ComplexField + RealField + Copy + PartialOrd + From<f64> + Send + Sync + std::fmt::Debug + std::fmt::LowerExp,
{
    type Error = KError;
    type Scalar = T;

    /// Solve Ax = b using LU factorization (full pivoting).
    ///
    /// # Arguments
    /// * `a` - Matrix (Faer Mat)
    /// * `pc` - (Unused) Preconditioner (not supported for direct solvers)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: ignored; on output: solution vector
    /// * `comm` - Communicator for parallel operations (unused for direct solvers)
    /// * `monitors` - Optional callbacks to invoke at each iteration
    /// * `work` - Optional workspace (unused for direct solvers)
    ///
    /// # Returns
    /// * `Ok(SolveStats)` (always converged in 1 iteration)
    fn solve(
        &mut self,
        a: &Mat<T>,
        pc: Option<&(dyn crate::preconditioner::legacy::Preconditioner<Mat<T>, Vec<T>> + '_)>,
        b: &Vec<T>,
        x: &mut Vec<T>,
        _comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
        _work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<crate::utils::convergence::SolveStats<T>, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("LuSolve");
        
        let _ = pc; // Direct solvers do not use preconditioner
        
        // Call monitors at start if provided
        if let Some(monitors) = monitors {
            for monitor in monitors {
                monitor(0, T::zero());
            }
        }
        
        #[cfg(feature = "logging")]
        let _fact_guard = StageGuard::new("LuFactor");
        
        // Compute LU factorization (overwrites any previous factor)
        let factor = FullPivLu::new(a.as_ref());
        self.factor = Some(factor);
        
        // Copy b into x
        x.clone_from(b);
        
        // Solve in-place: x = A^{-1} b
        let n = x.len();
        let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
        self.factor
            .as_ref()
            .unwrap()
            .solve_in_place_with_conj(Conj::No, x_mat);
        
        // Call monitors at end if provided
        if let Some(monitors) = monitors {
            for monitor in monitors {
                monitor(1, T::zero());
            }
        }
        
        // For direct solvers, always converged in 1 iteration
        Ok(crate::utils::convergence::SolveStats {
            iterations: 1,
            final_residual: T::zero(),
            reason: crate::utils::convergence::ConvergedReason::ConvergedAtol,
        })
    }
}

impl<T: faer::traits::ComplexField + faer::traits::RealField> Default for LuSolver<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// QR solver using Faer (for square or rectangular systems).
pub struct QrSolver;

impl QrSolver {
    /// Create a new QR solver.
    pub fn new() -> Self {
        QrSolver
    }
}

impl<T: ComplexField + RealField + Copy + PartialOrd + From<f64>> LinearSolver<Mat<T>, Vec<T>> for QrSolver {
    type Error = KError;
    type Scalar = T;

    /// Solve Ax = b using QR factorization.
    ///
    /// # Arguments
    /// * `a` - Matrix (Faer Mat)
    /// * `pc` - (Unused) Preconditioner (not supported for direct solvers)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: ignored; on output: solution vector
    /// * `comm` - Communicator for parallel operations (unused for direct solvers)
    /// * `monitors` - Optional callbacks to invoke at each iteration
    /// * `work` - Optional workspace (unused for direct solvers)
    ///
    /// # Returns
    /// * `Ok(SolveStats)` (always converged in 1 iteration)
    fn solve(
        &mut self,
        a: &Mat<T>,
        pc: Option<&(dyn crate::preconditioner::legacy::Preconditioner<Mat<T>, Vec<T>> + '_)>,
        b: &Vec<T>,
        x: &mut Vec<T>,
        _comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
        _work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<crate::utils::convergence::SolveStats<T>, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("QrSolve");
        
        let _ = pc; // Direct solvers do not use preconditioner
        
        // Call monitors at start if provided
        if let Some(monitors) = monitors {
            for monitor in monitors {
                monitor(0, T::zero());
            }
        }
        
        // Compute QR factorization
        let factor = Qr::new(a.as_ref());
        x.clone_from(b);
        let n = x.len();
        let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
        factor.solve_in_place_with_conj(Conj::No, x_mat);
        
        // Call monitors at end if provided
        if let Some(monitors) = monitors {
            for monitor in monitors {
                monitor(1, T::zero());
            }
        }
        
        Ok(crate::utils::convergence::SolveStats {
            iterations: 1,
            final_residual: T::zero(),
            reason: crate::utils::convergence::ConvergedReason::ConvergedAtol,
        })
    }
}

impl Default for QrSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Mat};
    use crate::solver::legacy::LinearSolver;

    #[test]
    fn lu_solver_solves_dense_system() {
        // 3x3 system: [[2,1,1],[1,3,2],[1,0,0]] x = [4,5,6]
        // True solution: [6,15,-23]
        let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0,0) => 2.0, (0,1) => 1.0, (0,2) => 1.0,
            (1,0) => 1.0, (1,1) => 3.0, (1,2) => 2.0,
            (2,0) => 1.0, (2,1) => 0.0, (2,2) => 0.0,
            _ => 0.0
        });
        let b = vec![4.0, 5.0, 6.0];
        let mut x = vec![0.0; 3];
        let mut solver = LuSolver::<f64>::new();
        let stats = solver.solve(&a, None, &b, &mut x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm), None, None).unwrap();
        let expected = vec![6.0, 15.0, -23.0];
        let tol = 1e-10;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((f64::from(*xi) - f64::from(*ei)).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(matches!(stats.reason,
            crate::utils::convergence::ConvergedReason::ConvergedAtol |
            crate::utils::convergence::ConvergedReason::ConvergedRtol), "LU did not report Converged reason");
    }

    #[test]
    fn qr_solver_solves_dense_system() {
        // 3x3 system: [[2,1,1],[1,3,2],[1,0,0]] x = [4,5,6]
        // True solution: [6,15,-23]
        let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0,0) => 2.0, (0,1) => 1.0, (0,2) => 1.0,
            (1,0) => 1.0, (1,1) => 3.0, (1,2) => 2.0,
            (2,0) => 1.0, (2,1) => 0.0, (2,2) => 0.0,
            _ => 0.0
        });
        let b = vec![4.0, 5.0, 6.0];
        let mut x = vec![0.0; 3];
        let mut solver = QrSolver::new();
        let stats = solver.solve(&a, None, &b, &mut x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm), None, None).unwrap();
        let expected = vec![6.0, 15.0, -23.0];
        let tol = 1e-10;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((f64::from(*xi) - f64::from(*ei)).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(matches!(stats.reason,
            crate::utils::convergence::ConvergedReason::ConvergedAtol |
            crate::utils::convergence::ConvergedReason::ConvergedRtol), "QR did not report Converged reason");
    }
}
