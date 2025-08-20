//! CGNR/CGNE solvers (Saad Ch 8.3)
//!
//! This module implements the CGNR (Conjugate Gradient on the Normal Residual) and CGNE (Conjugate Gradient on the Normal Equations)
//! methods for solving least-squares problems and non-square linear systems. Both methods reduce the original system to a symmetric
//! positive definite system and apply the Conjugate Gradient algorithm.
//!
//! # Overview
//!
//! - CGNR solves (AᵗA)x = Aᵗb by applying CG to the normal equations for the residual.
//! - CGNE solves (AAᵗ)y = b, then x = Aᵗy, by applying CG to the normal equations for the error.
//! - Both are suitable for overdetermined or underdetermined systems and least-squares problems.
//!
//! # Usage
//!
//! - Create a `CgnrSolver` or `CgneSolver` with the desired tolerance and maximum iterations.
//! - Call `solve` with the system matrix, right-hand side, and initial guess.
//! - The solver returns convergence statistics and the solution vector.
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Section 8.3.
//! - https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_normal_equations

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{SolveStats, Convergence, ConvergedReason};
use crate::error::KError;
#[cfg(feature = "logging")]
use log::trace;
#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;

/// CGNR solver struct.
///
/// Stores convergence parameters.
pub struct CgnrSolver<T> {
    pub conv: Convergence<T>,
}

/// CGNE solver struct.
///
/// Stores convergence parameters.
pub struct CgneSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float + From<f64>> CgnrSolver<T> {
    /// Create a new CGNR solver with the given tolerance and maximum iterations.
    pub fn new(rtol: T, max_iters: usize) -> Self {
        let atol = <T as From<f64>>::from(1e-12);
        let dtol = <T as From<f64>>::from(1e3);
        Self { conv: Convergence::new(rtol, atol, dtol, max_iters) }
    }
}

impl<T: num_traits::Float + From<f64>> CgneSolver<T> {
    /// Create a new CGNE solver with the given tolerance and maximum iterations.
    pub fn new(rtol: T, max_iters: usize) -> Self {
        let atol = <T as From<f64>>::from(1e-12);
        let dtol = <T as From<f64>>::from(1e3);
        Self { conv: Convergence::new(rtol, atol, dtol, max_iters) }
    }
}

impl<M: ?Sized, V, T> LinearSolver<M, V> for CgnrSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone + From<f64> + std::fmt::Debug,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the least-squares problem using CGNR (CG on the normal residual).
    ///
    /// # Arguments
    /// * `a` - System matrix
    /// * `pc` - Optional preconditioner (currently unused)
    /// * `b` - Right-hand side vector
    /// * `x` - Initial guess (input/output)
    /// * `comm` - Communication object for parallel computation
    /// * `monitors` - Optional monitor callbacks for iteration progress
    /// * `work` - Optional workspace for reusable allocations
    ///
    /// Returns convergence statistics and the solution vector.
    fn solve(
        &mut self,
        a: &M,
        pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
        work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<T>, KError> {
        // Check runtime profiling and monitoring flags
        let use_monitors = monitors.is_some();
        let _has_workspace = work.is_some();

        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CGNRSolve");
        #[cfg(feature = "logging")]
        trace!("Starting CGNR solve, monitoring: {}, workspace: {}", use_monitors, _has_workspace);

        let _ = pc; // CGNR does not use preconditioner (yet)
        let n = b.as_ref().len();
        let mut xk = x.as_ref().to_vec();
        let ip = ();
        
        // Compute initial residual r = b - A x
        let mut r = {
            #[cfg(feature = "logging")]
            let _matvec_stage = StageGuard::new("CGNRMatVec");
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = b.as_ref().iter().zip(tmp.as_ref()).map(|(&bi, &axi)| bi - axi).collect::<Vec<_>>();
            V::from(r_vec)
        };
        
        let mut z = {
            #[cfg(feature = "logging")]
            let _matvec_stage = StageGuard::new("CGNRMatVec");
            let mut z = V::from(vec![T::zero(); n]);
            a.matvec(&r, &mut z); // z = A^T r (for CGNR, A^T = A^T)
            z
        };
        
        let mut p = z.clone();
        let mut rz = ip.dot(&z, &z, comm);
        let res0 = ip.norm(&r, comm);
        let mut stats = SolveStats { 
            iterations: 0, 
            final_residual: res0, 
            reason: ConvergedReason::Continued 
        };

        // Invoke monitors for iteration 0
        if use_monitors {
            for monitor in monitors.unwrap() {
                monitor(0, res0);
            }
        }

        #[cfg(feature = "logging")]
        trace!("CGNR initial residual: {:?}", res0);

        for i in 1..=self.conv.max_iters {
            #[cfg(feature = "logging")]
            let _iter_stage = StageGuard::new("CGNRIteration");
            
            #[cfg(feature = "logging")]
            trace!("CGNR iteration {}", i);

            // Compute Ap = A p
            let ap = {
                #[cfg(feature = "logging")]
                let _matvec_stage = StageGuard::new("CGNRMatVec");
                let mut ap = V::from(vec![T::zero(); n]);
                a.matvec(&p, &mut ap);
                ap
            };
            
            // Compute AtAp = A^T (A p)
            let at_ap = {
                #[cfg(feature = "logging")]
                let _matvec_stage = StageGuard::new("CGNRMatVec");
                let mut at_ap = V::from(vec![T::zero(); n]);
                a.matvec(&ap, &mut at_ap);
                at_ap
            };
            
            // Compute step size alpha
            let denom = {
                #[cfg(feature = "logging")]
                let _dot_stage = StageGuard::new("CGNRDotProduct");
                ip.dot(&at_ap, &at_ap, comm)
            };
            
            if denom <= T::zero() {
                #[cfg(feature = "logging")]
                trace!("CGNR indefinite matrix detected at iter {}", i);
                stats.iterations = i;
                stats.final_residual = ip.norm(&r, comm);
                stats.reason = ConvergedReason::DivergedDtol;
                return Err(KError::IndefiniteMatrix);
            }
            
            let alpha = rz / denom;
            
            // Update x and r
            {
                #[cfg(feature = "logging")]
                let _axpy_stage = StageGuard::new("CGNRAxpy");
                for (xj, pj) in xk.iter_mut().zip(p.as_ref()) {
                    *xj = *xj + alpha * *pj;
                }
                for (rj, apj) in r.as_mut().iter_mut().zip(ap.as_ref()) {
                    *rj = *rj - alpha * *apj;
                }
            }
            
            {
                #[cfg(feature = "logging")]
                let _matvec_stage = StageGuard::new("CGNRMatVec");
                a.matvec(&r, &mut z); // z = A^T r
            }
            
            let rz_new = {
                #[cfg(feature = "logging")]
                let _dot_stage = StageGuard::new("CGNRDotProduct");
                ip.dot(&z, &z, comm)
            };
            
            let res_norm = {
                #[cfg(feature = "logging")]
                let _norm_stage = StageGuard::new("CGNRNorm");
                ip.norm(&r, comm)
            };
            
            // Invoke monitors for current iteration
            if use_monitors {
                for monitor in monitors.unwrap() {
                    monitor(i, res_norm);
                }
            }

            #[cfg(feature = "logging")]
            trace!("CGNR iteration {}: residual = {:?}", i, res_norm);
            
            // Check convergence using new interface
            let (reason, new_stats) = self.conv.check(res_norm, res0, i);
            stats = new_stats;
            if reason != ConvergedReason::Continued {
                *x = V::from(xk.clone());
                
                #[cfg(feature = "logging")]
                trace!("CGNR converged after {} iterations: {:?}", i, reason);
                return Ok(stats);
            }
            
            // Update search direction
            let beta = rz_new / rz;
            {
                #[cfg(feature = "logging")]
                let _axpy_stage = StageGuard::new("CGNRAxpy");
                let p_old = p.clone();
                for ((pj, zj), old_pj) in p.as_mut().iter_mut().zip(z.as_ref()).zip(p_old.as_ref()) {
                    *pj = *zj + beta * *old_pj;
                }
            }
            rz = rz_new;
        }
        
        *x = V::from(xk);
        
        #[cfg(feature = "logging")]
        trace!("CGNR solve completed: {} iterations, final residual: {:?}", 
               stats.iterations, stats.final_residual);
        
        Ok(stats)
    }
}

impl<M, V, T> LinearSolver<M, V> for CgneSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone + From<f64>,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the least-squares problem using CGNE (CG on the normal equations for the error).
    ///
    /// # Arguments
    /// * `a` - System matrix
    /// * `pc` - Optional preconditioner (currently unused)
    /// * `b` - Right-hand side vector
    /// * `x` - Initial guess (input/output)
    /// * `comm` - Communication object for parallel computation
    /// * `monitors` - Optional monitor callbacks for iteration progress
    /// * `work` - Optional workspace for reusable allocations
    ///
    /// Returns convergence statistics and the solution vector.
    fn solve(
        &mut self,
        a: &M,
        pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
        work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<T>, KError> {
        // Check runtime profiling and monitoring flags
        let use_monitors = monitors.is_some();
        let _has_workspace = work.is_some();

        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CGNESolve");
        #[cfg(feature = "logging")]
        trace!("Starting CGNE solve, monitoring: {}, workspace: {}", use_monitors, _has_workspace);

        let _ = pc; // CGNE does not use preconditioner (yet)
        let n = b.as_ref().len();
        let mut xk = x.as_ref().to_vec();
        let ip = ();
        
        // Compute initial residual r = b - A x
        let mut r = {
            #[cfg(feature = "logging")]
            let _matvec_stage = StageGuard::new("CGNEMatVec");
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = b.as_ref().iter().zip(tmp.as_ref()).map(|(&bi, &axi)| bi - axi).collect::<Vec<_>>();
            V::from(r_vec)
        };
        
        let mut z = {
            #[cfg(feature = "logging")]
            let _matvec_stage = StageGuard::new("CGNEMatVec");
            let mut z = V::from(vec![T::zero(); n]);
            a.matvec(&r, &mut z); // z = A^T r (for CGNE, A^T = A^T)
            z
        };
        
        let mut p = z.clone();
        let mut rz = ip.dot(&z, &z, comm);
        let res0 = ip.norm(&r, comm);
        let mut stats = SolveStats { 
            iterations: 0, 
            final_residual: res0, 
            reason: ConvergedReason::Continued 
        };

        // Invoke monitors for iteration 0
        if use_monitors {
            for monitor in monitors.unwrap() {
                monitor(0, res0);
            }
        }

        #[cfg(feature = "logging")]
        trace!("CGNE initial residual computed");

        for i in 1..=self.conv.max_iters {
            #[cfg(feature = "logging")]
            let _iter_stage = StageGuard::new("CGNEIteration");
            
            #[cfg(feature = "logging")]
            trace!("CGNE iteration {}", i);

            // Compute At_p = A p
            let at_p = {
                #[cfg(feature = "logging")]
                let _matvec_stage = StageGuard::new("CGNEMatVec");
                let mut at_p = V::from(vec![T::zero(); n]);
                a.matvec(&p, &mut at_p);
                at_p
            };
            
            // Compute Ap = A^T (A p)
            let ap = {
                #[cfg(feature = "logging")]
                let _matvec_stage = StageGuard::new("CGNEMatVec");
                let mut ap = V::from(vec![T::zero(); n]);
                a.matvec(&at_p, &mut ap);
                ap
            };
            
            // Compute step size alpha
            let denom = {
                #[cfg(feature = "logging")]
                let _dot_stage = StageGuard::new("CGNEDotProduct");
                ip.dot(&ap, &ap, comm)
            };
            
            if denom <= T::zero() {
                #[cfg(feature = "logging")]
                trace!("CGNE indefinite matrix detected at iter {}", i);
                stats.iterations = i;
                stats.final_residual = ip.norm(&r, comm);
                stats.reason = ConvergedReason::DivergedDtol;
                return Err(KError::IndefiniteMatrix);
            }
            
            let alpha = rz / denom;
            
            // Update x and r
            {
                #[cfg(feature = "logging")]
                let _axpy_stage = StageGuard::new("CGNEAxpy");
                for (xj, pj) in xk.iter_mut().zip(p.as_ref()) {
                    *xj = *xj + alpha * *pj;
                }
                for (rj, at_pj) in r.as_mut().iter_mut().zip(at_p.as_ref()) {
                    *rj = *rj - alpha * *at_pj;
                }
            }
            
            {
                #[cfg(feature = "logging")]
                let _matvec_stage = StageGuard::new("CGNEMatVec");
                a.matvec(&r, &mut z); // z = A^T r
            }
            
            let rz_new = {
                #[cfg(feature = "logging")]
                let _dot_stage = StageGuard::new("CGNEDotProduct");
                ip.dot(&z, &z, comm)
            };
            
            let res_norm = {
                #[cfg(feature = "logging")]
                let _norm_stage = StageGuard::new("CGNENorm");
                ip.norm(&r, comm)
            };
            
            // Invoke monitors for current iteration
            if use_monitors {
                for monitor in monitors.unwrap() {
                    monitor(i, res_norm);
                }
            }

            #[cfg(feature = "logging")]
            trace!("CGNE iteration {}: residual computed", i);
            
            // Check convergence using new interface
            let (reason, new_stats) = self.conv.check(res_norm, res0, i);
            stats = new_stats;
            if reason != ConvergedReason::Continued {
                *x = V::from(xk.clone());
                
                #[cfg(feature = "logging")]
                trace!("CGNE converged after {} iterations", i);
                return Ok(stats);
            }
            
            // Update search direction
            let beta = rz_new / rz;
            {
                #[cfg(feature = "logging")]
                let _axpy_stage = StageGuard::new("CGNEAxpy");
                let p_old = p.clone();
                for ((pj, zj), old_pj) in p.as_mut().iter_mut().zip(z.as_ref()).zip(p_old.as_ref()) {
                    *pj = *zj + beta * *old_pj;
                }
            }
            rz = rz_new;
        }
        
        *x = V::from(xk);
        
        #[cfg(feature = "logging")]
        trace!("CGNE solve completed: {} iterations", stats.iterations);
        
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    #[derive(Clone)]
    struct DenseMat {
        data: Vec<Vec<f64>>,
    }
    impl MatVec<Vec<f64>> for DenseMat {
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            for (i, row) in self.data.iter().enumerate() {
                y[i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
        }
    }

    #[test]
    fn cgnr_solves_simple_least_squares() {
        // Overdetermined system: minimize ||Ax - b||
        // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 3]
        // Least squares solution: x = [1, 2]
        let a = DenseMat { data: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]] };
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0, 0.0];
        let mut solver = CgnrSolver::new(1e-10, 50);
        let stats = solver.solve(&a, None, &b, &mut x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm), None, None).unwrap();
        let expected = vec![1.0, 2.0];
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(matches!(stats.reason, ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol), 
                "CGNR did not converge, reason: {:?}", stats.reason);
    }

    #[test]
    fn cgne_solves_simple_least_squares() {
        // Same system as above
        let a = DenseMat { data: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]] };
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0, 0.0];
        let mut solver = CgneSolver::new(1e-10, 50);
        let stats = solver.solve(&a, None, &b, &mut x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm), None, None).unwrap();
        let expected = vec![1.0, 2.0];
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(matches!(stats.reason, ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol), 
                "CGNE did not converge, reason: {:?}", stats.reason);
    }
}
