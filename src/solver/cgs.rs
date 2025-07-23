//! Conjugate Gradient Squared (CGS) Solver
//!
//! This module implements the CGS iterative method for solving nonsymmetric linear systems Ax = b.
//! The CGS algorithm is based on the BiConjugate Gradient (BiCG) method, but squares the residual
//! polynomials to achieve faster convergence in some cases. It is suitable for large, sparse, nonsymmetric
//! systems, but may suffer from breakdowns or instability for ill-conditioned problems.
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd Edition. SIAM. §7.2
//! - https://en.wikipedia.org/wiki/Conjugate_gradient_squared_method

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
#[cfg(feature = "logging")]
use log::trace;
#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;

/// CGS solver struct, holding convergence parameters.
///
/// # Type Parameters
/// * `T` - Scalar type (e.g., f32, f64)
pub struct CgsSolver<T> {
    /// Convergence criteria (multi-threshold, max iterations)
    pub conv: Convergence<T>,
}

impl<T> CgsSolver<T>
where T: num_traits::Float + Clone + Send + Sync + std::fmt::Debug + std::fmt::LowerExp,
{
    /// Create a new CGS solver with given tolerance and maximum iterations.
    ///
    /// # Arguments
    /// * `rtol` - Relative residual tolerance for convergence
    /// * `max_iters` - Maximum number of iterations
    pub fn new(rtol: T, max_iters: usize) -> Self {
        let atol = num_traits::cast::<f64, T>(1e-12).unwrap();
        let dtol = num_traits::cast::<f64, T>(1e3).unwrap();
        Self {
            conv: Convergence {
                rtol,
                atol,
                dtol,
                max_iters,
            },
        }
    }
}

impl<M, V, T> LinearSolver<M, V> for CgsSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone + From<f64> + std::fmt::Debug + std::ops::AddAssign + std::ops::SubAssign + Send + Sync + std::fmt::LowerExp,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the linear system Ax = b using the CGS algorithm.
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
        let _guard = StageGuard::new("CGSSolve");
        #[cfg(feature = "logging")]
        trace!("Starting CGS solve, monitoring: {}, workspace: {}", use_monitors, _has_workspace);

        let _ = pc; // CGS does not use preconditioner (yet)
        let n = b.as_ref().len();
        let mut xk = x.as_ref().to_vec();
        let ip = ();
        
        // Compute initial residual r = b - A x
        let mut r = {
            #[cfg(feature = "logging")]
            let _matvec_stage = StageGuard::new("CGSMatVec");
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = b.as_ref().iter().zip(tmp.as_ref()).map(|(&bi, &axi)| bi - axi).collect::<Vec<_>>();
            V::from(r_vec)
        };
        
        let r_tld = r.clone(); // Shadow residual (fixed for all iterations)
        let mut p = r.clone(); // Search direction
        let mut q = V::from(vec![T::zero(); n]); // Auxiliary vector
        let mut u = V::from(vec![T::zero(); n]); // Auxiliary vector
        let mut rho = ip.dot(&r_tld, &r, comm); // BiCG-like scalar
        let mut rho_old = T::zero();
        let res0 = ip.norm(&r, comm); // Initial residual norm
        
        let mut stats = SolveStats {
            iterations: 0,
            final_residual: res0,
            reason: crate::utils::convergence::ConvergedReason::Continued,
        };

        // Invoke monitors for iteration 0
        if use_monitors {
            for monitor in monitors.unwrap() {
                monitor(0, res0);
            }
        }

        #[cfg(feature = "logging")]
        trace!("CGS initial residual: {:?}", res0);

        for i in 1..=self.conv.max_iters {
            #[cfg(feature = "logging")]
            let _iter_stage = StageGuard::new("CGSIteration");
            
            #[cfg(feature = "logging")]
            trace!("CGS iteration {}", i);

            // Check for breakdown (division by zero)
            if rho.abs() < T::epsilon() {
                #[cfg(feature = "logging")]
                trace!("CGS breakdown detected at iteration {}", i);
                stats.iterations = i;
                stats.final_residual = ip.norm(&r, comm);
                stats.reason = crate::utils::convergence::ConvergedReason::DivergedDtol;
                return Err(KError::IndefiniteMatrix);
            }
            
            if i == 1 {
                // First iteration: initialize u and p
                u = r.clone();
                p = u.clone();
            } else {
                #[cfg(feature = "logging")]
                let _update_stage = StageGuard::new("CGSUpdate");
                let beta = rho / rho_old;
                // Save q and p from previous iteration
                let q_old = q.clone();
                let p_old = p.clone();
                // u = r + beta * q_old
                for (u_j, (r_j, qj_old)) in u.as_mut().iter_mut().zip(r.as_ref().iter().zip(q_old.as_ref())) {
                    *u_j = *r_j + beta * *qj_old;
                }
                // p = u + beta * (q_old + beta * p_old)
                for ((p_j, u_j), (qj_old, p_oldj)) in p.as_mut().iter_mut().zip(u.as_ref()).zip(q_old.as_ref().iter().zip(p_old.as_ref())) {
                    *p_j = *u_j + beta * (*qj_old + beta * *p_oldj);
                }
            }
            
            // v = A p
            let v = {
                #[cfg(feature = "logging")]
                let _matvec_stage = StageGuard::new("CGSMatVec");
                let mut v_tmp = V::from(vec![T::zero(); n]);
                a.matvec(&p, &mut v_tmp);
                v_tmp
            };
            
            // alpha = rho / (r_tld, v)
            let alpha = {
                #[cfg(feature = "logging")]
                let _dot_stage = StageGuard::new("CGSDotProduct");
                rho / ip.dot(&r_tld, &v, comm)
            };
            
            // q = u - alpha * v
            {
                #[cfg(feature = "logging")]
                let _axpy_stage = StageGuard::new("CGSAxpy");
                for (q_j, (u_j, v_j)) in q.as_mut().iter_mut().zip(u.as_ref().iter().zip(v.as_ref())) {
                    *q_j = *u_j - alpha * *v_j;
                }
            }
            
            // x = x + alpha * (u + q)
            {
                #[cfg(feature = "logging")]
                let _axpy_stage = StageGuard::new("CGSAxpy");
                for (xj, (u_j, q_j)) in xk.iter_mut().zip(u.as_ref().iter().zip(q.as_ref())) {
                    *xj += alpha * (*u_j + *q_j);
                }
            }
            
            // r = r - alpha * A(u + q)
            {
                #[cfg(feature = "logging")]
                let _matvec_stage = StageGuard::new("CGSMatVec");
                let mut upq = u.clone();
                for (upq_i, q_i) in upq.as_mut().iter_mut().zip(q.as_ref()) {
                    *upq_i += *q_i;
                }
                let mut w = V::from(vec![T::zero(); n]);
                a.matvec(&upq, &mut w);
                for (rj, wj) in r.as_mut().iter_mut().zip(w.as_ref()) {
                    *rj = *rj - alpha * *wj;
                }
            }
            
            let res_norm = {
                #[cfg(feature = "logging")]
                let _norm_stage = StageGuard::new("CGSNorm");
                ip.norm(&r, comm)
            };
            
            // Invoke monitors for current iteration
            if use_monitors {
                for monitor in monitors.unwrap() {
                    monitor(i, res_norm);
                }
            }

            #[cfg(feature = "logging")]
            trace!("CGS iteration {}: residual = {:?}", i, res_norm);
            
            // Check convergence using new interface
            let (reason, new_stats) = self.conv.check(res_norm, res0, i);
            stats = new_stats;
            if reason != crate::utils::convergence::ConvergedReason::Continued {
                *x = V::from(xk.clone());
                
                #[cfg(feature = "logging")]
                trace!("CGS converged after {} iterations: {:?}", i, reason);
                return Ok(stats);
            }
            
            rho_old = rho;
            rho = ip.dot(&r_tld, &r, comm);
        }
        
        *x = V::from(xk);
        
        #[cfg(feature = "logging")]
        trace!("CGS solve completed: {} iterations, final residual: {:?}", 
               stats.iterations, stats.final_residual);
        
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    /// Simple dense matrix for testing
    #[derive(Clone)]
    struct DenseMat {
        data: Vec<Vec<f64>>,
    }
    impl MatVec<Vec<f64>> for DenseMat {
        /// Matrix-vector multiplication: y = A x
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            for (i, row) in self.data.iter().enumerate() {
                y[i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
        }
    }

    #[test]
    fn cgs_solves_large_well_conditioned_nonsym() {
        // 5x5 diagonally dominant, non-symmetric system
        // A = [[10,2,0,0,0],[3,15,4,0,0],[0,-2,8,1,0],[0,0,1,7,3],[0,0,0,2,12]]
        // x_true = [1,2,3,4,5]
        // b = A * x_true
        let a = DenseMat {
            data: vec![
                vec![10.0, 2.0, 0.0, 0.0, 0.0],
                vec![3.0, 15.0, 4.0, 0.0, 0.0],
                vec![0.0, -2.0, 8.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0, 7.0, 3.0],
                vec![0.0, 0.0, 0.0, 2.0, 12.0],
            ]
        };
        let x_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = {
            let mut b = vec![0.0; 5];
            a.matvec(&x_true, &mut b);
            b
        };
        let mut x = vec![0.0; 5];
        let mut solver = CgsSolver::new(1e-10, 200);
        let stats = solver.solve(&a, None, &b, &mut x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm), None, None).unwrap();
        let tol = 1e-6;
        for (xi, ei) in x.iter().zip(x_true.iter()) {
            assert!((xi - ei).abs() <= tol, "xi = {:.6}, expected = {:.6}", xi, ei);
        }
        assert!(matches!(stats.reason,
            crate::utils::convergence::ConvergedReason::ConvergedRtol |
            crate::utils::convergence::ConvergedReason::ConvergedAtol),
            "CGS did not report Converged reason");
    }
}
