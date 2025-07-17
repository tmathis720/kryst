//! BiCGStab solver (Saad §7.1)
//!
//! This module implements the BiConjugate Gradient Stabilized (BiCGStab) method for solving
//! non-symmetric linear systems. BiCGStab is a popular Krylov subspace method that combines the
//! robustness of BiCG with smoother convergence, making it suitable for a wide range of problems.
//!
//! # Overview
//!
//! The BiCGStab algorithm iteratively refines the solution to Ax = b by constructing two coupled
//! sequences of vectors and scalars, using short recurrences and a shadow residual. It is especially
//! effective for large, sparse, and non-symmetric systems.
//!
//! # Usage
//!
//! - Create a `BiCgStabSolver` with the desired tolerance and maximum iterations.
//! - Call `solve` with the system matrix, right-hand side, and initial guess.
//! - The solver returns convergence statistics and the solution vector.
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Section 7.1.
//! - https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

use crate::core::traits::{InnerProduct, MatVec};
use crate::error::KError;
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats, ConvergedReason};
use crate::utils::profiling::StageGuard;

#[cfg(feature = "logging")]
use log::trace;

#[cfg(feature = "rayon")]
use rayon::prelude::*;
/// BiCGStab solver struct
///
/// Stores convergence parameters.
pub struct BiCgStabSolver<T>
where
    T: Send + Sync,
{
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float + From<f64> + std::ops::Mul<Output = T> + Send + Sync> BiCgStabSolver<T> {
    /// Create a new BiCGStab solver with the given tolerance and maximum iterations.
    pub fn new(tol: T, max_iters: usize) -> Self {
        let atol = <T as From<f64>>::from(1e-12);
        let dtol = <T as From<f64>>::from(1e3);
        Self { conv: Convergence::new(tol, atol, dtol, max_iters) }
    }
}

impl<M, V, T> LinearSolver<M, V> for BiCgStabSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone + From<f64> + Send + Sync + std::fmt::LowerExp,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the linear system Ax = b using the BiCGStab algorithm.
    ///
    /// # Arguments
    /// * `a` - System matrix
    /// * `pc` - Optional preconditioner (currently unused)
    /// * `b` - Right-hand side vector
    /// * `x` - Initial guess (input/output)
    /// * `comm` - Communicator for parallel reductions
    ///
    /// Returns convergence statistics and the solution vector.
    fn solve(&mut self, a: &M, pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>, b: &V, x: &mut V, comm: &crate::parallel::UniverseComm) -> Result<SolveStats<T>, KError> {
        // Call solve_with_monitors with empty monitor list
        self.solve_with_monitors(a, pc, b, x, comm, &[])
    }

    /// Solve the linear system Ax = b using the BiCGStab algorithm with iteration monitoring.
    ///
    /// # Arguments
    /// * `a` - System matrix
    /// * `pc` - Optional preconditioner (currently unused)
    /// * `b` - Right-hand side vector
    /// * `x` - Initial guess (input/output)
    /// * `comm` - Communicator for parallel reductions
    /// * `monitors` - Callbacks to invoke at each iteration with (iteration, residual_norm)
    ///
    /// Returns convergence statistics and the solution vector.
    fn solve_with_monitors(&mut self, a: &M, pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>, b: &V, x: &mut V, comm: &crate::parallel::UniverseComm, monitors: &[Box<dyn Fn(usize, T) + Send + Sync>]) -> Result<SolveStats<T>, KError> {
        let _solve_stage = StageGuard::new("BiCGStabSolve");
        
        #[cfg(feature = "logging")]
        trace!("Starting BiCGStab solve with {} monitors", monitors.len());

        let _ = pc; // BiCGStab does not use preconditioner (yet)
        let n = b.as_ref().len();
        let ip = ();
        let mut xk = x.as_ref().to_vec();
        
        // r0 = b - A x0
        let mut tmp = V::from(vec![T::zero(); n]);
        {
            let _matvec_stage = StageGuard::new("BiCGStabMatVec");
            a.matvec(&V::from(xk.clone()), &mut tmp);
        }
        let mut r = V::from(tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>());
        let r_hat = r.clone(); // shadow residual
        let mut rho_prev = T::one();
        let mut alpha = T::one();
        let mut omega_prev = T::one();
        let mut v = V::from(vec![T::zero(); n]);
        let mut p = r.clone(); // Properly initialize p = r
        
        // Compute initial residual norm
        let res0 = {
            let _norm_stage = StageGuard::new("BiCGStabNorm");
            ip.norm(&r, comm)
        };
        
        // Invoke monitors for iteration 0
        for monitor in monitors {
            monitor(0, res0);
        }

        let mut stats = SolveStats { 
            iterations: 0, 
            final_residual: res0, 
            reason: ConvergedReason::Continued 
        };
        
        // Check initial convergence using new interface
        let (reason, initial_stats) = self.conv.check(res0, res0, 0);
        if reason != ConvergedReason::Continued {
            *x = V::from(xk);
            return Ok(initial_stats);
        }

        #[cfg(feature = "logging")]
        trace!("BiCGStab initial residual: {:.3e}", res0);
        
        for i in 1..=self.conv.max_iters {
            let _iter_stage = StageGuard::new("BiCGStabIteration");
            
            #[cfg(feature = "logging")]
            trace!("BiCGStab iteration {}", i);

            // rho = <r_hat, r>
            let rho = {
                let _dot_stage = StageGuard::new("BiCGStabDotProduct");
                ip.dot(&r_hat, &r, comm)
            };
            
            if rho.abs() < T::epsilon() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: rho = {:.3e}", rho);
                break; // breakdown
            }
            
            let beta = if i == 1 {
                T::zero()
            } else {
                (rho / rho_prev) * (alpha / omega_prev)
            };
            
            // p = r + beta * (p - omega_prev * v)
            {
                let _axpy_stage = StageGuard::new("BiCGStabAxpy");
                #[cfg(feature = "rayon")]
                {
                    let beta = beta;
                    let omega = omega_prev;
                    p.as_mut().par_iter_mut()
                        .zip(r.as_ref().par_iter())
                        .zip(v.as_ref().par_iter())
                        .for_each(|((p_j, &r_j), &v_j)| {
                            *p_j = r_j + beta * (*p_j - omega * v_j);
                        });
                }
                #[cfg(not(feature = "rayon"))]
                {
                    for ((p_j, r_j), v_j) in p.as_mut().iter_mut().zip(r.as_ref()).zip(v.as_ref()) {
                        *p_j = *r_j + beta * (*p_j - omega_prev * *v_j);
                    }
                }
            }
            
            // v = A p
            {
                let _matvec_stage = StageGuard::new("BiCGStabMatVec");
                let mut v_tmp = V::from(vec![T::zero(); n]);
                a.matvec(&p.clone(), &mut v_tmp);
                v = v_tmp;
            }
            
            let alpha_num = rho;
            // alpha_den = <r_hat, v>
            let alpha_den = {
                let _dot_stage = StageGuard::new("BiCGStabDotProduct");
                ip.dot(&r_hat, &v, comm)
            };
            
            if alpha_den.abs() < T::epsilon() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: alpha_den = {:.3e}", alpha_den);
                break; // breakdown
            }
            alpha = alpha_num / alpha_den;
            
            // s = r - alpha * v
            let s = {
                let _axpy_stage = StageGuard::new("BiCGStabAxpy");
                #[cfg(feature = "rayon")]
                {
                    V::from(r.as_ref().par_iter().zip(v.as_ref().par_iter()).map(|(&rj, &vj)| rj - alpha * vj).collect::<Vec<_>>())
                }
                #[cfg(not(feature = "rayon"))]
                {
                    V::from(r.as_ref().iter().zip(v.as_ref()).map(|(&rj, &vj)| rj - alpha * vj).collect::<Vec<_>>())
                }
            };
            
            // Compute norm of s
            let s_norm = {
                let _norm_stage = StageGuard::new("BiCGStabNorm");
                ip.norm(&s, comm)
            };
            
            // Invoke monitors for current iteration (intermediate)
            for monitor in monitors {
                monitor(i, s_norm);
            }

            #[cfg(feature = "logging")]
            trace!("BiCGStab iteration {}: s_norm = {:.3e}", i, s_norm);
            
            // Check convergence for s using new interface  
            let (s_reason, s_stats) = self.conv.check(s_norm, res0, i);
            if s_reason != ConvergedReason::Continued {
                // Early convergence: update x and return
                {
                    let _axpy_stage = StageGuard::new("BiCGStabAxpy");
                    #[cfg(feature = "rayon")]
                    {
                        xk.par_iter_mut().zip(p.as_ref().par_iter()).for_each(|(xj, &pj)| {
                            *xj = *xj + alpha * pj;
                        });
                    }
                    #[cfg(not(feature = "rayon"))]
                    {
                        for (xj, pj) in xk.iter_mut().zip(p.as_ref()) {
                            *xj = *xj + alpha * *pj;
                        }
                    }
                }
                *x = V::from(xk);
                
                #[cfg(feature = "logging")]
                trace!("BiCGStab early convergence after {} iterations: {:?}", i, s_reason);
                return Ok(s_stats);
            }
            
            // t = A s
            let mut t = V::from(vec![T::zero(); n]);
            {
                let _matvec_stage = StageGuard::new("BiCGStabMatVec");
                a.matvec(&s, &mut t);
            }
            
            // omega = <t, s> / <t, t>
            let (omega_num, omega_den) = {
                let _dot_stage = StageGuard::new("BiCGStabDotProduct");
                let omega_num = ip.dot(&t, &s, comm);
                let omega_den = ip.dot(&t, &t, comm);
                (omega_num, omega_den)
            };
            
            if omega_den.abs() < T::epsilon() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: omega_den = {:.3e}", omega_den);
                break; // breakdown
            }
            let omega = omega_num / omega_den;
            
            // x = x + alpha * p + omega * s
            {
                let _axpy_stage = StageGuard::new("BiCGStabAxpy");
                #[cfg(feature = "rayon")]
                {
                    xk.par_iter_mut()
                        .zip(p.as_ref().par_iter())
                        .zip(s.as_ref().par_iter())
                        .for_each(|((xj, &pj), &sj)| {
                            *xj = *xj + alpha * pj + omega * sj;
                        });
                }
                #[cfg(not(feature = "rayon"))]
                {
                    for ((xj, pj), sj) in xk.iter_mut().zip(p.as_ref()).zip(s.as_ref()) {
                        *xj = *xj + alpha * *pj + omega * *sj;
                    }
                }
            }
            
            // r = s - omega * t
            let r_new = {
                let _axpy_stage = StageGuard::new("BiCGStabAxpy");
                #[cfg(feature = "rayon")]
                {
                    V::from(s.as_ref().par_iter().zip(t.as_ref().par_iter()).map(|(&sj, &tj)| sj - omega * tj).collect::<Vec<_>>())
                }
                #[cfg(not(feature = "rayon"))]
                {
                    V::from(s.as_ref().iter().zip(t.as_ref()).map(|(&sj, &tj)| sj - omega * tj).collect::<Vec<_>>())
                }
            };
            r = r_new;
            
            // Compute norm of r
            let r_norm = {
                let _norm_stage = StageGuard::new("BiCGStabNorm");
                ip.norm(&r, comm)
            };
            
            // Invoke monitors for current iteration (final)
            for monitor in monitors {
                monitor(i, r_norm);
            }

            #[cfg(feature = "logging")]
            trace!("BiCGStab iteration {}: residual = {:.3e}", i, r_norm);
            
            // Check convergence using new interface
            let (r_reason, r_stats) = self.conv.check(r_norm, res0, i);
            stats = r_stats;
            if r_reason != ConvergedReason::Continued {
                *x = V::from(xk);
                
                #[cfg(feature = "logging")]
                trace!("BiCGStab converged after {} iterations: {:?}", i, r_reason);
                return Ok(stats);
            }
            
            if omega.abs() < T::epsilon() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: omega = {:.3e}", omega);
                break; // breakdown
            }
            
            rho_prev = rho;
            omega_prev = omega;
        }
        
        *x = V::from(xk);
        
        #[cfg(feature = "logging")]
        trace!("BiCGStab solve completed: {} iterations, final residual: {:.3e}", 
               stats.iterations, stats.final_residual);
        
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use approx::assert_abs_diff_eq;

    // Helper: random well-conditioned non-symmetric 3x3 matrix
    fn nonsym_3x3() -> (Mat<f64>, Vec<f64>) {
        let a = Mat::from_fn(3, 3, |i, j| if i == j { 4.0 } else { (i + 2 * j) as f64 + 1.0 });
        let x_true = vec![1.0, 2.0, 3.0];
        let mut b = vec![0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                b[i] += a[(i, j)] * x_true[j];
            }
        }
        (a, b)
    }

    #[test]
    fn bicgstab_solves_well_conditioned_nonsym() {
        let (a, b) = nonsym_3x3();
        let mut x = vec![0.0; 3];
        let mut solver = BiCgStabSolver::new(1e-10, 100);
        let stats = solver.solve(&a, None, &b, &mut x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm)).unwrap();
        eprintln!("BiCGStab stats: {{ reason: {:?}, iters: {}, final_res: {:e} }}", stats.reason, stats.iterations, stats.final_residual);
        // Compare to true solution
        let x_true = vec![1.0, 2.0, 3.0];
        for i in 0..3 {
            assert_abs_diff_eq!(x[i], x_true[i], epsilon = 1e-8);
        }
        assert!(matches!(stats.reason, ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol), 
                "BiCGStab did not converge: stats = {:?}", stats);
    }
}
