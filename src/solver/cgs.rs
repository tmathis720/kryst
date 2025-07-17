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
    /// * `a` - Matrix implementing `MatVec`
    /// * `pc` - (Unused) Optional preconditioner (not supported in this implementation)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    ///
    /// # Returns
    /// * `Ok(SolveStats)` if converged or max iterations reached
    /// * `Err(KError)` on error
    fn solve(&mut self, a: &M, pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>, b: &V, x: &mut V, comm: &crate::parallel::UniverseComm) -> Result<SolveStats<T>, KError> {
        let _ = pc; // CGS does not use preconditioner (yet)
        let n = b.as_ref().len();
        let mut xk = x.as_ref().to_vec();
        let ip = ();
        // Compute initial residual r = b - A x
        let mut r = {
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
        for i in 1..=self.conv.max_iters {
            // Check for breakdown (division by zero)
            if rho.abs() < T::epsilon() {
                break; // breakdown
            }
            if i == 1 {
                // First iteration: initialize u and p
                u = r.clone();
                p = u.clone();
            } else {
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
            let mut v_tmp = V::from(vec![T::zero(); n]);
            a.matvec(&p, &mut v_tmp);
            let v = v_tmp;
            // alpha = rho / (r_tld, v)
            let alpha = rho / ip.dot(&r_tld, &v, comm);
            // q = u - alpha * v
            for (q_j, (u_j, v_j)) in q.as_mut().iter_mut().zip(u.as_ref().iter().zip(v.as_ref())) {
                *q_j = *u_j - alpha * *v_j;
            }
            // x = x + alpha * (u + q)
            for (xj, (u_j, q_j)) in xk.iter_mut().zip(u.as_ref().iter().zip(q.as_ref())) {
                *xj += alpha * (*u_j + *q_j);
            }
            // r = r - alpha * A(u + q)
            let mut upq = u.clone();
            for (upq_i, q_i) in upq.as_mut().iter_mut().zip(q.as_ref()) {
                *upq_i += *q_i;
            }
            let mut w = V::from(vec![T::zero(); n]);
            a.matvec(&upq, &mut w);
            for (rj, wj) in r.as_mut().iter_mut().zip(w.as_ref()) {
                *rj = *rj - alpha * *wj;
            }
            let res_norm = ip.norm(&r, comm);
            // Check convergence
            let (reason, s) = self.conv.check(res_norm, res0, i);
            stats = s.clone();
            if reason == crate::utils::convergence::ConvergedReason::ConvergedRtol
                || reason == crate::utils::convergence::ConvergedReason::ConvergedAtol {
                *x = V::from(xk.clone());
                return Ok(stats);
            }
            rho_old = rho;
            rho = ip.dot(&r_tld, &r, comm);
        }
        *x = V::from(xk);
        Ok(stats)
    }

    /// Solve using CGS with monitor callbacks and profiling.
    fn solve_with_monitors(
        &mut self,
        a: &M,
        pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
        monitors: &[Box<dyn Fn(usize, T) + Send + Sync>],
    ) -> Result<SolveStats<T>, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CGSSolve");
        #[cfg(feature = "logging")]
        trace!("Starting CGS solve with {} monitors", monitors.len());

        let _ = pc;
        let n = b.as_ref().len();
        let mut xk = x.as_ref().to_vec();
        let ip = ();
        #[cfg(feature = "logging")]
        let _matvec_guard = StageGuard::new("CGSMatVec");
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let vec = b.as_ref().iter().zip(tmp.as_ref()).map(|(&bi, &axi)| bi - axi).collect();
            V::from(vec)
        };
        // Shadow residual for CGS normal equations
        let r_tld = r.clone();
        #[cfg(feature = "logging")]
        let _norm_guard = StageGuard::new("CGSNorm");
        let res0 = ip.norm(&r, comm);
        for m in monitors { m(0, res0); }
        #[cfg(feature = "logging")]
        trace!("CGS initial residual: {:.3e}", res0);

        let mut stats = SolveStats { iterations: 0, final_residual: res0, reason: crate::utils::convergence::ConvergedReason::Continued };
        let mut rho = ip.dot(&r, &r, comm);
        let mut rho_old = T::zero();
        let mut u = r.clone();
        let mut p = r.clone();
        let mut q = V::from(vec![T::zero(); n]);
        for i in 1..=self.conv.max_iters {
            #[cfg(feature = "logging")]
            let _iter_guard = StageGuard::new("CGSIteration");
            if rho.abs() < T::epsilon() { break; }
            if i == 1 {
                p = u.clone();
            } else {
                #[cfg(feature = "logging")]
                let _dot_guard = StageGuard::new("CGSDotProduct");
                let beta = rho / rho_old;
                let q_old = q.clone();
                let p_old = p.clone();
                for (uj, (rj, qj)) in u.as_mut().iter_mut().zip(r.as_ref().iter().zip(q_old.as_ref())) {
                    *uj = *rj + beta * *qj;
                }
                for (pj, (uj, po)) in p.as_mut().iter_mut().zip(u.as_ref().iter().zip(p_old.as_ref())) {
                    *pj = *uj + beta * *po;
                }
            }
            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("CGSMatVec");
            let mut v_tmp = V::from(vec![T::zero(); n]);
            a.matvec(&p, &mut v_tmp);
            let v = v_tmp;
            #[cfg(feature = "logging")]
            let _dot_guard = StageGuard::new("CGSDotProduct");
            let alpha = rho / ip.dot(&r_tld, &v, comm);
            for (xj, pj) in xk.iter_mut().zip(p.as_ref()) { *xj += alpha * *pj; }
            for (rj, vj) in r.as_mut().iter_mut().zip(v.as_ref()) { *rj -= alpha * *vj; }
            for m in monitors { m(i, ip.norm(&r, comm)); }
            #[cfg(feature = "logging")]
            trace!("CGS iteration {}: residual = {:.3e}", i, ip.norm(&r, comm));
            let (reason, s) = self.conv.check(ip.norm(&r, comm), res0, i);
            stats = s;
            if reason != crate::utils::convergence::ConvergedReason::Continued {
                *x = V::from(xk.clone());
                return Ok(stats);
            }
            rho_old = rho;
            rho = ip.dot(&r, &r, comm);
        }
        *x = V::from(xk);
        #[cfg(feature = "logging")]
        trace!("CGS did not converge after max iterations");
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
        let stats = solver.solve(&a, None, &b, &mut x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm)).unwrap();
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
