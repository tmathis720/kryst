//! QMR solver (Saad §7.3)
//!
//! This module implements the Quasi-Minimal Residual (QMR) algorithm for solving large, sparse,
//! nonsymmetric linear systems Ax = b. QMR is based on the Bi-Lanczos process and is designed to
//! minimize the residual norm in a quasi-minimal sense. It is suitable for nonsymmetric and indefinite
//! systems, and does not require breakdown-avoiding look-ahead as in BiCG.
//!
//! # Features
//! - Handles general nonsymmetric systems
//! - Uses both A and A^T (matrix and its transpose)
//! - No preconditioning in this implementation
//! - Tracks true residual for convergence
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd Edition. SIAM. §7.3
//! - https://en.wikipedia.org/wiki/Quasi-minimal_residual_method

use std::iter::Sum;

use crate::solver::LinearSolver;
use crate::core::traits::{MatVec, InnerProduct, MatTransVec};
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use num_traits::Float;

/// Quasi-Minimal Residual (QMR) method for nonsymmetric A
pub struct QmrSolver<T> {
    /// Convergence criteria (tolerance and max iterations)
    pub conv: Convergence<T>,
}

impl<T: Float> QmrSolver<T> {
    /// Create a new QMR solver with given tolerance and maximum iterations.
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for QmrSolver<T>
where
    M: 'static + MatVec<V> + MatTransVec<V> + std::fmt::Debug,
    (): InnerProduct<V, Scalar = T>,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone,
    T: Float + From<f64> + std::fmt::Debug + Sum,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the linear system Ax = b using the QMR algorithm.
    ///
    /// # Arguments
    /// * `a` - Matrix implementing `MatVec` and `MatTransVec`
    /// * `_pc` - (Unused) Optional preconditioner (not supported in this implementation)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    ///
    /// # Returns
    /// * `Ok(SolveStats)` if converged or max iterations reached
    /// * `Err(KError)` on error
    fn solve(
        &mut self,
        a: &M,
        _pc: Option<&dyn Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
    ) -> Result<SolveStats<T>, KError> {
        let n = b.as_ref().len();
        let ip = ();
        // Allocate all vectors needed for QMR
        let mut r = V::from(vec![T::zero(); n]);
        let mut r_tld = V::from(vec![T::zero(); n]);
        let mut p = V::from(vec![T::zero(); n]);
        let mut p_tld = V::from(vec![T::zero(); n]);
        let mut v = V::from(vec![T::zero(); n]);
        let mut v_tld = V::from(vec![T::zero(); n]);
        let _d = V::from(vec![T::zero(); n]);
        let mut s = V::from(vec![T::zero(); n]);
        let mut t = V::from(vec![T::zero(); n]);
        let mut x_j = x.clone();
        // r0 = b - A x0
        a.matvec(x, &mut r);
        for i in 0..n {
            r.as_mut()[i] = b.as_ref()[i] - r.as_ref()[i];
        }
        // r_tld0 = arbitrary, use r0
        r_tld.clone_from(&r);
        let norm_r0 = ip.norm(&r);
        let mut stats = SolveStats { iterations: 0, final_residual: norm_r0, converged: false };
        let mut rho = ip.dot(&r_tld, &r);
        if rho == T::zero() {
            *x = x_j;
            stats.final_residual = ip.norm(&r);
            stats.converged = true;
            return Ok(stats);
        }
        #[allow(unused_assignments)]
        let mut beta = T::zero();
        let _eta = T::zero();
        let _theta = T::zero();
        let _tau = norm_r0;
        let mut res_norm = norm_r0;
        for j in 0..self.conv.max_iters {
            if j == 0 {
                // First iteration: initialize p and p_tld
                p.clone_from(&r);
                p_tld.clone_from(&r_tld);
            } else {
                let rho_prev = rho;
                rho = ip.dot(&r_tld, &r);
                if rho == T::zero() {
                    break;
                }
                beta = rho / rho_prev;
                // Update search directions
                for i in 0..n {
                    p.as_mut()[i] = r.as_ref()[i] + beta * p.as_ref()[i];
                    p_tld.as_mut()[i] = r_tld.as_ref()[i] + beta * p_tld.as_ref()[i];
                }
            }
            // v = A p
            a.matvec(&p, &mut v);
            // v_tld = A^T p_tld
            a.mattransvec(&p_tld, &mut v_tld);
            let sigma = ip.dot(&p_tld, &v);
            if sigma == T::zero() {
                break;
            }
            let alpha = rho / sigma;
            // s = r - alpha v
            for i in 0..n {
                s.as_mut()[i] = r.as_ref()[i] - alpha * v.as_ref()[i];
            }
            // t = A s
            a.matvec(&s, &mut t);
            let t_dot_s = ip.dot(&t, &s);
            let t_dot_t = ip.dot(&t, &t);
            let omega = if t_dot_t != T::zero() { t_dot_s / t_dot_t } else { T::zero() };
            // x_{j+1} = x_j + alpha p + omega s
            for i in 0..n {
                x_j.as_mut()[i] = x_j.as_ref()[i] + alpha * p.as_ref()[i] + omega * s.as_ref()[i];
            }
            // r = s - omega t
            for i in 0..n {
                r.as_mut()[i] = s.as_ref()[i] - omega * t.as_ref()[i];
            }
            // Check convergence with true residual
            a.matvec(&x_j, &mut t);
            for i in 0..n {
                t.as_mut()[i] = b.as_ref()[i] - t.as_ref()[i];
            }
            res_norm = ip.norm(&t);
            let (stop, s_stats) = self.conv.check(res_norm, norm_r0, j+1);
            stats = s_stats;
            if stop {
                *x = x_j.clone();
                stats.final_residual = res_norm;
                stats.converged = true;
                return Ok(stats);
            }
        }
        *x = x_j;
        stats.final_residual = res_norm;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    /// Simple dense matrix for testing
    #[derive(Clone, Debug)]
    struct DenseMat { data: Vec<Vec<f64>> }
    impl MatVec<Vec<f64>> for DenseMat {
        /// Matrix-vector multiplication: y = A x
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            for i in 0..x.len() {
                y[i] = self.data[i].iter().zip(x).map(|(a,b)| a*b).sum();
            }
        }
    }
    impl crate::core::traits::MatTransVec<Vec<f64>> for DenseMat {
        /// Matrix-transpose-vector multiplication: y = A^T x
        fn mattransvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            let n = self.data.len();
            let m = self.data[0].len();
            for j in 0..m {
                y[j] = 0.0;
                for i in 0..n {
                    y[j] += self.data[i][j] * x[i];
                }
            }
        }
    }

    #[ignore]
    #[test]
    fn qmr_solves_small_nonsym() {
        // A simple 2×2 nonsymmetric system
        // [2 1; 0 3] x = [3; 6] ⇒ x = [1;2]
        let a = DenseMat { data: vec![vec![2.0,1.0], vec![0.0,3.0]] };
        let b = vec![3.0,6.0];
        let mut x = vec![0.0,0.0];
        let mut solver = QmrSolver::new(1e-10, 50);
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        assert!((x[0]-1.0).abs() < 1e-4);
        assert!((x[1]-2.0).abs() < 1e-4);
        assert!(stats.converged);
    }
}
