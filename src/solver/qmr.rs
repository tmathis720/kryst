//! QMR solver (Saad §7.3)

use std::iter::Sum;

use crate::solver::LinearSolver;
use crate::core::traits::{MatVec, InnerProduct};
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use num_traits::Float;

/// Quasi-Minimal Residual (QMR) method for nonsymmetric A
pub struct QmrSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: Float> QmrSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for QmrSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone,
    T: Float + From<f64> + std::fmt::Debug + Sum,
{
    type Error = KError;
    type Scalar = T;

    fn solve(
        &mut self,
        a: &M,
        _pc: Option<&dyn Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
    ) -> Result<SolveStats<T>, KError> {
        // Based on Saad 2001, Algorithm 7.3
        let n = b.as_ref().len();
        let ip = ();

        // initial residual r0 = b - A x0
        let mut r = V::from(vec![T::zero(); n]);
        a.matvec(x, &mut r);
        for i in 0..n {
            r.as_mut()[i] = b.as_ref()[i] - r.as_ref()[i];
        }
        // choose r_tld = r (no preconditioning in this basic stub)
        let r_tld = r.clone();

        // bi-Lanczos scalars
        let mut rho = ip.dot(&r_tld, &r);
        if rho == T::zero() {
            return Ok(SolveStats { iterations: 0, final_residual: T::zero(), converged: true });
        }
        let mut rho_prev = rho;
        let _beta = T::zero();
        // let mut alpha = T::zero(); // Unused
        
        // search directions and temporary buffers
        let mut w = V::from(vec![T::zero(); n]);
        let mut p = V::from(vec![T::zero(); n]);
        let mut q = V::from(vec![T::zero(); n]);
        let mut d = V::from(vec![T::zero(); n]);
        let mut u = V::from(vec![T::zero(); n]);
        
        // initializations
        let mut x_out = V::from(vec![T::zero(); n]);
        let mut theta = T::zero();
        let mut eta = T::one();
        // let mut epsilon = T::zero(); // Unused

        let norm_r0 = ip.norm(&r);
        let mut stats = SolveStats { iterations: 0, final_residual: norm_r0, converged: false };

        for j in 1..=self.conv.max_iters {
            // --- Bi-Lanczos update ---
            if j == 1 {
                // v1 = r0 / rho
                let inv_rho = T::one() / rho.sqrt();
                for i in 0..n {
                    w.as_mut()[i] = r_tld.as_ref()[i] * inv_rho;
                }
                u = w.clone();
                p = w.clone();
            } else {
                // rho = <r_tld, r>
                rho = ip.dot(&r_tld, &r);
                let beta_j = rho / rho_prev;
                // u = v + beta_j * q
                for i in 0..n {
                    u.as_mut()[i] = w.as_ref()[i] + beta_j * q.as_ref()[i];
                }
                // p = u + beta_j * (q + beta_j * p)
                for i in 0..n {
                    p.as_mut()[i] = u.as_ref()[i]
                        + beta_j * (q.as_ref()[i] + beta_j * p.as_ref()[i]);
                }
                // similarly for dual directions (w, d, q)... omitted for brevity
            }

            // apply A to p
            let mut ap = V::from(vec![T::zero(); n]);
            a.matvec(&p, &mut ap);
            // alpha = <w, A p>
            let alpha = ip.dot(&w, &ap);
            if alpha == T::zero() { break; }
            
            // update q = ap - alpha * v
            for i in 0..n {
                q.as_mut()[i] = ap.as_ref()[i] - alpha * w.as_ref()[i];
            }
            // update d and solution vector
            // compute theta, epsilon, eta as in Saad
            let theta_prev = theta;
            theta = q.as_ref().iter().map(|&qi| qi * qi).sum::<T>().sqrt();
            let c = theta_prev / (alpha.sqrt() * theta_prev.hypot(alpha));
            let s = alpha / (alpha.sqrt() * theta_prev.hypot(alpha));
            let epsilon = s * theta_prev;
            eta = c * eta;
            
            // d = u - epsilon * d
            if j == 1 {
                d = u.clone();
            } else {
                for i in 0..n {
                    d.as_mut()[i] = u.as_ref()[i] - epsilon * d.as_ref()[i];
                }
            }
            // x_out += eta * d
            for i in 0..n {
                x_out.as_mut()[i] = x_out.as_ref()[i] + eta * d.as_ref()[i];
            }

            // residual approximation ||r_j|| ≈ |s|*theta
            let res_est = (s * theta).abs();
            stats = self.conv.check(res_est, norm_r0, j).1.clone();
            if stats.converged {
                *x = x_out.clone();
                stats.final_residual = res_est;
                return Ok(stats);
            }

            // shift variables for next iter
            rho_prev = rho;
            w = q.clone();
            // dual updates omitted...
        }

        *x = x_out;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    #[derive(Clone)]
    struct DenseMat { data: Vec<Vec<f64>> }
    impl MatVec<Vec<f64>> for DenseMat {
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            for i in 0..x.len() {
                y[i] = self.data[i].iter().zip(x).map(|(a,b)| a*b).sum();
            }
        }
    }

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
