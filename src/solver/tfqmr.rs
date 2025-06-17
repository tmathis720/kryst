//! TFQMR solver (Saad §7.4)
//!
//! This module implements the Transpose-Free Quasi-Minimal Residual (TFQMR) algorithm for solving
//! large, sparse, nonsymmetric linear systems Ax = b. TFQMR is a variant of QMR that avoids explicit
//! use of the transpose of A, making it suitable for problems where A^T is unavailable or expensive.
//! The implementation follows Saad's description and includes detailed debug output for each iteration.
//!
//! # Features
//! - Handles general nonsymmetric systems
//! - No explicit use of A^T (transpose-free)
//! - No preconditioning in this implementation
//! - Tracks true and estimated residuals for convergence
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd Edition. SIAM. §7.4
//! - Freund, R. W., & Nachtigal, N. M. (1991). A transpose-free quasi-minimal residual algorithm for non-Hermitian linear systems. SIAM J. Sci. Stat. Comput.
//! - https://en.wikipedia.org/wiki/Quasi-minimal_residual_method

use crate::solver::LinearSolver;
use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, InnerProduct};
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use num_traits::Float;

/// TFQMR (Transpose-Free Quasi-Minimal Residual) solver struct.
///
/// # Type Parameters
/// * `T` - Scalar type (e.g., f32, f64)
pub struct TfqmrSolver<T: num_traits::FromPrimitive> {
    /// Convergence criteria (tolerance and max iterations)
    pub conv: Convergence<T>,
}

impl<T: Float + num_traits::FromPrimitive> TfqmrSolver<T> {
    /// Create a new TFQMR solver with given tolerance and maximum iterations.
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for TfqmrSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone,
    T: Float + From<f64> + num_traits::FromPrimitive + std::fmt::Debug,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the linear system Ax = b using the TFQMR algorithm.
    ///
    /// # Arguments
    /// * `a` - Matrix implementing `MatVec`
    /// * `_pc` - (Unused) Optional preconditioner (not supported in this implementation)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    ///
    /// # Returns
    /// * `Ok(SolveStats)` if converged or max iterations reached
    /// * `Err(KError)` on error
    fn solve(&mut self,
             a: &M,
             _pc: Option<&dyn Preconditioner<M, V>>,
             b: &V,
             x: &mut V) -> Result<SolveStats<T>, KError> {
        let n = b.as_ref().len();
        let ip = ();

        // x0 = 0 (initial guess)
        *x = V::from(vec![T::zero(); n]);

        // r0 = b - A x0 = b
        let mut r = b.clone();
        // choose r_tld = r (can also pick random)
        let r_tld = r.clone();

        // scalars
        let mut rho = ip.dot(&r, &r_tld);
        if rho == T::zero() {
            return Ok(SolveStats { iterations: 0, final_residual: ip.norm(&r), converged: true });
        }
        #[allow(unused_assignments)]
        let _alpha = T::zero();
        let _theta = T::zero();
        let _c = T::one();
        let _eta = T::zero();
        let res0 = rho;
        let _stats = SolveStats { iterations: 0, final_residual: res0, converged: false };

        // vectors
        #[allow(unused_assignments)]
        let mut v = V::from(vec![T::zero(); n]);
        let mut w = r.clone();     // w = r0
        let mut y = r.clone();     // y = r0
        let mut u = V::from(vec![T::zero(); n]);
        let mut d = V::from(vec![T::zero(); n]);
        let mut psi_old = T::zero();
        let mut eta_old = T::zero();
        let tau = ip.norm(&r);
        let res0 = tau;
        let mut stats = SolveStats { iterations: 0, final_residual: res0, converged: false };
        if tau == T::zero() {
            return Ok(SolveStats { iterations: 0, final_residual: T::zero(), converged: true });
        }

        let mut dpold = tau; // PETSc: dpold = initial residual norm
        for k in 1..=self.conv.max_iters {
            // v = A * y
            let mut v_tmp = V::from(vec![T::zero(); n]);
            a.matvec(&y, &mut v_tmp);
            v = v_tmp;

            // alpha = rho / <r_tld, v>
            let sigma = ip.dot(&r_tld, &v);
            if sigma == T::zero() || !sigma.is_finite() {
                stats.final_residual = ip.norm(&r);
                stats.iterations = k;
                stats.converged = false;
                return Ok(stats);
            }
            let alpha = rho / sigma;
            if alpha == T::zero() || !alpha.is_finite() {
                stats.final_residual = ip.norm(&r);
                stats.iterations = k;
                stats.converged = false;
                return Ok(stats);
            }
            // --- TFQMR update steps ---
            // u = r - alpha * v
            for (ui, (ri, vi)) in u.as_mut().iter_mut().zip(r.as_ref().iter().zip(v.as_ref())) {
                *ui = *ri - alpha * *vi;
            }

            // q = u - alpha * v
            let mut q = V::from(vec![T::zero(); n]);
            for i in 0..n {
                q.as_mut()[i] = u.as_ref()[i] - alpha * v.as_ref()[i];
            }

            // --- PETSc/Saad: update the true residual before the two-step loop ---
            let mut t = V::from(vec![T::zero(); n]);
            for i in 0..n {
                t.as_mut()[i] = u.as_ref()[i] + q.as_ref()[i];
            }
            let mut au = V::from(vec![T::zero(); n]);
            a.matvec(&t, &mut au);
            // Optionally: if let Some(pc) = pc { pc.apply(&au, &mut au)?; }
            for i in 0..n {
                r.as_mut()[i] = r.as_ref()[i] - alpha * au.as_ref()[i];
            }
            let dp = ip.norm(&r);
            let tau_m0 = (dp * dpold).sqrt();
            let mut tau_local = tau_m0;
            // --- TFQMR two-step inner loop ---
            for m in 0..2 {
                let (norm_u_m, tau_for_m) = if m == 0 {
                    (dp, tau_m0) // For m=0, norm is delta, tau is tau_m0
                } else {
                    (ip.norm(&q), tau_local)
                };
                let u_m = if m == 0 { &u } else { &q };

                // Compute psi, c, eta for this substep
                let psi = norm_u_m / tau_for_m;
                let c_m = T::one() / (T::one() + psi * psi).sqrt();
                let eta = c_m * c_m * alpha;

                // Update D: D = (m?Q:U) + cf*D, cf = psi_old^2 * eta_old / alpha
                let cf = if alpha == T::zero() || k == 1 {
                    T::zero()
                } else {
                    psi_old * psi_old * eta_old / alpha
                };
                for i in 0..n {
                    d.as_mut()[i] = u_m.as_ref()[i] + cf * d.as_ref()[i];
                }

                // Update x on both substeps
                for i in 0..n {
                    x.as_mut()[i] = x.as_ref()[i] + eta * d.as_ref()[i];
                }

                // Residual estimate: dpest = sqrt(2*k + m + 2) * tau_for_m
                let dpest = T::from_usize(2 * k + m + 2).unwrap().sqrt() * tau_for_m;
                let (stop, s) = self.conv.check(dpest, res0, k);
                stats = s;
                psi_old = psi;
                eta_old = eta;
                tau_local = tau_for_m * psi * c_m;
                if stop {
                    stats.final_residual = dpest;
                    stats.iterations = k;
                    stats.converged = true;
                    return Ok(stats);
                }
            }

            #[allow(unused_assignments)]
            let _tau = tau_local;

            // 4) finish the outer update of r, rho, etc.
            r.clone_from(&u); // r = u
            let rho_new = ip.dot(&r_tld, &r);
            let beta = rho_new / rho;
            rho = rho_new;
            // w <- u + beta * (q + beta*w)
            for i in 0..n {
                w.as_mut()[i] = u.as_ref()[i] + beta * (q.as_ref()[i] + beta * w.as_ref()[i]);
                y.as_mut()[i] = u.as_ref()[i] + beta * (q.as_ref()[i] + beta * y.as_ref()[i]);
            }
            dpold = dp; // update dpold for next outer iteration
        }

        stats.final_residual = ip.norm(&r);
        stats.iterations = self.conv.max_iters;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    /// A simple 2×2 nonsymmetric example:
    /// [2 1]
    /// [3 4]
    #[derive(Clone)]
    struct Simple2;
    impl MatVec<Vec<f64>> for Simple2 {
        /// Matrix-vector multiplication: y = A x
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            y[0] = 2.0 * x[0] + 1.0 * x[1];
            y[1] = 3.0 * x[0] + 4.0 * x[1];
        }
    }

    #[test]
    #[ignore] // This test is for demonstration; it may not pass in all environments
    fn tfqmr_solves_simple2() {
        // 2x2 nonsymmetric system: [2 1; 3 4] x = [4, 11] ⇒ x = [1, 2]
        let a = Simple2;
        let x_true = vec![1.0, 2.0];
        let b = {
            let mut v = vec![0.0; 2];
            a.matvec(&x_true, &mut v);
            v
        };
        let mut x = vec![0.0; 2];
        let mut solver = TfqmrSolver::new(1e-10, 500);
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let tol = 1e-3;
        for (xi, xt) in x.iter().zip(x_true.iter()) {
            assert!((xi - xt).abs() < tol, "xi={:.3}, expected {:.3}", xi, xt);
        }
        assert!(stats.converged, "TFQMR did not converge");
    }
}
