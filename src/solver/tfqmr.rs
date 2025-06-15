//! TFQMR solver (Saad §7.4)

use crate::solver::LinearSolver;
use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, InnerProduct};
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use num_traits::Float;

/// TFQMR (Transpose-Free Quasi-Minimal Residual) solver
pub struct TfqmrSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: Float> TfqmrSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for TfqmrSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone,
    T: Float + From<f64>,
{
    type Error = KError;
    type Scalar = T;

    fn solve(&mut self,
             a: &M,
             pc: Option<&dyn Preconditioner<M, V>>,
             b: &V,
             x: &mut V) -> Result<SolveStats<T>, KError> {
        let n = b.as_ref().len();
        let ip = ();

        // x0 = 0
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
        let mut alpha = T::zero();
        let mut theta = T::zero();
        let mut eta = T::zero();
        let mut c = T::one();
        let _d_scalar = T::zero();

        // vectors
        #[allow(unused_assignments)]
        let mut v = V::from(vec![T::zero(); n]);
        let _p = V::from(vec![T::zero(); n]);
        let mut w = r.clone();     // w = r0
        let mut y = r.clone();     // y = r0
        let mut u = V::from(vec![T::zero(); n]);
        let mut d = V::from(vec![T::zero(); n]);

        let res0 = ip.norm(&r);
        let mut stats = SolveStats { iterations: 0, final_residual: res0, converged: false };

        for k in 1..=self.conv.max_iters {
            // v = A * y
            let mut v_tmp = V::from(vec![T::zero(); n]);
            a.matvec(&y, &mut v_tmp);
            v = v_tmp;
            if let Some(pc) = pc {
                let mut z = V::from(vec![T::zero(); n]);
                pc.apply(&v, &mut z)?;
                v = z;
            }

            // alpha = rho / <r_tld, v>
            let sigma = ip.dot(&r_tld, &v);
            if sigma == T::zero() {
                break; // breakdown
            }
            alpha = rho / sigma;

            // u = r - alpha * v
            for (ui, (ri, vi)) in u.as_mut().iter_mut().zip(r.as_ref().iter().zip(v.as_ref())) {
                *ui = *ri - alpha * *vi;
            }

            // w = u + theta^2 * eta / alpha * w
            if k == 1 {
                w = u.clone();
            } else {
                let coef = theta * theta * eta / alpha;
                let w_old = w.as_ref().to_vec();
                for (wi, (ui, wi_old)) in w.as_mut().iter_mut().zip(u.as_ref().iter().zip(w_old.iter())) {
                    *wi = *ui + coef * *wi_old;
                }
            }

            // d = u + (theta^2 * eta / alpha) * d
            if k == 1 {
                d = u.clone();
            } else {
                let coef = theta * theta * eta / alpha;
                let d_old = d.as_ref().to_vec();
                for (di, (ui, di_old)) in d.as_mut().iter_mut().zip(u.as_ref().iter().zip(d_old.iter())) {
                    *di = *ui + coef * *di_old;
                }
            }

            // x = x + alpha * d
            for (xi, di) in x.as_mut().iter_mut().zip(d.as_ref()) {
                *xi = *xi + alpha * *di;
            }

            // y = y - alpha * v
            for (yi, vi) in y.as_mut().iter_mut().zip(v.as_ref()) {
                *yi = *yi - alpha * *vi;
            }

            // update variables for next iteration
            let rho_new = ip.dot(&u, &r_tld);
            theta = ip.norm(&u) / alpha;
            let c_new = T::one() / ( (T::one() + theta * theta).sqrt() );
            eta = c * c_new * alpha;
            c = c_new;
            rho = rho_new;
            r = u.clone();

            // compute residual norm estimate
            let res_norm = if k % 2 == 0 {
                ip.norm(&r) // accurate on even steps
            } else {
                // quasi-minimal residual estimate: ||w|| * theta
                ip.norm(&w) * theta
            };
            let (stop, s) = self.conv.check(res_norm, res0, k);
            stats = s;
            if stop {
                stats.final_residual = res_norm;
                stats.iterations = k;
                return Ok(stats);
            }
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
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            y[0] = 2.0 * x[0] + 1.0 * x[1];
            y[1] = 3.0 * x[0] + 4.0 * x[1];
        }
    }

    #[test]
    fn tfqmr_solves_simple2() {
        let a = Simple2;
        let x_true = vec![1.0, 2.0];
        let b = {
            let mut v = vec![0.0; 2];
            a.matvec(&x_true, &mut v);
            v
        };
        let mut x = vec![0.0; 2];
        let mut solver = TfqmrSolver::new(1e-10, 50);
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let tol = 1e-3;
        for (xi, xt) in x.iter().zip(x_true.iter()) {
            assert!((xi - xt).abs() < tol, "xi={:.3}, expected {:.3}", xi, xt);
        }
        assert!(stats.converged, "TFQMR did not converge");
    }
}
