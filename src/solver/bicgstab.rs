//! BiCGStab solver (Saad §7.1)

use crate::core::traits::{InnerProduct, MatVec};
use crate::error::KError;
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};

pub struct BiCgStabSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float> BiCgStabSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for BiCgStabSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone + From<f64>,
{
    type Error = KError;
    type Scalar = T;

    fn solve(&mut self, a: &M, pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>, b: &V, x: &mut V) -> Result<SolveStats<T>, KError> {
        let _ = pc; // BiCGStab does not use preconditioner (yet)
        let n = b.as_ref().len();
        let ip = ();
        let mut xk = x.as_ref().to_vec();
        // r0 = b - A x0
        let mut tmp = V::from(vec![T::zero(); n]);
        a.matvec(&V::from(xk.clone()), &mut tmp);
        let mut r = V::from(tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>());
        let r_hat = r.clone(); // shadow residual
        let mut rho_prev = T::one();
        let mut alpha = T::one();
        let mut omega_prev = T::one();
        let mut v = V::from(vec![T::zero(); n]);
        let mut p = r.clone(); // Properly initialize p = r
        let res0 = ip.norm(&r);
        let mut stats = SolveStats { iterations: 0, final_residual: res0, converged: false };
        if res0 <= self.conv.tol {
            *x = V::from(xk);
            stats.converged = true;
            return Ok(stats);
        }
        for i in 1..=self.conv.max_iters {
            let rho = ip.dot(&r_hat, &r);
            if rho.abs() < T::epsilon() {
                break; // breakdown
            }
            let beta = if i == 1 {
                T::zero()
            } else {
                (rho / rho_prev) * (alpha / omega_prev)
            };
            // p = r + beta * (p - omega_prev * v)
            for ((p_j, r_j), v_j) in p.as_mut().iter_mut().zip(r.as_ref()).zip(v.as_ref()) {
                *p_j = *r_j + beta * (*p_j - omega_prev * *v_j);
            }
            // v = A p
            let mut v_tmp = V::from(vec![T::zero(); n]);
            a.matvec(&p.clone(), &mut v_tmp);
            v = v_tmp;
            let alpha_num = rho;
            let alpha_den = ip.dot(&r_hat, &v);
            if alpha_den.abs() < T::epsilon() {
                break; // breakdown
            }
            alpha = alpha_num / alpha_den;
            // s = r - alpha * v
            let s = V::from(r.as_ref().iter().zip(v.as_ref()).map(|(&rj, &vj)| rj - alpha * vj).collect::<Vec<_>>());
            let s_norm = ip.norm(&s);
            if s_norm <= self.conv.tol {
                for (xj, pj) in xk.iter_mut().zip(p.as_ref()) {
                    *xj = *xj + alpha * *pj;
                }
                *x = V::from(xk);
                stats = SolveStats { iterations: i, final_residual: s_norm, converged: true };
                return Ok(stats);
            }
            // t = A s
            let mut t = V::from(vec![T::zero(); n]);
            a.matvec(&s, &mut t);
            let omega_num = ip.dot(&t, &s);
            let omega_den = ip.dot(&t, &t);
            if omega_den.abs() < T::epsilon() {
                break; // breakdown
            }
            let omega = omega_num / omega_den;
            // x = x + alpha * p + omega * s
            for ((xj, pj), sj) in xk.iter_mut().zip(p.as_ref()).zip(s.as_ref()) {
                *xj = *xj + alpha * *pj + omega * *sj;
            }
            // r = s - omega * t (fix: do not update r in-place from previous r)
            r = V::from(s.as_ref().iter().zip(t.as_ref()).map(|(&sj, &tj)| sj - omega * tj).collect::<Vec<_>>());
            let r_norm = ip.norm(&r);
            stats = SolveStats { iterations: i, final_residual: r_norm, converged: r_norm <= self.conv.tol };
            if r_norm <= self.conv.tol {
                *x = V::from(xk);
                return Ok(stats);
            }
            if omega.abs() < T::epsilon() {
                break; // breakdown
            }
            rho_prev = rho;
            omega_prev = omega;
        }
        *x = V::from(xk);
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
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        eprintln!("BiCGStab stats: {{ converged: {}, iters: {}, final_res: {:e} }}", stats.converged, stats.iterations, stats.final_residual);
        // Compare to true solution
        let x_true = vec![1.0, 2.0, 3.0];
        for i in 0..3 {
            assert_abs_diff_eq!(x[i], x_true[i], epsilon = 1e-8);
        }
        assert!(stats.converged, "BiCGStab did not converge: stats = {:?}", stats);
    }
}
