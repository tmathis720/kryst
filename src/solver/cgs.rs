//! CGS solver (Saad §7.2)

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;

pub struct CgsSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float> CgsSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for CgsSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone + From<f64> + std::fmt::Debug + std::ops::AddAssign,
{
    type Error = KError;
    type Scalar = T;

    fn solve(&mut self, a: &M, pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>, b: &V, x: &mut V) -> Result<SolveStats<T>, KError> {
        let _ = pc; // CGS does not use preconditioner (yet)
        let n = b.as_ref().len();
        let mut xk = x.as_ref().to_vec();
        let ip = ();
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = b.as_ref().iter().zip(tmp.as_ref()).map(|(&bi, &axi)| bi - axi).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let r_tld = r.clone();
        let mut p = r.clone();
        let mut q = V::from(vec![T::zero(); n]);
        let mut u = V::from(vec![T::zero(); n]);
        let mut v = V::from(vec![T::zero(); n]);
        let mut rho = ip.dot(&r_tld, &r);
        let mut rho_old = T::zero();
        let res0 = ip.norm(&r);
        let mut stats = SolveStats { iterations: 0, final_residual: res0, converged: false };
        for i in 1..=self.conv.max_iters {
            if rho.abs() < T::epsilon() {
                break; // breakdown
            }
            if i == 1 {
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
            v = v_tmp;
            let alpha = rho / ip.dot(&r_tld, &v);
            // q = u - alpha * v
            for (q_j, (u_j, v_j)) in q.as_mut().iter_mut().zip(u.as_ref().iter().zip(v.as_ref())) {
                *q_j = *u_j - alpha * *v_j;
            }
            // x = x + alpha * (u + q)
            for (xj, (u_j, q_j)) in xk.iter_mut().zip(u.as_ref().iter().zip(q.as_ref())) {
                *xj = *xj + alpha * (*u_j + *q_j);
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
            let res_norm = ip.norm(&r);
            // eprintln!("CGS iter {}: res_norm={:.3e}", i, res_norm);
            let (stop, s) = self.conv.check(res_norm, res0, i);
            stats = s.clone();
            if stop && s.converged {
                *x = V::from(xk.clone());
                return Ok(stats);
            }
            rho_old = rho;
            rho = ip.dot(&r_tld, &r);
        }
        *x = V::from(xk);
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
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let tol = 1e-6;
        for (xi, ei) in x.iter().zip(x_true.iter()) {
            assert!((xi - ei).abs() <= tol, "xi = {:.6}, expected = {:.6}", xi, ei);
        }
        assert!(stats.converged, "CGS did not converge");
    }
}
