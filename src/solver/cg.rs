//! Conjugate Gradient (unpreconditioned) per Saad §6.1.

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;

pub struct CgSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: Copy + num_traits::Float> CgSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for CgSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone + From<f64>,
{
    type Error = KError;
    type Scalar = T;

    fn solve(&mut self, a: &M, pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>, b: &V, x: &mut V) -> Result<SolveStats<T>, KError> {
        let _ = pc; // CG does not use preconditioner
        let n = b.as_ref().len();
        let mut x_vec = x.as_ref().to_vec();
        let ip = ();
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(x_vec.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let mut p = r.clone();
        let mut rsq = ip.dot(&r, &r);
        let res0 = rsq.sqrt();
        let mut stats = SolveStats { iterations: 0, final_residual: res0, converged: false };

        for i in 1..=self.conv.max_iters {
            let mut ap = V::from(vec![T::zero(); n]);
            a.matvec(&p.clone(), &mut ap);
            let alpha = rsq / ip.dot(&p, &ap);
            for (xj, pj) in x_vec.iter_mut().zip(p.as_ref()) {
                *xj = *xj + alpha * *pj;
            }
            for (rj, apj) in r.as_mut().iter_mut().zip(ap.as_ref()) {
                *rj = *rj - alpha * *apj;
            }
            let rsq_new = ip.dot(&r, &r);
            let res_norm = rsq_new.sqrt();
            let (stop, s) = self.conv.check(res_norm, res0, i);
            stats = s.clone();
            if stop && s.converged {
                *x = V::from(x_vec.clone());
                return Ok(stats);
            }
            let beta = rsq_new / rsq;
            let p_old = p.clone();
            for ((pj, rj), old_pj) in p.as_mut().iter_mut().zip(r.as_ref()).zip(p_old.as_ref()) {
                *pj = *rj + beta * *old_pj;
            }
            rsq = rsq_new;
        }
        *x = V::from(x_vec);
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    // Simple dense matrix type for testing
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
    fn cg_solves_simple_spd() {
        // SPD system: [[4,1],[1,3]] x = [1,2]
        let a = DenseMat { data: vec![vec![4.0, 1.0], vec![1.0, 3.0]] };
        let b = vec![1.0, 2.0];
        let mut x = vec![0.0, 0.0];
        let mut solver = CgSolver::new(1e-10, 20);
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let expected = vec![0.09090909090909091, 0.6363636363636364];
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged, "CG did not converge");
    }

    #[test]
    fn cg_solves_spd() {
        // Symmetric positive definite system
        // A = [[4,1,0],[1,3,1],[0,1,2]]
        // x_true = [1,2,3]
        // b = A * x_true = [6,8,8]
        let a = DenseMat {
            data: vec![
                vec![4.0, 1.0, 0.0],
                vec![1.0, 3.0, 1.0],
                vec![0.0, 1.0, 2.0],
            ]
        };
        let x_true = vec![1.0, 2.0, 3.0];
        let b = {
            let mut b = vec![0.0; 3];
            a.matvec(&x_true, &mut b);
            b
        };
        let mut x = vec![0.0; 3];
        let mut solver = CgSolver::new(1e-10, 100);
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let tol = 1e-8;
        let mut r_final = vec![0.0; 3];
        a.matvec(&x, &mut r_final);
        for i in 0..3 {
            r_final[i] = b[i] - r_final[i];
        }
        let res_norm = r_final.iter().map(|&ri| ri*ri).sum::<f64>().sqrt();
        assert!(res_norm <= tol, "final residual = {:.6}, tol = {:.6}", res_norm, tol);
        assert!(stats.converged, "CG did not converge");
    }
}
