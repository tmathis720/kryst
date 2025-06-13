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

    fn solve(&mut self, a: &M, b: &V, x: &mut V) -> Result<SolveStats<T>, KError> {
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
