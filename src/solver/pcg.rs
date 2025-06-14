//! Preconditioned Conjugate Gradient (PCG) per Saad §9.2

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;

pub struct PcgSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: Copy + num_traits::Float> PcgSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for PcgSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone + From<f64>,
{
    type Error = KError;
    type Scalar = T;

    fn solve(&mut self, a: &M, pc: Option<&dyn Preconditioner<M, V>>, b: &V, x: &mut V) -> Result<SolveStats<T>, KError> {
        let n = b.as_ref().len();
        let ip = ();
        let mut x_vec = x.as_ref().to_vec();
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(x_vec.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let mut z = V::from(vec![T::zero(); n]);
        if let Some(pc) = pc {
            pc.apply(&r, &mut z)?;
        } else {
            z.clone_from(&r);
        }
        let mut p = z.clone();
        let mut rz = ip.dot(&r, &z);
        let res0 = rz.abs().sqrt();
        let mut stats = SolveStats { iterations: 0, final_residual: res0, converged: false };
        for i in 0..self.conv.max_iters {
            let mut ap = V::from(vec![T::zero(); n]);
            a.matvec(&p, &mut ap);
            let alpha = rz / ip.dot(&p, &ap);
            for (xj, pj) in x_vec.iter_mut().zip(p.as_ref()) {
                *xj = *xj + alpha * *pj;
            }
            for (rj, apj) in r.as_mut().iter_mut().zip(ap.as_ref()) {
                *rj = *rj - alpha * *apj;
            }
            if let Some(pc) = pc {
                pc.apply(&r, &mut z)?;
            } else {
                z.clone_from(&r);
            }
            let rz_new = ip.dot(&r, &z);
            let res_norm = rz_new.abs().sqrt();
            let (stop, s) = self.conv.check(res_norm, res0, i+1);
            stats = s.clone();
            if stop && s.converged {
                *x = V::from(x_vec.clone());
                return Ok(stats);
            }
            let beta = rz_new / rz;
            let p_old = p.clone();
            for ((pj, zj), old_pj) in p.as_mut().iter_mut().zip(z.as_ref()).zip(p_old.as_ref()) {
                *pj = *zj + beta * *old_pj;
            }
            rz = rz_new;
        }
        *x = V::from(x_vec);
        Ok(stats)
    }
}
