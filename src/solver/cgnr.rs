//! CGNR/CGNE solvers (Saad Ch 8.3)

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::SolveStats;
use crate::error::KError;

use crate::utils::convergence::Convergence;

pub struct CgnrSolver<T> {
    pub conv: Convergence<T>,
}

pub struct CgneSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float> CgnrSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<T: num_traits::Float> CgneSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for CgnrSolver<T>
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
        let mut xk = x.as_ref().to_vec();
        let ip = ();
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = b.as_ref().iter().zip(tmp.as_ref()).map(|(&bi, &axi)| bi - axi).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let mut z = V::from(vec![T::zero(); n]);
        a.matvec(&r, &mut z); // z = A^T r (for CGNR, A^T = A^T)
        let mut p = z.clone();
        let mut rz = ip.dot(&z, &z);
        let res0 = ip.norm(&r);
        let mut stats = SolveStats { iterations: 0, final_residual: res0, converged: false };

        for i in 1..=self.conv.max_iters {
            let mut ap = V::from(vec![T::zero(); n]);
            a.matvec(&p, &mut ap); // ap = A p
            let mut at_ap = V::from(vec![T::zero(); n]);
            a.matvec(&ap, &mut at_ap); // at_ap = A^T (A p)
            let alpha = rz / ip.dot(&at_ap, &at_ap);
            for (xj, pj) in xk.iter_mut().zip(p.as_ref()) {
                *xj = *xj + alpha * *pj;
            }
            for (rj, apj) in r.as_mut().iter_mut().zip(ap.as_ref()) {
                *rj = *rj - alpha * *apj;
            }
            a.matvec(&r, &mut z); // z = A^T r
            let rz_new = ip.dot(&z, &z);
            let res_norm = ip.norm(&r);
            let (stop, s) = self.conv.check(res_norm, res0, i);
            stats = s.clone();
            if stop && s.converged {
                *x = V::from(xk.clone());
                return Ok(stats);
            }
            let beta = rz_new / rz;
            let p_old = p.clone();
            for ((pj, zj), old_pj) in p.as_mut().iter_mut().zip(z.as_ref()).zip(p_old.as_ref()) {
                *pj = *zj + beta * *old_pj;
            }
            rz = rz_new;
        }
        *x = V::from(xk);
        Ok(stats)
    }
}

impl<M, V, T> LinearSolver<M, V> for CgneSolver<T>
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
        let mut xk = x.as_ref().to_vec();
        let ip = ();
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = b.as_ref().iter().zip(tmp.as_ref()).map(|(&bi, &axi)| bi - axi).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let mut z = V::from(vec![T::zero(); n]);
        a.matvec(&r, &mut z); // z = A^T r (for CGNE, A^T = A^T)
        let mut p = z.clone();
        let mut rz = ip.dot(&z, &z);
        let res0 = ip.norm(&r);
        let mut stats = SolveStats { iterations: 0, final_residual: res0, converged: false };

        for i in 1..=self.conv.max_iters {
            let mut at_p = V::from(vec![T::zero(); n]);
            a.matvec(&p, &mut at_p); // at_p = A p
            let mut ap = V::from(vec![T::zero(); n]);
            a.matvec(&at_p, &mut ap); // ap = A^T (A p)
            let alpha = rz / ip.dot(&ap, &ap);
            for (xj, pj) in xk.iter_mut().zip(p.as_ref()) {
                *xj = *xj + alpha * *pj;
            }
            for (rj, at_pj) in r.as_mut().iter_mut().zip(at_p.as_ref()) {
                *rj = *rj - alpha * *at_pj;
            }
            a.matvec(&r, &mut z); // z = A^T r
            let rz_new = ip.dot(&z, &z);
            let res_norm = ip.norm(&r);
            let (stop, s) = self.conv.check(res_norm, res0, i);
            stats = s.clone();
            if stop && s.converged {
                *x = V::from(xk.clone());
                return Ok(stats);
            }
            let beta = rz_new / rz;
            let p_old = p.clone();
            for ((pj, zj), old_pj) in p.as_mut().iter_mut().zip(z.as_ref()).zip(p_old.as_ref()) {
                *pj = *zj + beta * *old_pj;
            }
            rz = rz_new;
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
    fn cgnr_solves_simple_least_squares() {
        // Overdetermined system: minimize ||Ax - b||
        // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 3]
        // Least squares solution: x = [1, 2]
        let a = DenseMat { data: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]] };
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0, 0.0];
        let mut solver = CgnrSolver::new(1e-10, 50);
        let stats = solver.solve(&a, &b, &mut x).unwrap();
        let expected = vec![1.0, 2.0];
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged, "CGNR did not converge");
    }

    #[test]
    fn cgne_solves_simple_least_squares() {
        // Same system as above
        let a = DenseMat { data: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]] };
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0, 0.0];
        let mut solver = CgneSolver::new(1e-10, 50);
        let stats = solver.solve(&a, &b, &mut x).unwrap();
        let expected = vec![1.0, 2.0];
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged, "CGNE did not converge");
    }
}
