//! Preconditioned Conjugate Gradient (PCG) per Saad §9.2

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;

pub enum CgNormType { Preconditioned, Unpreconditioned, Natural, None }

pub struct PcgSolver<T> {
    pub conv: Convergence<T>,
    pub norm_type: CgNormType,
    pub single_reduction: bool,
    pub radius: Option<T>,
    pub obj_target: Option<T>,
    pub monitor: Option<Box<dyn FnMut(usize, T)>>,
    pub residual_history: Vec<T>,
}

impl<T: Copy + num_traits::Float> PcgSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self {
            conv: Convergence { tol, max_iters },
            norm_type: CgNormType::Unpreconditioned,
            single_reduction: false,
            radius: None,
            obj_target: None,
            monitor: None,
            residual_history: Vec::new(),
        }
    }
    pub fn with_norm(mut self, norm_type: CgNormType) -> Self {
        self.norm_type = norm_type;
        self
    }
    pub fn with_single_reduction(mut self, flag: bool) -> Self {
        self.single_reduction = flag;
        self
    }
    pub fn with_radius(mut self, radius: T) -> Self {
        self.radius = Some(radius);
        self
    }
    pub fn with_obj_target(mut self, obj: T) -> Self {
        self.obj_target = Some(obj);
        self
    }
    pub fn with_monitor<F>(mut self, f: F) -> Self
    where F: FnMut(usize, T) + 'static {
        self.monitor = Some(Box::new(f));
        self
    }
    pub fn clear_history(&mut self) {
        self.residual_history.clear();
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
        let dp = match self.norm_type {
            CgNormType::Preconditioned => ip.dot(&z, &z),
            CgNormType::Unpreconditioned => ip.dot(&r, &r),
            CgNormType::Natural => ip.dot(&r, &z),
            CgNormType::None => T::zero(),
        };
        if let Some(ref mut monitor) = self.monitor {
            monitor(0, dp.sqrt());
        }
        self.residual_history.push(dp.sqrt());
        for i in 0..self.conv.max_iters {
            let mut ap = V::from(vec![T::zero(); n]);
            a.matvec(&p, &mut ap);
            let p_dot_ap = if self.single_reduction {
                // Fused dot product: p^T A p (before r/z update)
                let mut p_dot_ap = T::zero();
                for i in 0..n {
                    p_dot_ap = p_dot_ap + p.as_ref()[i] * ap.as_ref()[i];
                }
                p_dot_ap
            } else {
                ip.dot(&p, &ap)
            };
            // Indefinite-matrix detection
            if p_dot_ap <= T::zero() {
                stats.iterations = i + 1;
                stats.final_residual = match self.norm_type {
                    CgNormType::Preconditioned => ip.dot(&z, &z).sqrt(),
                    CgNormType::Unpreconditioned => ip.dot(&r, &r).sqrt(),
                    CgNormType::Natural => ip.dot(&r, &z).abs().sqrt(),
                    CgNormType::None => T::zero(),
                };
                stats.converged = false;
                return Err(KError::IndefiniteMatrix);
            }
            let alpha = rz / p_dot_ap;
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
            let res_norm = match self.norm_type {
                CgNormType::Preconditioned => ip.dot(&z, &z).sqrt(),
                CgNormType::Unpreconditioned => ip.dot(&r, &r).sqrt(),
                CgNormType::Natural => ip.dot(&r, &z).abs().sqrt(),
                CgNormType::None => T::zero(),
            };
            if let Some(ref mut monitor) = self.monitor {
                monitor(i+1, res_norm);
            }
            self.residual_history.push(res_norm);
            let (stop, s) = self.conv.check(res_norm, res0, i+1);
            stats = s.clone();
            if stop && s.converged {
                *x = V::from(x_vec.clone());
                return Ok(stats);
            }
            let beta = rz_new / rz;
            // Indefinite-preconditioner detection
            if beta < T::zero() {
                stats.iterations = i + 1;
                stats.final_residual = res_norm;
                stats.converged = false;
                return Err(KError::IndefinitePreconditioner);
            }
            for (pj, zj) in p.as_mut().iter_mut().zip(z.as_ref()) {
                *pj = *zj + beta * *pj;
            }
            rz = rz_new;
        }
        *x = V::from(x_vec);
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;
    use crate::preconditioner::Preconditioner;

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
    struct IdentityPC;
    impl Preconditioner<DenseMat, Vec<f64>> for IdentityPC {
        fn apply(&self, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), crate::error::KError> {
            z.copy_from_slice(r);
            Ok(())
        }
    }

    #[test]
    fn pcg_single_reduction_equivalence() {
        // SPD system: [[4,1],[1,3]] x = [1,2]
        let a = DenseMat { data: vec![vec![4.0, 1.0], vec![1.0, 3.0]] };
        let b = vec![1.0, 2.0];
        let mut x_std = vec![0.0, 0.0];
        let mut x_single = vec![0.0, 0.0];
        let mut solver_std = PcgSolver::new(1e-10, 20);
        let mut solver_single = PcgSolver::new(1e-10, 20).with_single_reduction(true);
        let pc = IdentityPC;
        let _stats_std = solver_std.solve(&a, Some(&pc), &b, &mut x_std).unwrap();
        let stats_single = solver_single.solve(&a, Some(&pc), &b, &mut x_single).unwrap();
        let tol = 1e-8;
        for (xi, xj) in x_std.iter().zip(x_single.iter()) {
            assert!((xi - xj).abs() < tol, "single-reduction and standard PCG differ: {} vs {}", xi, xj);
        }
        assert!(stats_single.converged, "Single-reduction PCG did not converge");
        // Also check against expected solution
        let expected = vec![0.09090909090909091, 0.6363636363636364];
        for (xi, ei) in x_single.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
    }
}
