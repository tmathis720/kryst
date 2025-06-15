//! Conjugate Gradient (unpreconditioned) per Saad §6.1.

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;

pub enum CgNormType { Preconditioned, Unpreconditioned, Natural, None }

pub struct CgSolver<T> {
    pub conv: Convergence<T>,
    pub norm_type: CgNormType,
    pub single_reduction: bool,
    pub radius: Option<T>,
    pub obj_target: Option<T>,
    pub monitor: Option<Box<dyn FnMut(usize, T)>>,
    pub residual_history: Vec<T>,
}

impl<T: Copy + num_traits::Float> CgSolver<T> {
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
        let dp = match self.norm_type {
            CgNormType::Preconditioned => ip.dot(&r, &r),
            CgNormType::Unpreconditioned => ip.dot(&r, &r),
            CgNormType::Natural => ip.dot(&r, &p),
            CgNormType::None => T::zero(),
        };
        if let Some(ref mut monitor) = self.monitor {
            monitor(0, dp.sqrt());
        }
        self.residual_history.push(dp.sqrt());
        for i in 1..=self.conv.max_iters {
            let mut ap = V::from(vec![T::zero(); n]);
            a.matvec(&p.clone(), &mut ap);
            let p_dot_ap = if self.single_reduction {
                // Fused dot product: p^T A p (before r update)
                let mut p_dot_ap = T::zero();
                for i in 0..n {
                    p_dot_ap = p_dot_ap + p.as_ref()[i] * ap.as_ref()[i];
                }
                p_dot_ap
            } else {
                ip.dot(&p, &ap)
            };
            let res_norm;
            // Indefinite-matrix detection
            if p_dot_ap <= T::zero() {
                res_norm = ip.dot(&r, &r).sqrt();
                stats.iterations = i;
                stats.final_residual = res_norm;
                stats.converged = false;
                return Err(KError::IndefiniteMatrix);
            }
            let alpha = rsq / p_dot_ap;
            // Trust-region (Steihaug–Toint) logic
            if let Some(radius) = self.radius {
                let p_norm = ip.dot(&p, &p).sqrt();
                let x_norm = ip.dot(&V::from(x_vec.clone()), &V::from(x_vec.clone())).sqrt();
                if x_norm + alpha.abs() * p_norm > radius {
                    // Step would exceed trust region: take as much as possible
                    let max_step = (radius - x_norm) / p_norm;
                    for (xj, pj) in x_vec.iter_mut().zip(p.as_ref()) {
                        *xj = *xj + max_step * *pj;
                    }
                    *x = V::from(x_vec.clone());
                    let res_norm_tr = ip.dot(&r, &r).sqrt();
                    stats.iterations = i;
                    stats.final_residual = res_norm_tr;
                    stats.converged = false;
                    return Ok(stats);
                }
            }
            for (xj, pj) in x_vec.iter_mut().zip(p.as_ref()) {
                *xj = *xj + alpha * *pj;
            }
            for (rj, apj) in r.as_mut().iter_mut().zip(ap.as_ref()) {
                *rj = *rj - alpha * *apj;
            }
            let rsq_new = ip.dot(&r, &r);
            res_norm = match self.norm_type {
                CgNormType::Preconditioned => rsq_new.sqrt(),
                CgNormType::Unpreconditioned => rsq_new.sqrt(),
                CgNormType::Natural => ip.dot(&r, &p).abs().sqrt(),
                CgNormType::None => T::zero(),
            };
            // Objective function tracking
            if let Some(obj_target) = self.obj_target {
                let obj = {
                    let mut ax = V::from(vec![T::zero(); n]);
                    a.matvec(&V::from(x_vec.clone()), &mut ax);
                    let x_dot_ax = ip.dot(&V::from(x_vec.clone()), &ax);
                    let x_dot_b = ip.dot(&V::from(x_vec.clone()), b);
                    num_traits::cast::<f64, T>(0.5).unwrap() * x_dot_ax - x_dot_b
                };
                let res_norm_obj = match self.norm_type {
                    CgNormType::Preconditioned => ip.dot(&r, &r).sqrt(),
                    CgNormType::Unpreconditioned => rsq_new.sqrt(),
                    CgNormType::Natural => ip.dot(&r, &p).abs().sqrt(),
                    CgNormType::None => T::zero(),
                };
                if obj <= obj_target {
                    *x = V::from(x_vec.clone());
                    stats.iterations = i;
                    stats.final_residual = res_norm_obj;
                    stats.converged = true;
                    return Ok(stats);
                }
            }
            // Indefinite-beta detection
            if rsq_new / rsq < T::zero() {
                stats.iterations = i;
                stats.final_residual = res_norm;
                stats.converged = false;
                return Err(KError::IndefinitePreconditioner);
            }
            if let Some(ref mut monitor) = self.monitor {
                monitor(i, res_norm);
            }
            self.residual_history.push(res_norm);
            let (stop, s) = self.conv.check(res_norm, res0, i);
            stats = s.clone();
            if stop && s.converged {
                *x = V::from(x_vec.clone());
                return Ok(stats);
            }
            let beta = rsq_new / rsq;
            for (pj, rj) in p.as_mut().iter_mut().zip(r.as_ref()) {
                *pj = *rj + beta * *pj;
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

    #[test]
    fn cg_single_reduction_equivalence() {
        // SPD system: [[4,1],[1,3]] x = [1,2]
        let a = DenseMat { data: vec![vec![4.0, 1.0], vec![1.0, 3.0]] };
        let b = vec![1.0, 2.0];
        let mut x_std = vec![0.0, 0.0];
        let mut x_single = vec![0.0, 0.0];
        let mut solver_std = CgSolver::new(1e-10, 20);
        let mut solver_single = CgSolver::new(1e-10, 20).with_single_reduction(true);
        let _stats_std = solver_std.solve(&a, None, &b, &mut x_std).unwrap();
        let stats_single = solver_single.solve(&a, None, &b, &mut x_single).unwrap();
        let tol = 1e-8;
        for (xi, xj) in x_std.iter().zip(x_single.iter()) {
            assert!((xi - xj).abs() < tol, "single-reduction and standard CG differ: {} vs {}", xi, xj);
        }
        assert!(stats_single.converged, "Single-reduction CG did not converge");
        // Also check against expected solution
        let expected = vec![0.09090909090909091, 0.6363636363636364];
        for (xi, ei) in x_single.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
    }

    #[test]
    fn cg_single_reduction_spd3() {
        // SPD system: [[4,1,0],[1,3,1],[0,1,2]] x = [1,2,3]
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
        let mut x_std = vec![0.0; 3];
        let mut x_single = vec![0.0; 3];
        let mut solver_std = CgSolver::new(1e-10, 100);
        let mut solver_single = CgSolver::new(1e-10, 100).with_single_reduction(true);
        let _stats_std = solver_std.solve(&a, None, &b, &mut x_std).unwrap();
        let stats_single = solver_single.solve(&a, None, &b, &mut x_single).unwrap();
        let tol = 1e-8;
        for (xi, xj) in x_std.iter().zip(x_single.iter()) {
            assert!((xi - xj).abs() < tol, "single-reduction and standard CG differ: {} vs {}", xi, xj);
        }
        let mut r_final = vec![0.0; 3];
        a.matvec(&x_single, &mut r_final);
        for i in 0..3 {
            r_final[i] = b[i] - r_final[i];
        }
        let res_norm = r_final.iter().map(|&ri| ri*ri).sum::<f64>().sqrt();
        assert!(res_norm <= tol, "final residual = {:.6}, tol = {:.6}", res_norm, tol);
        assert!(stats_single.converged, "Single-reduction CG did not converge");
    }
}
