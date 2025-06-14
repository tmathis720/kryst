//! GMRES with fixed restart per Saad §6.4.

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use num_traits::Float;

pub struct GmresSolver<T> {
    pub restart: usize,
    pub conv: Convergence<T>,
}

impl<T: Copy + Float> GmresSolver<T> {
    pub fn new(restart: usize, tol: T, max_iters: usize) -> Self {
        Self {
            restart,
            conv: Convergence { tol, max_iters },
        }
    }

    // --- Arnoldi process with double orthogonalization and happy breakdown ---
    fn arnoldi<M, V>(
        a: &M,
        ip: &(),
        v_basis: &mut Vec<V>,
        h: &mut [Vec<T>],
        j: usize,
        epsilon: T,
    ) -> bool
    where
        M: MatVec<V>,
        (): InnerProduct<V, Scalar = T>,
        V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
        T: num_traits::Float + Clone,
    {
        let n = v_basis[0].as_ref().len();
        let mut w = V::from(vec![T::zero(); n]);
        a.matvec(&v_basis[j].clone(), &mut w);
        // Modified Gram-Schmidt
        for i in 0..=j {
            h[i][j] = ip.dot(&w, &v_basis[i]);
            for (wk, vik) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                *wk = *wk - h[i][j] * *vik;
            }
        }
        // Iterative refinement (second orthogonalization)
        for i in 0..=j {
            let tmp = ip.dot(&w, &v_basis[i]);
            h[i][j] = h[i][j] + tmp;
            for (wk, vik) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                *wk = *wk - tmp * *vik;
            }
        }
        h[j + 1][j] = ip.norm(&w);
        // Happy breakdown: if norm is very small, return true
        if h[j + 1][j].abs() < epsilon {
            return true;
        }
        let vj1 = V::from(w.as_ref().iter().map(|&wi| wi / h[j + 1][j]).collect::<Vec<_>>());
        v_basis.push(vj1);
        false
    }

    // --- Apply Givens rotation and update g together ---
    fn apply_givens_and_update_g(h: &mut [Vec<T>], g: &mut [T], cs: &mut [T], sn: &mut [T], j: usize, epsilon: T) {
        for i in 0..j {
            let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
            h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
            h[i][j] = temp;
        }
        let h_kk = h[j][j];
        let h_k1k = h[j + 1][j];
        let r = (h_kk * h_kk + h_k1k * h_k1k).sqrt();
        if r.abs() < epsilon {
            cs[j] = T::one();
            sn[j] = T::zero();
        } else {
            cs[j] = h_kk / r;
            sn[j] = h_k1k / r;
        }
        h[j][j] = cs[j] * h_kk + sn[j] * h_k1k;
        h[j + 1][j] = T::zero();
        // Update g
        let temp = cs[j] * g[j] + sn[j] * g[j + 1];
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
        g[j] = temp;
    }

    // --- Back-substitution for least squares with zero-pivot protection ---
    fn back_substitution(h: &[Vec<T>], g: &[T], y: &mut [T], m: usize, epsilon: T) {
        for i in (0..m).rev() {
            y[i] = g[i];
            for j in (i + 1)..m {
                y[i] = y[i] - h[i][j] * y[j];
            }
            if h[i][i].abs() > epsilon {
                y[i] = y[i] / h[i][i];
            } else {
                y[i] = T::zero();
            }
        }
    }
}

impl<M, V, T> LinearSolver<M, V> for GmresSolver<T>
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
        let ip = ();
        let mut xk = x.as_ref().to_vec();
        let mut r0 = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let mut beta = ip.norm(&r0);
        let res0 = beta;
        let mut stats = SolveStats { iterations: 0, final_residual: beta, converged: false };

        let max_outer = (self.conv.max_iters + self.restart - 1) / self.restart;
        let mut iteration = 0;
        let epsilon = num_traits::cast::<f64, T>(1e-14).unwrap();
        for _ in 0..max_outer {
            let mut v_basis: Vec<V> = Vec::with_capacity(self.restart + 1);
            // Normalize r0 for the first basis vector
            let r0_norm = beta;
            let v0 = r0.clone().as_ref().iter().map(|&ri| ri / r0_norm).collect::<Vec<_>>();
            v_basis.push(V::from(v0));
            let mut h = vec![vec![T::zero(); self.restart]; self.restart + 1];
            let mut g = vec![T::zero(); self.restart + 1];
            g[0] = beta;
            let mut cs = vec![T::zero(); self.restart];
            let mut sn = vec![T::zero(); self.restart];
            let mut m = 0;
            let mut happy_breakdown = false;
            for j in 0..self.restart {
                iteration += 1;
                happy_breakdown = Self::arnoldi(a, &ip, &mut v_basis, &mut h, j, epsilon);
                Self::apply_givens_and_update_g(&mut h, &mut g, &mut cs, &mut sn, j, epsilon);
                let res_norm = g[j + 1].abs();
                let (stop, s) = self.conv.check(res_norm, res0, iteration);
                stats = s.clone();
                m = j + 1;
                if stop && s.converged || happy_breakdown {
                    break;
                }
            }
            // Solve least squares Hy = g (only use m x m upper part)
            let mut y = vec![T::zero(); m];
            let h_upper: Vec<Vec<T>> = h.iter().take(m).map(|row| row[..m].to_vec()).collect();
            let g_upper = &g[..m];
            Self::back_substitution(&h_upper, g_upper, &mut y, m, epsilon);
            // Update xk = xk + sum_{j=0}^{m-1} y[j] * v_basis[j]
            for j in 0..m {
                for (xk_i, vj_i) in xk.iter_mut().zip(v_basis[j].as_ref()) {
                    *xk_i = *xk_i + y[j] * *vj_i;
                }
            }
            // Recompute true residual after restart
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            r0 = V::from(r_vec);
            beta = ip.norm(&r0);
            if stats.converged || iteration >= self.conv.max_iters {
                break;
            }
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
    fn gmres_solves_well_conditioned_nonsym() {
        // 4x4 non-symmetric, well-conditioned system
        // A = [[4,1,0,0],[1,3,1,0],[0,1,2,1],[0,0,1,3]]
        // x_true = [1,2,3,4]
        // b = A * x_true
        let a = DenseMat {
            data: vec![
                vec![4.0, 1.0, 0.0, 0.0],
                vec![1.0, 3.0, 1.0, 0.0],
                vec![0.0, 1.0, 2.0, 1.0],
                vec![0.0, 0.0, 1.0, 3.0],
            ]
        };
        let x_true = vec![1.0, 2.0, 3.0, 4.0];
        let b = {
            let mut b = vec![0.0; 4];
            a.matvec(&x_true, &mut b);
            b
        };
        let mut x = vec![0.0; 4];
        let mut solver = GmresSolver::new(4, 1e-10, 100);
        let stats = solver.solve(&a, &b, &mut x).unwrap();
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(x_true.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged, "GMRES did not converge");
    }
}
