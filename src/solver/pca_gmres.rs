//! Pipelined, Communication-Avoiding GMRES for Kryst
//! Initial implementation with block orthogonalization and pipelining skeleton

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Preconditioning modes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preconditioning { None, Left, Right }

/// Pipelined Communication-Avoiding GMRES solver
pub struct PcaGmresSolver<T> {
    /// Restart length (m)
    pub restart: usize,
    /// Pipeline depth (ℓ)
    pub pipeline_depth: usize,
    /// Block size for CA orthogonalization (s)
    pub block_size: usize,
    /// Drop tolerance for partial change-of-basis (τ)
    pub tau: Option<T>,
    /// Convergence criteria
    pub conv: Convergence<T>,
    /// Preconditioning mode
    pub preconditioning: Preconditioning,
}

impl<T: num_traits::Float + Send + Sync + From<f64> + std::ops::SubAssign + std::ops::MulAssign> PcaGmresSolver<T> {
    /// Create a new solver: restart=m, pipeline depth=ℓ, block size=s, tolerance tol, max_iters
    pub fn new(restart: usize, pipeline_depth: usize, block_size: usize, tol: T, max_iters: usize) -> Self {
        Self {
            restart,
            pipeline_depth,
            block_size,
            tau: None,
            conv: Convergence { tol, max_iters },
            preconditioning: Preconditioning::Left,
        }
    }

    /// Set preconditioning mode
    pub fn with_preconditioning(mut self, mode: Preconditioning) -> Self {
        self.preconditioning = mode;
        self
    }

    /// Set partial change-of-basis drop tolerance
    pub fn with_tau(mut self, tau: T) -> Self {
        self.tau = Some(tau);
        self
    }
}

impl<M, V, T> LinearSolver<M, V> for PcaGmresSolver<T>
where
    M: MatVec<V> + Sync,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone + Send + Sync,
    T: num_traits::Float + Send + Sync + From<f64> + Clone + std::ops::SubAssign + std::ops::MulAssign,
{
    type Error = KError;
    type Scalar = T;

    fn solve(&mut self,
             a: &M,
             pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>,
             b: &V,
             x: &mut V) -> Result<SolveStats<T>, KError> {
        let n = b.as_ref().len();
        let ip = ();
        // Initial guess xk = 0
        let mut xk = vec![T::zero(); n];
        // r0 = b - A xk
        let mut tmp = V::from(vec![T::zero(); n]);
        a.matvec(&V::from(xk.clone()), &mut tmp);
        let mut r0_vec = tmp.as_ref().iter().zip(b.as_ref())
            .map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
        let mut r0 = V::from(r0_vec);
        let mut beta = ip.norm(&r0);
        let res0 = beta;
        let mut stats = SolveStats { iterations: 0, final_residual: beta, converged: false };

        let mut iteration = 0;
        // Number of outer cycles
        let n_outer = (self.conv.max_iters + self.restart - 1) / self.restart;
        for _outer in 0..n_outer {
            // Build initial Arnoldi vector
            let mut v_basis: Vec<V> = Vec::with_capacity(self.restart + 1);
            // Normalize r0 into v0
            let v0 = r0.as_ref().iter().map(|&ri| ri / beta).collect::<Vec<_>>();
            v_basis.push(V::from(v0));

            // Hessenberg and Givens data
            let m = self.restart;
            let mut h = vec![vec![T::zero(); m]; m+1];
            let mut g = vec![T::zero(); m+1];
            g[0] = beta;
            let mut cs = vec![T::zero(); m];
            let mut sn = vec![T::zero(); m];

            // Iterate m steps (block at a time)
            let s = self.block_size;
            let mut j = 0;
            while j < m {
                let t = std::cmp::min(s, m - j);
                // 1) Generate block of t Krylov vectors
                #[cfg(feature = "rayon")]
                let mut v_block: Vec<V> = if pc.is_none() {
                    (0..t).into_par_iter().map(|k| {
                        let mut w = V::from(vec![T::zero(); n]);
                        a.matvec(&v_basis[j+k], &mut w);
                        w
                    }).collect()
                } else {
                    (0..t).map(|k| {
                        let mut w = V::from(vec![T::zero(); n]);
                        a.matvec(&v_basis[j+k], &mut w);
                        if let (Preconditioning::Right, Some(pc)) = (self.preconditioning, pc) {
                            let mut z = V::from(vec![T::zero(); n]);
                            pc.apply(&w, &mut z).unwrap();
                            w = z;
                        }
                        w
                    }).collect()
                };
                #[cfg(not(feature = "rayon"))]
                let mut v_block: Vec<V> = (0..t).map(|k| {
                    let mut w = V::from(vec![T::zero(); n]);
                    a.matvec(&v_basis[j+k], &mut w);
                    if let (Preconditioning::Right, Some(pc)) = (self.preconditioning, pc) {
                        let mut z = V::from(vec![T::zero(); n]);
                        pc.apply(&w, &mut z).unwrap();
                        w = z;
                    }
                    w
                }).collect();

                // 2) Block Classical Gram-Schmidt with overlapped reduction
                // Gather local inner-products into a temp array
                let mut local_dot = vec![T::zero(); (j+1)*t];
                for i in 0..=j {
                    for k in 0..t {
                        local_dot[i*t + k] = ip.dot(&v_basis[i], &v_block[k]);
                    }
                }
                // Kick off a non-blocking all-reduce on local_dot → global_dot
                #[cfg(feature = "mpi")]
                let global_dot = {
                    use mpi::traits::*;
                    use mpi::collective::SystemOperation;
                    // TODO: Pass a properly initialized communicator (e.g., MpiComm) into the solver.
                    // let world = ...;
                    // For now, this is a placeholder and will not work as-is:
                    let universe = mpi::initialize().unwrap();
                    let world = universe.world();
                    let mut req = world.immediate_all_reduce_into(&local_dot, &mut local_dot.clone(), &SystemOperation::sum());
                    // do the local subtractions while communication proceeds:
                    for i in 0..=j {
                        for k in 0..t {
                            let coeff = local_dot[i*t + k];
                            let qi = v_basis[i].as_ref();
                            v_block[k].as_mut().iter_mut()
                                .zip(qi)
                                .for_each(|(vk, &q)| *vk -= coeff * q);
                        }
                    }
                    let mut global_dot = vec![T::zero(); (j+1)*t];
                    req.wait_into(&mut global_dot);
                    global_dot
                };
                #[cfg(not(feature = "mpi"))]
                let global_dot = local_dot;
                // write back the fully reduced coefficients:
                for i in 0..=j {
                    for k in 0..t {
                        h[i][j+k] = global_dot[i*t + k];
                    }
                }
                // Intra-block orthogonalization
                for k in 0..t {
                    let vk = &mut v_block[k];
                    for i in 0..k {
                        let r_ij = ip.dot(&v_basis[j+i], &v_block[k]);
                        h[j+i][j+k] = r_ij;
                        let qji = v_basis[j+i].as_ref();
                        v_block[k].as_mut().iter_mut()
                          .zip(qji)
                          .for_each(|(vki, &qii)| *vki -= r_ij * qii);
                    }
                    // Normalize v_block[k]
                    let norm_vk = ip.norm(&v_block[k]);
                    h[j+k+1][j+k] = norm_vk;
                    let inv = T::one() / norm_vk;
                    v_block[k].as_mut().iter_mut().for_each(|vki| *vki *= inv);
                }

                // 3) Append new orthonormalized vectors to basis
                for k in 0..t {
                    v_basis.push(v_block[k].clone());
                }

                // 4) Apply Givens rotations to H and update g (pipeline step)
                for k in 0..t {
                    let col = j + k;
                    // apply previous rotations
                    for i in 0..col {
                        let temp = cs[i] * h[i][col] + sn[i] * h[i+1][col];
                        h[i+1][col] = -sn[i] * h[i][col] + cs[i] * h[i+1][col];
                        h[i][col] = temp;
                    }
                    // form new rotation for column col
                    let h_kk = h[col][col];
                    let h_k1k = h[col+1][col];
                    let r = (h_kk*h_kk + h_k1k*h_k1k).sqrt();
                    if r.abs() < T::epsilon() {
                        cs[col] = T::one();
                        sn[col] = T::zero();
                    } else {
                        cs[col] = h_kk / r;
                        sn[col] = h_k1k / r;
                    }
                    // apply rotation
                    h[col][col] = cs[col] * h_kk + sn[col] * h_k1k;
                    h[col+1][col] = T::zero();
                    // update g
                    let temp = cs[col] * g[col] + sn[col] * g[col+1];
                    g[col+1] = -sn[col] * g[col] + cs[col] * g[col+1];
                    g[col] = temp;
                }

                // 5) Check convergence on residual
                let gnorm = g[j + t].abs();
                iteration += t;
                let (stop, sstats) = self.conv.check(gnorm, res0, iteration);
                stats = sstats.clone();
                if stop {
                    break;
                }
                j += t;
            }

            // 6) Solve least-squares H y = g via back-substitution
            let m_eff = j;
            let mut y = vec![T::zero(); m_eff];
            // Back-substitution
            for i in (0..m_eff).rev() {
                let mut sum = g[i];
                for k in i+1..m_eff { sum = sum - h[i][k] * y[k]; }
                if h[i][i].abs() > T::epsilon() {
                    y[i] = sum / h[i][i];
                }
            }

            // 7) Update solution xk += Q[:,0..m_eff] * y
            for i in 0..m_eff {
                let coeff = y[i];
                let qi = &v_basis[i];
                xk.iter_mut()
                   .zip(qi.as_ref())
                   .for_each(|(xi, &qi)| *xi = *xi + coeff * qi);
            }

            // 8) Compute new residual r0 = b - A xk
            a.matvec(&V::from(xk.clone()), &mut tmp);
            r0_vec = tmp.as_ref().iter().zip(b.as_ref())
                        .map(|(&ax, &bi)| bi - ax).collect();
            r0 = V::from(r0_vec.clone());
            beta = ip.norm(&r0);
            stats.final_residual = beta;
            stats.converged = beta <= self.conv.tol * res0;
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
    use crate::utils::convergence::Convergence;

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
    fn pca_gmres_solves_small_system() {
        // 3x3 system: A = [[4,1,2],[1,3,1],[2,1,3]], x_true = [1,2,3]
        let a = DenseMat {
            data: vec![
                vec![4.0, 1.0, 2.0],
                vec![1.0, 3.0, 1.0],
                vec![2.0, 1.0, 3.0],
            ]
        };
        let x_true = vec![1.0, 2.0, 3.0];
        let mut b = vec![0.0; 3];
        a.matvec(&x_true, &mut b);
        let mut x = vec![0.0; 3];
        let mut solver = PcaGmresSolver::new(6, 2, 2, 1e-10, 30);
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(x_true.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged, "PCA-GMRES did not converge");
    }
}
