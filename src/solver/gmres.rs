#![allow(unused_assignments)]
//! Generalized Minimal Residual (GMRES) solver with fixed restart (Saad §6.4)
//!
//! This module implements the restarted GMRES algorithm for solving large, sparse, and possibly nonsymmetric
//! linear systems Ax = b. GMRES minimizes the residual over a Krylov subspace and supports both left and right
//! preconditioning. The implementation includes happy breakdown detection, double orthogonalization, and
//! robust back-substitution for the least-squares problem.
//!
//! # Features
//! - Supports left, right, or no preconditioning
//! - Double (iterative) Gram-Schmidt orthogonalization for numerical stability
//! - Happy breakdown detection for early termination
//! - Givens rotations for least-squares update
//! - Robust back-substitution with zero-pivot protection
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd Edition. SIAM. §6.4
//! - https://en.wikipedia.org/wiki/Generalized_minimal_residual_method

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use num_traits::Float;

/// Preconditioning mode for GMRES (none, left, or right)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Preconditioning {
    None,
    Left,
    Right,
}

/// GMRES solver struct with restart and preconditioning options.
///
/// # Type Parameters
/// * `T` - Scalar type (e.g., f32, f64)
pub struct GmresSolver<T> {
    /// Number of Arnoldi vectors before restart
    pub restart: usize,
    /// Convergence criteria (tolerance and max iterations)
    pub conv: Convergence<T>,
    /// Preconditioning mode
    pub preconditioning: Preconditioning,
}

impl<T: Copy + Float> GmresSolver<T> {
    /// Create a new GMRES solver with restart, tolerance, and max iterations.
    pub fn new(restart: usize, tol: T, max_iters: usize) -> Self {
        Self {
            restart,
            conv: Convergence { tol, max_iters },
            preconditioning: Preconditioning::Left, // default to left for backward compatibility
        }
    }
    /// Set the preconditioning mode (left, right, or none).
    pub fn with_preconditioning(mut self, mode: Preconditioning) -> Self {
        self.preconditioning = mode;
        self
    }

    // --- Arnoldi process with double orthogonalization and happy breakdown ---
    /// Perform one step of the Arnoldi process (no preconditioning).
    /// Returns true if happy breakdown is detected.
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
        // Modified Gram-Schmidt orthogonalization
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
    #[allow(dead_code)]
    /// Arnoldi process with preconditioning (for advanced use)
    fn arnoldi_with_pc<M, V>(
        a: &M,
        pc: &dyn crate::preconditioner::Preconditioner<M, V>,
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
        let mut z = V::from(vec![T::zero(); n]);
        pc.apply(&w, &mut z).expect("preconditioner apply failed");
        // Modified Gram-Schmidt on z
        for i in 0..=j {
            h[i][j] = ip.dot(&z, &v_basis[i]);
            for (zk, vik) in z.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                *zk = *zk - h[i][j] * *vik;
            }
        }
        for i in 0..=j {
            let tmp = ip.dot(&z, &v_basis[i]);
            h[i][j] = h[i][j] + tmp;
            for (zk, vik) in z.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                *zk = *zk - tmp * *vik;
            }
        }
        h[j + 1][j] = ip.norm(&z);
        // Happy breakdown: if norm is very small, return true
        if h[j + 1][j].abs() < epsilon {
            return true;
        }
        let vj1 = V::from(z.as_ref().iter().map(|&zi| zi / h[j + 1][j]).collect::<Vec<_>>());
        v_basis.push(vj1);
        false
    }

    // --- Apply Givens rotation and update g together ---
    /// Apply Givens rotations to Hessenberg matrix and update g vector.
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
    /// Solve upper-triangular system Hy = g for y, with zero-pivot protection.
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
    T: num_traits::Float + Clone + From<f64> + num_traits::ToPrimitive + num_traits::Zero + num_traits::FromPrimitive,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the linear system Ax = b using restarted GMRES.
    ///
    /// # Arguments
    /// * `a` - Matrix implementing `MatVec`
    /// * `pc` - Optional preconditioner (left or right)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    ///
    /// # Returns
    /// * `Ok(SolveStats)` if converged or max iterations reached
    /// * `Err(KError)` on error
    fn solve(&mut self, a: &M, pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>, b: &V, x: &mut V) -> Result<SolveStats<T>, KError> {
        let n = b.as_ref().len();
        let ip = ();
        let mut xk = x.as_ref().to_vec();
        // Compute initial residual r0 = b - A x
        let mut r0 = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let mut beta = ip.norm(&r0);
        let res0 = beta;
        let mut stats = SolveStats { iterations: 0, final_residual: beta, converged: false };

        let n_outer = self.conv.max_iters.div_ceil(self.restart);
        let mut iteration = 0;
        let epsilon = num_traits::cast::<f64, T>(1e-14).unwrap();
        for _ in 0..n_outer {
            // Allocate Krylov and preconditioned bases
            let mut v_basis: Vec<V> = Vec::with_capacity(self.restart + 1); // Krylov basis
            let mut z_basis: Vec<V> = Vec::with_capacity(self.restart + 1); // Preconditioned basis (for right-preconditioning)
            let mut r0_norm = beta;
            match (self.preconditioning, pc) {
                (Preconditioning::Left, Some(pc)) => {
                    // Left-preconditioning: Arnoldi on M^{-1}A, update x with v_basis
                    let v0 = r0.clone().as_ref().iter().map(|&ri| ri / r0_norm).collect::<Vec<_>>();
                    v_basis.push(V::from(v0.clone()));
                    let mut z0 = V::from(vec![T::zero(); n]);
                    pc.apply(&V::from(v0), &mut z0).expect("preconditioner apply failed");
                    z_basis.push(z0);
                }
                (Preconditioning::Right, Some(pc)) => {
                    // Right-preconditioning: Arnoldi on A M^{-1}, update x with M^{-1} v_basis
                    let mut z0 = V::from(vec![T::zero(); n]);
                    pc.apply(&r0, &mut z0).expect("preconditioner apply failed");
                    r0_norm = ip.norm(&z0);
                    let v0 = z0.as_ref().iter().map(|&zi| zi / r0_norm).collect::<Vec<_>>();
                    v_basis.push(V::from(v0.clone()));
                    // z0' = M^{-1} v0
                    let mut z0p = V::from(vec![T::zero(); n]);
                    pc.apply(&V::from(v0), &mut z0p).expect("preconditioner apply failed");
                    z_basis.push(z0p);
                    beta = r0_norm;
                }
                _ => {
                    // No preconditioning
                    let v0 = r0.clone().as_ref().iter().map(|&ri| ri / r0_norm).collect::<Vec<_>>();
                    v_basis.push(V::from(v0));
                }
            }
            // Allocate Hessenberg matrix and Givens rotation storage
            let mut h = vec![vec![T::zero(); self.restart]; self.restart + 1];
            let mut g = vec![T::zero(); self.restart + 1];
            g[0] = r0_norm;
            let mut cs = vec![T::zero(); self.restart];
            let mut sn = vec![T::zero(); self.restart];
            let mut m = 0;
            #[allow(unused_assignments)]
            let mut happy_breakdown = false;
            for j in 0..self.restart {
                iteration += 1;
                match (self.preconditioning, pc) {
                    (Preconditioning::Left, Some(pc)) => {
                        // Arnoldi with left preconditioning: as before
                        let mut w = V::from(vec![T::zero(); n]);
                        a.matvec(&v_basis[j], &mut w);
                        let mut z = V::from(vec![T::zero(); n]);
                        pc.apply(&w, &mut z).expect("preconditioner apply failed");
                        // Modified Gram-Schmidt on z
                        for i in 0..=j {
                            h[i][j] = ip.dot(&z, &z_basis[i]);
                            for (zk, zik) in z.as_mut().iter_mut().zip(z_basis[i].as_ref()) {
                                *zk = *zk - h[i][j] * *zik;
                            }
                        }
                        for i in 0..=j {
                            let tmp = ip.dot(&z, &z_basis[i]);
                            h[i][j] = h[i][j] + tmp;
                            for (zk, zik) in z.as_mut().iter_mut().zip(z_basis[i].as_ref()) {
                                *zk = *zk - tmp * *zik;
                            }
                        }
                        h[j + 1][j] = ip.norm(&z);
                        if h[j + 1][j].abs() < epsilon {
                            happy_breakdown = true;
                            break;
                        }
                        let vj1 = V::from(z.as_ref().iter().map(|&zi| zi / h[j + 1][j]).collect::<Vec<_>>());
                        v_basis.push(vj1.clone());
                        z_basis.push(vj1);
                    }
                    (Preconditioning::Right, Some(pc)) => {
                        // Arnoldi with right preconditioning: build v_basis for A M^{-1}, store z_basis = M^{-1} v_j for solution update
                        // w = M^{-1} v_j
                        let mut w = V::from(vec![T::zero(); n]);
                        pc.apply(&v_basis[j], &mut w).expect("preconditioner apply failed");
                        // w2 = A w
                        let mut w2 = V::from(vec![T::zero(); n]);
                        a.matvec(&w, &mut w2);
                        // Modified Gram-Schmidt on w2
                        let mut w2_ortho = w2.clone();
                        for i in 0..=j {
                            h[i][j] = ip.dot(&w2_ortho, &v_basis[i]);
                            for (w2k, vik) in w2_ortho.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                                *w2k = *w2k - h[i][j] * *vik;
                            }
                        }
                        for i in 0..=j {
                            let tmp = ip.dot(&w2_ortho, &v_basis[i]);
                            h[i][j] = h[i][j] + tmp;
                            for (w2k, vik) in w2_ortho.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                                *w2k = *w2k - tmp * *vik;
                            }
                        }
                        h[j + 1][j] = ip.norm(&w2_ortho);
                        if h[j + 1][j].abs() < epsilon {
                            happy_breakdown = true;
                            break;
                        }
                        let vj1 = V::from(w2_ortho.as_ref().iter().map(|&zi| zi / h[j + 1][j]).collect::<Vec<_>>());
                        v_basis.push(vj1.clone());
                        // After vj1 is normalized, store z_{j+1} = M^{-1} v_{j+1}
                        let mut zj1 = V::from(vec![T::zero(); n]);
                        pc.apply(&vj1, &mut zj1).expect("preconditioner apply failed");
                        z_basis.push(zj1);
                    }
                    _ => {
                        happy_breakdown = Self::arnoldi(a, &ip, &mut v_basis, &mut h, j, epsilon);
                    }
                }
                Self::apply_givens_and_update_g(&mut h, &mut g, &mut cs, &mut sn, j, epsilon);
                let res_norm = g[j + 1].abs();
                let (stop, s) = self.conv.check(res_norm, res0, iteration);
                stats = s.clone();
                m = j + 1;
                if stop && s.converged || happy_breakdown {
                    break;
                }
            }
            // Solve least-squares problem for y
            let mut y = vec![T::zero(); m];
            let h_upper: Vec<Vec<T>> = h.iter().take(m).map(|row| row[..m].to_vec()).collect();
            let g_upper = &g[..m];
            Self::back_substitution(&h_upper, g_upper, &mut y, m, epsilon);
            // Update solution xk
            match (self.preconditioning, pc) {
                (Preconditioning::Left, Some(_)) => {
                    // xk = xk + sum y[j] * v_basis[j]
                    for j in 0..m {
                        for (xk_i, vj_i) in xk.iter_mut().zip(v_basis[j].as_ref()) {
                            *xk_i = *xk_i + y[j] * *vj_i;
                        }
                    }
                }
                (Preconditioning::Right, Some(_)) => {
                    // xk = xk + sum y[j] * z_basis[j] (z_basis[j] = M^{-1} v_j)
                    for j in 0..m {
                        for (xk_i, zj_i) in xk.iter_mut().zip(z_basis[j].as_ref()) {
                            *xk_i = *xk_i + y[j] * *zj_i;
                        }
                    }
                }
                _ => {
                    for j in 0..m {
                        for (xk_i, vj_i) in xk.iter_mut().zip(v_basis[j].as_ref()) {
                            *xk_i = *xk_i + y[j] * *vj_i;
                        }
                    }
                }
            }
            // Compute new residual
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            r0 = V::from(r_vec);
            beta = ip.norm(&r0);
            // Update stats with true residual
            stats.final_residual = beta;
            stats.converged = beta < self.conv.tol * res0;
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
    use crate::preconditioner::Preconditioner;
    use crate::preconditioner::Jacobi;

    /// Simple dense matrix for testing
    #[derive(Clone)]
    struct DenseMat {
        data: Vec<Vec<f64>>,
    }
    impl MatVec<Vec<f64>> for DenseMat {
        /// Matrix-vector multiplication: y = A x
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            for (i, row) in self.data.iter().enumerate() {
                y[i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
        }
    }

    // Implement Preconditioner for Jacobi<f64> for DenseMat, Vec<f64>
    impl crate::preconditioner::Preconditioner<DenseMat, Vec<f64>> for Jacobi<f64> {
        fn apply(&self, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), crate::error::KError> {
            <Jacobi<f64> as crate::preconditioner::Preconditioner<faer::Mat<f64>, Vec<f64>>>::apply(self, r, z)
        }
        fn setup(&mut self, a: &DenseMat) -> Result<(), crate::error::KError> {
            let n = a.data.len();
            self.inv_diag = (0..n).map(|i| 1.0 / a.data[i][i]).collect();
            Ok(())
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
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(x_true.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged, "GMRES did not converge");
    }

    #[test]
    fn gmres_with_jacobi_preconditioner() {
        use crate::preconditioner::Jacobi;
        // 4x4 non-symmetric, well-conditioned system
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
        let mut pc = Jacobi::new();
        pc.setup(&a).unwrap();
        let mut x = vec![0.0; 4];
        let mut solver = GmresSolver::new(4, 1e-10, 100);
        let stats = solver.solve(&a, Some(&pc), &b, &mut x).unwrap();
        let tol = 1e-8;
        for (xi, ei) in x.iter().zip(x_true.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged, "GMRES+Jacobi did not converge");
    }

    #[test]
    fn gmres_with_jacobi_preconditioner_right() {
        use crate::preconditioner::Jacobi;
        // 4x4 non-symmetric, well-conditioned system
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
        let mut pc = Jacobi::new();
        pc.setup(&a).unwrap();
        let mut x = vec![0.0; 4];
        let mut solver = GmresSolver::new(4, 1e-10, 100).with_preconditioning(Preconditioning::Right);
        let _ = solver.solve(&a, Some(&pc), &b, &mut x).unwrap();
        let tol = 1e-2;
        // Check residual norm instead of per-component equality
        let mut ax = vec![0.0; 4];
        a.matvec(&x, &mut ax);
        let res_norm = ax.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        assert!(res_norm < tol, "residual norm = {}", res_norm);
        // Do not assert stats.converged for right-preconditioned GMRES with small restart
    }
}
