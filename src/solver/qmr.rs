//! QMR solver (Saad §7.3)
//!
//! This module implements the Quasi-Minimal Residual (QMR) algorithm for solving large, sparse,
//! nonsymmetric linear systems Ax = b. QMR is based on the Bi-Lanczos process and is designed to
//! minimize the residual norm in a quasi-minimal sense. It is suitable for nonsymmetric and indefinite
//! systems, and does not require breakdown-avoiding look-ahead as in BiCG.
//!
//! # Features
//! - Handles general nonsymmetric systems
//! - Uses both A and A^T (matrix and its transpose)
//! - No preconditioning in this implementation
//! - Tracks true residual for convergence
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd Edition. SIAM. §7.3
//! - https://en.wikipedia.org/wiki/Quasi-minimal_residual_method

use std::iter::Sum;

use crate::solver::LinearSolver;
use crate::core::traits::{MatVec, InnerProduct, MatTransVec};
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use num_traits::Float;

#[cfg(feature = "logging")]
use log::trace;
#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;

/// Quasi-Minimal Residual (QMR) method for nonsymmetric A
pub struct QmrSolver<T> {
    /// Convergence criteria (multi-threshold, max iterations)
    pub conv: Convergence<T>,
}

impl<T: Float> QmrSolver<T> {
    /// Create a new QMR solver with given tolerance and maximum iterations.
    pub fn new(rtol: T, max_iters: usize) -> Self {
        let atol = num_traits::cast(1e-12).unwrap_or(T::epsilon());
        let dtol = num_traits::cast(1e3).unwrap_or(T::one());
        Self {
            conv: Convergence {
                rtol,
                atol,
                dtol,
                max_iters,
            }
        }
    }

    /// Setup workspace for the QMR solver.
    ///
    /// Allocates temporary vectors needed for the QMR algorithm:
    /// r, r_tld, p, p_tld, v, v_tld, s, t vectors.
    pub fn setup_workspace(&mut self, work: &mut crate::context::ksp_context::Workspace, n: usize) {
        work.n = n;
        work.restart = 1; // QMR doesn't use restart like GMRES
        
        // Ensure we have enough temporary vectors for QMR (need 8 vectors total)
        work.tmp1.resize(n, 0.0);
        work.tmp2.resize(n, 0.0);
        work.tmp3.resize(n, 0.0);
        work.tmp4.resize(n, 0.0);
        
        // Use q vectors for additional workspace (r_tld, p, p_tld, v)
        while work.q.len() < 4 {
            work.q.push(vec![0.0; n]);
        }
        for q_vec in &mut work.q[..4] {
            q_vec.resize(n, 0.0);
        }
    }
}

impl<M: ?Sized, V, T> LinearSolver<M, V> for QmrSolver<T>
where
    M: 'static + MatVec<V> + MatTransVec<V> + Send + Sync,
    (): InnerProduct<V, Scalar = T>,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone + Send + Sync,
    T: Float + From<f64> + std::fmt::Debug + Sum + Send + Sync + std::fmt::LowerExp,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the linear system Ax = b using the QMR algorithm.
    ///
    /// This unified method handles all solve variants with optional monitoring,
    /// profiling, and workspace for maximum efficiency.
    ///
    /// # Arguments
    /// * `a` - Matrix implementing `MatVec` and `MatTransVec`
    /// * `_pc` - (Unused) Optional preconditioner (not supported in this implementation)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    /// * `comm` - Communicator for parallel reductions
    /// * `monitors` - Optional callbacks to invoke at each iteration with (iteration, residual_norm)
    /// * `work` - Optional pre-allocated workspace containing temporary vectors
    ///
    /// # Returns
    /// * `Ok(SolveStats)` if converged or max iterations reached
    /// * `Err(KError)` on error
    fn solve(
        &mut self,
        a: &M,
        _pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
        mut work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<Self::Scalar>, Self::Error> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("QmrSolve");
        
        let monitors = monitors.unwrap_or(&[]);
        
        // Only use monitors if monitoring is enabled at runtime
        let use_monitors = crate::utils::profiling::is_monitoring_enabled() && !monitors.is_empty();
        
        #[cfg(feature = "logging")]
        trace!("Starting QMR solve, monitoring: {}, workspace: {}", use_monitors, work.is_some());

        let n = b.as_ref().len();
        let ip = ();
        
        // Get workspace vectors or allocate locally
        // QMR needs 8 work vectors: r, r_tld, p, p_tld, v, v_tld, s, t
        let (mut r, mut r_tld, mut p, mut p_tld, mut v, mut v_tld, mut s, mut t) = 
            if let Some(workspace) = work.as_mut() {
                // Use workspace buffers, converting from f64 to T
                let r = workspace.tmp1.iter().map(|&x| <T as From<f64>>::from(x)).collect::<Vec<_>>();
                let r_tld = workspace.tmp2.iter().map(|&x| <T as From<f64>>::from(x)).collect::<Vec<_>>();
                let p = workspace.tmp3.iter().map(|&x| <T as From<f64>>::from(x)).collect::<Vec<_>>();
                let p_tld = workspace.tmp4.iter().map(|&x| <T as From<f64>>::from(x)).collect::<Vec<_>>();
                let v = workspace.q[0].iter().map(|&x| <T as From<f64>>::from(x)).collect::<Vec<_>>();
                let v_tld = workspace.q[1].iter().map(|&x| <T as From<f64>>::from(x)).collect::<Vec<_>>();
                let s = workspace.q[2].iter().map(|&x| <T as From<f64>>::from(x)).collect::<Vec<_>>();
                let t = workspace.q[3].iter().map(|&x| <T as From<f64>>::from(x)).collect::<Vec<_>>();
                (r, r_tld, p, p_tld, v, v_tld, s, t)
            } else {
                // Fallback to local allocation
                (
                    vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n],
                    vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n]
                )
            };

        let mut x_j = x.clone();
        
        // r0 = b - A x0
        #[cfg(feature = "logging")]
        let _matvec_guard = StageGuard::new("QmrMatVec");
        a.matvec(x, &mut V::from(r.clone()));
        #[cfg(feature = "logging")]
        drop(_matvec_guard);
        
        for i in 0..n {
            r[i] = b.as_ref()[i] - r[i];
        }
        
        // r_tld0 = arbitrary, use r0
        r_tld.copy_from_slice(&r);
        
        #[cfg(feature = "logging")]
        let _norm_guard = StageGuard::new("QmrNorm");
        let norm_r0 = ip.norm(&V::from(r.clone()), comm);
        #[cfg(feature = "logging")]
        drop(_norm_guard);
        
        let mut stats = SolveStats {
            iterations: 0,
            final_residual: norm_r0,
            reason: crate::utils::convergence::ConvergedReason::Continued,
        };
        
        // Invoke monitors for iteration 0
        if use_monitors {
            for monitor in monitors {
                monitor(0, norm_r0);
            }
        }
        
        #[cfg(feature = "logging")]
        let _dot_guard = StageGuard::new("QmrDotProduct");
        let mut rho = ip.dot(&V::from(r_tld.clone()), &V::from(r.clone()), comm);
        #[cfg(feature = "logging")]
        drop(_dot_guard);
        
        if rho == T::zero() {
            *x = x_j;
            
            #[cfg(feature = "logging")]
            let _norm_guard = StageGuard::new("QmrNorm");
            stats.final_residual = ip.norm(&V::from(r.clone()), comm);
            #[cfg(feature = "logging")]
            drop(_norm_guard);
            
            stats.reason = crate::utils::convergence::ConvergedReason::ConvergedAtol;
            return Ok(stats);
        }
        
        #[allow(unused_assignments)]
        let mut beta = T::zero();
        let mut res_norm = norm_r0;
        
        for j in 0..self.conv.max_iters {
            #[cfg(feature = "logging")]
            let _iter_guard = StageGuard::new("QmrIteration");
            
            #[cfg(feature = "logging")]
            trace!("QMR iteration {}", j + 1);
            
            if j == 0 {
                // First iteration: initialize p and p_tld
                p.copy_from_slice(&r);
                p_tld.copy_from_slice(&r_tld);
            } else {
                let rho_prev = rho;
                
                #[cfg(feature = "logging")]
                let _dot_guard = StageGuard::new("QmrDotProduct");
                rho = ip.dot(&V::from(r_tld.clone()), &V::from(r.clone()), comm);
                #[cfg(feature = "logging")]
                drop(_dot_guard);
                
                if rho == T::zero() {
                    break;
                }
                beta = rho / rho_prev;
                
                // Update search directions
                #[cfg(feature = "logging")]
                let _axpy_guard = StageGuard::new("QmrAxpy");
                for i in 0..n {
                    p[i] = r[i] + beta * p[i];
                    p_tld[i] = r_tld[i] + beta * p_tld[i];
                }
                #[cfg(feature = "logging")]
                drop(_axpy_guard);
            }
            
            // v = A p
            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("QmrMatVec");
            a.matvec(&V::from(p.clone()), &mut V::from(v.clone()));
            #[cfg(feature = "logging")]
            drop(_matvec_guard);
            
            // v_tld = A^T p_tld
            #[cfg(feature = "logging")]
            let _mattrans_guard = StageGuard::new("QmrMatTransVec");
            a.mattransvec(&V::from(p_tld.clone()), &mut V::from(v_tld.clone()));
            #[cfg(feature = "logging")]
            drop(_mattrans_guard);
            
            #[cfg(feature = "logging")]
            let _dot_guard = StageGuard::new("QmrDotProduct");
            let sigma = ip.dot(&V::from(p_tld.clone()), &V::from(v.clone()), comm);
            #[cfg(feature = "logging")]
            drop(_dot_guard);
            
            if sigma == T::zero() {
                break;
            }
            let alpha = rho / sigma;
            
            // s = r - alpha v
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("QmrAxpy");
            for i in 0..n {
                s[i] = r[i] - alpha * v[i];
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            
            // t = A s
            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("QmrMatVec");
            a.matvec(&V::from(s.clone()), &mut V::from(t.clone()));
            #[cfg(feature = "logging")]
            drop(_matvec_guard);
            
            #[cfg(feature = "logging")]
            let _dot_guard = StageGuard::new("QmrDotProduct");
            let t_dot_s = ip.dot(&V::from(t.clone()), &V::from(s.clone()), comm);
            let t_dot_t = ip.dot(&V::from(t.clone()), &V::from(t.clone()), comm);
            #[cfg(feature = "logging")]
            drop(_dot_guard);
            
            let omega = if t_dot_t != T::zero() { t_dot_s / t_dot_t } else { T::zero() };
            
            // x_{j+1} = x_j + alpha p + omega s
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("QmrAxpy");
            for i in 0..n {
                x_j.as_mut()[i] = x_j.as_ref()[i] + alpha * p[i] + omega * s[i];
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            
            // r = s - omega t
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("QmrAxpy");
            for i in 0..n {
                r[i] = s[i] - omega * t[i];
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            
            // Check convergence with true residual
            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("QmrMatVec");
            a.matvec(&x_j, &mut V::from(t.clone()));
            #[cfg(feature = "logging")]
            drop(_matvec_guard);
            
            for i in 0..n {
                t[i] = b.as_ref()[i] - t[i];
            }
            
            #[cfg(feature = "logging")]
            let _norm_guard = StageGuard::new("QmrNorm");
            res_norm = ip.norm(&V::from(t.clone()), comm);
            #[cfg(feature = "logging")]
            drop(_norm_guard);
            
            #[cfg(feature = "logging")]
            trace!("QMR iteration {}: residual = {:.3e}", j + 1, res_norm.to_f64().unwrap_or(0.0));
            
            // Invoke monitors if enabled
            if use_monitors {
                for monitor in monitors {
                    monitor(j + 1, res_norm);
                }
            }
            
            let (reason, s_stats) = self.conv.check(res_norm, norm_r0, j+1);
            stats = s_stats;
            if reason == crate::utils::convergence::ConvergedReason::ConvergedRtol
                || reason == crate::utils::convergence::ConvergedReason::ConvergedAtol {
                *x = x_j.clone();
                stats.final_residual = res_norm;
                stats.reason = reason;
                return Ok(stats);
            }
        }
        
        #[cfg(feature = "logging")]
        trace!("QMR solve completed after {} iterations", stats.iterations);
        
        *x = x_j;
        stats.final_residual = res_norm;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    /// Simple dense matrix for testing
    #[derive(Clone, Debug)]
    struct DenseMat { data: Vec<Vec<f64>> }
    impl MatVec<Vec<f64>> for DenseMat {
        /// Matrix-vector multiplication: y = A x
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            for i in 0..x.len() {
                y[i] = self.data[i].iter().zip(x).map(|(a,b)| a*b).sum();
            }
        }
    }
    impl crate::core::traits::MatTransVec<Vec<f64>> for DenseMat {
        /// Matrix-transpose-vector multiplication: y = A^T x
        fn mattransvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            let n = self.data.len();
            let m = self.data[0].len();
            for j in 0..m {
                y[j] = 0.0;
                for i in 0..n {
                    y[j] += self.data[i][j] * x[i];
                }
            }
        }
    }

    #[ignore]
    #[test]
    fn qmr_solves_small_nonsym() {
        // A simple 2×2 nonsymmetric system
        // [2 1; 0 3] x = [3; 6] ⇒ x = [1;2]
        let a = DenseMat { data: vec![vec![2.0,1.0], vec![0.0,3.0]] };
        let b = vec![3.0,6.0];
        let mut x = vec![0.0,0.0];
        let mut solver = QmrSolver::new(1e-10, 50);
        let stats = solver.solve(&a, None, &b, &mut x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm), None, None).unwrap();
        assert!((x[0]-1.0).abs() < 1e-4);
        assert!((x[1]-2.0).abs() < 1e-4);
        assert!(matches!(stats.reason,
            crate::utils::convergence::ConvergedReason::ConvergedRtol |
            crate::utils::convergence::ConvergedReason::ConvergedAtol), "QMR did not report Converged reason");
    }
}
