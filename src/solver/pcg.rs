//! Preconditioned Conjugate Gradient (PCG) per Saad §9.2
//!
//! This module implements the Preconditioned Conjugate Gradient (PCG) algorithm for solving large, sparse,
//! symmetric positive definite (SPD) linear systems Ax = b. The implementation supports flexible norm types,
//! single-reduction variants, trust-region (radius) and objective targeting, and per-iteration monitoring.
//!
//! # Features
//! - Supports preconditioning (left, right, or identity)
//! - Multiple norm types for convergence checks (preconditioned, unpreconditioned, natural, none)
//! - Single-reduction variant for reduced communication
//! - Optional trust-region (radius) and objective targeting
//! - Residual history and per-iteration monitoring
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd Edition. SIAM. §9.2
//! - https://en.wikipedia.org/wiki/Conjugate_gradient_method

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;

#[cfg(feature = "logging")]
use log::trace;
#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;

/// Norm type for PCG convergence and monitoring
pub enum CgNormType { Preconditioned, Unpreconditioned, Natural, None }

/// Preconditioned Conjugate Gradient (PCG) solver struct.
///
/// # Type Parameters
/// * `T` - Scalar type (e.g., f32, f64)
pub struct PcgSolver<T> {
    /// Convergence criteria (multi-threshold, max iterations)
    pub conv: Convergence<T>,
    /// Norm type for convergence and monitoring
    pub norm_type: CgNormType,
    /// Use single-reduction variant (fused dot products)
    pub single_reduction: bool,
    /// Optional trust-region radius
    pub radius: Option<T>,
    /// Optional objective target
    pub obj_target: Option<T>,
    /// Optional per-iteration monitor callback
    pub monitor: Option<Box<dyn FnMut(usize, T)>>,
    /// History of residual norms for each iteration
    pub residual_history: Vec<T>,
}

impl<T: Copy + num_traits::Float> PcgSolver<T> {
    /// Create a new PCG solver with given tolerance and maximum iterations.
    pub fn new(rtol: T, max_iters: usize) -> Self {
        let atol = num_traits::cast(1e-12).unwrap_or(T::epsilon());
        let dtol = num_traits::cast(1e3).unwrap_or(T::one());
        Self {
            conv: Convergence {
                rtol,
                atol,
                dtol,
                max_iters,
            },
            norm_type: CgNormType::Unpreconditioned,
            single_reduction: false,
            radius: None,
            obj_target: None,
            monitor: None,
            residual_history: Vec::new(),
        }
    }
    /// Set the norm type for convergence and monitoring.
    pub fn with_norm(mut self, norm_type: CgNormType) -> Self {
        self.norm_type = norm_type;
        self
    }
    /// Enable or disable single-reduction variant.
    pub fn with_single_reduction(mut self, flag: bool) -> Self {
        self.single_reduction = flag;
        self
    }
    /// Set trust-region radius.
    pub fn with_radius(mut self, radius: T) -> Self {
        self.radius = Some(radius);
        self
    }
    /// Set objective target value.
    pub fn with_obj_target(mut self, obj: T) -> Self {
        self.obj_target = Some(obj);
        self
    }
    /// Set a per-iteration monitor callback.
    pub fn with_monitor<F>(mut self, f: F) -> Self
    where F: FnMut(usize, T) + 'static {
        self.monitor = Some(Box::new(f));
        self
    }
    /// Clear the residual history.
    pub fn clear_history(&mut self) {
        self.residual_history.clear();
    }
}

impl<M, V, T> LinearSolver<M, V> for PcgSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone + Send + Sync,
    T: num_traits::Float + Clone + From<f64> + Send + Sync + std::fmt::Debug + std::fmt::LowerExp,
{
    type Error = KError;
    type Scalar = T;

    /// Solve the SPD linear system Ax = b using the PCG algorithm.
    ///
    /// # Arguments
    /// * `a` - Matrix implementing `MatVec` (must be SPD)
    /// * `pc` - Optional preconditioner
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    ///
    /// # Returns
    /// * `Ok(SolveStats)` if converged or max iterations reached
    /// * `Err(KError)` on error (e.g., indefinite matrix or preconditioner)
    fn solve(&mut self, a: &M, pc: Option<&dyn Preconditioner<M, V>>, b: &V, x: &mut V, comm: &crate::parallel::UniverseComm) -> Result<SolveStats<T>, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("PcgSolve");
        
        #[cfg(feature = "logging")]
        trace!("Starting PCG solve");

        let n = b.as_ref().len();
        let ip = ();
        let mut x_vec = x.as_ref().to_vec();
        // Compute initial residual r = b - A x
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            
            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("PcgMatVec");
            a.matvec(&V::from(x_vec.clone()), &mut tmp);
            #[cfg(feature = "logging")]
            drop(_matvec_guard);
            
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            V::from(r_vec)
        };
        // Apply preconditioner: z = M^{-1} r
        let mut z = V::from(vec![T::zero(); n]);
        if let Some(pc) = pc {
            pc.apply(crate::preconditioner::PcSide::Left, &r, &mut z)?;
        } else {
            z.clone_from(&r);
        }
        let mut p = z.clone();
        
        #[cfg(feature = "logging")]
        let _dot_guard = StageGuard::new("PcgDotProduct");
        let mut rz = ip.dot(&r, &z, comm);
        #[cfg(feature = "logging")]
        drop(_dot_guard);
        
        let res0 = rz.abs().sqrt();
        let mut stats = SolveStats {
            iterations: 0,
            final_residual: res0,
            reason: crate::utils::convergence::ConvergedReason::Continued,
        };
        // Choose norm for convergence check
        #[cfg(feature = "logging")]
        let _norm_guard = StageGuard::new("PcgNorm");
        let dp = match self.norm_type {
            CgNormType::Preconditioned => ip.dot(&z, &z, comm),
            CgNormType::Unpreconditioned => ip.dot(&r, &r, comm),
            CgNormType::Natural => ip.dot(&r, &z, comm),
            CgNormType::None => T::zero(),
        };
        #[cfg(feature = "logging")]
        drop(_norm_guard);
        if let Some(ref mut monitor) = self.monitor {
            monitor(0, dp.sqrt());
        }
        self.residual_history.push(dp.sqrt());
        for i in 0..self.conv.max_iters {
            #[cfg(feature = "logging")]
            let _iter_guard = StageGuard::new("PcgIteration");
            
            #[cfg(feature = "logging")]
            trace!("PCG iteration {}", i + 1);
            
            // Compute A p
            let mut ap = V::from(vec![T::zero(); n]);
            
            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("PcgMatVec");
            a.matvec(&p, &mut ap);
            #[cfg(feature = "logging")]
            drop(_matvec_guard);
            
            let p_dot_ap = if self.single_reduction {
                // Fused dot product: p^T A p (before r/z update)
                let mut p_dot_ap = T::zero();
                for i in 0..n {
                    p_dot_ap = p_dot_ap + p.as_ref()[i] * ap.as_ref()[i];
                }
                p_dot_ap
            } else {
                #[cfg(feature = "logging")]
                let _dot_guard = StageGuard::new("PcgDotProduct");
                let result = ip.dot(&p, &ap, comm);
                #[cfg(feature = "logging")]
                drop(_dot_guard);
                result
            };
            // Indefinite-matrix detection
            if p_dot_ap <= T::zero() {
                stats.iterations = i + 1;
                stats.final_residual = match self.norm_type {
                    CgNormType::Preconditioned => ip.dot(&z, &z, comm).sqrt(),
                    CgNormType::Unpreconditioned => ip.dot(&r, &r, comm).sqrt(),
                    CgNormType::Natural => ip.dot(&r, &z, comm).abs().sqrt(),
                    CgNormType::None => T::zero(),
                };
                // stats.converged field removed in new SolveStats
                return Err(KError::IndefiniteMatrix);
            }
            let alpha = rz / p_dot_ap;
            // Update solution: x = x + alpha * p
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("PcgAxpy");
            for (xj, pj) in x_vec.iter_mut().zip(p.as_ref()) {
                *xj = *xj + alpha * *pj;
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            
            // Update residual: r = r - alpha * A p
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("PcgAxpy");
            for (rj, apj) in r.as_mut().iter_mut().zip(ap.as_ref()) {
                *rj = *rj - alpha * *apj;
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            // Apply preconditioner: z = M^{-1} r
            if let Some(pc) = pc {
                pc.apply(crate::preconditioner::PcSide::Left, &r, &mut z)?;
            } else {
                z.clone_from(&r);
            }
            
            #[cfg(feature = "logging")]
            let _dot_guard = StageGuard::new("PcgDotProduct");
            let rz_new = ip.dot(&r, &z, comm);
            #[cfg(feature = "logging")]
            drop(_dot_guard);
            
            // Compute norm for convergence check
            #[cfg(feature = "logging")]
            let _norm_guard = StageGuard::new("PcgNorm");
            let res_norm = match self.norm_type {
                CgNormType::Preconditioned => ip.dot(&z, &z, comm).sqrt(),
                CgNormType::Unpreconditioned => ip.dot(&r, &r, comm).sqrt(),
                CgNormType::Natural => ip.dot(&r, &z, comm).abs().sqrt(),
                CgNormType::None => T::zero(),
            };
            #[cfg(feature = "logging")]
            drop(_norm_guard);
            
            #[cfg(feature = "logging")]
            trace!("PCG iteration {}: residual = {:.3e}", i + 1, res_norm.to_f64().unwrap_or(0.0));
            if let Some(ref mut monitor) = self.monitor {
                monitor(i+1, res_norm);
            }
            self.residual_history.push(res_norm);
            let (reason, s) = self.conv.check(res_norm, res0, i+1);
            stats = s.clone();
            if reason == crate::utils::convergence::ConvergedReason::ConvergedRtol
                || reason == crate::utils::convergence::ConvergedReason::ConvergedAtol {
                *x = V::from(x_vec.clone());
                return Ok(stats);
            }
            let beta = rz_new / rz;
            // Indefinite-preconditioner detection
            if beta < T::zero() {
                stats.iterations = i + 1;
                stats.final_residual = res_norm;
                stats.reason = crate::utils::convergence::ConvergedReason::DivergedDtol;
                return Err(KError::IndefinitePreconditioner);
            }
            // Update search direction: p = z + beta * p
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("PcgAxpy");
            for (pj, zj) in p.as_mut().iter_mut().zip(z.as_ref()) {
                *pj = *zj + beta * *pj;
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            
            rz = rz_new;
        }
        
        #[cfg(feature = "logging")]
        trace!("PCG solve completed after {} iterations", stats.iterations);
        
        *x = V::from(x_vec);
        Ok(stats)
    }

    fn solve_with_monitors(
        &mut self,
        a: &M,
        pc: Option<&dyn crate::preconditioner::Preconditioner<M, V>>,
        b: &V,
        x: &mut V,
        comm: &crate::parallel::UniverseComm,
        monitors: &[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]
    ) -> Result<SolveStats<Self::Scalar>, Self::Error> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("PcgSolve");
        
        #[cfg(feature = "logging")]
        trace!("Starting PCG solve with {} monitors", monitors.len());

        let n = b.as_ref().len();
        let ip = ();
        let mut x_vec = x.as_ref().to_vec();
        // Compute initial residual r = b - A x
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            
            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("PcgMatVec");
            a.matvec(&V::from(x_vec.clone()), &mut tmp);
            #[cfg(feature = "logging")]
            drop(_matvec_guard);
            
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            V::from(r_vec)
        };
        // Apply preconditioner: z = M^{-1} r
        let mut z = V::from(vec![T::zero(); n]);
        if let Some(pc) = pc {
            pc.apply(crate::preconditioner::PcSide::Left, &r, &mut z)?;
        } else {
            z.clone_from(&r);
        }
        let mut p = z.clone();
        
        #[cfg(feature = "logging")]
        let _dot_guard = StageGuard::new("PcgDotProduct");
        let mut rz = ip.dot(&r, &z, comm);
        #[cfg(feature = "logging")]
        drop(_dot_guard);
        
        let res0 = rz.abs().sqrt();
        let mut stats = SolveStats {
            iterations: 0,
            final_residual: res0,
            reason: crate::utils::convergence::ConvergedReason::Continued,
        };
        // Choose norm for convergence check
        #[cfg(feature = "logging")]
        let _norm_guard = StageGuard::new("PcgNorm");
        let dp = match self.norm_type {
            CgNormType::Preconditioned => ip.dot(&z, &z, comm),
            CgNormType::Unpreconditioned => ip.dot(&r, &r, comm),
            CgNormType::Natural => ip.dot(&r, &z, comm),
            CgNormType::None => T::zero(),
        };
        #[cfg(feature = "logging")]
        drop(_norm_guard);
        
        // Call monitors for initial state
        for monitor in monitors {
            monitor(0, dp.sqrt());
        }
        
        if let Some(ref mut monitor) = self.monitor {
            monitor(0, dp.sqrt());
        }
        self.residual_history.push(dp.sqrt());
        
        for i in 0..self.conv.max_iters {
            #[cfg(feature = "logging")]
            let _iter_guard = StageGuard::new("PcgIteration");
            
            #[cfg(feature = "logging")]
            trace!("PCG iteration {}", i + 1);
            
            // Compute A p
            let mut ap = V::from(vec![T::zero(); n]);
            
            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("PcgMatVec");
            a.matvec(&p, &mut ap);
            #[cfg(feature = "logging")]
            drop(_matvec_guard);
            
            let p_dot_ap = if self.single_reduction {
                // Fused dot product: p^T A p (before r/z update)
                let mut p_dot_ap = T::zero();
                for i in 0..n {
                    p_dot_ap = p_dot_ap + p.as_ref()[i] * ap.as_ref()[i];
                }
                p_dot_ap
            } else {
                #[cfg(feature = "logging")]
                let _dot_guard = StageGuard::new("PcgDotProduct");
                let result = ip.dot(&p, &ap, comm);
                #[cfg(feature = "logging")]
                drop(_dot_guard);
                result
            };
            // Indefinite-matrix detection
            if p_dot_ap <= T::zero() {
                stats.iterations = i + 1;
                stats.final_residual = match self.norm_type {
                    CgNormType::Preconditioned => ip.dot(&z, &z, comm).sqrt(),
                    CgNormType::Unpreconditioned => ip.dot(&r, &r, comm).sqrt(),
                    CgNormType::Natural => ip.dot(&r, &z, comm).abs().sqrt(),
                    CgNormType::None => T::zero(),
                };
                // stats.converged field removed in new SolveStats
                return Err(KError::IndefiniteMatrix);
            }
            let alpha = rz / p_dot_ap;
            // Update solution: x = x + alpha * p
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("PcgAxpy");
            for (xj, pj) in x_vec.iter_mut().zip(p.as_ref()) {
                *xj = *xj + alpha * *pj;
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            
            // Update residual: r = r - alpha * A p
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("PcgAxpy");
            for (rj, apj) in r.as_mut().iter_mut().zip(ap.as_ref()) {
                *rj = *rj - alpha * *apj;
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            
            // Apply preconditioner: z = M^{-1} r
            if let Some(pc) = pc {
                pc.apply(crate::preconditioner::PcSide::Left, &r, &mut z)?;
            } else {
                z.clone_from(&r);
            }
            
            #[cfg(feature = "logging")]
            let _dot_guard = StageGuard::new("PcgDotProduct");
            let rz_new = ip.dot(&r, &z, comm);
            #[cfg(feature = "logging")]
            drop(_dot_guard);
            
            // Compute norm for convergence check
            #[cfg(feature = "logging")]
            let _norm_guard = StageGuard::new("PcgNorm");
            let res_norm = match self.norm_type {
                CgNormType::Preconditioned => ip.dot(&z, &z, comm).sqrt(),
                CgNormType::Unpreconditioned => ip.dot(&r, &r, comm).sqrt(),
                CgNormType::Natural => ip.dot(&r, &z, comm).abs().sqrt(),
                CgNormType::None => T::zero(),
            };
            #[cfg(feature = "logging")]
            drop(_norm_guard);
            
            #[cfg(feature = "logging")]
            trace!("PCG iteration {}: residual = {:.3e}", i + 1, res_norm.to_f64().unwrap_or(0.0));
            
            // Call monitors
            for monitor in monitors {
                monitor(i + 1, res_norm);
            }
            
            if let Some(ref mut monitor) = self.monitor {
                monitor(i+1, res_norm);
            }
            self.residual_history.push(res_norm);
            let (reason, s) = self.conv.check(res_norm, res0, i+1);
            stats = s.clone();
            if reason == crate::utils::convergence::ConvergedReason::ConvergedRtol
                || reason == crate::utils::convergence::ConvergedReason::ConvergedAtol {
                *x = V::from(x_vec.clone());
                return Ok(stats);
            }
            let beta = rz_new / rz;
            // Indefinite-preconditioner detection
            if beta < T::zero() {
                stats.iterations = i + 1;
                stats.final_residual = res_norm;
                stats.reason = crate::utils::convergence::ConvergedReason::DivergedDtol;
                return Err(KError::IndefinitePreconditioner);
            }
            // Update search direction: p = z + beta * p
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("PcgAxpy");
            for (pj, zj) in p.as_mut().iter_mut().zip(z.as_ref()) {
                *pj = *zj + beta * *pj;
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);
            
            rz = rz_new;
        }
        
        #[cfg(feature = "logging")]
        trace!("PCG solve completed after {} iterations", stats.iterations);
        
        *x = V::from(x_vec);
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;
    use crate::preconditioner::Preconditioner;

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
    /// Identity preconditioner for testing
    struct IdentityPC;
    impl Preconditioner<DenseMat, Vec<f64>> for IdentityPC {
        fn setup(&mut self, _a: &DenseMat) -> Result<(), crate::error::KError> {
            Ok(())
        }
        
        fn apply(&self, _side: crate::preconditioner::PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), crate::error::KError> {
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
        let _stats_std = solver_std.solve(&a, Some(&pc), &b, &mut x_std, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm)).unwrap();
        let stats_single = solver_single.solve(&a, Some(&pc), &b, &mut x_single, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm)).unwrap();
        let tol = 1e-8;
        for (xi, xj) in x_std.iter().zip(x_single.iter()) {
            assert!((xi - xj).abs() < tol, "single-reduction and standard PCG differ: {} vs {}", xi, xj);
        }
        assert!(matches!(stats_single.reason,
            crate::utils::convergence::ConvergedReason::ConvergedRtol |
            crate::utils::convergence::ConvergedReason::ConvergedAtol), "Single-reduction PCG did not report Converged reason");
        // Also check against expected solution
        let expected = vec![0.09090909090909091, 0.6363636363636364];
        for (xi, ei) in x_single.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
    }
}
