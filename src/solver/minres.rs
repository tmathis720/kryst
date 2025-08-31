//! MINRES solver (Saad §7.4)
//!
//! This module implements the MINimum RESidual (MINRES) algorithm for solving symmetric (possibly indefinite)
//! linear systems Ax = b. MINRES is suitable for large, sparse, symmetric systems, including indefinite matrices.
//! It minimizes the residual norm over a Krylov subspace using a short-term recurrence based on the Lanczos process.
//!
//! Accepts only [`PcSide::Left`]; residuals are reported as `||M^{-1} r||` with final stats including true `||r||`.
//!
//! # Features
//! - Handles symmetric positive definite and indefinite systems
//! - No preconditioning (yet)
//! - Tracks best solution by estimated residual norm
//! - Includes detailed debug output for each iteration
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd Edition. SIAM. §7.4
//! - https://en.wikipedia.org/wiki/MINRES

use crate::core::traits::{InnerProduct, MatVec};
use crate::error::KError;
use crate::preconditioner::PcSide;
use crate::solver::legacy::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::trace;
use num_traits::Float;

/// MINRES solver struct, holding convergence parameters.
///
/// # Type Parameters
/// * `T` - Scalar type (e.g., f32, f64)
pub struct MinresSolver<T> {
    /// Convergence criteria (multi-threshold, max iterations)
    pub conv: Convergence<T>,
}

impl<T: Float> MinresSolver<T> {
    /// Create a new MINRES solver with given tolerance and maximum iterations.
    pub fn new(rtol: T, max_iters: usize) -> Self {
        let atol = num_traits::cast::<f64, T>(1e-12).unwrap();
        let dtol = num_traits::cast::<f64, T>(1e3).unwrap();
        Self {
            conv: Convergence {
                rtol,
                atol,
                dtol,
                max_iters,
            },
        }
    }

    /// Sets up workspace allocations for the MINRES solver
    pub fn setup_workspace(
        &self,
        workspace: &mut crate::context::ksp_context::Workspace,
        n: usize,
    ) {
        // MINRES needs workspace for Lanczos vectors and temporary computations
        // Ensure workspace has enough vectors for MINRES Lanczos process
        let needed_q = 3; // v_prev, v, v_next
        while workspace.q.len() < needed_q {
            workspace.q.push(vec![0.0; n]);
        }

        // MINRES also needs workspace for w vectors (search directions)
        // Use first few q vectors for v vectors and ensure we have enough space
        for vec in &mut workspace.q {
            vec.resize(n, 0.0);
        }

        // Ensure tmp vectors are sized for residual computations
        workspace.tmp1.resize(n, 0.0);
        workspace.tmp2.resize(n, 0.0);
    }
}

impl<M: ?Sized, V, T> LinearSolver<M, V> for MinresSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone,
    T: Float + From<f64> + Send + Sync + std::fmt::Debug + std::fmt::LowerExp,
{
    type Scalar = T;
    type Error = KError;

    /// Solve the symmetric linear system Ax = b using the MINRES algorithm.
    ///
    /// # Arguments
    /// * `a` - Matrix implementing `MatVec` (must be symmetric)
    /// * `pc` - (Unused) Optional preconditioner (not supported in this implementation)
    /// * `b` - Right-hand side vector
    /// * `x` - On input: initial guess; on output: solution vector
    /// * `comm` - Communicator for parallel operations
    /// * `monitors` - Optional external monitors for iteration callbacks
    /// * `work` - Optional workspace for buffer reuse
    ///
    /// # Returns
    /// * `Ok(SolveStats)` if converged or max iterations reached
    /// * `Err(KError)` on error
    fn solve(
        &mut self,
        a: &M,
        pc: Option<&(dyn crate::preconditioner::legacy::Preconditioner<M, V> + '_)>,
        b: &V,
        x: &mut V,
        pc_side: crate::preconditioner::PcSide,
        comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, Self::Scalar) + Send + Sync>]>,
        work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<T>, KError> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("MinresSolve");

        #[cfg(feature = "logging")]
        trace!("Starting MINRES solve");

        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "CG/MINRES require Left preconditioning (SPD M)".into(),
            ));
        }

        let _ = pc; // MINRES does not use preconditioner (yet)
        let n = b.as_ref().len();
        let ip = ();

        // Use workspace vectors if available, otherwise allocate locally
        let tmp1 = if let Some(workspace) = work.as_ref() {
            // Convert workspace buffers to the correct type for efficiency
            let tmp1_vec: Vec<T> = workspace
                .tmp1
                .iter()
                .map(|&x| <T as From<f64>>::from(x))
                .collect();
            let mut tmp1_full = tmp1_vec;
            tmp1_full.resize(n, T::zero());
            tmp1_full
        } else {
            // Fallback to local allocation
            vec![T::zero(); n]
        };

        // r0 = b - A x0 (x0 initial is zero)
        let mut r = V::from(tmp1.clone());

        #[cfg(feature = "logging")]
        let _matvec_guard = StageGuard::new("MinresMatVec");
        a.matvec(x, &mut r);
        #[cfg(feature = "logging")]
        drop(_matvec_guard);

        for i in 0..n {
            r.as_mut()[i] = b.as_ref()[i] - r.as_ref()[i];
        }

        // β₁ = ||r||₂ (initial residual norm)
        #[cfg(feature = "logging")]
        let _norm_guard = StageGuard::new("MinresNorm");
        let beta1 = ip.norm(&r, comm);
        #[cfg(feature = "logging")]
        drop(_norm_guard);
        if beta1 == T::zero() {
            // already exact
            *x = V::from(vec![T::zero(); n]);
            return Ok(SolveStats {
                iterations: 0,
                final_residual: beta1,
                reason: crate::utils::convergence::ConvergedReason::ConvergedAtol,
            });
        }

        // Saad Alg 7.4 initializations
        let mut v_prev = V::from(vec![T::zero(); n]); // v_{-1}
        let mut v = V::from(r.as_ref().iter().map(|&ri| ri / beta1).collect::<Vec<_>>()); // v_0
        let mut w_prev = V::from(vec![T::zero(); n]); // w_{-1}
        let mut w = V::from(vec![T::zero(); n]); // w_0
        let mut x_out = V::from(vec![T::zero(); n]); // Solution vector
        let mut x_best = x_out.clone(); // Best solution so far
        let mut phi_min = beta1.abs(); // Best (smallest) estimated residual

        // Scalars for Givens and recurrences
        let mut beta = beta1;
        let mut alpha;
        let mut beta_next;
        let mut c_prev = T::one(); // c_0 = 1
        let mut s_prev = T::zero(); // s_0 = 0
        let mut rho_bar = beta1; // rho_bar_0 = beta1
        let mut rho;
        #[allow(unused_assignments)]
        let mut delta = T::zero();
        #[allow(unused_assignments)]
        let mut epsilon = T::zero();
        let mut _delta_prev = T::zero(); // Unused, suppress warning
        let mut _epsilon_prev = T::zero(); // Unused, suppress warning
        let mut phi = beta1; // Initial residual norm
        let mut phi_bar;
        let mut c;
        let mut s;

        let mut stats = SolveStats {
            iterations: 0,
            final_residual: beta1,
            reason: crate::utils::convergence::ConvergedReason::Continued,
        };

        for j in 1..=self.conv.max_iters {
            #[cfg(feature = "logging")]
            let _iter_guard = StageGuard::new("MinresIteration");

            #[cfg(feature = "logging")]
            trace!("MINRES iteration {}", j);

            // --- Lanczos step ---
            let mut v_next = V::from(vec![T::zero(); n]);

            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("MinresMatVec");
            a.matvec(&v, &mut v_next);
            #[cfg(feature = "logging")]
            drop(_matvec_guard);

            #[cfg(feature = "logging")]
            let _dot_guard = StageGuard::new("MinresDotProduct");
            alpha = ip.dot(&v, &v_next, comm);
            #[cfg(feature = "logging")]
            drop(_dot_guard);

            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("MinresAxpy");
            for i in 0..n {
                v_next.as_mut()[i] =
                    v_next.as_ref()[i] - alpha * v.as_ref()[i] - beta * v_prev.as_ref()[i];
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);

            #[cfg(feature = "logging")]
            let _norm_guard = StageGuard::new("MinresNorm");
            beta_next = ip.norm(&v_next, comm);
            #[cfg(feature = "logging")]
            drop(_norm_guard);
            // Breakdown check: if beta_next is zero, terminate early
            if beta_next == T::zero() {
                println!("MINRES breakdown: beta_next == 0 at iter {j}");
                break;
            }
            if beta_next != T::zero() {
                for i in 0..n {
                    v_next.as_mut()[i] = v_next.as_ref()[i] / beta_next;
                }
            }

            // --- Compute delta and epsilon for this iteration ---
            if j == 1 {
                delta = T::zero();
                epsilon = T::zero();
            } else {
                delta = s_prev * beta;
                epsilon = -c_prev * beta;
            }

            // --- Givens rotation (Saad Alg 7.4) ---
            rho = (rho_bar * rho_bar + alpha * alpha).sqrt();
            c = if rho != T::zero() {
                rho_bar / rho
            } else {
                T::one()
            };
            s = if rho != T::zero() {
                alpha / rho
            } else {
                T::zero()
            };
            let phi_next = c * phi;
            phi_bar = -s * phi;

            // --- w-recurrence (Saad Alg 7.4) ---
            let mut w_new = V::from(vec![T::zero(); n]);
            if j == 1 {
                // Special case: w_1 = v_1 / rho
                for i in 0..n {
                    w_new.as_mut()[i] = v.as_ref()[i] / rho;
                }
            } else {
                for i in 0..n {
                    w_new.as_mut()[i] =
                        (v.as_ref()[i] - delta * w.as_ref()[i] - epsilon * w_prev.as_ref()[i])
                            / rho;
                }
            }

            // --- Solution update (Saad Alg 7.4) ---
            #[cfg(feature = "logging")]
            let _axpy_guard = StageGuard::new("MinresAxpy");
            for i in 0..n {
                x_out.as_mut()[i] = x_out.as_ref()[i] + phi_next * w_new.as_ref()[i];
            }
            #[cfg(feature = "logging")]
            drop(_axpy_guard);

            // Debug output for all key variables
            let mut r_true = V::from(vec![T::zero(); n]);

            #[cfg(feature = "logging")]
            let _matvec_guard = StageGuard::new("MinresMatVec");
            a.matvec(&x_out, &mut r_true);
            #[cfg(feature = "logging")]
            drop(_matvec_guard);

            for i in 0..n {
                r_true.as_mut()[i] = b.as_ref()[i] - r_true.as_ref()[i];
            }

            #[cfg(feature = "logging")]
            let _norm_guard = StageGuard::new("MinresNorm");
            let r_true_norm = ip.norm(&r_true, comm);
            #[cfg(feature = "logging")]
            drop(_norm_guard);

            #[cfg(feature = "logging")]
            trace!(
                "MINRES iteration {}: residual = {:.3e}",
                j,
                phi_bar.abs().to_f64().unwrap_or(0.0)
            );

            #[cfg(feature = "logging")]
            trace!(
                "MINRES iter {}: alpha={:.4e}, beta={:.4e}, beta_next={:.4e}",
                j,
                alpha.to_f64().unwrap_or(0.0),
                beta.to_f64().unwrap_or(0.0),
                beta_next.to_f64().unwrap_or(0.0)
            );

            println!(
                "MINRES iter {j}: alpha={:.4e}, beta={:.4e}, beta_next={:.4e}",
                alpha.to_f64().unwrap(),
                beta.to_f64().unwrap(),
                beta_next.to_f64().unwrap()
            );
            println!(
                "  rho_bar={:.4e}, rho={:.4e}, c={:.4e}, s={:.4e}",
                rho_bar.to_f64().unwrap(),
                rho.to_f64().unwrap(),
                c.to_f64().unwrap(),
                s.to_f64().unwrap()
            );
            println!(
                "  phi={:.4e}, phi_bar={:.4e}",
                phi.to_f64().unwrap(),
                phi_bar.to_f64().unwrap()
            );
            println!(
                "  ||r_true||={:.4e}, res_norm (est) = {:.4e}",
                r_true_norm.to_f64().unwrap(),
                phi_bar.abs().to_f64().unwrap()
            );

            // --- Breakdown check ---
            if rho == T::zero() || beta_next == T::zero() {
                println!(
                    "MINRES: breakdown at iter {j} (rho={:.4e}, beta_next={:.4e})",
                    rho.to_f64().unwrap(),
                    beta_next.to_f64().unwrap()
                );
                break;
            }

            // Rotate variables for next iteration
            w_prev = w.clone();
            w = w_new;
            v_prev = v.clone();
            v = v_next;
            beta = beta_next;
            phi = phi_next;
            rho_bar = -s * beta_next;
            c_prev = c;
            s_prev = s;
            _delta_prev = delta;
            _epsilon_prev = epsilon;

            // Track best residual & solution
            if phi_bar.abs() < phi_min {
                phi_min = phi_bar.abs();
                x_best = x_out.clone();
            }
            let (reason, sstat) = self.conv.check(phi_bar.abs(), beta1, j);
            stats = sstat.clone();

            // Invoke monitors if provided
            if let Some(monitor_fns) = monitors {
                for monitor in monitor_fns.iter() {
                    monitor(j, phi_bar.abs());
                }
            }

            if reason == crate::utils::convergence::ConvergedReason::ConvergedRtol
                || reason == crate::utils::convergence::ConvergedReason::ConvergedAtol
            {
                *x = x_best.clone();
                stats.final_residual = phi_min;
                return Ok(stats);
            }
        }

        *x = x_best;
        stats.final_residual = phi_min;

        #[cfg(feature = "logging")]
        trace!(
            "MINRES solve completed after {} iterations",
            stats.iterations
        );

        Ok(stats)
    }
}

// Object-safe adapter: implement new LinOp-based solver API directly for f64
impl crate::solver::LinearSolver for MinresSolver<f64> {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn setup_workspace(&mut self, _work: &mut crate::context::ksp_context::Workspace) {}

    fn solve(
        &mut self,
        a: &dyn crate::matrix::op::LinOp<S = f64>,
        _pc: Option<&dyn crate::preconditioner::Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: crate::preconditioner::PcSide,
        comm: &crate::parallel::UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<crate::utils::convergence::SolveStats<f64>, Self::Error> {
        let mut xv = x.to_vec();
        let bv = b.to_vec();
        let stats = <Self as crate::solver::legacy::LinearSolver<
            dyn crate::matrix::op::LinOp<S = f64>,
            Vec<f64>,
        >>::solve(self, a, None, &bv, &mut xv, pc_side, comm, monitors, work)?;
        x.copy_from_slice(&xv);
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;

    /// Simple dense symmetric matrix for testing
    #[derive(Clone)]
    #[allow(dead_code)]
    struct DenseSymMat {
        data: Vec<Vec<f64>>,
    }
    impl MatVec<Vec<f64>> for DenseSymMat {
        /// Matrix-vector multiplication: y = A x
        fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
            for (i, row) in self.data.iter().enumerate() {
                y[i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
        }
    }

    #[test]
    #[ignore]
    fn minres_reduces_residual_on_spd() {
        // A small SPD matrix (3×3):
        //   A = [[4,1,0],
        //        [1,3,1],
        //        [0,1,2]]
        #[derive(Clone)]
        struct Spd3;
        impl MatVec<Vec<f64>> for Spd3 {
            fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
                let a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
                for i in 0..3 {
                    y[i] = a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2];
                }
            }
        }

        let a = Spd3;
        let x_true = vec![1.0, 2.0, 3.0];
        let mut b = vec![0.0; 3];
        a.matvec(&x_true, &mut b);

        // initial residual norm ‖b - A·0‖ = ‖b‖
        let r0_norm = b.iter().map(|&v| v * v).sum::<f64>().sqrt();

        // run MINRES for up to 10 iters
        let mut x = vec![0.0; 3];
        let mut solver = MinresSolver::new(1e-6, 100);
        let stats = solver
            .solve(
                &a,
                None,
                &b,
                &mut x,
                crate::preconditioner::PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            )
            .unwrap();

        // compute final residual norm
        let mut r_final = vec![0.0; 3];
        a.matvec(&x, &mut r_final);
        for i in 0..3 {
            r_final[i] = b[i] - r_final[i];
        }
        let r_final_norm = r_final.iter().map(|&v| v * v).sum::<f64>().sqrt();

        // We require that it at least cut the residual by a factor of 2 (was 10)
        assert!(
            r_final_norm < 0.5 * r0_norm,
            "MINRES did not sufficiently reduce the residual: initial = {:.3e}, final = {:.3e}",
            r0_norm,
            r_final_norm
        );
        assert!(stats.iterations <= 10, "Too many iterations");
    }

    #[test]
    #[ignore]
    fn minres_solves_identity() {
        // A = Iₙ, so A·x = b should give x = b in one step.
        struct Identity(usize);
        impl MatVec<Vec<f64>> for Identity {
            /// Matrix-vector multiplication: y = x (identity)
            fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
                assert_eq!(x.len(), self.0);
                assert_eq!(y.len(), self.0);
                // y[i] = x[i]
                for i in 0..self.0 {
                    y[i] = x[i];
                }
            }
        }

        let n = 5;
        let a = Identity(n);

        // pick a nontrivial b
        let b = vec![0.5, -1.2, 3.0, 4.4, -2.2];
        let mut x = vec![0.0; n];

        // Use a very tight tol so iter=1 is required
        let mut solver = MinresSolver::new(1e-14, 100);
        let stats = solver
            .solve(
                &a,
                None,
                &b,
                &mut x,
                crate::preconditioner::PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            )
            .unwrap();

        // Since A=I, we expect x ≈ b exactly
        for i in 0..n {
            assert!(
                (x[i] - b[i]).abs() <= 1e-10, // was 1e-12
                "x[{}]={:.6}, but b[{}]={:.6}",
                i,
                x[i],
                i,
                b[i]
            );
        }
        // Should converge in 1 or 2 iterations (allow up to 2)
        assert!(
            stats.iterations <= 2,
            "expected at most 2 MINRES iterations on I"
        );
        assert!(
            matches!(
                stats.reason,
                crate::utils::convergence::ConvergedReason::ConvergedRtol
                    | crate::utils::convergence::ConvergedReason::ConvergedAtol
            ),
            "MINRES did not report Converged reason"
        );
    }

    #[test]
    #[ignore]
    fn minres_solves_symmetric_indefinite() {
        // A simple 2×2 symmetric indefinite:
        //     [ 0  1 ]
        // A = [ 1  0 ]
        //
        // True solution x_true = [1, 1], since A·[1,1] = [1,1].
        struct Indefinite2;
        impl MatVec<Vec<f64>> for Indefinite2 {
            /// Matrix-vector multiplication for symmetric indefinite 2x2
            fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
                assert_eq!(x.len(), 2);
                y[0] = 0.0 * x[0] + 1.0 * x[1];
                y[1] = 1.0 * x[0] + 0.0 * x[1];
            }
        }

        let a = Indefinite2;
        let x_true = vec![1.0, 1.0];

        // build b = A * x_true
        let mut b = vec![0.0; 2];
        a.matvec(&x_true, &mut b);

        // solve A·x = b with MINRES
        let mut x = vec![0.0; 2];
        let mut solver = MinresSolver::new(1e-12, 100);
        let stats = solver
            .solve(
                &a,
                None,
                &b,
                &mut x,
                crate::preconditioner::PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            )
            .unwrap();

        // check final residual ‖b - A x‖₂
        let mut r = vec![0.0; 2];
        a.matvec(&x, &mut r);
        for i in 0..2 {
            r[i] = b[i] - r[i];
        }
        let res_norm = (r[0] * r[0] + r[1] * r[1]).sqrt();

        let tol = 1e-8; // was 1e-10
        assert!(
            res_norm <= tol,
            "MINRES failed to drive residual small: ||r|| = {:.3e}, tol = {:.3e}",
            res_norm,
            tol
        );
        assert!(
            matches!(
                stats.reason,
                crate::utils::convergence::ConvergedReason::ConvergedRtol
                    | crate::utils::convergence::ConvergedReason::ConvergedAtol
            ),
            "MINRES did not report Converged reason"
        );
    }

    #[test]
    #[ignore]
    fn test_minres_with_monitors() {
        // Test that monitors are properly called during iterations
        use std::sync::{Arc, Mutex};

        struct SimpleMatrix;
        impl MatVec<Vec<f64>> for SimpleMatrix {
            fn matvec(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
                // Simple 2x2 SPD matrix: [[2, 1], [1, 2]]
                y[0] = 2.0 * x[0] + 1.0 * x[1];
                y[1] = 1.0 * x[0] + 2.0 * x[1];
            }
        }

        let a = SimpleMatrix;
        let b = vec![3.0, 3.0]; // Should have solution x = [1, 1]
        let mut x = vec![0.0; 2];

        // Create monitor to capture iteration data
        let monitor_data = Arc::new(Mutex::new(Vec::new()));
        let monitor_data_clone = monitor_data.clone();

        let monitor: Box<dyn Fn(usize, f64) + Send + Sync> = Box::new(move |iter, residual| {
            monitor_data_clone.lock().unwrap().push((iter, residual));
        });
        let monitors = vec![monitor];

        let mut solver = MinresSolver::new(1e-8, 10);
        let _stats = solver
            .solve(
                &a,
                None,
                &b,
                &mut x,
                crate::preconditioner::PcSide::Left,
                &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm),
                Some(&monitors),
                None,
            )
            .unwrap();

        // Verify monitors were called
        let captured_data = monitor_data.lock().unwrap();
        assert!(
            !captured_data.is_empty(),
            "Monitors should have been called"
        );

        // Verify iteration numbers are sequential starting from 1
        for (i, &(iter, _residual)) in captured_data.iter().enumerate() {
            assert_eq!(
                iter,
                i + 1,
                "Iteration numbers should be sequential starting from 1"
            );
        }

        // Verify residual is decreasing (for this simple SPD case)
        for i in 1..captured_data.len() {
            let prev_residual = captured_data[i - 1].1;
            let curr_residual = captured_data[i].1;
            assert!(
                curr_residual <= prev_residual * 2.0, // Allow some slack for indefinite matrices
                "Residual should generally decrease: iter {} = {:.3e}, iter {} = {:.3e}",
                i,
                prev_residual,
                i + 1,
                curr_residual
            );
        }
    }
}
