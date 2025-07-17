//! Convergence tracking & tolerance checks for iterative solvers.

/// Convergence criteria for iterative solvers.
///
/// This struct defines four types of stopping criteria:
/// - **Relative tolerance**: `‖r‖/‖b‖ ≤ rtol`
/// - **Absolute tolerance**: `‖r‖ ≤ atol` 
/// - **Divergence threshold**: `‖r‖ ≥ dtol * ‖b‖`
/// - **Maximum iterations**: `iterations ≥ max_iters`
pub struct Convergence<T> {
    /// Relative tolerance: ‖r‖/‖b‖ ≤ rtol ⇒ converge
    pub rtol: T,
    /// Absolute tolerance: ‖r‖ ≤ atol ⇒ converge
    pub atol: T,
    /// Divergence threshold: ‖r‖ ≥ dtol * ‖b‖ ⇒ diverge
    pub dtol: T,
    /// Maximum iterations
    pub max_iters: usize,
}

/// Reason for convergence or divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergedReason {
    /// Converged due to relative tolerance: ‖r‖/‖b‖ ≤ rtol
    ConvergedRtol,
    /// Converged due to absolute tolerance: ‖r‖ ≤ atol
    ConvergedAtol,
    /// Diverged due to divergence tolerance: ‖r‖ ≥ dtol * ‖b‖
    DivergedDtol,
    /// Diverged due to maximum iterations reached
    DivergedMaxIts,
    /// Continue iterating (none of the stopping criteria met)
    Continued,
}

/// Statistics from a solve operation.
#[derive(Clone, Debug)]
pub struct SolveStats<T> {
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub final_residual: T,
    /// Reason for stopping
    pub reason: ConvergedReason,
}

impl<T: Copy + PartialOrd + From<f64> + std::ops::Mul<Output = T>> Convergence<T> {
    /// Create new convergence criteria.
    pub fn new(rtol: T, atol: T, dtol: T, max_iters: usize) -> Self {
        Self { rtol, atol, dtol, max_iters }
    }

    /// Check convergence/divergence criteria.
    ///
    /// Returns (reason, SolveStats) based on current residual norm and iteration count.
    ///
    /// # Arguments
    /// * `rnorm` - Current residual norm ‖r‖
    /// * `bnorm` - Right-hand side norm ‖b‖
    /// * `iters` - Current iteration count
    ///
    /// # Returns
    /// Tuple of (ConvergedReason, SolveStats) indicating the stopping reason.
    pub fn check(&self, rnorm: T, bnorm: T, iters: usize) -> (ConvergedReason, SolveStats<T>) {
        // Absolute tolerance test first (most restrictive)
        if rnorm <= self.atol {
            let stats = SolveStats { 
                iterations: iters, 
                final_residual: rnorm, 
                reason: ConvergedReason::ConvergedAtol 
            };
            return (ConvergedReason::ConvergedAtol, stats);
        }
        
        // Relative tolerance test
        if rnorm <= self.rtol * bnorm {
            let stats = SolveStats { 
                iterations: iters, 
                final_residual: rnorm, 
                reason: ConvergedReason::ConvergedRtol 
            };
            return (ConvergedReason::ConvergedRtol, stats);
        }
        
        // Divergence test
        if rnorm >= self.dtol * bnorm {
            let stats = SolveStats { 
                iterations: iters, 
                final_residual: rnorm, 
                reason: ConvergedReason::DivergedDtol 
            };
            return (ConvergedReason::DivergedDtol, stats);
        }
        
        // Maximum iterations test
        if iters >= self.max_iters {
            let stats = SolveStats { 
                iterations: iters, 
                final_residual: rnorm, 
                reason: ConvergedReason::DivergedMaxIts 
            };
            return (ConvergedReason::DivergedMaxIts, stats);
        }
        
        // Continue iterating
        let stats = SolveStats { 
            iterations: iters, 
            final_residual: rnorm, 
            reason: ConvergedReason::Continued 
        };
        (ConvergedReason::Continued, stats)
    }
}

// Legacy convenience method for backward compatibility
impl<T: Copy + num_traits::Float + std::ops::Mul<Output = T> + From<f64>> Convergence<T> {
    /// Legacy method for backward compatibility.
    /// Returns (should_stop, stats) given current `res_norm` and iteration `i`.
    /// 
    /// **Deprecated**: Use `check()` instead for more detailed convergence information.
    #[deprecated(since = "0.1.0", note = "use check() method instead")]
    pub fn check_legacy(
        &self,
        res_norm: T,
        res0_norm: T,
        i: usize,
    ) -> (bool, SolveStats<T>) {
        let (reason, stats) = self.check(res_norm, res0_norm, i);
        let converged = matches!(reason, ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol);
        let legacy_stats = SolveStats {
            iterations: stats.iterations,
            final_residual: stats.final_residual,
            reason: stats.reason,
        };
        (converged || reason != ConvergedReason::Continued, legacy_stats)
    }
}
