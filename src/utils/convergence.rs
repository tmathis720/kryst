//! Convergence tracking & tolerance checks for iterative solvers.

#[allow(unused_imports)]
use crate::algebra::prelude::*;

/// Convergence criteria for iterative solvers.
///
/// This struct defines four types of stopping criteria:
/// - **Relative tolerance**: `‖r‖/‖b‖ ≤ rtol`
/// - **Absolute tolerance**: `‖r‖ ≤ atol`
/// - **Divergence threshold**: `‖r‖ ≥ dtol * ‖b‖`
/// - **Maximum iterations**: `iterations ≥ max_iters`
pub struct Convergence {
    /// Relative tolerance: ‖r‖/‖b‖ ≤ rtol ⇒ converge
    pub rtol: R,
    /// Absolute tolerance: ‖r‖ ≤ atol ⇒ converge
    pub atol: R,
    /// Divergence threshold: ‖r‖ ≥ dtol * ‖b‖ ⇒ diverge
    pub dtol: R,
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
    /// Converged because the step reached a trust-region bound
    ConvergedTrustRegion,
    /// Converged due to a happy breakdown (e.g., `pᵀAp` ≈ 0)
    ConvergedHappyBreakdown,
    /// Diverged due to divergence tolerance: ‖r‖ ≥ dtol * ‖b‖
    DivergedDtol,
    /// Diverged due to maximum iterations reached
    DivergedMaxIts,
    /// Continue iterating (none of the stopping criteria met)
    Continued,
}

/// Statistics from a solve operation.
#[derive(Clone, Debug, Default)]
pub struct SolverCounters {
    /// Number of global reduction operations executed by the solver.
    pub num_global_reductions: usize,
    /// Number of residual replacement events performed during the solve.
    pub residual_replacements: usize,
}

/// Statistics from a solve operation.
#[derive(Clone, Debug)]
pub struct SolveStats<R> {
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub final_residual: R,
    /// Reason for stopping
    pub reason: ConvergedReason,
    /// Additional counters collected during the solve.
    pub counters: SolverCounters,
    /// Total number of complex drift events observed during reductions.
    pub complex_drift_events: usize,
    /// Per-kind complex drift counts captured by the solver.
    pub complex_drift_counts: [usize; 6],
    /// Maximum relative imaginary magnitude observed.
    pub complex_drift_max_rel: R,
}

impl<R: Default> SolveStats<R> {
    /// Construct a new statistics record with zeroed counters.
    pub fn new(iterations: usize, final_residual: R, reason: ConvergedReason) -> Self {
        Self {
            iterations,
            final_residual,
            reason,
            counters: SolverCounters::default(),
            complex_drift_events: 0,
            complex_drift_counts: [0; 6],
            complex_drift_max_rel: R::default(),
        }
    }

    /// Attach solver counters to an existing statistics record.
    pub fn with_counters(mut self, counters: SolverCounters) -> Self {
        self.counters = counters;
        self
    }
}

impl Convergence {
    /// Create new convergence criteria.
    pub fn new(rtol: R, atol: R, dtol: R, max_iters: usize) -> Self {
        Self {
            rtol,
            atol,
            dtol,
            max_iters,
        }
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
    pub fn check(&self, rnorm: R, bnorm: R, iters: usize) -> (ConvergedReason, SolveStats<R>) {
        // Absolute tolerance test first (most restrictive)
        if rnorm <= self.atol {
            let stats = SolveStats::new(iters, rnorm, ConvergedReason::ConvergedAtol);
            return (ConvergedReason::ConvergedAtol, stats);
        }

        // Relative tolerance test
        if rnorm <= self.rtol * bnorm {
            let stats = SolveStats::new(iters, rnorm, ConvergedReason::ConvergedRtol);
            return (ConvergedReason::ConvergedRtol, stats);
        }

        // Divergence test
        if rnorm >= self.dtol * bnorm {
            let stats = SolveStats::new(iters, rnorm, ConvergedReason::DivergedDtol);
            return (ConvergedReason::DivergedDtol, stats);
        }

        // Maximum iterations test
        if iters >= self.max_iters {
            let stats = SolveStats::new(iters, rnorm, ConvergedReason::DivergedMaxIts);
            return (ConvergedReason::DivergedMaxIts, stats);
        }

        // Continue iterating
        let stats = SolveStats::new(iters, rnorm, ConvergedReason::Continued);
        (ConvergedReason::Continued, stats)
    }
}

// Legacy convenience method for backward compatibility
impl Convergence {
    /// Legacy method for backward compatibility.
    /// Returns (should_stop, stats) given current `res_norm` and iteration `i`.
    ///
    /// **Deprecated**: Use `check()` instead for more detailed convergence information.
    #[deprecated(since = "0.1.0", note = "use check() method instead")]
    pub fn check_legacy(&self, res_norm: R, res0_norm: R, i: usize) -> (bool, SolveStats<R>) {
        let (reason, stats) = self.check(res_norm, res0_norm, i);
        let converged = matches!(
            reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        );
        let mut legacy_stats =
            SolveStats::new(stats.iterations, stats.final_residual, stats.reason);
        legacy_stats.counters = stats.counters;
        (
            converged || reason != ConvergedReason::Continued,
            legacy_stats,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_new() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 1000);
        assert_eq!(conv.rtol, 1e-6);
        assert_eq!(conv.atol, 1e-12);
        assert_eq!(conv.dtol, 1e3);
        assert_eq!(conv.max_iters, 1000);
    }

    #[test]
    fn test_converged_absolute_tolerance() {
        let conv = Convergence::new(1e-6, 1e-8, 1e3, 100);
        let rnorm = 1e-9; // Less than atol
        let bnorm = 1.0;
        let iters = 5;

        let (reason, stats) = conv.check(rnorm, bnorm, iters);

        assert_eq!(reason, ConvergedReason::ConvergedAtol);
        assert_eq!(stats.reason, ConvergedReason::ConvergedAtol);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.final_residual, 1e-9);
    }

    #[test]
    fn test_converged_relative_tolerance() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 100);
        let rnorm = 1e-7; // Greater than atol but satisfies rtol
        let bnorm = 1.0;
        let iters = 10;

        let (reason, stats) = conv.check(rnorm, bnorm, iters);

        assert_eq!(reason, ConvergedReason::ConvergedRtol);
        assert_eq!(stats.reason, ConvergedReason::ConvergedRtol);
        assert_eq!(stats.iterations, 10);
        assert_eq!(stats.final_residual, 1e-7);
    }

    #[test]
    fn test_diverged_tolerance() {
        let conv = Convergence::new(1e-6, 1e-12, 2.0, 100);
        let rnorm = 3.0; // Greater than dtol * bnorm
        let bnorm = 1.0;
        let iters = 5;

        let (reason, stats) = conv.check(rnorm, bnorm, iters);

        assert_eq!(reason, ConvergedReason::DivergedDtol);
        assert_eq!(stats.reason, ConvergedReason::DivergedDtol);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.final_residual, 3.0);
    }

    #[test]
    fn test_diverged_max_iterations() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 10);
        let rnorm = 1e-3; // Not converged but within tolerance bounds
        let bnorm = 1.0;
        let iters = 10; // Equal to max_iters

        let (reason, stats) = conv.check(rnorm, bnorm, iters);

        assert_eq!(reason, ConvergedReason::DivergedMaxIts);
        assert_eq!(stats.reason, ConvergedReason::DivergedMaxIts);
        assert_eq!(stats.iterations, 10);
        assert_eq!(stats.final_residual, 1e-3);
    }

    #[test]
    fn test_continued() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 100);
        let rnorm = 1e-3; // Not converged, not diverged, within iteration limit
        let bnorm = 1.0;
        let iters = 5;

        let (reason, stats) = conv.check(rnorm, bnorm, iters);

        assert_eq!(reason, ConvergedReason::Continued);
        assert_eq!(stats.reason, ConvergedReason::Continued);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.final_residual, 1e-3);
    }

    #[test]
    fn test_convergence_precedence() {
        // Absolute tolerance takes precedence over relative
        let conv = Convergence::new(1e-6, 1e-8, 1e3, 100);
        let rnorm = 1e-9; // Satisfies both atol and rtol
        let bnorm = 1.0;
        let iters = 5;

        let (reason, _) = conv.check(rnorm, bnorm, iters);
        assert_eq!(reason, ConvergedReason::ConvergedAtol);
    }

    #[test]
    fn test_converged_reason_equality() {
        assert_eq!(
            ConvergedReason::ConvergedRtol,
            ConvergedReason::ConvergedRtol
        );
        assert_eq!(
            ConvergedReason::ConvergedAtol,
            ConvergedReason::ConvergedAtol
        );
        assert_eq!(ConvergedReason::DivergedDtol, ConvergedReason::DivergedDtol);
        assert_eq!(
            ConvergedReason::DivergedMaxIts,
            ConvergedReason::DivergedMaxIts
        );
        assert_eq!(ConvergedReason::Continued, ConvergedReason::Continued);

        assert_ne!(
            ConvergedReason::ConvergedRtol,
            ConvergedReason::ConvergedAtol
        );
        assert_ne!(
            ConvergedReason::DivergedDtol,
            ConvergedReason::DivergedMaxIts
        );
    }

    #[test]
    fn test_converged_reason_debug() {
        let reason = ConvergedReason::ConvergedRtol;
        let debug_str = format!("{:?}", reason);
        assert!(debug_str.contains("ConvergedRtol"));
    }

    #[test]
    fn test_solve_stats_clone() {
        let stats = SolveStats::new(42, 1e-8, ConvergedReason::ConvergedRtol);

        let cloned = stats.clone();
        assert_eq!(cloned.iterations, 42);
        assert_eq!(cloned.final_residual, 1e-8);
        assert_eq!(cloned.reason, ConvergedReason::ConvergedRtol);
    }

    #[test]
    fn test_solve_stats_debug() {
        let stats = SolveStats::new(10, 1e-6, ConvergedReason::ConvergedAtol);

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("ConvergedAtol"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_legacy_check_convergence() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 100);
        let res_norm = 1e-8;
        let res0_norm = 1.0;
        let iters = 5;

        let (should_stop, stats) = conv.check_legacy(res_norm, res0_norm, iters);

        assert!(should_stop);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.final_residual, 1e-8);
    }

    #[test]
    #[allow(deprecated)]
    fn test_legacy_check_continue() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 100);
        let res_norm = 1e-3;
        let res0_norm = 1.0;
        let iters = 5;

        let (should_stop, stats) = conv.check_legacy(res_norm, res0_norm, iters);

        assert!(!should_stop);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.final_residual, 1e-3);
        assert_eq!(stats.reason, ConvergedReason::Continued);
    }

    #[test]
    fn test_different_numeric_types() {
        // Test with f64 (which implements From<f64>)
        let conv_f64 = Convergence::new(1e-6f64, 1e-12f64, 1e3f64, 100);
        let (reason, _) = conv_f64.check(1e-8f64, 1.0f64, 5);
        assert_eq!(reason, ConvergedReason::ConvergedRtol);

        // Test with different tolerances
        let conv2 = Convergence::new(1e-8, 1e-16, 1e6, 50);
        let (reason2, _) = conv2.check(1e-10, 1.0, 10);
        assert_eq!(reason2, ConvergedReason::ConvergedRtol);
    }
}
