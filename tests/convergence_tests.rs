//! Tests for flexible convergence and divergence criteria

use kryst::utils::convergence::{ConvergedReason, Convergence};
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_rtol() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 100);
        let bnorm = 1.0;
        let rnorm = 1e-7; // Less than rtol * bnorm = 1e-6

        let (reason, stats) = conv.check(rnorm, bnorm, 5);

        assert_eq!(reason, ConvergedReason::ConvergedRtol);
        assert_eq!(stats.reason, ConvergedReason::ConvergedRtol);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.final_residual, rnorm);
    }

    #[test]
    fn test_convergence_atol() {
        let conv = Convergence::new(1e-6, 1e-8, 1e3, 100);
        let bnorm = 1.0;
        let rnorm = 1e-9; // Less than atol = 1e-8

        let (reason, stats) = conv.check(rnorm, bnorm, 3);

        assert_eq!(reason, ConvergedReason::ConvergedAtol);
        assert_eq!(stats.reason, ConvergedReason::ConvergedAtol);
        assert_eq!(stats.iterations, 3);
        assert_eq!(stats.final_residual, rnorm);
    }

    #[test]
    fn test_divergence_dtol() {
        let conv = Convergence::new(1e-6, 1e-12, 1e2, 100);
        let bnorm = 1.0;
        let rnorm = 150.0; // Greater than dtol * bnorm = 100

        let (reason, stats) = conv.check(rnorm, bnorm, 10);

        assert_eq!(reason, ConvergedReason::DivergedDtol);
        assert_eq!(stats.reason, ConvergedReason::DivergedDtol);
        assert_eq!(stats.iterations, 10);
        assert_eq!(stats.final_residual, rnorm);
    }

    #[test]
    fn test_divergence_max_iters() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 50);
        let bnorm = 1.0;
        let rnorm = 0.1; // Not converged but not diverged

        let (reason, stats) = conv.check(rnorm, bnorm, 50);

        assert_eq!(reason, ConvergedReason::DivergedMaxIts);
        assert_eq!(stats.reason, ConvergedReason::DivergedMaxIts);
        assert_eq!(stats.iterations, 50);
        assert_eq!(stats.final_residual, rnorm);
    }

    #[test]
    fn test_continued() {
        let conv = Convergence::new(1e-6, 1e-12, 1e3, 100);
        let bnorm = 1.0;
        let rnorm = 1e-3; // Not converged yet

        let (reason, stats) = conv.check(rnorm, bnorm, 10);

        assert_eq!(reason, ConvergedReason::Continued);
        assert_eq!(stats.reason, ConvergedReason::Continued);
        assert_eq!(stats.iterations, 10);
        assert_eq!(stats.final_residual, rnorm);
    }

    #[test]
    fn test_convergence_order_atol_first() {
        // Test that absolute tolerance is checked before relative tolerance
        let conv = Convergence::new(1e-3, 1e-6, 1e3, 100);
        let bnorm = 1.0;
        let rnorm = 1e-7; // Satisfies both atol and rtol, should be atol

        let (reason, _) = conv.check(rnorm, bnorm, 5);

        assert_eq!(reason, ConvergedReason::ConvergedAtol);
    }

    #[test]
    fn test_multiple_threshold_precedence() {
        // Test the order: atol > rtol > dtol > maxits
        let conv = Convergence::new(1e-3, 1e-4, 1e2, 10);

        // Case 1: Only atol satisfied (smallest residual)
        let (reason, _) = conv.check(1e-5, 1.0, 5);
        assert_eq!(reason, ConvergedReason::ConvergedAtol);

        // Case 2: Only rtol satisfied (residual between atol and rtol*bnorm)
        let (reason, _) = conv.check(5e-4, 1.0, 5); // 5e-4 > 1e-4 (atol) but < 1e-3 (rtol)
        assert_eq!(reason, ConvergedReason::ConvergedRtol);

        // Case 3: Divergence
        let (reason, _) = conv.check(200.0, 1.0, 5);
        assert_eq!(reason, ConvergedReason::DivergedDtol);

        // Case 4: Max iterations
        let (reason, _) = conv.check(0.1, 1.0, 10);
        assert_eq!(reason, ConvergedReason::DivergedMaxIts);
    }
}
