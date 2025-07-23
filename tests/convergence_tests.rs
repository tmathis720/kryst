//! Tests for flexible convergence and divergence criteria

use kryst::utils::convergence::{Convergence, ConvergedReason};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

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
    fn test_custom_convergence_test() {
        // Test custom convergence test functionality in KspContext
        let mut ksp = KspContext::new();
        
        // Set a custom convergence test
        ksp.set_convergence_test(|iters, rnorm, bnorm| {
            if rnorm / bnorm < 1e-3 {
                ConvergedReason::ConvergedRtol
            } else if iters >= 5 {
                ConvergedReason::DivergedMaxIts
            } else {
                ConvergedReason::Continued
            }
        });
        
        // Verify the custom test was set
        assert!(ksp.has_custom_convergence_test());
        
        // Test clearing the custom test
        ksp.clear_convergence_test();
        assert!(!ksp.has_custom_convergence_test());
    }

    #[test]
    fn test_ksp_context_with_custom_convergence() -> Result<(), Box<dyn std::error::Error>> {
        // Create a simple test problem
        let n = 3;
        let a = Mat::from_fn(n, n, |i, j| {
            if i == j { 4.0 } else if (i as isize - j as isize).abs() == 1 { -1.0 } else { 0.0 }
        });
        let b = vec![3.0, 2.0, 3.0];
        let mut x = vec![0.0; n];
        
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Cg)?
           .set_pc_type(PcType::None)?;
        
        // Set a lenient custom convergence test
        ksp.set_convergence_test(|iters, rnorm, bnorm| {
            if rnorm / bnorm < 1e-2 {  // Very lenient
                ConvergedReason::ConvergedRtol
            } else if iters >= 100 {
                ConvergedReason::DivergedMaxIts
            } else {
                ConvergedReason::Continued
            }
        });
        
        // Solve the system
        let stats = ksp.solve(&a, &b, &mut x)?;
        
        // The custom convergence test should have been applied
        assert!(matches!(stats.reason, ConvergedReason::ConvergedRtol | ConvergedReason::DivergedMaxIts));
        
        Ok(())
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
