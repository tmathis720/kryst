//! Example demonstrating flexible convergence and divergence tests.
//!
//! This example shows how to:
//! 1. Use the default convergence criteria (rtol, atol, dtol, maxits)
//! 2. Set custom convergence tests for specialized stopping criteria
//! 3. Inspect convergence reasons in solve statistics
//!
//! # Usage
//! 
//! ```bash
//! cargo run --example convergence_demo
//! ```

use faer::Mat;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::utils::convergence::ConvergedReason;

fn create_test_matrix(n: usize) -> Mat<f64> {
    // Create a symmetric positive definite tridiagonal matrix
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            4.0  // Main diagonal
        } else if (i as isize - j as isize).abs() == 1 {
            -1.0  // Super/sub diagonals
        } else {
            0.0
        }
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Flexible Convergence & Divergence Tests Demo");
    println!("============================================");
    println!();

    let n = 50;
    let a = create_test_matrix(n);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    // =================================================================
    // Example 1: Default convergence criteria
    // =================================================================
    println!("Example 1: Default Convergence Criteria");
    println!("---------------------------------------");

    let mut ksp1 = KspContext::new();
    ksp1.set_type(SolverType::Cg)?
        .set_pc_type(PcType::Jacobi)?
        .set_tolerances(1e-8, 1e-12, 1e3, 1000);

    let mut x1 = vec![0.0; n];
    let stats1 = ksp1.solve(&a, &b, &mut x1)?;

    println!("Default criteria result:");
    println!("  Reason: {:?}", stats1.reason);
    println!("  Iterations: {}", stats1.iterations);
    println!("  Final residual: {:.2e}", stats1.final_residual);
    println!();

    // =================================================================
    // Example 2: Custom convergence test - Early stopping
    // =================================================================
    println!("Example 2: Custom Convergence Test - Early Stopping");
    println!("---------------------------------------------------");

    let mut ksp2 = KspContext::new();
    ksp2.set_type(SolverType::Cg)?
        .set_pc_type(PcType::Jacobi)?;

    // Custom test: stop early if we get "good enough" convergence
    ksp2.set_convergence_test(|iters, rnorm, bnorm| {
        let rel_res = rnorm / bnorm;
        if rel_res < 1e-4 {
            println!("  Custom: Early convergence at iteration {} (rel_res = {:.2e})", iters, rel_res);
            ConvergedReason::ConvergedRtol
        } else if iters >= 20 {
            println!("  Custom: Stopping at iteration limit {}", iters);
            ConvergedReason::DivergedMaxIts
        } else {
            ConvergedReason::Continued
        }
    });

    let mut x2 = vec![0.0; n];
    let stats2 = ksp2.solve(&a, &b, &mut x2)?;

    println!("Custom early stopping result:");
    println!("  Reason: {:?}", stats2.reason);
    println!("  Iterations: {}", stats2.iterations);
    println!("  Final residual: {:.2e}", stats2.final_residual);
    println!();

    // =================================================================
    // Example 3: Custom convergence test - Stagnation detection
    // =================================================================
    println!("Example 3: Custom Convergence Test - Stagnation Detection");
    println!("---------------------------------------------------------");

    let mut ksp3 = KspContext::new();
    ksp3.set_type(SolverType::Cg)?
        .set_pc_type(PcType::None)?;

    // Custom test: detect when convergence stagnates
    use std::cell::Cell;
    let stagnation_threshold = 1e-2; // If relative improvement < 1%

    ksp3.set_convergence_test({
        let prev_residual = Cell::new(f64::INFINITY);
        move |iters, rnorm, bnorm| {
            let rel_res = rnorm / bnorm;
            if rel_res < 1e-6 {
                ConvergedReason::ConvergedRtol
            } else if iters > 5 {
                let prev = prev_residual.get();
                let improvement = (prev - rnorm) / prev;
                if improvement < stagnation_threshold {
                    println!("  Custom: Stagnation detected at iteration {} (improvement = {:.1e})", 
                             iters, improvement);
                    ConvergedReason::DivergedDtol  // Using dtol to indicate stagnation
                } else {
                    prev_residual.set(rnorm);
                    ConvergedReason::Continued
                }
            } else {
                prev_residual.set(rnorm);
                ConvergedReason::Continued
            }
        }
    });

    let mut x3 = vec![0.0; n];
    let stats3 = ksp3.solve(&a, &b, &mut x3)?;

    println!("Stagnation detection result:");
    println!("  Reason: {:?}", stats3.reason);
    println!("  Iterations: {}", stats3.iterations);
    println!("  Final residual: {:.2e}", stats3.final_residual);
    println!();

    // =================================================================
    // Example 4: Multiple thresholds demonstration
    // =================================================================
    println!("Example 4: Multiple Thresholds Demonstration");
    println!("--------------------------------------------");

    // Test different threshold scenarios
    let test_cases = vec![
        ("Very tight absolute tolerance", 1e-16, 1e-6, 1e3, 1000),
        ("Loose relative tolerance", 1e-2, 1e-12, 1e3, 1000),
        ("Low divergence threshold", 1e-6, 1e-12, 10.0, 1000),
        ("Few max iterations", 1e-6, 1e-12, 1e3, 5),
    ];

    for (name, rtol, atol, dtol, maxits) in test_cases {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Cg)?
           .set_pc_type(PcType::None)?
           .set_tolerances(rtol, atol, dtol, maxits);

        let mut x = vec![0.0; n];
        let stats = ksp.solve(&a, &b, &mut x)?;

        println!("  {}: {:?} ({} iters, res = {:.2e})", 
                 name, stats.reason, stats.iterations, stats.final_residual);
    }
    println!();

    // =================================================================
    // Example 5: Clearing custom convergence test
    // =================================================================
    println!("Example 5: Clearing Custom Convergence Test");
    println!("-------------------------------------------");

    let mut ksp5 = KspContext::new();
    ksp5.set_type(SolverType::Cg)?
        .set_pc_type(PcType::Jacobi)?
        .set_tolerances(1e-8, 1e-12, 1e3, 1000);

    // Set a custom test
    ksp5.set_convergence_test(|_iters, _rnorm, _bnorm| {
        ConvergedReason::DivergedMaxIts  // Always diverge (for demo)
    });

    // Clear it and use default
    ksp5.clear_convergence_test();

    let mut x5 = vec![0.0; n];
    let stats5 = ksp5.solve(&a, &b, &mut x5)?;

    println!("After clearing custom test (should converge normally):");
    println!("  Reason: {:?}", stats5.reason);
    println!("  Iterations: {}", stats5.iterations);
    println!("  Final residual: {:.2e}", stats5.final_residual);
    println!();

    println!("Demo completed successfully!");
    println!();
    println!("Key takeaways:");
    println!("1. Default convergence uses rtol, atol, dtol, and maxits thresholds");
    println!("2. Custom convergence tests allow specialized stopping criteria");
    println!("3. ConvergedReason enum provides detailed information about why solving stopped");
    println!("4. Custom tests can implement early stopping, stagnation detection, etc.");
    println!("5. Use clear_convergence_test() to revert to default behavior");

    Ok(())
}
