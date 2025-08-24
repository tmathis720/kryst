//! Example demonstrating iteration monitoring and logging features.
//!
//! This example shows how to:
//! 1. Register iteration monitors to track solver progress
//! 2. Enable trace logging to see internal solver operations
//! 3. Use different solver types with monitoring
//!
//! Run with: RUST_LOG=trace cargo run --example monitor --features=logging

use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::error::KError;
use kryst::matrix::op::LinOp;
use faer::Mat;
use std::sync::{Arc, Mutex};

#[cfg(feature = "logging")]
use log::{info, debug};

fn main() -> Result<(), KError> {
    // Initialize logger if logging feature is enabled
    #[cfg(feature = "logging")]
    env_logger::init();
    
    #[cfg(feature = "logging")]
    info!("Starting KSP monitoring example");

    // Create a simple test problem: 3x3 SPD matrix
    let n = 3;
    let mut a_data = vec![0.0; n * n];
    a_data[0] = 4.0; a_data[1] = -1.0; a_data[2] = 0.0;   // [4 -1  0]
    a_data[3] = -1.0; a_data[4] = 4.0; a_data[5] = -1.0;  // [-1 4 -1]  
    a_data[6] = 0.0; a_data[7] = -1.0; a_data[8] = 4.0;   // [0 -1  4]
    
    let a = Mat::from_fn(n, n, |i, j| a_data[i * n + j]);
    let a_op: Arc<dyn LinOp<S = f64>> = Arc::new(a.clone());
    let b = vec![3.0, 2.0, 3.0];
    let mut x = vec![0.0; n];
    
    println!("Solving Ax = b where:");
    println!("A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]");
    println!("b = [3, 2, 3]");
    println!();

    // Example 1: Simple iteration monitor
    println!("=== Example 1: Simple CG with iteration monitor ===");
    let mut ksp1 = KspContext::new();
    
    // Register a simple monitor that prints iteration and residual
    ksp1.add_monitor(|iter, residual| {
        println!("  Iter {}: residual = {:.3e}", iter, residual);
    });
    
    ksp1.set_operators(a_op.clone(), None)
        .set_type(SolverType::Cg)?
        .set_pc_type(PcType::None, None)?
        .set_tolerances(1e-6, 1e-12, 1e3, 100);

    let stats1 = ksp1.solve(&b, &mut x)?;
    println!("Solution: x = [{:.6}, {:.6}, {:.6}]", x[0], x[1], x[2]);
    println!("Converged in {} iterations with final residual {:.3e}", 
             stats1.iterations, stats1.final_residual);
    println!();

    // Example 2: Multiple monitors with different purposes
    println!("=== Example 2: GMRES with multiple monitors ===");
    let mut ksp2 = KspContext::new();
    
    // Shared data between monitors using Arc<Mutex<>>
    let max_residual = Arc::new(Mutex::new(0.0f64));
    let max_residual_clone = Arc::clone(&max_residual);
    
    // Monitor 1: Track maximum residual seen
    ksp2.add_monitor(move |_iter, residual| {
        let mut max_res = max_residual_clone.lock().unwrap();
        if residual > *max_res {
            *max_res = residual;
        }
    });
    
    // Monitor 2: Print progress every iteration
    ksp2.add_monitor(|iter, residual| {
        if iter % 1 == 0 || residual < 1e-6 {
            println!("  GMRES iter {}: ||r|| = {:.3e}", iter, residual);
        }
    });
    
    // Monitor 3: Check for stagnation using Arc<Mutex<>>
    let prev_residual = Arc::new(Mutex::new(std::f64::INFINITY));
    let prev_residual_clone = Arc::clone(&prev_residual);
    ksp2.add_monitor(move |iter, residual| {
        if iter > 0 {
            let mut prev_res = prev_residual_clone.lock().unwrap();
            let reduction = *prev_res / residual;
            if reduction < 1.01 && residual > 1e-8 {
                println!("  WARNING: Slow convergence at iter {}: reduction factor = {:.3}", 
                         iter, reduction);
            }
            *prev_res = residual;
        } else {
            let mut prev_res = prev_residual_clone.lock().unwrap();
            *prev_res = residual;
        }
    });
    
    x.fill(0.0); // Reset solution
    ksp2
        .set_operators(a_op.clone(), None)
        .set_type(SolverType::Gmres)?
        .set_pc_type(PcType::Jacobi, None)?;
    ksp2.set_restart(10);
    ksp2.set_tolerances(1e-8, 1e-12, 1e3, 100);

    #[cfg(feature = "logging")]
    debug!("Starting GMRES solve with Jacobi preconditioning");

    let stats2 = ksp2.solve(&b, &mut x)?;
    println!("Solution: x = [{:.6}, {:.6}, {:.6}]", x[0], x[1], x[2]);
    println!("Converged in {} iterations with final residual {:.3e}", 
             stats2.iterations, stats2.final_residual);
    
    let final_max_residual = *max_residual.lock().unwrap();
    println!("Maximum residual during iteration: {:.3e}", final_max_residual);
    println!();

    // Example 3: Monitor count and clearing
    println!("=== Example 3: Monitor management ===");
    let mut ksp3 = KspContext::new();
    
    println!("Initial monitor count: {}", ksp3.num_monitors());
    
    ksp3.add_monitor(|iter, residual| {
        println!("  Monitor A: iter={}, res={:.2e}", iter, residual);
    });
    
    ksp3.add_monitor(|iter, residual| {
        println!("  Monitor B: iter={}, res={:.2e}", iter, residual);
    });
    
    println!("After adding 2 monitors: {}", ksp3.num_monitors());
    
    ksp3.clear_monitors();
    println!("After clearing monitors: {}", ksp3.num_monitors());
    
    // Add a single final monitor
    ksp3.add_monitor(|iter, residual| {
        println!("  Final monitor: iter={}, res={:.3e}", iter, residual);
    });
    
    x.fill(0.0); // Reset solution
    ksp3.set_operators(a_op, None)
        .set_type(SolverType::BiCgStab)?
        .set_pc_type(PcType::None, None)?
        .set_tolerances(1e-6, 1e-12, 1e3, 100);

    let stats3 = ksp3.solve(&b, &mut x)?;
    println!("BiCGStab converged in {} iterations", stats3.iterations);
    println!();

    #[cfg(feature = "logging")]
    info!("All monitoring examples completed successfully");
    
    println!("=== Logging Instructions ===");
    println!("To see detailed internal logging, run with:");
    println!("  RUST_LOG=trace cargo run --example monitor --features=logging");
    println!("  RUST_LOG=debug cargo run --example monitor --features=logging");
    println!("  RUST_LOG=info cargo run --example monitor --features=logging");
    
    Ok(())
}
