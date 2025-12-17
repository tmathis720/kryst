//! Example demonstrating explicit setup phase and workspace reuse for efficient repeated solves.
//!
//! This example shows how to use the two-phase API:
//! 1. Setup phase: Configure solver and allocate workspace once
//! 2. Solve phase: Efficiently solve multiple systems with the same matrix structure
//!
//! # Usage
//!
//! ```bash
//! cargo run --example setup_reuse_demo
//! ```

#[cfg(feature = "complex")]
fn main() {
    eprintln!("setup_reuse_demo is disabled when the complex feature is enabled.");
}

#[cfg(all(not(feature = "backend-faer"), not(feature = "complex")))]
fn main() {
    eprintln!("setup_reuse_demo requires the backend-faer feature.");
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::context::ksp_context::{KspContext, SolverType};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::context::pc_context::PcType;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::matrix::op::LinOp;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::time::Instant;

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn create_test_matrix(n: usize) -> Mat<f64> {
    // Create a symmetric positive definite tridiagonal matrix
    // This is a common pattern in finite difference discretizations
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            4.0 // Main diagonal
        } else if (i as isize - j as isize).abs() == 1 {
            -1.0 // Super/sub diagonals
        } else {
            0.0
        }
    })
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("KSP Context Setup & Workspace Reuse Demo");
    println!("========================================");
    println!();

    let n = 100; // Problem size
    let num_solves = 5; // Number of right-hand sides to solve

    // Create the system matrix (represents a discrete Laplacian)
    let a: Arc<dyn LinOp<S = f64>> = Arc::new(create_test_matrix(n));
    println!("Created {}×{} tridiagonal system matrix", n, n);

    // Create multiple right-hand side vectors
    let mut rhs_vectors = Vec::new();
    for i in 0..num_solves {
        let b: Vec<f64> = (0..n)
            .map(|j| ((i + 1) as f64) * (j as f64 + 1.0).sin())
            .collect();
        rhs_vectors.push(b);
    }
    println!("Generated {} right-hand side vectors", num_solves);
    println!();

    // =================================================================
    // Method 1: Without explicit setup (auto-setup on each solve)
    // =================================================================
    println!("Method 1: Auto-setup (less efficient)");
    println!("--------------------------------------");

    let mut ksp_auto = KspContext::new();
    ksp_auto
        .set_type(SolverType::Cg)?
        .set_pc_type(PcType::Jacobi, None)?
        .set_tolerances(1e-8, 1e-12, 1e3, 1000)
        .set_operators(a.clone(), None);

    let start_auto = Instant::now();
    for (i, b) in rhs_vectors.iter().enumerate() {
        let mut x = vec![0.0; n];

        // Each solve may trigger setup overhead
        let stats = ksp_auto.solve(b, &mut x)?;

        println!(
            "  Solve {}: {} iterations, residual = {:.2e}",
            i + 1,
            stats.iterations,
            stats.final_residual
        );
    }
    let auto_time = start_auto.elapsed();
    println!("  Total time: {:.3} ms", auto_time.as_secs_f64() * 1000.0);
    println!();

    // =================================================================
    // Method 2: With explicit setup (setup once, solve many)
    // =================================================================
    println!("Method 2: Explicit setup (more efficient)");
    println!("------------------------------------------");

    let mut ksp_explicit = KspContext::new();
    ksp_explicit
        .set_type(SolverType::Cg)?
        .set_pc_type(PcType::Jacobi, None)?
        .set_tolerances(1e-8, 1e-12, 1e3, 1000)
        .set_operators(a.clone(), None);

    // Explicit setup phase (done once)
    let setup_start = Instant::now();
    ksp_explicit.setup()?;
    let setup_time = setup_start.elapsed();
    println!("  Setup time: {:.3} ms", setup_time.as_secs_f64() * 1000.0);
    println!("  Setup status: {}", ksp_explicit.is_setup());

    // Solve phase (reuses workspace)
    let solve_start = Instant::now();
    for (i, b) in rhs_vectors.iter().enumerate() {
        let mut x = vec![0.0; n];

        // This solve reuses the preconditioner and workspace
        let stats = ksp_explicit.solve(b, &mut x)?;

        println!(
            "  Solve {}: {} iterations, residual = {:.2e}",
            i + 1,
            stats.iterations,
            stats.final_residual
        );
    }
    let explicit_solve_time = solve_start.elapsed();
    let explicit_total_time = setup_time + explicit_solve_time;

    println!(
        "  Solve time: {:.3} ms",
        explicit_solve_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Total time: {:.3} ms",
        explicit_total_time.as_secs_f64() * 1000.0
    );
    println!();

    // =================================================================
    // Performance comparison
    // =================================================================
    println!("Performance Comparison");
    println!("----------------------");
    println!(
        "Auto-setup method:     {:.3} ms",
        auto_time.as_secs_f64() * 1000.0
    );
    println!(
        "Explicit setup method: {:.3} ms",
        explicit_total_time.as_secs_f64() * 1000.0
    );

    if explicit_total_time < auto_time {
        let speedup = auto_time.as_secs_f64() / explicit_total_time.as_secs_f64();
        println!("Speedup: {:.2}×", speedup);
    } else {
        let slowdown = explicit_total_time.as_secs_f64() / auto_time.as_secs_f64();
        println!(
            "Slowdown: {:.2}× (setup overhead for small problems)",
            slowdown
        );
    }
    println!();

    // =================================================================
    // Demonstrate workspace invalidation and re-setup
    // =================================================================
    println!("Workspace Management");
    println!("--------------------");

    let mut ksp_mgmt = KspContext::new();
    ksp_mgmt
        .set_type(SolverType::Gmres)?
        .set_pc_type(PcType::None, None)?
        .set_operators(a.clone(), None);

    // Initial setup
    ksp_mgmt.setup()?;
    println!("Initial setup: {}", ksp_mgmt.is_setup());

    // Change restart parameter (invalidates workspace)
    ksp_mgmt.set_restart(20);
    println!("After changing restart: {}", ksp_mgmt.is_setup());

    // Re-setup with new parameters
    ksp_mgmt.setup()?;
    println!("After re-setup: {}", ksp_mgmt.is_setup());
    println!();

    println!("Demo completed successfully!");
    println!();
    println!("Key takeaways:");
    println!("1. Use explicit setup() for repeated solves with the same matrix");
    println!("2. Workspace is automatically allocated and reused");
    println!("3. Changing solver parameters invalidates workspace");

    Ok(())
}
