//! Example demonstrating PETSc-style command-line options for configuring KSP solvers.
//!
//! This example shows how to use the options database to configure solvers and
//! preconditioners at runtime using command-line arguments, similar to PETSc.
//!
//! # Usage
//!
//! ```bash
//! # Run with default settings
//! cargo run --example options_demo
//!
//! # Configure CG solver with Jacobi preconditioner
//! cargo run --example options_demo -- -ksp_type cg -ksp_rtol 1e-8 -pc_type jacobi
//!
//! # Configure GMRES with custom restart
//! cargo run --example options_demo -- -ksp_type gmres -ksp_gmres_restart 100 -pc_type ilu0
//!
//! # Use direct LU solver (no Krylov iteration)
//! cargo run --example options_demo --features backend-faer,dense-direct,mpi -- -ksp_type preonly -pc_type lu
//!
//! # Use direct QR solver (for least squares problems)
//! cargo run --example options_demo --features backend-faer,dense-direct,mpi -- -ksp_type preonly -pc_type qr
//!
//! # Show all available options
//! cargo run --example options_demo -- -help
//! ```
#[cfg(feature = "complex")]
fn main() {
    eprintln!("options_demo is disabled when the complex feature is enabled.");
}

#[cfg(all(not(feature = "backend-faer"), not(feature = "complex")))]
#[cfg(not(feature = "complex"))]
fn main() {
    eprintln!("options_demo requires the backend-faer feature.");
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::config::options::parse_all_options;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::context::ksp_context::KspContext;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::env;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    let (ksp_opts, pc_opts) = parse_all_options(&args)?;

    println!("Kryst Options Demo");
    println!("==================");

    // Show what options were parsed
    println!("Parsed KSP options: {:?}", ksp_opts);
    println!("Parsed PC options: {:?}", pc_opts);
    println!();

    // Create and configure KSP context
    let mut ksp = KspContext::new();
    ksp.set_from_all_options(&ksp_opts, &pc_opts)?;

    println!("KSP Context Configuration:");
    println!("  Relative tolerance: {}", ksp.rtol);
    println!("  Absolute tolerance: {}", ksp.atol);
    println!("  Divergence tolerance: {}", ksp.dtol);
    println!("  Maximum iterations: {}", ksp.maxits);
    println!("  GMRES restart: {}", ksp.restart);
    println!();

    // Create a simple test problem: 2x2 SPD system
    // [4  1] [x1]   [1]
    // [1  3] [x2] = [2]
    // Solution: x1 ≈ 0.09090909, x2 ≈ 0.63636364
    let a = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 4.0,
        (0, 1) | (1, 0) => 1.0,
        (1, 1) => 3.0,
        _ => 0.0,
    });

    let b = vec![1.0, 2.0];
    let mut x = vec![0.0, 0.0]; // Initial guess

    println!("Solving test system:");
    println!("Matrix A:");
    for i in 0..2 {
        print!("  [");
        for j in 0..2 {
            print!("{:8.3}", a[(i, j)]);
        }
        println!(" ]");
    }
    println!("Right-hand side b: {:?}", b);
    println!("Initial guess x: {:?}", x);
    println!();

    // Solve the system
    ksp.set_operators(Arc::new(a.clone()), None);
    match ksp.solve(&b, &mut x) {
        Ok(stats) => {
            println!("Solution Results:");
            println!(
                "  Converged: {}",
                matches!(
                    stats.reason,
                    kryst::utils::convergence::ConvergedReason::ConvergedRtol
                        | kryst::utils::convergence::ConvergedReason::ConvergedAtol
                )
            );
            println!("  Iterations: {}", stats.iterations);
            println!("  Final residual norm: {:.2e}", stats.final_residual);
            println!("  Solution x: [{:.8}, {:.8}]", x[0], x[1]);

            // Verify solution by computing residual
            let mut r = vec![0.0, 0.0];
            for i in 0..2 {
                r[i] = b[i];
                for j in 0..2 {
                    r[i] -= a[(i, j)] * x[j];
                }
            }
            let residual_norm: f64 = (r[0] * r[0] + r[1] * r[1]).sqrt();
            println!("  Verification residual: {:.2e}", residual_norm);

            if residual_norm < 1e-10 {
                println!("  ✓ Solution is accurate!");
            } else {
                println!("  ⚠ Solution may not be accurate");
            }
        }
        Err(e) => {
            println!("Solver failed: {}", e);
            return Err(e.into());
        }
    }

    println!();
    println!("Demo completed successfully!");
    println!();
    println!("Try different options:");
    println!("  cargo run --example options_demo -- -ksp_type gmres -ksp_rtol 1e-12");
    println!("  cargo run --example options_demo -- -ksp_type bicgstab -pc_type jacobi");
    println!("  cargo run --example options_demo -- -ksp_type preonly -pc_type lu");
    println!("  cargo run --example options_demo -- -ksp_type preonly -pc_type qr");
    println!("  cargo run --example options_demo -- -help");

    Ok(())
}
