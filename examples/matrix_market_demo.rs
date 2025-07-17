//! Example demonstrating Matrix Market I/O with Kryst solvers.
//!
//! This example shows how to:
//! 1. Read a sparse matrix and RHS vector from Matrix Market files
//! 2. Convert them to Kryst formats
//! 3. Solve the linear system using a Krylov solver
//! 4. Write the solution back to a Matrix Market file

use kryst::utils::matrix_market::{read_matrix_market, write_vector_market};
use kryst::solver::{LinearSolver, GmresSolver};
use kryst::matrix::sparse::SparseMatrix;
use kryst::parallel::{UniverseComm, NoComm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging if available
    #[cfg(feature = "logging")]
    env_logger::init();

    println!("Matrix Market I/O Example");
    println!("=========================");

    // Read the sparse matrix
    println!("Reading matrix from examples/e05r0000/e05r0000.mtx...");
    let matrix_data = read_matrix_market("examples/e05r0000/e05r0000.mtx")?;
    println!("Matrix: {}x{} with {} non-zeros", 
             matrix_data.rows, matrix_data.cols, matrix_data.values.len());

    // Read the right-hand side vector
    println!("Reading RHS from examples/e05r0000/e05r0000_rhs1.mtx...");
    let rhs_data = read_matrix_market("examples/e05r0000/e05r0000_rhs1.mtx")?;
    println!("RHS: {}x{} vector", rhs_data.rows, rhs_data.cols);

    // Convert to Kryst formats
    let matrix = matrix_data.to_csr_matrix()?;
    let rhs = rhs_data.to_vector()?;
    let mut solution = vec![0.0; rhs.len()];

    println!("Converting to CSR matrix and dense vector...");
    println!("Matrix dimensions: {}x{}", matrix.nrows(), matrix.ncols());
    println!("RHS length: {}", rhs.len());

    // Set up the solver
    let mut solver = GmresSolver::new(30, 1e-6, 1000); // restart = 30, rtol = 1e-6, max_iters = 1000
    let comm = UniverseComm::NoComm(NoComm);

    println!("Solving linear system with GMRES (no preconditioner)...");

    // Solve the system
    let stats = solver.solve(&matrix, None, &rhs, &mut solution, &comm)?;
    
    println!("Solution completed!");
    println!("Iterations: {}", stats.iterations);
    println!("Final residual: {:.2e}", stats.final_residual);
    println!("Convergence reason: {:?}", stats.reason);

    // Write the solution to a Matrix Market file
    println!("Writing solution to solution.mtx...");
    write_vector_market("solution.mtx", &solution)?;

    // Display some solution statistics
    let solution_norm = solution.iter().map(|x| x * x).sum::<f64>().sqrt();
    let solution_max = solution.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let solution_min = solution.iter().fold(f64::INFINITY, |a, &b| a.min(b.abs()));

    println!("Solution statistics:");
    println!("  Norm: {:.6e}", solution_norm);
    println!("  Max absolute value: {:.6e}", solution_max);
    println!("  Min absolute value: {:.6e}", solution_min);

    // Verify the solution by computing residual
    let mut residual = rhs.clone();
    matrix.spmv(&solution, &mut residual);
    for (r, &b) in residual.iter_mut().zip(rhs.iter()) {
        *r = b - *r; // residual = b - A*x
    }
    
    let residual_norm = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
    let rhs_norm = rhs.iter().map(|x| x * x).sum::<f64>().sqrt();
    let relative_residual = residual_norm / rhs_norm;

    println!("Verification:");
    println!("  Residual norm: {:.6e}", residual_norm);
    println!("  RHS norm: {:.6e}", rhs_norm);
    println!("  Relative residual: {:.6e}", relative_residual);

    if relative_residual < 1e-6 {
        println!("✓ Solution verified successfully!");
    } else {
        println!("⚠ Solution verification failed - high residual");
    }

    println!("Example completed successfully!");
    Ok(())
}
