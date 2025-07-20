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

    // Try to read from example files, but generate test data if they don't exist
    let (matrix_data, rhs_data) = match (
        read_matrix_market("examples/e05r0000/e05r0000.mtx"),
        read_matrix_market("examples/e05r0000/e05r0000_rhs1.mtx")
    ) {
        (Ok(matrix), Ok(rhs)) => {
            println!("Reading matrix from examples/e05r0000/e05r0000.mtx...");
            println!("Matrix: {}x{} with {} non-zeros", 
                     matrix.rows, matrix.cols, matrix.values.len());
            println!("Reading RHS from examples/e05r0000/e05r0000_rhs1.mtx...");
            (matrix, rhs)
        },
        _ => {
            println!("Example Matrix Market files not found, generating test data...");
            // Generate a simple test matrix (5x5 tridiagonal)
            let n = 5;
            let mut matrix = kryst::utils::matrix_market::MatrixMarketData {
                rows: n,
                cols: n,
                nonzeros: 2 * n - 1, // diagonal + off-diagonals
                values: Vec::new(),
                row_indices: Vec::new(),
                col_indices: Vec::new(),
                is_symmetric: true,
                is_coordinate: true,
            };
            
            // Create tridiagonal matrix: [2 -1 0 0 0; -1 2 -1 0 0; ...]
            for i in 0..n {
                // Diagonal
                matrix.values.push(2.0);
                matrix.row_indices.push(i);
                matrix.col_indices.push(i);
                
                // Super-diagonal
                if i < n - 1 {
                    matrix.values.push(-1.0);
                    matrix.row_indices.push(i);
                    matrix.col_indices.push(i + 1);
                }
                
                // Sub-diagonal (if not symmetric)
                if !matrix.is_symmetric && i > 0 {
                    matrix.values.push(-1.0);
                    matrix.row_indices.push(i);
                    matrix.col_indices.push(i - 1);
                }
            }
            
            let rhs = kryst::utils::matrix_market::MatrixMarketData {
                rows: n,
                cols: 1,
                nonzeros: n,
                values: vec![1.0; n],
                row_indices: (0..n).collect(),
                col_indices: vec![0; n],
                is_symmetric: false,
                is_coordinate: true,
            };
            
            println!("Generated {}x{} tridiagonal matrix with {} non-zeros", 
                     matrix.rows, matrix.cols, matrix.values.len());
            (matrix, rhs)
        }
    };
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
