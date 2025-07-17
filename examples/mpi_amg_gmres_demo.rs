//! Large-scale MPI example demonstrating Matrix Market I/O with Jacobi-preconditioned GMRES.
//!
//! This example shows how to:
//! 1. Read a large sparse matrix and RHS vector from Matrix Market files
//! 2. Set up distributed parallel computation using MPI
//! 3. Construct a Jacobi preconditioner for efficient sparse preconditioning
//! 4. Solve the linear system using GMRES with Jacobi preconditioning
//! 5. Perform convergence analysis and write results
//!
//! Usage: cargo mpirun -n 4 --example mpi_amg_gmres_demo

use kryst::utils::matrix_market::{read_matrix_market, write_vector_market};
use kryst::solver::{LinearSolver, GmresSolver};
use kryst::preconditioner::{Jacobi, Preconditioner};
use kryst::matrix::sparse::SparseMatrix;
use kryst::parallel::{UniverseComm, Comm};
use std::time::Instant;

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize MPI and create communicator
    #[cfg(feature = "mpi")]
    let comm = UniverseComm::Mpi(MpiComm::new());
    #[cfg(not(feature = "mpi"))]
    let comm = UniverseComm::NoComm(kryst::parallel::NoComm);

    let rank = comm.rank();
    let size = comm.size();

    // Only rank 0 prints headers to avoid output clutter
    if rank == 0 {
        println!("Kryst MPI AMG-GMRES Demo");
        println!("========================");
        println!("Running on {} MPI processes", size);
        println!();
    }

    // Read the large sparse matrix (only rank 0 reads, then broadcasts)
    let start_io = Instant::now();
    
    let (matrix_data, rhs_data) = if rank == 0 {
        println!("Reading matrix from examples/e30r1000/e30r1000.mtx...");
        let matrix_data = read_matrix_market("examples/e30r1000/e30r1000.mtx")?;
        println!("Matrix: {}x{} with {} non-zeros", 
                 matrix_data.rows, matrix_data.cols, matrix_data.values.len());

        println!("Reading RHS from examples/e30r1000/e30r1000_rhs1.mtx...");
        let rhs_data = read_matrix_market("examples/e30r1000/e30r1000_rhs1.mtx")?;
        println!("RHS: {}x{} vector", rhs_data.rows, rhs_data.cols);
        
        (matrix_data, rhs_data)
    } else {
        // For this example, we'll have all processes read the data
        // In a real distributed implementation, you'd broadcast the data
        let matrix_data = read_matrix_market("examples/e30r1000/e30r1000.mtx")?;
        let rhs_data = read_matrix_market("examples/e30r1000/e30r1000_rhs1.mtx")?;
        (matrix_data, rhs_data)
    };

    // Convert to Kryst formats
    let matrix = matrix_data.to_csr_matrix()?;
    let rhs = rhs_data.to_vector()?;
    let mut solution = vec![0.0; rhs.len()];

    let io_time = start_io.elapsed();
    if rank == 0 {
        println!("I/O completed in {:.3}s", io_time.as_secs_f64());
        println!("Matrix dimensions: {}x{}", matrix.nrows(), matrix.ncols());
        println!("RHS length: {}", rhs.len());
        println!();
    }

    // Synchronize all processes before setup
    comm.barrier();

    // Set up Jacobi preconditioner
    let start_setup = Instant::now();
    if rank == 0 {
        println!("Setting up Jacobi preconditioner...");
    }

    // Create Jacobi preconditioner - simple but effective for many problems
    let mut jacobi = Jacobi::new();
    // Setup with the matrix to extract diagonal
    jacobi.setup(&matrix)?;

    let setup_time = start_setup.elapsed();
    if rank == 0 {
        println!("Jacobi setup completed in {:.3}s", setup_time.as_secs_f64());
    }

    // Set up GMRES solver with generous parameters for this large problem
    let restart = 50;          // GMRES restart parameter
    let rtol = 1e-8;          // Relative tolerance
    let max_iters = 2000;     // Maximum iterations
    let mut solver = GmresSolver::new(restart, rtol, max_iters);

    if rank == 0 {
        println!("Solver configuration:");
        println!("  GMRES restart: {}", restart);
        println!("  Relative tolerance: {:.1e}", rtol);
        println!("  Maximum iterations: {}", max_iters);
        println!();
    }

    comm.barrier();

    // Solve the system
    let start_solve = Instant::now();
    if rank == 0 {
        println!("Solving linear system with Jacobi-preconditioned GMRES...");
    }

    let stats = solver.solve(&matrix, Some(&jacobi), &rhs, &mut solution, &comm)?;
    
    let solve_time = start_solve.elapsed();

    // Print results (only rank 0)
    if rank == 0 {
        println!();
        println!("Solution completed!");
        println!("===================");
        println!("Solve time: {:.3}s", solve_time.as_secs_f64());
        println!("Iterations: {}", stats.iterations);
        println!("Final residual: {:.2e}", stats.final_residual);
        println!("Convergence reason: {:?}", stats.reason);
        println!();

        // Performance metrics
        let total_time = io_time + setup_time + solve_time;
        println!("Performance breakdown:");
        println!("  I/O time:    {:.3}s ({:.1}%)", io_time.as_secs_f64(), 100.0 * io_time.as_secs_f64() / total_time.as_secs_f64());
        println!("  Setup time:  {:.3}s ({:.1}%)", setup_time.as_secs_f64(), 100.0 * setup_time.as_secs_f64() / total_time.as_secs_f64());
        println!("  Solve time:  {:.3}s ({:.1}%)", solve_time.as_secs_f64(), 100.0 * solve_time.as_secs_f64() / total_time.as_secs_f64());
        println!("  Total time:  {:.3}s", total_time.as_secs_f64());
        println!();

        // Estimate performance metrics
        let nnz = matrix_data.values.len();
        let dof = matrix.nrows();
        println!("Problem characteristics:");
        println!("  Degrees of freedom: {}", dof);
        println!("  Non-zeros: {}", nnz);
        println!("  Fill factor: {:.2}", nnz as f64 / (dof * dof) as f64);
        println!("  Time per iteration: {:.1}ms", 1000.0 * solve_time.as_secs_f64() / stats.iterations as f64);
        println!("  MPI processes: {}", size);
        println!();
    }

    // Write solution (only rank 0)
    if rank == 0 {
        println!("Writing solution to mpi_jacobi_solution.mtx...");
        write_vector_market("mpi_jacobi_solution.mtx", &solution)?;
    }

    // Display solution statistics
    if rank == 0 {
        let solution_norm = solution.iter().map(|x| x * x).sum::<f64>().sqrt();
        let solution_max = solution.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let solution_min = solution.iter().filter(|&&x| x != 0.0).fold(f64::INFINITY, |a, &b| a.min(b.abs()));

        println!("Solution statistics:");
        println!("  Norm: {:.6e}", solution_norm);
        println!("  Max absolute value: {:.6e}", solution_max);
        println!("  Min non-zero absolute value: {:.6e}", solution_min);
        println!();
    }

    // Verify the solution by computing residual
    let mut residual = rhs.clone();
    matrix.spmv(&solution, &mut residual);
    for (r, &b) in residual.iter_mut().zip(rhs.iter()) {
        *r = b - *r; // residual = b - A*x
    }
    
    // Compute norms using parallel reduction
    let local_residual_sq = residual.iter().map(|x| x * x).sum::<f64>();
    let local_rhs_sq = rhs.iter().map(|x| x * x).sum::<f64>();
    
    // All-reduce to get global norms
    let global_residual_sq = comm.all_reduce_f64(local_residual_sq);
    let global_rhs_sq = comm.all_reduce_f64(local_rhs_sq);
    
    let residual_norm = global_residual_sq.sqrt();
    let rhs_norm = global_rhs_sq.sqrt();
    let relative_residual = residual_norm / rhs_norm;

    if rank == 0 {
        println!("Verification (MPI-parallel residual computation):");
        println!("  Residual norm: {:.6e}", residual_norm);
        println!("  RHS norm: {:.6e}", rhs_norm);
        println!("  Relative residual: {:.6e}", relative_residual);
        println!();

        if relative_residual < 1e-6 {
            println!("✓ Solution verified successfully!");
        } else if relative_residual < 1e-3 {
            println!("⚠ Solution marginally acceptable (high residual)");
        } else {
            println!("❌ Solution verification failed - high residual");
        }
        
        println!();
        println!("Example completed successfully!");
    }

    // Final barrier to ensure all processes complete together
    comm.barrier();

    Ok(())
}
