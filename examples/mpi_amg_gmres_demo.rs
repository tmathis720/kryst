#![cfg(feature = "mpi")]
//! Large-scale MPI example demonstrating Matrix Market I/O with configurable solvers and preconditioners.
//!
//! This example shows how to:
//! 1. Read a large sparse matrix and RHS vector from Matrix Market files
//! 2. Set up distributed parallel computation using MPI
//! 3. Use the unified KspContext for runtime solver and preconditioner selection
//! 4. Solve the linear system using the configured iterative solver
//! 5. Perform convergence analysis and write results
//!
//! Usage:
//!   cargo mpirun -n 4 --example mpi_amg_gmres_demo [options]
//!   
//! PETSc-style Options:
//!   -ksp_type <solver>         Solver type (cg, pcg, gmres, bicgstab, cgs, qmr, tfqmr, minres, cgnr, preonly)
//!   -pc_type <precond>         Preconditioner type (jacobi, ilu0, none, ilu, ilut, ilup, blockjacobi, sor, asm, chebyshev, amg, approxinverse, lu, qr)
//!   -ksp_rtol <tol>            Relative tolerance [default: 1e-5]
//!   -ksp_atol <tol>            Absolute tolerance [default: 1e-50]
//!   -ksp_dtol <tol>            Divergence tolerance [default: 1e5]
//!   -ksp_max_it <iters>        Maximum iterations [default: 10000]
//!   -ksp_gmres_restart <n>     GMRES restart parameter [default: 50]
//!   -ksp_pc_side <side>        Preconditioning side (left, right, symmetric) [default: left]
//!   -matrix <path>             Matrix file path [default: examples/e05r0300/e05r0300.mtx]
//!   -rhs <path>                RHS vector file path [default: examples/e05r0300/e05r0300_rhs1.mtx]
//!   -help                      Show all available options

#[cfg(feature = "complex")]
fn main() {
    eprintln!("mpi_amg_gmres_demo.rs is unavailable when built with --features complex");
}



use kryst::config::options::parse_all_options;
use kryst::context::ksp_context::KspContext;
use kryst::matrix::op::{CsrOp, wrap_with_comm};
use kryst::matrix::sparse::SparseMatrix;
use kryst::parallel::{Comm, UniverseComm};
use kryst::utils::matrix_market::{read_matrix_market, write_vector_market};
use std::env;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;

#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize MPI and create communicator
    #[cfg(feature = "mpi")]
    let comm = UniverseComm::Mpi(std::sync::Arc::new(MpiComm::new()));
    #[cfg(not(feature = "mpi"))]
    let comm = UniverseComm::NoComm(kryst::parallel::NoComm);

    let rank = comm.rank();
    let size = comm.size();

    // Parse command line arguments using the existing options system
    let args: Vec<String> = env::args().collect();
    let (ksp_opts, pc_opts) = parse_all_options(&args)?;

    // Create KspContext and configure from options
    let mut ksp = KspContext::new();

    // Set solver type (default: gmres - more robust for general problems)
    let solver_type = ksp_opts.ksp_type.as_deref().unwrap_or("gmres");
    ksp.set_type_from_str(solver_type)?;

    // Set preconditioner type (default: ilu0 - more robust than jacobi)
    let pc_type = pc_opts.pc_type.as_deref().unwrap_or("ilu0");
    ksp.set_pc_type_from_str(pc_type)?;

    // Set tolerances and iteration limits (more conservative defaults)
    let rtol = ksp_opts.rtol.unwrap_or(1e-8); // Tighter tolerance for better convergence
    let atol = ksp_opts.atol.unwrap_or(1e-12);
    let dtol = ksp_opts.dtol.unwrap_or(1e5); // More lenient divergence threshold
    let max_iters = ksp_opts.maxits.unwrap_or(2000); // More iterations for difficult problems
    ksp.set_tolerances(rtol, atol, dtol, max_iters);

    // Set restart parameter for GMRES-type solvers (larger restart for better convergence)
    let restart = ksp_opts.restart.unwrap_or(50);
    ksp.set_restart(restart);

    // Set preconditioning side (default to left preconditioning)
    let pc_side = ksp_opts.pc_side.as_deref().unwrap_or("left");
    ksp.set_pc_side_from_str(pc_side)?;

    // Set up file paths
    let matrix_file = ksp_opts
        .matrix_file
        .as_deref()
        .unwrap_or("examples/e05r0300/e05r0300.mtx");
    let rhs_file = ksp_opts
        .rhs_file
        .as_deref()
        .unwrap_or("examples/e05r0300/e05r0300_rhs1.mtx");

    // Only rank 0 prints headers to avoid output clutter
    if rank == 0 {
        println!("Kryst MPI Unified KSP Context Demo");
        println!("===================================");
        println!("Running on {} MPI processes", size);
        println!("Configuration:");
        println!("  Solver: {}", solver_type);
        println!("  Preconditioner: {}", pc_type);
        println!("  Relative tolerance: {:.1e}", rtol);
        println!("  Absolute tolerance: {:.1e}", atol);
        println!("  Divergence tolerance: {:.1e}", dtol);
        println!("  Max iterations: {}", max_iters);
        if solver_type.contains("gmres") || solver_type == "fgmres" {
            println!("  GMRES restart: {}", restart);
        }
        println!("  PC side: {}", pc_side);
        println!("  Matrix file: {}", matrix_file);
        println!("  RHS file: {}", rhs_file);
        println!();
    }

    // Read the sparse matrix (only rank 0 reads, then broadcasts)
    let start_io = Instant::now();

    let (matrix_data, rhs_data) = if rank == 0 {
        println!("Reading matrix from {}...", matrix_file);
        let matrix_data = read_matrix_market(matrix_file)?;
        println!(
            "Matrix: {}x{} with {} non-zeros",
            matrix_data.rows,
            matrix_data.cols,
            matrix_data.values.len()
        );

        println!("Reading RHS from {}...", rhs_file);
        let rhs_data = read_matrix_market(rhs_file)?;
        println!("RHS: {}x{} vector", rhs_data.rows, rhs_data.cols);

        (matrix_data, rhs_data)
    } else {
        // For this example, we'll have all processes read the data
        // In a real distributed implementation, you'd broadcast the data
        let matrix_data = read_matrix_market(matrix_file)?;
        let rhs_data = read_matrix_market(rhs_file)?;
        (matrix_data, rhs_data)
    };

    // Convert to Kryst formats
    let matrix = matrix_data.to_csr_matrix()?;
    let rhs = rhs_data.to_vector()?;
    let mut solution = vec![0.0; rhs.len()];

    // Wrap CSR matrix in an Arc and create a CsrOp (avoid densification)
    let csr_arc = Arc::new(matrix);
    let csr_op = CsrOp::new(csr_arc.clone());

    let io_time = start_io.elapsed();
    if rank == 0 {
        println!("I/O completed in {:.3}s", io_time.as_secs_f64());
        println!(
            "Matrix dimensions: {}x{}",
            matrix_data.rows, matrix_data.cols
        );
        println!("RHS length: {}", rhs.len());
        println!();
    }

    // Synchronize all processes before setup
    // Note: We cannot create a new MpiComm instance because MPI can only be initialized once
    // Instead, we'll use barriers through the existing comm after we give it to KSP setup
    // For now, skip this barrier since we'll have one after setup anyway

    // Set up the KSP context (preconditioner setup, etc.)
    let start_setup = Instant::now();

    if rank == 0 {
        println!(
            "Setting up {} preconditioner and {} solver...",
            pc_type, solver_type
        );
    }

    // Set up monitoring callback for convergence tracking
    let monitor_data = Arc::new(Mutex::new(Vec::<(usize, f64)>::new()));
    let monitor_data_clone = monitor_data.clone();

    let monitor = Box::new(move |iter: usize, residual: f64| {
        if let Ok(mut data) = monitor_data_clone.lock() {
            data.push((iter, residual));
            // Print every 10th iteration for rank 0 to avoid spam
            if rank == 0 && (iter == 0 || iter % 10 == 0 || residual < rtol) {
                println!("    Iteration {:4}: residual = {:.6e}", iter, residual);
            }
        }
    });

    // Setup KSP with monitoring and workspace optimization
    ksp.add_monitor(monitor);

    // Enable profiling if available
    #[cfg(feature = "logging")]
    {
        env_logger::init();
        if rank == 0 {
            println!("Profiling enabled - detailed timing information will be logged");
        }
    }

    // Attach operator and set up KSP, attaching the communicator
    let op_arc: Arc<dyn kryst::matrix::op::LinOp<S = f64>> =
        wrap_with_comm(Arc::new(csr_op), comm.clone());
    ksp.set_operators_with_comm(op_arc, None, comm.clone());
    ksp.setup()?;

    let setup_time = start_setup.elapsed();
    if rank == 0 {
        println!("KSP setup completed in {:.3}s", setup_time.as_secs_f64());
        println!();
    }

    // Barrier before solving (use our UniverseComm)
    comm.barrier();

    // Solve the system using the unified KSP context
    let start_solve = Instant::now();
    if rank == 0 {
        println!(
            "Solving linear system with {}-preconditioned {}...",
            pc_type,
            solver_type.to_uppercase()
        );
        println!("Convergence history:");
    }

    let stats = ksp.solve(&rhs, &mut solution)?;

    let solve_time = start_solve.elapsed();

    // Extract monitoring data for analysis
    let convergence_history = if let Ok(data) = monitor_data.lock() {
        data.clone()
    } else {
        Vec::new()
    };

    // Print results (only rank 0)
    if rank == 0 {
        println!();
        println!("Solution completed!");
        println!("===================");
        println!("Solve time: {:.3}s", solve_time.as_secs_f64());
        println!("Iterations: {}", stats.iterations);
        println!("Final residual: {:.2e}", stats.final_residual);
        println!("Convergence reason: {:?}", stats.reason);

        // Analyze convergence history
        if !convergence_history.is_empty() {
            let initial_residual = convergence_history[0].1;
            let final_residual = convergence_history.last().unwrap().1;
            let reduction_factor = final_residual / initial_residual;

            println!("Convergence analysis:");
            println!("  Initial residual: {:.2e}", initial_residual);
            println!("  Final residual: {:.2e}", final_residual);
            println!("  Reduction factor: {:.2e}", reduction_factor);

            if convergence_history.len() > 1 {
                let avg_reduction =
                    reduction_factor.powf(1.0 / (convergence_history.len() - 1) as f64);
                println!("  Average reduction per iteration: {:.3}", avg_reduction);
            }
        }

        println!();

        // Performance metrics
        let total_time = io_time + setup_time + solve_time;
        println!("Performance breakdown:");
        println!(
            "  I/O time:    {:.3}s ({:.1}%)",
            io_time.as_secs_f64(),
            100.0 * io_time.as_secs_f64() / total_time.as_secs_f64()
        );
        println!(
            "  Setup time:  {:.3}s ({:.1}%)",
            setup_time.as_secs_f64(),
            100.0 * setup_time.as_secs_f64() / total_time.as_secs_f64()
        );
        println!(
            "  Solve time:  {:.3}s ({:.1}%)",
            solve_time.as_secs_f64(),
            100.0 * solve_time.as_secs_f64() / total_time.as_secs_f64()
        );
        println!("  Total time:  {:.3}s", total_time.as_secs_f64());
        println!();

        // Estimate performance metrics
        let nnz = matrix_data.values.len();
        let dof = matrix_data.rows;
        let time_per_iter = if stats.iterations > 0 {
            solve_time.as_secs_f64() / stats.iterations as f64
        } else {
            0.0
        };

        println!("Problem characteristics:");
        println!("  Degrees of freedom: {}", dof);
        println!("  Non-zeros: {}", nnz);
        println!("  Fill factor: {:.4}", nnz as f64 / (dof * dof) as f64);
        println!("  Time per iteration: {:.1}ms", 1000.0 * time_per_iter);
        println!("  MPI processes: {}", size);
        println!(
            "  Solver efficiency: {:.0} DOF/s",
            dof as f64 / solve_time.as_secs_f64()
        );

        println!();
    }

    // Write solution (only rank 0)
    if rank == 0 {
        println!("Writing solution to mpi_ksp_solution.mtx...");
        write_vector_market("mpi_ksp_solution.mtx", &solution)?;
    }

    // Display solution statistics
    if rank == 0 {
        let solution_norm = solution.iter().map(|x| x * x).sum::<f64>().sqrt();
        let solution_max = solution.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let solution_min = solution
            .iter()
            .filter(|&&x| x != 0.0)
            .fold(f64::INFINITY, |a, &b| a.min(b.abs()));

        println!("Solution statistics:");
        println!("  Norm: {:.6e}", solution_norm);
        println!("  Max absolute value: {:.6e}", solution_max);
        println!("  Min non-zero absolute value: {:.6e}", solution_min);
        println!();
    }

    // Verify the solution by computing residual A*x - b
    let mut ax = vec![0.0; rhs.len()];
    csr_arc.spmv(&solution, &mut ax);

    let mut residual = rhs.clone();
    for (r, &ax_val) in residual.iter_mut().zip(ax.iter()) {
        *r = *r - ax_val; // residual = b - A*x
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
