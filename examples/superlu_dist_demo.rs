//! Comprehensive SuperLU_DIST distributed direct solver demonstration.
//!
//! This example demonstrates the complete SuperLU_DIST distributed sparse direct
//! factorization and solve functionality using real-world matrices from the
//! Matrix Market collection.
//!
//! **Recommended Usage:**
//!   `cargo mpirun -n 4 --features mpi,logging --example superlu_dist_demo`
//!
//! **Alternative Usage:**
//!   `cargo mpirun -n 2 --features mpi,logging --example superlu_dist_demo`
//!   `cargo mpirun -n 8 --features mpi,logging --example superlu_dist_demo -- --matrix fidap005`
//!   `cargo run --features logging --example superlu_dist_demo`  (serial mode)
//!
//! **Features Demonstrated:**
//! - Matrix Market I/O for real-world problems
//! - Phase 7 builder pattern and fluent configuration
//! - Command-line options integration
//! - MPI distributed factorization and solve
//! - Performance analysis and timing comparisons
//! - Error analysis and iterative refinement
//! - Memory usage profiling with StageGuard
//!
//! **Test Matrix:** fidapm11 (FIDAP model, structural problem, 22,294 x 22,294, 623,554 nnz)
#[cfg(feature = "complex")]
fn main() {
    eprintln!("superlu_dist_demo.rs is unavailable when built with --features complex");
}

#[cfg(all(not(feature = "backend-faer"), not(feature = "complex")))]
fn main() {
    eprintln!("superlu_dist_demo requires the backend-faer feature.");
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::parallel::UniverseComm;
#[cfg(feature = "superlu_dist")]
use kryst::solver::SuperLuDistSolver;
#[cfg(feature = "superlu_dist")]
use kryst::solver::superlu_dist::{
    ColumnPermutation, IterativeRefinement, RefinementConfig, ResidualMethod, RowPermutation,
    SuperLuDistBuilder,
};

#[cfg(not(feature = "mpi"))]
use kryst::parallel::NoComm;
use std::env;

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let matrix_name = parse_matrix_name(&args);
    let enable_analysis = parse_flag(&args, "--analysis");
    let enable_refinement = parse_flag(&args, "--refinement");

    // Create communicator
    #[cfg(feature = "mpi")]
    let comm = {
        let mpi_comm = MpiComm::new();
        let rank = mpi_comm.rank();
        let size = mpi_comm.size();
        if rank == 0 {
            println!("SuperLU_DIST Distributed Direct Solver Demo");
            println!("==========================================");
            println!("Running on {} MPI processes", size);
            println!(
                "Matrix: {}, Analysis: {}, Refinement: {}",
                matrix_name, enable_analysis, enable_refinement
            );
            println!();
        }
        UniverseComm::Mpi(std::sync::Arc::new(mpi_comm))
    };

    #[cfg(not(feature = "mpi"))]
    let comm = {
        println!("SuperLU_DIST Distributed Direct Solver Demo (Serial)");
        println!("===================================================");
        println!("Running in serial mode (MPI not available)");
        println!(
            "Matrix: {}, Analysis: {}, Refinement: {}",
            matrix_name, enable_analysis, enable_refinement
        );
        println!();
        UniverseComm::NoComm(NoComm)
    };

    // Example 1: Load real Matrix Market data
    #[cfg(feature = "superlu_dist")]
    example_matrix_market_solve(&comm, &matrix_name)?;

    // Example 2: Builder pattern demonstration with Phase 7 API
    #[cfg(feature = "superlu_dist")]
    example_builder_pattern(&comm, &matrix_name)?;

    // Example 3: Performance analysis with different configurations
    if enable_analysis {
        #[cfg(feature = "superlu_dist")]
        example_performance_analysis(&comm, &matrix_name)?;
    }

    // Example 4: Iterative refinement analysis
    if enable_refinement {
        #[cfg(feature = "superlu_dist")]
        example_refinement_analysis(&comm, &matrix_name)?;
    }

    // Only rank 0 prints final message
    match &comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => {
            if mpi_comm.rank() == 0 {
                println!("Demo completed successfully!");
                println!("For detailed performance analysis, run with --analysis flag");
                println!("For refinement testing, run with --refinement flag");
            }
        }
        _ => {
            println!("Demo completed successfully!");
            println!("For detailed performance analysis, run with --analysis flag");
            println!("For refinement testing, run with --refinement flag");
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[allow(dead_code)]
struct MatrixInfo {
    name: String,
    matrix_file: String,
    rhs_file: String,
    description: String,
}

/// Parse matrix name from command line arguments
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn parse_matrix_name(args: &[String]) -> String {
    for i in 0..args.len() {
        if args[i] == "--matrix" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    "fidapm11".to_string() // Default to fidapm11
}

/// Parse boolean flag from command line arguments
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn parse_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

/// Get available matrices for testing
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[allow(dead_code)]
fn get_available_matrices() -> Vec<MatrixInfo> {
    vec![
        MatrixInfo {
            name: "fidapm11".to_string(),
            matrix_file: "examples/mtx/fidapm11.mtx".to_string(),
            rhs_file: "examples/mtx/fidapm11_rhs1.mtx".to_string(),
            description: "FIDAP model (structural, 22,294 x 22,294, 623,554 nnz)".to_string(),
        },
        MatrixInfo {
            name: "fidap005".to_string(),
            matrix_file: "examples/mtx/fidap005.mtx".to_string(),
            rhs_file: "examples/mtx/fidap005_rhs1.mtx".to_string(),
            description: "FIDAP model (smaller test case)".to_string(),
        },
        MatrixInfo {
            name: "sherman5".to_string(),
            matrix_file: "examples/mtx/sherman5.mtx".to_string(),
            rhs_file: "examples/mtx/sherman5_rhs1.mtx".to_string(),
            description: "Sherman matrix (oil reservoir simulation)".to_string(),
        },
    ]
}

/// Load and solve a real Matrix Market problem
#[cfg(feature = "superlu_dist")]
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn example_matrix_market_solve(
    comm: &UniverseComm,
    matrix_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let is_rank_0 = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => mpi_comm.rank() == 0,
        _ => true,
    };

    if is_rank_0 {
        println!("1. Matrix Market Problem: {}", matrix_name);
        println!("{}={}", "=".repeat(25), "=".repeat(matrix_name.len()));
    }

    // Find the requested matrix
    let matrices = get_available_matrices();
    let matrix_info = matrices
        .iter()
        .find(|m| m.name == matrix_name)
        .ok_or_else(|| {
            format!(
                "Matrix '{}' not found. Available: {:?}",
                matrix_name,
                matrices.iter().map(|m| &m.name).collect::<Vec<_>>()
            )
        })?;

    if is_rank_0 {
        println!("Description: {}", matrix_info.description);
        println!("Loading matrix from: {}", matrix_info.matrix_file);
        println!("Loading RHS from: {}", matrix_info.rhs_file);
    }

    // Load matrix and RHS
    let matrix_data = read_matrix_market(&matrix_info.matrix_file)?;
    let rhs_data = read_matrix_market(&matrix_info.rhs_file)?;

    let matrix = matrix_data.to_csr_matrix()?;
    let b = rhs_data.to_vector()?;
    let mut x = vec![0.0; b.len()];

    if is_rank_0 {
        println!(
            "Matrix: {}x{} with {} non-zeros",
            matrix.nrows(),
            matrix.ncols(),
            matrix.nnz()
        );
        println!("RHS vector: {} elements", b.len());
        println!(
            "Density: {:.4}%",
            100.0 * matrix.nnz() as f64 / (matrix.nrows() * matrix.ncols()) as f64
        );
    }

    // Determine optimal process grid for MPI
    let process_grid = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => {
            let size = mpi_comm.size();
            let prows = (size as f64).sqrt().floor() as usize;
            let pcols = size / prows;
            Some((prows.max(1), pcols.max(1)))
        }
        _ => Some((1, 1)),
    };

    if is_rank_0 {
        println!("Process grid: {:?}", process_grid);
    }

    // Create solver with default options but set process grid
    let mut solver = SuperLuDistSolver::new();
    if let Some((rows, cols)) = process_grid {
        solver.set_process_grid(rows, cols);
    }

    // Solve the system
    if is_rank_0 {
        println!("Starting factorization and solve...");
    }

    let start = Instant::now();
    let stats = solver.solve(&matrix, None, &b, &mut x, PcSide::Left, comm, None, None)?;
    let solve_time = start.elapsed();

    if is_rank_0 {
        println!("✓ Solution computed successfully!");
        println!("  Total time: {:.3}s", solve_time.as_secs_f64());
        println!("  Iterations: {} (direct solver)", stats.iterations);
        println!("  Convergence: {:?}", stats.reason);

        // Basic solution analysis
        let solution_norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rhs_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        println!("  ||x||_2 = {:.6e}", solution_norm);
        println!("  ||b||_2 = {:.6e}", rhs_norm);

        // Show first few solution components
        let show_count = std::cmp::min(5, x.len());
        println!("  x[0:{}] = {:?}", show_count, &x[0..show_count]);
        println!();
    }

    Ok(())
}

/// Demonstrate Phase 7 builder pattern and fluent
#[cfg(feature = "superlu_dist")]
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn example_builder_pattern(
    comm: &UniverseComm,
    _matrix_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let is_rank_0 = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => mpi_comm.rank() == 0,
        _ => true,
    };

    if is_rank_0 {
        println!("2. Phase 7 Builder Pattern and Fluent API");
        println!("=========================================");
    }

    // Find the requested matrix (reuse smaller matrix for builder demo)
    let matrices = get_available_matrices();
    let matrix_info = matrices
        .iter()
        .find(|m| m.name == "fidapm05") // Use smaller matrix for detailed demo
        .unwrap_or(&matrices[0]);

    if is_rank_0 {
        println!("Using matrix: {}", matrix_info.description);
    }

    // Load matrix and RHS
    let matrix_data = read_matrix_market(&matrix_info.matrix_file)?;
    let rhs_data = read_matrix_market(&matrix_info.rhs_file)?;

    let matrix = matrix_data.to_csr_matrix()?;
    let b = rhs_data.to_vector()?;
    let mut x = vec![0.0; b.len()];

    // Demonstrate builder pattern with fluent configuration
    if is_rank_0 {
        println!("Building solver with fluent API...");
    }

    let process_grid = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => {
            let size = mpi_comm.size();
            let prows = (size as f64).sqrt().floor() as usize;
            let pcols = size / prows;
            (prows.max(1), pcols.max(1))
        }
        _ => (1, 1),
    };

    // Build solver using Phase 7 fluent API
    let mut solver = SuperLuDistBuilder::new()
        .column_permutation(ColumnPermutation::Metis)
        .row_permutation(RowPermutation::LargeDiag)
        .diagonal_pivot_threshold(0.01)
        .iterative_refinement(IterativeRefinement::Double)
        .process_grid(process_grid.0, process_grid.1)
        .print_level(if is_rank_0 { 1 } else { 0 })
        .replace_tiny_pivots(true)
        .static_pivoting(false)
        .refinement_config(RefinementConfig {
            max_iterations: 3,
            tolerance: 1e-12,
            relative_tolerance: 1e-10,
            min_improvement_factor: 0.9,
        })
        .residual_method(ResidualMethod::Scaled)
        .build();

    if is_rank_0 {
        println!("✓ Solver configured with advanced options:");
        println!(
            "  Column permutation: {:?}",
            solver.options().column_permutation
        );
        println!("  Row permutation: {:?}", solver.options().row_permutation);
        println!(
            "  Diagonal pivot threshold: {}",
            solver.options().diagonal_pivot_threshold
        );
        println!(
            "  Iterative refinement: {:?}",
            solver.options().iterative_refinement
        );
        println!("  Process grid: ({}, {})", process_grid.0, process_grid.1);
        println!("  Print level: {}", solver.options().print_level);
    }

    // Solve with advanced configuration
    let start = Instant::now();
    let stats = solver.solve(&matrix, None, &b, &mut x, PcSide::Left, comm, None, None)?;
    let solve_time = start.elapsed();

    if is_rank_0 {
        println!("✓ Solution with advanced configuration:");
        println!("  Total time: {:.3}s", solve_time.as_secs_f64());
        println!("  Convergence: {:?}", stats.reason);

        // Check if refinement was performed
        if let Some(ref_stats) = solver.refinement_stats() {
            println!("  Refinement iterations: {}", ref_stats.iterations);
            println!(
                "  Initial residual: {:.2e}",
                ref_stats.initial_residual_norm
            );
            println!("  Final residual: {:.2e}", ref_stats.final_residual_norm);
            println!(
                "  Improvement factor: {:.2e}",
                ref_stats.initial_residual_norm / ref_stats.final_residual_norm.max(1e-16)
            );
        }

        // Memory usage information
        if let Some(mem_stats) = solver.workspace_memory_stats() {
            println!(
                "  Total memory usage: {:.2} MB",
                mem_stats.total_memory as f64 / (1024.0 * 1024.0)
            );
            println!("  Temp vectors: {}", mem_stats.temp_vectors_count);
        }

        println!();
    }

    Ok(())
}

/// Demonstrate performance analysis with different configurations
#[cfg(feature = "superlu_dist")]
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn example_performance_analysis(
    comm: &UniverseComm,
    matrix_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let is_rank_0 = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => mpi_comm.rank() == 0,
        _ => true,
    };

    if is_rank_0 {
        println!("3. Performance Analysis with Different Configurations");
        println!("===================================================");
    }

    // Find the requested matrix
    let matrices = get_available_matrices();
    let matrix_info = matrices
        .iter()
        .find(|m| m.name == matrix_name)
        .unwrap_or(&matrices[0]);

    if is_rank_0 {
        println!("Using matrix: {}", matrix_info.description);
    }

    // Load matrix and RHS
    let matrix_data = read_matrix_market(&matrix_info.matrix_file)?;
    let rhs_data = read_matrix_market(&matrix_info.rhs_file)?;

    let matrix = matrix_data.to_csr_matrix()?;
    let b = rhs_data.to_vector()?;

    let process_grid = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => {
            let size = mpi_comm.size();
            let prows = (size as f64).sqrt().floor() as usize;
            let pcols = size / prows;
            (prows.max(1), pcols.max(1))
        }
        _ => (1, 1),
    };

    if is_rank_0 {
        println!("Process grid: ({}, {})", process_grid.0, process_grid.1);
        println!();
    }

    // Test different configurations
    let configurations = vec![
        (
            "Default",
            SuperLuDistBuilder::new()
                .process_grid(process_grid.0, process_grid.1)
                .build(),
        ),
        (
            "METIS+NoRowPerm",
            SuperLuDistBuilder::new()
                .column_permutation(ColumnPermutation::Metis)
                .row_permutation(RowPermutation::NoRowPerm)
                .process_grid(process_grid.0, process_grid.1)
                .build(),
        ),
        (
            "Conservative+Refinement",
            SuperLuDistBuilder::new()
                .column_permutation(ColumnPermutation::Metis)
                .diagonal_pivot_threshold(0.001)
                .iterative_refinement(IterativeRefinement::Double)
                .process_grid(process_grid.0, process_grid.1)
                .build(),
        ),
        (
            "Aggressive+Static",
            SuperLuDistBuilder::new()
                .column_permutation(ColumnPermutation::Metis)
                .diagonal_pivot_threshold(0.1)
                .static_pivoting(true)
                .process_grid(process_grid.0, process_grid.1)
                .build(),
        ),
    ];

    for (config_name, mut solver) in configurations {
        if is_rank_0 {
            println!("Testing configuration: {}", config_name);
        }

        let mut x = vec![0.0; b.len()];

        let start = Instant::now();
        let stats = solver.solve(&matrix, None, &b, &mut x, PcSide::Left, comm, None, None)?;
        let solve_time = start.elapsed();

        if is_rank_0 {
            println!("  Time: {:.3}s", solve_time.as_secs_f64());
            println!("  Convergence: {:?}", stats.reason);

            // Solution quality analysis
            let solution_norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            println!("  ||x||_2: {:.6e}", solution_norm);

            // Memory usage if available
            if let Some(mem_stats) = solver.workspace_memory_stats() {
                println!(
                    "  Total memory: {:.2} MB",
                    mem_stats.total_memory as f64 / (1024.0 * 1024.0)
                );
            }

            // Refinement stats if available
            if let Some(ref_stats) = solver.refinement_stats() {
                println!("  Refinement iters: {}", ref_stats.iterations);
                println!(
                    "  Residual reduction: {:.2e}",
                    ref_stats.initial_residual_norm / ref_stats.final_residual_norm.max(1e-16)
                );
            }

            println!();
        }
    }

    if is_rank_0 {
        println!("Performance Insights:");
        println!("• METIS ordering typically reduces fill-in for sparse matrices");
        println!("• Lower pivot thresholds improve stability but may increase solve time");
        println!("• Iterative refinement improves accuracy at computational cost");
        println!("• Static pivoting can be faster but less numerically stable");
        println!("• Optimal configuration depends on matrix properties and accuracy requirements");
        println!();
    }

    Ok(())
}

/// Demonstrate iterative refinement analysis
#[cfg(feature = "superlu_dist")]
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn example_refinement_analysis(
    comm: &UniverseComm,
    matrix_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let is_rank_0 = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => mpi_comm.rank() == 0,
        _ => true,
    };

    if is_rank_0 {
        println!("4. Iterative Refinement Analysis");
        println!("===============================");
    }

    // Find the requested matrix
    let matrices = get_available_matrices();
    let matrix_info = matrices
        .iter()
        .find(|m| m.name == matrix_name)
        .unwrap_or(&matrices[0]);

    if is_rank_0 {
        println!("Using matrix: {}", matrix_info.description);
    }

    // Load matrix and RHS
    let matrix_data = read_matrix_market(&matrix_info.matrix_file)?;
    let rhs_data = read_matrix_market(&matrix_info.rhs_file)?;

    let matrix = matrix_data.to_csr_matrix()?;
    let b = rhs_data.to_vector()?;

    let process_grid = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => {
            let size = mpi_comm.size();
            let prows = (size as f64).sqrt().floor() as usize;
            let pcols = size / prows;
            (prows.max(1), pcols.max(1))
        }
        _ => (1, 1),
    };

    // Test different refinement configurations
    let refinement_configs = vec![
        ("No Refinement", IterativeRefinement::NoRefine),
        ("Single Refinement", IterativeRefinement::Single),
        ("Double Refinement", IterativeRefinement::Double),
    ];

    for (config_name, refinement_type) in refinement_configs {
        if is_rank_0 {
            println!("Testing: {}", config_name);
        }

        let mut solver = SuperLuDistBuilder::new()
            .column_permutation(ColumnPermutation::Metis)
            .iterative_refinement(refinement_type)
            .process_grid(process_grid.0, process_grid.1)
            .refinement_config(RefinementConfig {
                max_iterations: 5,
                tolerance: 1e-14,
                relative_tolerance: 1e-12,
                min_improvement_factor: 0.95,
            })
            .residual_method(ResidualMethod::Scaled)
            .build();

        let mut x = vec![0.0; b.len()];

        let start = Instant::now();
        let _stats = solver.solve(&matrix, None, &b, &mut x, PcSide::Left, comm, None, None)?;
        let solve_time = start.elapsed();

        if is_rank_0 {
            println!("  Time: {:.3}s", solve_time.as_secs_f64());

            // Detailed refinement analysis
            if let Some(ref_stats) = solver.refinement_stats() {
                println!("  Refinement iterations: {}", ref_stats.iterations);
                println!(
                    "  Initial residual: {:.6e}",
                    ref_stats.initial_residual_norm
                );
                println!("  Final residual: {:.6e}", ref_stats.final_residual_norm);
                println!("  Convergence: {:?}", ref_stats.convergence_reason);
                println!("  Time in refinement: {:.3}s", ref_stats.refinement_time);

                if ref_stats.residual_history.len() > 1 {
                    println!("  Residual history:");
                    for (i, &residual) in ref_stats.residual_history.iter().enumerate() {
                        println!("    Iter {}: {:.6e}", i, residual);
                    }
                }
            } else {
                println!("  No refinement performed");
            }

            // Compute actual residual for verification
            let mut residual = vec![0.0; b.len()];
            matrix.spmv(&x, &mut residual);
            for (r, &b_val) in residual.iter_mut().zip(&b) {
                *r = (*r - b_val).abs();
            }
            let residual_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
            println!("  Actual ||Ax - b||_2: {:.6e}", residual_norm);

            println!();
        }
    }

    if is_rank_0 {
        println!("Refinement Insights:");
        println!("• Iterative refinement can significantly improve solution accuracy");
        println!("• Double precision refinement uses extended precision arithmetics");
        println!("• Convergence depends on matrix conditioning and pivot quality");
        println!("• Monitor residual reduction to assess refinement effectiveness");
        println!("• Balance accuracy gains against computational overhead");
        println!();
    }

    Ok(())
}
