//! Example demonstrating the SuperLU_DIST distributed direct solver with MPI.
//!
//! This example shows how to use the SuperLU_DIST solver for distributed sparse direct
//! factorization and solve in an MPI environment. While this implementation is currently 
//! a placeholder (since we don't have the actual SuperLU_DIST library linked), it 
//! demonstrates the interface and usage patterns for distributed direct solvers.
//!
//! To run with MPI:
//!   `cargo mpirun -n 2 --example superlu_dist_demo`
//!   `cargo mpirun -n 4 --example superlu_dist_demo -- --size large`
//!   `cargo mpirun -n 8 --example superlu_dist_demo -- --grid-size 10`
//!
//! To run serially:
//!   `cargo run --example superlu_dist_demo`

use kryst::solver::{LinearSolver, SuperLuDistSolver};
use kryst::solver::superlu_dist::{
    SuperLuDistOptions, ColumnPermutation, IterativeRefinement, RowPermutation
};
use kryst::matrix::sparse::{CsrMatrix, SparseMatrix};
use kryst::parallel::{UniverseComm, Comm};

#[cfg(not(feature = "mpi"))]
use kryst::parallel::NoComm;
use std::time::Instant;
use std::env;

#[cfg(feature = "mpi")]
use kryst::parallel::MpiComm;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let problem_size = parse_problem_size(&args);
    let grid_size = parse_grid_size(&args);
    
    // Create communicator
    #[cfg(feature = "mpi")]
    let comm = {
        let mpi_comm = MpiComm::new();
        let rank = mpi_comm.rank();
        let size = mpi_comm.size();
        if rank == 0 {
            println!("SuperLU_DIST Distributed Direct Solver Demo (MPI)");
            println!("=================================================");
            println!("Running on {} MPI processes", size);
            println!("Problem size: {:?}, Grid size: {}", problem_size, grid_size);
            println!();
        }
        UniverseComm::Mpi(mpi_comm)
    };
    
    #[cfg(not(feature = "mpi"))]
    let comm = {
        println!("SuperLU_DIST Distributed Direct Solver Demo (Serial)");
        println!("===================================================");
        println!("Running in serial mode (MPI not available)");
        println!("Problem size: {:?}, Grid size: {}", problem_size, grid_size);
        println!();
        UniverseComm::NoComm(NoComm)
    };

    // Example 1: Basic usage with default options
    example_basic_usage(&comm, problem_size)?;
    
    // Example 2: Advanced options configuration
    example_advanced_options(&comm, grid_size)?;
    
    // Example 3: Performance comparison scenarios
    example_performance_scenarios(&comm)?;
    
    // Only rank 0 prints final message
    match &comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => {
            if mpi_comm.rank() == 0 {
                println!("Demo completed successfully!");
            }
        },
        _ => println!("Demo completed successfully!"),
    }
    
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum ProblemSize {
    Small,
    Medium,
    Large,
}

/// Parse problem size from command line arguments
fn parse_problem_size(args: &[String]) -> ProblemSize {
    for i in 0..args.len() {
        if args[i] == "--size" && i + 1 < args.len() {
            match args[i + 1].as_str() {
                "small" => return ProblemSize::Small,
                "medium" => return ProblemSize::Medium,
                "large" => return ProblemSize::Large,
                _ => {}
            }
        }
    }
    ProblemSize::Medium // Default
}

/// Parse grid size from command line arguments
fn parse_grid_size(args: &[String]) -> usize {
    for i in 0..args.len() {
        if args[i] == "--grid-size" && i + 1 < args.len() {
            if let Ok(size) = args[i + 1].parse::<usize>() {
                return size;
            }
        }
    }
    4 // Default
}

/// Demonstrate basic SuperLU_DIST usage with default options
fn example_basic_usage(comm: &UniverseComm, problem_size: ProblemSize) -> Result<(), Box<dyn std::error::Error>> {
    let is_rank_0 = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => mpi_comm.rank() == 0,
        _ => true,
    };
    
    if is_rank_0 {
        println!("1. Basic SuperLU_DIST Usage");
        println!("---------------------------");
    }
    
    // Create a sparse matrix based on problem size
    let n = match problem_size {
        ProblemSize::Small => 5,
        ProblemSize::Medium => 10,
        ProblemSize::Large => 20,
    };
    
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..n {
        // Subdiagonal
        if i > 0 {
            col_idx.push(i - 1);
            values.push(-1.0);
        }
        
        // Diagonal
        col_idx.push(i);
        values.push(4.0);
        
        // Superdiagonal
        if i < n - 1 {
            col_idx.push(i + 1);
            values.push(-1.0);
        }
        
        row_ptr.push(col_idx.len());
    }
    
    let matrix = CsrMatrix::from_csr(n, n, row_ptr, col_idx, values);
    
    if is_rank_0 {
        println!("Matrix: {}x{} tridiagonal with {} non-zeros", 
                 matrix.nrows(), matrix.ncols(), matrix.nnz());
    }
    
    // Create right-hand side vector
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let mut x = vec![0.0; n];
    
    // Create solver with default options
    let mut solver = SuperLuDistSolver::new();
    
    // Solve the system
    let start = Instant::now();
    let stats = solver.solve(&matrix, None, &b, &mut x, comm, None, None)?;
    let solve_time = start.elapsed();
    
    if is_rank_0 {
        println!("Solution: {:?}", x);
        println!("Solve time: {:.3}ms", solve_time.as_millis());
        println!("Iterations: {} (direct solver)", stats.iterations);
        println!("Convergence: {:?}", stats.reason);
        println!();
    }
    
    Ok(())
}

/// Demonstrate advanced SuperLU_DIST options configuration
fn example_advanced_options(comm: &UniverseComm, grid_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    let is_rank_0 = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => mpi_comm.rank() == 0,
        _ => true,
    };
    
    if is_rank_0 {
        println!("2. Advanced SuperLU_DIST Options");
        println!("--------------------------------");
    }
    
    // Create a Poisson 2D matrix based on grid size
    let matrix = create_poisson_2d_matrix(grid_size)?;
    let n = matrix.nrows();
    
    if is_rank_0 {
        println!("Matrix: {}x{} Poisson 2D with {} non-zeros", 
                 matrix.nrows(), matrix.ncols(), matrix.nnz());
    }
    
    // Create right-hand side
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let mut x = vec![0.0; n];
    
    // Determine optimal process grid for MPI
    let process_grid = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => {
            let size = mpi_comm.size();
            let prows = (size as f64).sqrt().floor() as usize;
            let pcols = size / prows;
            Some((prows.max(1), pcols.max(1)))
        },
        _ => Some((1, 1)),
    };
    
    // Configure advanced options
    let options = SuperLuDistOptions {
        process_grid,
        column_permutation: ColumnPermutation::Metis,
        diagonal_pivot_threshold: 0.1,
        replace_tiny_pivots: true,
        iterative_refinement: IterativeRefinement::Double,
        print_level: if is_rank_0 { 1 } else { 0 }, // Only rank 0 prints
        static_pivoting: false,
        row_permutation: RowPermutation::LargeDiag,
    };
    
    let mut solver = SuperLuDistSolver::with_options(options);
    
    // Alternative: configure via method chaining
    solver.set_diagonal_pivot_threshold(0.1)
          .set_column_permutation(ColumnPermutation::Metis)
          .set_iterative_refinement(IterativeRefinement::Double)
          .set_print_level(if is_rank_0 { 1 } else { 0 });
    
    let start = Instant::now();
    let stats = solver.solve(&matrix, None, &b, &mut x, comm, None, None)?;
    let solve_time = start.elapsed();
    
    if is_rank_0 {
        println!("Advanced configuration results:");
        println!("  Process grid: {:?}", process_grid);
        println!("  Column permutation: {:?}", solver.options().column_permutation);
        println!("  Iterative refinement: {:?}", solver.options().iterative_refinement);
        println!("  Diagonal pivot threshold: {}", solver.options().diagonal_pivot_threshold);
        println!("  Solution computed in {:.3}ms", solve_time.as_millis());
        println!("  Convergence: {:?}", stats.reason);
        println!();
    }
    
    Ok(())
}

/// Demonstrate performance considerations for different problem types
fn example_performance_scenarios(comm: &UniverseComm) -> Result<(), Box<dyn std::error::Error>> {
    let is_rank_0 = match comm {
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(mpi_comm) => mpi_comm.rank() == 0,
        _ => true,
    };
    
    if is_rank_0 {
        println!("3. Performance Scenarios");
        println!("------------------------");
    }
    
    let scenarios = vec![
        ("Identity", create_identity_matrix(10)),
        ("Tridiagonal", create_tridiagonal_matrix(10)),
        ("Poisson 2D", create_poisson_2d_matrix(3)), // 3x3 grid = 9x9 matrix
    ];
    
    for (name, matrix_result) in scenarios {
        if is_rank_0 {
            println!("Testing {} matrix:", name);
        }
        
        let matrix = matrix_result?;
        let n = matrix.nrows();
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        
        // Determine process grid for this configuration
        let process_grid = match comm {
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(mpi_comm) => {
                let size = mpi_comm.size();
                let prows = (size as f64).sqrt().floor() as usize;
                let pcols = size / prows;
                Some((prows.max(1), pcols.max(1)))
            },
            _ => Some((1, 1)),
        };
        
        // Test different configurations
        let configs = vec![
            ("Default", SuperLuDistOptions::default()),
            ("METIS+Refinement", SuperLuDistOptions {
                process_grid,
                column_permutation: ColumnPermutation::Metis,
                iterative_refinement: IterativeRefinement::Double,
                print_level: 0, // Disable printing for performance tests
                ..SuperLuDistOptions::default()
            }),
            ("High Threshold", SuperLuDistOptions {
                process_grid,
                diagonal_pivot_threshold: 0.001,
                replace_tiny_pivots: true,
                print_level: 0,
                ..SuperLuDistOptions::default()
            }),
        ];
        
        for (config_name, mut options) in configs {
            // Ensure process grid is set properly
            options.process_grid = process_grid;
            
            let mut solver = SuperLuDistSolver::with_options(options);
            
            let start = Instant::now();
            let stats = solver.solve(&matrix, None, &b, &mut x, comm, None, None)?;
            let solve_time = start.elapsed();
            
            if is_rank_0 {
                println!("  {}: {:.3}ms, converged: {:?}", 
                         config_name, solve_time.as_millis(), stats.reason);
            }
        }
        if is_rank_0 {
            println!();
        }
    }
    
    if is_rank_0 {
        println!("Performance Analysis:");
        println!("  • SuperLU_DIST is designed for large distributed sparse systems");
        println!("  • For small problems, overhead may dominate solve time");
        println!("  • METIS ordering typically improves factorization quality");
        println!("  • Iterative refinement improves accuracy at the cost of extra work");
        println!("  • Adjust diagonal pivot threshold based on matrix conditioning");
        println!("  • Process grid should be chosen to balance load and communication");
        
        match comm {
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(mpi_comm) => {
                println!("  • Current MPI configuration: {} processes", mpi_comm.size());
            },
            _ => {
                println!("  • Running in serial mode (no MPI)");
            }
        }
        println!();
    }
    
    Ok(())
}

/// Create an identity matrix for testing
fn create_identity_matrix(n: usize) -> Result<CsrMatrix<f64>, Box<dyn std::error::Error>> {
    let row_ptr: Vec<usize> = (0..=n).collect();
    let col_idx: Vec<usize> = (0..n).collect();
    let values = vec![1.0; n];
    
    Ok(CsrMatrix::from_csr(n, n, row_ptr, col_idx, values))
}

/// Create a tridiagonal matrix for testing
fn create_tridiagonal_matrix(n: usize) -> Result<CsrMatrix<f64>, Box<dyn std::error::Error>> {
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..n {
        // Subdiagonal
        if i > 0 {
            col_idx.push(i - 1);
            values.push(-1.0);
        }
        
        // Diagonal
        col_idx.push(i);
        values.push(2.0);
        
        // Superdiagonal
        if i < n - 1 {
            col_idx.push(i + 1);
            values.push(-1.0);
        }
        
        row_ptr.push(col_idx.len());
    }
    
    Ok(CsrMatrix::from_csr(n, n, row_ptr, col_idx, values))
}

/// Create a 2D Poisson matrix (5-point stencil on grid)
fn create_poisson_2d_matrix(grid_size: usize) -> Result<CsrMatrix<f64>, Box<dyn std::error::Error>> {
    let n = grid_size * grid_size;
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..n {
        let row = i / grid_size;
        let col = i % grid_size;
        
        let mut neighbors = Vec::new();
        
        // West neighbor
        if col > 0 {
            neighbors.push((i - 1, -1.0));
        }
        
        // South neighbor
        if row > 0 {
            neighbors.push((i - grid_size, -1.0));
        }
        
        // Center (diagonal)
        neighbors.push((i, 4.0));
        
        // North neighbor
        if row < grid_size - 1 {
            neighbors.push((i + grid_size, -1.0));
        }
        
        // East neighbor
        if col < grid_size - 1 {
            neighbors.push((i + 1, -1.0));
        }
        
        // Sort neighbors by column index to ensure CSR format
        neighbors.sort_by_key(|&(col, _)| col);
        
        for (col_id, val) in neighbors {
            col_idx.push(col_id);
            values.push(val);
        }
        
        row_ptr.push(col_idx.len());
    }
    
    Ok(CsrMatrix::from_csr(n, n, row_ptr, col_idx, values))
}
