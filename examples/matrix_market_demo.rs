//! Comprehensive example demonstrating Matrix Market I/O with iterative vs direct solver comparison.
//!
//! This example shows how to:
//! 1. Read challenging sparse matrices from Matrix Market files (e.g., driven cavity problems)
//! 2. Analyze matrix properties and conditioning
//! 3. Compare iterative methods with different preconditioners
//! 4. Compare with direct methods when available
//! 5. Provide robust solver recommendations based on matrix characteristics
//!
//! The driven cavity matrices (e05r0000, e30r0000, etc.) are particularly challenging:
//! - Non-symmetric and indefinite from 2D fluid flow modeling
//! - Difficult for iterative methods due to poor conditioning
//! - May have zeros on diagonal from incompressibility conditions
//! - ILU preconditioners can be unstable due to poor factorization quality

use kryst::utils::matrix_market::read_matrix_market;
use kryst::context::ksp_context::KspContext;
use kryst::matrix::sparse::{SparseMatrix, CsrMatrix};
use std::time::Instant;

/// Analyze matrix properties and provide diagnostics
fn analyze_matrix(matrix: &CsrMatrix<f64>) -> (f64, bool) {
    let n = matrix.nrows();
    let nnz = matrix.nnz();
    let density = nnz as f64 / (n * n) as f64;
    
    // Convert to dense to check diagonal (for small matrices only)
    let has_diag_zeros = if n < 1000 {
        let dense = matrix.to_dense();
        let mut zeros = 0;
        for i in 0..n {
            if dense[(i, i)].abs() < 1e-15 {
                zeros += 1;
            }
        }
        zeros > 0
    } else {
        false // Skip check for large matrices
    };
    
    println!("Matrix Analysis:");
    println!("  Dimensions: {}x{}", n, n);
    println!("  Non-zeros: {} ({:.4}% density)", nnz, density * 100.0);
    if n < 1000 {
        println!("  Diagonal zeros: {}", if has_diag_zeros { "detected" } else { "none" });
    }
    
    (density, has_diag_zeros)
}

/// Test a solver configuration and return timing/convergence results
fn test_solver_config(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    solver_name: &str,
    pc_name: &str,
) -> Result<(usize, f64, f64, bool), Box<dyn std::error::Error>> {
    let mut solution = vec![0.0; rhs.len()];
    
    // Convert sparse matrix to dense for KspContext
    let dense_matrix = matrix.to_dense();
    let rhs_vec = rhs.to_vec();
    
    // Create and configure KSP context
    let mut ksp = KspContext::new();
    ksp.set_type_from_str(solver_name)?
       .set_pc_type_from_str(pc_name)?
       .set_tolerances(1e-6, 1e-12, 1e3, 1000);
    
    let start = Instant::now();
    let stats = ksp.solve(&dense_matrix, &rhs_vec, &mut solution)?;
    let solve_time = start.elapsed().as_secs_f64();
    
    let converged = stats.final_residual < 1e-6;
    
    Ok((stats.iterations, stats.final_residual, solve_time, converged))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging if available
    #[cfg(feature = "logging")]
    env_logger::init();

    println!("Comprehensive Matrix Market Solver Comparison");
    println!("============================================");
    println!();

    // Test multiple matrix files, focusing on driven cavity problems
    let test_matrices = vec![
        ("examples/e05r0000/e05r0000.mtx", "examples/e05r0000/e05r0000_rhs1.mtx", "Driven cavity (Re=0)"),
        ("examples/e05r0300/e05r0300.mtx", "examples/e05r0300/e05r0300_rhs1.mtx", "Driven cavity (Re=300)"),
        ("examples/e30r0000/e30r0000.mtx", "examples/e30r0000/e30r0000_rhs1.mtx", "Driven cavity 30x30 (Re=0)"),
        ("examples/e30r1000/e30r1000.mtx", "examples/e30r1000/e30r1000_rhs1.mtx", "Driven cavity 30x30 (Re=1000)"),
    ];

    for (matrix_path, rhs_path, description) in test_matrices {
        println!("Testing: {}", description);
        println!("Matrix: {}", matrix_path);
        println!("RHS: {}", rhs_path);
        
        // Try to read the matrix and RHS
        let (matrix_data, rhs_data) = match (
            read_matrix_market(matrix_path),
            read_matrix_market(rhs_path)
        ) {
            (Ok(matrix), Ok(rhs)) => (matrix, rhs),
            _ => {
                println!("⚠ Files not found, skipping {}", description);
                println!();
                continue;
            }
        };

        // Convert to Kryst formats
        let matrix = matrix_data.to_csr_matrix()?;
        let rhs = rhs_data.to_vector()?;

        // Analyze matrix properties
        let (density, has_zeros) = analyze_matrix(&matrix);
        
        if has_zeros {
            println!("⚠ Matrix has zeros on diagonal - typical of driven cavity problems");
            println!("  This makes diagonal-based preconditioners unstable");
        }
        
        println!();

        // Test different solver/preconditioner combinations
        let solver_configs = vec![
            // // Robust combinations for difficult problems
            // ("gmres", "none", "GMRES (no preconditioner)"),
            // ("gmres", "ilu0", "GMRES + ILU(0)"),
            // ("bicgstab", "none", "BiCGStab (no preconditioner)"),
            // ("bicgstab", "ilu0", "BiCGStab + ILU(0)"),
            // ("tfqmr", "none", "TFQMR (no preconditioner)"),
            
            // // These may fail for driven cavity matrices
            // ("gmres", "jacobi", "GMRES + Jacobi (may fail with diagonal zeros)"),
            // ("cg", "none", "CG (will fail for non-SPD)"),
            
            // Direct solver as reference
            ("preonly", "lu", "Direct LU"),
        ];

        println!("Solver Comparison Results:");
        println!("{:<35} {:>8} {:>12} {:>10} {:>8}", "Method", "Iters", "Residual", "Time(s)", "Status");
        println!("{}", "-".repeat(75));

        for (solver_type, pc_type, description) in solver_configs {
            match test_solver_config(&matrix, &rhs, solver_type, pc_type) {
                Ok((iters, residual, time, converged)) => {
                    let status = if converged { "✓" } else { "✗" };
                    println!("{:<35} {:>8} {:>12.2e} {:>10.3} {:>8}", 
                             description, iters, residual, time, status);
                    
                    if converged && solver_type != "preonly" {
                        // Calculate iteration efficiency
                        let dof_per_sec = matrix.nrows() as f64 / time;
                        if dof_per_sec > 1000.0 {
                            println!("    → High efficiency: {:.0} DOF/s", dof_per_sec);
                        }
                    }
                },
                Err(e) => {
                    println!("{:<35} {:>8} {:>12} {:>10} {:>8}", 
                             description, "FAIL", "N/A", "N/A", "✗");
                    println!("    → Error: {}", e);
                }
            }
        }
        
        println!();
        
        // Provide recommendations based on matrix characteristics
        println!("Recommendations for this problem:");
        if has_zeros {
            println!("  • Avoid Jacobi, SOR preconditioners (diagonal zeros)");
            println!("  • ILU(0) may be unstable - consider GMRES without preconditioning");
        }
        if density < 0.001 {
            println!("  • Very sparse matrix - iterative methods preferred over direct");
        }
        if matrix.nrows() > 10000 {
            println!("  • Large problem - direct methods may require excessive memory");
            println!("  • Focus on robust iterative methods (GMRES, BiCGStab, TFQMR)");
        }
        
        println!();
        println!("{}", "=".repeat(80));
        println!();
    }

    // If no test matrices were found, generate a simple test case
    println!("Note: For driven cavity matrices, download from Matrix Market:");
    println!("  https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/drivcav/");
    println!("  These matrices are specifically designed to test iterative solver robustness.");
    println!();
    println!("Key insights about driven cavity problems:");
    println!("  • Non-symmetric and indefinite from incompressible Navier-Stokes");  
    println!("  • Diagonal zeros from incompressibility condition");
    println!("  • ILU preconditioners often unstable due to poor factorization");
    println!("  • Direct methods work but become memory-intensive for large Re");
    println!("  • Robust iterative methods: GMRES, BiCGStab, TFQMR without preconditioning");
    
    Ok(())
}
