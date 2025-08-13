//! Simple test to debug the matrix loading and solver issues
//!
//! This is a minimal test to identify why the optimized demo is hanging.

use kryst::utils::matrix_market::read_matrix_market;
use kryst::context::ksp_context::KspContext;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Debug Matrix Market Test");
    println!("========================");

    // Check sizes of different matrices
    let test_matrices = vec![
        "fidap001", "fidap002", "fidap005", "add20", "e05r0100"
    ];
    
    for matrix_name in test_matrices {
        let matrix_path = format!("../mtx/{}.mtx", matrix_name);
        let _rhs_path = format!("../mtx/{}_rhs1.mtx", matrix_name);

        match read_matrix_market(&matrix_path) {
            Ok(matrix_data) => {
                let matrix = matrix_data.to_csr_matrix()?;
                println!("{}: {}x{} matrix, {} nnz", matrix_name, matrix.nrows(), matrix.ncols(), matrix.nnz());
            },
            Err(_) => {
                println!("{}: Failed to load", matrix_name);
            }
        }
    }
    
    // Now test with the smallest matrix (likely add20 or fidap001)
    let matrix_path = "examples/mtx/add20.mtx";
    let rhs_path = "examples/mtx/add20_rhs1.mtx";
    
    println!("\nTesting with add20 matrix:");
    println!("Loading matrix: {}", matrix_path);
    let matrix_data = read_matrix_market(matrix_path)?;
    
    println!("Loading RHS: {}", rhs_path);
    let rhs_data = read_matrix_market(rhs_path)?;
    
    println!("Converting to CSR format...");
    let matrix = matrix_data.to_csr_matrix()?;
    let rhs = rhs_data.to_vector()?;
    
    println!("Matrix size: {}x{}", matrix.nrows(), matrix.ncols());
    println!("Matrix nnz: {}", matrix.nnz());
    println!("RHS size: {}", rhs.len());
    
    if matrix.nrows() > 2500 {
        println!("Matrix too large for quick test, skipping solve...");
        return Ok(());
    }
    
    println!("Converting to dense format...");
    let dense_matrix = matrix.to_dense();
    
    let mut solution = vec![0.0; rhs.len()];
    let rhs_vec = rhs.to_vec();
    
    println!("Setting up KSP context...");
    let mut ksp = KspContext::new();
    ksp.set_type_from_str("cg")?  // Try CG instead of GMRES
       .set_pc_type_from_str("none")?
       .set_tolerances(1e-6, 1e-12, 1e3, 50); // Very low iteration limit
    
    println!("Starting solve...");
    let start = Instant::now();
    let result = ksp.solve(&dense_matrix, &rhs_vec, &mut solution);
    let solve_time = start.elapsed().as_secs_f64();
    
    match result {
        Ok(stats) => {
            println!("Solve completed in {:.3}s", solve_time);
            println!("Iterations: {}", stats.iterations);
            println!("Final residual: {:.2e}", stats.final_residual);
            println!("Converged: {}", stats.final_residual < 1e-6);
        },
        Err(e) => {
            println!("Solve failed: {}", e);
        }
    }
    
    Ok(())
}
