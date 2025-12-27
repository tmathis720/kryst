//! Simple test to debug the matrix loading and solver issues
//!
//! This version keeps the CSR path only (no dense transformation).
//!
//! to run:
//! cargo run --example debug_matrix_test
#[cfg(feature = "complex")]
fn main() {
    eprintln!("debug_matrix_test is disabled when the complex feature is enabled.");
}

#[cfg(not(feature = "complex"))]
use kryst::algebra::prelude::*;
#[cfg(not(feature = "complex"))]
use kryst::context::ksp_context::{KspContext, SolverType};
#[cfg(not(feature = "complex"))]
use kryst::context::pc_context::PcType;
#[cfg(not(feature = "complex"))]
use kryst::utils::matrix_market::read_matrix_market;
#[cfg(not(feature = "complex"))]
use std::path::PathBuf;
#[cfg(not(feature = "complex"))]
use std::sync::Arc;
#[cfg(not(feature = "complex"))]
use std::time::Instant;

#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Debug Matrix Market Test (CSR-only)");
    println!("===================================");

    // Check sizes of different matrices (paths resolved relative to crate dir so the example
    // works regardless of the current working directory when the binary is run).
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_matrices = vec!["fidap001", "fidap002", "fidap005", "add20", "e05r0100"];

    for matrix_name in test_matrices {
        let matrix_path = base_dir
            .join("examples")
            .join("mtx")
            .join(format!("{}.mtx", matrix_name));
        let _rhs_path = base_dir
            .join("examples")
            .join("mtx")
            .join(format!("{}_rhs1.mtx", matrix_name));

        let matrix_path_str = matrix_path.to_str().unwrap();
        match read_matrix_market(matrix_path_str) {
            Ok(matrix_data) => {
                let matrix = matrix_data.to_csr_matrix()?;
                println!(
                    "{}: {}x{} matrix, {} nnz",
                    matrix_name,
                    matrix.nrows(),
                    matrix.ncols(),
                    matrix.nnz()
                );
            }
            Err(_) => {
                println!("{}: Failed to load ({})", matrix_name, matrix_path_str);
            }
        }
    }

    // Now test with the smallest matrix (likely add20 or fidap001)
    let matrix_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("mtx")
        .join("add20.mtx");
    let rhs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("mtx")
        .join("add20_rhs1.mtx");

    println!("\nTesting with add20 matrix:");
    println!("Loading matrix: {}", matrix_path.display());
    let matrix_data = read_matrix_market(matrix_path.to_str().unwrap())?;

    println!("Loading RHS: {}", rhs_path.display());
    let rhs_data = read_matrix_market(rhs_path.to_str().unwrap())?;

    println!("Converting to CSR format...");
    let matrix = matrix_data.to_csr_matrix_scalar()?;
    let rhs: Vec<S> = rhs_data.to_vector_scalar()?;

    println!("Matrix size: {}x{}", matrix.nrows(), matrix.ncols());
    println!("Matrix nnz: {}", matrix.nnz());
    println!("RHS size: {}", rhs.len());

    if matrix.nrows() > 2500 {
        println!("Matrix too large for quick test, skipping solve...");
        return Ok(());
    }

    let mut solution = vec![S::default(); rhs.len()];

    println!("Setting up KSP context (CSR operator)...");
    let mut ksp = KspContext::new();
    // Replace your GMRES/Jacobi block with:
    ksp.set_type(SolverType::Gmres)?
        .set_pc_type(PcType::Ilu, None)? // Start with no PC
        .set_tolerances(1e-6, 1e-12, 2000.0, 50); // (rtol, atol, maxit, restart=unused for CG)

    // Keep the CSR operator:
    ksp.set_operators(Arc::new(matrix), None);
    ksp.setup()?;

    println!("Starting solve (CSR)...");
    let start = Instant::now();
    let result = ksp.solve(&rhs, &mut solution);
    let solve_time = start.elapsed().as_secs_f64();

    match result {
        Ok(stats) => {
            println!("Solve completed in {:.3}s", solve_time);
            println!("Iterations: {}", stats.iterations);
            println!("Final residual: {:.2e}", stats.final_residual);
            println!("Converged: {}", stats.final_residual < 1e-6);
        }
        Err(e) => {
            println!("Solve failed: {}", e);
        }
    }

    Ok(())
}
