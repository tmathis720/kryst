//! Comprehensive example demonstrating Matrix Market I/O with iterative vs direct solver comparison (CSR-only).
//!
//! This example shows how to:
//! 1. Read challenging sparse matrices from Matrix Market files (e.g., driven cavity problems)
//! 2. Analyze matrix properties and conditioning (without dense conversions)
//! 3. Compare iterative methods with different preconditioners
//! 4. Use a "preonly" direct solve through KSP + PC (still via CSR operator)
//! 5. Provide robust solver recommendations based on matrix characteristics
//!
//! Notes on driven cavity matrices (e05r0000, e30r0000, etc.):
//! - Often non-symmetric / indefinite (from incompressible flow discretizations)
//! - May contain zeros on the diagonal
//! - Diagonal-based preconditioners (Jacobi/SOR) can fail
//! - ILU may be unstable depending on ordering & fill strategy

use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::matrix::op::CsrOp; // CSR -> LinOp wrapper
use kryst::matrix::sparse::CsrMatrix;
use kryst::utils::matrix_market::read_matrix_market;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

/// Analyze matrix properties and provide diagnostics (CSR-only)
fn analyze_matrix(matrix: &CsrMatrix<f64>) -> (f64, bool) {
    let n = matrix.nrows();
    let nnz = matrix.nnz();
    let density = nnz as f64 / (n * n) as f64;

    // Check diagonal directly from CSR structure for modest sizes
    let mut has_diag_zeros = false;
    if n > 0 && n <= 20000 {
        // Access CSR internals
        let rp = matrix.row_ptr();
        let ci = matrix.col_idx();
        let va = matrix.values();

        let mut diag_zero_or_missing = 0usize;
        for i in 0..n {
            let start = rp[i];
            let end = rp[i + 1];
            let mut found = false;
            for k in start..end {
                if ci[k] == i {
                    found = true;
                    if va[k].abs() < 1e-15 {
                        diag_zero_or_missing += 1;
                    }
                    break;
                }
            }
            if !found {
                // Missing diagonal is effectively a zero diagonal entry
                diag_zero_or_missing += 1;
            }
        }
        has_diag_zeros = diag_zero_or_missing > 0;
    }

    println!("Matrix Analysis (CSR):");
    println!("  Dimensions: {}x{}", n, n);
    println!("  Non-zeros: {} ({:.4}% density)", nnz, density * 100.0);
    if n <= 20000 {
        println!(
            "  Diagonal zeros/missing: {}",
            if has_diag_zeros { "detected" } else { "none" }
        );
    }

    (density, has_diag_zeros)
}

/// Test a solver configuration (CSR operator end-to-end)
fn test_solver_config(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    solver_name: &str,
    pc_name: &str,
) -> Result<(usize, f64, f64, bool), Box<dyn std::error::Error>> {
    let mut solution = vec![0.0; rhs.len()];

    // KSP + PC configured to operate on CSR via CsrOp wrapper (no dense conversion)
    let mut ksp = KspContext::new();

    let st = SolverType::from_str(solver_name)?;
    let pct = PcType::from_str(pc_name)?;
    ksp.set_type(st)?
        .set_pc_type(pct, None)?
        .set_tolerances(1e-6, 1e-12, 1e3, 1000);

    // Provide CSR operator and prepare workspace
    let op = CsrOp::new(matrix.clone().into());
    ksp.set_operators(Arc::new(op), None);
    ksp.setup()?;

    // Solve
    let start = Instant::now();
    let stats = ksp.solve(rhs, &mut solution)?;
    let solve_time = start.elapsed().as_secs_f64();

    let converged = stats.final_residual < 1e-6;
    Ok((
        stats.iterations,
        stats.final_residual,
        solve_time,
        converged,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging if available
    #[cfg(feature = "logging")]
    env_logger::init();

    println!("Comprehensive Matrix Market Solver Comparison (CSR-only)");
    println!("========================================================");
    println!();

    // Test multiple matrix files, focusing on driven cavity problems.
    // Resolve paths relative to the crate so the example is robust to CWD.
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let raw_list = vec![
        (
            "e05r0000/e05r0000.mtx",
            "e05r0000/e05r0000_rhs1.mtx",
            "Driven cavity (Re=0)",
        ),
        (
            "e05r0300/e05r0300.mtx",
            "e05r0300/e05r0300_rhs1.mtx",
            "Driven cavity (Re=300)",
        ),
        (
            "e30r0000/e30r0000.mtx",
            "e30r0000/e30r0000_rhs1.mtx",
            "Driven cavity 30x30 (Re=0)",
        ),
        (
            "e30r1000/e30r1000.mtx",
            "e30r1000/e30r1000_rhs1.mtx",
            "Driven cavity 30x30 (Re=1000)",
        ),
    ];

    for (mat_rel, rhs_rel, description) in raw_list {
        let matrix_path = base.join("examples").join("").join(mat_rel);
        let rhs_path = base.join("examples").join("").join(rhs_rel);
        let matrix_path_s = matrix_path.to_str().unwrap();
        let rhs_path_s = rhs_path.to_str().unwrap();

        println!("Testing: {}", description);
        println!("Matrix: {}", matrix_path_s);
        println!("RHS: {}", rhs_path_s);

        // Try to read the matrix and RHS
        let (matrix_data, rhs_data) = match (
            read_matrix_market(matrix_path_s),
            read_matrix_market(rhs_path_s),
        ) {
            (Ok(matrix), Ok(rhs)) => (matrix, rhs),
            _ => {
                println!("⚠ Files not found, skipping {}", description);
                println!();
                continue;
            }
        };

        // Convert to Kryst formats (CSR + vector)
        let matrix = matrix_data.to_csr_matrix()?;
        let rhs = rhs_data.to_vector()?;

        // Analyze matrix properties (CSR-only)
        let (density, has_zeros) = analyze_matrix(&matrix);
        if has_zeros {
            println!("⚠ Matrix has zeros or missing entries on the diagonal (common in driven cavity)");
            println!("  → Diagonal-based preconditioners (Jacobi/SOR) may fail.");
        }
        println!();

        // Solver/preconditioner combinations (all via CSR operator)
        let solver_configs = vec![
            // Iterative baselines
            ("gmres", "none", "GMRES (no preconditioner)"),
            ("bicgstab", "none", "BiCGStab (no preconditioner)"),
            // Safer ILU when available; may still struggle on highly indefinite cases
            ("gmres", "ilu0", "GMRES + ILU(0)"),
            ("bicgstab", "ilu0", "BiCGStab + ILU(0)"),
            // Direct solve through KSP pipeline (no explicit dense path)
            ("preonly", "lu", "Direct LU (preonly)"),
        ];

        println!("Solver Comparison Results (CSR operator):");
        println!(
            "{:<35} {:>8} {:>12} {:>10} {:>8}",
            "Method", "Iters", "Residual", "Time(s)", "Status"
        );
        println!("{}", "-".repeat(75));

        for (solver_type, pc_type, label) in solver_configs {
            match test_solver_config(&matrix, &rhs, solver_type, pc_type) {
                Ok((iters, residual, time, converged)) => {
                    let status = if converged { "✓" } else { "✗" };
                    println!(
                        "{:<35} {:>8} {:>12.2e} {:>10.3} {:>8}",
                        label, iters, residual, time, status
                    );

                    if converged && solver_type != "preonly" {
                        let dof_per_sec = matrix.nrows() as f64 / time.max(1e-12);
                        if dof_per_sec > 1_000.0 {
                            println!("    → High efficiency: {:.0} DOF/s", dof_per_sec);
                        }
                    }
                }
                Err(e) => {
                    println!(
                        "{:<35} {:>8} {:>12} {:>10} {:>8}",
                        label, "FAIL", "N/A", "N/A", "✗"
                    );
                    println!("    → Error: {}", e);
                }
            }
        }

        println!();

        // Provide recommendations based on matrix characteristics
        println!("Recommendations for this problem:");
        if has_zeros {
            println!("  • Avoid Jacobi / SOR (diagonal zeros or missing diagonal entries).");
            println!("  • ILU(0) may be unstable; GMRES without PC can be a safer baseline.");
        }
        if density < 0.001 {
            println!("  • Very sparse: prefer iterative methods over dense/direct approaches.");
        }
        if matrix.nrows() > 10000 {
            println!("  • Large problem size: direct methods may require excessive memory.");
            println!("  • Focus on GMRES / BiCGStab with robust PC (ILU, AMG where available).");
        }

        println!();
        println!("{}", "=".repeat(80));
        println!();
    }

    println!("Note: For driven cavity matrices, download from Matrix Market:");
    println!("  https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/drivcav/");
    println!("Key insights about driven cavity problems:");
    println!("  • Non-symmetric and indefinite from incompressible Navier–Stokes.");
    println!("  • Diagonal zeros/missing entries are common.");
    println!("  • ILU preconditioners can be fragile depending on fill/ordering.");
    println!("  • Direct methods can work but often become memory-bound at scale.");
    println!("  • Robust iterative methods: GMRES, BiCGStab, TFQMR (with careful PC).");

    Ok(())
}
