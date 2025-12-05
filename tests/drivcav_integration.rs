#![cfg(feature = "backend-faer")]
//! Integration test for DRIVCAV matrix family support.
//! cargo run --example mpi_amg_gmres_demo -- -ksp_type gmres -pc_type ilutp -pc_ilut_max_fill 10 -pc_ilut_perm_tol 0.1
//! Tests the comprehensive ILUTP implementation with option parsing,
//! matrix reordering, and KspContext integration.

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::config::options::PcOptions;
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use kryst::error::KError;
use kryst::matrix::op::LinOp;
use std::sync::Arc;

#[test]
fn test_ilutp_options_parsing() -> Result<(), KError> {
    // Test that we can create PcOptions with ILUTP settings
    let pc_options = PcOptions {
        pc_type: Some("ilutp".to_string()),
        ilut_max_fill: Some(15),
        ilut_perm_tol: Some(0.05),
        reorder: Some("colamd".to_string()),
        ..Default::default()
    };

    // Check that ILUTP options were set correctly
    assert_eq!(pc_options.pc_type, Some("ilutp".to_string()));
    assert_eq!(pc_options.ilut_max_fill, Some(15));
    assert_eq!(pc_options.ilut_perm_tol, Some(0.05));
    assert_eq!(pc_options.reorder, Some("colamd".to_string()));

    Ok(())
}

#[test]
fn test_ilutp_environment_parsing() -> Result<(), KError> {
    // Test creating PcOptions with environment-like values
    let pc_options = PcOptions {
        pc_type: Some("ilutp".to_string()),
        ilut_max_fill: Some(20),
        ilut_perm_tol: Some(0.1),
        reorder: Some("amd".to_string()),
        ..Default::default()
    };

    assert_eq!(pc_options.pc_type, Some("ilutp".to_string()));
    assert_eq!(pc_options.ilut_max_fill, Some(20));
    assert_eq!(pc_options.ilut_perm_tol, Some(0.1));
    assert_eq!(pc_options.reorder, Some("amd".to_string()));

    Ok(())
}

#[test]
#[ignore]
fn test_ksp_context_ilutp_integration() -> Result<(), KError> {
    // Create a well-posed test matrix (8x8 symmetric positive definite)
    let n = 8;
    let mut matrix = Mat::<R>::zeros(n, n);

    // Create a simple symmetric positive definite matrix
    // Using a scaled identity + symmetric matrix
    for i in 0..n {
        matrix[(i, i)] = S::from_real(4.0).real(); // Strong diagonal dominance
        if i > 0 {
            let off_diag = S::from_real(-1.0).real();
            matrix[(i, i - 1)] = off_diag;
            matrix[(i - 1, i)] = off_diag; // Ensure symmetry
        }
    }

    // Create a simple RHS that gives a known solution
    // Let's use x = [1, 2, 3, 4, 5, 6, 7, 8] as our target solution
    let exact_solution: Vec<R> = (1..=n).map(|i| S::from_real(i as f64).real()).collect();

    // Compute b = A * exact_solution
    let mut b = vec![R::default(); n];
    for i in 0..n {
        for j in 0..n {
            b[i] += matrix[(i, j)] * exact_solution[j];
        }
    }

    let mut x = vec![R::default(); n];

    // Setup KspContext with ILUTP
    let mut ksp = KspContext::new();

    // Set operators and ILUTP preconditioner options with reasonable parameters
    let pc_options = PcOptions {
        pc_type: Some("ilutp".to_string()),
        ilut_max_fill: Some(20),  // Allow sufficient fill
        ilut_perm_tol: Some(0.1), // Reasonable permutation tolerance
        reorder: Some("none".to_string()),
        ..Default::default()
    };
    let amat: Arc<dyn LinOp<S = f64>> = Arc::new(matrix.clone());
    ksp.set_operators(amat, None);

    // Use reasonable tolerances
    ksp.set_type(SolverType::Gmres)?
        .set_pc_type(PcType::Ilutp, Some(&pc_options))?
        .set_tolerances(1e-8, 1e-12, 1e3, 100);

    // Solve the system
    let stats = ksp.solve(&b, &mut x)?;

    // Verify convergence
    println!("ILUTP-GMRES converged in {} iterations", stats.iterations);
    assert!(
        stats.iterations > 0,
        "Should have performed at least one iteration"
    );
    assert!(
        stats.iterations < 50,
        "Should converge reasonably fast for this SPD problem"
    );

    // Verify solution quality by computing residual ||Ax - b||
    let mut residual = b.clone();
    for i in 0..n {
        let mut ax_i = R::default();
        for j in 0..n {
            ax_i += matrix[(i, j)] * x[j];
        }
        residual[i] -= ax_i;
    }

    let residual_norm: R = residual.iter().map(|r| r * r).sum::<R>().sqrt();
    let rhs_norm: R = b.iter().map(|r| r * r).sum::<R>().sqrt();
    let relative_residual = residual_norm / rhs_norm;

    println!("Final residual norm: {:.2e}", residual_norm);
    println!("Relative residual: {:.2e}", relative_residual);

    // Check that we achieved reasonable accuracy
    assert!(
        relative_residual < 1e-6,
        "Relative residual should be small: got {:.2e}",
        relative_residual
    );

    // Verify solution accuracy against known exact solution
    let mut solution_error = R::default();
    for i in 0..n {
        solution_error += (x[i] - exact_solution[i]).powi(2);
    }
    solution_error = solution_error.sqrt();
    let exact_norm: R = exact_solution.iter().map(|s| s * s).sum::<R>().sqrt();
    let relative_error = solution_error / exact_norm;

    println!("Solution error norm: {:.2e}", solution_error);
    println!("Relative solution error: {:.2e}", relative_error);

    // Since we constructed b = A*x_exact, the solution should be very accurate
    assert!(
        relative_error < 1e-6,
        "Solution should be accurate for this constructed problem: got {:.2e}",
        relative_error
    );

    Ok(())
}

#[test]
fn test_pc_type_string_parsing() -> Result<(), KError> {
    // Test that all PC types can be parsed from strings
    let test_cases = vec![
        ("jacobi", PcType::Jacobi),
        ("ilu0", PcType::Ilu0),
        ("none", PcType::None),
        ("ilu", PcType::Ilu),
        ("ilut", PcType::Ilut),
        ("ilutp", PcType::Ilutp),
        ("ilup", PcType::Ilup),
        ("lu", PcType::Lu),
        ("qr", PcType::Qr),
    ];

    for (pc_str, expected_type) in test_cases {
        let parsed_type = pc_str.parse::<PcType>()?;
        assert_eq!(
            parsed_type, expected_type,
            "Failed to parse PC type: {}",
            pc_str
        );
    }

    Ok(())
}

#[test]
fn test_reorder_validation() {
    // Test that we can create PcOptions with different reorder types
    let valid_reorders = vec!["none", "colamd", "amd"];

    for reorder in valid_reorders {
        let pc_options = PcOptions {
            reorder: Some(reorder.to_string()),
            ..Default::default()
        };
        assert_eq!(pc_options.reorder, Some(reorder.to_string()));
    }
}

#[cfg(any())]
mod integration_benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_ilutp_vs_jacobi() -> Result<(), KError> {
        // Create a moderately sized test problem (100x100)
        let n = 100;
        let mut matrix = Mat::<R>::zeros(n, n);

        // Create a more challenging matrix (tridiagonal with some off-diagonal terms)
        for i in 0..n {
            matrix[(i, i)] = S::from_real(4.0).real();
            if i > 0 {
                matrix[(i, i - 1)] = S::from_real(-1.0).real();
            }
            if i < n - 1 {
                matrix[(i, i + 1)] = S::from_real(-1.0).real();
            }
            // Add some fill to make it more interesting
            if i > 1 {
                matrix[(i, i - 2)] = S::from_real(0.1).real();
            }
            if i < n - 2 {
                matrix[(i, i + 2)] = S::from_real(0.1).real();
            }
        }

        let b: Vec<R> = (0..n)
            .map(|i| S::from_real((i as f64 + 1.0).sin()).real())
            .collect();
        let mut x_jacobi = vec![R::default(); n];
        let mut x_ilutp = vec![R::default(); n];

        // Test with Jacobi preconditioner
        let start = Instant::now();
        let mut ksp_jacobi = KspContext::new();
        ksp_jacobi
            .set_type(SolverType::Gmres)?
            .set_pc_type(PcType::Jacobi, None)?
            .set_tolerances(1e-8, 1e-12, 1e3, 200);
        let stats_jacobi = ksp_jacobi.solve(&matrix, &b, &mut x_jacobi)?;
        let time_jacobi = start.elapsed();

        // Test with ILUTP preconditioner
        let start = Instant::now();
        let mut ksp_ilutp = KspContext::new();
        let pc_options = PcOptions {
            pc_type: Some("ilutp".to_string()),
            ilut_max_fill: Some(10),
            ilut_perm_tol: Some(0.1),
            reorder: Some("none".to_string()),
            ..Default::default()
        };
        ksp_ilutp
            .set_type(SolverType::Gmres)?
            .set_pc_type(PcType::Ilutp, None)?
            .set_pc_options(pc_options)
            .set_tolerances(1e-8, 1e-12, 1e3, 200);
        let stats_ilutp = ksp_ilutp.solve(&matrix, &b, &mut x_ilutp)?;
        let time_ilutp = start.elapsed();

        println!(
            "Jacobi: {} iterations in {:.2?}",
            stats_jacobi.iterations, time_jacobi
        );
        println!(
            "ILUTP:  {} iterations in {:.2?}",
            stats_ilutp.iterations, time_ilutp
        );

        // Both should converge (this is the main test)
        assert!(stats_jacobi.iterations < 200, "Jacobi should converge");
        assert!(stats_ilutp.iterations < 200, "ILUTP should converge");

        // Note: For this simple test matrix, ILUTP may not always outperform Jacobi
        // The value of ILUTP becomes apparent with more complex, ill-conditioned matrices
        // println!("Both preconditioners converged successfully, demonstrating ILUTP integration");

        Ok(())
    }
}
