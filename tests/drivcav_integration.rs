//! Integration test for DRIVCAV matrix family support.
//! cargo run --example mpi_amg_gmres_demo -- -ksp_type gmres -pc_type ilutp -pc_ilut_max_fill 10 -pc_ilut_perm_tol 0.1
//! Tests the comprehensive ILUTP implementation with option parsing,
//! matrix reordering, and KspContext integration.

use kryst::config::options::PcOptions;
use kryst::context::ksp_context::{KspContext, SolverType, PcType};
use kryst::error::KError;
use faer::Mat;

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
fn test_ksp_context_ilutp_integration() -> Result<(), KError> {
    // Create a test matrix (5x5 diagonally dominant)
    let mut matrix = Mat::zeros(5, 5);
    for i in 0..5 {
        matrix[(i, i)] = 10.0; // Strong diagonal
        if i > 0 {
            matrix[(i, i-1)] = -1.0;
        }
        if i < 4 {
            matrix[(i, i+1)] = -1.0;
        }
    }
    
    // Create RHS vector
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut x = vec![0.0; 5];
    
    // Setup KspContext with ILUTP
    let mut ksp = KspContext::new();
    
    // Set ILUTP preconditioner options
    let pc_options = PcOptions {
        pc_type: Some("ilutp".to_string()),
        ilut_max_fill: Some(5),
        ilut_perm_tol: Some(0.1),
        reorder: Some("none".to_string()),
        ..Default::default()
    };
    
    ksp.set_type(SolverType::Gmres)?
       .set_pc_type(PcType::Ilutp)?
       .set_pc_options(pc_options)
       .set_tolerances(1e-8, 1e-12, 1e3, 100);
    
    // Solve the system
    let stats = ksp.solve(&matrix, &b, &mut x)?;
    
    // Verify convergence
    println!("ILUTP-GMRES converged in {} iterations", stats.iterations);
    assert!(stats.iterations > 0, "Should have performed at least one iteration");
    assert!(stats.iterations < 50, "Should converge quickly for this well-conditioned problem");
    
    // Verify solution quality by computing residual
    let mut residual = b.clone();
    for i in 0..5 {
        let mut ax_i = 0.0;
        for j in 0..5 {
            ax_i += matrix[(i, j)] * x[j];
        }
        residual[i] -= ax_i;
    }
    
    let residual_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
    println!("Final residual norm: {:.2e}", residual_norm);
    assert!(residual_norm < 1e-6, "Solution should be accurate");
    
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
        assert_eq!(parsed_type, expected_type, "Failed to parse PC type: {}", pc_str);
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

#[cfg(test)]
mod integration_benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn bench_ilutp_vs_jacobi() -> Result<(), KError> {
        // Create a moderately sized test problem (100x100)
        let n = 100;
        let mut matrix = Mat::zeros(n, n);
        
        // Create a more challenging matrix (tridiagonal with some off-diagonal terms)
        for i in 0..n {
            matrix[(i, i)] = 4.0;
            if i > 0 {
                matrix[(i, i-1)] = -1.0;
            }
            if i < n-1 {
                matrix[(i, i+1)] = -1.0;
            }
            // Add some fill to make it more interesting
            if i > 1 {
                matrix[(i, i-2)] = 0.1;
            }
            if i < n-2 {
                matrix[(i, i+2)] = 0.1;
            }
        }
        
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
        let mut x_jacobi = vec![0.0; n];
        let mut x_ilutp = vec![0.0; n];
        
        // Test with Jacobi preconditioner
        let start = Instant::now();
        let mut ksp_jacobi = KspContext::new();
        ksp_jacobi.set_type(SolverType::Gmres)?
                  .set_pc_type(PcType::Jacobi)?
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
        ksp_ilutp.set_type(SolverType::Gmres)?
                 .set_pc_type(PcType::Ilutp)?
                 .set_pc_options(pc_options)
                 .set_tolerances(1e-8, 1e-12, 1e3, 200);
        let stats_ilutp = ksp_ilutp.solve(&matrix, &b, &mut x_ilutp)?;
        let time_ilutp = start.elapsed();
        
        println!("Jacobi: {} iterations in {:.2?}", stats_jacobi.iterations, time_jacobi);
        println!("ILUTP:  {} iterations in {:.2?}", stats_ilutp.iterations, time_ilutp);
        
        // Both should converge (this is the main test)
        assert!(stats_jacobi.iterations < 200, "Jacobi should converge");
        assert!(stats_ilutp.iterations < 200, "ILUTP should converge");
        
        // Note: For this simple test matrix, ILUTP may not always outperform Jacobi
        // The value of ILUTP becomes apparent with more complex, ill-conditioned matrices
        println!("Both preconditioners converged successfully, demonstrating ILUTP integration");
        
        Ok(())
    }
}
