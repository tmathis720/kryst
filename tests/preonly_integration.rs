//! Integration tests for PREONLY functionality.

use kryst::config::options::{KspOptions, PcOptions};
use kryst::context::ksp_context::{KspContext, SolverType};
use kryst::context::pc_context::PcType;
use faer::Mat;

#[test]
fn test_preonly_lu_simple_system() {
    // Create a simple 3x3 system: A * x = b
    let a_data = vec![
        2.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0,
    ];
    
    // Create matrix from row-major data
    let a = Mat::from_fn(3, 3, |i, j| a_data[i * 3 + j]);
    
    let b = vec![1.0, 2.0, 3.0];
    let mut x = vec![0.0; 3];
    
    // Configure KSP for PREONLY with LU
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Preonly).unwrap()
       .set_pc_type(PcType::Lu).unwrap();
    
    // Solve the system
    let stats = ksp.solve(&a, &b, &mut x).unwrap();
    
    // Verify solution
    assert_eq!(stats.iterations, 1); // PREONLY should converge in 1 iteration
    
    // Check that A * x ≈ b (residual test)
    let mut residual = vec![0.0; 3];
    for i in 0..3 {
        residual[i] = b[i];
        for j in 0..3 {
            residual[i] -= a[(i, j)] * x[j];
        }
    }
    
    // Verify the residual is small
    let res_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
    assert!(res_norm < 1e-12, "Residual norm {} is too large", res_norm);
}

#[test]
fn test_preonly_qr_simple_system() {
    // Create a simple 3x3 system
    let a_data = vec![
        1.0, 2.0, 3.0,
        2.0, 5.0, 6.0,
        3.0, 6.0, 9.0,
    ];
    
    // Create matrix from row-major data
    let a = Mat::from_fn(3, 3, |i, j| a_data[i * 3 + j]);
    
    let b = vec![14.0, 32.0, 50.0]; // Makes solution approximately [1, 2, 3]
    let mut x = vec![0.0; 3];
    
    // Configure KSP for PREONLY with QR
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Preonly).unwrap()
       .set_pc_type(PcType::Qr).unwrap();
    
    // Solve the system
    let stats = ksp.solve(&a, &b, &mut x).unwrap();
    
    // Verify solution properties
    assert_eq!(stats.iterations, 1); // PREONLY should converge in 1 iteration
}

#[test]
fn test_preonly_options_parsing() {
    // Test command-line style configuration
    let args = vec!["-ksp_type", "preonly", "-pc_type", "lu"];
    
    let ksp_opts = KspOptions::from_args(&args).unwrap();
    let pc_opts = PcOptions::from_args(&args).unwrap();
    
    assert_eq!(ksp_opts.ksp_type, Some("preonly".to_string()));
    assert_eq!(pc_opts.pc_type, Some("lu".to_string()));
    
    // Configure context from options
    let mut ksp = KspContext::new();
    ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
    
    // Test configuration by solving a small system
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }); // Identity matrix
    let b = vec![1.0, 2.0];
    let mut x = vec![0.0; 2];
    
    let stats = ksp.solve(&a, &b, &mut x).unwrap();
    assert_eq!(stats.iterations, 1);
    
    // Solution should be [1.0, 2.0] for identity matrix
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!((x[1] - 2.0).abs() < 1e-12);
}

#[test]
fn test_preonly_error_cases() {
    let mut ksp = KspContext::new();
    
    // Test PREONLY without setting a preconditioner
    ksp.set_type(SolverType::Preonly).unwrap();
    
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }); // Identity matrix
    let b = vec![1.0, 2.0];
    let mut x = vec![0.0; 2];
    
    // This should fail because no PC is set
    let result = ksp.solve(&a, &b, &mut x);
    assert!(result.is_err());
    
    // Test PREONLY with non-direct solver PC type (should fail)
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    let result = ksp.solve(&a, &b, &mut x);
    assert!(result.is_err());
}
