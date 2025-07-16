//! Tests for workspace reuse and explicit setup phase.

use kryst::context::ksp_context::{KspContext, SolverType, PcType};
use faer::Mat;

#[test]
fn test_explicit_setup() {
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::Jacobi).unwrap();
    
    // Initially, setup should not be called
    assert!(!ksp.is_setup());
    
    // Create a test matrix
    let a = Mat::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 0.0 });
    
    // Call explicit setup
    ksp.setup(&a, 3).unwrap();
    
    // Now setup should be complete
    assert!(ksp.is_setup());
}

#[test]
fn test_auto_setup_on_solve() {
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::None).unwrap();
    
    // Create test problem: 2x2 identity system
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let b = vec![1.0, 2.0];
    let mut x = vec![0.0, 0.0];
    
    // Setup should happen automatically on first solve
    assert!(!ksp.is_setup());
    let _stats = ksp.solve(&a, &b, &mut x).unwrap();
    assert!(ksp.is_setup());
    
    // Solution should be approximately [1.0, 2.0] for identity matrix
    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_workspace_reuse() {
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::None).unwrap();
    
    // Create test problem
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let b1 = vec![1.0, 2.0];
    let b2 = vec![3.0, 4.0];
    let mut x1 = vec![0.0, 0.0];
    let mut x2 = vec![0.0, 0.0];
    
    // First solve (setup happens)
    let _stats1 = ksp.solve(&a, &b1, &mut x1).unwrap();
    assert!(ksp.is_setup());
    
    // Second solve (should reuse workspace)
    let _stats2 = ksp.solve(&a, &b2, &mut x2).unwrap();
    assert!(ksp.is_setup());
    
    // Verify solutions
    assert!((x1[0] - 1.0).abs() < 1e-10);
    assert!((x1[1] - 2.0).abs() < 1e-10);
    assert!((x2[0] - 3.0).abs() < 1e-10);
    assert!((x2[1] - 4.0).abs() < 1e-10);
}

#[test]
fn test_restart_parameter_invalidates_workspace() {
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    
    // Create test problem
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    
    // Setup with default restart
    ksp.setup(&a, 2).unwrap();
    assert!(ksp.is_setup());
    
    // Changing restart should invalidate workspace
    ksp.set_restart(100);
    assert!(!ksp.is_setup());
}

#[test]
fn test_invalidate_setup() {
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    
    // Create test problem
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    
    // Setup
    ksp.setup(&a, 2).unwrap();
    assert!(ksp.is_setup());
    
    // Invalidate
    ksp.invalidate_setup();
    assert!(!ksp.is_setup());
}

#[test]
fn test_workspace_size_compatibility() {
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::None).unwrap();
    
    // Setup for size 2
    let a2 = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    ksp.setup(&a2, 2).unwrap();
    assert!(ksp.is_setup());
    
    // Try to solve with different size (should fail)
    let a3 = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
    let b3 = vec![1.0, 2.0, 3.0];
    let mut x3 = vec![0.0, 0.0, 0.0];
    
    let result = ksp.solve(&a3, &b3, &mut x3);
    assert!(result.is_err());
    
    // Error message should mention workspace compatibility
    if let Err(e) = result {
        assert!(e.to_string().contains("Workspace incompatible"));
    }
}

#[test]
fn test_no_solver_configured_error() {
    let mut ksp = KspContext::new();
    // Don't configure any solver
    
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let b = vec![1.0, 2.0];
    let mut x = vec![0.0, 0.0];
    
    let result = ksp.solve(&a, &b, &mut x);
    assert!(result.is_err());
    
    if let Err(e) = result {
        assert!(e.to_string().contains("No solver configured"));
    }
}
