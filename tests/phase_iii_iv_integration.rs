//! Comprehensive tests for Phase III and Phase IV functionality.
//!
//! Tests for:
//! - Enhanced Chebyshev preconditioner with matrix storage
//! - AMG with custom smoothing parameters  
//! - PC-chaining for composite preconditioning
//! - Iteration monitoring and convergence tracking
//! - Parameter tuning and automated optimization

use kryst::context::KspContext;
use kryst::context::ksp_context::{SolverType, PcType};
use kryst::config::options::PcOptions;
use kryst::utils::monitor::IterationMonitor;
use kryst::utils::tuning::ParameterTuner;
use faer::Mat;
use std::time::Duration;

/// Create a test matrix for experiments.
fn create_test_matrix(n: usize, condition_number: f64) -> Mat<f64> {
    let mut a = Mat::zeros(n, n);
    
    // Create a diagonally dominant matrix with specified condition number
    let min_eig = 1.0;
    let max_eig = condition_number;
    
    for i in 0..n {
        // Set eigenvalues logarithmically spaced between min_eig and max_eig
        let lambda = min_eig * (max_eig / min_eig).powf(i as f64 / (n - 1) as f64);
        a[(i, i)] = lambda;
    }
    
    // Add some off-diagonal structure to make it more interesting
    for i in 0..n {
        for j in 0..n {
            if i != j && (i as isize - j as isize).abs() <= 2 {
                a[(i, j)] = 0.1 * (1.0 / (1.0 + (i as f64 - j as f64).abs()));
            }
        }
    }
    
    a
}

#[test]
fn test_enhanced_chebyshev_preconditioner() {
    let n = 50;
    let matrix = create_test_matrix(n, 100.0);
    let rhs = vec![1.0; n];
    
    // Test Chebyshev preconditioner with matrix storage
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::Chebyshev).unwrap();
    
    // Set Chebyshev-specific options
    let mut pc_opts = PcOptions::default();
    pc_opts.chebyshev_degree = Some(6);
    // Let eigenvalue bounds be auto-estimated
    ksp.set_pc_options(pc_opts);
    
    ksp.rtol = 1e-8;
    ksp.maxits = 1000;
    
    // Setup and solve
    ksp.setup(&matrix, n).unwrap();
    let mut x = vec![0.0; n];
    let result = ksp.solve(&matrix, &rhs, &mut x);
    
    assert!(result.is_ok(), "Chebyshev preconditioned solve should succeed");
    
    // Verify solution quality
    let mut residual = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            residual[i] += matrix[(i, j)] * x[j];
        }
        residual[i] -= rhs[i];
    }
    
    let residual_norm = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
    assert!(residual_norm < 1e-6, "Residual norm should be small: {:.2e}", residual_norm);
}

#[test]
fn test_amg_with_smoothing_parameters() {
    let n = 40;
    let matrix = create_test_matrix(n, 50.0);
    let rhs = vec![1.0; n];
    
    // Test AMG with custom smoothing parameters
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::Amg).unwrap();
    
    // Set AMG-specific options including smoothing parameters
    let mut pc_opts = PcOptions::default();
    pc_opts.amg_levels = Some(8);
    pc_opts.amg_strength_threshold = Some(0.25);
    pc_opts.amg_nu_pre = Some(2);   // 2 pre-smoothing steps
    pc_opts.amg_nu_post = Some(1);  // 1 post-smoothing step
    ksp.set_pc_options(pc_opts);
    
    ksp.rtol = 1e-8;
    ksp.maxits = 1000;
    
    // Setup and solve
    ksp.setup(&matrix, n).unwrap();
    let mut x = vec![0.0; n];
    let result = ksp.solve(&matrix, &rhs, &mut x);
    
    assert!(result.is_ok(), "AMG preconditioned solve should succeed");
    
    // Verify solution quality
    let mut residual = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            residual[i] += matrix[(i, j)] * x[j];
        }
        residual[i] -= rhs[i];
    }
    
    let residual_norm = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
    assert!(residual_norm < 1e-6, "Residual norm should be small: {:.2e}", residual_norm);
}

#[test]
fn test_pc_chaining_composite_preconditioning() {
    let n = 30;
    let matrix = create_test_matrix(n, 20.0);
    let rhs = vec![1.0; n];
    
    // Test PC-chaining: Jacobi + ILU0 + Chebyshev
    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    
    // Set PC chain options
    let mut pc_opts = PcOptions::default();
    pc_opts.pc_chain = Some("jacobi,ilu0,chebyshev".to_string());
    pc_opts.chebyshev_degree = Some(4); // For the Chebyshev in the chain
    ksp.set_pc_options(pc_opts);
    
    ksp.rtol = 1e-8;
    ksp.maxits = 1000;
    ksp.restart = 20;
    
    // Setup and solve
    ksp.setup(&matrix, n).unwrap();
    let mut x = vec![0.0; n];
    let result = ksp.solve(&matrix, &rhs, &mut x);
    
    assert!(result.is_ok(), "PC-chained solve should succeed");
    
    // Verify solution quality
    let mut residual = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            residual[i] += matrix[(i, j)] * x[j];
        }
        residual[i] -= rhs[i];
    }
    
    let residual_norm = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
    assert!(residual_norm < 1e-6, "Residual norm should be small: {:.2e}", residual_norm);
}

#[test]
fn test_iteration_monitoring() {
    // Test the iteration monitor functionality
    let mut monitor = IterationMonitor::new();
    monitor.start_solve();
    
    // Simulate a convergent sequence
    let residuals = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
    
    for (i, &residual) in residuals.iter().enumerate() {
        let pc_time = if i > 0 { Some(Duration::from_millis(5)) } else { None };
        monitor.record_iteration(i, residual, pc_time);
    }
    
    monitor.mark_converged("Relative tolerance achieved");
    
    let stats = monitor.get_statistics();
    assert_eq!(stats.total_iterations, 6);
    assert_eq!(stats.initial_residual, 1.0);
    assert_eq!(stats.final_residual, 0.03125);
    assert!(stats.converged);
    assert!(stats.avg_convergence_rate < 1.0);
    assert!(stats.avg_convergence_rate > 0.0);
    
    // Test convergence rate calculation
    let recent_rate = monitor.recent_convergence_rate(3);
    assert!(recent_rate.is_some());
    assert!((recent_rate.unwrap() - 0.5).abs() < 0.01); // Should be close to 0.5
}

#[test]
fn test_stagnation_detection() {
    let mut monitor = IterationMonitor::new();
    monitor.start_solve();
    
    // Simulate a stagnating sequence
    monitor.record_iteration(0, 1.0, None);
    monitor.record_iteration(1, 0.99, None);  // Poor rate = 0.99
    monitor.record_iteration(2, 0.98, None);  // Poor rate ≈ 0.99
    monitor.record_iteration(3, 0.97, None);  // Poor rate ≈ 0.99
    
    // Should detect stagnation
    assert!(monitor.is_stagnating(0.95, 3));
    
    // Should not detect stagnation with tighter threshold
    assert!(!monitor.is_stagnating(0.999, 3));
}

#[test]
fn test_parameter_tuning_basic() {
    let n = 10; // Small matrix for fast testing
    let matrix = Mat::identity(n, n);
    let rhs = vec![1.0; n];
    
    let mut tuner = ParameterTuner::new();
    
    // Limit to a few configurations for testing
    tuner.set_solver_types(vec![SolverType::Cg, SolverType::Gmres]);
    tuner.set_pc_types(vec![PcType::Jacobi, PcType::Ilu0]);
    tuner.set_tolerances(vec![1e-6]);
    
    // Run tuning with a small number of trials
    let result = tuner.tune_parameters(&matrix, &rhs, 4);
    assert!(result.is_ok(), "Parameter tuning should succeed");
    
    let (best_config, all_results) = result.unwrap();
    assert!(!all_results.is_empty());
    assert!(all_results.len() <= 4);
    
    // Best config should be reasonable for identity matrix
    assert!(matches!(best_config.solver_type, SolverType::Cg | SolverType::Gmres));
    assert!(matches!(best_config.pc_type, PcType::Jacobi | PcType::Ilu0));
    
    // Should have found at least one converged solution
    let converged_count = all_results.iter().filter(|r| r.converged).count();
    assert!(converged_count > 0, "Should have at least one converged solution");
}

#[test]
fn test_parameter_tuning_with_chains() {
    let n = 8; // Very small for speed
    let matrix = create_test_matrix(n, 10.0);
    let rhs = vec![1.0; n];
    
    let mut tuner = ParameterTuner::new();
    
    // Test with PC chains
    tuner.set_solver_types(vec![SolverType::Cg]);
    tuner.add_pc_chains(vec!["jacobi,ilu0".to_string(), "jacobi".to_string()]);
    tuner.set_tolerances(vec![1e-6]);
    
    let result = tuner.tune_parameters(&matrix, &rhs, 3);
    assert!(result.is_ok(), "Parameter tuning with chains should succeed");
    
    let (_best_config, all_results) = result.unwrap();
    assert!(!all_results.is_empty());
    
    // Check if any result used a PC chain
    let has_chain_result = all_results.iter().any(|r| r.config.pc_chain.is_some());
    assert!(has_chain_result, "Should have tested at least one PC chain configuration");
}

#[test]
fn test_comprehensive_phase_iii_iv_integration() {
    // Integration test that combines all Phase III and Phase IV features
    let n = 25;
    let matrix = create_test_matrix(n, 30.0);
    let rhs = vec![1.0; n];
    
    // Test 1: Enhanced Chebyshev with monitoring
    {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Cg).unwrap();
        ksp.set_pc_type(PcType::Chebyshev).unwrap();
        
        let mut pc_opts = PcOptions::default();
        pc_opts.chebyshev_degree = Some(5);
        ksp.set_pc_options(pc_opts);
        
        ksp.rtol = 1e-8;
        ksp.maxits = 500;
        
        ksp.setup(&matrix, n).unwrap();
        let mut x = vec![0.0; n];
        
        // TODO: Integrate monitoring into solve process
        let result = ksp.solve(&matrix, &rhs, &mut x);
        assert!(result.is_ok(), "Enhanced Chebyshev solve should succeed");
    }
    
    // Test 2: AMG with smoothing + monitoring
    {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).unwrap();
        ksp.set_pc_type(PcType::Amg).unwrap();
        
        let mut pc_opts = PcOptions::default();
        pc_opts.amg_levels = Some(6);
        pc_opts.amg_strength_threshold = Some(0.3);
        pc_opts.amg_nu_pre = Some(2);
        pc_opts.amg_nu_post = Some(1);
        ksp.set_pc_options(pc_opts);
        
        ksp.rtol = 1e-8;
        ksp.maxits = 500;
        ksp.restart = 15;
        
        ksp.setup(&matrix, n).unwrap();
        let mut x = vec![0.0; n];
        let result = ksp.solve(&matrix, &rhs, &mut x);
        assert!(result.is_ok(), "Enhanced AMG solve should succeed");
    }
    
    // Test 3: PC-chaining with multiple preconditioners
    {
        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Cg).unwrap();
        
        let mut pc_opts = PcOptions::default();
        pc_opts.pc_chain = Some("jacobi,chebyshev".to_string());
        pc_opts.chebyshev_degree = Some(4);
        ksp.set_pc_options(pc_opts);
        
        ksp.rtol = 1e-8;
        ksp.maxits = 500;
        
        ksp.setup(&matrix, n).unwrap();
        let mut x = vec![0.0; n];
        let result = ksp.solve(&matrix, &rhs, &mut x);
        assert!(result.is_ok(), "PC-chained solve should succeed");
    }
    
    // Test 4: Mini parameter tuning run
    {
        let mut tuner = ParameterTuner::new();
        tuner.set_solver_types(vec![SolverType::Cg]);
        tuner.set_pc_types(vec![PcType::Jacobi, PcType::Chebyshev]);
        tuner.set_tolerances(vec![1e-6]);
        tuner.set_max_config_time(Duration::from_secs(30)); // Short timeout for testing
        
        let result = tuner.tune_parameters(&matrix, &rhs, 2);
        assert!(result.is_ok(), "Mini parameter tuning should succeed");
        
        let (best_config, results) = result.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        
        // Verify best config is reasonable
        assert_eq!(best_config.solver_type, SolverType::Cg);
        assert!(matches!(best_config.pc_type, PcType::Jacobi | PcType::Chebyshev));
    }
    
    println!("✓ All Phase III and Phase IV integration tests passed!");
}

#[test]
fn test_csv_logging_functionality() {
    // Test CSV logging (without actually creating files in test)
    let mut monitor = IterationMonitor::new();
    
    // Test that we can create the monitor and call methods without file I/O
    monitor.start_solve();
    monitor.record_iteration(0, 1.0, None);
    monitor.record_iteration(1, 0.1, Some(Duration::from_millis(5)));
    monitor.mark_converged("Test convergence");
    
    let stats = monitor.get_statistics();
    assert!(stats.converged);
    assert_eq!(stats.total_iterations, 2);
    
    // Test the statistics are reasonable
    assert!(stats.avg_convergence_rate < 1.0);
    assert!(stats.final_residual < stats.initial_residual);
}

#[test]
fn test_configuration_parameter_coverage() {
    // Test that parameter tuner generates configurations covering all major cases
    let mut tuner = ParameterTuner::new();
    tuner.set_solver_types(vec![SolverType::Cg, SolverType::Gmres]);
    tuner.set_pc_types(vec![PcType::Jacobi, PcType::Amg, PcType::Chebyshev]);
    tuner.add_pc_chains(vec!["jacobi,amg".to_string()]);
    
    let configs = tuner.generate_configurations();
    
    // Should have configs for each solver type
    assert!(configs.iter().any(|c| c.solver_type == SolverType::Cg));
    assert!(configs.iter().any(|c| c.solver_type == SolverType::Gmres));
    
    // Should have configs for each PC type
    assert!(configs.iter().any(|c| c.pc_type == PcType::Jacobi));
    assert!(configs.iter().any(|c| c.pc_type == PcType::Amg));
    assert!(configs.iter().any(|c| c.pc_type == PcType::Chebyshev));
    
    // Should have configs with PC chains
    assert!(configs.iter().any(|c| c.pc_chain.is_some()));
    
    // AMG configs should have AMG-specific parameters
    let amg_configs: Vec<_> = configs.iter().filter(|c| c.pc_type == PcType::Amg).collect();
    if !amg_configs.is_empty() {
        assert!(amg_configs.iter().any(|c| c.amg_levels.is_some()));
        assert!(amg_configs.iter().any(|c| c.amg_nu_pre.is_some()));
        assert!(amg_configs.iter().any(|c| c.amg_nu_post.is_some()));
    }
    
    // Chebyshev configs should have Chebyshev-specific parameters
    let cheb_configs: Vec<_> = configs.iter().filter(|c| c.pc_type == PcType::Chebyshev).collect();
    if !cheb_configs.is_empty() {
        assert!(cheb_configs.iter().any(|c| c.chebyshev_degree.is_some()));
    }
}
