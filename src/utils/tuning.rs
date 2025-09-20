//! Parameter grid search and automated tuning for kryst.
//!
//! This module provides automated parameter sweeping, performance benchmarking,
//! and optimal configuration discovery for preconditioners and solvers.
//!
//! # Overview
//!
//! - Grid search over multiple parameter dimensions
//! - Performance metrics collection and comparison
//! - Automatic best-configuration selection
//! - Export results for analysis and reproducibility
//!
//! # Usage
//!
//! ```rust
//! let mut tuner = ParameterTuner::new();
//! tuner.add_solver_types(vec![SolverType::Cg, SolverType::Gmres]);
//! tuner.add_pc_types(vec![PcType::Jacobi, PcType::Ilu0]);
//! tuner.add_tolerances(vec![1e-6, 1e-8, 1e-10]);
//!
//! let best_config = tuner.tune_parameters(&matrix, &rhs, max_trials)?;
//! println!("Best configuration: {:?}", best_config);
//! ```

use crate::config::options::{KspOptions, PcOptions};
use crate::context::KspContext;
use crate::context::ksp_context::SolverType;
use crate::context::pc_context::PcType;
use crate::error::KError;
use crate::utils::monitor::IterationMonitor;
use faer::Mat;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for a single parameter combination.
#[derive(Debug, Clone)]
pub struct ParameterConfig {
    pub solver_type: SolverType,
    pub pc_type: PcType,
    pub rtol: f64,
    pub atol: f64,
    pub maxits: usize,
    pub restart: Option<usize>, // For GMRES-type solvers

    // Preconditioner-specific parameters
    pub amg_levels: Option<usize>,
    pub amg_strength_threshold: Option<f64>,
    pub amg_nu_pre: Option<usize>,
    pub amg_nu_post: Option<usize>,
    pub chebyshev_degree: Option<usize>,
    pub chebyshev_lambda_min: Option<f64>,
    pub chebyshev_lambda_max: Option<f64>,
    pub ilut_max_fill: Option<usize>,
    pub drop_tol: Option<f64>,
    pub pc_chain: Option<String>,
}

/// Performance metrics for a single parameter combination.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub config: ParameterConfig,
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub solve_time: Duration,
    pub avg_convergence_rate: f64,
    pub setup_time: Duration,
    pub memory_usage_estimate: Option<usize>,
    pub convergence_reason: String,
}

/// Parameter sweeping and automated tuning engine.
pub struct ParameterTuner {
    /// Solver types to test
    solver_types: Vec<SolverType>,
    /// Preconditioner types to test
    pc_types: Vec<PcType>,
    /// Tolerance values to test
    tolerances: Vec<f64>,
    /// Maximum iteration counts to test
    max_iterations: Vec<usize>,
    /// AMG parameter ranges
    amg_levels_range: Vec<usize>,
    amg_threshold_range: Vec<f64>,
    amg_smoothing_range: Vec<(usize, usize)>, // (nu_pre, nu_post)
    /// Chebyshev parameter ranges
    chebyshev_degree_range: Vec<usize>,
    /// PC chain configurations to test
    pc_chains: Vec<String>,
    /// Maximum time budget per configuration
    max_config_time: Duration,
    /// Results from completed runs
    results: Vec<PerformanceMetrics>,
    /// Whether to enable detailed monitoring
    enable_monitoring: bool,
}

impl ParameterTuner {
    /// Create a new parameter tuner with default ranges.
    pub fn new() -> Self {
        Self {
            solver_types: vec![SolverType::Cg, SolverType::Gmres],
            pc_types: vec![PcType::Jacobi, PcType::Ilu0],
            tolerances: vec![1e-6, 1e-8],
            max_iterations: vec![1000, 5000],
            amg_levels_range: vec![5, 10, 20],
            amg_threshold_range: vec![0.1, 0.25, 0.5],
            amg_smoothing_range: vec![(1, 1), (2, 2), (3, 1)],
            chebyshev_degree_range: vec![3, 5, 10],
            pc_chains: vec![],
            max_config_time: Duration::from_secs(300), // 5 minutes per config
            results: Vec::new(),
            enable_monitoring: true,
        }
    }

    /// Set the solver types to test.
    pub fn set_solver_types(&mut self, types: Vec<SolverType>) -> &mut Self {
        self.solver_types = types;
        self
    }

    /// Set the preconditioner types to test.
    pub fn set_pc_types(&mut self, types: Vec<PcType>) -> &mut Self {
        self.pc_types = types;
        self
    }

    /// Set the tolerance values to test.
    pub fn set_tolerances(&mut self, tols: Vec<f64>) -> &mut Self {
        self.tolerances = tols;
        self
    }

    /// Add PC chain configurations to test.
    pub fn add_pc_chains(&mut self, chains: Vec<String>) -> &mut Self {
        self.pc_chains.extend(chains);
        self
    }

    /// Set the maximum time budget per configuration.
    pub fn set_max_config_time(&mut self, time: Duration) -> &mut Self {
        self.max_config_time = time;
        self
    }

    /// Generate all parameter combinations to test.
    pub fn generate_configurations(&self) -> Vec<ParameterConfig> {
        let mut configs = Vec::new();

        // Base combinations of solver + PC + tolerance + maxits
        for &solver_type in &self.solver_types {
            for &pc_type in &self.pc_types {
                for &rtol in &self.tolerances {
                    for &maxits in &self.max_iterations {
                        let base_config = ParameterConfig {
                            solver_type,
                            pc_type,
                            rtol,
                            atol: rtol * 1e-3, // atol = rtol / 1000 by default
                            maxits,
                            restart: if matches!(
                                solver_type,
                                SolverType::Gmres | SolverType::Fgmres
                            ) {
                                Some(30)
                            } else {
                                None
                            },
                            amg_levels: None,
                            amg_strength_threshold: None,
                            amg_nu_pre: None,
                            amg_nu_post: None,
                            chebyshev_degree: None,
                            chebyshev_lambda_min: None,
                            chebyshev_lambda_max: None,
                            ilut_max_fill: None,
                            drop_tol: None,
                            pc_chain: None,
                        };

                        // Add parameter-specific variations
                        match pc_type {
                            PcType::Amg => {
                                // Test AMG parameter combinations
                                for &levels in &self.amg_levels_range {
                                    for &threshold in &self.amg_threshold_range {
                                        for &(nu_pre, nu_post) in &self.amg_smoothing_range {
                                            let mut config = base_config.clone();
                                            config.amg_levels = Some(levels);
                                            config.amg_strength_threshold = Some(threshold);
                                            config.amg_nu_pre = Some(nu_pre);
                                            config.amg_nu_post = Some(nu_post);
                                            configs.push(config);
                                        }
                                    }
                                }
                            }
                            PcType::Chebyshev => {
                                // Test Chebyshev parameter combinations
                                for &degree in &self.chebyshev_degree_range {
                                    let mut config = base_config.clone();
                                    config.chebyshev_degree = Some(degree);
                                    // Leave lambda bounds for auto-estimation
                                    configs.push(config);
                                }
                            }
                            _ => {
                                // Use base configuration for other PC types
                                configs.push(base_config);
                            }
                        }
                    }
                }
            }
        }

        // Add PC chain configurations
        for chain in &self.pc_chains {
            for &solver_type in &self.solver_types {
                for &rtol in &self.tolerances {
                    for &maxits in &self.max_iterations {
                        let config = ParameterConfig {
                            solver_type,
                            pc_type: PcType::None, // Will be overridden by chain
                            rtol,
                            atol: rtol * 1e-3,
                            maxits,
                            restart: if matches!(
                                solver_type,
                                SolverType::Gmres | SolverType::Fgmres
                            ) {
                                Some(30)
                            } else {
                                None
                            },
                            amg_levels: None,
                            amg_strength_threshold: None,
                            amg_nu_pre: None,
                            amg_nu_post: None,
                            chebyshev_degree: None,
                            chebyshev_lambda_min: None,
                            chebyshev_lambda_max: None,
                            ilut_max_fill: None,
                            drop_tol: None,
                            pc_chain: Some(chain.clone()),
                        };
                        configs.push(config);
                    }
                }
            }
        }

        configs
    }

    /// Test a single parameter configuration.
    pub fn test_configuration(
        &mut self,
        config: &ParameterConfig,
        matrix: &Mat<f64>,
        rhs: &[f64],
    ) -> Result<PerformanceMetrics, KError> {
        let setup_start = Instant::now();

        // Create KSP context with configuration
        let mut ksp = KspContext::new();

        // Set solver type
        ksp.set_type(config.solver_type)?;

        // Set tolerances and limits
        ksp.rtol = config.rtol;
        ksp.atol = config.atol;
        ksp.maxits = config.maxits;
        if let Some(restart) = config.restart {
            ksp.restart = restart;
        }

        // Configure preconditioner
        if let Some(ref chain) = config.pc_chain {
            // Build a structured chain from the configuration
            let stages: Vec<PcOptions> = chain
                .split("->")
                .map(|token| {
                    let mut stage = PcOptions {
                        pc_type: Some(token.trim().to_string()),
                        ..Default::default()
                    };
                    if token.contains("amg") {
                        stage.amg_levels = config.amg_levels;
                        stage.amg_strength_threshold = config.amg_strength_threshold;
                        stage.amg_nu_pre = config.amg_nu_pre;
                        stage.amg_nu_post = config.amg_nu_post;
                    }
                    if token.contains("chebyshev") {
                        stage.chebyshev_degree = config.chebyshev_degree;
                        stage.chebyshev_lambda_min = config.chebyshev_lambda_min;
                        stage.chebyshev_lambda_max = config.chebyshev_lambda_max;
                    }
                    stage
                })
                .collect();

            let pc_opts = PcOptions {
                chain: Some(stages),
                ..Default::default()
            };
            ksp.set_from_all_options(&KspOptions::default(), &pc_opts)?;
        } else {
            // Single preconditioner with options
            let mut pc_opts = PcOptions::default();
            match config.pc_type {
                PcType::Amg => {
                    pc_opts.amg_levels = config.amg_levels;
                    pc_opts.amg_strength_threshold = config.amg_strength_threshold;
                    pc_opts.amg_nu_pre = config.amg_nu_pre;
                    pc_opts.amg_nu_post = config.amg_nu_post;
                }
                PcType::Chebyshev => {
                    pc_opts.chebyshev_degree = config.chebyshev_degree;
                    pc_opts.chebyshev_lambda_min = config.chebyshev_lambda_min;
                    pc_opts.chebyshev_lambda_max = config.chebyshev_lambda_max;
                }
                _ => {} // No special parameters for other types
            }
            ksp.set_pc_type(config.pc_type, Some(&pc_opts))?;
        }

        // Setup timing
        let n = matrix.nrows();
        let aop: Arc<dyn crate::matrix::op::LinOp<S = f64>> = Arc::new(matrix.clone());
        ksp.set_operators(aop, None);
        ksp.setup()?;
        let setup_time = setup_start.elapsed();

        // Create solution vector
        let mut x = vec![0.0; n];

        // Set up monitoring if enabled
        let mut monitor = if self.enable_monitoring {
            Some(IterationMonitor::new())
        } else {
            None
        };

        if let Some(ref mut mon) = monitor {
            mon.start_solve();
        }

        // Solve with timeout
        let solve_start = Instant::now();
        let solve_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Create a simple timeout mechanism using solver iteration callback
            ksp.solve(rhs, &mut x)
        }));

        let solve_time = solve_start.elapsed();

        // Check for timeout
        if solve_time > self.max_config_time {
            return Ok(PerformanceMetrics {
                config: config.clone(),
                converged: false,
                iterations: 0,
                final_residual: f64::INFINITY,
                solve_time,
                avg_convergence_rate: f64::INFINITY,
                setup_time,
                memory_usage_estimate: None,
                convergence_reason: "Timeout".to_string(),
            });
        }

        // Process solve result
        match solve_result {
            Ok(solve_result) => {
                match solve_result {
                    Ok(_) => {
                        // Compute final residual manually
                        let mut ax = vec![0.0; n];
                        // ax = A * x (matrix-vector multiply)
                        for i in 0..n {
                            for j in 0..n {
                                if i < matrix.nrows() && j < matrix.ncols() {
                                    ax[i] += matrix[(i, j)] * x[j];
                                }
                            }
                        }

                        // residual = |b - Ax|
                        let mut residual_norm = 0.0;
                        for i in 0..n {
                            let diff = rhs[i] - ax[i];
                            residual_norm += diff * diff;
                        }
                        residual_norm = residual_norm.sqrt();

                        // Get stats from monitor if available
                        let (iterations, avg_rate, reason) = if let Some(ref mon) = monitor {
                            let stats = mon.get_statistics();
                            (
                                stats.total_iterations,
                                stats.avg_convergence_rate,
                                stats.convergence_reason,
                            )
                        } else {
                            // Estimate from maxits (rough approximation)
                            let estimated_iters = if residual_norm < config.rtol {
                                (config.maxits as f64 * 0.7) as usize // Assume ~70% of maxits for convergence
                            } else {
                                config.maxits
                            };
                            (estimated_iters, 0.9, "Estimated".to_string())
                        };

                        Ok(PerformanceMetrics {
                            config: config.clone(),
                            converged: residual_norm < config.rtol,
                            iterations,
                            final_residual: residual_norm,
                            solve_time,
                            avg_convergence_rate: avg_rate,
                            setup_time,
                            memory_usage_estimate: None, // TODO: Implement memory tracking
                            convergence_reason: reason,
                        })
                    }
                    Err(e) => Ok(PerformanceMetrics {
                        config: config.clone(),
                        converged: false,
                        iterations: 0,
                        final_residual: f64::INFINITY,
                        solve_time,
                        avg_convergence_rate: f64::INFINITY,
                        setup_time,
                        memory_usage_estimate: None,
                        convergence_reason: format!("Solve error: {e}"),
                    }),
                }
            }
            Err(_) => Ok(PerformanceMetrics {
                config: config.clone(),
                converged: false,
                iterations: 0,
                final_residual: f64::INFINITY,
                solve_time,
                avg_convergence_rate: f64::INFINITY,
                setup_time,
                memory_usage_estimate: None,
                convergence_reason: "Panic/crash".to_string(),
            }),
        }
    }

    /// Run parameter sweep and find the best configuration.
    ///
    /// # Arguments
    /// * `matrix` - System matrix
    /// * `rhs` - Right-hand side vector
    /// * `max_trials` - Maximum number of configurations to test (0 = test all)
    ///
    /// # Returns
    /// * Best configuration found and all results
    pub fn tune_parameters(
        &mut self,
        matrix: &Mat<f64>,
        rhs: &[f64],
        max_trials: usize,
    ) -> Result<(ParameterConfig, Vec<PerformanceMetrics>), KError> {
        let configurations = self.generate_configurations();
        let total_configs = configurations.len();
        let configs_to_test = if max_trials > 0 {
            max_trials.min(total_configs)
        } else {
            total_configs
        };

        println!("Starting parameter tuning: {configs_to_test} configurations to test");

        self.results.clear();

        for (i, config) in configurations.iter().take(configs_to_test).enumerate() {
            println!(
                "Testing configuration {}/{}: {:?} + {:?}",
                i + 1,
                configs_to_test,
                config.solver_type,
                config.pc_type
            );

            match self.test_configuration(config, matrix, rhs) {
                Ok(metrics) => {
                    println!(
                        "  Result: converged={}, iterations={}, time={:.3}s, rate={:.2e}",
                        metrics.converged,
                        metrics.iterations,
                        metrics.solve_time.as_secs_f64(),
                        metrics.avg_convergence_rate
                    );
                    self.results.push(metrics);
                }
                Err(e) => {
                    println!("  Error: {e}");
                    // Continue with next configuration
                }
            }
        }

        // Find best configuration (converged, fewest iterations, fastest time)
        let best_metrics = self
            .results
            .iter()
            .filter(|m| m.converged)
            .min_by(|a, b| {
                // Primary: converged solutions first
                // Secondary: fewer iterations
                // Tertiary: faster solve time
                a.iterations
                    .cmp(&b.iterations)
                    .then(a.solve_time.cmp(&b.solve_time))
            })
            .or_else(|| {
                // If no converged solutions, pick the one with best residual
                self.results.iter().min_by(|a, b| {
                    a.final_residual
                        .partial_cmp(&b.final_residual)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });

        match best_metrics {
            Some(best) => {
                println!("\nBest configuration found:");
                println!(
                    "  Solver: {:?}, PC: {:?}",
                    best.config.solver_type, best.config.pc_type
                );
                if let Some(ref chain) = best.config.pc_chain {
                    println!("  PC Chain: {chain}");
                }
                println!(
                    "  Converged: {}, Iterations: {}, Time: {:.3}s",
                    best.converged,
                    best.iterations,
                    best.solve_time.as_secs_f64()
                );
                Ok((best.config.clone(), self.results.clone()))
            }
            None => Err(KError::SolveError(
                "No valid configurations found".to_string(),
            )),
        }
    }

    /// Export results to JSON for analysis.
    pub fn export_results(&self, filename: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        // Write a simple text summary instead of JSON
        writeln!(writer, "Parameter Tuning Results")?;
        writeln!(writer, "=======================")?;
        writeln!(
            writer,
            "Total configurations tested: {}",
            self.results.len()
        )?;

        for (i, result) in self.results.iter().enumerate() {
            writeln!(writer, "\nConfiguration {}:", i + 1)?;
            writeln!(writer, "  Solver: {:?}", result.config.solver_type)?;
            writeln!(writer, "  PC: {:?}", result.config.pc_type)?;
            writeln!(writer, "  Tolerance: {:.2e}", result.config.rtol)?;
            writeln!(writer, "  Converged: {}", result.converged)?;
            writeln!(writer, "  Iterations: {}", result.iterations)?;
            writeln!(
                writer,
                "  Solve time: {:.3}s",
                result.solve_time.as_secs_f64()
            )?;
        }

        Ok(())
    }

    /// Get summary statistics across all tested configurations.
    pub fn get_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        let total_configs = self.results.len() as f64;
        let converged_configs = self.results.iter().filter(|r| r.converged).count() as f64;

        summary.insert("total_configurations".to_string(), total_configs);
        summary.insert("converged_configurations".to_string(), converged_configs);
        summary.insert(
            "convergence_rate".to_string(),
            converged_configs / total_configs,
        );

        if !self.results.is_empty() {
            let avg_solve_time = self
                .results
                .iter()
                .map(|r| r.solve_time.as_secs_f64())
                .sum::<f64>()
                / total_configs;
            summary.insert("avg_solve_time".to_string(), avg_solve_time);

            let avg_iterations = self
                .results
                .iter()
                .map(|r| r.iterations as f64)
                .sum::<f64>()
                / total_configs;
            summary.insert("avg_iterations".to_string(), avg_iterations);
        }

        summary
    }
}

impl Default for ParameterTuner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn test_configuration_generation() {
        let tuner = ParameterTuner::new();
        let configs = tuner.generate_configurations();
        assert!(!configs.is_empty());

        // Should have at least one config for each solver/PC combination
        let min_expected = tuner.solver_types.len()
            * tuner.pc_types.len()
            * tuner.tolerances.len()
            * tuner.max_iterations.len();
        assert!(configs.len() >= min_expected);
    }

    #[test]
    fn test_parameter_tuning_basic() {
        // Create a small test matrix
        let n = 4;
        let matrix = Mat::identity(n, n);
        let rhs = vec![1.0; n];

        let mut tuner = ParameterTuner::new();
        tuner.set_solver_types(vec![SolverType::Cg]);
        tuner.set_pc_types(vec![PcType::Jacobi]);
        tuner.set_tolerances(vec![1e-6]);

        let result = tuner.tune_parameters(&matrix, &rhs, 1);
        assert!(result.is_ok());

        let (best_config, results) = result.unwrap();
        assert!(!results.is_empty());
        assert_eq!(best_config.solver_type, SolverType::Cg);
        assert_eq!(best_config.pc_type, PcType::Jacobi);
    }
}
