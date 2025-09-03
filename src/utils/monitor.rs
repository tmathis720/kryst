//! Iteration monitoring and convergence logging for kryst.
//!
//! This module provides real-time monitoring of solver convergence, parameter tracking,
//! and automated logging for performance analysis and tuning.
//!
//! # Overview
//!
//! - Track iteration-by-iteration residual norms, convergence rates
//! - Monitor preconditioner performance and timing
//! - Log parameter effectiveness for automated tuning
//! - Export data in various formats (CSV, JSON) for analysis
//!
//! # Usage
//!
//! ```rust
//! let mut monitor = IterationMonitor::new();
//! monitor.enable_logging("solver_convergence.csv");
//!
//! // During solve iterations
//! monitor.record_iteration(iter, residual_norm, Some(pc_apply_time));
//!
//! // After solve
//! let stats = monitor.get_statistics();
//! println!("Convergence rate: {:.2e}", stats.avg_convergence_rate);
//! ```

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

use crate::preconditioner::stats::ParIluIterSample;

pub enum Event<'a> {
    IluSetupBegin {
        opts_hash: u64,
    },
    IluSetupIter {
        sample: &'a ParIluIterSample,
    },
    IluSetupEnd {
        iters: u32,
        converged: bool,
        setup_time_s: f64,
    },
}

pub trait Monitor: Send + Sync {
    fn on_event(&self, ev: Event<'_>);
}

pub struct NullMonitor;
impl Monitor for NullMonitor {
    fn on_event(&self, _: Event<'_>) {}
}

pub struct TextMonitor {
    pub rank0: bool,
}

impl Monitor for TextMonitor {
    fn on_event(&self, ev: Event<'_>) {
        #[cfg(feature = "logging")]
        if self.rank0 {
            match ev {
                Event::IluSetupBegin { opts_hash } => {
                    log::info!("ILU: setup begin (opts={:016x})", opts_hash)
                }
                Event::IluSetupIter { sample } => log::info!(
                    "ILU: it {:>3}  corr(L)={:.3e}  corr(U)={:.3e}  res≈{:.3e}  maxΔpivot={:.3e}",
                    sample.iter,
                    sample.rel_corr_L,
                    sample.rel_corr_U,
                    sample.rel_residual,
                    sample.max_pivot_rel
                ),
                Event::IluSetupEnd {
                    iters,
                    converged,
                    setup_time_s,
                } => log::info!(
                    "ILU: setup end iters={} converged={} time={:.3}s",
                    iters,
                    converged,
                    setup_time_s
                ),
            }
        }
    }
}

/// Convergence statistics computed from iteration history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStats {
    /// Total number of iterations performed
    pub total_iterations: usize,
    /// Initial residual norm
    pub initial_residual: f64,
    /// Final residual norm  
    pub final_residual: f64,
    /// Average convergence rate per iteration
    pub avg_convergence_rate: f64,
    /// Best (fastest) convergence rate observed
    pub best_convergence_rate: f64,
    /// Worst (slowest) convergence rate observed
    pub worst_convergence_rate: f64,
    /// Total solve time
    pub total_solve_time: Duration,
    /// Average time per iteration
    pub avg_iteration_time: Duration,
    /// Average preconditioner application time per iteration
    pub avg_pc_time: Option<Duration>,
    /// Did the solver converge?
    pub converged: bool,
    /// Final convergence criterion met
    pub convergence_reason: String,
}

/// Data for a single iteration.
#[derive(Debug, Clone)]
pub struct IterationData {
    /// Iteration number (0-indexed)
    pub iteration: usize,
    /// Residual norm at this iteration
    pub residual_norm: f64,
    /// Convergence rate from previous iteration
    pub convergence_rate: Option<f64>,
    /// Time taken for this iteration
    pub iteration_time: Duration,
    /// Time spent in preconditioner application
    pub pc_time: Option<Duration>,
    /// Timestamp when iteration completed
    pub timestamp: Instant,
}

/// Monitor that tracks solver iteration progress and performance.
pub struct IterationMonitor {
    /// History of iteration data
    history: Vec<IterationData>,
    /// Optional file writer for CSV logging
    csv_writer: Option<BufWriter<File>>,
    /// Start time of the solve
    solve_start_time: Option<Instant>,
    /// Maximum number of iterations to store in history
    max_history: usize,
    /// Whether to compute convergence rates
    compute_rates: bool,
    /// Current convergence status
    converged: bool,
    /// Reason for convergence/divergence
    convergence_reason: String,
}

impl IterationMonitor {
    /// Create a new iteration monitor.
    ///
    /// # Arguments
    /// * `max_history` - Maximum number of iterations to keep in memory (0 = unlimited)
    pub fn new_with_capacity(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            csv_writer: None,
            solve_start_time: None,
            max_history,
            compute_rates: true,
            converged: false,
            convergence_reason: "In progress".to_string(),
        }
    }

    /// Create a new iteration monitor with default settings.
    pub fn new() -> Self {
        Self::new_with_capacity(1000) // Keep last 1000 iterations by default
    }

    /// Enable CSV logging to a file.
    ///
    /// # Arguments
    /// * `filename` - Path to CSV file for logging iteration data
    pub fn enable_csv_logging(&mut self, filename: &str) -> Result<(), std::io::Error> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        // Write CSV header (elapsed time since solve start in seconds)
        writeln!(
            writer,
            "iteration,residual_norm,convergence_rate,iteration_time_ms,pc_time_ms,elapsed_s"
        )?;

        self.csv_writer = Some(writer);
        Ok(())
    }

    /// Start monitoring a new solve.
    pub fn start_solve(&mut self) {
        self.solve_start_time = Some(Instant::now());
        self.history.clear();
        self.converged = false;
        self.convergence_reason = "In progress".to_string();
    }

    /// Record data for a single iteration.
    ///
    /// # Arguments
    /// * `iteration` - Iteration number (0-indexed)
    /// * `residual_norm` - Current residual norm
    /// * `pc_time` - Optional time spent in preconditioner application
    pub fn record_iteration(
        &mut self,
        iteration: usize,
        residual_norm: f64,
        pc_time: Option<Duration>,
    ) {
        let now = Instant::now();
        let iteration_time = if iteration == 0 {
            Duration::from_nanos(0)
        } else if let Some(prev) = self.history.last() {
            now.duration_since(prev.timestamp)
        } else {
            Duration::from_nanos(0)
        };

        // Compute convergence rate
        let convergence_rate = if iteration > 0 && self.compute_rates {
            if let Some(prev) = self.history.last() {
                if prev.residual_norm > 0.0 && residual_norm > 0.0 {
                    Some(residual_norm / prev.residual_norm)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let iter_data = IterationData {
            iteration,
            residual_norm,
            convergence_rate,
            iteration_time,
            pc_time,
            timestamp: now,
        };

        // Add to history with capacity management
        if self.max_history > 0 && self.history.len() >= self.max_history {
            self.history.remove(0); // Remove oldest
        }
        self.history.push(iter_data.clone());

        // Write to CSV if enabled
        if let Some(ref mut writer) = self.csv_writer {
            let rate_str = iter_data
                .convergence_rate
                .map(|r| format!("{:.6e}", r))
                .unwrap_or_default();
            let pc_time_str = iter_data
                .pc_time
                .map(|t| format!("{:.3}", t.as_secs_f64() * 1000.0))
                .unwrap_or_default();

            // Elapsed since solve start; default to 0.0s if not started
            let elapsed_s = self
                .solve_start_time
                .map(|t0| t0.elapsed().as_secs_f64())
                .unwrap_or(0.0);

            let _ = writeln!(
                writer,
                "{},{:.6e},{},{:.3},{},{:.6}",
                iteration,
                residual_norm,
                rate_str,
                iteration_time.as_secs_f64() * 1000.0,
                pc_time_str,
                elapsed_s
            );
            let _ = writer.flush();
        }
    }

    /// Mark the solve as converged with a reason.
    pub fn mark_converged(&mut self, reason: &str) {
        self.converged = true;
        self.convergence_reason = reason.to_string();
    }

    /// Mark the solve as diverged with a reason.
    pub fn mark_diverged(&mut self, reason: &str) {
        self.converged = false;
        self.convergence_reason = reason.to_string();
    }

    /// Get comprehensive convergence statistics.
    pub fn get_statistics(&self) -> ConvergenceStats {
        let total_iterations = self.history.len();

        let initial_residual = self.history.first().map(|d| d.residual_norm).unwrap_or(0.0);

        let final_residual = self.history.last().map(|d| d.residual_norm).unwrap_or(0.0);

        let total_solve_time = self
            .solve_start_time
            .and_then(|start| {
                self.history
                    .last()
                    .map(|last| last.timestamp.duration_since(start))
            })
            .unwrap_or_default();

        let avg_iteration_time = if total_iterations > 1 {
            let total_nanos = self
                .history
                .iter()
                .skip(1) // Skip first iteration (setup time)
                .map(|d| d.iteration_time.as_nanos())
                .sum::<u128>();
            let avg_nanos = (total_nanos / ((total_iterations - 1) as u128)) as u64;
            Duration::from_nanos(avg_nanos)
        } else {
            Duration::from_nanos(0)
        };

        let avg_pc_time = {
            let pc_times: Vec<_> = self.history.iter().filter_map(|d| d.pc_time).collect();
            if !pc_times.is_empty() {
                let total_pc_nanos: u128 = pc_times.iter().map(|t| t.as_nanos()).sum();
                let avg_nanos = (total_pc_nanos / (pc_times.len() as u128)) as u64;
                Some(Duration::from_nanos(avg_nanos))
            } else {
                None
            }
        };

        let rates: Vec<f64> = self
            .history
            .iter()
            .filter_map(|d| d.convergence_rate)
            .collect();

        let (avg_convergence_rate, best_convergence_rate, worst_convergence_rate) =
            if !rates.is_empty() {
                let avg = rates.iter().sum::<f64>() / (rates.len() as f64);
                let best = rates.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let worst = rates.iter().fold(0.0f64, |a, &b| a.max(b));
                (avg, best, worst)
            } else {
                (1.0, 1.0, 1.0) // Default to no convergence
            };

        ConvergenceStats {
            total_iterations,
            initial_residual,
            final_residual,
            avg_convergence_rate,
            best_convergence_rate,
            worst_convergence_rate,
            total_solve_time,
            avg_iteration_time,
            avg_pc_time,
            converged: self.converged,
            convergence_reason: self.convergence_reason.clone(),
        }
    }

    /// Get the current iteration count.
    pub fn current_iteration(&self) -> usize {
        self.history.len()
    }

    /// Get the current residual norm (if any iterations recorded).
    pub fn current_residual(&self) -> Option<f64> {
        self.history.last().map(|d| d.residual_norm)
    }

    /// Get the recent convergence rate (average of last few iterations).
    pub fn recent_convergence_rate(&self, window: usize) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }

        let start_idx = self
            .history
            .len()
            .saturating_sub(window.min(self.history.len()));
        let recent_rates: Vec<f64> = self.history[start_idx..]
            .iter()
            .filter_map(|d| d.convergence_rate)
            .collect();

        if recent_rates.is_empty() {
            None
        } else {
            Some(recent_rates.iter().sum::<f64>() / (recent_rates.len() as f64))
        }
    }

    /// Check if convergence has stagnated (poor convergence rate recently).
    pub fn is_stagnating(&self, threshold: f64, window: usize) -> bool {
        if let Some(recent_rate) = self.recent_convergence_rate(window) {
            recent_rate > threshold
        } else {
            false
        }
    }
}

impl Default for IterationMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IterationMonitor {
    fn drop(&mut self) {
        // Ensure CSV file is properly closed
        if let Some(ref mut writer) = self.csv_writer {
            let _ = writer.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_monitor_basic_functionality() {
        let mut monitor = IterationMonitor::new();
        monitor.start_solve();

        // Record some iterations
        monitor.record_iteration(0, 1.0, None);
        monitor.record_iteration(1, 0.5, Some(Duration::from_millis(10)));
        monitor.record_iteration(2, 0.25, Some(Duration::from_millis(12)));

        monitor.mark_converged("Relative tolerance achieved");

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_iterations, 3);
        assert_eq!(stats.initial_residual, 1.0);
        assert_eq!(stats.final_residual, 0.25);
        assert!(stats.converged);
        assert!(stats.avg_convergence_rate < 1.0); // Should be improving
    }

    #[test]
    fn test_convergence_rate_calculation() {
        let mut monitor = IterationMonitor::new();
        monitor.start_solve();

        monitor.record_iteration(0, 1.0, None);
        monitor.record_iteration(1, 0.1, None); // Rate = 0.1
        monitor.record_iteration(2, 0.01, None); // Rate = 0.1

        let recent_rate = monitor.recent_convergence_rate(2);
        assert!((recent_rate.unwrap() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_stagnation_detection() {
        let mut monitor = IterationMonitor::new();
        monitor.start_solve();

        monitor.record_iteration(0, 1.0, None);
        monitor.record_iteration(1, 0.99, None); // Poor rate = 0.99
        monitor.record_iteration(2, 0.98, None); // Poor rate ≈ 0.99

        assert!(monitor.is_stagnating(0.95, 2));
        assert!(!monitor.is_stagnating(0.999, 2));
    }
}
