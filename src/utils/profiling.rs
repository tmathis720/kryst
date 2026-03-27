//! Lightweight profiling utilities for Krylov solvers.
//!
//! This module provides RAII-based stage guards for timing major solver phases,
//! similar to PETSc's profiling stages. Profiling can be enabled/disabled at runtime
//! via global configuration.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Global flag to enable/disable profiling at runtime.
static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Global flag to enable/disable monitoring at runtime.
static MONITORING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable profiling globally.
pub fn enable_profiling() {
    PROFILING_ENABLED.store(true, Ordering::Relaxed);
}

/// Disable profiling globally.
pub fn disable_profiling() {
    PROFILING_ENABLED.store(false, Ordering::Relaxed);
}

/// Check if profiling is currently enabled.
pub fn is_profiling_enabled() -> bool {
    PROFILING_ENABLED.load(Ordering::Relaxed)
}

/// Enable monitoring globally.
pub fn enable_monitoring() {
    MONITORING_ENABLED.store(true, Ordering::Relaxed);
}

/// Disable monitoring globally.
pub fn disable_monitoring() {
    MONITORING_ENABLED.store(false, Ordering::Relaxed);
}

/// Check if monitoring is currently enabled.
pub fn is_monitoring_enabled() -> bool {
    MONITORING_ENABLED.load(Ordering::Relaxed)
}

/// RAII guard for profiling solver stages.
///
/// This struct automatically logs stage entry on creation and stage exit on drop
/// when profiling is enabled. When profiling is disabled, this becomes a very
/// low-cost abstraction.
///
/// # Example
/// ```rust
/// use kryst::utils::profiling::{StageGuard, enable_profiling};
///
/// enable_profiling();
/// {
///     let _setup = StageGuard::new("KSPSetup");
///     // ... setup code ...
/// } // "leave stage: KSPSetup" logged here if profiling enabled
/// ```
pub struct StageGuard {
    name: &'static str,
    start_time: Option<Instant>,
}

impl StageGuard {
    /// Create a new stage guard and log entry if profiling is enabled.
    ///
    /// # Arguments
    /// * `name` - Static string name of the stage
    ///
    /// # Returns
    /// * A new `StageGuard` that will log stage exit on drop if profiling is enabled
    pub fn new(name: &'static str) -> Self {
        let start_time = if is_profiling_enabled() {
            #[cfg(feature = "logging")]
            log::trace!("---- enter stage: {name}");
            #[cfg(not(feature = "logging"))]
            eprintln!("---- enter stage: {}", name);
            Some(Instant::now())
        } else {
            None
        };

        StageGuard { name, start_time }
    }

    /// Get the elapsed time since the stage started.
    /// Returns None if profiling was disabled when the stage started.
    pub fn elapsed(&self) -> Option<std::time::Duration> {
        self.start_time.map(|start| start.elapsed())
    }
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        if let Some(start_time) = self.start_time {
            let elapsed = start_time.elapsed();
            #[cfg(feature = "logging")]
            log::trace!("---- leave stage: {} (took {:?})", self.name, elapsed);
            #[cfg(not(feature = "logging"))]
            eprintln!("---- leave stage: {} (took {:?})", self.name, elapsed);
        }
    }
}

/// Macro for timing a code block with automatic stage guard creation.
///
/// This is a convenience macro that creates a stage guard for a named block.
/// The guard is automatically dropped at the end of the block.
///
/// # Example
/// ```rust,ignore
/// use kryst::utils::profiling::time_stage;
///
/// time_stage!("MatVec", {
///     // ... matrix-vector multiplication code ...
/// });
/// ```
#[macro_export]
macro_rules! time_stage {
    ($name:expr, $block:block) => {{
        let _guard = $crate::utils::profiling::StageGuard::new($name);
        $block
    }};
}

/// Conditionally execute profiling code only when profiling is enabled.
///
/// This macro helps avoid the overhead of profiling operations when profiling
/// is disabled at runtime.
///
/// # Example
/// ```rust,ignore
/// use kryst::utils::profiling::with_profiling;
///
/// with_profiling!(|| {
///     eprintln!("Starting expensive operation");
///     let start = std::time::Instant::now();
///     // ... operation ...
///     eprintln!("Operation took {:?}", start.elapsed());
/// });
/// ```
#[macro_export]
macro_rules! with_profiling {
    ($closure:expr) => {
        if $crate::utils::profiling::is_profiling_enabled() {
            $closure();
        }
    };
}

/// Conditionally execute monitoring code only when monitoring is enabled.
///
/// This macro helps avoid the overhead of monitoring operations when monitoring
/// is disabled at runtime.
///
/// # Example
/// ```rust,ignore
/// use kryst::utils::profiling::with_monitoring;
///
/// with_monitoring!(|| {
///     eprintln!("Iteration {}: residual = {:.2e}", iter, residual);
/// });
/// ```
#[macro_export]
macro_rules! with_monitoring {
    ($closure:expr) => {
        if $crate::utils::profiling::is_monitoring_enabled() {
            $closure();
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_guard_creation() {
        // This should not panic regardless of logging feature
        let _guard = StageGuard::new("TestStage");
    }

    #[test]
    fn test_nested_stages() {
        let _outer = StageGuard::new("OuterStage");
        {
            let _inner = StageGuard::new("InnerStage");
            // Inner should drop first
        }
        // Outer drops last
    }
}
