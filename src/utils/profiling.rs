//! Lightweight profiling utilities for Krylov solvers.
//!
//! This module provides RAII-based stage guards for timing major solver phases,
//! similar to PETSc's profiling stages. When the `logging` feature is enabled,
//! stage entries and exits are logged at trace level.

#[cfg(feature = "logging")]
use log::trace;
#[cfg(feature = "logging")]
use std::time::Instant;

/// RAII guard for profiling solver stages.
///
/// This struct automatically logs stage entry on creation and stage exit on drop.
/// When the `logging` feature is disabled, this becomes a zero-cost abstraction.
///
/// # Example
/// ```rust
/// use kryst::utils::profiling::StageGuard;
/// 
/// {
///     let _setup = StageGuard::new("KSPSetup");
///     // ... setup code ...
/// } // "leave stage: KSPSetup" logged here
/// ```
pub struct StageGuard {
    #[cfg(feature = "logging")]
    name: &'static str,
    #[cfg(feature = "logging")]
    start_time: Instant,
}

impl StageGuard {
    /// Create a new stage guard and log entry.
    ///
    /// # Arguments
    /// * `name` - Static string name of the stage
    ///
    /// # Returns
    /// * A new `StageGuard` that will log stage exit on drop
    pub fn new(name: &'static str) -> Self {
        #[cfg(feature = "logging")]
        {
            trace!("---- enter stage: {}", name);
            StageGuard {
                name,
                start_time: Instant::now(),
            }
        }
        #[cfg(not(feature = "logging"))]
        {
            let _ = name; // Silence unused variable warning
            StageGuard {}
        }
    }
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        #[cfg(feature = "logging")]
        {
            let elapsed = self.start_time.elapsed();
            trace!("---- leave stage: {} (took {:?})", self.name, elapsed);
        }
    }
}

/// Macro for timing a code block with automatic stage guard creation.
///
/// This is a convenience macro that creates a stage guard for a named block.
/// The guard is automatically dropped at the end of the block.
///
/// # Example
/// ```rust
/// use kryst::utils::profiling::time_stage;
/// 
/// time_stage!("MatVec", {
///     // ... matrix-vector multiplication code ...
/// });
/// ```
#[macro_export]
macro_rules! time_stage {
    ($name:expr, $block:block) => {
        {
            let _guard = $crate::utils::profiling::StageGuard::new($name);
            $block
        }
    };
}

/// Conditionally execute timing code only when logging feature is enabled.
///
/// This macro helps avoid the overhead of timing operations when logging
/// is disabled.
///
/// # Example
/// ```rust
/// use kryst::utils::profiling::with_timing;
/// 
/// with_timing!(|| {
///     log::trace!("Starting expensive operation");
///     let start = std::time::Instant::now();
///     // ... operation ...
///     log::trace!("Operation took {:?}", start.elapsed());
/// });
/// ```
#[macro_export]
macro_rules! with_timing {
    ($closure:expr) => {
        #[cfg(feature = "logging")]
        $closure();
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
