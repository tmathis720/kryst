//! Invariant helpers for cheap runtime validation.
//!
//! When the `invariants` feature is enabled, violations return
//! [`KError::InvalidInput`], making test failures easier to diagnose.
//! Without the feature, invariants degrade to `debug_assert!` checks so
//! release builds stay lean.

#[cfg(feature = "invariants")]
#[macro_export]
macro_rules! invariant {
    ($cond:expr, $($msg:tt)*) => {{
        if !$cond {
            return Err(crate::error::KError::InvalidInput(format!($($msg)*)));
        }
    }};
}

#[cfg(not(feature = "invariants"))]
#[macro_export]
macro_rules! invariant {
    ($cond:expr, $($msg:tt)*) => {{
        debug_assert!($cond, $($msg)*);
    }};
}
