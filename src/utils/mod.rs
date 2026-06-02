//! Utility modules for logging, convergence checks, graph coloring, reordering, and profiling.

pub mod buffer_pool;
pub mod coloring;
pub mod conditioning;
pub mod convergence;
pub mod diagnostics;
pub mod direct_reference;
pub mod invariants;
#[cfg(feature = "backend-faer")]
pub mod matrix_market;
pub mod matrix_screening;
pub mod merge;
pub mod metrics;
pub mod monitor;
pub mod partition;
pub mod permutation;
#[cfg(feature = "backend-faer")]
pub mod preconditioning_pipeline;
pub mod profiling;
pub mod reduction;
#[cfg(feature = "backend-faer")]
pub mod reordering;
pub mod solver_ladder;
pub mod solver_policy;
#[cfg(all(feature = "tuning", not(feature = "complex")))]
pub mod tuning;
pub mod verification;

pub use convergence::{AcceptanceStatus, classify_acceptance_status};
pub use diagnostics::{DirectVerificationCapability, format_direct_verification_status};
pub use direct_reference::{
    DirectReferenceComparison, DirectReferencePolicyInput, compare_solution_vectors,
    direct_reference_policy, global_direct_reference_policy_allows,
};
pub use metrics::true_residual_norm;
pub use monitor::{Event, Monitor, NullMonitor, TextMonitor};
pub use solver_ladder::{
    AcceptanceContract, AttemptRecord, FallbackStep, SolverTestResult, TruthReference,
    classify_acceptance, classify_failure, execute_fallback_ladder, render_attempt_chain,
    solver_reason_code,
};
pub use verification::{
    DirectReferenceLike, VerificationStatus, verification_status_from_direct_reference,
};
