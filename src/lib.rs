#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # kryst
//!
//! PETSc-like KSP/PC in Rust with MPI-friendly design.
//!
//! ## Architecture decisions (read first)
//! - [`LinOp::comm()`] is the single source of truth for parallel context (ADR-001).
//! - Preconditioners compute **z = M⁻¹ x**; solvers decide left/right placement (ADR-002).
//! - Flexible PCs use [`Preconditioner::apply_mut`] (FGMRES, etc.).
//! - Options map to builders; no in-solver construction.
//!
//! See `docs/adr/` for rationale and trade-offs.

pub mod parallel;

pub mod config;
pub mod context;
pub mod core;
pub mod error;
pub mod matrix;
pub mod preconditioner;
pub mod solver;
pub mod utils;

// Re-exports for convenience
pub use config::*;
pub use context::*;
pub use core::*;
pub use error::*;
pub use matrix::*;
pub use preconditioner::*;
pub use solver::*;
pub use utils::*;

// Re-export SolveStats at the crate root for convenience
pub use utils::convergence::SolveStats;
