#![deny(rustdoc::broken_intra_doc_links)]
#![deny(unused_must_use)]
#![warn(clippy::float_cmp)]
#![deny(clippy::manual_memcpy)]
#![cfg_attr(feature = "complex", allow(clippy::approx_constant))]
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
//!
//! ### Parallelism
//! With the `rayon` feature, [`CsrOp::matvec`](crate::matrix::op::CsrOp) can use a
//! parallel SpMV path for local-only matrices larger than `KRYST_PAR_CUTOFF`
//! (default [`crate::parallel::threads::DEFAULT_PAR_CUTOFF`]). Control the pool
//! size with `KRYST_THREADS` or `RAYON_NUM_THREADS`. See
//! [`crate::parallel::threads`] for details.
//!
//! ### Scalars
//! Internal linear algebra now operates on a crate-wide scalar alias `S`.
//! The default build keeps `S = f64`, while enabling the `complex` feature
//! switches to `Complex64` without changing public solver interfaces.

pub mod parallel;

pub mod algebra;
pub mod config;
pub mod context;
pub mod core;
pub mod error;
pub mod matrix;
pub mod ops;
pub mod preconditioner;
pub mod reduction;
pub mod solver;
pub mod testkit;
pub mod utils;

pub mod prelude {
    pub use crate::algebra::{dense as algebra_dense, *};
    pub use crate::matrix::{CsrMatrix, DenseMatrix, dense as matrix_dense};
}

pub use crate::prelude::*;
pub use config::*;
pub use context::*;
pub use core::*;
pub use error::*;
pub use reduction::*;
pub use utils::*;

// Re-export SolveStats at the crate root for convenience
pub use utils::convergence::SolveStats;
