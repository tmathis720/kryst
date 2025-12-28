#![deny(rustdoc::broken_intra_doc_links)]
#![deny(unused_must_use)]
#![warn(clippy::float_cmp)]
#![deny(clippy::manual_memcpy)]
#![cfg_attr(feature = "complex", allow(clippy::approx_constant))]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # kryst
//! PETSc-like Krylov solvers (KSP) and preconditioners (PC) in Rust.
//!
//! ## Quick start (serial)
//! ```rust,no_run
//! use kryst::prelude::*;
//! use kryst::matrix::MatShell;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let n = 4;
//! let a = MatShell::new(n, n, move |x, y| {
//!     for i in 0..n {
//!         let mut sum = 2.0 * x[i];
//!         if i > 0 {
//!             sum -= x[i - 1];
//!         }
//!         if i + 1 < n {
//!             sum -= x[i + 1];
//!         }
//!         y[i] = sum;
//!     }
//! });
//! let a = Arc::new(a) as Arc<dyn LinOp<S = R>>;
//! let b = vec![S::from_real(1.0); n];
//! let mut x = vec![S::zero(); n];
//!
//! let mut ksp = KspContext::new();
//! ksp.set_type(SolverType::Gmres)?;
//! ksp.set_pc_type(PcType::None, None)?;
//! ksp.set_operators(a, None);
//! ksp.setup()?;
//! let stats = ksp.solve(&b, &mut x)?;
//! println!("reason={:?} residual={:.3e}", stats.reason, stats.final_residual);
//! # Ok(()) }
//! ```
//!
//! ## Lifecycle
//! 1. `set_type` / `set_pc_type` / `set_from_options`
//! 2. `set_operators(Amat, Pmat)`
//! 3. `setup()` (idempotent; reuses structure/values when possible)
//! 4. `solve(b, x)` (may call `setup()` automatically if needed)
//!
//! ## Reuse model
//! - `LinOp::structure_id` changes when the sparsity pattern or dimensions change.
//! - `LinOp::values_id` changes when only numeric values change.
//! - `StructureId(0)` / `ValuesId(0)` mean "unknown" and force conservative rebuilds.
//! - Wrappers like `CsrOp` / `DenseOp` expose `mark_structure_changed` and
//!   `mark_values_changed` helpers.
//!
//! ## MPI communicator rules
//! - `KspContext` is bound to the operator communicator when operators are set.
//! - `try_set_operators` checks communicator congruence and dimension congruence.
//! - `try_set_operators_with_comm` wraps operators with an explicit communicator.
//! - Communicators must be IDENT or CONGRUENT (MPI_Comm_compare semantics).
//!
//! ## Feature flags
//! | Feature | Enables | Notes |
//! | --- | --- | --- |
//! | `mpi` | MPI communication backend | Requires MPI installed; MPI examples run via `mpirun` |
//! | `complex` | Complex scalar `S` (Complex64) | Classical and pipelined GMRES/FGMRES variants are supported |
//! | `backend-faer` | Dense/CSR backends and most PCs | Default feature |
//!
//! `S` is the internal scalar alias and `R` is its real partner. In real builds
//! `S = R = f64`. In complex builds `S = Complex64` and `R = f64`.
//!
//! ## Next steps
//! - `docs/communicators.md` for communicator details
//! - `docs/petsc_mapping.md` for PETSc to kryst mapping
//! - `examples/` for runnable demos

pub mod algebra;
pub mod config;
pub mod context;
pub mod core;
pub mod error;
pub mod matrix;
pub mod parallel;
pub mod preconditioner;
pub mod solver;
pub mod utils;

#[doc(hidden)]
pub mod ops;
#[doc(hidden)]
pub mod reduction;
#[doc(hidden)]
pub mod testkit;

pub mod prelude;

pub use crate::algebra::{R, S};
pub use crate::config::options::{KspOptions, PcOptions};
pub use crate::context::KspContext;
pub use crate::context::ksp_context::SolverType;
pub use crate::context::pc_context::PcType;
pub use crate::error::KError;
pub use crate::matrix::LinOp;
pub use crate::parallel::{Comm, UniverseComm};
pub use crate::preconditioner::{PcSide, Preconditioner};
pub use crate::utils::convergence::{ConvergedReason, SolveStats};
