//! Context module for KrylovKit linear algebra library.
//!
//! This module provides context/factory types for configuring and managing solver and preconditioner objects.
//! Contexts encapsulate algorithm selection, parameter management, and construction of solver/preconditioner pipelines.
//!
//! Modules:
//! - [`ksp_context`]: Contains the `KspContext` struct for Krylov subspace solver configuration and management.
//! - [`pc_context`]: Contains the preconditioner context types and factories.
//!
//! Usage:
//! Import the desired context type and use it to configure and instantiate solvers or preconditioners.
//!
//! # Example
//! ```rust,ignore
//! use crate::context::KspContext;
//! let ksp = KspContext::new();
//! // Configure and use the context...
//! ```
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.
//! - PETSc documentation: https://petsc.org/release/docs/manualpages/KSP/

pub mod ksp_context;
pub use ksp_context::KspContext;
pub mod pc_context;
