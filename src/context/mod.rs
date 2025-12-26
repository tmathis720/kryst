//! KSP/PC context types and factories.
//!
//! Most users configure solvers through [`KspContext`], which owns the solver type,
//! preconditioner choice, operator bindings, and the `setup -> solve` lifecycle.
//! Preconditioner construction helpers live in [`pc_context`].
//!
//! ## Lifecycle (summary)
//! 1. `set_type` / `set_pc_type` / `set_from_options`
//! 2. `set_operators(Amat, Pmat)`
//! 3. `setup()` (idempotent; reuses cached structure/values when possible)
//! 4. `solve(b, x)`
//!
//! ## Example
//! ```rust,no_run
//! use kryst::prelude::*;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! struct IdentityOp(usize);
//! impl LinOp for IdentityOp {
//!     type S = R;
//!     fn dims(&self) -> (usize, usize) {
//!         (self.0, self.0)
//!     }
//!     fn matvec(&self, x: &[R], y: &mut [R]) {
//!         y.copy_from_slice(x);
//!     }
//!     fn as_any(&self) -> &dyn std::any::Any {
//!         self
//!     }
//! }
//!
//! let op = Arc::new(IdentityOp(4)) as Arc<dyn LinOp<S = R>>;
//! let b = vec![S::from_real(1.0); 4];
//! let mut x = vec![S::zero(); 4];
//! let mut ksp = KspContext::new();
//! ksp.set_type(SolverType::Gmres)?;
//! ksp.set_pc_type(PcType::None, None)?;
//! ksp.set_operators(op, None);
//! let _stats = ksp.solve(&b, &mut x)?;
//! # Ok(()) }
//! ```

pub mod ksp_context;
pub use ksp_context::KspContext;
pub mod pc_context;
pub use pc_context::{DeferredPcInfo, NoOpPreconditioner, PC, PcFactory, PcType, SparsityPattern};
