//! Preconditioners for linear solvers.
//!
//! This module defines the Preconditioner trait and includes implementations such as Jacobi, ILU, SOR, AMG, Additive Schwarz, and more.

use crate::error::KError;

/// A preconditioner M ≈ A⁻¹.
pub trait Preconditioner<M, V> {
    /// Apply M⁻¹ to r, writing z = M⁻¹ r
    fn apply(&self, r: &V, z: &mut V) -> Result<(), KError>;
    /// Optionally: setup/factorize from A
    fn setup(&mut self, _a: &M) -> Result<(), KError> { Ok(()) }
}

/// A preconditioner whose action M⁻¹ may change at every iteration.
pub trait FlexiblePreconditioner<M, V> {
    /// Given the current residual `r`, produce `z ≈ Mₖ⁻¹ r`.
    fn apply(&mut self, r: &V, z: &mut V) -> Result<(), crate::error::KError>;
}

// Submodules for various preconditioners
pub mod block_jacobi;
pub mod ilu;
pub mod jacobi;
pub mod sor;
pub mod amg;
pub mod asm;
pub mod ilut;
pub mod ilup;
pub mod chebyshev;
pub mod approxinv;

// Re-exports for convenience
pub use jacobi::Jacobi;
pub use sor::Sor;
pub use ilu::Ilu0;
pub use amg::AMG;
pub use asm::AdditiveSchwarz;
pub use ilut::Ilut;
pub use ilup::Ilup;
pub use chebyshev::Chebyshev;
pub use approxinv::ApproxInv;
pub use self::sor::MatSorType;

/// Unified preconditioner enum for all supported types.
pub use crate::context::pc_context::{PC, SparsityPattern};
