//! Preconditioners for linear solvers.

use crate::error::KError;

/// A preconditioner M ≈ A⁻¹.
pub trait Preconditioner<M, V> {
    /// Apply M⁻¹ to r, writing z = M⁻¹ r
    fn apply(&self, r: &V, z: &mut V) -> Result<(), KError>;
    /// Optionally: setup/factorize from A
    fn setup(&mut self, a: &M) -> Result<(), KError> { Ok(()) }
}

pub mod block_jacobi;
pub mod ilu;
pub mod jacobi;
pub mod ssor;
pub use jacobi::Jacobi;
pub use ssor::Ssor;
pub use ilu::Ilu0;
