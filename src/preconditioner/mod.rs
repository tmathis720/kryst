//! Preconditioners for linear solvers.

use crate::error::KError;

/// A preconditioner M ≈ A⁻¹.
/// - `setup` is called once to factor or build M.
/// - `apply` applies M to a vector: y ← M x.
pub trait Preconditioner<M, V>: Send + Sync {
    /// Factor or initialize the preconditioner from A.
    fn setup(&mut self, a: &M) -> Result<(), KError>;

    /// Apply preconditioner: y ← M x.
    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError>;
}

pub mod block_jacobi;
pub mod ilu;
pub mod jacobi;
pub mod ssor;
pub use jacobi::Jacobi;
pub use ssor::Ssor;
pub use ilu::Ilu0;
