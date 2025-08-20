//! Preconditioners for linear solvers.
//!
//! This module defines the Preconditioner trait and includes implementations such as Jacobi, ILU, SOR, AMG, Additive Schwarz, and more.

use crate::error::KError;
use std::str::FromStr;

/// Which side to apply M⁻¹ on in preconditioning.
///
/// For the linear system Ax = b with preconditioner M ≈ A:
/// - Left: Solve M⁻¹Ax = M⁻¹b
/// - Right: Solve AM⁻¹y = b, then x = M⁻¹y  
/// - Symmetric: Apply both left and right preconditioning (M₁⁻¹AM₂⁻¹)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcSide {
    /// Left preconditioning: M⁻¹Ax = M⁻¹b
    Left,
    /// Right preconditioning: AM⁻¹y = b, x = M⁻¹y
    Right,
    /// Symmetric preconditioning: M₁⁻¹AM₂⁻¹y = M₁⁻¹b, x = M₂⁻¹y
    Symmetric,
}

impl FromStr for PcSide {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "left" => Ok(PcSide::Left),
            "right" => Ok(PcSide::Right),
            "symmetric" => Ok(PcSide::Symmetric),
            _ => Err(KError::UnrecognizedPcSide(s.to_string())),
        }
    }
}

impl Default for PcSide {
    fn default() -> Self {
        PcSide::Left
    }
}

/// A preconditioner M ≈ A⁻¹.
pub trait Preconditioner<M: ?Sized, V> {
    /// Build any factorization/hierarchy once from the system matrix.
    ///
    /// # Arguments
    /// * `a` - System matrix to build preconditioner from
    ///
    /// # Returns
    /// * `Ok(())` on successful setup
    /// * `Err(KError)` if setup fails
    fn setup(&mut self, a: &M) -> Result<(), KError>;
    
    /// Apply M⁻¹ to input vector, writing result to output vector.
    ///
    /// # Arguments
    /// * `side` - Which side to apply preconditioning (Left/Right/Symmetric)
    /// * `x` - Input vector
    /// * `y` - Output vector (will be overwritten)
    ///
    /// # Returns
    /// * `Ok(())` on successful application
    /// * `Err(KError)` if application fails
    fn apply(&self, side: PcSide, x: &V, y: &mut V) -> Result<(), KError>;
}

/// A preconditioner whose action M⁻¹ may change at every iteration.
pub trait FlexiblePreconditioner<M: ?Sized, V> {
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
pub mod ilutp;
pub mod ilup;
pub mod chebyshev;
pub mod approxinv;
pub mod chain;

// Re-exports for convenience
pub use jacobi::Jacobi;
pub use sor::Sor;
pub use ilu::Ilu0;
pub use amg::AMG;
pub use asm::AdditiveSchwarz;
pub use ilut::Ilut;
pub use ilutp::Ilutp;
pub use ilup::Ilup;
pub use chebyshev::{Chebyshev, ChebyshevPre};
pub use approxinv::ApproxInv;
pub use chain::PcChain;
pub use self::sor::MatSorType;

/// Unified preconditioner enum for all supported types.
pub use crate::context::pc_context::{PC, SparsityPattern};
