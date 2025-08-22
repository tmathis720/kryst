//! Preconditioners for linear solvers.
//!
//! This module defines the Preconditioner trait and includes implementations such as Jacobi, ILU, SOR, AMG, Additive Schwarz, and more.

use crate::error::KError;
use faer::Mat;
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

/// Object-safe preconditioner operating on `f64` slices and [`LinOp`] matrices.
pub trait Preconditioner: Send + Sync {
    /// Build any factorization/hierarchy once from the system matrix.
    fn setup(
        &mut self,
        a: &dyn crate::matrix::op::LinOp<S = f64>,
    ) -> Result<(), KError>;

    /// Apply M⁻¹ to input vector, writing result to output slice.
    fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError>;
}

/// Internal trait for preconditioners that operate directly on dense matrices.
pub trait PreconditionerMat: Send + Sync {
    /// Set up the preconditioner from a dense matrix.
    fn setup_mat(&mut self, a: &Mat<f64>) -> Result<(), KError>;

    /// Apply M⁻¹ to input slice.
    fn apply_vec(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError>;
}

/// Legacy generic preconditioner traits retained for transitional adapters.
pub mod legacy {
    use super::PcSide;
    use crate::error::KError;

    /// Generic preconditioner operating on matrix and vector types.
    pub trait Preconditioner<M: ?Sized, V> {
        fn setup(&mut self, a: &M) -> Result<(), KError>;
        fn apply(&self, side: PcSide, r: &V, z: &mut V) -> Result<(), KError>;
    }

    /// Flexible preconditioner for FGMRES-style solvers.
    pub trait FlexiblePreconditioner<M: ?Sized, V> {
        fn setup(&mut self, a: &M) -> Result<(), KError>;
        fn apply(&mut self, r: &V, z: &mut V) -> Result<(), KError>;
    }
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
