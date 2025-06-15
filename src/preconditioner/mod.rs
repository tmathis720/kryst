//! Preconditioners for linear solvers.

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

pub mod block_jacobi;
pub mod ilu;
pub mod jacobi;
pub mod sor;
pub mod amg;
pub mod ilut;
pub mod ilup;
pub mod chebyshev;
pub mod approxinv;

pub use jacobi::Jacobi;
pub use sor::Sor;
pub use ilu::Ilu0;
pub use amg::AMG;
pub use ilut::Ilut;
pub use ilup::Ilup;
pub use chebyshev::Chebyshev;
pub use approxinv::ApproxInv;
pub use self::sor::MatSorType;

/// Unified preconditioner enum for all supported types.
pub enum PC<T> {
    Jacobi,
    Ssor,
    Ilu0,
    Ilup { fill: usize },
    Ilut { fill: usize, droptol: T },
    Chebyshev { degree: usize, emin: Option<T>, emax: Option<T> },
    ApproxInv { pattern: SparsityPattern, tol: T, max_iter: usize },
    BlockJacobi { blocks: Vec<Vec<usize>> },
    Multicolor { colors: Vec<usize> },
    AMG,
}

/// Sparsity pattern for approximate inverse preconditioners.
pub enum SparsityPattern {
    Auto,
    Manual(Vec<Vec<usize>>), // for each row, the list of column indices
}
