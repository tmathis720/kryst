//! Preconditioner context and configuration for Krylov solvers.
//!
//! This module defines the `PC` enum, which provides a unified interface for specifying
//! and configuring all supported preconditioner types in the library. Preconditioners
//! are used to accelerate the convergence of iterative solvers by transforming the
//! linear system into a more favorable form. Each variant of the `PC` enum corresponds
//! to a specific preconditioning strategy, with associated parameters where applicable.
//!
//! # Supported Preconditioners
//!
//! - Jacobi: Diagonal scaling preconditioner.
//! - Ssor: Symmetric Successive Over-Relaxation.
//! - Ilu0: Incomplete LU factorization with zero fill-in.
//! - Ilup: Incomplete LU with fixed fill-in level.
//! - Ilut: Incomplete LU with threshold-based dropping.
//! - Chebyshev: Polynomial preconditioner using Chebyshev polynomials.
//! - ApproxInv: Approximate inverse preconditioner with configurable sparsity.
//! - BlockJacobi: Block-diagonal Jacobi preconditioner.
//! - Multicolor: Multicoloring-based preconditioner.
//! - AMG: Algebraic Multigrid preconditioner.
//! - AdditiveSchwarz: Additive Schwarz domain decomposition preconditioner.
//!
//! # Example
//!
//! ```rust
//! use crate::context::pc_context::{PC, SparsityPattern};
//! let pc = PC::Ilut { fill: 10, droptol: 1e-3 };
//! ```

/// Unified preconditioner enum for all supported types.
///
/// Each variant represents a different preconditioning strategy. Some variants
/// include parameters to control their behavior (e.g., fill level, drop tolerance,
/// polynomial degree, etc.).
#[derive(Debug, Clone)]
pub enum PC<T> {
    /// Jacobi (diagonal scaling) preconditioner.
    Jacobi,
    /// Symmetric Successive Over-Relaxation (SSOR) preconditioner.
    Ssor,
    /// Incomplete LU factorization with zero fill-in (ILU(0)).
    Ilu0,
    /// Incomplete LU factorization with fixed fill-in level (ILU(p)).
    ///
    /// - `fill`: The fill-in level (number of allowed extra nonzeros per row).
    Ilup { fill: usize },
    /// Incomplete LU factorization with threshold-based dropping (ILUT).
    ///
    /// - `fill`: Maximum number of nonzeros per row.
    /// - `droptol`: Drop tolerance for discarding small elements.
    Ilut { fill: usize, droptol: T },
    /// Chebyshev polynomial preconditioner.
    ///
    /// - `degree`: Degree of the Chebyshev polynomial.
    /// - `emin`: Optional lower bound on the spectrum.
    /// - `emax`: Optional upper bound on the spectrum.
    Chebyshev { degree: usize, emin: Option<T>, emax: Option<T> },
    /// Approximate inverse preconditioner.
    ///
    /// - `pattern`: Sparsity pattern for the approximate inverse.
    /// - `tol`: Convergence tolerance for the iterative construction.
    /// - `max_iter`: Maximum number of iterations for the construction algorithm.
    ApproxInv { pattern: SparsityPattern, tol: T, max_iter: usize },
    /// Block Jacobi preconditioner.
    ///
    /// - `blocks`: List of index blocks, each block is a list of row/column indices.
    BlockJacobi { blocks: Vec<Vec<usize>> },
    /// Multicolor preconditioner.
    ///
    /// - `colors`: Color assignment for each row/column (for parallelization).
    Multicolor { colors: Vec<usize> },
    /// Algebraic Multigrid (AMG) preconditioner.
    AMG,
    /// Additive Schwarz domain decomposition preconditioner.
    AdditiveSchwarz,
}

/// Sparsity pattern for approximate inverse preconditioners.
///
/// Used to control the nonzero structure of the approximate inverse. The `Auto` variant
/// lets the library choose a pattern automatically, while `Manual` allows the user to
/// specify the sparsity structure explicitly.
#[derive(Debug, Clone)]
pub enum SparsityPattern {
    /// Let the library choose the sparsity pattern automatically.
    Auto,
    /// User-specified sparsity pattern.
    ///
    /// Each inner vector contains the column indices for the corresponding row.
    Manual(Vec<Vec<usize>>), // for each row, the list of column indices
}
