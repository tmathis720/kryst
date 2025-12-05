use crate::error::KError;

/// Abstract interface for triangular solves used by preconditioners.
///
/// Implementors typically wrap sparse L/U factors and provide in-place solves.
pub trait TriangularSolve<S> {
    /// Solve `L * x = b` (forward substitution), writing the solution into `x`.
    fn solve_lower_in_place(&self, x: &mut [S]) -> Result<(), KError>;

    /// Solve `U * x = b` (backward substitution), writing the solution into `x`.
    fn solve_upper_in_place(&self, x: &mut [S]) -> Result<(), KError>;
}
