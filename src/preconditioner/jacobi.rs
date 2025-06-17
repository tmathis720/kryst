// Jacobi preconditioner implementation
//
// This module implements the Jacobi (diagonal) preconditioner, which uses the inverse of the diagonal
// of the system matrix as a simple preconditioner for iterative solvers. The Jacobi preconditioner is
// effective for diagonally dominant matrices and is often used as a baseline or smoother in multigrid methods.
//
// # Overview
//
// The Jacobi preconditioner approximates the inverse of the matrix A by the inverse of its diagonal D:
//     M⁻¹ ≈ D⁻¹
//
// # Usage
//
// - Create a `Jacobi` preconditioner with `new()` or `default()`.
// - Call `setup` with the system matrix to extract the diagonal and compute its inverse.
// - Use `apply` to apply the preconditioner to a vector.

use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, Indexing};
use crate::error::KError;
use num_traits::Float;

/// Jacobi preconditioner: M⁻¹ = D⁻¹
///
/// Stores the inverse of the diagonal of the system matrix.
pub struct Jacobi<T> {
    /// Inverse of the diagonal entries of the matrix
    pub(crate) inv_diag: Vec<T>,
}

impl<T: Float> Jacobi<T> {
    /// Create a new Jacobi preconditioner with empty state; user must call `setup`.
    pub fn new() -> Self {
        Self { inv_diag: Vec::new() }
    }
}

impl<T: num_traits::Float> Default for Jacobi<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, V, T> Preconditioner<M, V> for Jacobi<T>
where
    M: MatVec<V> + Indexing,
    V: AsRef<[T]> + AsMut<[T]> + From<Vec<T>>,
    T: Float + Send + Sync,
{
    /// Setup the Jacobi preconditioner by extracting and inverting the diagonal of the matrix.
    ///
    /// For each row i, computes the diagonal entry by applying the matrix to the i-th unit vector.
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        let n = a.nrows();
        let mut diag = vec![T::zero(); n];
        let mut e = vec![T::zero(); n];
        let mut col = vec![T::zero(); n];
        for i in 0..n {
            // Set e to the i-th unit vector
            e.iter_mut().for_each(|x| *x = T::zero());
            e[i] = T::one();
            let e_v = V::from(e.clone());
            let mut col_v = V::from(col.clone());
            a.matvec(&e_v, &mut col_v);
            col = col_v.as_ref().to_vec();
            diag[i] = col[i];
        }
        // Compute inverse diagonal, handling zeros safely
        self.inv_diag = diag.into_iter()
            .map(|d| if d != T::zero() { T::one() / d } else { T::zero() })
            .collect();
        Ok(())
    }

    /// Apply the Jacobi preconditioner to a vector x, storing the result in y.
    ///
    /// Computes y[i] = inv_diag[i] * x[i] for all i.
    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError> {
        let x_ref = x.as_ref();
        let y_mut = y.as_mut();
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            y_mut.par_iter_mut().enumerate().for_each(|(i, yval)| {
                *yval = self.inv_diag[i] * x_ref[i];
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for i in 0..x_ref.len() {
                y_mut[i] = self.inv_diag[i] * x_ref[i];
            }
        }
        Ok(())
    }
}
