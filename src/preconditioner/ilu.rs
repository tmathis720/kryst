//! ILU(0) factorization with zero fill (Saad §10.3).
//!
//! This module implements the Incomplete LU factorization with zero fill (ILU(0)),
//! a classic preconditioner for sparse linear systems. The ILU(0) factorization
//! produces lower (L) and upper (U) triangular matrices with the same sparsity
//! pattern as the original matrix, but does not introduce any new nonzero entries (zero fill).
//!
//! # Overview
//!
//! Given a matrix A, ILU(0) computes L and U such that A ≈ LU, where L is unit lower triangular
//! and U is upper triangular. The factorization is performed in-place, and only nonzero entries
//! in the original matrix are considered for fill-in.
//!
//! # Usage
//!
//! - Create an `Ilu0` struct (optionally with `new` or `default`).
//! - Call `setup` with the system matrix to compute the factors.
//! - Use `apply` to solve M⁻¹x ≈ A⁻¹x using the computed factors.
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. Section 10.3.

use crate::preconditioner::Preconditioner;
use crate::error::KError;
use num_traits::Float;
use faer::traits::ComplexField;
use faer::Mat;

/// ILU(0) preconditioner struct
///
/// Stores the lower (L) and upper (U) triangular factors as dense matrices.
pub struct Ilu0<T> {
    /// Lower triangular factor (unit diagonal)
    pub(crate) l: Mat<T>,
    /// Upper triangular factor
    pub(crate) u: Mat<T>,
}

impl<T: Float + Send + Sync + ComplexField> Ilu0<T> {
    /// Create a new, empty ILU(0) preconditioner
    pub fn new() -> Self {
        Self {
            l: Mat::zeros(0, 0),
            u: Mat::zeros(0, 0),
        }
    }
}

impl<T: num_traits::Float + Send + Sync + faer::traits::ComplexField> Default for Ilu0<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Sync + ComplexField> Preconditioner<Mat<T>, Vec<T>> for Ilu0<T> {
    /// Compute the ILU(0) factorization of the matrix `a`.
    ///
    /// The factors L and U are stored in the struct. Only nonzero entries in `a` are considered for fill-in.
    fn setup(&mut self, a: &Mat<T>) -> Result<(), KError> {
        let (n, _) = (a.nrows(), a.ncols());
        let mut l = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);

        for i in 0..n {
            // Set diagonal of U
            u[(i, i)] = a[(i, i)];
            // Fill upper triangle of U (row i)
            for j in (i+1)..n {
                if a[(i, j)] != T::zero() {
                    u[(i, j)] = a[(i, j)];
                }
            }
            // Set diagonal of L
            l[(i, i)] = T::one();
            // Fill lower triangle of L (column i)
            for j in (i+1)..n {
                if a[(j, i)] != T::zero() {
                    l[(j, i)] = a[(j, i)] / u[(i, i)];
                }
            }
            // Update remaining entries (Schur complement update)
            for j in (i+1)..n {
                for k in (i+1)..n {
                    if a[(j, k)] != T::zero() {
                        let v = a[(j, k)] - l[(j, i)] * u[(i, k)];
                        if v != T::zero() {
                            if k >= j {
                                u[(j, k)] = v;
                            } else {
                                l[(j, k)] = v;
                            }
                        }
                    }
                }
            }
        }
        self.l = l;
        self.u = u;
        Ok(())
    }

    /// Apply the ILU(0) preconditioner to a vector x, storing the result in y.
    ///
    /// Solves Ly1 = x (forward substitution), then Uy = y1 (backward substitution).
    fn apply(&self, x: &Vec<T>, y: &mut Vec<T>) -> Result<(), KError> {
        // Forward substitution: solve L y1 = x
        let mut y1 = x.clone();
        let n = x.len();
        for i in 0..n {
            for j in 0..i {
                y1[i] = y1[i] - self.l[(i, j)] * y1[j];
            }
        }
        // Backward substitution: solve U y = y1
        for i in (0..n).rev() {
            for j in (i+1)..n {
                y1[i] = y1[i] - self.u[(i, j)] * y1[j];
            }
        }
        y.copy_from_slice(&y1);
        Ok(())
    }
}
