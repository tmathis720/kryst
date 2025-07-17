//! Matrix-free ("shell") operators for kryst.
//!
//! This module provides `ShellMat`, which allows users to define matrix operations
//! via callbacks rather than storing matrix entries explicitly. This is useful for:
//! - Large matrices that are expensive to store
//! - Matrices defined by algorithms (e.g., finite difference operators)
//! - Hierarchical or adaptive methods
//! - GPU-based or distributed matrix operations
//!
//! # Usage
//!
//! ```rust
//! use kryst::core::mat::shell::ShellMat;
//! 
//! // Create a 3x3 diagonal matrix with entries [2.0, 3.0, 4.0]
//! let shell = ShellMat::new(
//!     3,
//!     |x, y| {
//!         let x_ref = x.as_ref();
//!         let y_mut = y.as_mut();
//!         y_mut[0] = 2.0 * x_ref[0];
//!         y_mut[1] = 3.0 * x_ref[1]; 
//!         y_mut[2] = 4.0 * x_ref[2];
//!     },
//!     |x, y| {
//!         // For a diagonal matrix, transpose is the same
//!         let x_ref = x.as_ref();
//!         let y_mut = y.as_mut();
//!         y_mut[0] = 2.0 * x_ref[0];
//!         y_mut[1] = 3.0 * x_ref[1];
//!         y_mut[2] = 4.0 * x_ref[2];
//!     },
//! );
//! ```

use crate::core::traits::{MatVec, MatTransVec, MatShape};

/// A "shell" matrix: user-supplied callbacks for A·x and Aᵀ·x
///
/// `ShellMat` provides a matrix-free interface where matrix operations are defined
/// by user-provided closures. This allows for efficient representation of matrices
/// that don't need to be stored explicitly.
pub struct ShellMat<V> {
    /// Dimension of the (square) operator
    pub dim: usize,
    /// Matrix-vector multiplication: y = A·x
    mult: Box<dyn Fn(&V, &mut V) + Send + Sync>,
    /// Matrix-transpose-vector multiplication: y = Aᵀ·x  
    mult_trans: Box<dyn Fn(&V, &mut V) + Send + Sync>,
}

impl<V> ShellMat<V> {
    /// Construct a new shell matrix of size `dim` with user-provided operations.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the square matrix
    /// * `mult` - Closure computing y = A·x
    /// * `mult_trans` - Closure computing y = Aᵀ·x
    ///
    /// # Example
    ///
    /// ```rust
    /// use kryst::core::mat::shell::ShellMat;
    /// 
    /// // Identity matrix
    /// let identity = ShellMat::new(
    ///     3,
    ///     |x, y| {
    ///         let x_ref = x.as_ref();
    ///         let y_mut = y.as_mut();
    ///         for i in 0..x_ref.len() {
    ///             y_mut[i] = x_ref[i];
    ///         }
    ///     },
    ///     |x, y| {
    ///         let x_ref = x.as_ref();
    ///         let y_mut = y.as_mut(); 
    ///         for i in 0..x_ref.len() {
    ///             y_mut[i] = x_ref[i];
    ///         }
    ///     },
    /// );
    /// ```
    pub fn new<F, G>(dim: usize, mult: F, mult_trans: G) -> Self
    where
        F: Fn(&V, &mut V) + Send + Sync + 'static,
        G: Fn(&V, &mut V) + Send + Sync + 'static,
    {
        ShellMat {
            dim,
            mult: Box::new(mult),
            mult_trans: Box::new(mult_trans),
        }
    }

    /// Create a shell matrix where the transpose operation is the same as the forward operation.
    /// This is useful for symmetric matrices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the square matrix
    /// * `mult` - Closure computing y = A·x (used for both A·x and Aᵀ·x)
    ///
    /// # Example
    ///
    /// ```rust
    /// use kryst::core::mat::shell::ShellMat;
    /// 
    /// // Symmetric diagonal matrix
    /// let symmetric = ShellMat::new_symmetric(
    ///     3,
    ///     |x, y| {
    ///         let x_ref = x.as_ref();
    ///         let y_mut = y.as_mut();
    ///         y_mut[0] = 5.0 * x_ref[0];
    ///         y_mut[1] = 3.0 * x_ref[1];
    ///         y_mut[2] = 7.0 * x_ref[2];
    ///     },
    /// );
    /// ```
    pub fn new_symmetric<F>(dim: usize, mult: F) -> Self
    where
        F: Fn(&V, &mut V) + Send + Sync + 'static + Clone,
    {
        let mult_clone = mult.clone();
        ShellMat {
            dim,
            mult: Box::new(mult),
            mult_trans: Box::new(mult_clone),
        }
    }

    /// Get the dimension of this shell matrix.
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

impl<V> MatVec<V> for ShellMat<V>
where
    V: AsRef<[f64]> + AsMut<[f64]>,
{
    /// Apply the matrix-vector product: y = A·x
    fn matvec(&self, x: &V, y: &mut V) {
        (self.mult)(x, y);
    }
}

impl<V> MatTransVec<V> for ShellMat<V>
where
    V: AsRef<[f64]> + AsMut<[f64]>,
{
    /// Apply the matrix-transpose-vector product: y = Aᵀ·x
    fn mattransvec(&self, x: &V, y: &mut V) {
        (self.mult_trans)(x, y);
    }
}

impl<V> MatShape for ShellMat<V> {
    /// Number of rows in the matrix
    fn nrows(&self) -> usize {
        self.dim
    }

    /// Number of columns in the matrix
    fn ncols(&self) -> usize {
        self.dim
    }
}

// Implementing Send and Sync for ShellMat
unsafe impl<V> Send for ShellMat<V> {}
unsafe impl<V> Sync for ShellMat<V> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::{MatVec, MatTransVec};

    #[test]
    fn test_shell_identity() {
        let identity = ShellMat::new(
            3,
            |x: &Vec<f64>, y: &mut Vec<f64>| {
                let x_ref: &[f64] = x.as_ref();
                let y_mut: &mut [f64] = y.as_mut();
                for i in 0..x_ref.len() {
                    y_mut[i] = x_ref[i];
                }
            },
            |x: &Vec<f64>, y: &mut Vec<f64>| {
                let x_ref: &[f64] = x.as_ref();
                let y_mut: &mut [f64] = y.as_mut();
                for i in 0..x_ref.len() {
                    y_mut[i] = x_ref[i];
                }
            },
        );

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        
        identity.matvec(&x, &mut y);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);

        let mut y_trans = vec![0.0; 3];
        identity.mattransvec(&x, &mut y_trans);
        assert_eq!(y_trans, vec![1.0, 2.0, 3.0]);

        assert_eq!(identity.nrows(), 3);
        assert_eq!(identity.ncols(), 3);
    }

    #[test]
    fn test_shell_diagonal() {
        let diag_entries = vec![2.0, 3.0, 4.0];
        let diag_clone = diag_entries.clone();
        
        let diagonal = ShellMat::new(
            3,
            move |x: &Vec<f64>, y: &mut Vec<f64>| {
                let x_ref: &[f64] = x.as_ref();
                let y_mut: &mut [f64] = y.as_mut();
                for i in 0..x_ref.len() {
                    y_mut[i] = diag_entries[i] * x_ref[i];
                }
            },
            move |x: &Vec<f64>, y: &mut Vec<f64>| {
                let x_ref: &[f64] = x.as_ref();
                let y_mut: &mut [f64] = y.as_mut();
                for i in 0..x_ref.len() {
                    y_mut[i] = diag_clone[i] * x_ref[i];
                }
            },
        );

        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];
        
        diagonal.matvec(&x, &mut y);
        assert_eq!(y, vec![2.0, 3.0, 4.0]);

        let mut y_trans = vec![0.0; 3];
        diagonal.mattransvec(&x, &mut y_trans);
        assert_eq!(y_trans, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_shell_symmetric() {
        let symmetric = ShellMat::new_symmetric(
            2,
            |x: &Vec<f64>, y: &mut Vec<f64>| {
                let x_ref: &[f64] = x.as_ref();
                let y_mut: &mut [f64] = y.as_mut();
                // Matrix [[2, 1], [1, 3]]
                y_mut[0] = 2.0 * x_ref[0] + 1.0 * x_ref[1];
                y_mut[1] = 1.0 * x_ref[0] + 3.0 * x_ref[1];
            },
        );

        let x = vec![1.0, 0.0];
        let mut y = vec![0.0; 2];
        
        symmetric.matvec(&x, &mut y);
        assert_eq!(y, vec![2.0, 1.0]);

        let mut y_trans = vec![0.0; 2];
        symmetric.mattransvec(&x, &mut y_trans);
        assert_eq!(y_trans, vec![2.0, 1.0]); // Same as forward for symmetric matrix

        let x2 = vec![0.0, 1.0];
        let mut y2 = vec![0.0; 2];
        symmetric.matvec(&x2, &mut y2);
        assert_eq!(y2, vec![1.0, 3.0]);
    }

    #[test]
    fn test_shell_transpose() {
        // Non-symmetric matrix: [[1, 2], [3, 4]]
        // Transpose: [[1, 3], [2, 4]]
        let matrix = ShellMat::new(
            2,
            |x: &Vec<f64>, y: &mut Vec<f64>| {
                let x_ref: &[f64] = x.as_ref();
                let y_mut: &mut [f64] = y.as_mut();
                y_mut[0] = 1.0 * x_ref[0] + 2.0 * x_ref[1];
                y_mut[1] = 3.0 * x_ref[0] + 4.0 * x_ref[1];
            },
            |x: &Vec<f64>, y: &mut Vec<f64>| {
                let x_ref: &[f64] = x.as_ref();
                let y_mut: &mut [f64] = y.as_mut();
                y_mut[0] = 1.0 * x_ref[0] + 3.0 * x_ref[1];
                y_mut[1] = 2.0 * x_ref[0] + 4.0 * x_ref[1];
            },
        );

        let x = vec![1.0, 0.0];
        let mut y = vec![0.0; 2];
        
        matrix.matvec(&x, &mut y);
        assert_eq!(y, vec![1.0, 3.0]); // First column of A

        let mut y_trans = vec![0.0; 2];
        matrix.mattransvec(&x, &mut y_trans);
        assert_eq!(y_trans, vec![1.0, 2.0]); // First column of Aᵀ

        let x2 = vec![0.0, 1.0];
        let mut y2 = vec![0.0; 2];
        matrix.matvec(&x2, &mut y2);
        assert_eq!(y2, vec![2.0, 4.0]); // Second column of A

        let mut y2_trans = vec![0.0; 2];
        matrix.mattransvec(&x2, &mut y2_trans);
        assert_eq!(y2_trans, vec![3.0, 4.0]); // Second column of Aᵀ
    }
}
