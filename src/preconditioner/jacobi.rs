//! Jacobi preconditioner implementation.
//!
//! This monomorphic version operates on `f64` values and dense [`faer::Mat`] matrices.

use crate::error::KError;
use crate::preconditioner::{PcSide, PreconditionerMat};
use faer::Mat;

/// Jacobi preconditioner: stores the inverse of the diagonal of the matrix.
pub struct Jacobi {
    /// Inverse diagonal entries.
    pub(crate) diag_inv: Vec<f64>,
}

impl Jacobi {
    /// Create an empty Jacobi preconditioner. Call [`setup_mat`] before use.
    pub fn new() -> Self {
        Self { diag_inv: Vec::new() }
    }

    fn build_from_dense(&mut self, a: &Mat<f64>) -> Result<(), KError> {
        let n = a.nrows();
        self.diag_inv.resize(n, 0.0);
        for i in 0..n {
            let val = a[(i, i)];
            self.diag_inv[i] = if val != 0.0 { 1.0 / val } else { 0.0 };
        }
        Ok(())
    }
}

impl PreconditionerMat for Jacobi {
    fn setup_mat(&mut self, a: &Mat<f64>) -> Result<(), KError> {
        self.build_from_dense(a)
    }

    fn apply_vec(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        assert_eq!(r.len(), z.len());
        for ((zi, di), ri) in z.iter_mut().zip(self.diag_inv.iter()).zip(r.iter()) {
            *zi = di * ri;
        }
        Ok(())
    }
}

