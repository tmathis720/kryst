use std::{any::Any, hash::{Hash, Hasher}};
use faer::traits::ComplexField;

/// Format-agnostic linear operator.
pub trait LinOp: Send + Sync {
    /// Scalar type.
    type S: ComplexField;

    /// Dimensions (rows, cols).
    fn dims(&self) -> (usize, usize);

    /// Compute y = A x.
    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]);

    /// Optional transpose/adjoint matvec. Default panics.
    fn matvec_t(&self, _x: &[Self::S], _y: &mut [Self::S]) {
        unimplemented!("transpose/adjoint not provided for this operator");
    }

    /// Lightweight identifiers for structure/value changes.
    fn structure_id(&self) -> u64 { 0 }
    fn values_id(&self) -> u64 { 0 }

    /// Downcast hook for specialized solvers/preconditioners.
    fn as_any(&self) -> &dyn Any;
}

// --- Dense adapter -------------------------------------------------------
use faer::Mat;
impl LinOp for Mat<f64> {
    type S = f64;

    fn dims(&self) -> (usize, usize) { (self.nrows(), self.ncols()) }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        for i in 0..self.nrows() {
            let mut sum = 0.0;
            for j in 0..self.ncols() {
                sum += self[(i, j)] * x[j];
            }
            y[i] = sum;
        }
    }

    fn structure_id(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.nrows().hash(&mut h);
        self.ncols().hash(&mut h);
        h.finish()
    }

    fn as_any(&self) -> &dyn Any { self }
}

// --- CSR adapter ---------------------------------------------------------
use crate::matrix::sparse::{CsrMatrix, SparseMatrix};
impl LinOp for CsrMatrix<f64> {
    type S = f64;

    fn dims(&self) -> (usize, usize) { (self.nrows(), self.ncols()) }

    fn matvec(&self, x: &[f64], y: &mut [f64]) { self.spmv(x, y); }

    fn structure_id(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.row_ptr().hash(&mut h);
        self.col_idx().hash(&mut h);
        h.finish()
    }

    fn as_any(&self) -> &dyn Any { self }
}
