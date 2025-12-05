// SparseMatrix trait and implementations (CSR, CSC)

use crate::algebra::prelude::*;
use crate::core::traits::{Indexing, SubmatrixExtract};

#[cfg(all(feature = "backend-faer", feature = "simd"))]
use crate::matrix::spmv::{SpmvPlan, SpmvTuning, build_plan_owned as build_spmv_plan};
use std::collections::HashMap;

/// A read‐only sparse matrix supporting CSR access.
pub trait SparseMatrix {
    /// Stored scalar type.
    type Scalar;

    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Borrow the CSR row pointer array (length = nrows + 1).
    fn row_ptr(&self) -> &[usize];
    /// Borrow the CSR column index array (length = nnz).
    fn col_idx(&self) -> &[usize];
    /// Borrow the CSR value array (length = nnz).
    fn values(&self) -> &[Self::Scalar];
}

/// CSR matrix with structural access available for any scalar type.
#[derive(Clone)]
pub struct CsrMatrix<T> {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
    /// Cached position of the diagonal entry in each row.
    ///
    /// `diag_pos[i]` stores `Some(k)` if column `i` appears in row `i` and
    /// `k` is the index into `values()`; otherwise `None` if the diagonal is
    /// structurally zero.  This enables O(1) access to the diagonal for
    /// factorization and triangular solve kernels without converting to a
    /// dense representation.
    diag_pos: Vec<Option<usize>>,
    #[cfg(all(feature = "backend-faer", feature = "simd"))]
    spmv_plan: Option<SpmvPlan<f64>>,
}

impl<T> CsrMatrix<T> {
    /// Build a CSR from raw row‐ptr, col‐idx, and values.
    pub fn from_csr(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        let mut this = Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
            diag_pos: Vec::new(),
            #[cfg(all(feature = "backend-faer", feature = "simd"))]
            spmv_plan: None,
        };
        this.build_diag_pos();
        this
    }

    /// Get matrix dimensions.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Get number of nonzeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Borrow the CSR row pointer array (length = nrows + 1).
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Borrow the CSR column index array (length = nnz).
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Borrow the CSR value array (length = nnz).
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Mutably borrow the CSR value array (length = nnz).
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        #[cfg(all(feature = "backend-faer", feature = "simd"))]
        self.invalidate_spmv_plan();
        &mut self.values
    }

    /// Borrow a row of the matrix as CSR slices `(col_idx, values)`.
    #[inline]
    pub fn row(&self, i: usize) -> (&[usize], &[T]) {
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        (&self.col_idx[start..end], &self.values[start..end])
    }

    /// Mutable borrow of the values of row `i`.
    ///
    /// The column indices remain immutable; the structure of the CSR matrix
    /// is fixed.  This is intended for in-place numeric operations such as
    /// ILU factorizations where the sparsity pattern does not change.
    #[inline]
    pub fn row_values_mut(&mut self, i: usize) -> &mut [T] {
        #[cfg(all(feature = "backend-faer", feature = "simd"))]
        self.invalidate_spmv_plan();
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        &mut self.values[start..end]
    }

    /// Immutable access to the diagonal entry of row `i` if present.
    #[inline]
    pub fn diag_ref(&self, i: usize) -> Option<&T> {
        self.diag_pos[i].map(|k| &self.values()[k])
    }

    /// Mutable access to the diagonal entry of row `i` if present.
    #[inline]
    pub fn diag_mut(&mut self, i: usize) -> Option<&mut T> {
        if let Some(k) = self.diag_pos[i] {
            Some(&mut self.values_mut()[k])
        } else {
            None
        }
    }

    /// Rebuild the cached diagonal positions.  Call after any operation that
    /// may have modified the sparsity structure.
    pub fn build_diag_pos(&mut self) {
        let n = self.nrows();
        self.diag_pos.resize(n, None);
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            if let Ok(off) = self.col_idx[start..end].binary_search(&i) {
                self.diag_pos[i] = Some(start + off);
            } else {
                self.diag_pos[i] = None;
            }
        }
    }
}

impl<T> SparseMatrix for CsrMatrix<T> {
    type Scalar = T;

    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn row_ptr(&self) -> &[usize] {
        self.row_ptr()
    }

    fn col_idx(&self) -> &[usize] {
        self.col_idx()
    }

    fn values(&self) -> &[Self::Scalar] {
        self.values()
    }
}

impl<T: KrystScalar> CsrMatrix<T> {
    /// Create an identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let values: Vec<T> = vec![T::one(); n];

        Self::from_csr(n, n, row_ptr, col_idx, values)
    }

    /// Extract diagonal as a vector.
    pub fn diagonal(&self) -> Vec<T> {
        let n = self.nrows().min(self.ncols());
        let mut diag = vec![T::zero(); n];

        for i in 0..n {
            let (cols, vals) = self.row(i);
            if let Some((_, &val)) = cols
                .iter()
                .copied()
                .zip(vals.iter())
                .find(|(col, _)| *col == i)
            {
                diag[i] = val;
            }
        }

        diag
    }

    /// Sparse matrix-vector product with default scaling: y = A * x.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        crate::matrix::spmv::csr_matvec(self, x, y).expect("spmv dimension mismatch");
    }

    /// Sparse matrix-vector product: y = alpha * A * x + beta * y.
    pub fn spmv_scaled(
        &self,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(crate::error::KError::InvalidInput(format!(
                "Dimension mismatch in spmv: A={}x{}, x.len()={}, y.len={}",
                self.nrows(),
                self.ncols(),
                x.len(),
                y.len()
            )));
        }

        crate::matrix::spmv::scalar::spmv_scaled_csr(
            self.nrows(),
            self.row_ptr(),
            self.col_idx(),
            self.values(),
            alpha,
            x,
            beta,
            y,
        );
        Ok(())
    }

    /// Sparse matrix-vector product with transpose: y = alpha * A^T * x + beta * y.
    pub fn spmv_transpose_scaled(
        &self,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.nrows() || y.len() != self.ncols() {
            return Err(crate::error::KError::InvalidInput(format!(
                "Dimension mismatch in spmv^T: A={}x{}, x.len()={}, y.len()={}",
                self.nrows(),
                self.ncols(),
                x.len(),
                y.len()
            )));
        }

        crate::matrix::spmv::scalar::spmv_t_scaled_csr(
            self.nrows(),
            self.row_ptr(),
            self.col_idx(),
            self.values(),
            alpha,
            x,
            beta,
            y,
        );
        Ok(())
    }
}

/// Methods that only work when `T::Real = f64` (for faer interop).
impl<T> CsrMatrix<T>
where
    T: KrystScalar<Real = f64>,
{
    /// Convert to dense faer::Mat with real (f64) entries. Works for any T: KrystScalar.
    ///
    /// In complex builds, this projects the matrix onto its real component;
    /// imaginary parts are discarded because AMG operates on real operators.
    #[cfg(feature = "backend-faer")]
    pub fn to_dense(&self) -> faer::Mat<f64> {
        let mut dense = faer::Mat::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            let (cols, vals) = self.row(i);
            for (&j, &v) in cols.iter().zip(vals.iter()) {
                dense[(i, j)] = v.real();
            }
        }
        dense
    }

    /// Convert from dense faer::Mat (with real entries) to sparse CSR format with drop tolerance.
    /// Works for any T: KrystScalar by converting each entry via T::from_real.
    ///
    /// When `T` is complex, the imaginary parts of the resulting matrix are
    /// zero-initialised because the construction lifts real values through
    /// [`KrystScalar::from_real`].
    #[cfg(feature = "backend-faer")]
    pub fn from_dense(dense: &faer::Mat<R>, drop_tol: R) -> Self {
        let nrows = dense.nrows();
        let ncols = dense.ncols();
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for i in 0..nrows {
            for j in 0..ncols {
                let val = dense[(i, j)];
                if val.abs() >= drop_tol {
                    col_idx.push(j);
                    values.push(T::from_real(val));
                }
            }
            row_ptr.push(col_idx.len());
        }

        Self::from_csr(nrows, ncols, row_ptr, col_idx, values)
    }

    /// Convert from an owned dense `faer::Mat<R>` to sparse CSR format with drop tolerance.
    #[cfg(feature = "backend-faer")]
    pub fn from_dense_owned(dense: faer::Mat<R>, drop_tol: R) -> Self {
        Self::from_dense(&dense, drop_tol)
    }
}

impl CsrMatrix<f64> {
    /// Convert this CSR matrix into the scalar-aware CSR wrapper.
    pub fn to_scalar_csr(&self) -> crate::matrix::csr::CsrMatrix<S> {
        let values = self.values().iter().copied().map(S::from_real).collect();
        crate::matrix::csr::CsrMatrix::new(
            self.nrows(),
            self.ncols(),
            self.row_ptr().to_vec(),
            self.col_idx().to_vec(),
            values,
        )
    }
}

impl<T> Indexing for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        self.nrows()
    }
}

impl<T: Clone> SubmatrixExtract for CsrMatrix<T> {
    type S = T;

    fn extract_submatrix(&self, rows: &[usize], cols: &[usize]) -> Self {
        let m = rows.len();
        let n = cols.len();
        let mut row_ptr = Vec::with_capacity(m + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        let mut g2l: HashMap<usize, usize> = HashMap::with_capacity(n);
        for (l, &g) in cols.iter().enumerate() {
            g2l.insert(g, l);
        }

        for &g_row in rows {
            let rs = self.row_ptr[g_row];
            let re = self.row_ptr[g_row + 1];
            for p in rs..re {
                let gcol = self.col_idx[p];
                if let Some(&lcol) = g2l.get(&gcol) {
                    col_idx.push(lcol);
                    values.push(self.values[p].clone());
                }
            }
            row_ptr.push(col_idx.len());
        }

        CsrMatrix::from_csr(m, n, row_ptr, col_idx, values)
    }
}

#[cfg(all(feature = "backend-faer", feature = "simd"))]
impl<T> CsrMatrix<T> {
    #[inline]
    fn invalidate_spmv_plan(&mut self) {
        self.spmv_plan = None;
    }
}

#[cfg(all(feature = "backend-faer", feature = "simd"))]
impl CsrMatrix<f64> {
    /// Builds (or rebuilds) the SIMD-aware SpMV plan using the provided tuning.
    pub fn build_spmv_plan(&mut self, tuning: &SpmvTuning) {
        let owned = crate::matrix::csr::CsrMatrix::new(
            self.nrows(),
            self.ncols(),
            self.row_ptr().to_vec(),
            self.col_idx().to_vec(),
            self.values().to_vec(),
        );
        self.spmv_plan = Some(build_spmv_plan(owned, tuning));
    }

    /// Clears any cached SpMV plan, forcing the scalar fallback on the next
    /// application until [`build_spmv_plan`] is invoked again.
    pub fn clear_spmv_plan(&mut self) {
        self.spmv_plan = None;
    }
}

#[cfg(feature = "rayon")]
impl<T> CsrMatrix<T>
where
    T: KrystScalar,
{
    /// Parallel SpMV using CSR structure directly.
    pub fn spmv_parallel(&self, x: &[T], y: &mut [T]) {
        crate::matrix::spmv::csr_matvec_par(self, x, y).expect("spmv_parallel dimension mismatch");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_spmv() {
        // 3×3 identity in CSR: row_ptr=[0,1,2,3], col_idx=[0,1,2], vals=[1,1,1]
        let m = CsrMatrix::from_csr(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![S::from_real(1.0), S::from_real(1.0), S::from_real(1.0)],
        );
        let x = vec![S::from_real(2.0), S::from_real(3.0), S::from_real(5.0)];
        let mut y = vec![S::zero(); 3];
        m.spmv_scaled(S::one(), &x, S::zero(), &mut y).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn simple_pattern() {
        // 2×3 matrix [[1,2,0],[0,3,4]]
        let m = CsrMatrix::from_csr(
            2,
            3,
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![
                S::from_real(1.0),
                S::from_real(2.0),
                S::from_real(3.0),
                S::from_real(4.0),
            ],
        );
        let x = vec![S::one(), S::one(), S::one()];
        let mut y = vec![S::zero(); 2];
        m.spmv_scaled(S::one(), &x, S::zero(), &mut y).unwrap();
        assert_eq!(y, vec![S::from_real(3.0), S::from_real(7.0)]);
    }

    #[test]
    fn transpose_spmv() {
        // 2×3 matrix [[1,2,0],[0,3,4]]; transpose is 3×2
        let m = CsrMatrix::from_csr(
            2,
            3,
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![
                S::from_real(1.0),
                S::from_real(2.0),
                S::from_real(3.0),
                S::from_real(4.0),
            ],
        );
        let x = vec![S::from_real(1.0), S::from_real(2.0)];
        let mut y = vec![S::zero(); 3];
        m.spmv_transpose_scaled(S::one(), &x, S::zero(), &mut y)
            .unwrap();
        assert_eq!(
            y,
            vec![S::from_real(1.0), S::from_real(8.0), S::from_real(8.0)]
        );
    }
}
