#![cfg(feature = "backend-faer")]

//! Faer-backed implementations of [`AsFormat`](crate::matrix::format::AsFormat) along with
//! fast CSR/CSC conversions.
//!
//! NOTE on caching and invalidation:
//! - Dense `faer::Mat<f64>` does not track `ValuesId` (returns 0). Conversions from raw Mat
//!   will not auto-invalidate on numeric changes.
//! - Wrap dense matrices in `DenseOp` and call `mark_values_changed()` after in-place updates
//!   to ensure CSC/CSR cache keys include the new `ValuesId`, triggering correct refreshes.

use std::sync::Arc;

use crate::algebra::prelude::*;
use crate::matrix::{
    backend::DefaultBackend,
    csc::CscMatrix,
    format::{AsFormat, format_key_from_ptr, get_or_insert_csc, get_or_insert_csr},
    op::{DenseOp, LinOp, StructureId, ValuesId},
    sparse::CsrMatrix,
};
use faer::Mat;

impl<S> AsFormat<S, DefaultBackend> for CsrMatrix<f64>
where
    S: KrystScalar<Real = f64>,
{
    fn as_csr(&self) -> Option<&CsrMatrix<f64>> {
        Some(self)
    }

    fn to_csr_cached(&self, _drop_tol: f64) -> Arc<CsrMatrix<f64>> {
        Arc::new(self.clone())
    }

    fn as_csc(&self) -> Option<&CscMatrix<f64>> {
        None
    }

    fn to_csc_cached(&self, _drop_tol: f64) -> Arc<CscMatrix<f64>> {
        Arc::new(csr_to_csc(self))
    }

    fn structure_id_for_cache(&self) -> StructureId {
        LinOp::structure_id(self)
    }

    fn values_id_for_cache(&self) -> ValuesId {
        LinOp::values_id(self)
    }
}

impl<S> AsFormat<S, DefaultBackend> for Mat<f64>
where
    S: KrystScalar<Real = f64>,
{
    fn to_csr_cached(&self, drop_tol: f64) -> Arc<CsrMatrix<f64>> {
        let key = format_key_from_ptr(
            self as *const Mat<f64> as usize,
            LinOp::structure_id(self),
            LinOp::values_id(self),
            drop_tol,
        );
        get_or_insert_csr(key, || {
            let csr = CsrMatrix::<f64>::from_dense(self, drop_tol)
                .expect("dense-to-CSR conversion should succeed for real scalars");
            Arc::new(csr)
        })
    }

    fn as_csc(&self) -> Option<&CscMatrix<f64>> {
        None
    }

    fn to_csc_cached(&self, drop_tol: f64) -> Arc<CscMatrix<f64>> {
        let key = format_key_from_ptr(
            self as *const Mat<f64> as usize,
            LinOp::structure_id(self),
            LinOp::values_id(self),
            drop_tol,
        );
        get_or_insert_csc(key, || {
            let csc = CscMatrix::<f64>::from_dense(self, drop_tol)
                .expect("dense-to-CSC conversion should succeed for real scalars");
            Arc::new(csc)
        })
    }

    fn structure_id_for_cache(&self) -> StructureId {
        LinOp::structure_id(self)
    }

    fn values_id_for_cache(&self) -> ValuesId {
        LinOp::values_id(self)
    }
}

impl<S> AsFormat<S, DefaultBackend> for DenseOp
where
    S: KrystScalar<Real = f64>,
{
    fn to_csr_cached(&self, drop_tol: f64) -> Arc<CsrMatrix<f64>> {
        let inner = self.inner();
        let key = format_key_from_ptr(
            inner as *const Mat<f64> as usize,
            self.structure_id(),
            self.values_id(),
            drop_tol,
        );
        get_or_insert_csr(key, || {
            let csr = CsrMatrix::<f64>::from_dense(inner, drop_tol)
                .expect("dense-to-CSR conversion should succeed for real scalars");
            Arc::new(csr)
        })
    }

    fn as_csc(&self) -> Option<&CscMatrix<f64>> {
        None
    }

    fn to_csc_cached(&self, drop_tol: f64) -> Arc<CscMatrix<f64>> {
        let inner = self.inner();
        let key = format_key_from_ptr(
            inner as *const Mat<f64> as usize,
            self.structure_id(),
            self.values_id(),
            drop_tol,
        );
        get_or_insert_csc(key, || {
            let csc = CscMatrix::<f64>::from_dense(inner, drop_tol)
                .expect("dense-to-CSC conversion should succeed for real scalars");
            Arc::new(csc)
        })
    }

    fn structure_id_for_cache(&self) -> StructureId {
        self.structure_id()
    }

    fn values_id_for_cache(&self) -> ValuesId {
        self.values_id()
    }
}

impl<S> AsFormat<S, DefaultBackend> for CscMatrix<f64>
where
    S: KrystScalar<Real = f64>,
{
    fn as_csr(&self) -> Option<&CsrMatrix<f64>> {
        None
    }

    fn to_csr_cached(&self, _drop_tol: f64) -> Arc<CsrMatrix<f64>> {
        Arc::new(csc_to_csr(self))
    }

    fn as_csc(&self) -> Option<&CscMatrix<f64>> {
        Some(self)
    }

    fn to_csc_cached(&self, _drop_tol: f64) -> Arc<CscMatrix<f64>> {
        Arc::new(self.clone())
    }

    fn structure_id_for_cache(&self) -> StructureId {
        LinOp::structure_id(self)
    }

    fn values_id_for_cache(&self) -> ValuesId {
        LinOp::values_id(self)
    }
}

// --- Local helpers: fast CSR<->CSC conversion without densifying ----------
pub(crate) fn csr_to_csc(a: &CsrMatrix<f64>) -> CscMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let ap = a.row_ptr();
    let aj = a.col_idx();
    let av = a.values();
    let nnz = av.len();

    let mut col_ptr = vec![0usize; n + 1];
    for &j in aj {
        col_ptr[j + 1] += 1;
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }

    let mut next = col_ptr.clone();
    let mut row_idx = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    for i in 0..m {
        for p in ap[i]..ap[i + 1] {
            let j = aj[p];
            let q = next[j];
            row_idx[q] = i;
            values[q] = av[p];
            next[j] += 1;
        }
    }
    CscMatrix::from_csc(m, n, col_ptr, row_idx, values)
}

pub(crate) fn csc_to_csr(a: &CscMatrix<f64>) -> CsrMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let cp = a.col_ptr();
    let ri = a.row_idx();
    let vv = a.values();
    let nnz = vv.len();

    let mut row_ptr = vec![0usize; m + 1];
    for &i in ri {
        row_ptr[i + 1] += 1;
    }
    for i in 0..m {
        row_ptr[i + 1] += row_ptr[i];
    }

    let mut next = row_ptr.clone();
    let mut col_idx = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    for j in 0..n {
        for p in cp[j]..cp[j + 1] {
            let i = ri[p];
            let q = next[i];
            col_idx[q] = j;
            values[q] = vv[p];
            next[i] += 1;
        }
    }
    CsrMatrix::from_csr(m, n, row_ptr, col_idx, values)
}
