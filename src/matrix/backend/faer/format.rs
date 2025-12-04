#![cfg(feature = "backend-faer")]

//! Faer-backed implementations of [`AsFormat`](crate::matrix::format::AsFormat) and
//! backend-specific format caches.
//!
//! NOTE on caching and invalidation:
//! - Dense `faer::Mat<f64>` does not track `ValuesId` (returns 0). Conversions from raw Mat
//!   will not auto-invalidate on numeric changes.
//! - Wrap dense matrices in `DenseOp` and call `mark_values_changed()` after in-place updates
//!   to ensure CSC/CSR cache keys include the new `ValuesId`, triggering correct refreshes.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};

use crate::matrix::{
    csc::CscMatrix,
    format::AsFormat,
    op::{DenseOp, LinOp},
    sparse::CsrMatrix,
};
use once_cell::sync::Lazy;
use faer::Mat;

#[derive(Clone, Copy, Debug)]
struct CsrKey {
    base_ptr: usize,
    structure_id: u64,
    values_id: u64,
    drop_tol_bits: u64,
}

impl PartialEq for CsrKey {
    fn eq(&self, other: &Self) -> bool {
        self.base_ptr == other.base_ptr
            && self.structure_id == other.structure_id
            && self.values_id == other.values_id
            && self.drop_tol_bits == other.drop_tol_bits
    }
}
impl Eq for CsrKey {}
impl Hash for CsrKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base_ptr.hash(state);
        self.structure_id.hash(state);
        self.values_id.hash(state);
        self.drop_tol_bits.hash(state);
    }
}

#[derive(Clone, Copy, Debug)]
struct CscKey {
    base_ptr: usize,
    structure_id: u64,
    values_id: u64,
    drop_tol_bits: u64,
}

impl PartialEq for CscKey {
    fn eq(&self, other: &Self) -> bool {
        self.base_ptr == other.base_ptr
            && self.structure_id == other.structure_id
            && self.values_id == other.values_id
            && self.drop_tol_bits == other.drop_tol_bits
    }
}
impl Eq for CscKey {}
impl Hash for CscKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base_ptr.hash(state);
        self.structure_id.hash(state);
        self.values_id.hash(state);
        self.drop_tol_bits.hash(state);
    }
}

/// Global cache of dense->CSR conversions.
static CSR_CACHE: Lazy<Mutex<HashMap<CsrKey, Weak<CsrMatrix<f64>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Global cache of dense->CSC conversions.
static CSC_CACHE: Lazy<Mutex<HashMap<CscKey, Weak<CscMatrix<f64>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[inline]
fn key_from_ptr(ptr: usize, structure_id: u64, values_id: u64, drop_tol: f64) -> CsrKey {
    CsrKey {
        base_ptr: ptr,
        structure_id,
        values_id,
        drop_tol_bits: drop_tol.to_bits(),
    }
}

#[inline]
fn csc_key_from_ptr(ptr: usize, structure_id: u64, values_id: u64, drop_tol: f64) -> CscKey {
    CscKey {
        base_ptr: ptr,
        structure_id,
        values_id,
        drop_tol_bits: drop_tol.to_bits(),
    }
}

impl AsFormat<f64> for CsrMatrix<f64> {
    type Csr = CsrMatrix<f64>;
    type Csc = CscMatrix<f64>;

    fn as_csr(&self) -> Option<&Self::Csr> {
        Some(self)
    }

    fn to_csr_cached(&self, _drop_tol: f64) -> Arc<Self::Csr> {
        Arc::new(self.clone())
    }

    fn as_csc(&self) -> Option<&Self::Csc> {
        None
    }

    fn to_csc_cached(&self, _drop_tol: f64) -> Arc<Self::Csc> {
        Arc::new(csr_to_csc(self))
    }
}

impl AsFormat<f64> for Mat<f64> {
    type Csr = CsrMatrix<f64>;
    type Csc = CscMatrix<f64>;

    fn to_csr_cached(&self, drop_tol: f64) -> Arc<Self::Csr> {
        let base_ptr = self as *const Mat<f64> as usize;
        let structure_id = LinOp::structure_id(self).0;
        let values_id = LinOp::values_id(self).0;
        let key = key_from_ptr(base_ptr, structure_id, values_id, drop_tol);
        if let Some(existing) = {
            let cache = CSR_CACHE.lock().unwrap();
            cache.get(&key).and_then(|w| w.upgrade())
        } {
            return existing;
        }
        let csr = CsrMatrix::from_dense(self, drop_tol);
        let arc = Arc::new(csr);
        let mut cache = CSR_CACHE.lock().unwrap();
        cache.insert(key, Arc::downgrade(&arc));
        arc
    }

    fn as_csc(&self) -> Option<&Self::Csc> {
        None
    }

    fn to_csc_cached(&self, drop_tol: f64) -> Arc<Self::Csc> {
        let base_ptr = self as *const Mat<f64> as usize;
        let structure_id = LinOp::structure_id(self).0;
        let values_id = LinOp::values_id(self).0;
        let key = csc_key_from_ptr(base_ptr, structure_id, values_id, drop_tol);
        if let Some(existing) = {
            let cache = CSC_CACHE.lock().unwrap();
            cache.get(&key).and_then(|w| w.upgrade())
        } {
            return existing;
        }
        let csc = CscMatrix::from_dense(self, drop_tol);
        let arc = Arc::new(csc);
        let mut cache = CSC_CACHE.lock().unwrap();
        cache.insert(key, Arc::downgrade(&arc));
        arc
    }
}

impl AsFormat<f64> for DenseOp {
    type Csr = CsrMatrix<f64>;
    type Csc = CscMatrix<f64>;

    fn to_csr_cached(&self, drop_tol: f64) -> Arc<Self::Csr> {
        let inner = self.inner();
        let base_ptr = inner as *const Mat<f64> as usize;
        let sid = self.structure_id().0;
        let vid = self.values_id().0;
        let key = key_from_ptr(base_ptr, sid, vid, drop_tol);
        if let Some(existing) = {
            let cache = CSR_CACHE.lock().unwrap();
            cache.get(&key).and_then(|w| w.upgrade())
        } {
            return existing;
        }
        let csr = CsrMatrix::from_dense(inner, drop_tol);
        let arc = Arc::new(csr);
        let mut cache = CSR_CACHE.lock().unwrap();
        cache.insert(key, Arc::downgrade(&arc));
        arc
    }

    fn as_csc(&self) -> Option<&Self::Csc> {
        None
    }

    fn to_csc_cached(&self, drop_tol: f64) -> Arc<Self::Csc> {
        let inner = self.inner();
        let base_ptr = inner as *const Mat<f64> as usize;
        let sid = self.structure_id().0;
        let vid = self.values_id().0;
        let key = csc_key_from_ptr(base_ptr, sid, vid, drop_tol);
        if let Some(existing) = {
            let cache = CSC_CACHE.lock().unwrap();
            cache.get(&key).and_then(|w| w.upgrade())
        } {
            return existing;
        }
        let csc = CscMatrix::from_dense(inner, drop_tol);
        let arc = Arc::new(csc);
        let mut cache = CSC_CACHE.lock().unwrap();
        cache.insert(key, Arc::downgrade(&arc));
        arc
    }
}

impl AsFormat<f64> for CscMatrix<f64> {
    type Csr = CsrMatrix<f64>;
    type Csc = CscMatrix<f64>;

    fn as_csr(&self) -> Option<&Self::Csr> {
        None
    }

    fn to_csr_cached(&self, _drop_tol: f64) -> Arc<Self::Csr> {
        Arc::new(csc_to_csr(self))
    }

    fn as_csc(&self) -> Option<&Self::Csc> {
        Some(self)
    }

    fn to_csc_cached(&self, _drop_tol: f64) -> Arc<Self::Csc> {
        Arc::new(self.clone())
    }
}

// --- Local helpers: fast CSR<->CSC conversion without densifying ----------
fn csr_to_csc(a: &CsrMatrix<f64>) -> CscMatrix<f64> {
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

fn csc_to_csr(a: &CscMatrix<f64>) -> CsrMatrix<f64> {
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
