use std::sync::Arc;

use faer::Mat;

use crate::matrix::{
    csc::CscMatrix,
    format::{AsFormat, CSC_CACHE, CSR_CACHE, csc_key_from_ptr, key_from_ptr},
    op::{DenseOp, LinOp},
    sparse::CsrMatrix,
};

impl AsFormat for CsrMatrix<f64> {
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
}

impl AsFormat for Mat<f64> {
    fn to_csr_cached(&self, drop_tol: f64) -> Arc<CsrMatrix<f64>> {
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

    fn as_csc(&self) -> Option<&CscMatrix<f64>> {
        None
    }

    fn to_csc_cached(&self, drop_tol: f64) -> Arc<CscMatrix<f64>> {
        let base_ptr = self as *const Mat<f64> as usize;
        let structure_id = LinOp::structure_id(self).0;
        let key = csc_key_from_ptr(base_ptr, structure_id, drop_tol);
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

impl AsFormat for DenseOp {
    fn to_csr_cached(&self, drop_tol: f64) -> Arc<CsrMatrix<f64>> {
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

    fn as_csc(&self) -> Option<&CscMatrix<f64>> {
        None
    }

    fn to_csc_cached(&self, drop_tol: f64) -> Arc<CscMatrix<f64>> {
        let inner = self.inner();
        let base_ptr = inner as *const Mat<f64> as usize;
        let sid = self.structure_id().0;
        let key = csc_key_from_ptr(base_ptr, sid, drop_tol);
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

impl AsFormat for CscMatrix<f64> {
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
