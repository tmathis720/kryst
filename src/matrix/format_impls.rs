use std::sync::Arc;

use faer::Mat;

use crate::matrix::{
    format::{AsFormat, CSR_CACHE, key_from_ptr},
    op::LinOp,
    sparse::CsrMatrix,
};

impl AsFormat for CsrMatrix<f64> {
    fn as_csr(&self) -> Option<&CsrMatrix<f64>> {
        Some(self)
    }

    fn to_csr_cached(&self, _drop_tol: f64) -> Arc<CsrMatrix<f64>> {
        Arc::new(self.clone())
    }
}

impl AsFormat for Mat<f64> {
    fn to_csr_cached(&self, drop_tol: f64) -> Arc<CsrMatrix<f64>> {
        let base_ptr = self as *const Mat<f64> as usize;
        let structure_id = LinOp::structure_id(self).0;
        let key = key_from_ptr(base_ptr, structure_id, drop_tol);
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
}
