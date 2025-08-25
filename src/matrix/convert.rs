use std::sync::Arc;

use faer::Mat;

use crate::{
    error::KError,
    matrix::{format::AsFormat, op::LinOp, sparse::CsrMatrix},
};

/// Try to borrow a CSR matrix if the operator is already CSR.
pub fn try_as_csr<'a>(pmat: &'a dyn LinOp<S = f64>) -> Option<&'a CsrMatrix<f64>> {
    pmat.as_any().downcast_ref::<CsrMatrix<f64>>()
}

/// Convert a matrix to CSR, caching dense conversions.
pub fn to_csr_cached(
    pmat: &dyn LinOp<S = f64>,
    drop_tol: f64,
) -> Result<Arc<CsrMatrix<f64>>, KError> {
    if let Some(csr) = try_as_csr(pmat) {
        return Ok(Arc::new(csr.clone()));
    }
    if let Some(mat) = pmat.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(mat.to_csr_cached(drop_tol));
    }
    Err(KError::InvalidInput(
        "to_csr_cached: unsupported LinOp type".into(),
    ))
}

/// Obtain a CSR matrix from a [`LinOp`], converting and caching if necessary.
#[inline]
pub fn csr_from_linop(
    op: &dyn LinOp<S = f64>,
    drop_tol: f64,
) -> Result<Arc<CsrMatrix<f64>>, KError> {
    to_csr_cached(op, drop_tol)
}

/// Obtain a dense matrix from a [`LinOp`], converting formats as needed.
pub fn dense_from_linop(op: &dyn LinOp<S = f64>) -> Result<Mat<f64>, KError> {
    if let Some(mat) = op.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(mat.clone());
    }
    if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<f64>>() {
        return Ok(csr.to_dense());
    }
    Err(KError::InvalidInput(
        "Unsupported operator type for Dense conversion".into(),
    ))
}
