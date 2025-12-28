//! Faer-backed matrix implementations of the abstraction traits.

use std::sync::Arc;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::backend::SparseBackend;
use crate::matrix::csc::CscMatrix;
use crate::matrix::format::{BackendFormatSupport, FormatHint, OpFormat};
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;

pub mod dense;
pub mod format;
pub mod sparse;

/// Marker type for the default Faer backend.
pub struct FaerBackend;

pub type DefaultDenseMat<S> = faer::Mat<S>;
pub type DefaultCsrMat<S> = crate::matrix::sparse::CsrMatrix<S>;
pub type DefaultCscMat<S> = crate::matrix::csc::CscMatrix<S>;

impl<S> SparseBackend<S> for FaerBackend
where
    S: KrystScalar<Real = f64>,
{
    const FORMAT_SUPPORT: BackendFormatSupport = BackendFormatSupport::new(true, true, true, false);

    type Csr = CsrMatrix<S::Real>;
    type Csc = CscMatrix<S::Real>;
    type Dense = faer::Mat<S::Real>;

    fn csr_from_dense(dense: &Self::Dense, drop_tol: S::Real) -> Result<Self::Csr, KError> {
        CsrMatrix::<S::Real>::from_dense(dense, drop_tol)
    }

    fn csc_from_csr(csr: &Self::Csr, _drop_tol: S::Real) -> Self::Csc {
        crate::matrix::backend::faer::format::csr_to_csc(csr)
    }

    fn csr_from_csc(csc: &Self::Csc, _drop_tol: S::Real) -> Self::Csr {
        crate::matrix::backend::faer::format::csc_to_csr(csc)
    }

    fn dense_from_csr(csr: &Self::Csr) -> Result<Self::Dense, KError> {
        csr.to_dense()
    }

    fn dense_from_csc(csc: &Self::Csc) -> Result<Self::Dense, KError> {
        csc.to_dense()
    }
}

fn map_format(want: OpFormat) -> Result<FormatHint, KError> {
    match want {
        OpFormat::Dense => Ok(FormatHint::Dense),
        OpFormat::Csr => Ok(FormatHint::Csr),
        OpFormat::Csc => Ok(FormatHint::Csc),
        OpFormat::Any | OpFormat::BlockCsr => Err(KError::Unsupported(
            "faer backend does not support the requested format",
        )),
    }
}

pub fn try_materialize(
    op: Arc<dyn LinOp<S = S>>,
    want: OpFormat,
    drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    if want.is_any() {
        return Ok(op);
    }
    let hint = map_format(want)?;
    crate::matrix::convert::materialize_linop_with_hint(op.as_ref(), hint, drop_tol)
}

pub fn try_materialize_ref(
    op: &dyn LinOp<S = S>,
    want: OpFormat,
    drop_tol: R,
) -> Result<Arc<dyn LinOp<S = S>>, KError> {
    let hint = map_format(want)?;
    crate::matrix::convert::materialize_linop_with_hint(op, hint, drop_tol)
}

#[cfg(test)]
mod tests {
    use crate::matrix::dense_api::DenseMatRef;
    use crate::matrix::sparse::CsrMatrix;
    use crate::matrix::sparse_api::CsrMatRef;
    use crate::matrix::spmv;

    fn assert_dense_ref<T: DenseMatRef<f64>>() {}
    fn assert_csr_ref<T: CsrMatRef<f64>>() {}

    #[test]
    fn faer_mat_satisfies_dense_traits() {
        assert_dense_ref::<faer::Mat<f64>>();
    }

    #[test]
    fn csr_matrix_drives_generic_spmv() {
        assert_csr_ref::<CsrMatrix<f64>>();
        let a = CsrMatrix::from_csr(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0, 2.0, 3.0]);
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        spmv::spmv_csr_serial(&a, &x, &mut y).unwrap();
        assert_eq!(y, vec![3.0, 3.0]);
    }
}
