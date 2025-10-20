use std::sync::Arc;

use faer::Mat;

use crate::error::KError;
use crate::matrix::{csc::CscMatrix, format::FormatHint, op::LinOp, sparse::CsrMatrix};

#[cfg(not(feature = "complex"))]
use crate::matrix::{
    DistCsrOp,
    csr::CsrMatrix as ScalarCsrMatrix,
    format::AsFormat,
    op::{DenseOp, GenericCsrOp, wrap_with_comm},
};

/// Build a helpful error for unsupported operator conversions.
///
/// `where_` is the function name (e.g., "to_csr_cached") and `target` is the
/// desired target format ("CSR", "CSC", "dense").
#[cfg(not(feature = "complex"))]
fn unsupported_linop_err(op: &dyn LinOp<S = f64>, where_: &str, target: &str) -> KError {
    let sid = op.structure_id().0;
    let vid = op.values_id().0;
    let has_ids = sid != 0 || vid != 0;

    let mut help = String::new();
    help.push_str(&format!(
        "convert::{where_}: unsupported LinOp type for conversion to {target}.\n"
    ));
    help.push_str("- Recovery options:\n");
    help.push_str("  • If you have a dense matrix (`faer::Mat<f64>`), wrap it with `DenseOp` so structure/values IDs are tracked and conversions can be cached:\n");
    help.push_str("      let op = DenseOp::new(Arc::new(mat));\n");
    help.push_str(
        "      // after in-place updates: op.mark_values_changed() / op.mark_structure_changed()\n",
    );
    help.push_str(
        "  • If you have a CSR matrix (`CsrMatrix<f64>`), wrap it with `CsrOp` likewise:\n",
    );
    help.push_str("      let op = CsrOp::new(Arc::new(csr));\n");
    help.push_str(
        "  • If you have a generic CSR operator (`GenericCsrOp<f64>`), conversions clone its storage automatically.\n",
    );
    help.push_str("  • If this is your own LinOp type, implement `matrix::format::AsFormat` for it to enable cached conversions.\n");
    help.push_str(
        "  • If running distributed, attach the communicator with `wrap_with_comm(op, comm)`.\n",
    );

    if !has_ids {
        help.push_str("\nNote: this operator reports unknown StructureId/ValuesId (both 0). \
                       Wrapping with `DenseOp`/`CsrOp` enables precise cache keys and efficient reuse.\n");
    }

    KError::InvalidInput(help)
}

#[cfg(not(feature = "complex"))]
fn scalar_csr_to_sparse(matrix: &ScalarCsrMatrix<f64>) -> CsrMatrix<f64> {
    CsrMatrix::from_csr(
        matrix.nrows,
        matrix.ncols,
        matrix.rowptr.clone(),
        matrix.colind.clone(),
        matrix.values.clone(),
    )
}

/// Try to borrow a CSR matrix if the operator is already CSR.
#[cfg(not(feature = "complex"))]
pub fn try_as_csr(pmat: &dyn LinOp<S = f64>) -> Option<&CsrMatrix<f64>> {
    pmat.as_any().downcast_ref::<CsrMatrix<f64>>()
}

#[cfg(feature = "complex")]
pub fn try_as_csr(_pmat: &dyn LinOp<S = f64>) -> Option<&CsrMatrix<f64>> {
    None
}

/// Convert a matrix to CSR, caching dense conversions.
///
/// # Errors
/// Returns a recoverable `KError::InvalidInput` with guidance when `pmat` is
/// an unsupported `LinOp` type. See message for how to wrap with
/// `DenseOp`/`CsrOp` or implement `AsFormat` to enable cached conversions.
#[cfg(not(feature = "complex"))]
pub fn to_csr_cached(
    pmat: &dyn LinOp<S = f64>,
    drop_tol: f64,
) -> Result<Arc<CsrMatrix<f64>>, KError> {
    if let Some(csr) = try_as_csr(pmat) {
        return Ok(Arc::new(csr.clone()));
    }
    if let Some(generic) = pmat.as_any().downcast_ref::<GenericCsrOp<f64>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(Arc::new(csr));
    }
    if let Some(mat) = pmat.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(mat.to_csr_cached(drop_tol));
    }
    Err(unsupported_linop_err(pmat, "to_csr_cached", "CSR"))
}

#[cfg(feature = "complex")]
pub fn to_csr_cached(
    _pmat: &dyn LinOp<S = f64>,
    _drop_tol: f64,
) -> Result<Arc<CsrMatrix<f64>>, KError> {
    Err(KError::InvalidInput(
        "to_csr_cached is not available with the `complex` feature enabled".into(),
    ))
}

/// Obtain a CSR matrix from a [`LinOp`], converting and caching if necessary.
#[inline]
#[cfg(not(feature = "complex"))]
pub fn csr_from_linop(
    op: &dyn LinOp<S = f64>,
    drop_tol: f64,
) -> Result<Arc<CsrMatrix<f64>>, KError> {
    to_csr_cached(op, drop_tol)
}

#[cfg(feature = "complex")]
pub fn csr_from_linop(
    op: &dyn LinOp<S = f64>,
    drop_tol: f64,
) -> Result<Arc<CsrMatrix<f64>>, KError> {
    to_csr_cached(op, drop_tol)
}

/// Try to borrow a CSC matrix if the operator is already CSC.
#[cfg(not(feature = "complex"))]
pub fn try_as_csc(pmat: &dyn LinOp<S = f64>) -> Option<&CscMatrix<f64>> {
    pmat.as_any().downcast_ref::<CscMatrix<f64>>()
}

#[cfg(feature = "complex")]
pub fn try_as_csc(_pmat: &dyn LinOp<S = f64>) -> Option<&CscMatrix<f64>> {
    None
}

/// Convert a matrix to CSC, caching dense/CSR conversions.
///
/// # Errors
/// Returns a recoverable `KError::InvalidInput` with guidance when `pmat` is
/// an unsupported `LinOp` type. See message for how to wrap with
/// `DenseOp`/`CsrOp` or implement `AsFormat` to enable cached conversions.
#[cfg(not(feature = "complex"))]
pub fn to_csc_cached(
    pmat: &dyn LinOp<S = f64>,
    drop_tol: f64,
) -> Result<Arc<CscMatrix<f64>>, KError> {
    if let Some(csc) = try_as_csc(pmat) {
        return Ok(Arc::new(csc.clone()));
    }
    if let Some(mat) = pmat.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(mat.to_csc_cached(drop_tol));
    }
    if let Some(generic) = pmat.as_any().downcast_ref::<GenericCsrOp<f64>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(AsFormat::to_csc_cached(&csr, drop_tol));
    }
    if let Some(csr) = pmat.as_any().downcast_ref::<CsrMatrix<f64>>() {
        return Ok(csr.to_csc_cached(drop_tol));
    }
    Err(unsupported_linop_err(pmat, "to_csc_cached", "CSC"))
}

#[cfg(feature = "complex")]
pub fn to_csc_cached(
    _pmat: &dyn LinOp<S = f64>,
    _drop_tol: f64,
) -> Result<Arc<CscMatrix<f64>>, KError> {
    Err(KError::InvalidInput(
        "to_csc_cached is not available with the `complex` feature enabled".into(),
    ))
}

/// Obtain a CSC matrix from a [`LinOp`], converting and caching if necessary.
#[inline]
#[cfg(not(feature = "complex"))]
pub fn csc_from_linop(
    op: &dyn LinOp<S = f64>,
    drop_tol: f64,
) -> Result<Arc<CscMatrix<f64>>, KError> {
    to_csc_cached(op, drop_tol)
}

#[cfg(feature = "complex")]
pub fn csc_from_linop(
    op: &dyn LinOp<S = f64>,
    drop_tol: f64,
) -> Result<Arc<CscMatrix<f64>>, KError> {
    to_csc_cached(op, drop_tol)
}

/// Obtain a dense matrix from a [`LinOp`], converting formats as needed.
///
/// # Errors
/// Returns a recoverable `KError::InvalidInput` with guidance when `op` is
/// an unsupported `LinOp` type. See message for how to wrap with `DenseOp` to
/// enable cached conversions.
#[cfg(not(feature = "complex"))]
pub fn dense_from_linop(op: &dyn LinOp<S = f64>) -> Result<Mat<f64>, KError> {
    if let Some(mat) = op.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(mat.clone());
    }
    if let Some(generic) = op.as_any().downcast_ref::<GenericCsrOp<f64>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(csr.to_dense());
    }
    if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<f64>>() {
        return Ok(csr.to_dense());
    }
    Err(unsupported_linop_err(op, "dense_from_linop", "dense"))
}

#[cfg(feature = "complex")]
pub fn dense_from_linop(_op: &dyn LinOp<S = f64>) -> Result<Mat<f64>, KError> {
    Err(KError::InvalidInput(
        "dense_from_linop is not available with the `complex` feature enabled".into(),
    ))
}

/// Ensure we have an owned `Mat<f64>` regardless of storage (view vs owned).
/// This clones data when necessary and returns an owned matrix.
pub fn owned_from_mat(mat: &Mat<f64>) -> Mat<f64> {
    mat.clone()
}

/// Convert `op` to a LinOp view with the requested `hint`, preserving communicator.
/// For Dense, returns an owned `faer::Mat<f64>` so preconditioners can safely factorize.
#[cfg(not(feature = "complex"))]
pub fn materialize_linop_with_hint(
    op: &dyn LinOp<S = f64>,
    hint: FormatHint,
    drop_tol: f64,
) -> Result<std::sync::Arc<dyn LinOp<S = f64>>, KError> {
    let comm = op.comm();

    // Dense matrix
    if let Some(m) = op.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(match hint {
            FormatHint::Csr => {
                let csr = m.to_csr_cached(drop_tol);
                wrap_with_comm(csr, comm)
            }
            FormatHint::Csc => {
                let csc = m.to_csc_cached(drop_tol);
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let owned = owned_from_mat(m);
                wrap_with_comm(std::sync::Arc::new(owned), comm)
            }
        });
    }

    // CSR matrix
    if let Some(generic) = op.as_any().downcast_ref::<GenericCsrOp<f64>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(match hint {
            FormatHint::Csr => wrap_with_comm(Arc::new(csr.clone()), comm),
            FormatHint::Csc => {
                let csc = AsFormat::to_csc_cached(&csr, drop_tol);
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let dense = csr.to_dense();
                wrap_with_comm(std::sync::Arc::new(dense), comm)
            }
        });
    }
    if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<f64>>() {
        return Ok(match hint {
            FormatHint::Csr => wrap_with_comm(std::sync::Arc::new(csr.clone()), comm),
            FormatHint::Csc => {
                let csc = AsFormat::to_csc_cached(csr, drop_tol);
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let dense = csr.to_dense();
                wrap_with_comm(std::sync::Arc::new(dense), comm)
            }
        });
    }

    // CSC matrix
    if let Some(csc) = op.as_any().downcast_ref::<CscMatrix<f64>>() {
        return Ok(match hint {
            FormatHint::Csr => {
                let csr = AsFormat::to_csr_cached(csc, drop_tol);
                wrap_with_comm(csr, comm)
            }
            FormatHint::Csc => wrap_with_comm(std::sync::Arc::new(csc.clone()), comm),
            FormatHint::Dense => {
                let dense = csc.to_dense();
                wrap_with_comm(std::sync::Arc::new(dense), comm)
            }
        });
    }

    // Distributed CSR operator — expose the on-processor block for lightweight PCs.
    if let Some(dist) = op.as_any().downcast_ref::<DistCsrOp>() {
        return Ok(match hint {
            FormatHint::Csr => {
                let csr = std::sync::Arc::new(dist.local_matrix());
                wrap_with_comm(csr, comm)
            }
            FormatHint::Csc => {
                let csr_local = dist.local_matrix();
                let csc = AsFormat::to_csc_cached(&csr_local, drop_tol);
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let dense = dist.local_matrix().to_dense();
                wrap_with_comm(std::sync::Arc::new(dense), comm)
            }
        });
    }

    // DenseOp wrapper
    if let Some(dense_op) = op.as_any().downcast_ref::<DenseOp>() {
        let inner = dense_op.inner();
        return Ok(match hint {
            FormatHint::Csr => {
                let csr = AsFormat::to_csr_cached(dense_op, drop_tol);
                wrap_with_comm(csr, comm)
            }
            FormatHint::Csc => {
                let csc = AsFormat::to_csc_cached(dense_op, drop_tol);
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let owned = owned_from_mat(inner);
                wrap_with_comm(std::sync::Arc::new(owned), comm)
            }
        });
    }

    // Unsupported operator for conversion (e.g., distributed CSR or custom LinOp)
    let target = match hint {
        FormatHint::Csr => "CSR",
        FormatHint::Csc => "CSC",
        FormatHint::Dense => "dense",
    };
    Err(unsupported_linop_err(
        op,
        "materialize_linop_with_hint",
        target,
    ))
}

#[cfg(feature = "complex")]
pub fn materialize_linop_with_hint(
    _op: &dyn LinOp<S = f64>,
    _hint: FormatHint,
    _drop_tol: f64,
) -> Result<Arc<dyn LinOp<S = f64>>, KError> {
    Err(KError::InvalidInput(
        "materialize_linop_with_hint is not available with the `complex` feature enabled".into(),
    ))
}

#[cfg(all(test, not(feature = "complex")))]
mod tests {
    use super::*;
    use crate::matrix::{DistCsrOp, op_shell::MatShell, sparse::CsrMatrix};
    use crate::parallel::{NoComm, UniverseComm};

    #[test]
    fn to_csr_cached_returns_guidance_on_unsupported_type() {
        // 3x3 shell op that cannot be converted by convert::*.
        let shell = MatShell::new(3, 3, |x, y| {
            y.copy_from_slice(x);
        });

        let err = to_csr_cached(&shell, 0.0).err().unwrap();
        let msg = format!("{err:?}");
        // The guidance should mention the recovery hints:
        assert!(msg.contains("DenseOp"), "error should suggest DenseOp");
        assert!(msg.contains("CsrOp"), "error should suggest CsrOp");
        assert!(msg.contains("AsFormat"), "error should suggest AsFormat");
        assert!(
            msg.contains("wrap_with_comm"),
            "error should suggest wrapping communicator"
        );
    }

    #[test]
    fn dense_from_linop_guidance() {
        let shell = MatShell::new(2, 2, |x, y| y.copy_from_slice(x));
        let err = dense_from_linop(&shell).err().unwrap();
        let msg = format!("{err:?}");
        assert!(
            msg.to_lowercase().contains("dense"),
            "should reference dense target"
        );
        assert!(msg.contains("DenseOp"), "should suggest DenseOp");
    }

    #[test]
    fn materialize_accepts_dist_csr_ops() {
        let comm = UniverseComm::NoComm(NoComm);
        let part = vec![0, 1];
        let local = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![2.0]);
        let dist = DistCsrOp::from_local_rows(1, 0, &local, &part, comm.clone()).unwrap();

        let view = materialize_linop_with_hint(&dist, FormatHint::Csr, 0.0).unwrap();
        let csr = view
            .as_any()
            .downcast_ref::<CsrMatrix<f64>>()
            .expect("converted CSR matrix");
        assert_eq!(csr.dims(), (1, 1));
        assert_eq!(csr.values().len(), 1);
        assert!((csr.values()[0] - 2.0).abs() <= f64::EPSILON);

        let dense = materialize_linop_with_hint(&dist, FormatHint::Dense, 0.0).unwrap();
        let mat = dense
            .as_any()
            .downcast_ref::<faer::Mat<f64>>()
            .expect("converted dense matrix");
        assert!((mat[(0, 0)] - 2.0).abs() <= f64::EPSILON);
    }
}
