//! Real-only conversion helpers for `LinOp<S = S>`.
//!
//! These functions are intended for AMG and factorization workflows that
//! operate on real-valued operators. All APIs here assume `S = f64` and should
//! not be used with complex `LinOp` implementations.
use std::sync::Arc;

use faer::Mat;

use crate::algebra::prelude::*;
use crate::error::KError;
#[cfg(not(feature = "complex"))]
use crate::matrix::DistCsrOp;
use crate::matrix::{
    backend::DefaultBackend,
    csc::CscMatrix,
    csr::CsrMatrix as ScalarCsrMatrix,
    format::{AsFormat, FormatHint},
    op::{DenseOp, GenericCsrOp, LinOp, wrap_with_comm},
    sparse::CsrMatrix,
};

/// Build a helpful error for unsupported operator conversions.
///
/// `where_` is the function name (e.g., "to_csr_cached") and `target` is the
/// desired target format ("CSR", "CSC", "dense").
fn unsupported_linop_err(op: &dyn LinOp<S = S>, where_: &str, target: &str) -> KError {
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

fn scalar_csr_to_sparse<S: KrystScalar>(matrix: &ScalarCsrMatrix<S>) -> CsrMatrix<S> {
    CsrMatrix::from_csr(
        matrix.nrows,
        matrix.ncols,
        matrix.rowptr.clone(),
        matrix.colind.clone(),
        matrix.values.clone(),
    )
}

#[cfg(feature = "complex")]
fn csr_to_dense_complex(csr: &CsrMatrix<S>) -> Mat<S> {
    let m = csr.nrows();
    let n = csr.ncols();
    let mut dense = Mat::<S>::zeros(m, n);
    let rp = csr.row_ptr();
    let ci = csr.col_idx();
    let vv = csr.values();
    for i in 0..m {
        for p in rp[i]..rp[i + 1] {
            dense[(i, ci[p])] = vv[p];
        }
    }
    dense
}

#[cfg(feature = "complex")]
fn csc_to_dense_complex(csc: &CscMatrix<S>) -> Mat<S> {
    let m = csc.nrows();
    let n = csc.ncols();
    let mut dense = Mat::<S>::zeros(m, n);
    let cp = csc.col_ptr();
    let ri = csc.row_idx();
    let vv = csc.values();
    for j in 0..n {
        for p in cp[j]..cp[j + 1] {
            dense[(ri[p], j)] = vv[p];
        }
    }
    dense
}

#[cfg(feature = "complex")]
fn dense_to_csr_complex(dense: &Mat<S>, drop_tol: R) -> CsrMatrix<S> {
    let nrows = dense.nrows();
    let ncols = dense.ncols();
    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..nrows {
        for j in 0..ncols {
            let v = dense[(i, j)];
            if v.abs() >= drop_tol {
                col_idx.push(j);
                values.push(v);
            }
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(nrows, ncols, row_ptr, col_idx, values)
}

#[cfg(feature = "complex")]
fn dense_to_csc_complex(dense: &Mat<S>, drop_tol: R) -> CscMatrix<S> {
    let nrows = dense.nrows();
    let ncols = dense.ncols();
    let mut col_ptr = Vec::with_capacity(ncols + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0);
    for j in 0..ncols {
        for i in 0..nrows {
            let v = dense[(i, j)];
            if v.abs() >= drop_tol {
                row_idx.push(i);
                values.push(v);
            }
        }
        col_ptr.push(row_idx.len());
    }
    CscMatrix::from_csc(nrows, ncols, col_ptr, row_idx, values)
}

#[cfg(feature = "complex")]
fn csr_to_csc_complex(csr: &CsrMatrix<S>) -> CscMatrix<S> {
    let m = csr.nrows();
    let n = csr.ncols();
    let ap = csr.row_ptr();
    let aj = csr.col_idx();
    let av = csr.values();
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
    let mut values = vec![S::zero(); nnz];
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

#[cfg(feature = "complex")]
fn csc_to_csr_complex(csc: &CscMatrix<S>) -> CsrMatrix<S> {
    let m = csc.nrows();
    let n = csc.ncols();
    let cp = csc.col_ptr();
    let ri = csc.row_idx();
    let vv = csc.values();
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
    let mut values = vec![S::zero(); nnz];
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
/// Try to borrow a CSR matrix if the operator is already CSR.
pub fn try_as_csr(pmat: &dyn LinOp<S = S>) -> Option<&CsrMatrix<f64>> {
    pmat.as_any().downcast_ref::<CsrMatrix<f64>>()
}

/// Convert a matrix to CSR, caching dense conversions.
///
/// This only applies to real-valued operators with `S = f64`. Complex builds
/// are expected to use the underlying real operator explicitly rather than
/// these helpers.
/// # Errors
/// Returns a recoverable `KError::InvalidInput` with guidance when `pmat` is
/// an unsupported `LinOp` type. See message for how to wrap with
/// `DenseOp`/`CsrOp` or implement `AsFormat` to enable cached conversions.
pub fn to_csr_cached(
    pmat: &dyn LinOp<S = S>,
    drop_tol: R,
) -> Result<Arc<CsrMatrix<f64>>, KError> {
    if let Some(csr) = try_as_csr(pmat) {
        return Ok(Arc::new(csr.clone()));
    }
    if let Some(csc) = pmat.as_any().downcast_ref::<CscMatrix<f64>>() {
        return Ok(<CscMatrix<f64> as AsFormat<f64, DefaultBackend>>::to_csr_cached(csc, drop_tol));
    }
    if let Some(generic) = pmat.as_any().downcast_ref::<GenericCsrOp<f64>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(Arc::new(csr));
    }
    if let Some(mat) = pmat.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(<Mat<f64> as AsFormat<f64, DefaultBackend>>::to_csr_cached(
            mat, drop_tol,
        ));
    }
    if let Some(dense_op) = pmat.as_any().downcast_ref::<DenseOp>() {
        return Ok(<DenseOp as AsFormat<f64, DefaultBackend>>::to_csr_cached(
            dense_op, drop_tol,
        ));
    }
    Err(unsupported_linop_err(pmat, "to_csr_cached", "CSR"))
}

/// Obtain a CSR matrix from a [`LinOp`], converting and caching if necessary.
#[inline]
pub fn csr_from_linop(
    op: &dyn LinOp<S = S>,
    drop_tol: R,
) -> Result<Arc<CsrMatrix<f64>>, KError> {
    to_csr_cached(op, drop_tol)
}

/// Try to borrow a CSC matrix if the operator is already CSC.
pub fn try_as_csc(pmat: &dyn LinOp<S = S>) -> Option<&CscMatrix<f64>> {
    pmat.as_any().downcast_ref::<CscMatrix<f64>>()
}

/// Convert a matrix to CSC, caching dense/CSR conversions.
///
/// This only applies to real-valued operators with `S = f64`. Complex builds
/// are expected to use the underlying real operator explicitly rather than
/// these helpers.
/// # Errors
/// Returns a recoverable `KError::InvalidInput` with guidance when `pmat` is
/// an unsupported `LinOp` type. See message for how to wrap with
/// `DenseOp`/`CsrOp` or implement `AsFormat` to enable cached conversions.
pub fn to_csc_cached(
    pmat: &dyn LinOp<S = S>,
    drop_tol: R,
) -> Result<Arc<CscMatrix<f64>>, KError> {
    if let Some(csc) = try_as_csc(pmat) {
        return Ok(Arc::new(csc.clone()));
    }
    if let Some(generic) = pmat.as_any().downcast_ref::<GenericCsrOp<f64>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(
            <CsrMatrix<f64> as AsFormat<f64, DefaultBackend>>::to_csc_cached(&csr, drop_tol),
        );
    }
    if let Some(csr) = pmat.as_any().downcast_ref::<CsrMatrix<f64>>() {
        return Ok(<CsrMatrix<f64> as AsFormat<f64, DefaultBackend>>::to_csc_cached(csr, drop_tol));
    }
    if let Some(mat) = pmat.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(<Mat<f64> as AsFormat<f64, DefaultBackend>>::to_csc_cached(
            mat, drop_tol,
        ));
    }
    if let Some(dense_op) = pmat.as_any().downcast_ref::<DenseOp>() {
        return Ok(<DenseOp as AsFormat<f64, DefaultBackend>>::to_csc_cached(
            dense_op, drop_tol,
        ));
    }
    Err(unsupported_linop_err(pmat, "to_csc_cached", "CSC"))
}

/// Obtain a CSC matrix from a [`LinOp`], converting and caching if necessary.
#[inline]
pub fn csc_from_linop(
    op: &dyn LinOp<S = S>,
    drop_tol: R,
) -> Result<Arc<CscMatrix<f64>>, KError> {
    to_csc_cached(op, drop_tol)
}

/// Obtain a dense matrix from a [`LinOp`], converting formats as needed.
///
/// This helper is restricted to real operators (`S = f64`) and materializes
/// the real part of the operator for AMG-oriented routines.
/// # Errors
/// Returns a recoverable `KError::InvalidInput` with guidance when `op` is
/// an unsupported `LinOp` type. See message for how to wrap with `DenseOp` to
/// enable cached conversions.
pub fn dense_from_linop(op: &dyn LinOp<S = S>) -> Result<Mat<f64>, KError> {
    if let Some(mat) = op.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(mat.clone());
    }
    if let Some(dense_op) = op.as_any().downcast_ref::<DenseOp>() {
        return Ok(owned_from_mat(dense_op.inner()));
    }
    if let Some(generic) = op.as_any().downcast_ref::<GenericCsrOp<f64>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(csr.to_dense()?);
    }
    if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<f64>>() {
        return Ok(csr.to_dense()?);
    }
    if let Some(csc) = op.as_any().downcast_ref::<CscMatrix<f64>>() {
        return Ok(csc.to_dense()?);
    }
    Err(unsupported_linop_err(op, "dense_from_linop", "dense"))
}

/// Ensure we have an owned `Mat<f64>` regardless of storage (view vs owned).
/// This clones data when necessary and returns an owned matrix.
pub fn owned_from_mat(mat: &Mat<f64>) -> Mat<f64> {
    mat.clone()
}

/// Convert `op` to a LinOp view with the requested `hint`, preserving communicator.
/// For Dense, returns an owned `faer::Mat<f64>` so preconditioners can safely factorize.
/// Only applies to real-valued operators (`S = f64`).
#[cfg(not(feature = "complex"))]
pub fn materialize_linop_with_hint(
    op: &dyn LinOp<S = S>,
    hint: FormatHint,
    drop_tol: R,
) -> Result<std::sync::Arc<dyn LinOp<S = S>>, KError> {
    let comm = op.comm();

    if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<f64>>() {
        return Ok(match hint {
            FormatHint::Csr => wrap_with_comm(Arc::new(csr.clone()), comm),
            FormatHint::Csc => {
                let csc =
                    <CsrMatrix<f64> as AsFormat<f64, DefaultBackend>>::to_csc_cached(csr, drop_tol);
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let dense = csr.to_dense()?;
                wrap_with_comm(Arc::new(dense), comm)
            }
        });
    }

    if let Some(csc) = op.as_any().downcast_ref::<CscMatrix<f64>>() {
        return Ok(match hint {
            FormatHint::Csr => {
                let csr =
                    <CscMatrix<f64> as AsFormat<f64, DefaultBackend>>::to_csr_cached(csc, drop_tol);
                wrap_with_comm(csr, comm)
            }
            FormatHint::Csc => wrap_with_comm(Arc::new(csc.clone()), comm),
            FormatHint::Dense => {
                let dense = csc.to_dense()?;
                wrap_with_comm(Arc::new(dense), comm)
            }
        });
    }

    if let Some(m) = op.as_any().downcast_ref::<Mat<f64>>() {
        return Ok(match hint {
            FormatHint::Csr => {
                let csr = <Mat<f64> as AsFormat<f64, DefaultBackend>>::to_csr_cached(m, drop_tol);
                wrap_with_comm(csr, comm)
            }
            FormatHint::Csc => {
                let csc = <Mat<f64> as AsFormat<f64, DefaultBackend>>::to_csc_cached(m, drop_tol);
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let owned = owned_from_mat(m);
                wrap_with_comm(Arc::new(owned), comm)
            }
        });
    }

    if let Some(dense_op) = op.as_any().downcast_ref::<DenseOp>() {
        return Ok(match hint {
            FormatHint::Csr => {
                let csr =
                    <DenseOp as AsFormat<f64, DefaultBackend>>::to_csr_cached(dense_op, drop_tol);
                wrap_with_comm(csr, comm)
            }
            FormatHint::Csc => {
                let csc =
                    <DenseOp as AsFormat<f64, DefaultBackend>>::to_csc_cached(dense_op, drop_tol);
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let owned = owned_from_mat(dense_op.inner());
                wrap_with_comm(Arc::new(owned), comm)
            }
        });
    }

    if let Some(generic) = op.as_any().downcast_ref::<GenericCsrOp<f64>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(match hint {
            FormatHint::Csr => wrap_with_comm(Arc::new(csr.clone()), comm),
            FormatHint::Csc => {
                let csc = <CsrMatrix<f64> as AsFormat<f64, DefaultBackend>>::to_csc_cached(
                    &csr, drop_tol,
                );
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let dense = csr.to_dense()?;
                wrap_with_comm(Arc::new(dense), comm)
            }
        });
    }

    #[cfg(not(feature = "complex"))]
    if let Some(dist) = op.as_any().downcast_ref::<DistCsrOp>() {
        let local = dist.local_matrix();
        return Ok(match hint {
            FormatHint::Csr => wrap_with_comm(Arc::new(local.clone()), comm),
            FormatHint::Csc => {
                let csc = <CsrMatrix<f64> as AsFormat<f64, DefaultBackend>>::to_csc_cached(
                    &local, drop_tol,
                );
                wrap_with_comm(csc, comm)
            }
            FormatHint::Dense => {
                let dense = local.to_dense()?;
                wrap_with_comm(Arc::new(dense), comm)
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

/// Real-only conversion helper for complex builds.
#[cfg(feature = "complex")]
pub fn materialize_linop_with_hint(
    op: &dyn LinOp<S = S>,
    hint: FormatHint,
    drop_tol: R,
) -> Result<std::sync::Arc<dyn LinOp<S = S>>, KError> {
    let comm = op.comm();

    if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<S>>() {
        return Ok(match hint {
            FormatHint::Csr => wrap_with_comm(Arc::new(csr.clone()), comm),
            FormatHint::Csc => {
                let csc = csr_to_csc_complex(csr);
                wrap_with_comm(Arc::new(csc), comm)
            }
            FormatHint::Dense => {
                let dense = csr_to_dense_complex(csr);
                wrap_with_comm(Arc::new(dense), comm)
            }
        });
    }

    if let Some(csc) = op.as_any().downcast_ref::<CscMatrix<S>>() {
        return Ok(match hint {
            FormatHint::Csr => {
                let csr = csc_to_csr_complex(csc);
                wrap_with_comm(Arc::new(csr), comm)
            }
            FormatHint::Csc => wrap_with_comm(Arc::new(csc.clone()), comm),
            FormatHint::Dense => {
                let dense = csc_to_dense_complex(csc);
                wrap_with_comm(Arc::new(dense), comm)
            }
        });
    }

    if let Some(m) = op.as_any().downcast_ref::<Mat<S>>() {
        return Ok(match hint {
            FormatHint::Csr => {
                let csr = dense_to_csr_complex(m, drop_tol);
                wrap_with_comm(Arc::new(csr), comm)
            }
            FormatHint::Csc => {
                let csc = dense_to_csc_complex(m, drop_tol);
                wrap_with_comm(Arc::new(csc), comm)
            }
            FormatHint::Dense => {
                let owned = m.clone();
                wrap_with_comm(Arc::new(owned), comm)
            }
        });
    }

    if let Some(generic) = op.as_any().downcast_ref::<GenericCsrOp<S>>() {
        let csr = scalar_csr_to_sparse(generic.matrix());
        return Ok(match hint {
            FormatHint::Csr => wrap_with_comm(Arc::new(csr.clone()), comm),
            FormatHint::Csc => {
                let csc = csr_to_csc_complex(&csr);
                wrap_with_comm(Arc::new(csc), comm)
            }
            FormatHint::Dense => {
                let dense = csr_to_dense_complex(&csr);
                wrap_with_comm(Arc::new(dense), comm)
            }
        });
    }

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
