//! Sparse matrix-vector multiplication backends.
//!
//! Canonical high-level entry points:
//! - [`csr_matvec`]     – serial `y = A * x`
//! - [`csr_matvec_par`] – parallel `y = A * x` (Rayon-aware)
//! - [`csr_t_matvec_par`] – `y = A^T * x` using CSC or CSR gather
//!
//! These operate on any type implementing [`CsrAccess`], including
//! [`crate::matrix::csr::CsrMatrix`] and [`crate::matrix::sparse::CsrMatrix`].
pub mod plan;
pub mod scalar;
#[cfg(all(feature = "simd", not(feature = "complex")))]
pub mod sellc;
#[cfg(all(feature = "simd", not(feature = "complex")))]
pub mod simd_csr;

pub use self::plan::{
    SpmvKernel, SpmvPlan, SpmvTuning, build as build_plan, build_owned as build_plan_owned,
};
pub use self::scalar::{spmv_csr_scalar, spmv_scaled_csr, spmv_t_scaled_csr};

use crate::algebra::prelude::*;
use crate::context::ksp_context::BlockVec;
use crate::error::KError;
use crate::matrix::sparse_api::{CscMatRef, CsrMatRef};
#[cfg(feature = "backend-faer")]
use faer::{MatMut, MatRef};

#[inline]
pub fn spmv_scaled_f32_on_pattern(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals32: &[f32],
    alpha: f32,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    assert_eq!(row_ptr.len(), n + 1);
    assert_eq!(y.len(), n);
    if beta == 0.0 {
        y.fill(0.0);
    } else if beta != 1.0 {
        for v in y.iter_mut() {
            *v *= beta;
        }
    }
    for i in 0..n {
        let mut acc = 0.0f32;
        let rs = row_ptr[i];
        let re = row_ptr[i + 1];
        for p in rs..re {
            acc += vals32[p] * x[col_idx[p]];
        }
        y[i] += alpha * acc;
    }
}

#[inline]
pub fn spmv_t_scaled_f32_on_pattern(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals32: &[f32],
    alpha: f32,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    assert_eq!(row_ptr.len(), n + 1);
    if beta == 0.0 {
        y.fill(0.0);
    } else if beta != 1.0 {
        for v in y.iter_mut() {
            *v *= beta;
        }
    }
    for i in 0..n {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let rs = row_ptr[i];
        let re = row_ptr[i + 1];
        for p in rs..re {
            y[col_idx[p]] += alpha * vals32[p] * xi;
        }
    }
}

/// y = A * x using CSR; parallel when `rayon` is enabled.
pub trait CsrAccess<S: KrystScalar>: CsrMatRef<S> {}
impl<S: KrystScalar, T: CsrMatRef<S>> CsrAccess<S> for T {}

#[inline(always)]
fn csr_row_dot<S: KrystScalar>(
    row: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[S],
    x: &[S],
) -> S {
    let start = row_ptr[row];
    let end = row_ptr[row + 1];
    let mut acc = S::zero();
    for idx in start..end {
        acc = acc + vals[idx] * x[col_idx[idx]];
    }
    acc
}

pub fn spmv_csr_serial<S: KrystScalar, A: CsrAccess<S>>(
    a: &A,
    x: &[S],
    y: &mut [S],
) -> Result<(), KError> {
    if x.len() != a.ncols() || y.len() != a.nrows() {
        return Err(KError::InvalidInput(
            "spmv_csr_serial: dimension mismatch".into(),
        ));
    }
    let row_ptr = a.row_ptr();
    let col_idx = a.col_idx();
    let vals = a.values();
    for row in 0..a.nrows() {
        y[row] = csr_row_dot(row, row_ptr, col_idx, vals, x);
    }
    Ok(())
}

#[cfg(feature = "rayon")]
pub fn spmv_csr_parallel<S, A>(a: &A, x: &[S], y: &mut [S]) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    if x.len() != a.ncols() || y.len() != a.nrows() {
        return Err(KError::InvalidInput(
            "spmv_csr_parallel: dimension mismatch".into(),
        ));
    }
    let nrows = a.nrows();
    let row_ptr = a.row_ptr();
    let col_idx = a.col_idx();
    let vals = a.values();
    let tune = crate::algebra::parallel_cfg::parallel_tune();
    if nrows < tune.min_rows_spmv {
        return spmv_csr_serial(a, x, y);
    }

    use rayon::prelude::*;
    let chunk = std::cmp::max(64, tune.chunk_rows_spmv);
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(chunk_id, y_chunk)| {
            let row_start = chunk_id * chunk;
            for (offset, yi) in y_chunk.iter_mut().enumerate() {
                let row = row_start + offset;
                if row >= nrows {
                    break;
                }
                *yi = csr_row_dot(row, row_ptr, col_idx, vals, x);
            }
        });
    Ok(())
}

pub fn spmm_csr_dense<A>(a: &A, x: &BlockVec, y: &mut BlockVec) -> Result<(), KError>
where
    A: CsrAccess<S>,
{
    let (m, n) = (a.nrows(), a.ncols());
    if x.nrows() != n || y.nrows() != m || x.ncols() != y.ncols() {
        return Err(KError::InvalidInput(
            "spmm_csr_dense: dimension mismatch".into(),
        ));
    }

    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    let p = x.ncols();
    let xn = x.nrows();
    let yn = y.nrows();
    let x_data = x.as_slice();
    let y_data = y.as_mut_slice();
    y_data.fill(S::zero());

    for i in 0..m {
        let row_start = rp[i];
        let row_end = rp[i + 1];
        for pos in row_start..row_end {
            let j = cj[pos];
            let aij = vv[pos];
            let x_base = j;
            for col in 0..p {
                let y_idx = col * yn + i;
                let x_idx = col * xn + x_base;
                y_data[y_idx] += aij * x_data[x_idx];
            }
        }
    }

    Ok(())
}

#[cfg(not(feature = "rayon"))]
pub fn spmv_csr_parallel<S, A>(a: &A, x: &[S], y: &mut [S]) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    spmv_csr_serial(a, x, y)
}

/// Backend selection for transpose SpMV.
pub enum TBackend<'a, S: KrystScalar> {
    Csc(&'a dyn CscMatRef<S>),
    CsrGather,
}

#[cfg(feature = "rayon")]
fn t_spmv_csr_parallel_csc<S: KrystScalar>(
    csc: &dyn CscMatRef<S>,
    x: &[S],
    y: &mut [S],
) -> Result<(), KError> {
    if x.len() != csc.nrows() || y.len() != csc.ncols() {
        return Err(KError::InvalidInput("t_spmv: dimension mismatch".into()));
    }
    use rayon::prelude::*;
    let cp = csc.col_ptr();
    let ri = csc.row_idx();
    let vv = csc.values();
    y.par_iter_mut().enumerate().for_each(|(j, yj)| {
        let mut sum = <S as KrystScalar>::zero();
        for p in cp[j]..cp[j + 1] {
            let row = ri[p];
            let val = unsafe { *vv.get_unchecked(p) }.conj();
            let xr = unsafe { *x.get_unchecked(row) };
            sum = val.mul_add(xr, sum);
        }
        *yj = sum;
    });
    Ok(())
}

#[cfg(not(feature = "rayon"))]
fn t_spmv_csr_parallel_csc<S: KrystScalar>(
    csc: &dyn CscMatRef<S>,
    x: &[S],
    y: &mut [S],
) -> Result<(), KError> {
    if x.len() != csc.nrows() || y.len() != csc.ncols() {
        return Err(KError::InvalidInput("t_spmv: dimension mismatch".into()));
    }
    csc.t_matvec(x, y);
    Ok(())
}

#[cfg(feature = "rayon")]
fn t_spmv_csr_parallel_gather<S, A>(a: &A, x: &[S], y: &mut [S]) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    let (m, n) = (a.nrows(), a.ncols());
    if x.len() != m || y.len() != n {
        return Err(KError::InvalidInput("t_spmv: dimension mismatch".into()));
    }
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    use rayon::prelude::*;
    let out = (0..m)
        .into_par_iter()
        .fold(
            || vec![<S as KrystScalar>::zero(); n],
            |mut y_chunk, i| {
                let xi = x[i];
                if xi != <S as KrystScalar>::zero() {
                    let (rs, re) = (rp[i], rp[i + 1]);
                    for p in rs..re {
                        let j = unsafe { *cj.get_unchecked(p) };
                        let aij = unsafe { *vv.get_unchecked(p) }.conj();
                        unsafe {
                            let slot = y_chunk.get_unchecked_mut(j);
                            *slot = aij.mul_add(xi, *slot);
                        }
                    }
                }
                y_chunk
            },
        )
        .reduce(
            || vec![<S as KrystScalar>::zero(); n],
            |mut a, b| {
                for j in 0..n {
                    a[j] = a[j] + b[j];
                }
                a
            },
        );

    y.copy_from_slice(&out);
    Ok(())
}

#[cfg(not(feature = "rayon"))]
fn t_spmv_csr_parallel_gather<S, A>(a: &A, x: &[S], y: &mut [S]) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    if x.len() != a.nrows() || y.len() != a.ncols() {
        return Err(KError::InvalidInput("t_spmv: dimension mismatch".into()));
    }
    y.fill(<S as KrystScalar>::zero());
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    for i in 0..a.nrows() {
        let xi = x[i];
        if xi != <S as KrystScalar>::zero() {
            for p in rp[i]..rp[i + 1] {
                y[cj[p]] = y[cj[p]] + vv[p].conj() * xi;
            }
        }
    }
    Ok(())
}

/// y = A^T * x using CSR input and selectable backend.
pub fn t_spmv_csr_parallel<S, A>(
    a: &A,
    t_backend: TBackend<'_, S>,
    x: &[S],
    y: &mut [S],
) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    match t_backend {
        TBackend::Csc(csc) => t_spmv_csr_parallel_csc(csc, x, y),
        TBackend::CsrGather => t_spmv_csr_parallel_gather(a, x, y),
    }
}

/// Y(s) = A * X(s) with s right-hand sides stored column-major.
pub fn spmm_csr_block<S, A>(
    a: &A,
    s: usize,
    x_cols: &[&[S]],
    y_cols: &mut [&mut [S]],
) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    let (m, n) = (a.nrows(), a.ncols());
    if x_cols.len() != s || y_cols.len() != s {
        return Err(KError::InvalidInput("spmm: bad s".into()));
    }
    for r in 0..s {
        if x_cols[r].len() != n || y_cols[r].len() != m {
            return Err(KError::InvalidInput("spmm: dimension mismatch".into()));
        }
        y_cols[r].fill(<S as KrystScalar>::zero());
    }

    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    let mut acc = vec![<S as KrystScalar>::zero(); s];
    for i in 0..m {
        acc.fill(<S as KrystScalar>::zero());
        let (rs, re) = (rp[i], rp[i + 1]);
        for p in rs..re {
            let j = cj[p];
            let aij = vv[p];
            for r in 0..s {
                acc[r] = acc[r] + aij * x_cols[r][j];
            }
        }
        for r in 0..s {
            y_cols[r][i] = acc[r];
        }
    }
    Ok(())
}

/// Dense sparse matrix-matrix product specialized for small dense blocks.
///
/// Computes `Y = A * X` where `A` is CSR and `X`, `Y` are column-major dense
/// matrices provided as [`MatRef`] and [`MatMut`] respectively. The caller must
/// zero `Y` prior to invocation if accumulation is not desired.
#[cfg(feature = "backend-faer")]
pub fn csr_spmm_dense<S, A>(a: &A, x: MatRef<'_, S>, mut y: MatMut<'_, S>) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    let (m, n) = (a.nrows(), a.ncols());
    if x.nrows() != n {
        return Err(KError::InvalidInput(
            "csr_spmm_dense: column count mismatch".into(),
        ));
    }
    if y.nrows() != m {
        return Err(KError::InvalidInput(
            "csr_spmm_dense: row count mismatch".into(),
        ));
    }
    if x.ncols() != y.ncols() {
        return Err(KError::InvalidInput(
            "csr_spmm_dense: rhs count mismatch".into(),
        ));
    }

    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    let k = x.ncols();

    for i in 0..m {
        // zero the output row explicitly to avoid accumulating stale data.
        for col in 0..k {
            y[(i, col)] = S::zero();
        }
        for p in rp[i]..rp[i + 1] {
            let col = cj[p];
            let val = vv[p];
            for rhs in 0..k {
                y[(i, rhs)] = y[(i, rhs)] + val * x[(col, rhs)];
            }
        }
    }

    Ok(())
}

/// Canonical serial CSR matvec: `y = A * x`.
pub fn csr_matvec<S, A>(a: &A, x: &[S], y: &mut [S]) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    spmv_csr_serial(a, x, y)
}

/// Canonical parallel CSR matvec: `y = A * x`.
pub fn csr_matvec_par<S, A>(a: &A, x: &[S], y: &mut [S]) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    spmv_csr_parallel(a, x, y)
}

/// Serial transpose CSR matvec: `y = A^T * x` (`A^H * x` in complex builds).
pub fn csr_t_matvec<S, A>(a: &A, x: &[S], y: &mut [S]) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    if x.len() != a.nrows() || y.len() != a.ncols() {
        return Err(KError::InvalidInput(
            "csr_t_matvec: dimension mismatch".into(),
        ));
    }
    scalar::spmv_t_scaled_csr(
        a.nrows(),
        a.row_ptr(),
        a.col_idx(),
        a.values(),
        S::one(),
        x,
        S::zero(),
        y,
    );
    Ok(())
}

/// Canonical transpose CSR matvec: `y = A^T * x` (`A^H * x` in complex builds).
///
/// Uses CSC backend where available, otherwise CSR gather path.
pub fn csr_t_matvec_par<S, A>(
    a: &A,
    t_backend: TBackend<'_, S>,
    x: &[S],
    y: &mut [S],
) -> Result<(), KError>
where
    S: KrystScalar,
    A: CsrAccess<S>,
{
    t_spmv_csr_parallel(a, t_backend, x, y)
}

#[cfg(test)]
mod tests;
