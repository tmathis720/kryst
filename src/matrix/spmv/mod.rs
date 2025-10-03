pub mod plan;
pub mod scalar;
#[cfg(feature = "simd")]
pub mod sellc;
#[cfg(feature = "simd")]
pub mod simd_csr;

pub use self::plan::{
    SpmvKernel, SpmvPlan, SpmvTuning, build as build_plan, build_owned as build_plan_owned,
};
pub use self::scalar::{spmv_csr_scalar, spmv_scaled_csr, spmv_t_scaled_csr};

use crate::context::ksp_context::BlockVec;
use crate::error::KError;
use crate::matrix::{csc::CscMatrix, sparse::CsrMatrix};
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
#[cfg(feature = "rayon")]
pub fn spmv_csr_parallel(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    if x.len() != a.ncols() || y.len() != a.nrows() {
        return Err(KError::InvalidInput(
            "spmv_csr_parallel: dimension mismatch".into(),
        ));
    }
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    use rayon::prelude::*;
    y.par_chunks_mut(1).enumerate().for_each(|(i, yi)| {
        let (rs, re) = (rp[i], rp[i + 1]);
        let sum = unsafe { lane_dot_gather_unchecked(&vv[rs..re], &cj[rs..re], x) };
        yi[0] = sum;
    });
    Ok(())
}

pub fn spmm_csr_dense(a: &CsrMatrix<f64>, x: &BlockVec, y: &mut BlockVec) -> Result<(), KError> {
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
    y_data.fill(0.0);

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
pub fn spmv_csr_parallel(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    a.spmv_scaled(1.0, x, 0.0, y)
}

#[cfg(feature = "rayon")]
#[inline]
unsafe fn fallback_lane_dot(vals: &[f64], cols: &[usize], x: &[f64]) -> f64 {
    debug_assert_eq!(vals.len(), cols.len());
    let mut acc = 0.0;
    for k in 0..vals.len() {
        let v = unsafe { *vals.get_unchecked(k) };
        let col = unsafe { *cols.get_unchecked(k) };
        let xv = unsafe { *x.get_unchecked(col) };
        acc += v * xv;
    }
    acc
}

#[cfg(feature = "rayon")]
#[cfg(not(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
)))]
#[inline]
unsafe fn lane_dot_gather_unchecked(vals: &[f64], cols: &[usize], x: &[f64]) -> f64 {
    unsafe { fallback_lane_dot(vals, cols, x) }
}

#[cfg(feature = "rayon")]
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
#[inline]
unsafe fn lane_dot_gather_unchecked(vals: &[f64], cols: &[usize], x: &[f64]) -> f64 {
    // placeholder for AVX2 gather implementation
    unsafe { fallback_lane_dot(vals, cols, x) }
}

/// Backend selection for transpose SpMV.
pub enum TBackend<'a> {
    Csc(&'a CscMatrix<f64>),
    CsrGather,
}

#[cfg(feature = "rayon")]
fn t_spmv_csr_parallel_csc(csc: &CscMatrix<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    if x.len() != csc.nrows() || y.len() != csc.ncols() {
        return Err(KError::InvalidInput("t_spmv: dimension mismatch".into()));
    }
    use rayon::prelude::*;
    let cp = csc.col_ptr();
    let ri = csc.row_idx();
    let vv = csc.values();
    y.par_iter_mut().enumerate().for_each(|(j, yj)| {
        let mut sum = 0.0;
        for p in cp[j]..cp[j + 1] {
            let row = ri[p];
            sum += unsafe { *vv.get_unchecked(p) * *x.get_unchecked(row) };
        }
        *yj = sum;
    });
    Ok(())
}

#[cfg(not(feature = "rayon"))]
fn t_spmv_csr_parallel_csc(csc: &CscMatrix<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    if x.len() != csc.nrows() || y.len() != csc.ncols() {
        return Err(KError::InvalidInput("t_spmv: dimension mismatch".into()));
    }
    csc.t_matvec(x, y);
    Ok(())
}

#[cfg(feature = "rayon")]
fn t_spmv_csr_parallel_gather(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
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
            || vec![0.0; n],
            |mut y_chunk, i| {
                let xi = x[i];
                if xi != 0.0 {
                    let (rs, re) = (rp[i], rp[i + 1]);
                    for p in rs..re {
                        let j = unsafe { *cj.get_unchecked(p) };
                        let aij = unsafe { *vv.get_unchecked(p) };
                        unsafe { *y_chunk.get_unchecked_mut(j) += aij * xi };
                    }
                }
                y_chunk
            },
        )
        .reduce(
            || vec![0.0; n],
            |mut a, b| {
                for j in 0..n {
                    a[j] += b[j];
                }
                a
            },
        );

    y.copy_from_slice(&out);
    Ok(())
}

#[cfg(not(feature = "rayon"))]
fn t_spmv_csr_parallel_gather(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    if x.len() != a.nrows() || y.len() != a.ncols() {
        return Err(KError::InvalidInput("t_spmv: dimension mismatch".into()));
    }
    y.fill(0.0);
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    for i in 0..a.nrows() {
        let xi = x[i];
        if xi != 0.0 {
            for p in rp[i]..rp[i + 1] {
                y[cj[p]] += vv[p] * xi;
            }
        }
    }
    Ok(())
}

/// y = A^T * x using CSR input and selectable backend.
pub fn t_spmv_csr_parallel(
    a: &CsrMatrix<f64>,
    t_backend: TBackend<'_>,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), KError> {
    match t_backend {
        TBackend::Csc(csc) => t_spmv_csr_parallel_csc(csc, x, y),
        TBackend::CsrGather => t_spmv_csr_parallel_gather(a, x, y),
    }
}

/// Y(s) = A * X(s) with s right-hand sides stored column-major.
pub fn spmm_csr_block(
    a: &CsrMatrix<f64>,
    s: usize,
    x_cols: &[&[f64]],
    y_cols: &mut [&mut [f64]],
) -> Result<(), KError> {
    let (m, n) = (a.nrows(), a.ncols());
    if x_cols.len() != s || y_cols.len() != s {
        return Err(KError::InvalidInput("spmm: bad s".into()));
    }
    for r in 0..s {
        if x_cols[r].len() != n || y_cols[r].len() != m {
            return Err(KError::InvalidInput("spmm: dimension mismatch".into()));
        }
        y_cols[r].fill(0.0);
    }

    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    let mut acc = vec![0.0f64; s];
    for i in 0..m {
        acc.fill(0.0);
        let (rs, re) = (rp[i], rp[i + 1]);
        for p in rs..re {
            let j = cj[p];
            let aij = vv[p];
            for r in 0..s {
                acc[r] += aij * x_cols[r][j];
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
pub fn csr_spmm_dense(
    a: &CsrMatrix<f64>,
    x: MatRef<'_, f64>,
    mut y: MatMut<'_, f64>,
) -> Result<(), KError> {
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
            y[(i, col)] = 0.0;
        }
        for p in rp[i]..rp[i + 1] {
            let col = cj[p];
            let val = vv[p];
            for rhs in 0..k {
                y[(i, rhs)] += val * x[(col, rhs)];
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
