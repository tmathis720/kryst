use crate::error::KError;
use crate::matrix::{csc::CscMatrix, sparse::CsrMatrix};

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

#[cfg(not(feature = "rayon"))]
pub fn spmv_csr_parallel(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    a.spmv_scaled(1.0, x, 0.0, y)
}

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

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2")))]
#[inline]
unsafe fn lane_dot_gather_unchecked(vals: &[f64], cols: &[usize], x: &[f64]) -> f64 {
    unsafe { fallback_lane_dot(vals, cols, x) }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
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
fn t_spmv_csr_parallel_csc(
    csc: &CscMatrix<f64>,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), KError> {
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
fn t_spmv_csr_parallel_csc(
    csc: &CscMatrix<f64>,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), KError> {
    if x.len() != csc.nrows() || y.len() != csc.ncols() {
        return Err(KError::InvalidInput("t_spmv: dimension mismatch".into()));
    }
    csc.t_matvec(x, y);
    Ok(())
}

#[cfg(feature = "rayon")]
fn t_spmv_csr_parallel_gather(
    a: &CsrMatrix<f64>,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), KError> {
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
fn t_spmv_csr_parallel_gather(
    a: &CsrMatrix<f64>,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), KError> {
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

    for i in 0..m {
        let (rs, re) = (rp[i], rp[i + 1]);
        let mut acc = vec![0.0f64; s];
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
