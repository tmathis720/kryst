//! SIMD-accelerated CSR sparse matrix-vector kernels built on the `wide` crate.
//!
//! The routines in this module provide portable gather-based SpMV
//! implementations that are selected at runtime by the plan builder. While the
//! API mirrors what a `std::simd` implementation would offer, the use of
//! `wide` keeps the kernels available on stable Rust. These kernels are
//! specialized for `f64` values and are only compiled in real builds.

use super::scalar;
use wide::{f64x2, f64x4};

#[inline]
fn load2(vals: &[f64]) -> f64x2 {
    f64x2::new([vals[0], vals[1]])
}

#[inline]
fn load4(vals: &[f64]) -> f64x4 {
    f64x4::new([vals[0], vals[1], vals[2], vals[3]])
}

#[inline]
unsafe fn gather2(x: &[f64], idx: &[usize]) -> f64x2 {
    let x0 = unsafe { *x.get_unchecked(idx[0]) };
    let x1 = unsafe { *x.get_unchecked(idx[1]) };
    f64x2::new([x0, x1])
}

#[inline]
unsafe fn gather4(x: &[f64], idx: &[usize]) -> f64x4 {
    let x0 = unsafe { *x.get_unchecked(idx[0]) };
    let x1 = unsafe { *x.get_unchecked(idx[1]) };
    let x2 = unsafe { *x.get_unchecked(idx[2]) };
    let x3 = unsafe { *x.get_unchecked(idx[3]) };
    f64x4::new([x0, x1, x2, x3])
}

fn spmv_scaled_csr_simd_gather_2(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    assert_eq!(row_ptr.len(), m + 1);
    assert_eq!(col_idx.len(), vals.len());
    if let Some(max_col) = col_idx.iter().copied().max() {
        assert!(x.len() > max_col);
    }
    assert!(y.len() >= m);

    if beta == 0.0 {
        y[..m].fill(0.0);
    } else if beta != 1.0 {
        for yi in &mut y[..m] {
            *yi *= beta;
        }
    }

    for i in 0..m {
        let mut sum = 0.0;
        let mut p = row_ptr[i];
        let end = row_ptr[i + 1];

        while p + 1 < end {
            let a_vec = load2(&vals[p..p + 2]);
            let idx = [col_idx[p], col_idx[p + 1]];
            let x_vec = unsafe { gather2(x, &idx) };
            let prod = (a_vec * x_vec).to_array();
            sum += prod[0] + prod[1];
            p += 2;
        }

        while p < end {
            sum += vals[p] * x[col_idx[p]];
            p += 1;
        }

        y[i] += alpha * sum;
    }
}

fn spmv_scaled_csr_simd_gather_4(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    assert_eq!(row_ptr.len(), m + 1);
    assert_eq!(col_idx.len(), vals.len());
    if let Some(max_col) = col_idx.iter().copied().max() {
        assert!(x.len() > max_col);
    }
    assert!(y.len() >= m);

    if beta == 0.0 {
        y[..m].fill(0.0);
    } else if beta != 1.0 {
        for yi in &mut y[..m] {
            *yi *= beta;
        }
    }

    for i in 0..m {
        let mut sum = 0.0;
        let mut p = row_ptr[i];
        let end = row_ptr[i + 1];

        while p + 3 < end {
            let a_vec = load4(&vals[p..p + 4]);
            let idx = [col_idx[p], col_idx[p + 1], col_idx[p + 2], col_idx[p + 3]];
            let x_vec = unsafe { gather4(x, &idx) };
            let prod = (a_vec * x_vec).to_array();
            sum += prod[0] + prod[1] + prod[2] + prod[3];
            p += 4;
        }

        while p < end {
            sum += vals[p] * x[col_idx[p]];
            p += 1;
        }

        y[i] += alpha * sum;
    }
}

fn spmv_t_scaled_csr_simd_gather_2(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    assert_eq!(row_ptr.len(), m + 1);
    assert_eq!(col_idx.len(), vals.len());

    if beta == 0.0 {
        y.fill(0.0);
    } else if beta != 1.0 {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }

    for i in 0..m {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let mut p = row_ptr[i];
        let end = row_ptr[i + 1];
        while p + 1 < end {
            let cols = [col_idx[p], col_idx[p + 1]];
            let vals_vec = load2(&vals[p..p + 2]);
            let contrib = vals_vec * f64x2::splat(alpha * xi);
            let arr = contrib.to_array();
            for lane in 0..2 {
                let col = cols[lane];
                y[col] += arr[lane];
            }
            p += 2;
        }
        while p < end {
            let col = col_idx[p];
            y[col] += alpha * vals[p] * xi;
            p += 1;
        }
    }
}

fn spmv_t_scaled_csr_simd_gather_4(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    assert_eq!(row_ptr.len(), m + 1);
    assert_eq!(col_idx.len(), vals.len());

    if beta == 0.0 {
        y.fill(0.0);
    } else if beta != 1.0 {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }

    for i in 0..m {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let mut p = row_ptr[i];
        let end = row_ptr[i + 1];
        while p + 3 < end {
            let cols = [col_idx[p], col_idx[p + 1], col_idx[p + 2], col_idx[p + 3]];
            let vals_vec = load4(&vals[p..p + 4]);
            let contrib = vals_vec * f64x4::splat(alpha * xi);
            let arr = contrib.to_array();
            for lane in 0..4 {
                let col = cols[lane];
                y[col] += arr[lane];
            }
            p += 4;
        }
        while p < end {
            let col = col_idx[p];
            y[col] += alpha * vals[p] * xi;
            p += 1;
        }
    }
}

/// Computes `y = alpha * A * x + beta * y` for a CSR matrix using SIMD gathers.
pub fn spmv_scaled_csr_simd_gather(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
    lanes: usize,
) {
    match lanes {
        4 => spmv_scaled_csr_simd_gather_4(m, row_ptr, col_idx, vals, alpha, x, beta, y),
        _ => spmv_scaled_csr_simd_gather_2(m, row_ptr, col_idx, vals, alpha, x, beta, y),
    }
}

/// Computes `y = alpha * A^T * x + beta * y` using the gather path.
pub fn spmv_t_scaled_csr_simd_gather(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
    lanes: usize,
) {
    match lanes {
        4 => spmv_t_scaled_csr_simd_gather_4(m, row_ptr, col_idx, vals, alpha, x, beta, y),
        _ => spmv_t_scaled_csr_simd_gather_2(m, row_ptr, col_idx, vals, alpha, x, beta, y),
    }
}

/// Selects an appropriate SIMD lane count for the current target at runtime.
#[inline]
pub fn detect_simd_lanes() -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return 4;
        }
    }
    2
}

/// Dispatch helper that selects the specialised gather kernel for `lanes`.
#[inline]
pub fn dispatch_spmv_scaled_csr_simd_gather(
    lanes: usize,
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    spmv_scaled_csr_simd_gather(m, row_ptr, col_idx, vals, alpha, x, beta, y, lanes);
}

/// Dispatch helper for transpose SIMD gather kernels.
#[inline]
pub fn dispatch_spmv_t_scaled_csr_simd_gather(
    lanes: usize,
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    spmv_t_scaled_csr_simd_gather(m, row_ptr, col_idx, vals, alpha, x, beta, y, lanes);
}

/// Fallback helper to execute the scalar kernel when the SIMD path is not
/// viable (e.g., lane detection picks 1).
#[inline]
pub fn fallback_scalar(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    scalar::spmv_scaled_csr(m, row_ptr, col_idx, vals, alpha, x, beta, y);
}
