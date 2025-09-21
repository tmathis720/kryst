//! Scalar sparse matrix-vector multiplication kernels.

/// Computes `y = alpha * A * x + beta * y` for a matrix stored in CSR format.
///
/// This is the baseline SpMV kernel and remains the reference implementation
/// for the higher-level SIMD paths that will be added in later iterations. It
/// keeps the loop order deterministic and performs a small unrolled
/// accumulation to improve ILP on modern CPUs without depending on explicit
/// SIMD instructions.
pub fn spmv_scaled_csr(
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
        let mut acc = 0.0f64;
        let mut p = row_ptr[i];
        let end = row_ptr[i + 1];

        while p + 3 < end {
            acc += vals[p] * x[col_idx[p]]
                + vals[p + 1] * x[col_idx[p + 1]]
                + vals[p + 2] * x[col_idx[p + 2]]
                + vals[p + 3] * x[col_idx[p + 3]];
            p += 4;
        }

        while p < end {
            acc += vals[p] * x[col_idx[p]];
            p += 1;
        }

        y[i] += alpha * acc;
    }
}

/// Computes `y = alpha * A^T * x + beta * y` for a CSR matrix.
pub fn spmv_t_scaled_csr(
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
    assert!(col_idx.len() == vals.len());

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
        let rs = row_ptr[i];
        let re = row_ptr[i + 1];
        for p in rs..re {
            y[col_idx[p]] += alpha * vals[p] * xi;
        }
    }
}
