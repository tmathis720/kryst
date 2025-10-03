//! Scalar sparse matrix-vector multiplication kernels.

#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::matrix::csr::CsrMatrix;

/// Portable CSR SpMV: computes `y = A * x` without conjugation.
///
/// This helper keeps the control flow simple to act as the scalar fallback for
/// both real and complex builds.  The loops intentionally mirror the
/// conventional row-wise CSR traversal so it is easy to audit and compare
/// against the existing `spmv_scaled_csr` entry point which supports
/// alpha/beta scaling factors.
#[inline]
pub fn spmv_csr_scalar<S: KrystScalar>(a: &CsrMatrix<S>, x: &[S], y: &mut [S]) {
    debug_assert_eq!(x.len(), a.ncols);
    debug_assert_eq!(y.len(), a.nrows);

    for i in 0..a.nrows {
        let start = a.rowptr[i];
        let end = a.rowptr[i + 1];
        let mut acc = S::zero();
        for k in start..end {
            let j = a.colind[k];
            acc = acc + a.values[k] * x[j];
        }
        y[i] = acc;
    }
}

/// Computes `y = alpha * A * x + beta * y` for a matrix stored in CSR format.
///
/// This is the baseline SpMV kernel and remains the reference implementation
/// for the higher-level SIMD paths that will be added in later iterations. It
/// keeps the loop order deterministic and performs a small unrolled
/// accumulation to improve ILP on modern CPUs without depending on explicit
/// SIMD instructions.
pub fn spmv_scaled_csr<S: KrystScalar>(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[S],
    alpha: S,
    x: &[S],
    beta: S,
    y: &mut [S],
) {
    assert_eq!(row_ptr.len(), m + 1);
    assert_eq!(col_idx.len(), vals.len());
    if let Some(max_col) = col_idx.iter().copied().max() {
        assert!(x.len() > max_col);
    }
    assert!(y.len() >= m);

    if beta == S::zero() {
        y[..m].fill(S::zero());
    } else if beta != S::one() {
        for yi in &mut y[..m] {
            *yi = (*yi) * beta;
        }
    }

    for i in 0..m {
        let mut acc = S::zero();
        let mut p = row_ptr[i];
        let end = row_ptr[i + 1];

        while p + 3 < end {
            acc = vals[p].mul_add(x[col_idx[p]], acc);
            acc = vals[p + 1].mul_add(x[col_idx[p + 1]], acc);
            acc = vals[p + 2].mul_add(x[col_idx[p + 2]], acc);
            acc = vals[p + 3].mul_add(x[col_idx[p + 3]], acc);
            p += 4;
        }

        while p < end {
            acc = vals[p].mul_add(x[col_idx[p]], acc);
            p += 1;
        }

        y[i] = y[i] + alpha * acc;
    }
}

/// Computes `y = alpha * A^T * x + beta * y` for a CSR matrix.
pub fn spmv_t_scaled_csr<S: KrystScalar>(
    m: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[S],
    alpha: S,
    x: &[S],
    beta: S,
    y: &mut [S],
) {
    assert_eq!(row_ptr.len(), m + 1);
    assert!(col_idx.len() == vals.len());

    if beta == S::zero() {
        y.fill(S::zero());
    } else if beta != S::one() {
        for yi in y.iter_mut() {
            *yi = (*yi) * beta;
        }
    }

    for i in 0..m {
        let xi = x[i];
        if xi == S::zero() {
            continue;
        }
        let rs = row_ptr[i];
        let re = row_ptr[i + 1];
        for p in rs..re {
            let idx = col_idx[p];
            y[idx] = y[idx] + alpha * vals[p] * xi;
        }
    }
}
