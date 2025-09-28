//! Core dense kernels used by block Krylov solvers.

use crate::context::ksp_context::BlockVec;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::spmv::spmm_csr_dense as global_spmm;

/// Perform `y[:,c] += a * x[:,c]` for each column `c`.
#[inline]
pub fn block_axpy(a: f64, x: &BlockVec, y: &mut BlockVec) {
    debug_assert_eq!(x.nrows(), y.nrows());
    debug_assert_eq!(x.ncols(), y.ncols());
    if a == 0.0 {
        return;
    }
    let p = x.ncols();
    for col in 0..p {
        let xcol = x.col(col);
        let ycol = y.col_mut(col);
        for (yi, &xi) in ycol.iter_mut().zip(xcol.iter()) {
            *yi += a * xi;
        }
    }
}

/// Project a block vector against a tall basis `V` using the coefficient matrix `C`.
#[inline]
pub fn block_project(v: &[&[f64]], c_row_major: &[f64], k: usize, p: usize, y: &mut BlockVec) {
    debug_assert_eq!(c_row_major.len(), k * p);
    debug_assert_eq!(y.ncols(), p);
    if k == 0 || p == 0 {
        return;
    }
    let n = y.nrows();
    for col in 0..p {
        let ycol = y.col_mut(col);
        for row in 0..k {
            let coeff = c_row_major[row * p + col];
            if coeff == 0.0 {
                continue;
            }
            let vrow = v[row];
            debug_assert_eq!(vrow.len(), n);
            for (yi, &vi) in ycol.iter_mut().zip(vrow.iter()) {
                *yi -= coeff * vi;
            }
        }
    }
}

/// Compute the Gram matrix `G = X^T Y` (row-major output).
#[inline]
pub fn gram_pxp(x: &BlockVec, y: &BlockVec, out: &mut [f64]) {
    debug_assert_eq!(x.ncols(), y.ncols());
    let p = x.ncols();
    let n = x.nrows();
    debug_assert_eq!(out.len(), p * p);
    for col_y in 0..p {
        let ycol = y.col(col_y);
        for col_x in 0..p {
            let xcol = x.col(col_x);
            let mut acc = 0.0;
            for i in 0..n {
                acc += xcol[i] * ycol[i];
            }
            out[col_x * p + col_y] = acc;
        }
    }
}

/// Compute `C = V^T * W` where `V` stores `k` tall vectors and `W` is an `n×p` block.
#[inline]
pub fn tall_t_times_block(v: &[&[f64]], w: &BlockVec, out: &mut [f64]) {
    let k = v.len();
    let p = w.ncols();
    debug_assert_eq!(out.len(), k * p);
    let n = w.nrows();
    for row in 0..k {
        let vrow = v[row];
        debug_assert_eq!(vrow.len(), n);
        for col in 0..p {
            let wcol = w.col(col);
            let mut acc = 0.0;
            for i in 0..n {
                acc += vrow[i] * wcol[i];
            }
            out[row * p + col] = acc;
        }
    }
}

/// Sparse matrix / dense block multiplication wrapper.
#[inline]
pub fn spmm_csr_dense(a: &CsrMatrix<f64>, x: &BlockVec, y: &mut BlockVec) -> Result<(), KError> {
    debug_assert_eq!(x.ncols(), y.ncols());
    debug_assert_eq!(x.nrows(), a.ncols());
    debug_assert_eq!(y.nrows(), a.nrows());
    global_spmm(a, x, y)
}
