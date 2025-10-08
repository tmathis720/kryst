//! Core dense kernels used by block Krylov solvers.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::BlockVec;
use crate::error::KError;
use crate::matrix::spmv::spmm_csr_dense as global_spmm;

/// Perform `y[:,c] += a * x[:,c]` for each column `c`.
#[inline]
pub fn block_axpy(a: S, x: &BlockVec, y: &mut BlockVec) {
    debug_assert_eq!(x.nrows(), y.nrows());
    debug_assert_eq!(x.ncols(), y.ncols());
    if a.abs() <= R::default() {
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
pub fn block_project(v: &[&[S]], c_row_major: &[S], k: usize, p: usize, y: &mut BlockVec) {
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
            if coeff.abs() <= R::default() {
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
pub fn gram_pxp(x: &BlockVec, y: &BlockVec, out: &mut [S]) {
    debug_assert_eq!(x.ncols(), y.ncols());
    let p = x.ncols();
    debug_assert_eq!(out.len(), p * p);
    for col_y in 0..p {
        let ycol = y.col(col_y);
        debug_assert_eq!(ycol.len(), x.nrows());
        for col_x in 0..p {
            let xcol = x.col(col_x);
            debug_assert_eq!(xcol.len(), x.nrows());
            out[col_x * p + col_y] = dot_conj(xcol, ycol);
        }
    }
}

/// Compute `C = V^T * W` where `V` stores `k` tall vectors and `W` is an `n×p` block.
#[inline]
pub fn tall_t_times_block(v: &[&[S]], w: &BlockVec, out: &mut [S]) {
    let k = v.len();
    let p = w.ncols();
    debug_assert_eq!(out.len(), k * p);
    for row in 0..k {
        let vrow = v[row];
        debug_assert_eq!(vrow.len(), w.nrows());
        for col in 0..p {
            let wcol = w.col(col);
            out[row * p + col] = dot_conj(vrow, wcol);
        }
    }
}

/// Sparse matrix / dense block multiplication wrapper.
#[inline]
pub fn spmm_csr_dense<A>(a: &A, x: &BlockVec, y: &mut BlockVec) -> Result<(), KError>
where
    A: crate::matrix::spmv::CsrAccess<S>,
{
    debug_assert_eq!(x.ncols(), y.ncols());
    debug_assert_eq!(x.nrows(), a.ncols());
    debug_assert_eq!(y.nrows(), a.nrows());
    global_spmm(a, x, y)
}
