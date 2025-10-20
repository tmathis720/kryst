//! Thread-parallel vector kernels with scalar fallback.
//!
//! - Works with or without `feature="rayon"`.
//! - Uses stable chunking (configurable) to keep reductions numerically steady.
//! - Provides scalar fallbacks for small problems or builds without Rayon.
//!
//! The kernels assume crate-level aliases/traits brought in via
//! [`crate::algebra::prelude`].

#![allow(clippy::needless_borrow)]

use crate::algebra::{parallel_cfg::parallel_tune, prelude::*};

#[cfg(feature = "rayon")]
use rayon::ThreadPoolBuilder;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

const VEC_CHUNK: usize = 1 << 14;
const REPRO_CHUNK: usize = 1 << 14;

/// Configure the Rayon thread pool. Safe to call multiple times; failures are ignored
/// when the global pool has already been built.
#[cfg(feature = "rayon")]
pub fn set_rayon_threads(n: usize) {
    let _ = ThreadPoolBuilder::new().num_threads(n).build_global();
}

// -------------------- scalar fallbacks --------------------

#[inline]
fn s_copy(src: &[S], dst: &mut [S]) {
    debug_assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

#[inline]
fn s_fill_zero(dst: &mut [S]) {
    for value in dst {
        *value = S::zero();
    }
}

#[inline]
fn s_scale(alpha: S, y: &mut [S]) {
    if alpha == S::from_real(1.0) {
        return;
    }
    if alpha == S::zero() {
        s_fill_zero(y);
        return;
    }
    for yi in y {
        *yi = alpha * *yi;
    }
}

#[inline]
fn s_axpy(x: &[S], alpha: S, y: &mut [S]) {
    debug_assert_eq!(x.len(), y.len());
    if alpha == S::zero() {
        return;
    }
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = *yi + alpha * xi;
    }
}

#[inline]
fn s_axpby(x: &[S], alpha: S, y: &mut [S], beta: S) {
    debug_assert_eq!(x.len(), y.len());
    if beta == S::zero() {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = alpha * xi;
        }
    } else if beta == S::from_real(1.0) {
        s_axpy(x, alpha, y);
    } else {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = alpha * xi + beta * *yi;
        }
    }
}

#[inline]
fn s_dot_conj_local(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = S::zero();
    const BLK: usize = 1 << 14;
    let mut i = 0;
    while i < x.len() {
        let end = (i + BLK).min(x.len());
        let mut blk = S::zero();
        for j in i..end {
            blk = blk + x[j].conj() * y[j];
        }
        acc = acc + blk;
        i = end;
    }
    acc
}

#[inline]
fn s_sum_abs2_local(x: &[S]) -> R {
    let mut acc = R::default();
    const BLK: usize = 1 << 14;
    let mut i = 0;
    while i < x.len() {
        let end = (i + BLK).min(x.len());
        let mut blk = R::default();
        for j in i..end {
            let a = x[j].abs();
            blk = blk + a * a;
        }
        acc = acc + blk;
        i = end;
    }
    acc
}

// -------------------- public API (dual-path) --------------------

#[inline]
pub fn par_copy(src: &[S], dst: &mut [S]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(feature = "rayon")]
    {
        let n = src.len();
        let min_len = parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            src.par_chunks(chunk)
                .zip(dst.par_chunks_mut(chunk))
                .for_each(|(s, d)| d.copy_from_slice(s));
            return;
        }
    }
    s_copy(src, dst);
}

#[inline]
pub fn par_fill_zero(dst: &mut [S]) {
    #[cfg(feature = "rayon")]
    {
        let n = dst.len();
        let min_len = parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            dst.par_chunks_mut(chunk)
                .for_each(|chunk| s_fill_zero(chunk));
            return;
        }
    }
    s_fill_zero(dst);
}

#[inline]
pub fn par_scale(alpha: S, y: &mut [S]) {
    #[cfg(feature = "rayon")]
    {
        let n = y.len();
        let min_len = parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            if alpha == S::from_real(1.0) {
                return;
            }
            if alpha == S::zero() {
                par_fill_zero(y);
                return;
            }
            y.par_chunks_mut(chunk).for_each(|yc| {
                for yi in yc {
                    *yi = alpha * *yi;
                }
            });
            return;
        }
    }
    s_scale(alpha, y);
}

#[inline]
pub fn par_axpy(x: &[S], alpha: S, y: &mut [S]) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = parallel_tune().min_len_vec;
        if n >= min_len {
            if alpha == S::zero() {
                return;
            }
            y.par_iter_mut()
                .zip(x.par_iter().copied())
                .for_each(|(yi, xi)| {
                    *yi = *yi + alpha * xi;
                });
            return;
        }
    }
    s_axpy(x, alpha, y);
}

#[inline]
pub fn par_axpby(x: &[S], alpha: S, y: &mut [S], beta: S) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = parallel_tune().min_len_vec;
        if n >= min_len {
            if beta == S::zero() {
                y.par_iter_mut()
                    .zip(x.par_iter().copied())
                    .for_each(|(yi, xi)| {
                        *yi = alpha * xi;
                    });
            } else if beta == S::from_real(1.0) {
                par_axpy(x, alpha, y);
            } else {
                y.par_iter_mut()
                    .zip(x.par_iter().copied())
                    .for_each(|(yi, xi)| {
                        *yi = alpha * xi + beta * *yi;
                    });
            }
            return;
        }
    }
    s_axpby(x, alpha, y, beta);
}

/// Compute `y = x + alpha * y`.
#[inline]
pub fn par_xpay(x: &[S], alpha: S, y: &mut [S]) {
    par_axpby(x, S::one(), y, alpha);
}

#[inline]
pub fn par_dot_conj_local(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            return x
                .par_chunks(chunk)
                .zip(y.par_chunks(chunk))
                .map(|(xc, yc)| {
                    let mut acc = S::zero();
                    for (&xi, &yi) in xc.iter().zip(yc) {
                        acc = acc + xi.conj() * yi;
                    }
                    acc
                })
                .reduce(S::zero, |a, b| a + b);
        }
    }
    s_dot_conj_local(x, y)
}

#[inline]
pub fn par_sum_abs2_local(x: &[S]) -> R {
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            return x
                .par_chunks(chunk)
                .map(|xc| {
                    let mut ssq = R::default();
                    for &value in xc {
                        let a = value.abs();
                        ssq = ssq + a * a;
                    }
                    ssq
                })
                .reduce(R::default, |a, b| a + b);
        }
    }
    s_sum_abs2_local(x)
}

/// Deterministic conjugated dot product using fixed chunking.
pub fn dot_conj_local_repro(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return S::zero();
    }

    let nchunks = (x.len() + REPRO_CHUNK - 1) / REPRO_CHUNK;
    let mut parts = vec![S::zero(); nchunks];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        parts.par_iter_mut().enumerate().for_each(|(cid, slot)| {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = S::zero();
            for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                acc = acc + xi.conj() * yi;
            }
            *slot = acc;
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for cid in 0..nchunks {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = S::zero();
            for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                acc = acc + xi.conj() * yi;
            }
            parts[cid] = acc;
        }
    }

    let mut total = S::zero();
    for part in parts {
        total = total + part;
    }
    total
}

/// Deterministic sum of squared magnitudes using fixed chunking.
pub fn sum_abs2_local_repro(x: &[S]) -> R {
    if x.is_empty() {
        return R::zero();
    }

    let nchunks = (x.len() + REPRO_CHUNK - 1) / REPRO_CHUNK;
    let mut parts = vec![R::zero(); nchunks];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        parts.par_iter_mut().enumerate().for_each(|(cid, slot)| {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = R::zero();
            for &value in &x[start..end] {
                let a = value.abs();
                acc = acc + a * a;
            }
            *slot = acc;
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for cid in 0..nchunks {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = R::zero();
            for &value in &x[start..end] {
                let a = value.abs();
                acc = acc + a * a;
            }
            parts[cid] = acc;
        }
    }

    let mut total = R::zero();
    for part in parts {
        total = total + part;
    }
    total
}
