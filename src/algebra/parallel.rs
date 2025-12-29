//! Thread-parallel vector kernels with scalar fallback.
//!
//! - Works with or without `feature="rayon"`.
//! - Uses stable chunking (configurable) to keep reductions numerically steady.
//! - Provides scalar fallbacks for small problems or builds without Rayon.
//!
//! The kernels assume crate-level aliases/traits brought in via
//! [`crate::algebra::prelude`].

#![allow(clippy::needless_borrow)]

use crate::algebra::prelude::*;
use crate::reduction::{Kahan, ReproMode, Accum};
use crate::utils::reduction::repro_mode_is_strict;

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use std::sync::atomic::{AtomicUsize, Ordering};

const VEC_CHUNK: usize = 1 << 14;
const REPRO_CHUNK: usize = 1 << 14;

#[cfg(feature = "rayon")]
static GLOBAL_RAYON_CONFIG_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Configure the global Rayon thread pool used by Kryst's parallel kernels.
///
/// This is a thin wrapper around [`rayon::ThreadPoolBuilder::build_global`]; it is
/// safe to call multiple times, but only the first successful invocation takes
/// effect. Subsequent calls are ignored once the global pool has been initialised.
#[cfg(feature = "rayon")]
pub fn set_rayon_threads(n: usize) {
    GLOBAL_RAYON_CONFIG_CALLS.fetch_add(1, Ordering::Relaxed);
    let _ = crate::parallel::threads::init_global_rayon_pool_with_threads(n);
}

#[cfg(all(feature = "rayon", test))]
pub(crate) fn global_rayon_config_calls() -> usize {
    GLOBAL_RAYON_CONFIG_CALLS.load(Ordering::Relaxed)
}

#[cfg(all(feature = "rayon", test))]
pub(crate) fn reset_global_rayon_config_calls() {
    GLOBAL_RAYON_CONFIG_CALLS.store(0, Ordering::Relaxed);
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
            blk = blk + x[j].abs2();
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
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
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
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
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
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
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
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
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
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
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

/// Apply `f(i)` for each index `i` in `0..len`, using Rayon when enabled and the
/// problem size exceeds the configured threshold.
#[inline]
pub fn par_for_each_index<F>(len: usize, f: F)
where
    F: Fn(usize) + Sync + Send,
{
    #[cfg(feature = "rayon")]
    {
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        if len >= min_len && !crate::algebra::parallel_cfg::force_serial() {
            (0..len).into_par_iter().for_each(|i| f(i));
            return;
        }
    }

    for i in 0..len {
        f(i);
    }
}

#[inline]
fn dot_conj_local_fast(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
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
fn sum_abs2_local_fast(x: &[S]) -> R {
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
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

#[inline]
pub fn par_dot_conj_local(x: &[S], y: &[S]) -> S {
    if repro_mode_is_strict() {
        return dot_conj_local_repro(x, y);
    }
    dot_conj_local_fast(x, y)
}

#[inline]
pub fn par_sum_abs2_local(x: &[S]) -> R {
    if repro_mode_is_strict() {
        return sum_abs2_local_repro(x);
    }
    sum_abs2_local_fast(x)
}

#[inline]
pub fn dot_conj_local_with_mode(x: &[S], y: &[S], mode: ReproMode) -> S {
    match mode {
        ReproMode::Fast => dot_conj_local_fast(x, y),
        ReproMode::Deterministic => dot_conj_local_repro(x, y),
        ReproMode::DeterministicAccurate => dot_conj_local_repro_accurate(x, y),
    }
}

#[inline]
pub fn sum_abs2_local_with_mode(x: &[S], mode: ReproMode) -> R {
    match mode {
        ReproMode::Fast => sum_abs2_local_fast(x),
        ReproMode::Deterministic => sum_abs2_local_repro(x),
        ReproMode::DeterministicAccurate => sum_abs2_local_repro_accurate(x),
    }
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
        if !crate::algebra::parallel_cfg::force_serial() {
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
        } else {
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

/// Deterministic conjugated dot product with compensated chunk sums.
pub fn dot_conj_local_repro_accurate(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return S::zero();
    }

    let nchunks = (x.len() + REPRO_CHUNK - 1) / REPRO_CHUNK;
    let mut parts = vec![S::zero(); nchunks];

    #[cfg(feature = "rayon")]
    {
        if !crate::algebra::parallel_cfg::force_serial() {
            use rayon::prelude::*;
            parts.par_iter_mut().enumerate().for_each(|(cid, slot)| {
                let start = cid * REPRO_CHUNK;
                let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
                #[cfg(feature = "complex")]
                {
                    let mut acc_re = Kahan::new();
                    let mut acc_im = Kahan::new();
                    for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                        let prod = xi.conj() * yi;
                        acc_re.add(prod.real());
                        acc_im.add(prod.imag());
                    }
                    *slot = S::from_parts(acc_re.finish(), acc_im.finish());
                }
                #[cfg(not(feature = "complex"))]
                {
                    let mut acc = Kahan::new();
                    for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                        acc.add(xi * yi);
                    }
                    *slot = S::from_real(acc.finish());
                }
            });
        } else {
            for cid in 0..nchunks {
                let start = cid * REPRO_CHUNK;
                let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
                #[cfg(feature = "complex")]
                {
                    let mut acc_re = Kahan::new();
                    let mut acc_im = Kahan::new();
                    for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                        let prod = xi.conj() * yi;
                        acc_re.add(prod.real());
                        acc_im.add(prod.imag());
                    }
                    parts[cid] = S::from_parts(acc_re.finish(), acc_im.finish());
                }
                #[cfg(not(feature = "complex"))]
                {
                    let mut acc = Kahan::new();
                    for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                        acc.add(xi * yi);
                    }
                    parts[cid] = S::from_real(acc.finish());
                }
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for cid in 0..nchunks {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            #[cfg(feature = "complex")]
            {
                let mut acc_re = Kahan::new();
                let mut acc_im = Kahan::new();
                for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                    let prod = xi.conj() * yi;
                    acc_re.add(prod.real());
                    acc_im.add(prod.imag());
                }
                parts[cid] = S::from_parts(acc_re.finish(), acc_im.finish());
            }
            #[cfg(not(feature = "complex"))]
            {
                let mut acc = Kahan::new();
                for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                    acc.add(xi * yi);
                }
                parts[cid] = S::from_real(acc.finish());
            }
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
        if !crate::algebra::parallel_cfg::force_serial() {
            use rayon::prelude::*;
            parts.par_iter_mut().enumerate().for_each(|(cid, slot)| {
                let start = cid * REPRO_CHUNK;
                let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
                let mut acc = R::zero();
                for &value in &x[start..end] {
                    acc = acc + value.abs2();
                }
                *slot = acc;
            });
        } else {
            for cid in 0..nchunks {
                let start = cid * REPRO_CHUNK;
                let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
                let mut acc = R::zero();
                for &value in &x[start..end] {
                    acc = acc + value.abs2();
                }
                parts[cid] = acc;
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for cid in 0..nchunks {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = R::zero();
            for &value in &x[start..end] {
                acc = acc + value.abs2();
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

/// Deterministic sum of squared magnitudes with compensated chunk sums.
pub fn sum_abs2_local_repro_accurate(x: &[S]) -> R {
    if x.is_empty() {
        return R::zero();
    }

    let nchunks = (x.len() + REPRO_CHUNK - 1) / REPRO_CHUNK;
    let mut parts = vec![R::zero(); nchunks];

    #[cfg(feature = "rayon")]
    {
        if !crate::algebra::parallel_cfg::force_serial() {
            use rayon::prelude::*;
            parts.par_iter_mut().enumerate().for_each(|(cid, slot)| {
                let start = cid * REPRO_CHUNK;
                let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
                let mut acc = Kahan::new();
                for &value in &x[start..end] {
                    acc.add(value.abs2());
                }
                *slot = acc.finish();
            });
        } else {
            for cid in 0..nchunks {
                let start = cid * REPRO_CHUNK;
                let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
                let mut acc = Kahan::new();
                for &value in &x[start..end] {
                    acc.add(value.abs2());
                }
                parts[cid] = acc.finish();
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for cid in 0..nchunks {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = Kahan::new();
            for &value in &x[start..end] {
                acc.add(value.abs2());
            }
            parts[cid] = acc.finish();
        }
    }

    let mut total = R::zero();
    for part in parts {
        total = total + part;
    }
    total
}
