//! Shared helpers so tests & examples work with S=f64 or S=Complex64.

use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::prelude::*;

/// Default tolerances (tweak to your solver accuracy)
pub const ATOL: R = 1e-12;
pub const RTOL: R = 1e-10;

/// Turn an f64 into S (real -> complex with im=0 under `complex`)
#[inline]
pub fn s(x: f64) -> S {
    S::from_real(x)
}

/// Is a scalar essentially zero? (`|z| < eps`)
#[inline]
pub fn is_zero(z: S, eps: R) -> bool {
    z.abs() < eps
}

/// Scalar approx equality using |a-b| <= atol + rtol*max(|a|,|b|)
#[inline]
pub fn approx_s(a: S, b: S, atol: R, rtol: R) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs());
    diff <= atol + rtol * scale
}

/// Assert scalar closeness (pretty message in both modes)
#[track_caller]
pub fn assert_s_close(label: &str, a: S, b: S, atol: R, rtol: R) {
    if !approx_s(a, b, atol, rtol) {
        panic!(
            "{label}: |a-b|={} (a={:?}, b={:?}) > atol+rtol*scale (atol={:.3e}, rtol={:.3e})",
            (a - b).abs(),
            a,
            b,
            atol,
            rtol
        );
    }
}

/// Vector 2-norm of difference (returns R)
#[inline]
pub fn vec_err(a: &[S], b: &[S]) -> R {
    let mut tmp = Vec::<S>::with_capacity(a.len());
    unsafe {
        tmp.set_len(a.len());
    }
    for i in 0..a.len() {
        tmp[i] = a[i] - b[i];
    }
    nrm2(&tmp)
}

/// Conjugate dot product helper (matches BLAS dot for reals)
#[inline]
pub fn vec_dot(a: &[S], b: &[S]) -> S {
    dot_conj(a, b)
}

/// Assert vector closeness via 2-norm
#[track_caller]
pub fn assert_vec_close(label: &str, a: &[S], b: &[S], atol: R, rtol: R) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} != {}",
        a.len(),
        b.len()
    );
    let err = vec_err(a, b);
    let na = nrm2(a);
    let nb = nrm2(b);
    let scale = na.max(nb);
    if err > atol + rtol * scale {
        panic!(
            "{label}: ||a-b||2={:.3e} > {:.3e} (atol+rtol*max(||a||,||b||))",
            err,
            atol + rtol * scale
        );
    }
}

/// Convenience macro: scalar close with defaults
#[macro_export]
macro_rules! assert_s_close {
    ($label:expr, $a:expr, $b:expr) => {{
        $crate::testkit::assert_s_close(
            $label,
            $a,
            $b,
            $crate::testkit::ATOL,
            $crate::testkit::RTOL,
        )
    }};
}

/// Convenience macro: vector close with defaults
#[macro_export]
macro_rules! assert_vec_close {
    ($label:expr, $a:expr, $b:expr) => {{
        $crate::testkit::assert_vec_close(
            $label,
            $a,
            $b,
            $crate::testkit::ATOL,
            $crate::testkit::RTOL,
        )
    }};
}
