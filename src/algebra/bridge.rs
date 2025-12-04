//! Temporary buffers and copy helpers for bridging real operators into the
//! active scalar domain.
//!
//! Complex builds project complex vectors to real workspace slices, apply a
//! `LinOp<S = f64>`, then lift the result back into scalars.
use crate::algebra::prelude::*;

/// Temporary buffers reused by solver bridges when converting between `S` and `f64`.
#[derive(Default, Clone, Debug)]
pub struct BridgeScratch {
    buf: Vec<f64>,
}

impl BridgeScratch {
    /// Create an empty scratch buffer.
    #[inline]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    #[inline]
    fn ensure(&mut self, want: usize) {
        if self.buf.len() < want {
            self.buf.resize(want, 0.0);
        }
    }

    /// Loan two disjoint real buffers of length `n` at once.
    #[inline]
    pub fn with_pair<F, Rv>(&mut self, n: usize, f: F) -> Rv
    where
        F: FnOnce(&mut [f64], &mut [f64]) -> Rv,
    {
        self.ensure(2 * n);
        let (xr, rest) = self.buf.split_at_mut(n);
        let (yr, _) = rest.split_at_mut(n);
        f(xr, yr)
    }

    /// Loan a single temporary buffer of length `n`.
    #[inline]
    pub fn with_one<F, Rv>(&mut self, n: usize, f: F) -> Rv
    where
        F: FnOnce(&mut [f64]) -> Rv,
    {
        self.ensure(n);
        f(&mut self.buf[..n])
    }
}

/// Copy from any scalar `T` whose real part is `f64` into a real buffer.
///
/// This helper is intentionally generic so bridge code can handle arbitrary
/// `KrystScalar` implementations without depending on the global alias `S`.
#[inline]
pub fn copy_scalar_to_real_in<T: KrystScalar<Real = f64>>(x: &[T], xr: &mut [f64]) {
    debug_assert_eq!(x.len(), xr.len());
    for (dst, &src) in xr.iter_mut().zip(x.iter()) {
        *dst = src.real();
    }
}

/// Copy from a real buffer into any scalar `T` whose real part is `f64`.
///
/// This helper is intentionally generic so bridge code can handle arbitrary
/// `KrystScalar` implementations without depending on the global alias `S`.
#[inline]
pub fn copy_real_into_scalar<T: KrystScalar<Real = f64>>(yr: &[f64], y: &mut [T]) {
    debug_assert_eq!(yr.len(), y.len());
    for (dst, &src) in y.iter_mut().zip(yr.iter()) {
        *dst = T::from_real(src);
    }
}
