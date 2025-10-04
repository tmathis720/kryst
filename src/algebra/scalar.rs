#![allow(clippy::excessive_precision)]

use core::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "complex")]
use num_complex::Complex64;

/// Scalar abstraction used internally by kryst.  The goal is to
/// keep the public API monomorphic (f64), while the internals use `S`.
pub trait KrystScalar:
    Copy
    + Clone
    + Send
    + Sync
    + 'static
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The corresponding real type (always `f64` for now).
    type Real: Copy
        + Clone
        + Send
        + Sync
        + 'static
        + Default
        + PartialEq
        + PartialOrd
        + Add<Output = Self::Real>
        + Sub<Output = Self::Real>
        + Mul<Output = Self::Real>
        + Div<Output = Self::Real>;

    // Constructors / constants
    fn zero() -> Self;
    fn one() -> Self;

    /// Convert from a real (`f64`) to this scalar type.
    fn from_real(x: Self::Real) -> Self;

    /// Extract the real part (identity for real, `.re` for complex).
    fn real(self) -> Self::Real;

    /// Extract the imaginary part (zero for real scalars).
    fn imag(self) -> Self::Real;

    /// Construct a scalar from its real and imaginary components.
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;

    // Basic ops we need everywhere
    fn abs(self) -> Self::Real; // |z| for complex, |x| for real
    fn conj(self) -> Self; // identity for real
    fn inv(self) -> Self; // 1/self (caller ensures nonzero)
    fn is_finite(self) -> bool;

    /// Fused multiply-add.  For `f64` we use HW FMA; for complex we fall back.
    fn mul_add(self, a: Self, b: Self) -> Self;
}

// ==================== Implementations ====================

impl KrystScalar for f64 {
    type Real = f64;

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn from_real(x: Self::Real) -> Self {
        x
    }

    #[inline]
    fn real(self) -> Self::Real {
        self
    }

    #[inline]
    fn imag(self) -> Self::Real {
        0.0
    }

    #[inline]
    fn from_parts(re: Self::Real, _im: Self::Real) -> Self {
        re
    }

    #[inline]
    fn abs(self) -> Self::Real {
        f64::abs(self)
    }

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn inv(self) -> Self {
        1.0 / self
    }

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
}

#[cfg(feature = "complex")]
impl KrystScalar for Complex64 {
    type Real = f64;

    #[inline]
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }

    #[inline]
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    #[inline]
    fn from_real(x: Self::Real) -> Self {
        Complex64::new(x, 0.0)
    }

    #[inline]
    fn real(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn imag(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn from_parts(re: Self::Real, im: Self::Real) -> Self {
        Complex64::new(re, im)
    }

    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }

    #[inline]
    fn conj(self) -> Self {
        Complex64::new(self.re, -self.im)
    }

    #[inline]
    fn inv(self) -> Self {
        let n2 = self.re * self.re + self.im * self.im;
        Complex64::new(self.re / n2, -self.im / n2)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
}

// ==================== Feature-gated scalar choice ====================

#[cfg(feature = "complex")]
pub type S = Complex64;
#[cfg(not(feature = "complex"))]
pub type S = f64;

/// Real partner of `S` (currently always `f64`)
pub type R = <S as KrystScalar>::Real;

#[cfg(feature = "complex")]
#[inline]
pub fn copy_scalar_to_real_in(z: &[S], out: &mut [f64]) {
    debug_assert_eq!(z.len(), out.len());
    for (dst, &src) in out.iter_mut().zip(z.iter()) {
        *dst = src.real();
    }
}

#[cfg(feature = "complex")]
#[inline]
pub fn copy_real_to_scalar_in(x: &[f64], out: &mut [S]) {
    debug_assert_eq!(x.len(), out.len());
    for (dst, &src) in out.iter_mut().zip(x.iter()) {
        *dst = S::from_real(src);
    }
}

#[cfg(not(feature = "complex"))]
#[inline]
pub fn copy_scalar_to_real_in(z: &[S], out: &mut [f64]) {
    debug_assert_eq!(z.len(), out.len());
    if core::ptr::eq(z.as_ptr() as *const f64, out.as_ptr()) {
        return;
    }
    // SAFETY: when the complex feature is disabled we have S == f64.
    let z_as_f64: &[f64] = unsafe { &*(z as *const [S] as *const [f64]) };
    out.copy_from_slice(z_as_f64);
}

#[cfg(not(feature = "complex"))]
#[inline]
pub fn copy_real_to_scalar_in(x: &[f64], out: &mut [S]) {
    debug_assert_eq!(x.len(), out.len());
    if core::ptr::eq(x.as_ptr(), out.as_ptr() as *const f64) {
        return;
    }
    // SAFETY: when the complex feature is disabled we have S == f64.
    let out_as_f64: &mut [f64] = unsafe { &mut *(out as *mut [S] as *mut [f64]) };
    out_as_f64.copy_from_slice(x);
}
