use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Sub};
#[cfg(feature = "complex")]
use num_complex::Complex64;

/// Scalar abstraction used throughout the crate.
///
/// Implementations should remain lightweight so the compiler can inline
/// aggressively in hot loops. The associated `Real` type corresponds to the
/// magnitude field for the scalar (always `f64` for the supported types).
pub trait KrystScalar: Copy + Send + Sync + 'static {
    type Real: Copy + Send + Sync + 'static;

    fn zero() -> Self;
    fn one() -> Self;

    fn abs(self) -> Self::Real;
    fn conj(self) -> Self;
    fn inv(self) -> Self;
    fn is_finite(self) -> bool;

    fn mul_add(self, a: Self, b: Self) -> Self;
}

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
    fn abs(self) -> Self::Real {
        self.abs()
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
        self.is_finite()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
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
    fn abs(self) -> Self::Real {
        self.norm()
    }

    #[inline]
    fn conj(self) -> Self {
        let Complex64 { re, im } = self;
        Complex64::new(re, -im)
    }

    #[inline]
    fn inv(self) -> Self {
        let Complex64 { re, im } = self;
        let denom = re * re + im * im;
        Complex64::new(re / denom, -im / denom)
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

#[cfg(feature = "complex")]
pub type S = Complex64;
#[cfg(not(feature = "complex"))]
pub type S = f64;

pub type R = <S as KrystScalar>::Real;

pub trait RealScalar:
    Copy
    + Send
    + Sync
    + 'static
    + PartialOrd
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn epsilon() -> Self;
    fn is_finite(self) -> bool;
    fn from_f64(x: f64) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
}

pub trait Scalar:
    KrystScalar<Real = f64>
    + Copy
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn from_real(x: <Self as KrystScalar>::Real) -> Self;
    fn real(self) -> <Self as KrystScalar>::Real;
}

impl RealScalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn epsilon() -> Self {
        f64::EPSILON
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.is_finite()
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Scalar for f64 {
    #[inline]
    fn from_real(x: <Self as KrystScalar>::Real) -> Self {
        x
    }

    #[inline]
    fn real(self) -> <Self as KrystScalar>::Real {
        self
    }
}

#[cfg(feature = "complex")]
impl Scalar for Complex64 {
    #[inline]
    fn from_real(x: <Self as KrystScalar>::Real) -> Self {
        Complex64::new(x, 0.0)
    }

    #[inline]
    fn real(self) -> <Self as KrystScalar>::Real {
        self.re
    }
}
