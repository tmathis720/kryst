use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Sub};

/// Trait for real scalar types used for magnitudes and comparisons.
///
/// This trait intentionally mirrors only the operations required by the
/// algorithms in this crate and is kept separate from `num_traits::Float`
/// so that complex support can be added without relying on external trait
/// hierarchies.
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
    /// Absolute value. For non-negative types, this is the identity.
    fn abs(self) -> Self;
}

/// Trait for scalar types, potentially complex, that algorithms operate on.
///
/// Every scalar has an associated real type used for magnitudes and
/// comparisons.
pub trait Scalar:
    Copy
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    type Real: RealScalar;

    fn zero() -> Self;
    fn one() -> Self;
    fn conj(self) -> Self;
    fn abs(self) -> Self::Real;
    fn is_finite(self) -> bool;
    fn from_real(x: Self::Real) -> Self;
    fn real(self) -> Self::Real;
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
        f64::is_finite(self)
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

impl Scalar for f64 {
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
    fn conj(self) -> Self {
        self
    }
    #[inline]
    fn abs(self) -> Self::Real {
        f64::abs(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn from_real(x: Self::Real) -> Self {
        x
    }
    #[inline]
    fn real(self) -> Self::Real {
        self
    }
}

pub trait RealField: private::Sealed + Copy + 'static {}
impl RealField for f64 {}

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
}
