use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::matrix::op::LinOpF64;

/// Internal, scalar-generic linear operator for solvers.
///
/// Forward-looking notes:
/// - Additional operator kinds (transpose/adjoint, batched calls) can be added later
///   without breaking object safety.
/// - When the public API becomes scalar generic, this trait can be re-exported
///   directly.
pub trait KLinOp: Send + Sync {
    type Scalar: KrystScalar;

    /// Dimensions of the operator `(nrows, ncols)`.
    fn dims(&self) -> (usize, usize);

    /// Perform `y <- A x`.
    ///
    /// `scratch` allows adapters to reuse temporary buffers when bridging
    /// between scalar types. Native-`S` implementations may ignore it.
    fn matvec_s(&self, x: &[Self::Scalar], y: &mut [Self::Scalar], scratch: &mut BridgeScratch);
}

impl<T> KLinOp for T
where
    T: LinOpF64 + Send + Sync,
{
    type Scalar = f64;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        <T as LinOpF64>::dims(self)
    }

    #[inline]
    fn matvec_s(&self, x: &[f64], y: &mut [f64], _scratch: &mut BridgeScratch) {
        <T as LinOpF64>::matvec(self, x, y);
    }
}
