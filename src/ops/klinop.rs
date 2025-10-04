use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;

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

    /// Whether the operator exposes a transpose/adjoint matvec.
    fn supports_t_matvec_s(&self) -> bool {
        false
    }

    /// Perform `y <- A^T x` (or `A^H x` in complex builds).
    ///
    /// Default implementation panics unless overridden by operators that
    /// advertise transpose support via [`supports_t_matvec_s`].
    fn t_matvec_s(
        &self,
        _x: &[Self::Scalar],
        _y: &mut [Self::Scalar],
        _scratch: &mut BridgeScratch,
    ) {
        panic!("KLinOp::t_matvec_s called but supports_t_matvec_s() == false");
    }
}
