use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::preconditioner::PcSide;
use crate::preconditioner::Preconditioner as PreconditionerF64;

/// Internal, scalar-generic preconditioner interface.
///
/// The trait is object safe so solvers can work with `Arc<dyn KPreconditioner>`. Future
/// extensions (transpose support, batched application) can be added without breaking
/// callers.
pub trait KPreconditioner: Send + Sync {
    type Scalar: KrystScalar;

    /// Dimensions of the preconditioner, typically `(n, n)`.
    fn dims(&self) -> (usize, usize);

    /// Apply the preconditioner.
    fn apply_s(
        &self,
        side: PcSide,
        x: &[Self::Scalar],
        y: &mut [Self::Scalar],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError>;
}

impl<T> KPreconditioner for T
where
    T: PreconditionerF64 + Send + Sync,
{
    type Scalar = f64;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        <T as PreconditionerF64>::dims(self)
    }

    #[inline]
    fn apply_s(
        &self,
        side: PcSide,
        x: &[f64],
        y: &mut [f64],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        <T as PreconditionerF64>::apply(self, side, x, y)
    }
}
