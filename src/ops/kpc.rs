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

    /// Apply the preconditioner in a mutable/flexible mode.
    ///
    /// By default this delegates to [`apply_s`], preserving backwards compatibility for
    /// immutable preconditioners while allowing flexible algorithms (e.g., FGMRES) to
    /// request a mutable handle when available.
    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[Self::Scalar],
        y: &mut [Self::Scalar],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        self.apply_s(side, x, y, scratch)
    }

    /// Optional hook invoked at solver restarts.
    #[allow(unused_variables)]
    fn on_restart_s(
        &mut self,
        outer_iter: usize,
        residual_norm: <Self::Scalar as KrystScalar>::Real,
    ) -> Result<(), KError> {
        let _ = (outer_iter, residual_norm);
        Ok(())
    }
}

#[cfg(not(feature = "complex"))]
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

    #[inline]
    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[f64],
        y: &mut [f64],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        <T as PreconditionerF64>::apply_mut(self, side, x, y)
    }

    #[inline]
    fn on_restart_s(
        &mut self,
        outer_iter: usize,
        residual_norm: <Self::Scalar as KrystScalar>::Real,
    ) -> Result<(), KError> {
        <T as PreconditionerF64>::on_restart(self, outer_iter, residual_norm)
    }
}
