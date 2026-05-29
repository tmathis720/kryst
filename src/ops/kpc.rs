use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::error::KError;
#[cfg(not(feature = "complex"))]
use crate::preconditioner::Preconditioner as PreconditionerF64;
use crate::preconditioner::{Op, PcCaps, PcSide};

/// Internal, scalar-generic preconditioner interface.
///
/// The trait is object safe so solvers can work with `Arc<dyn KPreconditioner<Scalar = S> + Sync>` when
/// sharing a handle between threads, or with mutable references when per-thread access is sufficient.
pub trait KPreconditioner: Send {
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

    /// Apply a selected matrix operation through the preconditioner.
    fn apply_op_s(
        &self,
        op: Op,
        x: &[Self::Scalar],
        y: &mut [Self::Scalar],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        match op {
            Op::NoTrans => self.apply_s(PcSide::Left, x, y, scratch),
            Op::Trans | Op::ConjTrans => self.apply_s(PcSide::Left, x, y, scratch),
        }
    }

    /// Capabilities exposed by the wrapped preconditioner.
    fn capabilities_s(&self) -> PcCaps {
        PcCaps::default()
    }

    /// Apply the preconditioner in a mutable/flexible mode.
    ///
    /// By default this delegates to `apply_s`, preserving backwards compatibility for
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
/// Blanket impl for native `f64` preconditioners. Complex builds should use the adapters in
/// [`crate::ops::wrap`] instead of this impl.
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
    fn apply_op_s(
        &self,
        op: Op,
        x: &[f64],
        y: &mut [f64],
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        <T as PreconditionerF64>::apply_op(self, op, x, y)
    }

    #[inline]
    fn capabilities_s(&self) -> PcCaps {
        <T as PreconditionerF64>::capabilities(self)
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
