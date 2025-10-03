use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
#[cfg(feature = "complex")]
use crate::algebra::scalar::{copy_real_to_scalar_in, copy_scalar_to_real_in};
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::{PcSide, Preconditioner as PreconditionerF64};

/// Adapter that exposes an `f64` operator via the scalar-generic [`KLinOp`] interface.
pub struct F64AsSOp<'a, A: LinOpF64 + LinOp<S = f64> + ?Sized> {
    inner: &'a A,
}

impl<'a, A> F64AsSOp<'a, A>
where
    A: LinOpF64 + LinOp<S = f64> + ?Sized,
{
    #[inline]
    pub fn new(inner: &'a A) -> Self {
        Self { inner }
    }
}

#[inline]
pub fn as_s_op<'a, A>(op: &'a A) -> F64AsSOp<'a, A>
where
    A: LinOpF64 + LinOp<S = f64> + ?Sized,
{
    F64AsSOp::new(op)
}

impl<'a, A> KLinOp for F64AsSOp<'a, A>
where
    A: LinOpF64 + LinOp<S = f64> + Send + Sync + ?Sized,
{
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        <A as LinOpF64>::dims(self.inner)
    }

    #[inline]
    fn matvec_s(&self, x: &[S], y: &mut [S], scratch: &mut BridgeScratch) {
        #[cfg(not(feature = "complex"))]
        {
            let _ = scratch;
            let x_r: &[f64] = unsafe { &*(x as *const [S] as *const [f64]) };
            let y_r: &mut [f64] = unsafe { &mut *(y as *mut [S] as *mut [f64]) };
            <A as LinOpF64>::matvec(self.inner, x_r, y_r);
        }

        #[cfg(feature = "complex")]
        {
            let n = x.len();
            let xr = scratch.xr(n);
            let yr = scratch.yr(n);
            copy_scalar_to_real_in(x, xr);
            <A as LinOpF64>::matvec(self.inner, xr, yr);
            copy_real_to_scalar_in(yr, y);
        }
    }
}

/// Adapter that exposes an `f64` preconditioner via the scalar-generic [`KPreconditioner`] interface.
pub struct F64AsSPc<'a, P: PreconditionerF64 + ?Sized> {
    inner: &'a P,
}

impl<'a, P: PreconditionerF64 + ?Sized> F64AsSPc<'a, P> {
    #[inline]
    pub fn new(inner: &'a P) -> Self {
        Self { inner }
    }
}

#[inline]
pub fn as_s_pc<'a, P: PreconditionerF64 + ?Sized>(pc: &'a P) -> F64AsSPc<'a, P> {
    F64AsSPc::new(pc)
}

impl<'a, P> KPreconditioner for F64AsSPc<'a, P>
where
    P: PreconditionerF64 + Send + Sync + ?Sized,
{
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        <P as PreconditionerF64>::dims(self.inner)
    }

    #[inline]
    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        #[cfg(not(feature = "complex"))]
        {
            let _ = scratch;
            let x_r: &[f64] = unsafe { &*(x as *const [S] as *const [f64]) };
            let y_r: &mut [f64] = unsafe { &mut *(y as *mut [S] as *mut [f64]) };
            return <P as PreconditionerF64>::apply(self.inner, side, x_r, y_r);
        }

        #[cfg(feature = "complex")]
        {
            let n = x.len();
            let xr = scratch.xr(n);
            let yr = scratch.yr(n);
            copy_scalar_to_real_in(x, xr);
            <P as PreconditionerF64>::apply(self.inner, side, xr, yr)?;
            copy_real_to_scalar_in(yr, y);
            Ok(())
        }
    }
}
