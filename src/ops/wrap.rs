use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
#[cfg(feature = "complex")]
use crate::algebra::scalar::{copy_real_to_scalar_in, copy_scalar_to_real_in};
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::matrix::op_bridge::matvec_s as bridge_matvec_s;
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::bridge::{
    apply_pc_mut_s as bridge_apply_pc_mut_s, apply_pc_s as bridge_apply_pc_s,
};
use crate::preconditioner::{PcSide, Preconditioner as PreconditionerF64};
use core::marker::PhantomData;

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
        bridge_matvec_s(self.inner, x, y, scratch);
    }

    #[inline]
    fn supports_t_matvec_s(&self) -> bool {
        <A as LinOp>::supports_transpose(self.inner)
    }

    #[inline]
    fn t_matvec_s(&self, x: &[S], y: &mut [S], scratch: &mut BridgeScratch) {
        #[cfg(not(feature = "complex"))]
        {
            let _ = scratch;
            let x_r: &[f64] = unsafe { &*(x as *const [S] as *const [f64]) };
            let y_r: &mut [f64] = unsafe { &mut *(y as *mut [S] as *mut [f64]) };
            <A as LinOp>::t_matvec(self.inner, x_r, y_r);
        }

        #[cfg(feature = "complex")]
        {
            let need = x.len().max(y.len());
            let (xr, yr) = scratch.real_pair(need);
            copy_scalar_to_real_in(x, &mut xr[..x.len()]);
            <A as LinOp>::t_matvec(self.inner, &xr[..x.len()], &mut yr[..y.len()]);
            copy_real_to_scalar_in(&yr[..y.len()], y);
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
        bridge_apply_pc_s(self.inner, side, x, y, scratch)
    }
}

/// Mutable adapter that exposes an `f64` preconditioner via [`KPreconditioner`].
pub struct F64AsSPcMut<'a, P: PreconditionerF64 + ?Sized> {
    inner: *mut P,
    _marker: PhantomData<&'a mut P>,
}

impl<'a, P: PreconditionerF64 + ?Sized> F64AsSPcMut<'a, P> {
    #[inline]
    pub fn new(inner: &'a mut P) -> Self {
        Self {
            inner: inner as *mut P,
            _marker: PhantomData,
        }
    }
}

#[inline]
pub fn as_s_pc_mut<'a, P: PreconditionerF64 + ?Sized>(pc: &'a mut P) -> F64AsSPcMut<'a, P> {
    F64AsSPcMut::new(pc)
}

unsafe impl<'a, P> Send for F64AsSPcMut<'a, P> where P: PreconditionerF64 + Send + Sync + ?Sized {}
unsafe impl<'a, P> Sync for F64AsSPcMut<'a, P> where P: PreconditionerF64 + Send + Sync + ?Sized {}

impl<'a, P> KPreconditioner for F64AsSPcMut<'a, P>
where
    P: PreconditionerF64 + Send + Sync + ?Sized,
{
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        unsafe { <P as PreconditionerF64>::dims(&*self.inner) }
    }

    #[inline]
    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_s(unsafe { &*self.inner }, side, x, y, scratch)
    }

    #[inline]
    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_mut_s(unsafe { &mut *self.inner }, side, x, y, scratch)
    }

    #[inline]
    fn on_restart_s(&mut self, outer_iter: usize, residual_norm: R) -> Result<(), KError> {
        unsafe { <P as PreconditionerF64>::on_restart(&mut *self.inner, outer_iter, residual_norm) }
    }
}
