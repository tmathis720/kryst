use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::preconditioner::{Op, PcSide, Preconditioner};

#[inline]
pub fn apply_pc_s<P>(
    pc: &P,
    side: PcSide,
    x: &[S],
    y: &mut [S],
    scratch: &mut BridgeScratch,
) -> Result<(), KError>
where
    P: Preconditioner + ?Sized,
{
    let _ = scratch;
    debug_assert_eq!(x.len(), y.len());
    pc.apply(side, x, y)
}

#[inline]
pub fn apply_pc_mut_s<P>(
    pc: &mut P,
    side: PcSide,
    x: &[S],
    y: &mut [S],
    scratch: &mut BridgeScratch,
) -> Result<(), KError>
where
    P: Preconditioner + ?Sized,
{
    let _ = scratch;
    debug_assert_eq!(x.len(), y.len());
    pc.apply_mut(side, x, y)
}

#[inline]
pub fn apply_pc_op_s<P>(
    pc: &P,
    op: Op,
    x: &[S],
    y: &mut [S],
    scratch: &mut BridgeScratch,
) -> Result<(), KError>
where
    P: Preconditioner + ?Sized,
{
    let _ = scratch;
    debug_assert_eq!(x.len(), y.len());
    pc.apply_op(op, x, y)
}
