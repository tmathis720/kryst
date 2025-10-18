use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::algebra::bridge::{copy_real_into_scalar, copy_scalar_to_real_in};
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::preconditioner::{PcSide, Preconditioner};

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
    debug_assert_eq!(x.len(), y.len());

    #[cfg(not(feature = "complex"))]
    {
        let _ = scratch;
        // SAFETY: when the complex feature is disabled we have S == f64.
        let x_r: &[f64] = unsafe { &*(x as *const [S] as *const [f64]) };
        let y_r: &mut [f64] = unsafe { &mut *(y as *mut [S] as *mut [f64]) };
        pc.apply(side, x_r, y_r)
    }

    #[cfg(feature = "complex")]
    {
        let n = x.len();
        scratch.with_pair(n, |xr, yr| {
            copy_scalar_to_real_in(x, xr);
            pc.apply(side, xr, yr)?;
            copy_real_into_scalar(yr, y);
            Ok(())
        })
    }
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
    debug_assert_eq!(x.len(), y.len());

    #[cfg(not(feature = "complex"))]
    {
        let _ = scratch;
        // SAFETY: when the complex feature is disabled we have S == f64.
        let x_r: &[f64] = unsafe { &*(x as *const [S] as *const [f64]) };
        let y_r: &mut [f64] = unsafe { &mut *(y as *mut [S] as *mut [f64]) };
        pc.apply_mut(side, x_r, y_r)
    }

    #[cfg(feature = "complex")]
    {
        let n = x.len();
        scratch.with_pair(n, |xr, yr| {
            copy_scalar_to_real_in(x, xr);
            pc.apply_mut(side, xr, yr)?;
            copy_real_into_scalar(yr, y);
            Ok(())
        })
    }
}
