use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
#[cfg(feature = "complex")]
use crate::algebra::scalar::{copy_real_to_scalar_in, copy_scalar_to_real_in};
use crate::matrix::op::LinOp;

#[inline]
pub fn matvec_s<A>(a: &A, x: &[S], y: &mut [S], scratch: &mut BridgeScratch)
where
    A: LinOp<S = f64> + ?Sized,
{
    debug_assert_eq!(x.len(), y.len());

    #[cfg(not(feature = "complex"))]
    {
        let _ = scratch;
        // SAFETY: when the complex feature is disabled we have S == f64.
        let x_r: &[f64] = unsafe { &*(x as *const [S] as *const [f64]) };
        let y_r: &mut [f64] = unsafe { &mut *(y as *mut [S] as *mut [f64]) };
        a.matvec(x_r, y_r);
        return;
    }

    #[cfg(feature = "complex")]
    {
        let n = x.len();
        let (xr, yr) = scratch.real_pair(n);
        copy_scalar_to_real_in(x, xr);
        a.matvec(xr, yr);
        copy_real_to_scalar_in(yr, y);
    }
}
