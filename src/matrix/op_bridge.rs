use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::algebra::bridge::{copy_real_into_scalar, copy_scalar_to_real_in};
use crate::algebra::prelude::*;
use crate::matrix::op::LinOp;

#[inline]
pub fn matvec_s<A>(a: &A, x: &[S], y: &mut [S], scratch: &mut BridgeScratch)
where
    A: LinOp<S = f64> + ?Sized,
{
    let (rows, cols) = a.dims();
    if rows != 0 || cols != 0 {
        debug_assert_eq!(x.len(), cols);
        debug_assert_eq!(y.len(), rows);
    }

    #[cfg(not(feature = "complex"))]
    {
        let _ = scratch;
        // SAFETY: when the complex feature is disabled we have S == f64.
        let x_r: &[f64] = unsafe { &*(x as *const [S] as *const [f64]) };
        let y_r: &mut [f64] = unsafe { &mut *(y as *mut [S] as *mut [f64]) };
        a.matvec(x_r, y_r);
    }

    #[cfg(feature = "complex")]
    {
        let n = x.len();
        scratch.with_pair(n, |xr, yr| {
            copy_scalar_to_real_in(x, xr);
            a.matvec(xr, yr);
            copy_real_into_scalar(yr, y);
        });
    }
}
