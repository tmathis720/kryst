//! Helpers for applying real-valued operators inside scalar-generic code.
//!
//! In complex builds, these bridges project complex vectors to real workspaces,
//! apply a `LinOp<S = f64>`, then lift the result back into the active scalar
//! domain.
use crate::algebra::bridge::BridgeScratch;
#[cfg(feature = "complex")]
use crate::algebra::bridge::{copy_real_into_scalar, copy_scalar_to_real_in};
use crate::algebra::prelude::*;
use crate::matrix::op::LinOp;

/// Apply a real-valued operator `A : R^n -> R^m` to the active scalar vector.
///
/// - In real builds (`feature = "complex"` disabled), `S = f64` and this is a
///   thin wrapper over `a.matvec(x, y)`.
/// - In complex builds, `S = Complex64` but `A` remains `LinOp<S = f64>`. The
///   bridge projects the complex input to a real workspace, applies `A`, and
///   lifts the result back. The operator therefore acts on the real component
///   only; the imaginary part of `y` is determined solely by
///   [`crate::algebra::bridge::copy_real_into_scalar`].
///
/// This is intended for workflows where the operator is real but the surrounding
/// solver or preconditioner works in the complex scalar domain.
#[inline]
pub fn matvec_s<A>(a: &A, x: &[S], y: &mut [S], scratch: &mut BridgeScratch)
where
    A: LinOp<S = f64> + ?Sized,
{
    let (mut rows, mut cols) = a.dims();
    if rows == 0 {
        rows = y.len();
    } else {
        debug_assert_eq!(y.len(), rows);
    }
    if cols == 0 {
        cols = x.len();
    } else {
        debug_assert_eq!(x.len(), cols);
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
        let n = rows.max(cols);
        scratch.with_pair(n, |xr_full, yr_full| {
            let xr = &mut xr_full[..cols];
            let yr = &mut yr_full[..rows];
            copy_scalar_to_real_in(x, xr);
            a.matvec(xr, yr);
            copy_real_into_scalar(yr, y);
        });
    }
}

#[cfg(all(test, feature = "complex"))]
mod tests {
    use super::*;
    use crate::algebra::bridge::BridgeScratch;
    use num_complex::Complex64;

    struct SimpleRealOp;
    impl LinOp for SimpleRealOp {
        type S = f64;

        fn dims(&self) -> (usize, usize) {
            (2, 2)
        }

        fn matvec(&self, x: &[f64], y: &mut [f64]) {
            y[0] = 2.0 * x[0];
            y[1] = 3.0 * x[1];
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn bridge_projects_complex_inputs_to_real_operator() {
        let op = SimpleRealOp;
        let x = [Complex64::new(1.0, 5.0), Complex64::new(-1.0, 7.0)];
        let mut y = [Complex64::new(0.0, 0.0); 2];
        let mut scratch = BridgeScratch::default();

        matvec_s(&op, &x, &mut y, &mut scratch);

        assert_eq!(y[0].re, 2.0);
        assert_eq!(y[1].re, -3.0);
        assert_eq!(y[0].im, 0.0);
        assert_eq!(y[1].im, 0.0);
    }
}
