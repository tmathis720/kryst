pub mod buffer;
pub mod givens;

use crate::algebra::blas::dot_conj;
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::matrix::op::LinOp;
use crate::ops::klinop::KLinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::utils::reduction::{AllreduceHandle, AsyncComm, ReductOptions};

pub use buffer::take_or_resize;

fn reduce_real<C: Comm>(comm: &C, value: R) -> R {
    if comm.size() == 1 {
        value
    } else {
        comm.allreduce_sum(value)
    }
}

/// Recompute the true residual norm ||r||_2 where r = b - A x.
///
/// This uses the provided `comm` for the dot-product reduction so it works in
/// both serial and distributed settings.
#[inline]
pub fn recompute_true_residual_norm<C: Comm>(
    a: &dyn LinOp<S = f64>,
    b: &[f64],
    x: &[f64],
    comm: &C,
    tmp: &mut [f64], // length = ncols
) -> f64 {
    a.matvec(x, tmp);
    let mut local = 0.0;
    for i in 0..tmp.len() {
        tmp[i] = b[i] - tmp[i];
        local += tmp[i] * tmp[i];
    }
    if comm.size() == 1 {
        local.sqrt()
    } else {
        comm.allreduce_sum(local).sqrt()
    }
}

#[inline]
pub fn recompute_true_residual_norm_s<A, C>(
    a: &A,
    b: &[S],
    x: &[S],
    comm: &C,
    tmp: &mut [S],
    scratch: &mut BridgeScratch,
) -> R
where
    A: KLinOp<Scalar = S> + ?Sized,
    C: Comm,
{
    debug_assert_eq!(b.len(), x.len());
    debug_assert_eq!(tmp.len(), x.len());

    a.matvec_s(x, tmp, scratch);
    for i in 0..tmp.len() {
        tmp[i] = b[i] - tmp[i];
    }
    let local = dot_conj(tmp, tmp).real();
    reduce_real(comm, local).sqrt()
}

/// Compute the residual norm used for iteration monitors (the "reported" norm):
/// - Left preconditioning:  ||M^{-1} r||_2
/// - Right/Symmetric:      ||r||_2
///
/// The `r_true` slice is modified in-place to hold `r = b - A x` on entry, and
/// when `side` is Left and a preconditioner is provided, `scratch` is used to
/// hold `z = M^{-1} r`.
#[inline]
pub fn reported_residual_norm(
    side: crate::preconditioner::PcSide,
    pc: Option<&dyn crate::preconditioner::Preconditioner>,
    r_true: &mut [f64],  // input: r = b - Ax, length = n
    scratch: &mut [f64], // length = n (used for M^{-1} r)
    comm: &UniverseComm,
) -> f64 {
    match side {
        crate::preconditioner::PcSide::Left => {
            if let Some(m) = pc {
                let _ = m.apply(crate::preconditioner::PcSide::Left, r_true, scratch);
                comm.dot(scratch, scratch).sqrt()
            } else {
                // no PC: Left semantics degrade to ||r||
                comm.dot(r_true, r_true).sqrt()
            }
        }
        crate::preconditioner::PcSide::Right | crate::preconditioner::PcSide::Symmetric => {
            comm.dot(r_true, r_true).sqrt()
        }
    }
}

/// Handle for a fused pair of asynchronous dot products.
#[derive(Debug)]
pub struct AsyncDot2 {
    pub handle: AllreduceHandle<(f64, f64)>,
    pub local: (f64, f64),
}

/// Launch a fused pair of dot products asynchronously.
pub fn dot2_async<C: AsyncComm + ?Sized>(
    comm: &C,
    x1: &[f64],
    y1: &[f64],
    x2: &[f64],
    y2: &[f64],
    opt: &ReductOptions,
) -> AsyncDot2 {
    debug_assert_eq!(x1.len(), y1.len());
    debug_assert_eq!(x2.len(), y2.len());
    let mut a = 0.0;
    let mut b = 0.0;
    for ((&xi, &yi), (&xj, &yj)) in x1.iter().zip(y1).zip(x2.iter().zip(y2)) {
        a += xi * yi;
        b += xj * yj;
    }
    let (handle, local) = comm
        .allreduce2_async(a, b, opt)
        .expect("async reduction launch");
    AsyncDot2 { handle, local }
}

/// Launch a single dot product asynchronously. The result is encoded in the
/// first entry of the returned pair.
pub fn dot1_async<C: AsyncComm + ?Sized>(
    comm: &C,
    x: &[f64],
    y: &[f64],
    opt: &ReductOptions,
) -> Result<(AllreduceHandle<(f64, f64)>, (f64, f64)), crate::error::KError> {
    debug_assert_eq!(x.len(), y.len());
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i] * y[i];
    }
    comm.allreduce2_async(sum, 0.0, opt)
}

/// Launch a single dot product asynchronously on scalar slices. The result is
/// encoded in the first entry of the returned pair.
pub fn dot1_async_s<C: AsyncComm + ?Sized>(
    comm: &C,
    x: &[S],
    y: &[S],
    opt: &ReductOptions,
) -> Result<(AllreduceHandle<(f64, f64)>, (f64, f64)), crate::error::KError> {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(not(feature = "complex"))]
    unsafe {
        let xr: &[f64] = &*(x as *const [S] as *const [f64]);
        let yr: &[f64] = &*(y as *const [S] as *const [f64]);
        return dot1_async(comm, xr, yr, opt);
    }

    #[cfg(feature = "complex")]
    {
        let sum = dot_conj(x, y).real();
        comm.allreduce2_async(sum, 0.0, opt)
    }
}

/// Handle for a batch of asynchronous dot products.
#[derive(Debug)]
pub struct AsyncDotN {
    pub handle: AllreduceHandle<Vec<f64>>,
    pub local: Vec<f64>,
}

/// Launch multiple dot products asynchronously.
pub fn dotn_async<C: AsyncComm + ?Sized>(
    comm: &C,
    pairs: &[(/*x*/ &[f64], /*y*/ &[f64])],
    opt: &ReductOptions,
) -> AsyncDotN {
    let mut loc = vec![0.0; pairs.len()];
    for (k, (x, y)) in pairs.iter().enumerate() {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += x[i] * y[i];
        }
        loc[k] = sum;
    }
    let (handle, local) = comm
        .allreduce_n_async(loc.clone(), opt)
        .expect("async reduction launch");
    AsyncDotN { handle, local }
}

/// Launch an asynchronous squared-norm reduction.
pub fn nrm2_async<C: AsyncComm + ?Sized>(
    comm: &C,
    x: &[f64],
    opt: &ReductOptions,
) -> (AllreduceHandle<(f64, f64)>, f64) {
    let mut sumsq = 0.0;
    for &xi in x {
        sumsq += xi * xi;
    }
    let (handle, local) = comm
        .allreduce2_async(sumsq, 0.0, opt)
        .expect("async reduction launch");
    (handle, local.0)
}

/// Launch an asynchronous squared-norm reduction on scalar slices.
pub fn nrm2_async_s<C: AsyncComm + ?Sized>(
    comm: &C,
    x: &[S],
    opt: &ReductOptions,
) -> (AllreduceHandle<(f64, f64)>, f64) {
    #[cfg(not(feature = "complex"))]
    unsafe {
        let xr: &[f64] = &*(x as *const [S] as *const [f64]);
        return nrm2_async(comm, xr, opt);
    }

    #[cfg(feature = "complex")]
    {
        let sumsq = dot_conj(x, x).real();
        let (handle, local) = comm
            .allreduce2_async(sumsq, 0.0, opt)
            .expect("async reduction launch");
        (handle, local.0)
    }
}
