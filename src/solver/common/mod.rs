pub mod buffer;
pub mod givens;

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::matrix::op::LinOp;
use crate::ops::klinop::KLinOp;
use crate::parallel::{Comm, UniverseComm, global_nrm2, global_reduction_mode};
use crate::reduction::{CommDeterministic, Packet, ReproMode, dot_local_slice};
#[cfg(feature = "complex")]
use crate::reduction::{DDP, KahanP, PacketAccum};
use crate::utils::reduction::{AllreduceHandle, AsyncComm, ReductOptions};

pub use buffer::take_or_resize;

/// Convert a conjugated dot-product result into its real scalar component.
///
/// Complex builds tolerate tiny imaginary drift introduced by roundoff during
/// distributed reductions.  The imaginary component is checked against a
/// scaled tolerance in debug builds; large drift is clamped by discarding the
/// imaginary part.
#[inline(always)]
pub fn dot_result_to_real(global: S) -> R {
    let real_part = global.real();
    #[cfg(feature = "complex")]
    {
        let imag_part = global.imag();
        let magnitude = global.abs();
        let eps = 128.0 * f64::EPSILON;
        let scale = 1.0 + magnitude;
        if imag_part.abs() > eps * scale {
            // Keep a debug-only guard to surface non-finite drift without
            // panicking on benign imaginary components.
            debug_assert!(
                imag_part.is_finite(),
                "dot_result_to_real: imaginary part is not finite: im={imag_part}, |s|={magnitude}"
            );
        }
    }
    real_part
}

/// Recompute the true residual norm ||r||_2 where r = b - A x.
///
/// This uses the provided `comm` for the dot-product reduction so it works in
/// both serial and distributed settings.
#[inline]
pub fn recompute_true_residual_norm<C: Comm + CommDeterministic>(
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
    let mode = global_reduction_mode();
    let summed = if comm.size() == 1 {
        local
    } else {
        match mode {
            ReproMode::Fast => dot_result_to_real(comm.allreduce_sum_scalar(S::from_real(local))),
            _ => {
                let packet = Packet::<1> { v: [local] };
                comm.allreduce_det(&packet, mode).v[0]
            }
        }
    };

    let clamped = if summed >= 0.0 { summed } else { 0.0 };
    clamped.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::{NoComm, UniverseComm, set_global_reduction_mode_scoped};
    use std::any::Any;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy)]
    struct IdentityOp;

    impl LinOp for IdentityOp {
        type S = f64;

        fn dims(&self) -> (usize, usize) {
            (2, 2)
        }

        fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
            y.copy_from_slice(x);
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    struct MockComm {
        fast_calls: AtomicUsize,
        det_calls: AtomicUsize,
        fast_value: f64,
        det_value: f64,
    }

    impl MockComm {
        fn new(fast_value: f64, det_value: f64) -> Self {
            Self {
                fast_calls: AtomicUsize::new(0),
                det_calls: AtomicUsize::new(0),
                fast_value,
                det_value,
            }
        }

        fn reset(&self) {
            self.fast_calls.store(0, Ordering::Relaxed);
            self.det_calls.store(0, Ordering::Relaxed);
        }
    }

    impl Comm for MockComm {
        type Vec = Vec<f64>;
        type Request<'a> = ();

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            2
        }

        fn barrier(&self) {}

        #[cfg(feature = "mpi")]
        fn scatter<T: Clone + mpi::datatype::Equivalence>(
            &self,
            _global: &[T],
            _out: &mut [T],
            _root: usize,
        ) {
            unimplemented!("scatter is not used in tests");
        }

        #[cfg(not(feature = "mpi"))]
        fn scatter<T: Clone>(&self, _global: &[T], _out: &mut [T], _root: usize) {
            unimplemented!("scatter is not used in tests");
        }

        #[cfg(feature = "mpi")]
        fn gather<T: Clone + mpi::datatype::Equivalence>(
            &self,
            _local: &[T],
            _out: &mut Vec<T>,
            _root: usize,
        ) {
            unimplemented!("gather is not used in tests");
        }

        #[cfg(not(feature = "mpi"))]
        fn gather<T: Clone>(&self, _local: &[T], _out: &mut Vec<T>, _root: usize) {
            unimplemented!("gather is not used in tests");
        }

        fn all_reduce_f64(&self, _local: f64) -> f64 {
            self.fast_value
        }

        fn allreduce_sum(&self, _x: f64) -> f64 {
            self.fast_value
        }

        fn allreduce_sum2(&self, _a: f64, _b: f64) -> (f64, f64) {
            (self.fast_value, self.fast_value)
        }

        fn allreduce_sum_slice(&self, _v: &mut [f64]) {
            unimplemented!("slice reductions are not used in tests");
        }

        fn allreduce_sum_scalar(&self, _z: S) -> S {
            self.fast_calls.fetch_add(1, Ordering::Relaxed);
            S::from_real(self.fast_value)
        }

        fn split(&self, _color: i32, _key: i32) -> UniverseComm {
            UniverseComm::NoComm(NoComm)
        }

        fn irecv_from<'a>(&'a self, _buf: &'a mut [f64], _src: i32) -> Self::Request<'a> {
            unimplemented!("irecv is not used in tests");
        }

        fn isend_to<'a>(&'a self, _buf: &'a [f64], _dest: i32) -> Self::Request<'a> {
            unimplemented!("isend is not used in tests");
        }

        fn irecv_from_u64<'a>(&'a self, _buf: &'a mut [u64], _src: i32) -> Self::Request<'a> {
            unimplemented!("irecv_u64 is not used in tests");
        }

        fn isend_to_u64<'a>(&'a self, _buf: &'a [u64], _dest: i32) -> Self::Request<'a> {
            unimplemented!("isend_u64 is not used in tests");
        }

        fn wait_all<'a>(&self, _reqs: &mut [Self::Request<'a>]) {}
    }

    impl CommDeterministic for MockComm {
        fn allreduce_det<const N: usize>(&self, _local: &Packet<N>, mode: ReproMode) -> Packet<N> {
            self.det_calls.fetch_add(1, Ordering::Relaxed);
            let value = match mode {
                ReproMode::Fast => self.fast_value,
                _ => self.det_value,
            };
            let mut out = Packet::<N>::default();
            for slot in out.v.iter_mut() {
                *slot = value;
            }
            out
        }
    }

    #[test]
    fn recompute_true_residual_norm_respects_global_mode() {
        let op = IdentityOp;
        let b = [2.0, -1.0];
        let x = [1.0, 0.0];
        let comm = MockComm::new(2.0, 4.0);
        let mut tmp = vec![0.0; 2];

        {
            let _guard = set_global_reduction_mode_scoped(ReproMode::Fast);
            tmp.fill(0.0);
            let norm = recompute_true_residual_norm(&op, &b, &x, &comm, &mut tmp);
            assert!((norm - 2.0_f64.sqrt()).abs() < 1e-12);
        }
        assert_eq!(comm.fast_calls.load(Ordering::Relaxed), 1);
        assert_eq!(comm.det_calls.load(Ordering::Relaxed), 0);

        comm.reset();
        {
            let _guard = set_global_reduction_mode_scoped(ReproMode::Deterministic);
            tmp.fill(0.0);
            let norm = recompute_true_residual_norm(&op, &b, &x, &comm, &mut tmp);
            assert!((norm - 2.0).abs() < 1e-12);
        }
        assert_eq!(comm.fast_calls.load(Ordering::Relaxed), 0);
        assert_eq!(comm.det_calls.load(Ordering::Relaxed), 1);

        comm.reset();
        {
            let _guard = set_global_reduction_mode_scoped(ReproMode::DeterministicAccurate);
            tmp.fill(0.0);
            let norm = recompute_true_residual_norm(&op, &b, &x, &comm, &mut tmp);
            assert!((norm - 2.0).abs() < 1e-12);
        }
        assert_eq!(comm.fast_calls.load(Ordering::Relaxed), 0);
        assert_eq!(comm.det_calls.load(Ordering::Relaxed), 1);
    }
}

#[inline]
pub fn recompute_true_residual_norm_s<A>(
    a: &A,
    b: &[S],
    x: &[S],
    comm: &UniverseComm,
    tmp: &mut [S],
    scratch: &mut BridgeScratch,
) -> R
where
    A: KLinOp<Scalar = S> + ?Sized,
{
    debug_assert_eq!(b.len(), tmp.len());

    let (rows, cols) = a.dims();
    if rows != 0 {
        debug_assert_eq!(b.len(), rows);
    }
    if cols != 0 {
        debug_assert_eq!(x.len(), cols);
    }

    a.matvec_s(x, tmp, scratch);
    for i in 0..tmp.len() {
        tmp[i] = b[i] - tmp[i];
    }
    global_nrm2(comm, tmp)
}

/// Compute the residual norm used for iteration monitors (the "reported" norm):
/// - Left preconditioning:  ||M^{-1} r||_2
/// - Right/Symmetric:      ||r||_2
///
/// The `r_true` slice is modified in-place to hold `r = b - A x` on entry, and
/// when `side` is Left and a preconditioner is provided, `scratch` is used to
/// hold `z = M^{-1} r`.
#[inline]
#[cfg(not(feature = "complex"))]
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
    pub handle: AllreduceHandle<(R, R)>,
    pub local: (R, R),
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
    let mode = opt.effective_mode();
    let a: R = dot_local_slice(x1, y1, mode);
    let b: R = dot_local_slice(x2, y2, mode);
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
) -> Result<(AllreduceHandle<(R, R)>, (R, R)), crate::error::KError> {
    debug_assert_eq!(x.len(), y.len());
    let mode = opt.effective_mode();
    let sum = dot_local_slice(x, y, mode);
    comm.allreduce2_async(sum, R::default(), opt)
}

/// Launch a single dot product asynchronously on scalar slices. The result is
/// encoded in the first entry of the returned pair.
pub fn dot1_async_s<C: AsyncComm + ?Sized>(
    comm: &C,
    x: &[S],
    y: &[S],
    opt: &ReductOptions,
) -> Result<(AllreduceHandle<(R, R)>, (R, R)), crate::error::KError> {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(not(feature = "complex"))]
    unsafe {
        let xr: &[f64] = &*(x as *const [S] as *const [f64]);
        let yr: &[f64] = &*(y as *const [S] as *const [f64]);
        dot1_async(comm, xr, yr, opt)
    }

    #[cfg(feature = "complex")]
    {
        let mode = opt.effective_mode();
        let Packet { v: [re, im] } = dot_conj_components(x, y, mode);
        comm.allreduce2_async(re, im, opt)
    }
}

/// Handle for a batch of asynchronous dot products.
#[derive(Debug)]
pub struct AsyncDotN {
    pub handle: AllreduceHandle<Vec<R>>,
    pub local: Vec<R>,
}

/// Launch multiple dot products asynchronously.
pub fn dotn_async<C: AsyncComm + ?Sized>(
    comm: &C,
    pairs: &[(/*x*/ &[f64], /*y*/ &[f64])],
    opt: &ReductOptions,
) -> AsyncDotN {
    let mut loc = vec![R::default(); pairs.len()];
    let mode = opt.effective_mode();
    for (k, (x, y)) in pairs.iter().enumerate() {
        debug_assert_eq!(x.len(), y.len());
        loc[k] = dot_local_slice(x, y, mode);
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
) -> (AllreduceHandle<(R, R)>, R) {
    let mode = opt.effective_mode();
    let sumsq: R = dot_local_slice(x, x, mode);
    let (handle, local) = comm
        .allreduce2_async(sumsq, R::default(), opt)
        .expect("async reduction launch");
    (handle, local.0)
}

/// Launch an asynchronous squared-norm reduction on scalar slices.
pub fn nrm2_async_s<C: AsyncComm + ?Sized>(
    comm: &C,
    x: &[S],
    opt: &ReductOptions,
) -> (AllreduceHandle<(R, R)>, R) {
    #[cfg(not(feature = "complex"))]
    unsafe {
        let xr: &[f64] = &*(x as *const [S] as *const [f64]);
        nrm2_async(comm, xr, opt)
    }

    #[cfg(feature = "complex")]
    {
        let mode = opt.effective_mode();
        let Packet { v: [re, im] } = dot_conj_components(x, x, mode);
        let (handle, local) = comm
            .allreduce2_async(re, im, opt)
            .expect("async reduction launch");
        (handle, local.0)
    }
}

#[cfg(feature = "complex")]
fn dot_conj_components(x: &[S], y: &[S], mode: ReproMode) -> Packet<2> {
    debug_assert_eq!(x.len(), y.len());
    match mode {
        ReproMode::Fast => {
            let mut re = 0.0;
            let mut im = 0.0;
            for (&xi, &yi) in x.iter().zip(y) {
                let prod = xi.conj() * yi;
                re += prod.real();
                im += prod.imag();
            }
            Packet { v: [re, im] }
        }
        ReproMode::Deterministic => {
            let mut acc = KahanP::<2>::new();
            for (&xi, &yi) in x.iter().zip(y) {
                let prod = xi.conj() * yi;
                acc.add(&Packet {
                    v: [prod.real(), prod.imag()],
                });
            }
            acc.finish()
        }
        ReproMode::DeterministicAccurate => {
            let mut acc = DDP::<2>::new();
            for (&xi, &yi) in x.iter().zip(y) {
                let prod = xi.conj() * yi;
                acc.add(&Packet {
                    v: [prod.real(), prod.imag()],
                });
            }
            acc.finish()
        }
    }
}
