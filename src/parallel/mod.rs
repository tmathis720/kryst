use crate::algebra::prelude::*;
use crate::reduction::ReproMode;
#[cfg(feature = "mpi")]
use mpi::datatype::Equivalence;
#[cfg(feature = "mpi")]
use mpi::raw::AsRaw;
use std::marker::PhantomData;
#[cfg(any(feature = "mpi", feature = "rayon"))]
use std::sync::Arc;

mod reduce;
mod reduce_async;
#[cfg(feature = "mpi")]
pub use reduce::allreduce_sum_scalar_mpi_sys;
pub use reduce::{
    GlobalReductionModeGuard, allreduce_sum_scalar_slice_in_place,
    allreduce_sum_scalar_slice_owned, allreduce_sum_scalar_slice_owned_with_mode,
    allreduce_sum_scalar_slice_with_mode, allreduce_sum_scalar_with_mode, global_dot_conj,
    global_dot_conj_accurate, global_dot_conj_many, global_dot_conj_many_accurate,
    global_dot_conj_many_into, global_dot_conj_many_into_accurate, global_dot_conj_many_into_repro,
    global_dot_conj_many_into_with_mode, global_dot_conj_many_repro,
    global_dot_conj_many_with_mode, global_dot_conj_repro, global_dot_conj_with_mode, global_nrm2,
    global_nrm2_accurate, global_nrm2_many, global_nrm2_many_accurate, global_nrm2_many_into,
    global_nrm2_many_into_accurate, global_nrm2_many_into_repro, global_nrm2_many_into_with_mode,
    global_nrm2_many_repro, global_nrm2_many_with_mode, global_nrm2_repro, global_nrm2_with_mode,
    global_reduce_tuple2, global_reduction_mode, set_global_reduction_mode,
    set_global_reduction_mode_scoped,
};
pub use reduce_async::{ReduceReqReal, ReduceReqScalar, ReduceReqScalars, ReduceReqTuple2};

// Opaque request that can represent multiple backends. Use a `PhantomData`
// to tie the lifetime when MPI support is disabled.
pub enum AnyRequest<'a> {
    #[cfg(feature = "mpi")]
    Mpi(MpiRequest<'a>),
    None(PhantomData<&'a ()>),
}

#[cfg(feature = "mpi")]
pub struct MpiRequest<'a> {
    pub(crate) handle: mpi::ffi::MPI_Request,
    pub(crate) _marker: std::marker::PhantomData<&'a mut [f64]>,
}

/// Abstract communicator for reductions & splits
pub trait Comm: Send + Sync + 'static {
    type Vec;
    type Request<'a>: 'a;
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    fn barrier(&self);
    #[cfg(feature = "mpi")]
    fn scatter<T: Clone + Equivalence>(&self, global: &[T], out: &mut [T], root: usize);
    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], root: usize);
    #[cfg(feature = "mpi")]
    fn gather<T: Clone + Equivalence>(&self, local: &[T], out: &mut Vec<T>, root: usize);
    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, root: usize);

    /// All‐reduce a scalar (sum) across ranks
    fn all_reduce_f64(&self, local: f64) -> f64;

    /// Reduce one scalar (sum) across ranks.
    fn allreduce_sum(&self, x: f64) -> f64 {
        self.all_reduce_f64(x)
    }

    /// Reduce two scalars (sum) in a single collective. Default: two single reductions.
    fn allreduce_sum2(&self, a: f64, b: f64) -> (f64, f64) {
        let a = self.allreduce_sum(a);
        let b = self.allreduce_sum(b);
        (a, b)
    }

    /// Reduce a slice of scalars in place (sum).
    fn allreduce_sum_slice(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = self.allreduce_sum(*x);
        }
    }

    /// Reduce a single scalar `S` (sum) across ranks.
    #[inline]
    fn allreduce_sum_scalar(&self, z: S) -> S {
        #[cfg(feature = "complex")]
        {
            let (re, im) = self.allreduce_sum2(z.real(), z.imag());
            S::from_parts(re, im)
        }
        #[cfg(not(feature = "complex"))]
        {
            S::from_real(self.allreduce_sum(z.real()))
        }
    }

    /// Split this communicator into sub‐colors
    fn split(&self, color: i32, key: i32) -> UniverseComm;

    /// Nonblocking receive into `buf` from `src`.
    fn irecv_from<'a>(&'a self, buf: &'a mut [f64], src: i32) -> Self::Request<'a>;
    /// Nonblocking send of `buf` to `dest`.
    fn isend_to<'a>(&'a self, buf: &'a [f64], dest: i32) -> Self::Request<'a>;

    /// Nonblocking receive of u64 words.
    fn irecv_from_u64<'a>(&'a self, buf: &'a mut [u64], src: i32) -> Self::Request<'a>;
    /// Nonblocking send of u64 words.
    fn isend_to_u64<'a>(&'a self, buf: &'a [u64], dest: i32) -> Self::Request<'a>;
    /// Wait for all requests to complete.
    fn wait_all<'a>(&self, reqs: &mut [Self::Request<'a>]);

    /// Legacy all_reduce method for backward compatibility
    fn all_reduce(&self, x: f64) -> f64 {
        self.all_reduce_f64(x)
    }

    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        let local = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f64>();
        self.all_reduce_f64(local)
    }
    // Parallel/distributed matrix-vector product
    fn parallel_mat_vec(&self, a: &faer::Mat<f64>, x: &[f64], y: &mut [f64]) {
        // Default: serial mat-vec
        assert_eq!(a.ncols(), x.len());
        assert_eq!(a.nrows(), y.len());
        for i in 0..a.nrows() {
            y[i] = 0.0;
            for j in 0..a.ncols() {
                y[i] += a[(i, j)] * x[j];
            }
        }
    }
}

/// Default no‐MPI/no‐parallel communicator for serial execution
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoComm;

impl Comm for NoComm {
    type Vec = Vec<f64>;
    type Request<'a> = ();

    fn rank(&self) -> usize {
        0
    }
    fn size(&self) -> usize {
        1
    }
    fn barrier(&self) {}

    #[cfg(feature = "mpi")]
    fn scatter<T: Clone + Equivalence>(&self, global: &[T], out: &mut [T], _root: usize) {
        // For no-comm case, just copy from first elements
        for (dst, src) in out.iter_mut().zip(global.iter()) {
            *dst = src.clone();
        }
    }
    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], _root: usize) {
        // For no-comm case, just copy from first elements
        for (dst, src) in out.iter_mut().zip(global.iter()) {
            *dst = src.clone();
        }
    }

    #[cfg(feature = "mpi")]
    fn gather<T: Clone + Equivalence>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        out.clear();
        out.extend_from_slice(local);
    }
    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        out.clear();
        out.extend_from_slice(local);
    }

    fn all_reduce_f64(&self, local: f64) -> f64 {
        local
    }

    fn allreduce_sum2(&self, a: f64, b: f64) -> (f64, f64) {
        (a, b)
    }

    fn allreduce_sum_slice(&self, _v: &mut [f64]) {}

    fn split(&self, _color: i32, _key: i32) -> UniverseComm {
        UniverseComm::NoComm(NoComm)
    }

    fn irecv_from<'a>(&'a self, _buf: &'a mut [f64], _src: i32) -> Self::Request<'a> {}
    fn isend_to<'a>(&'a self, _buf: &'a [f64], _dest: i32) -> Self::Request<'a> {}
    fn irecv_from_u64<'a>(&'a self, _buf: &'a mut [u64], _src: i32) -> Self::Request<'a> {}
    fn isend_to_u64<'a>(&'a self, _buf: &'a [u64], _dest: i32) -> Self::Request<'a> {}
    fn wait_all<'a>(&self, _reqs: &mut [Self::Request<'a>]) {}
}

impl NoComm {
    #[inline]
    pub fn allreduce_sum_scalar(&self, v: S) -> S {
        v
    }

    #[inline]
    pub fn allreduce_sum_scalars(&self, _buf: &mut [S]) {}

    #[inline]
    pub fn allreduce_sum_real(&self, v: R) -> R {
        v
    }
}

#[cfg(feature = "mpi")]
pub mod mpi_comm;
#[cfg(feature = "mpi")]
pub use mpi_comm::MpiComm;

pub mod threads;

#[cfg(feature = "rayon")]
pub mod rayon_comm;
#[cfg(feature = "rayon")]
pub use rayon_comm::RayonComm;

/// `UniverseComm` is a stable logical handle for MPI (or `NoComm` for serial).
/// It is `Clone + Eq` and cheap to pass; use it only for initialization or
/// when launching collectives. Do not store per-rank derived state here.
#[derive(Clone)]
pub enum UniverseComm {
    NoComm(NoComm),
    #[cfg(feature = "mpi")]
    Mpi(Arc<MpiComm>),
    #[cfg(feature = "rayon")]
    Rayon(RayonComm),
    #[cfg(not(any(feature = "mpi", feature = "rayon")))]
    Serial,
}

impl std::fmt::Debug for UniverseComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UniverseComm::NoComm(_) => f.write_str("NoComm"),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => write!(f, "MpiComm {{ id: {:?} }}", comm.world.as_raw()),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_) => f.write_str("Rayon"),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => f.write_str("Serial"),
        }
    }
}

impl PartialEq for UniverseComm {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (UniverseComm::NoComm(_), UniverseComm::NoComm(_)) => true,
            #[cfg(feature = "mpi")]
            (UniverseComm::Mpi(a), UniverseComm::Mpi(b)) => a.world.as_raw() == b.world.as_raw(),
            #[cfg(feature = "rayon")]
            (UniverseComm::Rayon(_), UniverseComm::Rayon(_)) => true,
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            (UniverseComm::Serial, UniverseComm::Serial) => true,
            _ => false,
        }
    }
}

impl Eq for UniverseComm {}

impl UniverseComm {
    #[cfg(feature = "mpi")]
    pub(crate) fn as_mpi(&self) -> Option<&mpi::topology::SimpleCommunicator> {
        match self {
            UniverseComm::Mpi(comm) => Some(&comm.world),
            _ => None,
        }
    }

    pub fn id(&self) -> u64 {
        match self {
            UniverseComm::NoComm(_) => 0,
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.world.as_raw() as u64,
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_) => 0,
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => 0,
        }
    }

    /// Sum a single scalar across all ranks (fast path).
    #[inline]
    pub fn allreduce_sum_scalar(&self, z: S) -> S {
        reduce::allreduce_sum_scalar_impl(self, z)
    }

    #[inline]
    pub fn iallreduce_sum_scalar<'a>(&self, local: S, out: &'a mut S) -> ReduceReqScalar<'a> {
        reduce_async::iallreduce_sum_scalar(self, local, out)
    }

    /// Sum an array of scalars in place across all ranks (fast path).
    #[inline]
    pub fn allreduce_sum_scalars(&self, buf: &mut [S]) {
        match self {
            UniverseComm::NoComm(comm) => comm.allreduce_sum_scalars(buf),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.allreduce_sum_scalars(buf),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_) => {}
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => {}
        }
    }

    #[inline]
    pub fn iallreduce_sum_scalars<'a>(&self, buf: &'a mut [S]) -> ReduceReqScalars<'a> {
        reduce_async::iallreduce_sum_scalars(self, buf)
    }

    /// Sum a single real value across all ranks (fast path).
    #[inline]
    pub fn allreduce_sum_real(&self, v: R) -> R {
        match self {
            UniverseComm::NoComm(comm) => comm.allreduce_sum_real(v),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.allreduce_sum_real(v),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.all_reduce_f64(v),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => v,
        }
    }

    #[inline]
    pub fn iallreduce_sum_real<'a>(&self, local: R, out: &'a mut R) -> ReduceReqReal<'a> {
        reduce_async::iallreduce_sum_real(self, local, out)
    }

    /// Deterministic scalar sum across all ranks.
    #[inline]
    pub fn allreduce_sum_scalar_repro(&self, z: S) -> S {
        reduce::allreduce_sum_scalar_repro_impl(self, z, ReproMode::Deterministic)
    }

    /// Accurate scalar sum across all ranks.
    #[inline]
    pub fn allreduce_sum_scalar_accurate(&self, z: S) -> S {
        reduce::allreduce_sum_scalar_repro_impl(self, z, ReproMode::DeterministicAccurate)
    }

    /// Deterministic scalar sum across all ranks using the requested reproducibility mode.
    #[inline]
    pub fn allreduce_sum_scalar_repro_with_mode(&self, z: S, mode: ReproMode) -> S {
        reduce::allreduce_sum_scalar_repro_impl(self, z, mode)
    }

    /// Sum a single scalar across all ranks using the raw `mpi_sys` interface.
    #[cfg(feature = "mpi")]
    #[inline]
    pub fn allreduce_sum_scalar_raw(&self, z: S) -> S {
        reduce::allreduce_sum_scalar_mpi_sys(self, z)
    }

    /// Sum a slice of scalars in place across all ranks (fast path).
    #[inline]
    pub fn allreduce_sum_scalar_slice(&self, data: &mut [S]) {
        reduce::allreduce_sum_scalar_slice_in_place(self, data)
    }

    /// Sum a slice of scalars across all ranks and return the result in a new buffer.
    #[inline]
    pub fn allreduce_sum_scalar_slice_owned(&self, data: &[S]) -> Vec<S> {
        reduce::allreduce_sum_scalar_slice_owned(self, data)
    }

    /// Sum a slice of scalars across all ranks using the requested reproducibility mode and
    /// return the result in a new buffer.
    #[inline]
    pub fn allreduce_sum_scalar_slice_owned_with_mode(
        &self,
        data: &[S],
        mode: ReproMode,
    ) -> Vec<S> {
        reduce::allreduce_sum_scalar_slice_owned_with_mode(self, data, mode)
    }

    #[inline]
    pub fn iallreduce_tuple2<'a>(
        &self,
        a_local: S,
        b_local: R,
        out_a: &'a mut S,
        out_b: &'a mut R,
    ) -> ReduceReqTuple2<'a> {
        reduce_async::iallreduce_tuple2(self, a_local, b_local, out_a, out_b)
    }
}

impl Comm for UniverseComm {
    type Vec = Vec<f64>; // Default, can be made generic
    type Request<'a> = AnyRequest<'a>;
    fn rank(&self) -> usize {
        match self {
            UniverseComm::NoComm(comm) => comm.rank(),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.rank(),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.rank(),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => 0,
        }
    }
    fn size(&self) -> usize {
        match self {
            UniverseComm::NoComm(comm) => comm.size(),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.size(),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.size(),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => 1,
        }
    }
    fn barrier(&self) {
        match self {
            UniverseComm::NoComm(comm) => comm.barrier(),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.barrier(),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.barrier(),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => {}
        }
    }
    #[cfg(feature = "mpi")]
    fn scatter<T: Clone + Equivalence>(&self, global: &[T], out: &mut [T], root: usize) {
        match self {
            UniverseComm::NoComm(comm) => comm.scatter(global, out, root),
            UniverseComm::Mpi(comm) => comm.scatter(global, out, root),
            _ => unreachable!(),
        }
    }
    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], root: usize) {
        match self {
            UniverseComm::NoComm(comm) => comm.scatter(global, out, root),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.scatter(global, out, root),
            #[cfg(not(feature = "rayon"))]
            UniverseComm::Serial => {
                for (dst, src) in out.iter_mut().zip(global.iter()) {
                    *dst = src.clone();
                }
            }
        }
    }
    #[cfg(feature = "mpi")]
    fn gather<T: Clone + Equivalence>(&self, local: &[T], out: &mut Vec<T>, root: usize) {
        match self {
            UniverseComm::NoComm(comm) => comm.gather(local, out, root),
            UniverseComm::Mpi(comm) => comm.gather(local, out, root),
            _ => unreachable!(),
        }
    }
    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, _root: usize) {
        match self {
            UniverseComm::NoComm(comm) => comm.gather(local, out, _root),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.gather(local, out, _root),
            #[cfg(not(feature = "rayon"))]
            UniverseComm::Serial => {
                out.clear();
                out.extend_from_slice(local);
            }
        }
    }
    fn all_reduce(&self, x: f64) -> f64 {
        match self {
            UniverseComm::NoComm(comm) => comm.all_reduce(x),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.all_reduce(x),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.all_reduce(x),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => x,
        }
    }

    fn all_reduce_f64(&self, local: f64) -> f64 {
        match self {
            UniverseComm::NoComm(comm) => comm.all_reduce_f64(local),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.all_reduce_f64(local),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.all_reduce_f64(local),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => local,
        }
    }

    fn split(&self, color: i32, key: i32) -> UniverseComm {
        match self {
            UniverseComm::NoComm(comm) => comm.split(color, key),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => comm.split(color, key),
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(comm) => comm.split(color, key),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => UniverseComm::Serial,
        }
    }

    fn irecv_from<'a>(&'a self, _buf: &'a mut [f64], _src: i32) -> Self::Request<'a> {
        match self {
            UniverseComm::NoComm(_comm) => AnyRequest::None(PhantomData),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => {
                // Post a true nonblocking receive via raw MPI and keep the handle
                let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
                let count = _buf.len() as i32;
                let src_rank = _src;
                let comm_raw = mpi::raw::AsRaw::as_raw(&comm.world);
                let rc = unsafe {
                    mpi::ffi::MPI_Irecv(
                        _buf.as_mut_ptr() as *mut std::ffi::c_void,
                        count,
                        mpi::ffi::RSMPI_DOUBLE,
                        src_rank,
                        0,
                        comm_raw,
                        &mut req,
                    )
                };
                debug_assert_eq!(rc, 0);
                AnyRequest::Mpi(MpiRequest {
                    handle: req,
                    _marker: std::marker::PhantomData,
                })
            }
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_comm) => AnyRequest::None(PhantomData),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => AnyRequest::None(PhantomData),
        }
    }
    fn isend_to<'a>(&'a self, _buf: &'a [f64], _dest: i32) -> Self::Request<'a> {
        match self {
            UniverseComm::NoComm(_comm) => AnyRequest::None(PhantomData),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => {
                // Post a true nonblocking send via raw MPI and keep the handle
                let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
                let count = _buf.len() as i32;
                let dest_rank = _dest;
                let comm_raw = mpi::raw::AsRaw::as_raw(&comm.world);
                let rc = unsafe {
                    mpi::ffi::MPI_Isend(
                        _buf.as_ptr() as *const std::ffi::c_void,
                        count,
                        mpi::ffi::RSMPI_DOUBLE,
                        dest_rank,
                        0,
                        comm_raw,
                        &mut req,
                    )
                };
                debug_assert_eq!(rc, 0);
                AnyRequest::Mpi(MpiRequest {
                    handle: req,
                    _marker: std::marker::PhantomData,
                })
            }
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_comm) => AnyRequest::None(PhantomData),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => AnyRequest::None(PhantomData),
        }
    }

    fn irecv_from_u64<'a>(&'a self, _buf: &'a mut [u64], _src: i32) -> Self::Request<'a> {
        match self {
            UniverseComm::NoComm(_comm) => AnyRequest::None(PhantomData),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => {
                // Nonblocking receive of u64 via raw MPI
                let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
                let count = _buf.len() as i32;
                let src_rank = _src;
                let comm_raw = mpi::raw::AsRaw::as_raw(&comm.world);
                let rc = unsafe {
                    mpi::ffi::MPI_Irecv(
                        _buf.as_mut_ptr() as *mut std::ffi::c_void,
                        count,
                        mpi::ffi::RSMPI_UINT64_T,
                        src_rank,
                        0,
                        comm_raw,
                        &mut req,
                    )
                };
                debug_assert_eq!(rc, 0);
                AnyRequest::Mpi(MpiRequest {
                    handle: req,
                    _marker: std::marker::PhantomData,
                })
            }
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_comm) => AnyRequest::None(PhantomData),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => AnyRequest::None(PhantomData),
        }
    }
    fn isend_to_u64<'a>(&'a self, _buf: &'a [u64], _dest: i32) -> Self::Request<'a> {
        match self {
            UniverseComm::NoComm(_comm) => AnyRequest::None(PhantomData),
            #[cfg(feature = "mpi")]
            UniverseComm::Mpi(comm) => {
                // Nonblocking send of u64 via raw MPI
                let mut req: mpi::ffi::MPI_Request = unsafe { std::mem::zeroed() };
                let count = _buf.len() as i32;
                let dest_rank = _dest;
                let comm_raw = mpi::raw::AsRaw::as_raw(&comm.world);
                let rc = unsafe {
                    mpi::ffi::MPI_Isend(
                        _buf.as_ptr() as *const std::ffi::c_void,
                        count,
                        mpi::ffi::RSMPI_UINT64_T,
                        dest_rank,
                        0,
                        comm_raw,
                        &mut req,
                    )
                };
                debug_assert_eq!(rc, 0);
                AnyRequest::Mpi(MpiRequest {
                    handle: req,
                    _marker: std::marker::PhantomData,
                })
            }
            #[cfg(feature = "rayon")]
            UniverseComm::Rayon(_comm) => AnyRequest::None(PhantomData),
            #[cfg(not(any(feature = "mpi", feature = "rayon")))]
            UniverseComm::Serial => AnyRequest::None(PhantomData),
        }
    }
    fn wait_all<'a>(&self, reqs: &mut [Self::Request<'a>]) {
        #[cfg(feature = "mpi")]
        {
            for r in reqs.iter_mut() {
                if let AnyRequest::Mpi(rq) = r {
                    let rc = unsafe {
                        mpi::ffi::MPI_Wait(&mut rq.handle, mpi::ffi::RSMPI_STATUS_IGNORE)
                    };
                    debug_assert_eq!(rc, 0);
                }
            }
        }
        // Clear handles so buffers can be reused immediately.
        for r in reqs.iter_mut() {
            *r = AnyRequest::None(PhantomData);
        }
    }
}

#[cfg(not(feature = "mpi"))]
pub trait Equivalence {}

pub enum ReduceOp {
    Sum,
    // Add more as needed
}
