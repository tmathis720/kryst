use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use kryst::parallel::{Comm, UniverseComm};
use kryst::reduction::{CommDeterministic, Packet, ReproMode};

#[derive(Clone)]
pub struct CountingComm<C> {
    pub inner: C,
    pub reduces: Arc<AtomicUsize>,
}

impl<C: Comm + Clone> CountingComm<C> {
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            reduces: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl<C: Comm + Clone> Comm for CountingComm<C> {
    type Vec = C::Vec;
    type Request<'a> = C::Request<'a>;

    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn size(&self) -> usize {
        self.inner.size()
    }
    fn barrier(&self) {
        self.inner.barrier()
    }

    #[cfg(feature = "mpi")]
    fn scatter<T: Clone + mpi::datatype::Equivalence>(
        &self,
        global: &[T],
        out: &mut [T],
        root: usize,
    ) {
        self.inner.scatter(global, out, root)
    }
    #[cfg(not(feature = "mpi"))]
    fn scatter<T: Clone>(&self, global: &[T], out: &mut [T], root: usize) {
        self.inner.scatter(global, out, root)
    }

    #[cfg(feature = "mpi")]
    fn gather<T: Clone + mpi::datatype::Equivalence>(
        &self,
        local: &[T],
        out: &mut Vec<T>,
        root: usize,
    ) {
        self.inner.gather(local, out, root)
    }
    #[cfg(not(feature = "mpi"))]
    fn gather<T: Clone>(&self, local: &[T], out: &mut Vec<T>, root: usize) {
        self.inner.gather(local, out, root)
    }

    fn all_reduce_f64(&self, x: f64) -> f64 {
        self.reduces.fetch_add(1, Ordering::Relaxed);
        self.inner.all_reduce_f64(x)
    }

    fn allreduce_sum(&self, x: f64) -> f64 {
        self.reduces.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_sum(x)
    }

    fn allreduce_sum2(&self, a: f64, b: f64) -> (f64, f64) {
        self.reduces.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_sum2(a, b)
    }

    fn allreduce_sum_slice(&self, v: &mut [f64]) {
        self.reduces.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_sum_slice(v)
    }

    fn split(&self, color: i32, key: i32) -> UniverseComm {
        self.inner.split(color, key)
    }

    fn irecv_from<'a>(&'a self, buf: &'a mut [f64], src: i32) -> Self::Request<'a> {
        self.inner.irecv_from(buf, src)
    }
    fn isend_to<'a>(&'a self, buf: &'a [f64], dest: i32) -> Self::Request<'a> {
        self.inner.isend_to(buf, dest)
    }

    fn irecv_from_u64<'a>(&'a self, buf: &'a mut [u64], src: i32) -> Self::Request<'a> {
        self.inner.irecv_from_u64(buf, src)
    }
    fn isend_to_u64<'a>(&'a self, buf: &'a [u64], dest: i32) -> Self::Request<'a> {
        self.inner.isend_to_u64(buf, dest)
    }

    fn wait_all<'a>(&self, reqs: &mut [Self::Request<'a>]) {
        self.inner.wait_all(reqs)
    }
}

impl<C: Comm + CommDeterministic + Clone> CommDeterministic for CountingComm<C> {
    fn allreduce_det<const N: usize>(&self, local: &Packet<N>, mode: ReproMode) -> Packet<N> {
        self.reduces.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_det(local, mode)
    }
}
