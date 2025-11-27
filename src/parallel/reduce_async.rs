use crate::algebra::prelude::*;
use crate::parallel::UniverseComm;
use std::marker::PhantomData;

#[cfg(feature = "mpi")]
use crate::parallel::mpi_comm::OwnedMpiRequest;

#[cfg(all(feature = "mpi", feature = "complex"))]
type SmallFixed = [R; 2];
#[cfg(all(feature = "mpi", not(feature = "complex")))]
type SmallFixed = [R; 1];

pub struct ReduceReqScalar<'a> {
    inner: ReduceReqScalarInner<'a>,
}

enum ReduceReqScalarInner<'a> {
    NoComm(PhantomData<&'a mut ()>),
    #[cfg(feature = "mpi")]
    Mpi {
        _send: SmallFixed,
        recv: SmallFixed,
        out: &'a mut S,
        req: OwnedMpiRequest,
    },
}

impl<'a> ReduceReqScalar<'a> {
    pub(crate) fn new_no_comm() -> Self {
        Self {
            inner: ReduceReqScalarInner::NoComm(PhantomData),
        }
    }

    #[cfg(feature = "mpi")]
    pub(crate) fn new_mpi(
        _send: SmallFixed,
        recv: SmallFixed,
        out: &'a mut S,
        req: OwnedMpiRequest,
    ) -> Self {
        Self {
            inner: ReduceReqScalarInner::Mpi {
                _send,
                recv,
                out,
                req,
            },
        }
    }

    pub fn wait(self) {
        match self.inner {
            ReduceReqScalarInner::NoComm(_) => {}
            #[cfg(feature = "mpi")]
            ReduceReqScalarInner::Mpi {
                recv, out, mut req, ..
            } => {
                req.wait();
                *out = super::reduce::unpack_rr_to_scalar_s(recv);
            }
        }
    }
}

pub struct ReduceReqScalars<'a> {
    inner: ReduceReqScalarsInner<'a>,
}

enum ReduceReqScalarsInner<'a> {
    NoComm(PhantomData<&'a mut ()>),
    #[cfg(feature = "mpi")]
    Mpi {
        _send: Vec<R>,
        recv: Vec<R>,
        out: &'a mut [S],
        req: OwnedMpiRequest,
        pack_stride: usize,
    },
}

impl<'a> ReduceReqScalars<'a> {
    pub(crate) fn new_no_comm() -> Self {
        Self {
            inner: ReduceReqScalarsInner::NoComm(PhantomData),
        }
    }

    #[cfg(feature = "mpi")]
    pub(crate) fn new_mpi(
        _send: Vec<R>,
        recv: Vec<R>,
        out: &'a mut [S],
        req: OwnedMpiRequest,
        pack_stride: usize,
    ) -> Self {
        Self {
            inner: ReduceReqScalarsInner::Mpi {
                _send,
                recv,
                out,
                req,
                pack_stride,
            },
        }
    }

    pub fn wait(self) {
        match self.inner {
            ReduceReqScalarsInner::NoComm(_) => {}
            #[cfg(feature = "mpi")]
            ReduceReqScalarsInner::Mpi {
                recv,
                out,
                mut req,
                pack_stride,
                ..
            } => {
                req.wait();
                if pack_stride == 1 {
                    for (slot, value) in out.iter_mut().zip(recv.iter()) {
                        *slot = S::from_real(*value);
                    }
                } else {
                    #[cfg(feature = "complex")]
                    {
                        for (slot, chunk) in out.iter_mut().zip(recv.chunks_exact(pack_stride)) {
                            if pack_stride == 2 {
                                *slot = S::from_parts(chunk[0], chunk[1]);
                            }
                        }
                    }
                    #[cfg(not(feature = "complex"))]
                    {
                        unreachable!("complex stride expected only for complex feature");
                    }
                }
            }
        }
    }
}

pub struct ReduceReqReal<'a> {
    inner: ReduceReqRealInner<'a>,
}

#[cfg(feature = "mpi")]
type RealFixed = [R; 1];

enum ReduceReqRealInner<'a> {
    NoComm(PhantomData<&'a mut ()>),
    #[cfg(feature = "mpi")]
    Mpi {
        _send: RealFixed,
        recv: RealFixed,
        out: &'a mut R,
        req: OwnedMpiRequest,
    },
}

impl<'a> ReduceReqReal<'a> {
    pub(crate) fn new_no_comm() -> Self {
        Self {
            inner: ReduceReqRealInner::NoComm(PhantomData),
        }
    }

    #[cfg(feature = "mpi")]
    pub(crate) fn new_mpi(
        _send: RealFixed,
        recv: RealFixed,
        out: &'a mut R,
        req: OwnedMpiRequest,
    ) -> Self {
        Self {
            inner: ReduceReqRealInner::Mpi {
                _send,
                recv,
                out,
                req,
            },
        }
    }

    pub fn wait(self) {
        match self.inner {
            ReduceReqRealInner::NoComm(_) => {}
            #[cfg(feature = "mpi")]
            ReduceReqRealInner::Mpi {
                recv, out, mut req, ..
            } => {
                req.wait();
                *out = recv[0];
            }
        }
    }
}

pub struct ReduceReqTuple2<'a> {
    inner: ReduceReqTuple2Inner<'a>,
}

enum ReduceReqTuple2Inner<'a> {
    NoComm(PhantomData<&'a mut ()>),
    #[cfg(feature = "mpi")]
    Mpi {
        _send: Vec<R>,
        recv: Vec<R>,
        out_a: &'a mut S,
        out_b: &'a mut R,
        req: OwnedMpiRequest,
    },
}

impl<'a> ReduceReqTuple2<'a> {
    pub(crate) fn new_no_comm() -> Self {
        Self {
            inner: ReduceReqTuple2Inner::NoComm(PhantomData),
        }
    }

    #[cfg(feature = "mpi")]
    pub(crate) fn new_mpi(
        _send: Vec<R>,
        recv: Vec<R>,
        out_a: &'a mut S,
        out_b: &'a mut R,
        req: OwnedMpiRequest,
    ) -> Self {
        Self {
            inner: ReduceReqTuple2Inner::Mpi {
                _send,
                recv,
                out_a,
                out_b,
                req,
            },
        }
    }

    pub fn wait(self) {
        match self.inner {
            ReduceReqTuple2Inner::NoComm(_) => {}
            #[cfg(feature = "mpi")]
            ReduceReqTuple2Inner::Mpi {
                recv,
                out_a,
                out_b,
                mut req,
                ..
            } => {
                req.wait();
                #[cfg(feature = "complex")]
                {
                    debug_assert_eq!(recv.len(), 3);
                    *out_a = S::from_parts(recv[0], recv[1]);
                    *out_b = recv[2];
                }
                #[cfg(not(feature = "complex"))]
                {
                    debug_assert_eq!(recv.len(), 2);
                    *out_a = S::from_real(recv[0]);
                    *out_b = recv[1];
                }
            }
        }
    }
}

pub(crate) fn iallreduce_sum_scalar<'a>(
    comm: &UniverseComm,
    local: S,
    out: &'a mut S,
) -> ReduceReqScalar<'a> {
    match comm {
        UniverseComm::NoComm(_) => {
            *out = local;
            ReduceReqScalar::new_no_comm()
        }
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => {
            let send = crate::parallel::reduce::pack_scalar_s_to_rr(local);
            let mut recv = send;
            let req = inner.immediate_allreduce_sum(&send[..], &mut recv[..]);
            ReduceReqScalar::new_mpi(send, recv, out, req)
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => {
            *out = local;
            ReduceReqScalar::new_no_comm()
        }
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => {
            *out = local;
            ReduceReqScalar::new_no_comm()
        }
    }
}

pub(crate) fn iallreduce_sum_scalars<'a>(
    comm: &UniverseComm,
    buf: &'a mut [S],
) -> ReduceReqScalars<'a> {
    #[cfg(not(feature = "mpi"))]
    let _ = buf;
    match comm {
        UniverseComm::NoComm(_) => ReduceReqScalars::new_no_comm(),
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => {
            #[cfg(feature = "complex")]
            let pack_stride = 2usize;
            #[cfg(not(feature = "complex"))]
            let pack_stride = 1usize;

            let mut recv = Vec::<R>::with_capacity(buf.len() * pack_stride);
            #[cfg(feature = "complex")]
            {
                for &value in buf.iter() {
                    recv.extend_from_slice(&crate::parallel::reduce::pack_scalar_s_to_rr(value));
                }
            }
            #[cfg(not(feature = "complex"))]
            {
                for &value in buf.iter() {
                    recv.extend_from_slice(&crate::parallel::reduce::pack_scalar_s_to_rr(value));
                }
            }
            let send = recv.clone();
            let req = inner.immediate_allreduce_sum(&send[..], recv.as_mut_slice());
            ReduceReqScalars::new_mpi(send, recv, buf, req, pack_stride)
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => ReduceReqScalars::new_no_comm(),
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => ReduceReqScalars::new_no_comm(),
    }
}

pub(crate) fn iallreduce_sum_real<'a>(
    comm: &UniverseComm,
    local: R,
    out: &'a mut R,
) -> ReduceReqReal<'a> {
    match comm {
        UniverseComm::NoComm(_) => {
            *out = local;
            ReduceReqReal::new_no_comm()
        }
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => {
            let send = [local];
            let mut recv = send;
            let req = inner.immediate_allreduce_sum(&send[..], &mut recv[..]);
            ReduceReqReal::new_mpi(send, recv, out, req)
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => {
            *out = local;
            ReduceReqReal::new_no_comm()
        }
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => {
            *out = local;
            ReduceReqReal::new_no_comm()
        }
    }
}

pub(crate) fn iallreduce_tuple2<'a>(
    comm: &UniverseComm,
    a_local: S,
    b_local: R,
    out_a: &'a mut S,
    out_b: &'a mut R,
) -> ReduceReqTuple2<'a> {
    match comm {
        UniverseComm::NoComm(_) => {
            *out_a = a_local;
            *out_b = b_local;
            ReduceReqTuple2::new_no_comm()
        }
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => {
            let mut recv = Vec::<R>::with_capacity(3);
            recv.extend_from_slice(&crate::parallel::reduce::pack_scalar_s_to_rr(a_local));
            recv.push(b_local);
            let send = recv.clone();
            let req = inner.immediate_allreduce_sum(&send[..], recv.as_mut_slice());
            ReduceReqTuple2::new_mpi(send, recv, out_a, out_b, req)
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => {
            *out_a = a_local;
            *out_b = b_local;
            ReduceReqTuple2::new_no_comm()
        }
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => {
            *out_a = a_local;
            *out_b = b_local;
            ReduceReqTuple2::new_no_comm()
        }
    }
}
