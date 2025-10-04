use crate::algebra::blas::dot_conj;
use crate::algebra::prelude::*;
use crate::parallel::{Comm, UniverseComm};
use crate::reduction::{CommDeterministic, Packet, ReproMode};

#[cfg(feature = "complex")]
#[inline]
fn pack_scalar(z: S) -> [f64; 2] {
    [z.real(), z.imag()]
}

#[cfg(feature = "complex")]
#[inline]
fn unpack_scalar(parts: [f64; 2]) -> S {
    S::from_parts(parts[0], parts[1])
}

#[inline]
pub(crate) fn allreduce_sum_scalar_impl(comm: &UniverseComm, z: S) -> S {
    match comm {
        UniverseComm::NoComm(_) => z,
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => {
            #[cfg(feature = "complex")]
            {
                let parts = pack_scalar(z);
                let (re, im) = inner.allreduce_sum2(parts[0], parts[1]);
                S::from_parts(re, im)
            }
            #[cfg(not(feature = "complex"))]
            {
                inner.all_reduce_f64(z)
            }
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => z,
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => z,
    }
}

#[inline]
pub(crate) fn allreduce_sum_scalar_repro_impl(comm: &UniverseComm, z: S) -> S {
    #[cfg(feature = "complex")]
    {
        let packet = Packet::<2> {
            v: [z.real(), z.imag()],
        };
        let reduced = comm.allreduce_det(&packet, ReproMode::Deterministic);
        return S::from_parts(reduced.v[0], reduced.v[1]);
    }

    #[cfg(not(feature = "complex"))]
    {
        let packet = Packet::<1> { v: [z] };
        let reduced = comm.allreduce_det(&packet, ReproMode::Deterministic);
        return reduced.v[0];
    }
}

/// Global conjugated dot product across all ranks.
#[inline]
pub fn global_dot_conj(comm: &UniverseComm, x: &[S], y: &[S]) -> S {
    let local = dot_conj(x, y);
    comm.allreduce_sum_scalar(local)
}

/// Deterministic global conjugated dot product across all ranks.
#[inline]
pub fn global_dot_conj_repro(comm: &UniverseComm, x: &[S], y: &[S]) -> S {
    let local = dot_conj(x, y);
    comm.allreduce_sum_scalar_repro(local)
}

#[inline]
pub fn allreduce_sum_scalar_slice_in_place(comm: &UniverseComm, data: &mut [S]) {
    match comm {
        UniverseComm::NoComm(_) => {}
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => {
            if data.is_empty() {
                return;
            }
            #[cfg(feature = "complex")]
            {
                let mut packed = vec![0.0f64; data.len() * 2];
                for (idx, &value) in data.iter().enumerate() {
                    let parts = pack_scalar(value);
                    packed[2 * idx] = parts[0];
                    packed[2 * idx + 1] = parts[1];
                }
                inner.allreduce_sum_slice(&mut packed);
                for (idx, slot) in data.iter_mut().enumerate() {
                    *slot = S::from_parts(packed[2 * idx], packed[2 * idx + 1]);
                }
            }
            #[cfg(not(feature = "complex"))]
            {
                let slice: &mut [f64] = unsafe { &mut *(data as *mut [S] as *mut [f64]) };
                inner.allreduce_sum_slice(slice);
            }
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => {}
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::NoComm;

    #[test]
    fn allreduce_scalar_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let z = S::from_parts(1.25, 0.75);
        let out = comm.allreduce_sum_scalar(z);
        assert_eq!(out, z);

        #[cfg(feature = "complex")]
        {
            assert!((out.imag() - 0.75).abs() < 1e-15);
        }

        let g = global_dot_conj(&comm, &[z], &[S::from_real(2.0)]);
        assert_eq!(g, S::from_parts(2.5, 1.5));

        #[cfg(feature = "complex")]
        {
            assert!((g.imag() - 1.5).abs() < 1e-15);
        }
    }

    #[test]
    fn repro_matches_fast_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let z = S::from_parts(-0.5, 0.125);
        let fast = comm.allreduce_sum_scalar(z);
        let repro = comm.allreduce_sum_scalar_repro(z);
        assert_eq!(fast, repro);

        let dot_fast = global_dot_conj(&comm, &[z], &[S::one()]);
        let dot_repro = global_dot_conj_repro(&comm, &[z], &[S::one()]);
        assert_eq!(dot_fast, dot_repro);
    }
}
