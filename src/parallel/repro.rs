use crate::algebra::prelude::*;

#[cfg(feature = "mpi")]
use super::MpiComm;
#[cfg(feature = "mpi")]
use mpi::traits::CommunicatorCollectives;

#[cfg(feature = "complex")]
#[inline]
pub fn pack_scalar_s(value: S) -> [R; 2] {
    [value.real(), value.imag()]
}

#[cfg(not(feature = "complex"))]
#[inline]
pub fn pack_scalar_s(value: S) -> [R; 1] {
    [value.real()]
}

#[cfg(feature = "complex")]
#[inline]
pub fn unpack_scalar_s(parts: [R; 2]) -> S {
    S::from_parts(parts[0], parts[1])
}

#[cfg(not(feature = "complex"))]
#[inline]
pub fn unpack_scalar_s(parts: [R; 1]) -> S {
    S::from_real(parts[0])
}

#[cfg(feature = "mpi")]
#[inline]
pub fn reduce_sum_real_rank_ordered(comm: &MpiComm, local: R) -> R {
    if comm.size <= 1 {
        return local;
    }

    let mut gathered = vec![0.0f64; comm.size];
    comm.world.all_gather_into(&local, gathered.as_mut_slice());
    let mut acc = 0.0f64;
    for value in gathered {
        acc += value;
    }
    acc
}

#[cfg(feature = "mpi")]
#[inline]
pub fn reduce_sum_scalar_rank_ordered(comm: &MpiComm, local: S) -> S {
    if comm.size <= 1 {
        return local;
    }

    #[cfg(feature = "complex")]
    const WIDTH: usize = 2;
    #[cfg(not(feature = "complex"))]
    const WIDTH: usize = 1;

    let packed = pack_scalar_s(local);
    let mut gathered = vec![0.0f64; WIDTH * comm.size];
    comm.world
        .all_gather_into(&packed[..], gathered.as_mut_slice());

    let mut acc = [0.0f64; WIDTH];
    for rank in 0..comm.size {
        for lane in 0..WIDTH {
            acc[lane] += gathered[WIDTH * rank + lane];
        }
    }

    #[cfg(feature = "complex")]
    {
        unpack_scalar_s([acc[0], acc[1]])
    }

    #[cfg(not(feature = "complex"))]
    {
        unpack_scalar_s([acc[0]])
    }
}

#[cfg(feature = "mpi")]
#[inline]
pub fn reduce_sum_scalars_rank_ordered(comm: &MpiComm, locals: &mut [S]) {
    if locals.is_empty() || comm.size <= 1 {
        return;
    }

    #[cfg(feature = "complex")]
    const WIDTH: usize = 2;
    #[cfg(not(feature = "complex"))]
    const WIDTH: usize = 1;

    let n = locals.len();
    let mut send = vec![0.0f64; WIDTH * n];
    for (idx, value) in locals.iter().enumerate() {
        let packed = pack_scalar_s(*value);
        for lane in 0..WIDTH {
            send[WIDTH * idx + lane] = packed[lane];
        }
    }

    let mut gathered = vec![0.0f64; WIDTH * n * comm.size];
    comm.world
        .all_gather_into(&send[..], gathered.as_mut_slice());

    for elem in 0..n {
        let mut acc = [0.0f64; WIDTH];
        for rank in 0..comm.size {
            for lane in 0..WIDTH {
                let offset = WIDTH * n * rank + WIDTH * elem + lane;
                acc[lane] += gathered[offset];
            }
        }
        locals[elem] = unpack_scalar_s(acc);
    }
}
