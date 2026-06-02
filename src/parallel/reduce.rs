use crate::algebra::parallel::{
    dot_conj_local_repro, par_dot_conj_local, par_sum_abs2_local, sum_abs2_local_repro,
};
use crate::algebra::prelude::*;
use crate::parallel::{Comm, UniverseComm};
use crate::reduction::{CommDeterministic, Packet, ReproMode};
use crate::utils::reduction::set_current_repro_mode;
use core::sync::atomic::{AtomicU8, Ordering};
use smallvec::SmallVec;

#[cfg(feature = "mpi")]
use core::ffi::c_void;
#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi")]
use mpi::traits::CommunicatorCollectives;
#[cfg(feature = "mpi")]
use mpi::{ffi, raw::AsRaw};

#[cfg(feature = "complex")]
#[inline]
pub(crate) fn pack_scalar_s_to_rr(v: S) -> [R; 2] {
    [v.real(), v.imag()]
}

#[cfg(feature = "complex")]
#[inline]
pub(crate) fn unpack_rr_to_scalar_s(rr: [R; 2]) -> S {
    S::from_parts(rr[0], rr[1])
}

#[cfg(not(feature = "complex"))]
#[allow(dead_code)]
#[inline]
pub(crate) fn pack_scalar_s_to_rr(v: S) -> [R; 1] {
    [v.real()]
}

#[cfg(not(feature = "complex"))]
#[allow(dead_code)]
#[inline]
pub(crate) fn unpack_rr_to_scalar_s(rr: [R; 1]) -> S {
    S::from_real(rr[0])
}

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

pub(crate) fn allreduce_rr_in_place(comm: &UniverseComm, rr: &mut [R]) {
    if rr.is_empty() || comm.size() <= 1 {
        return;
    }

    match comm {
        UniverseComm::NoComm(_) => {}
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => inner.blocking_allreduce_sum_in_place(rr),
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => {}
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => {}
    }
}

pub(crate) fn allreduce_vec_s_in_place(comm: &UniverseComm, s: &mut [S]) {
    if s.is_empty() || comm.size() <= 1 {
        return;
    }

    match comm {
        UniverseComm::NoComm(_) => {}
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(_) => {
            #[cfg(feature = "complex")]
            {
                let mut tmp: Vec<R> = Vec::with_capacity(s.len() * 2);
                for &value in s.iter() {
                    tmp.extend_from_slice(&pack_scalar_s_to_rr(value));
                }
                allreduce_rr_in_place(comm, tmp.as_mut_slice());
                for (slot, chunk) in s.iter_mut().zip(tmp.chunks_exact(2)) {
                    *slot = unpack_rr_to_scalar_s([chunk[0], chunk[1]]);
                }
            }

            #[cfg(not(feature = "complex"))]
            {
                let mut tmp: Vec<R> = s.iter().map(|&value| value.real()).collect();
                allreduce_rr_in_place(comm, tmp.as_mut_slice());
                for (slot, &value) in s.iter_mut().zip(tmp.iter()) {
                    *slot = S::from_real(value);
                }
            }
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => {}
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => {}
    }
}

#[inline]
pub(crate) fn allreduce_sum_scalar_impl(comm: &UniverseComm, z: S) -> S {
    match comm {
        UniverseComm::NoComm(_) => z,
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(_) => {
            let world = comm
                .as_mpi()
                .expect("MPI communicator should be available for this variant");
            #[cfg(feature = "complex")]
            {
                let send = pack_scalar(z);
                let mut recv = [0.0f64; 2];
                world.all_reduce_into(&send, &mut recv, SystemOperation::sum());
                unpack_scalar(recv)
            }
            #[cfg(not(feature = "complex"))]
            {
                let mut out = z;
                world.all_reduce_into(&z, &mut out, SystemOperation::sum());
                out
            }
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => z,
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => z,
    }
}

#[cfg(feature = "mpi")]
#[inline]
pub fn allreduce_sum_scalar_mpi_sys(comm: &UniverseComm, z: S) -> S {
    if comm.size() <= 1 {
        return z;
    }

    let Some(world) = comm.as_mpi() else {
        return z;
    };

    unsafe { mpi_allreduce_sum_scalar_raw(world, z) }
}

#[cfg(feature = "mpi")]
unsafe fn mpi_allreduce_sum_scalar_raw(world: &mpi::topology::SimpleCommunicator, z: S) -> S {
    let raw_comm = world.as_raw();

    #[cfg(not(feature = "complex"))]
    {
        let send = [z.real()];
        let mut recv = [0.0f64; 1];
        let datatype = unsafe { ffi::RSMPI_DOUBLE };
        let op = unsafe { ffi::RSMPI_SUM };
        let status = unsafe {
            ffi::MPI_Allreduce(
                send.as_ptr() as *const c_void,
                recv.as_mut_ptr() as *mut c_void,
                1,
                datatype,
                op,
                raw_comm,
            )
        };
        debug_assert_eq!(status, 0);
        S::from_real(recv[0])
    }

    #[cfg(feature = "complex")]
    {
        let send = [z.real(), z.imag()];
        let mut recv = [0.0f64; 2];
        let datatype = unsafe { ffi::RSMPI_DOUBLE };
        let op = unsafe { ffi::RSMPI_SUM };
        let status = unsafe {
            ffi::MPI_Allreduce(
                send.as_ptr() as *const c_void,
                recv.as_mut_ptr() as *mut c_void,
                2,
                datatype,
                op,
                raw_comm,
            )
        };
        debug_assert_eq!(status, 0);
        S::from_parts(recv[0], recv[1])
    }
}

#[inline]
pub(crate) fn allreduce_sum_scalar_repro_impl(comm: &UniverseComm, z: S, mode: ReproMode) -> S {
    if comm.size() <= 1 {
        return z;
    }

    let mode = effective_mode(comm, mode);
    match mode {
        ReproMode::Fast => comm.allreduce_sum_scalar(z),
        ReproMode::Deterministic | ReproMode::DeterministicAccurate => {
            comm.reduce_sum_scalar_s_repro(z)
        }
    }
}

const MODE_FAST: u8 = 0;
const MODE_DETERMINISTIC: u8 = 1;
const MODE_DETERMINISTIC_ACCURATE: u8 = 2;

static GLOBAL_REDUCTION_MODE: AtomicU8 = AtomicU8::new(MODE_FAST);

/// Guard that restores the previous global reduction mode when dropped.
#[derive(Debug)]
pub struct GlobalReductionModeGuard {
    previous: ReproMode,
}

impl Drop for GlobalReductionModeGuard {
    fn drop(&mut self) {
        set_global_reduction_mode(self.previous);
    }
}

#[inline]
fn encode_mode(mode: ReproMode) -> u8 {
    match mode {
        ReproMode::Fast => MODE_FAST,
        ReproMode::Deterministic => MODE_DETERMINISTIC,
        ReproMode::DeterministicAccurate => MODE_DETERMINISTIC_ACCURATE,
    }
}

#[inline]
fn decode_mode(tag: u8) -> ReproMode {
    match tag {
        MODE_DETERMINISTIC => ReproMode::Deterministic,
        MODE_DETERMINISTIC_ACCURATE => ReproMode::DeterministicAccurate,
        _ => ReproMode::Fast,
    }
}

/// Update the process-wide reduction mode used by the global helpers.
#[inline]
pub fn set_global_reduction_mode(mode: ReproMode) {
    GLOBAL_REDUCTION_MODE.store(encode_mode(mode), Ordering::Relaxed);
    set_current_repro_mode(mode);
}

/// Current process-wide reduction mode.
#[inline]
pub fn global_reduction_mode() -> ReproMode {
    decode_mode(GLOBAL_REDUCTION_MODE.load(Ordering::Relaxed))
}

/// Temporarily install a new global reduction mode, restoring the previous
/// configuration when the returned guard is dropped.
#[inline]
pub fn set_global_reduction_mode_scoped(mode: ReproMode) -> GlobalReductionModeGuard {
    let prev = global_reduction_mode();
    set_global_reduction_mode(mode);
    GlobalReductionModeGuard { previous: prev }
}

#[inline]
fn effective_mode(comm: &UniverseComm, mode: ReproMode) -> ReproMode {
    if comm.is_reproducible() && matches!(mode, ReproMode::Fast) {
        ReproMode::Deterministic
    } else {
        mode
    }
}

/// Sum a single scalar across all ranks using the requested reproducibility mode.
#[inline]
pub fn allreduce_sum_scalar_with_mode(comm: &UniverseComm, z: S, mode: ReproMode) -> S {
    let mode = effective_mode(comm, mode);
    match mode {
        ReproMode::Fast => comm.allreduce_sum_scalar(z),
        ReproMode::Deterministic | ReproMode::DeterministicAccurate => {
            comm.allreduce_sum_scalar_repro_with_mode(z, mode)
        }
    }
}

#[inline]
pub fn allreduce_sum_real_with_mode(comm: &UniverseComm, v: R, mode: ReproMode) -> R {
    let mode = effective_mode(comm, mode);
    match mode {
        ReproMode::Fast => comm.allreduce_sum_real(v),
        ReproMode::Deterministic | ReproMode::DeterministicAccurate => {
            comm.reduce_sum_real_repro(v)
        }
    }
}

/// Global conjugated dot product across all ranks using the requested mode.
#[inline]
pub fn global_dot_conj_with_mode(comm: &UniverseComm, x: &[S], y: &[S], mode: ReproMode) -> S {
    assert_eq!(
        x.len(),
        y.len(),
        "global_dot_conj length mismatch: {} vs {}",
        x.len(),
        y.len()
    );
    let mode = effective_mode(comm, mode);
    let local = if matches!(mode, ReproMode::Fast) {
        par_dot_conj_local(x, y)
    } else {
        dot_conj_local_repro(x, y)
    };
    allreduce_sum_scalar_with_mode(comm, local, mode)
}

/// Compute multiple global conjugated dot products in a single collective.
#[inline]
pub fn global_dot_conj_many_with_mode(
    comm: &UniverseComm,
    pairs: &[(&[S], &[S])],
    mode: ReproMode,
) -> Vec<S> {
    if pairs.is_empty() {
        return Vec::new();
    }

    let mut results: SmallVec<[S; 8]> = SmallVec::with_capacity(pairs.len());
    results.resize(pairs.len(), S::zero());
    global_dot_conj_many_into_with_mode(comm, pairs, results.as_mut_slice(), mode);
    results.into_vec()
}

/// Compute multiple global conjugated dot products, writing the results into the
/// provided output slice.
#[inline]
pub fn global_dot_conj_many_into_with_mode(
    comm: &UniverseComm,
    pairs: &[(&[S], &[S])],
    out: &mut [S],
    mode: ReproMode,
) {
    assert_eq!(
        pairs.len(),
        out.len(),
        "global_dot_conj_many_into output length mismatch: {} pairs for {} slots",
        pairs.len(),
        out.len()
    );

    if pairs.is_empty() {
        return;
    }

    let mode = effective_mode(comm, mode);

    for (idx, ((x, y), slot)) in pairs.iter().zip(out.iter_mut()).enumerate() {
        assert_eq!(
            x.len(),
            y.len(),
            "global_dot_conj_many length mismatch at pair {}: {} vs {}",
            idx,
            x.len(),
            y.len()
        );
        *slot = if matches!(mode, ReproMode::Fast) {
            par_dot_conj_local(x, y)
        } else {
            dot_conj_local_repro(x, y)
        };
    }

    match mode {
        ReproMode::Fast => allreduce_vec_s_in_place(comm, out),
        ReproMode::Deterministic | ReproMode::DeterministicAccurate => {
            allreduce_sum_scalar_slice_with_mode(comm, out, mode)
        }
    }
}

/// Global Euclidean norm of a vector across all ranks using the requested mode.
#[inline]
pub fn global_nrm2_with_mode(comm: &UniverseComm, x: &[S], mode: ReproMode) -> R {
    let mode = effective_mode(comm, mode);
    let ssq = if matches!(mode, ReproMode::Fast) {
        par_sum_abs2_local(x)
    } else {
        sum_abs2_local_repro(x)
    };
    let global = allreduce_sum_real_with_mode(comm, ssq, mode);
    let clamped = if global >= 0.0 { global } else { 0.0 };
    clamped.sqrt()
}

/// Global Euclidean norm of a vector across all ranks.
#[inline]
pub fn global_nrm2(comm: &UniverseComm, x: &[S]) -> R {
    global_nrm2_with_mode(comm, x, global_reduction_mode())
}

/// Reduce a scalar and a real value together using a single collective.
#[inline]
pub fn global_reduce_tuple2(comm: &UniverseComm, a: S, b: R) -> (S, R) {
    #[cfg(feature = "complex")]
    let mut packed = [a.real(), a.imag(), b];
    #[cfg(not(feature = "complex"))]
    let mut packed = [a.real(), b];
    allreduce_rr_in_place(comm, packed.as_mut_slice());

    #[cfg(feature = "complex")]
    {
        (S::from_parts(packed[0], packed[1]), packed[2])
    }

    #[cfg(not(feature = "complex"))]
    {
        (S::from_real(packed[0]), packed[1])
    }
}

/// Accurate global Euclidean norm of a vector across all ranks.
#[inline]
pub fn global_nrm2_accurate(comm: &UniverseComm, x: &[S]) -> R {
    global_nrm2_with_mode(comm, x, ReproMode::DeterministicAccurate)
}

/// Deterministic global Euclidean norm of a vector across all ranks.
#[inline]
pub fn global_nrm2_repro(comm: &UniverseComm, x: &[S]) -> R {
    global_nrm2_with_mode(comm, x, ReproMode::Deterministic)
}

#[inline]
fn clamp_and_sqrt(values: &mut [R]) {
    for value in values.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
        *value = (*value).sqrt();
    }
}

/// Compute multiple global Euclidean norms using the requested mode.
#[inline]
pub fn global_nrm2_many_with_mode(comm: &UniverseComm, vecs: &[&[S]], mode: ReproMode) -> Vec<R> {
    if vecs.is_empty() {
        return Vec::new();
    }

    let mode = effective_mode(comm, mode);
    let mut sums: Vec<R> = vec![R::zero(); vecs.len()];
    for (slot, &vec) in sums.iter_mut().zip(vecs.iter()) {
        *slot = if matches!(mode, ReproMode::Fast) {
            par_sum_abs2_local(vec)
        } else {
            sum_abs2_local_repro(vec)
        };
    }

    allreduce_sum_real_slice_with_mode(comm, sums.as_mut_slice(), mode);
    clamp_and_sqrt(sums.as_mut_slice());
    sums
}

/// Compute multiple global Euclidean norms into an output slice using the requested mode.
#[inline]
pub fn global_nrm2_many_into_with_mode(
    comm: &UniverseComm,
    vecs: &[&[S]],
    out: &mut [R],
    mode: ReproMode,
) {
    assert_eq!(
        vecs.len(),
        out.len(),
        "global_nrm2_many_into output length mismatch: {} vectors for {} slots",
        vecs.len(),
        out.len()
    );

    if vecs.is_empty() {
        return;
    }

    let mode = effective_mode(comm, mode);
    for (slot, &vec) in out.iter_mut().zip(vecs.iter()) {
        *slot = if matches!(mode, ReproMode::Fast) {
            par_sum_abs2_local(vec)
        } else {
            sum_abs2_local_repro(vec)
        };
    }

    allreduce_sum_real_slice_with_mode(comm, out, mode);
    clamp_and_sqrt(out);
}

/// Compute multiple global Euclidean norms using the current global reduction mode.
#[inline]
pub fn global_nrm2_many(comm: &UniverseComm, vecs: &[&[S]]) -> Vec<R> {
    global_nrm2_many_with_mode(comm, vecs, global_reduction_mode())
}

/// Compute multiple accurate global Euclidean norms across all ranks.
#[inline]
pub fn global_nrm2_many_accurate(comm: &UniverseComm, vecs: &[&[S]]) -> Vec<R> {
    global_nrm2_many_with_mode(comm, vecs, ReproMode::DeterministicAccurate)
}

/// Compute multiple deterministic global Euclidean norms across all ranks.
#[inline]
pub fn global_nrm2_many_repro(comm: &UniverseComm, vecs: &[&[S]]) -> Vec<R> {
    global_nrm2_many_with_mode(comm, vecs, ReproMode::Deterministic)
}

/// Compute multiple global Euclidean norms into an output slice using the current mode.
#[inline]
pub fn global_nrm2_many_into(comm: &UniverseComm, vecs: &[&[S]], out: &mut [R]) {
    let mode = global_reduction_mode();
    global_nrm2_many_into_with_mode(comm, vecs, out, mode);
}

/// Compute multiple accurate global Euclidean norms into an output slice.
#[inline]
pub fn global_nrm2_many_into_accurate(comm: &UniverseComm, vecs: &[&[S]], out: &mut [R]) {
    global_nrm2_many_into_with_mode(comm, vecs, out, ReproMode::DeterministicAccurate);
}

/// Compute multiple deterministic global Euclidean norms into an output slice.
#[inline]
pub fn global_nrm2_many_into_repro(comm: &UniverseComm, vecs: &[&[S]], out: &mut [R]) {
    global_nrm2_many_into_with_mode(comm, vecs, out, ReproMode::Deterministic);
}

/// Global conjugated dot product across all ranks.
#[inline]
pub fn global_dot_conj(comm: &UniverseComm, x: &[S], y: &[S]) -> S {
    global_dot_conj_with_mode(comm, x, y, global_reduction_mode())
}

/// Accurate global conjugated dot product across all ranks.
#[inline]
pub fn global_dot_conj_accurate(comm: &UniverseComm, x: &[S], y: &[S]) -> S {
    global_dot_conj_with_mode(comm, x, y, ReproMode::DeterministicAccurate)
}

/// Deterministic global conjugated dot product across all ranks.
#[inline]
pub fn global_dot_conj_repro(comm: &UniverseComm, x: &[S], y: &[S]) -> S {
    global_dot_conj_with_mode(comm, x, y, ReproMode::Deterministic)
}

/// Compute multiple global conjugated dot products in the current global mode.
#[inline]
pub fn global_dot_conj_many(comm: &UniverseComm, pairs: &[(&[S], &[S])]) -> Vec<S> {
    global_dot_conj_many_with_mode(comm, pairs, global_reduction_mode())
}

/// Compute multiple accurate global conjugated dot products.
#[inline]
pub fn global_dot_conj_many_accurate(comm: &UniverseComm, pairs: &[(&[S], &[S])]) -> Vec<S> {
    global_dot_conj_many_with_mode(comm, pairs, ReproMode::DeterministicAccurate)
}

/// Compute multiple deterministic global conjugated dot products.
#[inline]
pub fn global_dot_conj_many_repro(comm: &UniverseComm, pairs: &[(&[S], &[S])]) -> Vec<S> {
    global_dot_conj_many_with_mode(comm, pairs, ReproMode::Deterministic)
}

/// Compute multiple global conjugated dot products into an output slice using
/// the current global reduction mode.
#[inline]
pub fn global_dot_conj_many_into(comm: &UniverseComm, pairs: &[(&[S], &[S])], out: &mut [S]) {
    let mode = global_reduction_mode();
    global_dot_conj_many_into_with_mode(comm, pairs, out, mode);
}

/// Compute multiple accurate global conjugated dot products into an output slice.
#[inline]
pub fn global_dot_conj_many_into_accurate(
    comm: &UniverseComm,
    pairs: &[(&[S], &[S])],
    out: &mut [S],
) {
    global_dot_conj_many_into_with_mode(comm, pairs, out, ReproMode::DeterministicAccurate);
}

/// Compute multiple deterministic global conjugated dot products into an output slice.
#[inline]
pub fn global_dot_conj_many_into_repro(comm: &UniverseComm, pairs: &[(&[S], &[S])], out: &mut [S]) {
    global_dot_conj_many_into_with_mode(comm, pairs, out, ReproMode::Deterministic);
}

fn allreduce_sum_scalar_slice_fast(comm: &UniverseComm, data: &mut [S]) {
    if data.is_empty() || comm.size() <= 1 {
        return;
    }

    match comm {
        UniverseComm::NoComm(_) => {}
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => {
            #[cfg(feature = "complex")]
            {
                let mut packed: SmallVec<[f64; 16]> = SmallVec::with_capacity(data.len() * 2);
                for &value in data.iter() {
                    let parts = pack_scalar(value);
                    packed.extend_from_slice(&parts);
                }
                inner.allreduce_sum_slice(packed.as_mut_slice());
                for (slot, chunk) in data.iter_mut().zip(packed.chunks_exact(2)) {
                    *slot = S::from_parts(chunk[0], chunk[1]);
                }
            }
            #[cfg(not(feature = "complex"))]
            {
                inner.allreduce_sum_slice(data);
            }
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => {}
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => {}
    }
}

fn allreduce_sum_real_slice_fast(comm: &UniverseComm, data: &mut [R]) {
    if data.is_empty() || comm.size() <= 1 {
        return;
    }

    match comm {
        UniverseComm::NoComm(_) => {}
        #[cfg(feature = "mpi")]
        UniverseComm::Mpi(inner) => {
            use mpi::collective::SystemOperation;
            let mut recv = vec![0.0f64; data.len()];
            inner
                .world
                .all_reduce_into(&data[..], &mut recv[..], SystemOperation::sum());
            data.copy_from_slice(&recv);
        }
        #[cfg(feature = "rayon")]
        UniverseComm::Rayon(_) => {}
        #[cfg(not(any(feature = "mpi", feature = "rayon")))]
        UniverseComm::Serial => {}
    }
}

#[inline]
pub fn allreduce_sum_real_slice_with_mode(comm: &UniverseComm, data: &mut [R], mode: ReproMode) {
    let mode = effective_mode(comm, mode);
    match mode {
        ReproMode::Fast => allreduce_sum_real_slice_fast(comm, data),
        ReproMode::Deterministic | ReproMode::DeterministicAccurate => {
            reduce_real_slice_deterministic(comm, data, mode);
        }
    }
}

fn reduce_real_slice_deterministic(comm: &UniverseComm, data: &mut [R], mode: ReproMode) {
    if data.is_empty() || comm.size() <= 1 {
        return;
    }

    let mut scratch: Vec<f64> = data.to_vec();
    reduce_buffer_in_packets(comm, scratch.as_mut_slice(), mode);
    data.copy_from_slice(&scratch);
}

/// Reduce a slice of scalars using the requested reproducibility mode.
#[inline]
pub fn allreduce_sum_scalar_slice_with_mode(comm: &UniverseComm, data: &mut [S], mode: ReproMode) {
    let mode = effective_mode(comm, mode);
    match mode {
        ReproMode::Fast => allreduce_sum_scalar_slice_fast(comm, data),
        ReproMode::Deterministic | ReproMode::DeterministicAccurate => {
            reduce_scalar_slice_deterministic(comm, data, mode);
        }
    }
}

/// Reduce a slice of scalars and return the summed values in a new buffer using the
/// requested reproducibility mode.
#[inline]
pub fn allreduce_sum_scalar_slice_owned_with_mode(
    comm: &UniverseComm,
    data: &[S],
    mode: ReproMode,
) -> Vec<S> {
    let mut out = Vec::from(data);
    allreduce_sum_scalar_slice_with_mode(comm, &mut out, mode);
    out
}

/// Reduce a slice of scalars and return the summed values in a new buffer.
#[inline]
pub fn allreduce_sum_scalar_slice_owned(comm: &UniverseComm, data: &[S]) -> Vec<S> {
    let mode = global_reduction_mode();
    allreduce_sum_scalar_slice_owned_with_mode(comm, data, mode)
}

fn reduce_scalar_slice_deterministic(comm: &UniverseComm, data: &mut [S], mode: ReproMode) {
    if data.is_empty() || comm.size() <= 1 {
        return;
    }
    match mode {
        ReproMode::Fast => {}
        ReproMode::Deterministic | ReproMode::DeterministicAccurate => {
            comm.reduce_sum_scalars_s_repro(data);
        }
    }
}

fn reduce_buffer_in_packets(comm: &UniverseComm, buf: &mut [f64], mode: ReproMode) {
    if buf.is_empty() {
        return;
    }

    let mut offset = 0;
    while offset < buf.len() {
        let remaining = buf.len() - offset;
        let width = remaining.min(4);
        match width {
            4 => {
                let mut packet = Packet::<4> { v: [0.0; 4] };
                packet.v.copy_from_slice(&buf[offset..offset + 4]);
                let reduced = comm.allreduce_det(&packet, mode);
                buf[offset..offset + 4].copy_from_slice(&reduced.v);
            }
            3 => {
                let mut packet = Packet::<3> { v: [0.0; 3] };
                packet.v.copy_from_slice(&buf[offset..offset + 3]);
                let reduced = comm.allreduce_det(&packet, mode);
                buf[offset..offset + 3].copy_from_slice(&reduced.v);
            }
            2 => {
                let mut packet = Packet::<2> { v: [0.0; 2] };
                packet.v.copy_from_slice(&buf[offset..offset + 2]);
                let reduced = comm.allreduce_det(&packet, mode);
                buf[offset..offset + 2].copy_from_slice(&reduced.v);
            }
            _ => {
                let mut packet = Packet::<1> { v: [0.0; 1] };
                packet.v[0] = buf[offset];
                let reduced = comm.allreduce_det(&packet, mode);
                buf[offset] = reduced.v[0];
            }
        }
        offset += width;
    }
}

/// Reduce a slice of scalars in place using the current global reduction mode.
#[inline]
pub fn allreduce_sum_scalar_slice_in_place(comm: &UniverseComm, data: &mut [S]) {
    let mode = global_reduction_mode();
    allreduce_sum_scalar_slice_with_mode(comm, data, mode);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::blas::dot_conj;
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
        assert_eq!(g, S::from_parts(2.5, -1.5));

        #[cfg(feature = "complex")]
        {
            assert!((g.imag() + 1.5).abs() < 1e-15);
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

    #[test]
    fn global_nrm2_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let vec = [S::from_real(3.0), S::from_real(4.0)];
        let norm = global_nrm2(&comm, &vec);
        assert!((norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn global_nrm2_repro_matches_fast_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let vec = [S::from_parts(1.0, 0.5), S::from_parts(-2.0, -0.25)];
        let fast = global_nrm2(&comm, &vec);
        let repro = global_nrm2_repro(&comm, &vec);
        assert!((fast - repro).abs() < 1e-15);
    }

    #[test]
    fn global_nrm2_accurate_matches_fast_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let vec = [S::from_parts(0.75, -0.5), S::from_parts(-1.25, 0.25)];
        let fast = global_nrm2(&comm, &vec);
        let accurate = global_nrm2_accurate(&comm, &vec);
        assert!((fast - accurate).abs() < 1e-15);
    }

    #[test]
    fn slice_reduction_respects_mode() {
        let comm = UniverseComm::NoComm(NoComm);
        let mut data_fast = [S::from_real(1.0), S::from_real(-2.0), S::from_real(3.5)];
        let mut data_det = data_fast;

        allreduce_sum_scalar_slice_with_mode(&comm, &mut data_fast, ReproMode::Fast);
        allreduce_sum_scalar_slice_with_mode(&comm, &mut data_det, ReproMode::Deterministic);

        assert_eq!(data_fast, data_det);
    }

    #[test]
    fn global_mode_controls_slice_reduction() {
        let comm = UniverseComm::NoComm(NoComm);
        let _guard = set_global_reduction_mode_scoped(ReproMode::DeterministicAccurate);
        let mut data = [S::from_real(0.5), S::from_real(-1.5)];
        allreduce_sum_scalar_slice_in_place(&comm, &mut data);
        assert_eq!(data, [S::from_real(0.5), S::from_real(-1.5)]);
    }

    #[test]
    #[should_panic(expected = "global_dot_conj length mismatch")]
    fn global_dot_conj_rejects_mismatched_lengths() {
        let comm = UniverseComm::NoComm(NoComm);
        let x = [S::from_real(1.0)];
        let y = [S::from_real(1.0), S::from_real(2.0)];
        let _ = global_dot_conj(&comm, &x, &y);
    }

    #[test]
    fn global_dot_conj_accurate_matches_fast_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let x = [S::from_parts(0.5, -0.25), S::from_parts(1.5, 0.75)];
        let y = [S::from_parts(1.0, 0.5), S::from_parts(-0.5, -0.5)];
        let fast = global_dot_conj(&comm, &x, &y);
        let accurate = global_dot_conj_accurate(&comm, &x, &y);
        assert_eq!(fast, accurate);
    }

    #[test]
    fn scoped_reduction_mode_guard_restores_state() {
        let comm = UniverseComm::NoComm(NoComm);
        let prev = global_reduction_mode();
        set_global_reduction_mode(ReproMode::Fast);
        {
            let _guard = set_global_reduction_mode_scoped(ReproMode::Deterministic);
            assert_eq!(global_reduction_mode(), ReproMode::Deterministic);

            let mut values = [S::from_real(1.0)];
            allreduce_sum_scalar_slice_in_place(&comm, &mut values);
            assert_eq!(values, [S::from_real(1.0)]);
        }
        assert_eq!(global_reduction_mode(), ReproMode::Fast);
        set_global_reduction_mode(prev);
    }

    #[test]
    fn owned_slice_reduction_single_rank_matches_input() {
        let comm = UniverseComm::NoComm(NoComm);
        let input = [
            S::from_parts(1.0, 0.25),
            S::from_parts(-2.0, 0.5),
            S::from_parts(0.75, -0.125),
        ];
        let summed = allreduce_sum_scalar_slice_owned(&comm, &input);
        assert_eq!(summed, input);

        let summed_det =
            allreduce_sum_scalar_slice_owned_with_mode(&comm, &input, ReproMode::Deterministic);
        assert_eq!(summed_det, input);
    }

    #[test]
    fn owned_slice_reduction_respects_global_mode() {
        let comm = UniverseComm::NoComm(NoComm);
        let prev = global_reduction_mode();
        set_global_reduction_mode(ReproMode::DeterministicAccurate);
        let input = [S::from_real(1.25), S::from_real(-0.75)];
        let summed = allreduce_sum_scalar_slice_owned(&comm, &input);
        assert_eq!(summed, input);
        set_global_reduction_mode(prev);
    }

    #[test]
    fn global_dot_conj_many_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let x0 = [S::from_real(1.0), S::from_real(2.0)];
        let y0 = [S::from_real(0.5), S::from_real(-1.0)];
        let x1 = [S::from_parts(0.75, 0.25)];
        let y1 = [S::from_parts(1.25, -0.5)];

        let results = global_dot_conj_many(&comm, &[(&x0, &y0), (&x1, &y1)]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], dot_conj(&x0, &y0));
        assert_eq!(results[1], dot_conj(&x1, &y1));

        let repro = global_dot_conj_many_repro(&comm, &[(&x0, &y0), (&x1, &y1)]);
        assert_eq!(repro, results);
    }

    #[test]
    fn global_dot_conj_many_into_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let x0 = [S::from_parts(0.25, 0.5), S::from_parts(1.0, -0.25)];
        let y0 = [S::from_parts(2.0, -1.5), S::from_parts(-0.5, 0.25)];
        let x1 = [S::from_parts(-1.5, 0.75), S::from_parts(0.0, -0.5)];
        let y1 = [S::from_parts(0.75, 0.5), S::from_parts(1.25, -0.75)];

        let mut out = [S::zero(), S::zero()];
        global_dot_conj_many_into(&comm, &[(&x0, &y0), (&x1, &y1)], &mut out);

        assert_eq!(out[0], dot_conj(&x0, &y0));
        assert_eq!(out[1], dot_conj(&x1, &y1));

        let mut repro = out;
        global_dot_conj_many_into_repro(&comm, &[(&x0, &y0), (&x1, &y1)], &mut repro);
        assert_eq!(out, repro);

        let mut accurate = out;
        global_dot_conj_many_into_accurate(&comm, &[(&x0, &y0), (&x1, &y1)], &mut accurate);
        assert_eq!(out, accurate);
    }

    #[test]
    fn global_nrm2_many_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let v0 = [S::from_real(3.0), S::from_real(4.0)];
        let v1 = [S::from_parts(1.0, 0.5), S::from_parts(-2.0, -0.25)];

        let norms = global_nrm2_many(&comm, &[&v0, &v1]);
        assert_eq!(norms.len(), 2);
        assert!((norms[0] - 5.0).abs() < 1e-12);

        let expected1 = global_nrm2(&comm, &v1);
        assert!((norms[1] - expected1).abs() < 1e-15);

        let repro = global_nrm2_many_repro(&comm, &[&v0, &v1]);
        assert_eq!(norms, repro);

        let accurate = global_nrm2_many_accurate(&comm, &[&v0, &v1]);
        assert_eq!(norms, accurate);
    }

    #[test]
    fn global_nrm2_many_into_single_rank() {
        let comm = UniverseComm::NoComm(NoComm);
        let v0 = [S::from_parts(0.5, -0.25), S::from_parts(1.5, 0.75)];
        let v1 = [S::from_parts(1.0, 0.5), S::from_parts(-0.5, -0.5)];

        let mut out = [0.0, 0.0];
        global_nrm2_many_into(&comm, &[&v0, &v1], &mut out);

        let expected0 = global_nrm2(&comm, &v0);
        let expected1 = global_nrm2(&comm, &v1);
        assert!((out[0] - expected0).abs() < 1e-15);
        assert!((out[1] - expected1).abs() < 1e-15);

        let mut repro = out;
        global_nrm2_many_into_repro(&comm, &[&v0, &v1], &mut repro);
        assert_eq!(repro, out);

        let mut accurate = out;
        global_nrm2_many_into_accurate(&comm, &[&v0, &v1], &mut accurate);
        assert_eq!(accurate, out);
    }
}
