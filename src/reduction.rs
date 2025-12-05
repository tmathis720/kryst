use crate::parallel::Comm;

#[cfg(feature = "mpi")]
use crate::parallel::MpiComm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReproMode {
    Fast,
    Deterministic,
    DeterministicAccurate,
}

#[derive(Clone, Copy, Debug)]
pub struct ReductionOptions {
    pub mode: ReproMode,
    pub single_thread_local: bool,
    pub chunk_len: usize,
    pub packet_width: usize,
}

impl Default for ReductionOptions {
    fn default() -> Self {
        Self {
            mode: ReproMode::Fast,
            single_thread_local: true,
            chunk_len: 32_768,
            packet_width: 1,
        }
    }
}

pub trait Accum {
    fn add(&mut self, x: f64);
    fn finish(self) -> f64;
}

#[derive(Clone, Copy)]
pub struct Kahan {
    pub sum: f64,
    pub c: f64,
}

impl Default for Kahan {
    fn default() -> Self {
        Self::new()
    }
}

impl Kahan {
    #[inline]
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }
}

impl Accum for Kahan {
    #[inline]
    fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }
    #[inline]
    fn finish(self) -> f64 {
        self.sum
    }
}

#[derive(Clone, Copy)]
pub struct DD {
    pub hi: f64,
    pub lo: f64,
}

impl Default for DD {
    fn default() -> Self {
        Self::new()
    }
}

impl DD {
    #[inline]
    pub fn new() -> Self {
        Self { hi: 0.0, lo: 0.0 }
    }
}

impl Accum for DD {
    #[inline]
    fn add(&mut self, x: f64) {
        let s = self.hi + x;
        let z = s - self.hi;
        let e = (self.hi - (s - z)) + (x - z) + self.lo;
        self.hi = s + e;
        self.lo = e - (self.hi - s);
    }
    #[inline]
    fn finish(self) -> f64 {
        self.hi + self.lo
    }
}

#[inline]
pub fn dot_local_slice(u: &[f64], v: &[f64], mode: ReproMode) -> f64 {
    debug_assert_eq!(u.len(), v.len());
    match mode {
        ReproMode::Fast => u.iter().zip(v).map(|(a, b)| a * b).sum(),
        ReproMode::Deterministic => {
            let mut acc = Kahan::new();
            for (&a, &b) in u.iter().zip(v) {
                acc.add(a * b);
            }
            acc.finish()
        }
        ReproMode::DeterministicAccurate => {
            let mut acc = DD::new();
            for (&a, &b) in u.iter().zip(v) {
                acc.add(a * b);
            }
            acc.finish()
        }
    }
}

#[allow(dead_code)]
pub fn dot_local_deterministic_parallel(
    u: &[f64],
    v: &[f64],
    _chunk_len: usize,
    mode: ReproMode,
) -> f64 {
    // Placeholder: currently executes serial deterministic dot.
    // A fully parallel deterministic implementation can be added later.
    dot_local_slice(u, v, mode)
}

#[repr(C)]
#[derive(Clone)]
pub struct Packet<const N: usize> {
    pub v: [f64; N],
}

impl<const N: usize> Default for Packet<N> {
    fn default() -> Self {
        Self { v: [0.0; N] }
    }
}

pub trait PacketAccum<const N: usize> {
    fn add(&mut self, x: &Packet<N>);
    fn finish(self) -> Packet<N>;
}

pub struct KahanP<const N: usize> {
    sum: [f64; N],
    c: [f64; N],
}

impl<const N: usize> Default for KahanP<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> KahanP<N> {
    pub fn new() -> Self {
        Self {
            sum: [0.0; N],
            c: [0.0; N],
        }
    }
}

impl<const N: usize> PacketAccum<N> for KahanP<N> {
    #[inline]
    fn add(&mut self, x: &Packet<N>) {
        for i in 0..N {
            let y = x.v[i] - self.c[i];
            let t = self.sum[i] + y;
            self.c[i] = (t - self.sum[i]) - y;
            self.sum[i] = t;
        }
    }
    #[inline]
    fn finish(self) -> Packet<N> {
        Packet { v: self.sum }
    }
}

pub struct DDP<const N: usize> {
    hi: [f64; N],
    lo: [f64; N],
}

impl<const N: usize> Default for DDP<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> DDP<N> {
    pub fn new() -> Self {
        Self {
            hi: [0.0; N],
            lo: [0.0; N],
        }
    }
}

impl<const N: usize> PacketAccum<N> for DDP<N> {
    #[inline]
    fn add(&mut self, x: &Packet<N>) {
        for i in 0..N {
            let s = self.hi[i] + x.v[i];
            let z = s - self.hi[i];
            let e = (self.hi[i] - (s - z)) + (x.v[i] - z) + self.lo[i];
            self.hi[i] = s + e;
            self.lo[i] = e - (self.hi[i] - s);
        }
    }
    #[inline]
    fn finish(self) -> Packet<N> {
        let mut out = [0.0f64; N];
        for i in 0..N {
            out[i] = self.hi[i] + self.lo[i];
        }
        Packet { v: out }
    }
}

pub trait CommDeterministic: Comm {
    fn allreduce_det<const N: usize>(&self, local: &Packet<N>, mode: ReproMode) -> Packet<N>;
}

fn allreduce_packet_det<const N: usize, C>(
    comm: &C,
    local: &Packet<N>,
    mode: ReproMode,
) -> Packet<N>
where
    C: Comm,
{
    if matches!(mode, ReproMode::Fast) {
        let mut tmp = local.clone();
        comm.allreduce_sum_slice(&mut tmp.v);
        return tmp;
    }

    let size = comm.size();
    let rank = comm.rank();
    if size == 1 {
        return local.clone();
    }

    if rank == 0 {
        match mode {
            ReproMode::DeterministicAccurate => {
                let mut acc = DDP::<N>::new();
                acc.add(local);
                for src in 1..size {
                    let mut buf = Packet::<N>::default();
                    {
                        let mut recv = comm.irecv_from(&mut buf.v, src as i32);
                        comm.wait_all(std::slice::from_mut(&mut recv));
                    }
                    acc.add(&buf);
                }
                let total = acc.finish();
                for dest in 1..size {
                    let mut send = comm.isend_to(&total.v, dest as i32);
                    comm.wait_all(std::slice::from_mut(&mut send));
                }
                total
            }
            _ => {
                let mut acc = KahanP::<N>::new();
                acc.add(local);
                for src in 1..size {
                    let mut buf = Packet::<N>::default();
                    {
                        let mut recv = comm.irecv_from(&mut buf.v, src as i32);
                        comm.wait_all(std::slice::from_mut(&mut recv));
                    }
                    acc.add(&buf);
                }
                let total = acc.finish();
                for dest in 1..size {
                    let mut send = comm.isend_to(&total.v, dest as i32);
                    comm.wait_all(std::slice::from_mut(&mut send));
                }
                total
            }
        }
    } else {
        let mut send = comm.isend_to(&local.v, 0);
        comm.wait_all(std::slice::from_mut(&mut send));
        let mut buf = Packet::<N>::default();
        {
            let mut recv = comm.irecv_from(&mut buf.v, 0);
            comm.wait_all(std::slice::from_mut(&mut recv));
        }
        buf
    }
}

use crate::parallel::{NoComm, UniverseComm};

impl CommDeterministic for UniverseComm {
    fn allreduce_det<const N: usize>(&self, local: &Packet<N>, mode: ReproMode) -> Packet<N> {
        allreduce_packet_det(self, local, mode)
    }
}

impl CommDeterministic for NoComm {
    fn allreduce_det<const N: usize>(&self, local: &Packet<N>, _mode: ReproMode) -> Packet<N> {
        local.clone()
    }
}

#[cfg(feature = "mpi")]
impl CommDeterministic for MpiComm {
    fn allreduce_det<const N: usize>(&self, local: &Packet<N>, mode: ReproMode) -> Packet<N> {
        allreduce_packet_det(self, local, mode)
    }
}

#[derive(Default)]
pub struct DotEngine {
    pub opts: ReductionOptions,
}

impl DotEngine {
    pub fn dot<C: Comm + CommDeterministic>(&self, u: &[f64], v: &[f64], comm: &C) -> f64 {
        let packet = Packet::<1> {
            v: [self.dot_local(u, v)],
        };
        let g = comm.allreduce_det(&packet, self.opts.mode);
        g.v[0]
    }

    pub fn dot2<C: Comm + CommDeterministic>(&self, a: f64, b: f64, comm: &C) -> (f64, f64) {
        let packet = Packet::<2> { v: [a, b] };
        let g = comm.allreduce_det(&packet, self.opts.mode);
        (g.v[0], g.v[1])
    }

    fn dot_local(&self, u: &[f64], v: &[f64]) -> f64 {
        if self.opts.mode == ReproMode::Fast {
            u.iter().zip(v).map(|(a, b)| a * b).sum()
        } else if self.opts.single_thread_local {
            dot_local_slice(u, v, self.opts.mode)
        } else {
            dot_local_deterministic_parallel(u, v, self.opts.chunk_len, self.opts.mode)
        }
    }

    pub fn dot_many_into<C: Comm + CommDeterministic>(
        &self,
        pairs: &[(&[f64], &[f64])],
        out: &mut [f64],
        comm: &C,
    ) {
        if pairs.len() != out.len() {
            panic!(
                "dot_many_into length mismatch: {} pairs for {} slots",
                pairs.len(),
                out.len()
            );
        }
        if pairs.is_empty() {
            return;
        }

        for ((u, v), slot) in pairs.iter().zip(out.iter_mut()) {
            if u.len() != v.len() {
                panic!(
                    "dot_many_into vector length mismatch: {} vs {}",
                    u.len(),
                    v.len()
                );
            }
            *slot = self.dot_local(u, v);
        }

        let width = self.opts.packet_width.max(1).min(4);
        let mode = self.opts.mode;
        let mut idx = 0;
        while idx < out.len() {
            let chunk_len = (out.len() - idx).min(width);
            match chunk_len {
                1 => {
                    let packet = Packet::<1> { v: [out[idx]] };
                    let reduced = comm.allreduce_det(&packet, mode);
                    out[idx] = reduced.v[0];
                }
                2 => {
                    let packet = Packet::<2> {
                        v: [out[idx], out[idx + 1]],
                    };
                    let reduced = comm.allreduce_det(&packet, mode);
                    out[idx..idx + 2].copy_from_slice(&reduced.v);
                }
                3 => {
                    let packet = Packet::<3> {
                        v: [out[idx], out[idx + 1], out[idx + 2]],
                    };
                    let reduced = comm.allreduce_det(&packet, mode);
                    out[idx..idx + 3].copy_from_slice(&reduced.v);
                }
                _ => {
                    let packet = Packet::<4> {
                        v: [out[idx], out[idx + 1], out[idx + 2], out[idx + 3]],
                    };
                    let reduced = comm.allreduce_det(&packet, mode);
                    out[idx..idx + 4].copy_from_slice(&reduced.v);
                }
            }
            idx += chunk_len;
        }
    }

    pub fn dot_many<C: Comm + CommDeterministic>(
        &self,
        pairs: &[(&[f64], &[f64])],
        comm: &C,
    ) -> Vec<f64> {
        let mut out = vec![0.0; pairs.len()];
        self.dot_many_into(pairs, &mut out, comm);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::NoComm;

    #[test]
    fn dot_engine_many_matches_individual_dots() {
        let mut opts = ReductionOptions::default();
        opts.packet_width = 4;
        let engine = DotEngine { opts };
        let a = vec![1.0, -2.0, 3.5, 0.75];
        let b = vec![0.5, 1.5, -2.0, 4.0];
        let c = vec![1.25, -0.5, 3.0, -1.0];
        let pairs = [(&a[..], &b[..]), (&a[..], &c[..])];
        let mut out = vec![0.0; pairs.len()];
        engine.dot_many_into(&pairs, &mut out, &NoComm);

        let single_ab = engine.dot(&a, &b, &NoComm);
        let single_ac = engine.dot(&a, &c, &NoComm);
        assert!((out[0] - single_ab).abs() < 1e-12);
        assert!((out[1] - single_ac).abs() < 1e-12);
    }

    #[test]
    fn dot_engine_many_batches_more_than_packet_width() {
        let mut opts = ReductionOptions::default();
        opts.packet_width = 3;
        let engine = DotEngine { opts };

        let inputs: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64 + 1.0, (i as f64 - 0.5) * 0.75])
            .collect();
        let mut pairs = Vec::new();
        for idx in 0..inputs.len() {
            let next = (idx + 1) % inputs.len();
            pairs.push((&inputs[idx][..], &inputs[next][..]));
        }
        let mut out = vec![0.0; pairs.len()];
        engine.dot_many_into(pairs.as_slice(), &mut out, &NoComm);

        for (idx, (u, v)) in pairs.iter().enumerate() {
            let expected = engine.dot(u, v, &NoComm);
            assert!((out[idx] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn dot_engine_modes_match_dot_many_for_each_mode() {
        let a = vec![1.0, -2.0, 3.5, 0.75];
        let b = vec![0.5, 1.5, -2.0, 4.0];
        let c = vec![1.25, -0.5, 3.0, -1.0];
        let pairs = [(&a[..], &b[..]), (&a[..], &c[..])];
        let comm = NoComm;

        for &mode in &[
            ReproMode::Fast,
            ReproMode::Deterministic,
            ReproMode::DeterministicAccurate,
        ] {
            let mut opts = ReductionOptions::default();
            opts.packet_width = 4;
            opts.mode = mode;
            let engine = DotEngine { opts };

            let many = engine.dot_many(&pairs, &comm);
            assert_eq!(many.len(), pairs.len());
            for (idx, (u, v)) in pairs.iter().enumerate() {
                let single = engine.dot(u, v, &comm);
                assert!(
                    (single - many[idx]).abs() < 1e-12,
                    "mode {mode:?} mismatch at idx {idx}"
                );
            }
        }
    }

    #[test]
    fn deterministic_modes_resist_cancellation() {
        let u = [1e16, 1.0, -1e16];
        let v = [1.0, 1.0, 1.0];
        let comm = NoComm;

        let mut fast_opts = ReductionOptions::default();
        fast_opts.mode = ReproMode::Fast;
        let fast_engine = DotEngine { opts: fast_opts };
        let fast = fast_engine.dot(&u, &v, &comm);
        assert_eq!(fast, 0.0);

        let mut det_opts = ReductionOptions::default();
        det_opts.mode = ReproMode::Deterministic;
        let det_engine = DotEngine { opts: det_opts };
        let det = det_engine.dot(&u, &v, &comm);
        assert!((det - 1.0).abs() < 1e-12);
        let det_many = det_engine.dot_many(&[(&u, &v)], &comm);
        assert!((det_many[0] - det).abs() < 1e-12);

        let mut accurate_opts = ReductionOptions::default();
        accurate_opts.mode = ReproMode::DeterministicAccurate;
        let accurate_engine = DotEngine {
            opts: accurate_opts,
        };
        let accurate = accurate_engine.dot(&u, &v, &comm);
        assert!((accurate - 1.0).abs() < 1e-12);
        let accurate_many = accurate_engine.dot_many(&[(&u, &v)], &comm);
        assert!((accurate_many[0] - accurate).abs() < 1e-12);
    }
}
