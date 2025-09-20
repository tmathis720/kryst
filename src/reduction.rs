use crate::parallel::Comm;

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

pub trait CommDeterministic {
    fn allreduce_det<const N: usize>(&self, local: &Packet<N>, mode: ReproMode) -> Packet<N>;
}

use crate::parallel::UniverseComm;

impl CommDeterministic for UniverseComm {
    fn allreduce_det<const N: usize>(&self, local: &Packet<N>, mode: ReproMode) -> Packet<N> {
        if matches!(mode, ReproMode::Fast) {
            let mut tmp = local.clone();
            self.allreduce_sum_slice(&mut tmp.v);
            return tmp;
        }
        let size = self.size();
        let rank = self.rank();
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
                        let mut r = self.irecv_from(&mut buf.v, src as i32);
                        self.wait_all(std::slice::from_mut(&mut r));
                        acc.add(&buf);
                    }
                    let total = acc.finish();
                    for dest in 1..size {
                        let mut s = self.isend_to(&total.v, dest as i32);
                        self.wait_all(std::slice::from_mut(&mut s));
                    }
                    total
                }
                _ => {
                    let mut acc = KahanP::<N>::new();
                    acc.add(local);
                    for src in 1..size {
                        let mut buf = Packet::<N>::default();
                        let mut r = self.irecv_from(&mut buf.v, src as i32);
                        self.wait_all(std::slice::from_mut(&mut r));
                        acc.add(&buf);
                    }
                    let total = acc.finish();
                    for dest in 1..size {
                        let mut s = self.isend_to(&total.v, dest as i32);
                        self.wait_all(std::slice::from_mut(&mut s));
                    }
                    total
                }
            }
        } else {
            let mut s = self.isend_to(&local.v, 0);
            self.wait_all(std::slice::from_mut(&mut s));
            let mut buf = Packet::<N>::default();
            let mut r = self.irecv_from(&mut buf.v, 0);
            self.wait_all(std::slice::from_mut(&mut r));
            buf
        }
    }
}

#[derive(Default)]
pub struct DotEngine {
    pub opts: ReductionOptions,
}

impl DotEngine {
    pub fn dot<C: Comm + CommDeterministic>(&self, u: &[f64], v: &[f64], comm: &C) -> f64 {
        let local = if self.opts.mode == ReproMode::Fast {
            u.iter().zip(v).map(|(a, b)| a * b).sum()
        } else if self.opts.single_thread_local {
            dot_local_slice(u, v, self.opts.mode)
        } else {
            dot_local_deterministic_parallel(u, v, self.opts.chunk_len, self.opts.mode)
        };
        let packet = Packet::<1> { v: [local] };
        let g = comm.allreduce_det(&packet, self.opts.mode);
        g.v[0]
    }

    pub fn dot2<C: Comm + CommDeterministic>(&self, a: f64, b: f64, comm: &C) -> (f64, f64) {
        let packet = Packet::<2> { v: [a, b] };
        let g = comm.allreduce_det(&packet, self.opts.mode);
        (g.v[0], g.v[1])
    }
}
