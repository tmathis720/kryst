#[cfg(feature = "metrics")]
mod enabled {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    pub struct Counters {
        pub total_nanos: AtomicU64,
        pub count: AtomicU64,
        pub last_nanos: AtomicU64,
    }

    impl Counters {
        pub const fn new() -> Self {
            Self {
                total_nanos: AtomicU64::new(0),
                count: AtomicU64::new(0),
                last_nanos: AtomicU64::new(0),
            }
        }

        #[inline]
        pub fn snapshot(&self) -> (u64, u64, u64) {
            (
                self.total_nanos.load(Ordering::Relaxed),
                self.count.load(Ordering::Relaxed),
                self.last_nanos.load(Ordering::Relaxed),
            )
        }
    }

    pub struct SolveTimer<'a> {
        start: Instant,
        ctrs: &'a Counters,
    }

    impl<'a> SolveTimer<'a> {
        #[inline]
        pub fn start(ctrs: &'a Counters) -> Self {
            Self {
                start: Instant::now(),
                ctrs,
            }
        }

        #[inline]
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
    }

    impl<'a> Drop for SolveTimer<'a> {
        #[inline]
        fn drop(&mut self) {
            let dt = self.start.elapsed().as_nanos() as u64;
            self.ctrs.total_nanos.fetch_add(dt, Ordering::Relaxed);
            self.ctrs.count.fetch_add(1, Ordering::Relaxed);
            self.ctrs.last_nanos.store(dt, Ordering::Relaxed);
        }
    }
}

#[cfg(feature = "metrics")]
pub use enabled::*;

#[cfg(not(feature = "metrics"))]
mod disabled {
    use std::time::Duration;

    pub struct Counters;
    impl Default for Counters {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Counters {
        pub const fn new() -> Self {
            Self
        }
        pub fn snapshot(&self) -> (u64, u64, u64) {
            (0, 0, 0)
        }
    }

    pub struct SolveTimer<'a> {
        _p: core::marker::PhantomData<&'a ()>,
    }
    impl<'a> SolveTimer<'a> {
        #[inline]
        pub fn start(_: &'a Counters) -> Self {
            Self {
                _p: core::marker::PhantomData,
            }
        }
        #[inline]
        pub fn elapsed(&self) -> Duration {
            Duration::ZERO
        }
    }
    impl<'a> Drop for SolveTimer<'a> {
        fn drop(&mut self) {}
    }
}

#[cfg(not(feature = "metrics"))]
pub use disabled::*;

/// Estimate the error of a preconditioner using random Rademacher probes.
///
/// The supplied closure should compute `M^{-1} A z` for an input vector `z`.
/// The function returns the maximum sampled value of `|| (I - M^{-1}A) z ||_2 / ||z||_2`.
pub fn proxy_operator_error<M: Fn(&[f64], &mut [f64])>(
    apply_m_inv_a: M,
    n: usize,
    samples: usize,
    seed: u64,
) -> f64 {
    let mut rng = oorandom::Rand64::new(seed.into());
    let mut z = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];
    let mut worst = 0.0f64;
    for _ in 0..samples {
        for i in 0..n {
            z[i] = if rng.rand_u64() & 1 == 1 { 1.0 } else { -1.0 };
        }
        apply_m_inv_a(&z, &mut w);
        for i in 0..n {
            w[i] = z[i] - w[i];
        }
        let r = (w.iter().map(|x| x * x).sum::<f64>()).sqrt() / (n as f64).sqrt();
        worst = worst.max(r);
    }
    worst
}
