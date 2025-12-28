use crate::algebra::parallel_cfg::serial_guard;
use crate::error::KError;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use once_cell::sync::OnceCell;

#[cfg(feature = "rayon")]
static SERIAL_POOL: OnceCell<rayon::ThreadPool> = OnceCell::new();

#[derive(Clone, Debug)]
pub enum ThreadingPolicy {
    /// Use Rayon global pool as-is (never reconfigure it).
    GlobalUnmodified,
    /// Run all parallel work inside this pool.
    #[cfg(feature = "rayon")]
    Pool(Arc<rayon::ThreadPool>),
    /// Force serial execution for Kryst-side parallel regions.
    Serial,
}

#[derive(Clone, Debug)]
pub struct ExecutionPolicy {
    pub threading: ThreadingPolicy,
    pub reproducible: bool,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            threading: ThreadingPolicy::GlobalUnmodified,
            reproducible: false,
        }
    }
}

impl ExecutionPolicy {
    pub fn with_reproducible(mut self, r: bool) -> Self {
        self.reproducible = r;
        self
    }

    #[cfg(feature = "rayon")]
    pub fn with_threads(mut self, n: usize) -> Result<Self, KError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| KError::InvalidInput(format!("rayon pool build failed: {e}")))?;
        self.threading = ThreadingPolicy::Pool(Arc::new(pool));
        Ok(self)
    }

    /// Run a closure under this policy (installs pool if present).
    pub fn install<T>(&self, f: impl FnOnce() -> T + Send) -> T
    where
        T: Send,
    {
        match &self.threading {
            ThreadingPolicy::Serial => {
                let _guard = serial_guard(true);
                #[cfg(feature = "rayon")]
                {
                    let pool = SERIAL_POOL.get_or_try_init(|| {
                        rayon::ThreadPoolBuilder::new().num_threads(1).build()
                    });
                    if let Ok(pool) = pool {
                        return pool.install(f);
                    }
                }
                f()
            }
            #[cfg(feature = "rayon")]
            ThreadingPolicy::Pool(pool) => pool.install(f),
            ThreadingPolicy::GlobalUnmodified => f(),
        }
    }
}
