use crate::algebra::parallel_cfg::serial_guard;
use crate::config::options::KspOptions;
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

/// Nested KSP execution policy for inner solver contexts (for example `pc_type=ksp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestedExecutionPolicy {
    /// Execute nested work in serial mode (safe fallback for MPI + threaded hosts).
    Serial,
    /// Reuse context-managed worker pools (or global executor when not configured).
    ContextPool,
    /// Use the global Rayon pool unchanged.
    Global,
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
    /// Resolve the execution policy for a nested KSP from parsed options.
    ///
    /// This centralizes nested MPI/thread semantics so all nested callers use the
    /// same precedence and validation:
    /// - `threads_mode=serial` always forces serial
    /// - `threads_mode=context` (or unset) selects context pools
    /// - `threads_mode=global` is allowed only for strictly local (non-MPI) solves
    /// - when `comm_size > 1` and `threads > 1`, force `serial` or return an error
    pub fn nested_from_options(opts: &KspOptions, comm_size: usize) -> Result<Self, KError> {
        let mode = opts.threads_mode.as_deref().unwrap_or("context");
        let policy = match mode {
            "serial" => NestedExecutionPolicy::Serial,
            "context" => NestedExecutionPolicy::ContextPool,
            "global" => {
                if comm_size > 1 {
                    return Err(KError::InvalidInput(
                        "nested pc_type=ksp with MPI does not allow ksp_threads_mode=global; use serial/context"
                            .into(),
                    ));
                }
                NestedExecutionPolicy::Global
            }
            other => {
                return Err(KError::InvalidInput(format!(
                    "unknown nested ksp_threads_mode: {other}"
                )));
            }
        };

        if comm_size > 1 && opts.threads.unwrap_or(1) > 1 && policy != NestedExecutionPolicy::Serial
        {
            return Err(KError::InvalidInput(
                "nested pc_type=ksp with MPI and threads>1 requires ksp_threads_mode=serial".into(),
            ));
        }

        let mut exec = ExecutionPolicy::default();
        match policy {
            NestedExecutionPolicy::Serial => {
                exec.threading = ThreadingPolicy::Serial;
            }
            NestedExecutionPolicy::ContextPool =>
            {
                #[cfg(feature = "rayon")]
                if let Some(n) = opts.threads {
                    exec = exec.with_threads(n)?;
                }
            }
            NestedExecutionPolicy::Global => {
                exec.threading = ThreadingPolicy::GlobalUnmodified;
            }
        }
        Ok(exec)
    }

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
                    let pool = SERIAL_POOL
                        .get_or_try_init(|| rayon::ThreadPoolBuilder::new().num_threads(1).build());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::options::KspOptions;

    #[test]
    fn nested_policy_rejects_global_with_mpi() {
        let opts = KspOptions {
            threads_mode: Some("global".into()),
            ..Default::default()
        };
        let err = ExecutionPolicy::nested_from_options(&opts, 2).unwrap_err();
        assert!(format!("{err}").contains("does not allow ksp_threads_mode=global"));
    }

    #[test]
    fn nested_policy_requires_serial_for_mpi_multithread() {
        let opts = KspOptions {
            threads_mode: Some("context".into()),
            threads: Some(4),
            ..Default::default()
        };
        let err = ExecutionPolicy::nested_from_options(&opts, 4).unwrap_err();
        assert!(format!("{err}").contains("requires ksp_threads_mode=serial"));
    }

    #[test]
    fn nested_policy_accepts_serial_for_mpi() {
        let opts = KspOptions {
            threads_mode: Some("serial".into()),
            threads: Some(8),
            ..Default::default()
        };
        let pol = ExecutionPolicy::nested_from_options(&opts, 8).unwrap();
        assert!(matches!(pol.threading, ThreadingPolicy::Serial));
    }
}
