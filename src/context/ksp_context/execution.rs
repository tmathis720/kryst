use crate::algebra::parallel_cfg::serial_guard;
use crate::algebra::parallel_cfg::{ParallelTune, parallel_tune};
use crate::config::options::KspOptions;
use crate::error::KError;
use crate::reduction::ReproMode;
use crate::utils::reduction::{ReductExec, ReductOptions};
use std::fmt;
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
    /// Run nested work with a bounded thread cap (MPI-safe compromise between serial and full parallel).
    Hybrid,
    /// Use the global Rayon pool unchanged.
    Global,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NestedPolicyContext<'a> {
    pub outer_threads: Option<usize>,
    pub outer_threads_mode: Option<&'a str>,
}

#[derive(Clone, Debug)]
pub struct ExecutionPolicy {
    pub threading: ThreadingPolicy,
    pub reproducible: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KrylovVariant {
    Classical,
    Pipelined,
    SStep,
}

#[derive(Clone, Debug)]
pub struct AutoExecutionReport {
    pub comm_size: usize,
    pub local_work: usize,
    pub restart_len: usize,
    pub reduction_latency_us: f64,
    pub variant: KrylovVariant,
    pub reduction: ReproMode,
    pub reduction_exec: ReductExec,
    pub overlap: OverlapStrategy,
    pub threading: &'static str,
    pub threads: usize,
}

impl AutoExecutionReport {
    pub fn concise(&self) -> String {
        format!(
            "ExecutionPolicy::Auto(variant={:?}, reduction={:?}/{:?}, overlap={:?}, threads={}:{}, comm={}, work={}, restart={}, latency_us={:.2})",
            self.variant,
            self.reduction,
            self.reduction_exec,
            self.overlap,
            self.threading,
            self.threads,
            self.comm_size,
            self.local_work,
            self.restart_len,
            self.reduction_latency_us
        )
    }
}

impl fmt::Display for AutoExecutionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.concise())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapStrategy {
    Disabled,
    Conservative,
    Aggressive,
}

#[derive(Clone, Debug)]
pub struct AdaptiveExecutionDecision {
    pub problem_size: usize,
    pub comm_size: usize,
    pub restart_len: usize,
    pub local_work: usize,
    pub reduction_latency_us: f64,
    pub tune: ParallelTune,
    pub threading: &'static str,
    pub recommended_threads: usize,
    pub threading_reason: &'static str,
    pub requested_reduction: ReproMode,
    pub selected_reduction: ReproMode,
    pub reduction_exec: ReductExec,
    pub variant: KrylovVariant,
    pub sstep_block: Option<usize>,
    pub overlap: OverlapStrategy,
    pub monitor_overhead_sensitive: bool,
    pub auto_report: AutoExecutionReport,
}

impl AdaptiveExecutionDecision {
    pub fn decide(
        problem_size: usize,
        comm_size: usize,
        restart_len: usize,
        local_work: usize,
        reduction_latency_us: f64,
        reproducible: bool,
        monitor_overhead_sensitive: bool,
        reduction: &ReductOptions,
    ) -> Self {
        let tune = parallel_tune();

        #[cfg(feature = "rayon")]
        let threading = {
            let rec =
                crate::parallel::threads::suggest_thread_policy(problem_size, comm_size, tune);
            (
                match rec.flavor {
                    crate::parallel::threads::ThreadExecFlavor::Serial => "serial",
                    crate::parallel::threads::ThreadExecFlavor::Rayon => "rayon",
                },
                rec.threads,
                rec.reason,
            )
        };
        #[cfg(not(feature = "rayon"))]
        let threading = ("serial", 1, "rayon feature disabled");

        let requested_reduction = reduction.mode;
        let selected_reduction = if reproducible {
            match requested_reduction {
                ReproMode::Fast => ReproMode::Deterministic,
                mode => mode,
            }
        } else if comm_size > 1 && (monitor_overhead_sensitive || reduction_latency_us > 35.0) {
            ReproMode::Fast
        } else {
            requested_reduction
        };

        let high_latency = reduction_latency_us > 25.0;
        let very_high_latency = reduction_latency_us > 60.0;

        let reduction_exec = if comm_size > 1
            && !reproducible
            && !monitor_overhead_sensitive
            && (problem_size >= tune.min_rows_spmv || high_latency)
        {
            ReductExec::Async
        } else {
            ReductExec::Sync
        };

        let variant = if comm_size <= 1 || restart_len <= 2 {
            KrylovVariant::Classical
        } else if very_high_latency && restart_len >= 12 && local_work >= tune.min_rows_spmv {
            KrylovVariant::SStep
        } else if high_latency || problem_size >= tune.min_rows_spmv {
            KrylovVariant::Pipelined
        } else {
            KrylovVariant::Classical
        };

        let sstep_block = match variant {
            KrylovVariant::SStep => Some((restart_len / 4).clamp(2, 8)),
            _ => None,
        };

        let overlap = if matches!(reduction_exec, ReductExec::Sync) {
            OverlapStrategy::Disabled
        } else if problem_size >= 4 * tune.min_rows_spmv {
            OverlapStrategy::Aggressive
        } else {
            OverlapStrategy::Conservative
        };

        Self {
            problem_size,
            comm_size,
            restart_len,
            local_work,
            reduction_latency_us,
            tune,
            threading: threading.0,
            recommended_threads: threading.1,
            threading_reason: threading.2,
            requested_reduction,
            selected_reduction,
            reduction_exec,
            variant,
            sstep_block,
            overlap,
            monitor_overhead_sensitive,
            auto_report: AutoExecutionReport {
                comm_size,
                local_work,
                restart_len,
                reduction_latency_us,
                variant,
                reduction: selected_reduction,
                reduction_exec,
                overlap,
                threading: threading.0,
                threads: threading.1,
            },
        }
    }
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
        Self::nested_from_options_with_context(opts, comm_size, NestedPolicyContext::default())
    }

    pub fn nested_from_options_with_context(
        opts: &KspOptions,
        comm_size: usize,
        context: NestedPolicyContext<'_>,
    ) -> Result<Self, KError> {
        let mode = opts
            .threads_mode
            .as_deref()
            .or(context.outer_threads_mode)
            .unwrap_or("context");
        let policy = match mode {
            "serial" => NestedExecutionPolicy::Serial,
            "context" => NestedExecutionPolicy::ContextPool,
            "hybrid" => NestedExecutionPolicy::Hybrid,
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
            if policy == NestedExecutionPolicy::Hybrid {
                // handled below via thread capping
            } else {
                return Err(KError::InvalidInput(
                "nested pc_type=ksp with MPI and threads>1 requires ksp_threads_mode=serial or hybrid".into(),
            ));
            }
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
            NestedExecutionPolicy::Hybrid => {
                #[cfg(feature = "rayon")]
                {
                    let rank_threads = crate::parallel::threads::current_rayon_threads().max(1);
                    let outer_cap = context.outer_threads.unwrap_or(rank_threads).max(1);
                    let mut budget = rank_threads.min(outer_cap);
                    if comm_size > 1 {
                        budget = budget.min((rank_threads / 2).max(1));
                    }
                    let requested = opts.threads.unwrap_or(budget).max(1);
                    let chosen = requested.min(budget).max(1);
                    if chosen <= 1 {
                        exec.threading = ThreadingPolicy::Serial;
                    } else {
                        exec = exec.with_threads(chosen)?;
                    }
                }
                #[cfg(not(feature = "rayon"))]
                {
                    exec.threading = ThreadingPolicy::Serial;
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
    use crate::reduction::ReproMode;
    use crate::utils::reduction::{ReductExec, ReductOptions};

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
        assert!(format!("{err}").contains("requires ksp_threads_mode=serial or hybrid"));
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

    #[test]
    fn nested_policy_inherits_outer_hybrid_mode() {
        let opts = KspOptions::default();
        let pol = ExecutionPolicy::nested_from_options_with_context(
            &opts,
            4,
            NestedPolicyContext {
                outer_threads_mode: Some("hybrid"),
                outer_threads: Some(4),
            },
        )
        .unwrap();
        #[cfg(feature = "rayon")]
        assert!(!matches!(pol.threading, ThreadingPolicy::GlobalUnmodified));
    }

    #[test]
    fn adaptive_policy_uses_fast_sync_when_monitor_heavy() {
        let opt = ReductOptions::default();
        let d = AdaptiveExecutionDecision::decide(2048, 4, 30, 2048 * 30, 10.0, false, true, &opt);
        assert_eq!(d.selected_reduction, ReproMode::Fast);
        assert!(matches!(d.reduction_exec, ReductExec::Sync));
    }

    #[test]
    fn adaptive_policy_forces_deterministic_when_reproducible() {
        let opt = ReductOptions::default();
        let d =
            AdaptiveExecutionDecision::decide(16384, 8, 40, 16384 * 40, 20.0, true, false, &opt);
        assert!(matches!(
            d.selected_reduction,
            ReproMode::Deterministic | ReproMode::DeterministicAccurate
        ));
    }

    #[test]
    fn adaptive_policy_selects_sstep_for_high_latency() {
        let opt = ReductOptions::default();
        let d =
            AdaptiveExecutionDecision::decide(32768, 16, 50, 32768 * 50, 80.0, false, false, &opt);
        assert!(matches!(d.variant, KrylovVariant::SStep));
        assert!(d.sstep_block.is_some());
    }
}
