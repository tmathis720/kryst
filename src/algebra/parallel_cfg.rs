use once_cell::sync::OnceCell;
use std::cell::Cell;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug)]
pub struct ParallelTune {
    /// Minimum vector length to enable Rayon in elementwise kernels.
    pub min_len_vec: usize,
    /// Minimum rows to enable Rayon in CSR SpMV.
    pub min_rows_spmv: usize,
    /// Target chunk size in rows for CSR SpMV (approx).
    pub chunk_rows_spmv: usize,
    /// Minimum work size (`rows * cols`) to enable Rayon in CSR SpMM dense.
    pub min_work_spmm_dense: usize,
    /// Target row block size for threaded CSR SpMM dense.
    pub chunk_rows_spmm_dense: usize,
    /// Target RHS-column block size for threaded CSR SpMM dense.
    pub chunk_cols_spmm_dense: usize,
    /// Minimum rows to enable Rayon in ILU factorization kernels.
    pub min_rows_ilu_factorization: usize,
    /// Minimum rows to enable Rayon in ILU triangular solves.
    pub min_rows_ilu_triangular: usize,
    /// Minimum total rows in a level-group to activate parallel level solve work.
    pub min_rows_ilu_triangular_level_parallel: usize,
    /// Minimum rows in a bucket-group; smaller consecutive buckets are coalesced.
    pub min_rows_ilu_triangular_bucket_coalesce: usize,
    /// Minimum rows to enable Rayon in ASM block application.
    pub min_rows_asm_apply: usize,
}

impl Default for ParallelTune {
    fn default() -> Self {
        Self {
            min_len_vec: 8192,
            min_rows_spmv: 2048,
            chunk_rows_spmv: 512,
            min_work_spmm_dense: 16_384,
            chunk_rows_spmm_dense: 128,
            chunk_cols_spmm_dense: 4,
            min_rows_ilu_factorization: 512,
            min_rows_ilu_triangular: 512,
            min_rows_ilu_triangular_level_parallel: 128,
            min_rows_ilu_triangular_bucket_coalesce: 64,
            min_rows_asm_apply: 512,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelTunerMode {
    /// Keep the configured/manual thresholds exactly as provided.
    Manual,
    /// Allow adaptation from measured timings.
    Adaptive,
    /// Freeze thresholds for reproducible/CI runs.
    Deterministic,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KernelTimingSnapshot {
    pub serial_ns_per_elem: Option<f64>,
    pub parallel_ns_per_elem: Option<f64>,
    pub serial_samples: u64,
    pub parallel_samples: u64,
}

#[derive(Clone, Debug)]
pub struct AdaptiveTuneDecision {
    pub mode: ParallelTunerMode,
    pub reduction_latency_us: f64,
    pub baseline: ParallelTune,
    pub selected: ParallelTune,
    pub kernel_timing: KernelTimingSnapshot,
    pub rationale: &'static str,
}

static PAR_TUNE: OnceCell<RwLock<ParallelTune>> = OnceCell::new();
static PAR_TUNER_MODE: OnceCell<RwLock<ParallelTunerMode>> = OnceCell::new();

static SERIAL_TIME_NS: AtomicU64 = AtomicU64::new(0);
static SERIAL_ELEMS: AtomicU64 = AtomicU64::new(0);
static SERIAL_SAMPLES: AtomicU64 = AtomicU64::new(0);

static PAR_TIME_NS: AtomicU64 = AtomicU64::new(0);
static PAR_ELEMS: AtomicU64 = AtomicU64::new(0);
static PAR_SAMPLES: AtomicU64 = AtomicU64::new(0);

fn cell() -> &'static RwLock<ParallelTune> {
    PAR_TUNE.get_or_init(|| RwLock::new(ParallelTune::default()))
}

fn mode_cell() -> &'static RwLock<ParallelTunerMode> {
    PAR_TUNER_MODE.get_or_init(|| RwLock::new(ParallelTunerMode::Manual))
}

pub fn set_parallel_tune(t: ParallelTune) {
    if let Ok(mut guard) = cell().write() {
        *guard = t;
    }
}

pub fn parallel_tune() -> ParallelTune {
    cell()
        .read()
        .map(|g| *g)
        .unwrap_or_else(|_| ParallelTune::default())
}

pub fn set_parallel_tuner_mode(mode: ParallelTunerMode) {
    if let Ok(mut guard) = mode_cell().write() {
        *guard = mode;
    }
}

pub fn parallel_tuner_mode() -> ParallelTunerMode {
    mode_cell()
        .read()
        .map(|g| *g)
        .unwrap_or(ParallelTunerMode::Manual)
}

pub fn observe_vector_kernel_timing(len: usize, used_parallel: bool, elapsed_ns: u64) {
    if len == 0 || elapsed_ns == 0 {
        return;
    }
    let len = len as u64;
    if used_parallel {
        PAR_TIME_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        PAR_ELEMS.fetch_add(len, Ordering::Relaxed);
        PAR_SAMPLES.fetch_add(1, Ordering::Relaxed);
    } else {
        SERIAL_TIME_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        SERIAL_ELEMS.fetch_add(len, Ordering::Relaxed);
        SERIAL_SAMPLES.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn kernel_timing_snapshot() -> KernelTimingSnapshot {
    let serial_time = SERIAL_TIME_NS.load(Ordering::Relaxed);
    let serial_elems = SERIAL_ELEMS.load(Ordering::Relaxed);
    let parallel_time = PAR_TIME_NS.load(Ordering::Relaxed);
    let parallel_elems = PAR_ELEMS.load(Ordering::Relaxed);

    KernelTimingSnapshot {
        serial_ns_per_elem: (serial_elems > 0).then(|| serial_time as f64 / serial_elems as f64),
        parallel_ns_per_elem: (parallel_elems > 0)
            .then(|| parallel_time as f64 / parallel_elems as f64),
        serial_samples: SERIAL_SAMPLES.load(Ordering::Relaxed),
        parallel_samples: PAR_SAMPLES.load(Ordering::Relaxed),
    }
}

pub fn adapt_parallel_tune(
    baseline: ParallelTune,
    reduction_latency_us: f64,
    reproducible: bool,
) -> AdaptiveTuneDecision {
    let mode = if reproducible {
        ParallelTunerMode::Deterministic
    } else {
        parallel_tuner_mode()
    };
    let timing = kernel_timing_snapshot();
    let mut selected = baseline;
    let mut rationale = "manual_or_reproducible";

    if matches!(mode, ParallelTunerMode::Adaptive) {
        if let (Some(serial), Some(par)) = (timing.serial_ns_per_elem, timing.parallel_ns_per_elem)
        {
            if timing.serial_samples >= 4 && timing.parallel_samples >= 4 {
                let speedup = if par > 0.0 { serial / par } else { 1.0 };
                if speedup > 1.2 {
                    selected.min_len_vec = baseline.min_len_vec.saturating_mul(3).max(512) / 4;
                    rationale = "parallel_vector_kernels_faster";
                } else if speedup < 1.05 {
                    selected.min_len_vec = baseline.min_len_vec.saturating_mul(5) / 4;
                    rationale = "parallel_vector_kernels_not_profitable";
                } else {
                    rationale = "vector_kernel_speedup_neutral";
                }
            } else {
                rationale = "insufficient_kernel_samples";
            }
        } else {
            rationale = "missing_kernel_samples";
        }

        selected.min_rows_spmv = (selected.min_len_vec / 4).clamp(256, 65_536);
        selected.min_rows_ilu_factorization = (selected.min_rows_spmv / 4).clamp(128, 8_192);
        selected.min_rows_ilu_triangular = selected.min_rows_ilu_factorization;
        selected.min_rows_ilu_triangular_level_parallel =
            (selected.min_rows_ilu_triangular / 4).clamp(32, 2_048);
        selected.min_rows_ilu_triangular_bucket_coalesce =
            (selected.min_rows_ilu_triangular_level_parallel / 2).max(8);
        selected.min_rows_asm_apply = (selected.min_rows_spmv / 3).clamp(128, 16_384);

        if reduction_latency_us > 50.0 {
            selected.min_rows_spmv = selected.min_rows_spmv.saturating_mul(3) / 4;
            selected.min_rows_asm_apply = selected.min_rows_asm_apply.saturating_mul(3) / 4;
            rationale = "high_reduction_latency_bias_parallel";
        }
    }

    AdaptiveTuneDecision {
        mode,
        reduction_latency_us,
        baseline,
        selected,
        kernel_timing: timing,
        rationale,
    }
}

thread_local! {
    static FORCE_SERIAL: Cell<bool> = Cell::new(false);
}

pub struct SerialGuard {
    prev: bool,
}

impl Drop for SerialGuard {
    fn drop(&mut self) {
        FORCE_SERIAL.with(|flag| flag.set(self.prev));
    }
}

pub fn force_serial() -> bool {
    FORCE_SERIAL.with(|flag| flag.get())
}

pub fn serial_guard(enable: bool) -> SerialGuard {
    let prev = FORCE_SERIAL.with(|flag| {
        let prev = flag.get();
        flag.set(enable);
        prev
    });
    SerialGuard { prev }
}

/// Configure Rayon for reproducible runs by constraining the global pool.
pub fn set_rayon_threads_for_repro(enable: bool) {
    #[cfg(feature = "rayon")]
    {
        if enable {
            let _ = crate::parallel::threads::init_global_rayon_pool_with_threads(1);
        }
    }
    let _ = enable;
}
