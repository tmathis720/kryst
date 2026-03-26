use once_cell::sync::OnceCell;
use std::cell::Cell;
use std::sync::RwLock;

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
            min_rows_asm_apply: 512,
        }
    }
}

static PAR_TUNE: OnceCell<RwLock<ParallelTune>> = OnceCell::new();

fn cell() -> &'static RwLock<ParallelTune> {
    PAR_TUNE.get_or_init(|| RwLock::new(ParallelTune::default()))
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
