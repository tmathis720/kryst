use once_cell::sync::OnceCell;
use std::sync::RwLock;

#[derive(Clone, Copy, Debug)]
pub struct ParallelTune {
    /// Minimum vector length to enable Rayon in elementwise kernels.
    pub min_len_vec: usize,
    /// Minimum rows to enable Rayon in CSR SpMV.
    pub min_rows_spmv: usize,
    /// Target chunk size in rows for CSR SpMV (approx).
    pub chunk_rows_spmv: usize,
}

impl Default for ParallelTune {
    fn default() -> Self {
        Self {
            min_len_vec: 8192,
            min_rows_spmv: 2048,
            chunk_rows_spmv: 512,
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
