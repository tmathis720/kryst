//! Runtime SpMV plan selection and metadata.

use std::time::Instant;

use crate::matrix::sparse::CsrMatrix;

use super::scalar;
#[cfg(feature = "simd")]
use super::{sellc, simd_csr};

/// Identifies the selected kernel implementation inside a [`SpmvPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    Scalar,
    #[cfg(feature = "simd")]
    CsrSimdGather,
    #[cfg(feature = "simd")]
    SellC,
}

/// Per-matrix runtime plan describing how sparse matrix-vector products should
/// be executed.
#[derive(Clone, Debug)]
pub struct SpmvPlan {
    pub kind: KernelKind,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<f64>,
    #[cfg(feature = "simd")]
    pub sell: Option<sellc::SellCStorage>,
    #[cfg(feature = "simd")]
    lanes: usize,
}

impl SpmvPlan {
    /// Applies the selected kernel to compute `y = alpha * A * x + beta * y`.
    #[inline]
    pub fn apply(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        match self.kind {
            KernelKind::Scalar => scalar::spmv_scaled_csr(
                self.nrows(),
                &self.row_ptr,
                &self.col_idx,
                &self.vals,
                alpha,
                x,
                beta,
                y,
            ),
            #[cfg(feature = "simd")]
            KernelKind::CsrSimdGather => {
                if self.lanes <= 1 {
                    simd_csr::fallback_scalar(
                        self.nrows(),
                        &self.row_ptr,
                        &self.col_idx,
                        &self.vals,
                        alpha,
                        x,
                        beta,
                        y,
                    );
                } else {
                    simd_csr::dispatch_spmv_scaled_csr_simd_gather(
                        self.lanes,
                        self.nrows(),
                        &self.row_ptr,
                        &self.col_idx,
                        &self.vals,
                        alpha,
                        x,
                        beta,
                        y,
                    );
                }
            }
            #[cfg(feature = "simd")]
            KernelKind::SellC => {
                let sell = self
                    .sell
                    .as_ref()
                    .expect("SELL-C plan missing storage for SellC kernel");
                dispatch_sellc(self.lanes, sell, alpha, x, beta, y);
            }
        }
    }

    /// Returns the number of rows stored in the CSR representation.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }

    /// Builds a scalar-only plan.
    pub fn build_scalar(matrix: &CsrMatrix<f64>) -> Self {
        Self {
            kind: KernelKind::Scalar,
            row_ptr: matrix.row_ptr().to_vec(),
            col_idx: matrix.col_idx().to_vec(),
            vals: matrix.values().to_vec(),
            #[cfg(feature = "simd")]
            sell: None,
            #[cfg(feature = "simd")]
            lanes: 1,
        }
    }
}

/// Tuning knobs influencing how [`SpmvPlan::build`] selects a kernel.
#[derive(Clone, Debug)]
pub struct SpmvTuning {
    pub allow_simd: bool,
    pub prefer_sellc: bool,
    pub sell_c: usize,
    pub sell_sigma: usize,
    pub bench_nsamples: usize,
    pub min_nnz_for_simd: usize,
}

impl Default for SpmvTuning {
    fn default() -> Self {
        Self {
            allow_simd: cfg!(feature = "simd"),
            prefer_sellc: true,
            sell_c: 16,
            sell_sigma: 64,
            bench_nsamples: 3,
            min_nnz_for_simd: 2_000,
        }
    }
}

/// Builds an SpMV plan using the provided tuning configuration.
pub fn build(matrix: &CsrMatrix<f64>, tuning: &SpmvTuning) -> SpmvPlan {
    let mut plan = SpmvPlan::build_scalar(matrix);

    #[cfg(feature = "simd")]
    {
        if !tuning.allow_simd {
            return plan;
        }
        let nnz = plan.col_idx.len();
        if nnz < tuning.min_nnz_for_simd {
            return plan;
        }

        let lanes = simd_csr::detect_simd_lanes();
        if lanes <= 1 {
            return plan;
        }

        let m = matrix.nrows();
        if m == 0 {
            return plan;
        }

        let mut lmin = usize::MAX;
        let mut lmax = 0usize;
        for row in 0..m {
            let len = plan.row_ptr[row + 1] - plan.row_ptr[row];
            if len == 0 {
                continue;
            }
            lmin = lmin.min(len);
            lmax = lmax.max(len);
        }
        let is_uniformish = if lmin == usize::MAX {
            true
        } else {
            lmax <= lmin.saturating_mul(2) && lmax <= 128
        };

        let mut best_kind = KernelKind::CsrSimdGather;
        let mut best_sell = None;
        let bench_runs = tuning.bench_nsamples;

        let mut y_buf = vec![0.0f64; matrix.nrows()];
        let x_buf = vec![1.0f64; matrix.ncols().max(1)];

        let gather_time = microbench(bench_runs, || {
            simd_csr::dispatch_spmv_scaled_csr_simd_gather(
                lanes,
                plan.nrows(),
                &plan.row_ptr,
                &plan.col_idx,
                &plan.vals,
                1.0,
                &x_buf,
                0.0,
                &mut y_buf,
            );
        });

        let prefer_sell = tuning.prefer_sellc || !is_uniformish;
        if prefer_sell {
            let sell_c = round_up_to_multiple(tuning.sell_c.max(lanes), lanes);
            let sell_sigma = tuning.sell_sigma.max(sell_c);
            let sell = sellc::csr_to_sellc(
                matrix.nrows(),
                matrix.ncols(),
                &plan.row_ptr,
                &plan.col_idx,
                &plan.vals,
                sell_c,
                sell_sigma,
            );
            let sell_time = microbench(bench_runs, || {
                dispatch_sellc(lanes, &sell, 1.0, &x_buf, 0.0, &mut y_buf);
            });
            if sell_time < gather_time {
                best_kind = KernelKind::SellC;
                best_sell = Some(sell);
            }
        }

        plan.kind = best_kind;
        plan.lanes = lanes;
        plan.sell = best_sell;
    }

    plan
}

#[cfg(feature = "simd")]
fn dispatch_sellc(
    lanes: usize,
    storage: &sellc::SellCStorage,
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    sellc::spmv_scaled_sellc(
        storage,
        alpha,
        x,
        beta,
        y,
        match lanes {
            4 => 4,
            _ => 2,
        },
    );
}

#[cfg(feature = "simd")]
fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return value;
    }
    ((value + multiple - 1) / multiple) * multiple
}

fn microbench<F: FnMut()>(nsamples: usize, mut f: F) -> f64 {
    if nsamples == 0 {
        f();
        return 0.0;
    }
    let mut best = f64::INFINITY;
    for _ in 0..nsamples {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}
