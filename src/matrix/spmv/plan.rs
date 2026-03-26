//! Runtime SpMV plan selection and metadata for real-valued CSR matrices.
//!
//! SIMD paths are intentionally limited to `S = f64` for AMG-oriented SpMV.
//! In complex builds (`feature = "complex"`), the plan always selects the
//! scalar kernel even when `feature = "simd"` is enabled.

#[cfg(all(feature = "simd", not(feature = "complex")))]
use core::any::TypeId;
use std::time::Instant;

use crate::algebra::scalar::KrystScalar;
use crate::matrix::csr::CsrMatrix;

use super::scalar;
#[cfg(all(feature = "simd", not(feature = "complex")))]
use super::{sellc, simd_csr};

/// Identifies the selected kernel implementation inside a [`SpmvPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpmvKernel {
    Scalar,
    #[cfg(all(feature = "simd", not(feature = "complex")))]
    CsrSimdGather,
    #[cfg(all(feature = "simd", not(feature = "complex")))]
    SellC,
}

/// Per-matrix runtime plan describing how sparse matrix-vector products should
/// be executed. Complex builds keep `kernel` at [`SpmvKernel::Scalar`] even
/// when SIMD is available.
#[derive(Clone, Debug)]
pub struct SpmvPlan<S: KrystScalar> {
    pub kernel: SpmvKernel,
    pub diagnostics: SpmvPlanDiagnostics,
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<S>,
    #[cfg(all(feature = "simd", not(feature = "complex")))]
    pub sell: Option<sellc::SellCStorage>,
    #[cfg(all(feature = "simd", not(feature = "complex")))]
    lanes: usize,
}

#[derive(Clone, Debug)]
pub struct SpmvPlanDiagnostics {
    pub selected_kernel: SpmvKernel,
    pub simd_lanes: usize,
    pub gather_bench_seconds: Option<f64>,
    pub sellc_bench_seconds: Option<f64>,
}

impl<S: KrystScalar> SpmvPlan<S> {
    /// Applies the selected kernel to compute `y = alpha * A * x + beta * y`.
    #[inline]
    pub fn apply_scaled(&self, alpha: S, x: &[S], beta: S, y: &mut [S]) {
        match self.kernel {
            SpmvKernel::Scalar => scalar::spmv_scaled_csr(
                self.nrows,
                &self.row_ptr,
                &self.col_idx,
                &self.vals,
                alpha,
                x,
                beta,
                y,
            ),
            #[cfg(all(feature = "simd", not(feature = "complex")))]
            SpmvKernel::CsrSimdGather => {
                debug_assert_eq!(TypeId::of::<S>(), TypeId::of::<f64>());
                let alpha = unsafe { *(&alpha as *const S as *const f64) };
                let beta = unsafe { *(&beta as *const S as *const f64) };
                let x = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
                let y =
                    unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
                let vals = unsafe {
                    std::slice::from_raw_parts(self.vals.as_ptr() as *const f64, self.vals.len())
                };

                if self.lanes <= 1 {
                    simd_csr::fallback_scalar(
                        self.nrows,
                        &self.row_ptr,
                        &self.col_idx,
                        vals,
                        alpha,
                        x,
                        beta,
                        y,
                    );
                } else {
                    simd_csr::dispatch_spmv_scaled_csr_simd_gather(
                        self.lanes,
                        self.nrows,
                        &self.row_ptr,
                        &self.col_idx,
                        vals,
                        alpha,
                        x,
                        beta,
                        y,
                    );
                }
            }
            #[cfg(all(feature = "simd", not(feature = "complex")))]
            SpmvKernel::SellC => {
                let sell = self
                    .sell
                    .as_ref()
                    .expect("SELL-C plan missing storage for SellC kernel");
                let alpha = unsafe { *(&alpha as *const S as *const f64) };
                let beta = unsafe { *(&beta as *const S as *const f64) };
                let x = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
                let y =
                    unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
                dispatch_sellc(self.lanes, sell, alpha, x, beta, y);
            }
        }
    }

    /// Returns the number of rows stored in the CSR representation.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Builds a scalar-only plan by cloning the CSR structure.
    pub fn build_scalar(matrix: &CsrMatrix<S>) -> Self {
        Self {
            kernel: SpmvKernel::Scalar,
            diagnostics: SpmvPlanDiagnostics {
                selected_kernel: SpmvKernel::Scalar,
                simd_lanes: 1,
                gather_bench_seconds: None,
                sellc_bench_seconds: None,
            },
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_ptr: matrix.rowptr.clone(),
            col_idx: matrix.colind.clone(),
            vals: matrix.values.clone(),
            #[cfg(all(feature = "simd", not(feature = "complex")))]
            sell: None,
            #[cfg(all(feature = "simd", not(feature = "complex")))]
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
            allow_simd: cfg!(all(feature = "simd", not(feature = "complex"))),
            prefer_sellc: true,
            sell_c: 16,
            sell_sigma: 64,
            bench_nsamples: 3,
            min_nnz_for_simd: 2_000,
        }
    }
}

/// Builds an SpMV plan from a borrowed CSR matrix.
pub fn build<S: KrystScalar>(matrix: &CsrMatrix<S>, tuning: &SpmvTuning) -> SpmvPlan<S> {
    build_owned(matrix.clone(), tuning)
}

/// Builds an SpMV plan while taking ownership of the CSR storage.
pub fn build_owned<S: KrystScalar>(matrix: CsrMatrix<S>, tuning: &SpmvTuning) -> SpmvPlan<S> {
    let CsrMatrix {
        nrows,
        ncols,
        rowptr,
        colind,
        values,
    } = matrix;

    #[allow(unused_mut)]
    let mut plan = SpmvPlan {
        kernel: SpmvKernel::Scalar,
        diagnostics: SpmvPlanDiagnostics {
            selected_kernel: SpmvKernel::Scalar,
            simd_lanes: 1,
            gather_bench_seconds: None,
            sellc_bench_seconds: None,
        },
        nrows,
        ncols,
        row_ptr: rowptr,
        col_idx: colind,
        vals: values,
        #[cfg(all(feature = "simd", not(feature = "complex")))]
        sell: None,
        #[cfg(all(feature = "simd", not(feature = "complex")))]
        lanes: 1,
    };

    #[cfg(not(all(feature = "simd", not(feature = "complex"))))]
    let _ = tuning;

    #[cfg(all(feature = "simd", not(feature = "complex")))]
    {
        if TypeId::of::<S>() == TypeId::of::<f64>() && tuning.allow_simd {
            let nnz = plan.col_idx.len();
            if nnz >= tuning.min_nnz_for_simd {
                let lanes = simd_csr::detect_simd_lanes();
                if lanes > 1 && plan.nrows > 0 {
                    let mut lmin = usize::MAX;
                    let mut lmax = 0usize;
                    for row in 0..plan.nrows {
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

                    let mut best_kernel = SpmvKernel::CsrSimdGather;
                    let mut best_sell = None;
                    let bench_runs = tuning.bench_nsamples;

                    let mut y_buf = vec![0.0f64; plan.nrows];
                    let x_buf = vec![1.0f64; plan.ncols.max(1)];
                    let vals = unsafe {
                        std::slice::from_raw_parts(
                            plan.vals.as_ptr() as *const f64,
                            plan.vals.len(),
                        )
                    };

                    let gather_time = microbench(bench_runs, || {
                        simd_csr::dispatch_spmv_scaled_csr_simd_gather(
                            lanes,
                            plan.nrows,
                            &plan.row_ptr,
                            &plan.col_idx,
                            vals,
                            1.0,
                            &x_buf,
                            0.0,
                            &mut y_buf,
                        );
                    });
                    plan.diagnostics.gather_bench_seconds = Some(gather_time);

                    let prefer_sell = tuning.prefer_sellc || !is_uniformish;
                    if prefer_sell {
                        let sell_c = round_up_to_multiple(tuning.sell_c.max(lanes), lanes);
                        let sell_sigma = tuning.sell_sigma.max(sell_c);
                        let sell = sellc::csr_to_sellc(
                            plan.nrows,
                            plan.ncols,
                            &plan.row_ptr,
                            &plan.col_idx,
                            vals,
                            sell_c,
                            sell_sigma,
                        );
                        let sell_time = microbench(bench_runs, || {
                            dispatch_sellc(lanes, &sell, 1.0, &x_buf, 0.0, &mut y_buf);
                        });
                        plan.diagnostics.sellc_bench_seconds = Some(sell_time);
                        if sell_time < gather_time {
                            best_kernel = SpmvKernel::SellC;
                            best_sell = Some(sell);
                        }
                    }

                    plan.kernel = best_kernel;
                    plan.lanes = lanes;
                    plan.sell = best_sell;
                    plan.diagnostics.selected_kernel = best_kernel;
                    plan.diagnostics.simd_lanes = lanes;
                }
            }
        }
    }

    #[cfg(feature = "complex")]
    debug_assert!(matches!(plan.kernel, SpmvKernel::Scalar));

    plan
}

#[cfg(all(feature = "simd", not(feature = "complex")))]
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

#[cfg(all(feature = "simd", not(feature = "complex")))]
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

#[cfg(all(test, feature = "complex"))]
mod tests {
    use super::*;
    use crate::algebra::prelude::*;

    #[test]
    fn complex_build_uses_scalar_kernel() {
        let matrix = CsrMatrix::<S>::new(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![S::from_real(1.0), S::from_real(2.0)],
        );
        let plan = build(&matrix, &SpmvTuning::default());
        assert!(matches!(plan.kernel, SpmvKernel::Scalar));
    }
}
