//! DistCsrOp: canonical distributed CSR linear operator.
//!
//! This is the preferred representation for distributed sparse matrices. Other
//! distributed matrix APIs (e.g. `parcsr::*`) are secondary and will gradually
//! be reworked to build on this abstraction.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::algebra::bridge::BridgeScratch;
use crate::algebra::scalar::{KrystScalar, S};
use crate::error::KError;
use crate::matrix::csr::CsrMatrix as PlanCsrMatrix;
use crate::matrix::dist::csr_types::{DistRowCsr, LocalSquareCsr};
use crate::matrix::dist::halo::{HaloIndexPlan, HaloPhaseTimings, HaloPlan, HaloTuning};
use crate::matrix::dist::spmv_dist::RowRanges;
use crate::matrix::op::{ChangeIds, DistLayout, LinOp, StructureId, ValuesId};
use crate::matrix::parcsr::ParCsrMatrix;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::spmv::plan::{self as spmv_plan, SpmvKernel, SpmvPlan, SpmvTuning};
use crate::ops::klinop::KLinOp;
use crate::parallel::{Comm, UniverseComm};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use faer::Mat;

fn owner_of(j: usize, row_part: &[usize]) -> usize {
    // Locate the owner rank such that row_part[r] <= j < row_part[r + 1].
    let mut lo = 0usize;
    let mut hi = row_part.len() - 2;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        if j < row_part[mid + 1] {
            if j >= row_part[mid] {
                return mid;
            }
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn self_idx(plan: &HaloIndexPlan, gcol: usize) -> usize {
    plan.n_local
        + *plan
            .ghost_index_of
            .get(&gcol)
            .expect("ghost column missing from halo plan")
}

/// Canonical distributed CSR operator with an MPI-backed halo plan.
///
/// # Thread-safety
/// `DistCsrOp` supports concurrent `matvec` calls on a single instance.
/// Halo exchange state is checked out per call from a pool managed by `HaloPlan`.
pub struct DistCsrOp {
    pub n_global: usize,
    pub row_start: usize,
    pub row_end: usize,
    pub n_local: usize,
    layout: DistLayout,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    vals: Vec<S>,
    row_is_local: Vec<bool>,
    #[cfg_attr(feature = "rayon", allow(dead_code))]
    local_only: RowRanges,
    border: RowRanges,
    border_ghost_row_ranges: Vec<Option<std::ops::Range<usize>>>,
    border_ghost_col_unified: Vec<usize>,
    border_ghost_vals: Vec<S>,
    local_diag_plan: SpmvPlan<S>,
    plan_diagnostics: DistributedPlanDiagnostics,
    halo: HaloPlan,
    overlap_mode: HaloOverlapMode,
    ids: ChangeIds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaloOverlapMode {
    Disabled,
    Interior,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistLocalKernelStrategy {
    RowSplitScalar,
    LocalDiagSpmvPlan,
}

#[derive(Debug, Clone)]
pub struct DistributedPlanMetrics {
    pub n_local_rows: usize,
    pub local_nnz: usize,
    pub local_diag_nnz: usize,
    pub ghost_nnz: usize,
    pub local_only_rows: usize,
    pub border_rows: usize,
    pub halo_recv_volume: usize,
    pub halo_send_volume: usize,
}

#[derive(Debug, Clone)]
pub struct DistributedPlanDiagnostics {
    pub overlap_mode: HaloOverlapMode,
    pub kernel_strategy: DistLocalKernelStrategy,
    pub local_spmv_kernel: Option<SpmvKernel>,
    pub row_locality_ratio: f64,
    pub border_ratio: f64,
    pub halo_recv_volume: usize,
    pub halo_send_volume: usize,
    pub expected_communication_fraction: f64,
    pub expected_computation_fraction: f64,
}

/// Per-phase timings from an explicitly profiled distributed SpMV.
///
/// The ordinary [`LinOp::matvec`] path does not collect timestamps. Benchmarks
/// can call [`DistCsrOp::profile_matvec`] when a phase breakdown is required.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DistSpmvProfile {
    pub total: Duration,
    pub local_compute: Duration,
    pub packing: Duration,
    /// Time spent posting point-to-point communication requests.
    pub communication: Duration,
    pub wait: Duration,
    pub unpack: Duration,
}

impl DistSpmvProfile {
    fn add_halo(&mut self, phase: HaloPhaseTimings) {
        self.packing += phase.packing;
        self.communication += phase.communication;
        self.wait += phase.wait;
        self.unpack += phase.unpack;
    }
}

pub fn choose_distributed_plan(
    metrics: &DistributedPlanMetrics,
    local_spmv_kernel: Option<SpmvKernel>,
) -> DistributedPlanDiagnostics {
    let n_rows = metrics.n_local_rows.max(1) as f64;
    let row_locality_ratio = (metrics.local_only_rows as f64 / n_rows).clamp(0.0, 1.0);
    let border_ratio = (metrics.border_rows as f64 / n_rows).clamp(0.0, 1.0);
    let halo_volume = metrics.halo_recv_volume + metrics.halo_send_volume;
    let halo_per_row = halo_volume as f64 / n_rows;
    let ghost_pressure = metrics.ghost_nnz as f64 / metrics.local_nnz.max(1) as f64;
    let communication_pressure =
        (0.5 * border_ratio + 0.3 * ghost_pressure + 0.2 * (halo_per_row / 8.0)).clamp(0.0, 1.0);

    let overlap_mode = if communication_pressure >= 0.28 {
        HaloOverlapMode::Interior
    } else {
        HaloOverlapMode::Disabled
    };

    let kernel_strategy = match local_spmv_kernel {
        Some(_) if row_locality_ratio >= 0.55 || communication_pressure < 0.25 => {
            DistLocalKernelStrategy::LocalDiagSpmvPlan
        }
        _ => DistLocalKernelStrategy::RowSplitScalar,
    };

    let mut expected_communication_fraction =
        (0.55 * border_ratio + 0.45 * ghost_pressure + (halo_per_row / 32.0)).clamp(0.0, 0.95);
    if overlap_mode == HaloOverlapMode::Interior {
        expected_communication_fraction *= 0.82;
    }
    let expected_computation_fraction = (1.0 - expected_communication_fraction).clamp(0.05, 1.0);

    DistributedPlanDiagnostics {
        overlap_mode,
        kernel_strategy,
        local_spmv_kernel,
        row_locality_ratio,
        border_ratio,
        halo_recv_volume: metrics.halo_recv_volume,
        halo_send_volume: metrics.halo_send_volume,
        expected_communication_fraction,
        expected_computation_fraction,
    }
}

impl DistCsrOp {
    /// Canonical balanced row partition helper for distributed operators.
    ///
    /// Returns a prefix array of length `comm.size() + 1` where each rank `r`
    /// owns rows `part[r]..part[r + 1]`.
    pub fn partition_rows_balanced(n_global: usize, comm: &UniverseComm) -> Vec<usize> {
        let p = comm.size();
        assert!(p > 0, "number of partitions must be positive");
        let base = n_global / p;
        let rem = n_global % p;
        let mut starts = Vec::with_capacity(p + 1);
        let mut s = 0usize;
        for k in 0..p {
            starts.push(s);
            s += base + usize::from(k < rem);
        }
        starts.push(n_global);
        starts
    }

    pub fn from_local_rows(
        n_global: usize,
        row_start: usize,
        local_rows: &CsrMatrix<S>,
        part_prefix: &[usize],
        comm: UniverseComm,
    ) -> Result<Self, KError> {
        Self::from_local_rows_with_halo_tuning(
            n_global,
            row_start,
            local_rows,
            part_prefix,
            comm,
            HaloTuning::default(),
        )
    }

    pub fn from_local_rows_with_halo_tuning(
        n_global: usize,
        row_start: usize,
        local_rows: &CsrMatrix<S>,
        part_prefix: &[usize],
        comm: UniverseComm,
        halo_tuning: HaloTuning,
    ) -> Result<Self, KError> {
        if part_prefix.len() != comm.size() + 1 {
            return Err(KError::InvalidInput(
                "partition vector length must be size + 1".into(),
            ));
        }
        let row_end = row_start + local_rows.nrows();
        let n_local = local_rows.nrows();
        let rank = comm.rank();

        let row_ptr = local_rows.row_ptr().to_vec();
        let col_idx = local_rows.col_idx().to_vec();
        let vals = local_rows.values().to_vec();

        let mut recv_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut row_is_local = vec![true; n_local];
        for i in 0..n_local {
            for idx in row_ptr[i]..row_ptr[i + 1] {
                let gcol = col_idx[idx];
                let owner = owner_of(gcol, part_prefix);
                if owner != rank {
                    row_is_local[i] = false;
                    recv_map.entry(owner).or_default().push(gcol);
                }
            }
        }

        let halo = HaloPlan::new_with_tuning(
            comm.clone(),
            Arc::new(part_prefix.to_vec()),
            row_start,
            row_end,
            recv_map,
            halo_tuning,
        )?;

        let local_only = RowRanges::from_mask(&row_is_local, true);
        let border = RowRanges::from_mask(&row_is_local, false);

        let mut border_ghost_row_ranges = vec![None; n_local];
        let mut border_ghost_col_unified = Vec::new();
        let mut border_ghost_vals = Vec::new();
        let mut local_diag_row_ptr = Vec::with_capacity(n_local + 1);
        let mut local_diag_col_idx = Vec::new();
        let mut local_diag_vals = Vec::new();
        local_diag_row_ptr.push(0);
        let mut local_diag_nnz = 0usize;
        let mut ghost_nnz = 0usize;
        let mut local_only_rows = 0usize;
        for i in 0..n_local {
            let start = border_ghost_col_unified.len();
            for idx in row_ptr[i]..row_ptr[i + 1] {
                let gcol = col_idx[idx];
                let owner = owner_of(gcol, halo.index.row_part.as_ref());
                if owner == rank {
                    local_diag_col_idx.push(gcol - row_start);
                    local_diag_vals.push(vals[idx]);
                    local_diag_nnz += 1;
                } else {
                    border_ghost_col_unified.push(self_idx(&halo.index, gcol));
                    border_ghost_vals.push(vals[idx]);
                    ghost_nnz += 1;
                }
            }
            local_diag_row_ptr.push(local_diag_col_idx.len());
            let end = border_ghost_col_unified.len();
            if end > start {
                border_ghost_row_ranges[i] = Some(start..end);
            } else {
                local_only_rows += 1;
            }
        }
        let local_diag = PlanCsrMatrix::new(
            n_local,
            n_local,
            local_diag_row_ptr,
            local_diag_col_idx,
            local_diag_vals,
        );
        let local_diag_plan = spmv_plan::build(&local_diag, &SpmvTuning::default());
        let metrics = DistributedPlanMetrics {
            n_local_rows: n_local,
            local_nnz: vals.len(),
            local_diag_nnz,
            ghost_nnz,
            local_only_rows,
            border_rows: n_local.saturating_sub(local_only_rows),
            halo_recv_volume: halo.recv_volume(),
            halo_send_volume: halo.send_volume(),
        };
        let plan_diagnostics = choose_distributed_plan(&metrics, Some(local_diag_plan.kernel));

        let ids = ChangeIds::default();
        ids.bump_structure();
        ids.bump_values();

        let layout = DistLayout {
            global_rows: n_global,
            global_cols: n_global,
            row_start,
            row_end,
            col_start: row_start,
            col_end: row_end,
        };

        Ok(Self {
            n_global,
            row_start,
            row_end,
            n_local,
            layout,
            row_ptr,
            col_idx,
            vals,
            row_is_local,
            local_only,
            border,
            border_ghost_row_ranges,
            border_ghost_col_unified,
            border_ghost_vals,
            local_diag_plan,
            plan_diagnostics: plan_diagnostics.clone(),
            halo,
            overlap_mode: plan_diagnostics.overlap_mode,
            ids,
        })
    }

    pub fn set_halo_overlap_mode(&mut self, mode: HaloOverlapMode) {
        self.overlap_mode = mode;
        self.plan_diagnostics.overlap_mode = mode;
    }

    pub fn plan_diagnostics(&self) -> &DistributedPlanDiagnostics {
        &self.plan_diagnostics
    }

    /// Build a distributed operator from a [`ParCsrMatrix`].
    ///
    /// This merges the diagonal and off-process blocks into a single local CSR
    pub fn from_parcsr(par: &ParCsrMatrix) -> Result<Self, KError> {
        let n_global = par.global_m;
        let local_rows = par.canonical_local_rows_csr()?;
        let part_prefix = Self::partition_rows_balanced(n_global, &par.comm);

        Self::from_local_rows(
            n_global,
            par.row_start,
            &local_rows,
            &part_prefix,
            par.comm.clone(),
        )
    }

    pub fn update_numeric(&mut self, new_vals: &[S]) -> Result<(), KError> {
        if new_vals.len() != self.vals.len() {
            return Err(KError::InvalidInput(
                "value array has incorrect length".into(),
            ));
        }
        self.vals.copy_from_slice(new_vals);
        let local = self.local_block_csr();
        let local_diag = PlanCsrMatrix::new(
            local.nrows(),
            local.ncols(),
            local.row_ptr().to_vec(),
            local.col_idx().to_vec(),
            local.values().to_vec(),
        );
        self.local_diag_plan = spmv_plan::build(&local_diag, &SpmvTuning::default());
        for row in 0..self.n_local {
            if let Some(range) = &self.border_ghost_row_ranges[row] {
                let mut slot = range.start;
                for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                    let owner = owner_of(self.col_idx[idx], self.halo.index.row_part.as_ref());
                    if owner != self.halo.index.rank {
                        self.border_ghost_vals[slot] = self.vals[idx];
                        slot += 1;
                    }
                }
            }
        }
        self.ids.bump_values();
        Ok(())
    }

    pub fn local_matrix(&self) -> CsrMatrix<S> {
        CsrMatrix::from_csr(
            self.n_local,
            self.n_global,
            self.row_ptr.clone(),
            self.col_idx.clone(),
            self.vals.clone(),
        )
    }

    /// Extract local owned rows with global column indexing.
    pub fn local_rows_csr(&self) -> DistRowCsr<S> {
        DistRowCsr::new(self.local_matrix(), self.row_start, self.n_global)
            .expect("DistCsrOp::local_matrix shape invariant violated")
    }

    /// Extract the owned diagonal block as a CSR matrix (local rows/cols only).
    pub fn local_block_csr(&self) -> CsrMatrix<S> {
        let n = self.n_local;
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        row_ptr.push(0);
        for row in 0..n {
            for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                let gcol = self.col_idx[idx];
                if gcol >= self.row_start && gcol < self.row_end {
                    col_idx.push(gcol - self.row_start);
                    vals.push(self.vals[idx]);
                }
            }
            row_ptr.push(col_idx.len());
        }
        CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
    }

    /// Extract the owned diagonal block with local square semantics.
    pub fn local_square_block(&self) -> LocalSquareCsr<S> {
        LocalSquareCsr::try_from(self.local_block_csr())
            .expect("DistCsrOp local block must be square by construction")
    }

    #[cfg(all(feature = "backend-faer", not(feature = "complex")))]
    /// Extract the owned diagonal block as a dense matrix (real builds only).
    pub fn local_block_dense(&self) -> Mat<f64> {
        let n = self.n_local;
        let mut local = Mat::zeros(n, n);
        for row in 0..n {
            for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                let gcol = self.col_idx[idx];
                if gcol >= self.row_start && gcol < self.row_end {
                    local[(row, gcol - self.row_start)] = self.vals[idx];
                }
            }
        }
        local
    }

    /// Return the global index of the first local row.
    pub fn local_row_offset(&self) -> usize {
        self.row_start
    }

    /// Return the global row partition used by the distributed operator.
    pub fn row_partition(&self) -> Arc<Vec<usize>> {
        self.halo.index.row_part.clone()
    }

    /// Return the halo index plan used by the distributed operator.
    pub fn halo_index(&self) -> Arc<HaloIndexPlan> {
        self.halo.index.clone()
    }

    /// Number of local rows stored on this rank.
    pub fn local_nrows(&self) -> usize {
        self.n_local
    }

    /// Compute `y = A x` while recording the major local and halo phases.
    ///
    /// This follows the same overlap mode and kernels as `matvec`, but adds
    /// timestamp collection. It is intended for baselines and diagnostics,
    /// not steady-state production calls.
    pub fn profile_matvec(&self, x: &[S], y: &mut [S]) -> Result<DistSpmvProfile, KError> {
        if x.len() != self.n_local || y.len() != self.n_local {
            return Err(KError::InvalidInput("dimension mismatch".into()));
        }

        let total_start = Instant::now();
        let mut profile = DistSpmvProfile::default();
        let compute_start = Instant::now();
        y.fill(S::zero());
        profile.local_compute += compute_start.elapsed();

        match self.overlap_mode {
            HaloOverlapMode::Disabled => {
                let halo_req =
                    if self.halo.index.n_ghost > 0 || !self.halo.index.send_local_idx.is_empty() {
                        let (request, phase) = self.halo.post_halo_profiled(x);
                        profile.add_halo(phase);
                        Some(request)
                    } else {
                        None
                    };
                if let Some(request) = halo_req {
                    let (ghost, phase) = self.halo.complete_halo_profiled(request);
                    profile.add_halo(phase);
                    let compute_start = Instant::now();
                    self.spmv_local_only(x, y);
                    self.spmv_border(y, &ghost);
                    profile.local_compute += compute_start.elapsed();
                } else {
                    let compute_start = Instant::now();
                    self.spmv_local_only(x, y);
                    self.spmv_border(y, &[]);
                    profile.local_compute += compute_start.elapsed();
                }
            }
            HaloOverlapMode::Interior => {
                let halo_req =
                    if self.halo.index.n_ghost > 0 || !self.halo.index.send_local_idx.is_empty() {
                        let (request, phase) = self.halo.post_halo_profiled(x);
                        profile.add_halo(phase);
                        Some(request)
                    } else {
                        None
                    };

                let compute_start = Instant::now();
                self.spmv_local_only(x, y);
                profile.local_compute += compute_start.elapsed();

                if let Some(request) = halo_req {
                    let (ghost, phase) = self.halo.complete_halo_profiled(request);
                    profile.add_halo(phase);
                    let compute_start = Instant::now();
                    self.spmv_border(y, &ghost);
                    profile.local_compute += compute_start.elapsed();
                } else {
                    let compute_start = Instant::now();
                    self.spmv_border(y, &[]);
                    profile.local_compute += compute_start.elapsed();
                }
            }
        }

        profile.total = total_start.elapsed();
        Ok(profile)
    }

    fn spmv_local_only(&self, x: &[S], y: &mut [S]) {
        if self.plan_diagnostics.kernel_strategy == DistLocalKernelStrategy::LocalDiagSpmvPlan {
            self.local_diag_plan.apply_scaled(S::one(), x, S::zero(), y);
            return;
        }
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            y.par_iter_mut()
                .enumerate()
                .filter(|(row, _)| self.row_is_local[*row])
                .for_each(|(row, slot)| {
                    let mut acc = S::zero();
                    for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                        let col = self.col_idx[idx] - self.row_start;
                        acc = acc + self.vals[idx] * x[col];
                    }
                    *slot = acc;
                });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for span in &self.local_only.spans {
                for row in span.clone() {
                    let mut acc = S::zero();
                    for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                        let col = self.col_idx[idx] - self.row_start;
                        acc = acc + self.vals[idx] * x[col];
                    }
                    y[row] = acc;
                }
            }
        }
    }

    fn spmv_border(&self, y: &mut [S], ghost: &[S]) {
        if self.border.is_empty() {
            return;
        }
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            y.par_iter_mut()
                .enumerate()
                .filter(|(row, _)| !self.row_is_local[*row])
                .for_each(|(row, slot)| {
                    if let Some(range) = &self.border_ghost_row_ranges[row] {
                        let mut acc = S::zero();
                        for k in range.clone() {
                            let col = self.border_ghost_col_unified[k] - self.n_local;
                            let val = self.border_ghost_vals[k];
                            acc = acc + val * ghost[col];
                        }
                        *slot = *slot + acc;
                    }
                });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for span in &self.border.spans {
                for row in span.clone() {
                    if let Some(range) = &self.border_ghost_row_ranges[row] {
                        let mut acc = S::zero();
                        for k in range.clone() {
                            let col = self.border_ghost_col_unified[k] - self.n_local;
                            let val = self.border_ghost_vals[k];
                            acc = acc + val * ghost[col];
                        }
                        y[row] = y[row] + acc;
                    }
                }
            }
        }
    }
}

impl KLinOp for DistCsrOp {
    type Scalar = S;

    fn dims(&self) -> (usize, usize) {
        (self.n_local, self.n_local)
    }

    fn matvec_s(&self, x: &[S], y: &mut [S], _scratch: &mut BridgeScratch) {
        assert_eq!(x.len(), self.n_local);
        assert_eq!(y.len(), self.n_local);
        for v in y.iter_mut() {
            *v = S::zero();
        }
        match self.overlap_mode {
            HaloOverlapMode::Disabled => {
                let halo_req =
                    if self.halo.index.n_ghost > 0 || !self.halo.index.send_local_idx.is_empty() {
                        Some(self.halo.post_halo(x))
                    } else {
                        None
                    };
                if let Some(req) = halo_req {
                    let ghost = self.halo.complete_halo(req);
                    self.spmv_local_only(x, y);
                    self.spmv_border(y, &ghost[..]);
                } else {
                    self.spmv_local_only(x, y);
                    self.spmv_border(y, &[]);
                }
            }
            HaloOverlapMode::Interior => {
                let halo_req =
                    if self.halo.index.n_ghost > 0 || !self.halo.index.send_local_idx.is_empty() {
                        Some(self.halo.post_halo(x))
                    } else {
                        None
                    };

                self.spmv_local_only(x, y);

                if let Some(req) = halo_req {
                    let ghost = self.halo.complete_halo(req);
                    self.spmv_border(y, &ghost[..]);
                } else {
                    self.spmv_border(y, &[]);
                }
            }
        }
    }
}

impl LinOp for DistCsrOp {
    type S = S;

    fn dims(&self) -> (usize, usize) {
        (self.n_local, self.n_local)
    }

    fn matvec(&self, x: &[S], y: &mut [S]) {
        let mut scratch = BridgeScratch::default();
        self.matvec_s(x, y, &mut scratch);
    }

    fn try_matvec(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != self.n_local || y.len() != self.n_local {
            return Err(KError::InvalidInput("dimension mismatch".into()));
        }
        self.matvec(x, y);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn structure_id(&self) -> StructureId {
        self.ids.structure_id()
    }

    fn values_id(&self) -> ValuesId {
        self.ids.values_id()
    }

    fn comm(&self) -> UniverseComm {
        self.halo.index.comm.clone()
    }

    fn dist_layout(&self) -> Option<&DistLayout> {
        Some(&self.layout)
    }

    fn format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Csr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_prefers_overlap_for_comm_heavy_metrics() {
        let metrics = DistributedPlanMetrics {
            n_local_rows: 4096,
            local_nnz: 80_000,
            local_diag_nnz: 30_000,
            ghost_nnz: 50_000,
            local_only_rows: 700,
            border_rows: 3396,
            halo_recv_volume: 12_000,
            halo_send_volume: 10_000,
        };
        let diag = choose_distributed_plan(&metrics, Some(SpmvKernel::Scalar));
        assert_eq!(diag.overlap_mode, HaloOverlapMode::Interior);
        assert_eq!(
            diag.kernel_strategy,
            DistLocalKernelStrategy::RowSplitScalar
        );
        assert!(diag.expected_communication_fraction > diag.expected_computation_fraction);
    }

    #[test]
    fn planner_prefers_local_diag_kernel_for_compute_heavy_metrics() {
        let metrics = DistributedPlanMetrics {
            n_local_rows: 4096,
            local_nnz: 80_000,
            local_diag_nnz: 76_000,
            ghost_nnz: 4_000,
            local_only_rows: 3600,
            border_rows: 496,
            halo_recv_volume: 500,
            halo_send_volume: 600,
        };
        let diag = choose_distributed_plan(&metrics, Some(SpmvKernel::Scalar));
        assert_eq!(diag.overlap_mode, HaloOverlapMode::Disabled);
        assert_eq!(
            diag.kernel_strategy,
            DistLocalKernelStrategy::LocalDiagSpmvPlan
        );
        assert!(diag.expected_computation_fraction > diag.expected_communication_fraction);
    }
}
