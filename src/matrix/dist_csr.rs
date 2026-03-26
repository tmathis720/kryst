//! DistCsrOp: canonical distributed CSR linear operator.
//!
//! This is the preferred representation for distributed sparse matrices. Other
//! distributed matrix APIs (e.g. `parcsr::*`) are secondary and will gradually
//! be reworked to build on this abstraction.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::algebra::bridge::BridgeScratch;
use crate::algebra::scalar::{KrystScalar, S};
use crate::error::KError;
use crate::matrix::dist::halo::{HaloIndexPlan, HaloPlan, HaloTuning};
use crate::matrix::dist::spmv_dist::RowRanges;
use crate::matrix::op::{ChangeIds, DistLayout, LinOp, StructureId, ValuesId};
use crate::matrix::parcsr::ParCsrMatrix;
use crate::matrix::sparse::CsrMatrix;
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
    border_row_ranges: Vec<Option<std::ops::Range<usize>>>,
    border_col_unified: Vec<usize>,
    border_vals: Vec<S>,
    halo: HaloPlan,
    overlap_mode: HaloOverlapMode,
    ids: ChangeIds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaloOverlapMode {
    Disabled,
    Interior,
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

        let mut border_row_ranges = vec![None; n_local];
        let mut border_col_unified = Vec::new();
        let mut border_vals = Vec::new();
        for i in 0..n_local {
            if row_is_local[i] {
                continue;
            }
            let start = border_col_unified.len();
            for idx in row_ptr[i]..row_ptr[i + 1] {
                let gcol = col_idx[idx];
                let owner = owner_of(gcol, halo.index.row_part.as_ref());
                let unified = if owner == rank {
                    gcol - row_start
                } else {
                    self_idx(&halo.index, gcol)
                };
                border_col_unified.push(unified);
                border_vals.push(vals[idx]);
            }
            let end = border_col_unified.len();
            border_row_ranges[i] = Some(start..end);
        }

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
            border_row_ranges,
            border_col_unified,
            border_vals,
            halo,
            overlap_mode: HaloOverlapMode::Interior,
            ids,
        })
    }

    pub fn set_halo_overlap_mode(&mut self, mode: HaloOverlapMode) {
        self.overlap_mode = mode;
    }

    /// Build a distributed operator from a [`ParCsrMatrix`].
    ///
    /// This merges the diagonal and off-process blocks into a single local CSR
    /// with global column indices before delegating to [`from_local_rows`].
    pub fn from_parcsr(par: &ParCsrMatrix) -> Result<Self, KError> {
        let n_local = par.local_n();
        let n_global = par.global_m;

        let mut row_ptr = Vec::with_capacity(n_local + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        row_ptr.push(0);

        for i in 0..n_local {
            let (diag_cols, diag_vals) = par.a_diag.row(i);
            let (off_cols, off_vals) = par.a_off.row(i);
            let mut entries = Vec::with_capacity(diag_cols.len() + off_cols.len());

            for (&local_j, &v) in diag_cols.iter().zip(diag_vals.iter()) {
                let gcol = *par
                    .colmap_owned
                    .get(local_j)
                    .ok_or_else(|| KError::InvalidInput("diag colmap missing entry".into()))?;
                entries.push((gcol, v));
            }
            for (&ghost_j, &v) in off_cols.iter().zip(off_vals.iter()) {
                let gcol = *par
                    .colmap_ghost
                    .get(ghost_j)
                    .ok_or_else(|| KError::InvalidInput("ghost colmap missing entry".into()))?;
                entries.push((gcol, v));
            }

            entries.sort_unstable_by_key(|(c, _)| *c);
            for (c, v) in entries {
                col_idx.push(c);
                vals.push(v);
            }
            row_ptr.push(col_idx.len());
        }

        let local_rows = CsrMatrix::from_csr(n_local, n_global, row_ptr, col_idx, vals);
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
        for row in 0..self.n_local {
            if let Some(range) = &self.border_row_ranges[row] {
                let mut idx = self.row_ptr[row];
                for slot in range.clone() {
                    self.border_vals[slot] = self.vals[idx];
                    idx += 1;
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

    fn spmv_local_only(&self, x: &[S], y: &mut [S]) {
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

    fn spmv_border(&self, x: &[S], y: &mut [S], ghost: &[S]) {
        if self.border.is_empty() {
            return;
        }
        let n_local = self.n_local;
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            y.par_iter_mut()
                .enumerate()
                .filter(|(row, _)| !self.row_is_local[*row])
                .for_each(|(row, slot)| {
                    if let Some(range) = &self.border_row_ranges[row] {
                        let mut acc = S::zero();
                        for k in range.clone() {
                            let col = self.border_col_unified[k];
                            let val = self.border_vals[k];
                            if col < n_local {
                                acc = acc + val * x[col];
                            } else {
                                acc = acc + val * ghost[col - n_local];
                            }
                        }
                        *slot = acc;
                    }
                });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for span in &self.border.spans {
                for row in span.clone() {
                    if let Some(range) = &self.border_row_ranges[row] {
                        let mut acc = S::zero();
                        for k in range.clone() {
                            let col = self.border_col_unified[k];
                            let val = self.border_vals[k];
                            if col < n_local {
                                acc = acc + val * x[col];
                            } else {
                                acc = acc + val * ghost[col - n_local];
                            }
                        }
                        y[row] = acc;
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
                    self.spmv_border(x, y, &ghost[..]);
                } else {
                    self.spmv_local_only(x, y);
                    self.spmv_border(x, y, &[]);
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
                    self.spmv_border(x, y, &ghost[..]);
                } else {
                    self.spmv_border(x, y, &[]);
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
