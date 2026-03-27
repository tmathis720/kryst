use crate::algebra::scalar::{KrystScalar, S, is_complex_scalar};
use crate::config::options::PcOptions;
use crate::context::pc_context::{PcFactory, PcType};
use crate::core::traits::SubmatrixExtract;
use crate::error::KError;
#[cfg(feature = "backend-faer")]
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::DistLayout;
use crate::matrix::op::LinOp;
use crate::matrix::op::{StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::utils::spgemm_with_drop_tol_generic;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
use crate::utils::convergence::{ConvergedReason, FailureReasonKind, NestedPcFailure};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct FieldSplitPc {
    block_sizes: Vec<usize>,
    block_spans: Vec<BlockSpan>,
    children: Vec<Box<dyn Preconditioner>>,
    split_type: FieldSplitType,
    full_matrix: Option<Arc<CsrMatrix<S>>>,
    block_matrices: Vec<Arc<CsrMatrix<S>>>,
    schur_blocks: Option<SchurBlocks>,
    schur_precondition_matrix: Option<Arc<CsrMatrix<S>>>,
    schur_apply_hook: Option<SchurApplyHook>,
    last_structure_id: Option<StructureId>,
    last_values_id: Option<ValuesId>,
    extraction_mode: BlockExtractionMode,
    comm_schedule: SplitCommSchedule,
    schur_approx_workflow: String,
    diagnostics: Mutex<Vec<SplitDiagnostics>>,
    apply_workspaces: Mutex<HashMap<ApplyWorkspaceKey, Vec<ApplyWorkspace>>>,
    all_children_local: bool,
}

type SchurApplyHook = Arc<dyn Fn(&[S], &mut [S]) -> Result<(), KError> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockSpan {
    start: usize,
    end: usize,
}

impl BlockSpan {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

#[derive(Debug, Clone, Copy)]
enum FieldSplitType {
    Additive,
    Multiplicative,
    Symmetric,
    Schur {
        factorization: SchurFactorization,
        precondition: SchurPrecondition,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SchurFactorization {
    Diag,
    Lower,
    Upper,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchurPrecondition {
    Self_,
    SelfP,
    Diag,
    A11,
    Full,
    FullMatFree,
    User,
}

#[derive(Debug, Clone)]
struct SchurBlocks {
    a12: Arc<CsrMatrix<S>>,
    a21: Arc<CsrMatrix<S>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockExtractionMode {
    Extract,
    Cached,
    ZeroCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitCommSchedule {
    Auto,
    LocalFirst,
    ExchangeFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SplitTypeKey {
    Additive,
    Multiplicative,
    Symmetric,
    Schur(SchurFactorization),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ApplyWorkspaceKey {
    split_type: SplitTypeKey,
    block_lens: Vec<usize>,
    n: usize,
}

#[derive(Debug, Clone, Default)]
struct ApplyWorkspace {
    y_accum: Vec<S>,
    residual: Vec<S>,
    ay: Vec<S>,
    block_outputs: Vec<Vec<S>>,
    schur_tmp0: Vec<S>,
    schur_tmp1: Vec<S>,
    schur_corr: Vec<S>,
}

#[derive(Debug, Clone, Default)]
pub struct SplitDiagnostics {
    pub apply_calls: usize,
    pub reduction_count: usize,
    pub local_work_time: Duration,
    pub exchange_time: Duration,
    pub allocation_free_applies: usize,
    pub local_first_apply_time: Duration,
    pub exchange_first_apply_time: Duration,
    pub comm_schedule_time_delta: Duration,
}

impl FieldSplitPc {
    fn materialize_csr(a: &dyn LinOp<S = S>) -> Result<Arc<CsrMatrix<S>>, KError> {
        #[cfg(feature = "backend-faer")]
        {
            return csr_from_linop(a, 0.0);
        }

        #[cfg(not(feature = "backend-faer"))]
        {
            if let Some(csr) = a.as_any().downcast_ref::<CsrMatrix<S>>() {
                return Ok(Arc::new(csr.clone()));
            }

            let (rows, cols) = a.dims();
            if rows != cols {
                return Err(KError::InvalidInput(
                    "FieldSplit setup requires a square operator".into(),
                ));
            }

            let mut row_ptr = Vec::with_capacity(rows + 1);
            let mut col_idx = Vec::new();
            let mut values = Vec::new();
            let mut ej = vec![S::zero(); cols];
            let mut col = vec![S::zero(); rows];
            row_ptr.push(0);
            let threshold = S::from_real(0.0).abs();
            let mut entries: Vec<Vec<(usize, S)>> = vec![Vec::new(); rows];
            for j in 0..cols {
                ej.fill(S::zero());
                ej[j] = S::one();
                a.try_matvec(&ej, &mut col)?;
                for i in 0..rows {
                    if col[i].abs() > threshold {
                        entries[i].push((j, col[i]));
                    }
                }
            }
            for row in entries {
                for (j, v) in row {
                    col_idx.push(j);
                    values.push(v);
                }
                row_ptr.push(col_idx.len());
            }

            Ok(Arc::new(CsrMatrix::from_csr(
                rows, cols, row_ptr, col_idx, values,
            )))
        }
    }

    pub fn new(
        block_sizes: Vec<usize>,
        child_pc_type: Option<String>,
        opts: PcOptions,
    ) -> Result<Self, KError> {
        let child_type = child_pc_type
            .as_deref()
            .map(PcType::from_str)
            .transpose()?
            .unwrap_or(PcType::Jacobi);
        let mut children = Vec::with_capacity(block_sizes.len());
        for (i, _) in block_sizes.iter().enumerate() {
            let mut child_opts = opts.clone();
            child_opts.pc_type = None;
            if let Some(scoped) = opts.fieldsplit_child_scoped_options(i) {
                child_opts.overlay_from(scoped.clone());
            }
            let scoped_type = child_opts
                .pc_type
                .as_deref()
                .map(PcType::from_str)
                .transpose()?;
            children.push(PcFactory::create_preconditioner(
                scoped_type.unwrap_or(child_type),
                Some(&child_opts),
            )?);
        }
        let all_children_local = children
            .iter()
            .all(|pc| pc.distributed_support() == PcDistributedSupport::LocalOnly);
        let split_type = Self::split_type_from_options(&opts)?;
        if matches!(split_type, FieldSplitType::Schur { .. }) && block_sizes.len() != 2 {
            return Err(KError::InvalidInput(
                "pc_fieldsplit_type=schur requires exactly two blocks".into(),
            ));
        }
        let split_count = block_sizes.len();
        Ok(Self {
            block_sizes,
            block_spans: Vec::new(),
            children,
            split_type,
            full_matrix: None,
            block_matrices: Vec::new(),
            schur_blocks: None,
            schur_precondition_matrix: None,
            schur_apply_hook: None,
            last_structure_id: None,
            last_values_id: None,
            extraction_mode: Self::extraction_mode_from_options(&opts)?,
            comm_schedule: Self::comm_schedule_from_options(&opts)?,
            schur_approx_workflow: opts.resolved_fieldsplit_schur_approx(),
            diagnostics: Mutex::new(vec![SplitDiagnostics::default(); split_count]),
            apply_workspaces: Mutex::new(HashMap::new()),
            all_children_local,
        })
    }

    fn extraction_mode_from_options(opts: &PcOptions) -> Result<BlockExtractionMode, KError> {
        match opts
            .pc_fieldsplit_extraction
            .as_deref()
            .unwrap_or("extract")
            .to_lowercase()
            .as_str()
        {
            "extract" => Ok(BlockExtractionMode::Extract),
            "cached" | "cache" => Ok(BlockExtractionMode::Cached),
            "zero_copy" | "zerocopy" | "view" => Ok(BlockExtractionMode::ZeroCopy),
            other => Err(KError::InvalidInput(format!(
                "unknown pc_fieldsplit_extraction: {other}"
            ))),
        }
    }

    fn split_type_from_options(opts: &PcOptions) -> Result<FieldSplitType, KError> {
        let kind = opts.resolved_fieldsplit_type();
        match kind.as_str() {
            "additive" | "diag" | "blockdiag" => Ok(FieldSplitType::Additive),
            "composite_additive" | "basic" => Ok(FieldSplitType::Additive),
            "multiplicative" | "mul" | "gs" | "gauss_seidel" => Ok(FieldSplitType::Multiplicative),
            "composite_multiplicative" => Ok(FieldSplitType::Multiplicative),
            "symmetric" | "sym" | "symmetric_multiplicative" => Ok(FieldSplitType::Symmetric),
            "composite_symmetric_multiplicative" | "multiplicative_symmetric" => {
                Ok(FieldSplitType::Symmetric)
            }
            "schur" => {
                let factorization = match opts.resolved_fieldsplit_schur_fact_type().as_str() {
                    "diag" => SchurFactorization::Diag,
                    "lower" => SchurFactorization::Lower,
                    "upper" => SchurFactorization::Upper,
                    "full" => SchurFactorization::Full,
                    other => {
                        return Err(KError::InvalidInput(format!(
                            "unknown pc_fieldsplit_schur_fact_type: {other}"
                        )));
                    }
                };
                let precondition = match opts.resolved_fieldsplit_schur_precondition().as_str() {
                    "self" => SchurPrecondition::Self_,
                    "selfp" | "self_p" => SchurPrecondition::SelfP,
                    "diag" => SchurPrecondition::Diag,
                    "a11" => SchurPrecondition::A11,
                    "full" => SchurPrecondition::Full,
                    "full_matfree" | "matfree" => SchurPrecondition::FullMatFree,
                    "user" => SchurPrecondition::User,
                    other => {
                        return Err(KError::InvalidInput(format!(
                            "unknown pc_fieldsplit_schur_precondition: {other}"
                        )));
                    }
                };
                Ok(FieldSplitType::Schur {
                    factorization,
                    precondition,
                })
            }
            other => Err(KError::InvalidInput(format!(
                "unknown pc_fieldsplit_type: {other}"
            ))),
        }
    }

    fn comm_schedule_from_options(opts: &PcOptions) -> Result<SplitCommSchedule, KError> {
        match opts.resolved_fieldsplit_comm_schedule().as_str() {
            "auto" => Ok(SplitCommSchedule::Auto),
            "local_first" | "local" => Ok(SplitCommSchedule::LocalFirst),
            "exchange_first" | "exchange" => Ok(SplitCommSchedule::ExchangeFirst),
            other => Err(KError::InvalidInput(format!(
                "unknown pc_fieldsplit_comm_schedule: {other}"
            ))),
        }
    }

    fn active_comm_schedule(&self) -> SplitCommSchedule {
        match self.comm_schedule {
            SplitCommSchedule::Auto
                if self.distributed_support() == PcDistributedSupport::Distributed =>
            {
                SplitCommSchedule::LocalFirst
            }
            SplitCommSchedule::Auto => SplitCommSchedule::ExchangeFirst,
            schedule => schedule,
        }
    }

    fn split_type_key(&self) -> SplitTypeKey {
        match self.split_type {
            FieldSplitType::Additive => SplitTypeKey::Additive,
            FieldSplitType::Multiplicative => SplitTypeKey::Multiplicative,
            FieldSplitType::Symmetric => SplitTypeKey::Symmetric,
            FieldSplitType::Schur { factorization, .. } => SplitTypeKey::Schur(factorization),
        }
    }

    fn apply_workspace_key(&self, n: usize) -> ApplyWorkspaceKey {
        ApplyWorkspaceKey {
            split_type: self.split_type_key(),
            block_lens: self.block_spans.iter().map(BlockSpan::len).collect(),
            n,
        }
    }

    fn build_workspace(&self, key: &ApplyWorkspaceKey) -> ApplyWorkspace {
        let mut block_outputs = Vec::with_capacity(key.block_lens.len());
        for len in &key.block_lens {
            block_outputs.push(vec![S::zero(); *len]);
        }
        let schur_len0 = key.block_lens.first().copied().unwrap_or(0);
        let schur_len1 = key.block_lens.get(1).copied().unwrap_or(0);
        ApplyWorkspace {
            y_accum: vec![S::zero(); key.n],
            residual: vec![S::zero(); key.n],
            ay: vec![S::zero(); key.n],
            block_outputs,
            schur_tmp0: vec![S::zero(); schur_len0],
            schur_tmp1: vec![S::zero(); schur_len1],
            schur_corr: vec![S::zero(); schur_len0],
        }
    }

    fn checkout_workspace(&self, n: usize) -> (ApplyWorkspaceKey, ApplyWorkspace, bool) {
        let key = self.apply_workspace_key(n);
        if let Ok(mut pools) = self.apply_workspaces.lock()
            && let Some(pool) = pools.get_mut(&key)
            && let Some(workspace) = pool.pop()
        {
            return (key, workspace, true);
        }
        let workspace = self.build_workspace(&key);
        (key, workspace, false)
    }

    fn checkin_workspace(&self, key: ApplyWorkspaceKey, workspace: ApplyWorkspace) {
        if let Ok(mut pools) = self.apply_workspaces.lock() {
            pools.entry(key).or_default().push(workspace);
        }
    }

    fn add_local_time(&self, idx: usize, elapsed: Duration) {
        if let Ok(mut diag) = self.diagnostics.lock()
            && let Some(d) = diag.get_mut(idx)
        {
            d.apply_calls += 1;
            d.local_work_time += elapsed;
        }
    }

    fn add_exchange_event(&self, idx: usize, elapsed: Duration) {
        if let Ok(mut diag) = self.diagnostics.lock()
            && let Some(d) = diag.get_mut(idx)
        {
            d.reduction_count += 1;
            d.exchange_time += elapsed;
        }
    }

    fn add_apply_schedule_stats(
        &self,
        schedule: SplitCommSchedule,
        elapsed: Duration,
        allocation_free: bool,
    ) {
        if let Ok(mut diag) = self.diagnostics.lock() {
            for (idx, span) in self.block_spans.iter().enumerate() {
                if span.len() == 0 {
                    continue;
                }
                if let Some(d) = diag.get_mut(idx) {
                    if allocation_free {
                        d.allocation_free_applies += 1;
                    }
                    match schedule {
                        SplitCommSchedule::LocalFirst => d.local_first_apply_time += elapsed,
                        SplitCommSchedule::ExchangeFirst => d.exchange_first_apply_time += elapsed,
                        SplitCommSchedule::Auto => {}
                    }
                    d.comm_schedule_time_delta = d
                        .local_first_apply_time
                        .abs_diff(d.exchange_first_apply_time);
                }
            }
        }
    }

    pub fn split_diagnostics(&self) -> Vec<SplitDiagnostics> {
        self.diagnostics
            .lock()
            .map(|d| d.clone())
            .unwrap_or_default()
    }

    fn block_spans_from_sizes(block_sizes: &[usize]) -> Vec<BlockSpan> {
        let mut spans = Vec::with_capacity(block_sizes.len());
        let mut off = 0usize;
        for size in block_sizes {
            let span = BlockSpan {
                start: off,
                end: off + *size,
            };
            spans.push(span);
            off = span.end;
        }
        spans
    }

    fn block_spans_from_sizes_with_layout(
        block_sizes: &[usize],
        local_n: usize,
        layout: Option<&DistLayout>,
    ) -> Result<Vec<BlockSpan>, KError> {
        if block_sizes.is_empty() {
            return Err(KError::InvalidInput(
                "pc_fieldsplit_block_sizes must contain at least one block".into(),
            ));
        }
        if block_sizes.contains(&0) {
            return Err(KError::InvalidInput(
                "pc_fieldsplit_block_sizes entries must all be > 0".into(),
            ));
        }
        if let Some(layout) = layout {
            if layout.row_end < layout.row_start {
                return Err(KError::InvalidInput(
                    "invalid distributed layout: row_end < row_start".into(),
                ));
            }
            if layout.row_end - layout.row_start != local_n {
                return Err(KError::InvalidInput(format!(
                    "distributed layout/local row mismatch: local_n={local_n}, layout_rows={}",
                    layout.row_end - layout.row_start
                )));
            }
        }

        let total: usize = block_sizes.iter().sum();
        if total == local_n {
            return Ok(Self::block_spans_from_sizes(block_sizes));
        }
        if let Some(layout) = layout {
            if total == layout.global_rows {
                let mut spans = Vec::with_capacity(block_sizes.len());
                let mut off = 0usize;
                for size in block_sizes {
                    let start = off;
                    let end = off + *size;
                    off = end;
                    let local_start = max(start, layout.row_start);
                    let local_end = min(end, layout.row_end);
                    if local_start >= local_end {
                        spans.push(BlockSpan { start: 0, end: 0 });
                    } else {
                        spans.push(BlockSpan {
                            start: local_start - layout.row_start,
                            end: local_end - layout.row_start,
                        });
                    }
                }
                return Ok(spans);
            }
            if total > local_n && total < layout.global_rows {
                return Err(KError::InvalidInput(format!(
                    "pc_fieldsplit_block_sizes appears mixed local/global (sum={total}, local={local_n}, global={}); provide all-local or all-global sizes",
                    layout.global_rows
                )));
            }
            return Err(KError::InvalidInput(format!(
                "pc_fieldsplit_block_sizes must sum to local ({local_n}) or global ({}) rows",
                layout.global_rows
            )));
        }
        Err(KError::InvalidInput(format!(
            "pc_fieldsplit_block_sizes must sum to matrix size ({local_n})"
        )))
    }

    fn extract_block_matrices(
        &self,
        csr: &CsrMatrix<S>,
        spans: &[BlockSpan],
    ) -> Vec<Arc<CsrMatrix<S>>> {
        if self.extraction_mode == BlockExtractionMode::Cached
            && spans.len() == self.block_spans.len()
            && spans == self.block_spans.as_slice()
            && !self.block_matrices.is_empty()
        {
            return self.block_matrices.clone();
        }
        spans
            .iter()
            .map(|span| {
                if self.extraction_mode == BlockExtractionMode::ZeroCopy
                    && span.start == 0
                    && span.end == csr.nrows()
                    && csr.nrows() == csr.ncols()
                {
                    return Arc::new(csr.clone());
                }
                let indices: Vec<usize> = (span.start..span.end).collect();
                Arc::new(csr.extract_submatrix(&indices, &indices))
            })
            .collect()
    }

    fn restrict_rhs<'a>(&self, x: &'a [S], span: BlockSpan) -> &'a [S] {
        &x[span.start..span.end]
    }

    fn extract_schur_blocks(&self, csr: &CsrMatrix<S>, spans: &[BlockSpan]) -> Option<SchurBlocks> {
        if spans.len() != 2 {
            return None;
        }
        let n = csr.nrows();
        if spans.iter().any(|s| s.end > n || s.start > s.end) {
            return None;
        }
        let rows_0: Vec<usize> = (spans[0].start..spans[0].end).collect();
        let rows_1: Vec<usize> = (spans[1].start..spans[1].end).collect();
        let a12 = Arc::new(csr.extract_submatrix(&rows_0, &rows_1));
        let a21 = Arc::new(csr.extract_submatrix(&rows_1, &rows_0));
        Some(SchurBlocks { a12, a21 })
    }

    fn schur_full_approx(
        &self,
        a11: &CsrMatrix<S>,
        a22: &CsrMatrix<S>,
        schur: &SchurBlocks,
        workflow: &str,
    ) -> Result<CsrMatrix<S>, KError> {
        match workflow {
            "diag" => self.schur_diag_approx(a11, a22, schur),
            "full" => {
                // Placeholder for stronger Schur workflows; currently uses the same
                // structurally complete sparse path as diag for determinism.
                self.schur_diag_approx(a11, a22, schur)
            }
            "distributed_diag" | "dist_diag" => self.schur_diag_approx(a11, a22, schur),
            "distributed_full" | "dist_full" => self.schur_diag_approx(a11, a22, schur),
            other => Err(KError::InvalidInput(format!(
                "unknown pc_fieldsplit_schur_approx: {other}"
            ))),
        }
    }

    fn schur_diag_approx(
        &self,
        a11: &CsrMatrix<S>,
        a22: &CsrMatrix<S>,
        schur: &SchurBlocks,
    ) -> Result<CsrMatrix<S>, KError> {
        let n1 = a11.nrows();
        let mut diag_inv = vec![S::zero(); n1];
        for i in 0..n1 {
            let rs = a11.row_ptr()[i];
            let re = a11.row_ptr()[i + 1];
            let mut aii = S::zero();
            for p in rs..re {
                if a11.col_idx()[p] == i {
                    aii = a11.values()[p];
                    break;
                }
            }
            diag_inv[i] = if aii.abs() > 1e-14 {
                aii.inv()
            } else {
                S::zero()
            };
        }
        let a12 = schur.a12.as_ref();
        let mut scaled_vals = vec![S::zero(); a12.values().len()];
        for row in 0..a12.nrows() {
            let rs = a12.row_ptr()[row];
            let re = a12.row_ptr()[row + 1];
            for p in rs..re {
                scaled_vals[p] = a12.values()[p] * diag_inv[row];
            }
        }
        let scaled_a12 = CsrMatrix::from_csr(
            a12.nrows(),
            a12.ncols(),
            a12.row_ptr().to_vec(),
            a12.col_idx().to_vec(),
            scaled_vals,
        );
        let product = spgemm_with_drop_tol_generic(schur.a21.as_ref(), &scaled_a12, 1e-12)?;
        Self::csr_subtract(a22, &product, 1e-12)
    }

    fn complex_safe_schur_precondition(
        &self,
        precondition: SchurPrecondition,
        a22: &Arc<CsrMatrix<S>>,
        schur_mat: Arc<CsrMatrix<S>>,
    ) -> (Option<Arc<CsrMatrix<S>>>, Option<SchurApplyHook>) {
        if !is_complex_scalar::<S>() {
            return match precondition {
                SchurPrecondition::Full => (Some(schur_mat), None),
                SchurPrecondition::FullMatFree | SchurPrecondition::User => {
                    let schur = schur_mat.clone();
                    (
                        None,
                        Some(Arc::new(move |rhs: &[S], out: &mut [S]| {
                            schur.try_spmv(rhs, out)
                        })),
                    )
                }
                _ => (None, None),
            };
        }
        match precondition {
            SchurPrecondition::Full
            | SchurPrecondition::FullMatFree
            | SchurPrecondition::User
            | SchurPrecondition::Self_
            | SchurPrecondition::SelfP => {
                // Complex-safe fallback mirrors PETSc's preference for stable Schur-side
                // composition semantics when inexpensive real-valued approximations are
                // unavailable.
                (Some(a22.clone()), None)
            }
            _ => (None, None),
        }
    }

    fn csr_subtract(
        a: &CsrMatrix<S>,
        b: &CsrMatrix<S>,
        drop_tol: <S as KrystScalar>::Real,
    ) -> Result<CsrMatrix<S>, KError> {
        if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
            return Err(KError::InvalidInput(format!(
                "csr_subtract dimension mismatch: A={}x{}, B={}x{}",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols()
            )));
        }
        let nrows = a.nrows();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for row in 0..nrows {
            let mut ia = a.row_ptr()[row];
            let mut ib = b.row_ptr()[row];
            let a_end = a.row_ptr()[row + 1];
            let b_end = b.row_ptr()[row + 1];
            while ia < a_end || ib < b_end {
                let (col, val) = if ib >= b_end || (ia < a_end && a.col_idx()[ia] < b.col_idx()[ib])
                {
                    let col = a.col_idx()[ia];
                    let val = a.values()[ia];
                    ia += 1;
                    (col, val)
                } else if ia >= a_end || b.col_idx()[ib] < a.col_idx()[ia] {
                    let col = b.col_idx()[ib];
                    let val = -b.values()[ib];
                    ib += 1;
                    (col, val)
                } else {
                    let col = a.col_idx()[ia];
                    let val = a.values()[ia] - b.values()[ib];
                    ia += 1;
                    ib += 1;
                    (col, val)
                };
                if val.abs() > drop_tol {
                    col_idx.push(col);
                    values.push(val);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Ok(CsrMatrix::from_csr(
            nrows,
            a.ncols(),
            row_ptr,
            col_idx,
            values,
        ))
    }

    fn update_residual(&self, x: &[S], y: &[S], r: &mut [S], ay: &mut [S]) -> Result<(), KError> {
        let a = self.full_matrix.as_ref().ok_or_else(|| {
            KError::InvalidInput("fieldsplit multiplicative requires CSR matrix".into())
        })?;
        a.try_spmv(y, ay)?;
        for i in 0..r.len() {
            r[i] = x[i] - ay[i];
        }
        Ok(())
    }

    fn nested_apply_failure(&self, idx: usize, side: PcSide, stage: &str, err: &KError) -> KError {
        match err {
            KError::NestedPcFailed(inner) => KError::NestedPcFailed(NestedPcFailure {
                component: "pc_fieldsplit",
                reason: inner.reason,
                iterations: inner.iterations,
                final_norm: inner.final_norm.clone(),
                residual_history_summary: inner.residual_history_summary.clone(),
                detail: format!(
                    "component=pc_fieldsplit split={:?} block={idx} side={side:?} stage={stage} inner_component={} inner_reason={} inner_detail={}",
                    self.split_type, inner.component, inner.reason, inner.detail
                ),
            }),
            _ => KError::NestedPcFailed(NestedPcFailure {
                component: "pc_fieldsplit",
                reason: ConvergedReason::from_failure_kind(FailureReasonKind::PcApply),
                iterations: 0,
                final_norm: None,
                residual_history_summary: None,
                detail: format!(
                    "component=pc_fieldsplit split={:?} block={idx} side={side:?} stage={stage} error={err}",
                    self.split_type
                ),
            }),
        }
    }

    fn apply_child(
        &self,
        idx: usize,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        stage: &str,
    ) -> Result<(), KError> {
        self.children[idx]
            .apply(side, x, y)
            .map_err(|err| self.nested_apply_failure(idx, side, stage, &err))
    }

    fn apply_additive(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        workspace: &mut ApplyWorkspace,
    ) -> Result<(), KError> {
        y.fill(S::zero());
        let mut order: Vec<usize> = (0..self.block_spans.len()).collect();
        if self.active_comm_schedule() == SplitCommSchedule::LocalFirst {
            order.sort_by_key(|idx| self.block_spans[*idx].len());
        }
        for idx in order {
            let span = self.block_spans[idx];
            if span.len() == 0 {
                continue;
            }
            let zout = &mut workspace.block_outputs[idx];
            zout.fill(S::zero());
            let start = Instant::now();
            self.apply_child(idx, side, self.restrict_rhs(x, span), zout, "additive")?;
            self.add_local_time(idx, start.elapsed());
            for (yi, zi) in y[span.start..span.end].iter_mut().zip(zout.iter()) {
                *yi += *zi;
            }
        }
        Ok(())
    }

    fn apply_multiplicative(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        workspace: &mut ApplyWorkspace,
    ) -> Result<(), KError> {
        let y_accum = &mut workspace.y_accum;
        y_accum.fill(S::zero());
        let residual = &mut workspace.residual;
        residual.copy_from_slice(x);
        for (idx, (span, _)) in self
            .block_spans
            .iter()
            .zip(self.children.iter())
            .enumerate()
        {
            if span.len() == 0 {
                continue;
            }
            let zout = &mut workspace.block_outputs[idx];
            zout.fill(S::zero());
            self.apply_child(
                idx,
                side,
                self.restrict_rhs(residual, *span),
                zout,
                "multiplicative",
            )?;
            for (i, val) in zout.iter().enumerate() {
                y_accum[span.start + i] += *val;
            }
            self.update_residual(x, y_accum, residual, &mut workspace.ay)?;
        }
        y.copy_from_slice(y_accum);
        Ok(())
    }

    fn apply_symmetric(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        workspace: &mut ApplyWorkspace,
    ) -> Result<(), KError> {
        let y_accum = &mut workspace.y_accum;
        y_accum.fill(S::zero());
        let residual = &mut workspace.residual;
        residual.copy_from_slice(x);
        for (idx, (span, _)) in self
            .block_spans
            .iter()
            .zip(self.children.iter())
            .enumerate()
        {
            if span.len() == 0 {
                continue;
            }
            let zout = &mut workspace.block_outputs[idx];
            zout.fill(S::zero());
            self.apply_child(
                idx,
                side,
                self.restrict_rhs(residual, *span),
                zout,
                "symmetric_forward",
            )?;
            for (i, val) in zout.iter().enumerate() {
                y_accum[span.start + i] += *val;
            }
            self.update_residual(x, y_accum, residual, &mut workspace.ay)?;
        }
        for (idx, (span, _)) in self
            .block_spans
            .iter()
            .zip(self.children.iter())
            .enumerate()
            .rev()
        {
            if span.len() == 0 {
                continue;
            }
            let zout = &mut workspace.block_outputs[idx];
            zout.fill(S::zero());
            self.apply_child(
                idx,
                side,
                self.restrict_rhs(residual, *span),
                zout,
                "symmetric_backward",
            )?;
            for (i, val) in zout.iter().enumerate() {
                y_accum[span.start + i] += *val;
            }
            self.update_residual(x, y_accum, residual, &mut workspace.ay)?;
        }
        y.copy_from_slice(y_accum);
        Ok(())
    }

    fn apply_schur(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        workspace: &mut ApplyWorkspace,
    ) -> Result<(), KError> {
        let FieldSplitType::Schur { factorization, .. } = self.split_type else {
            return Err(KError::InvalidInput(
                "apply_schur called for non-schur fieldsplit".into(),
            ));
        };
        let spans = &self.block_spans;
        let schur = self
            .schur_blocks
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("missing Schur blocks for fieldsplit".into()))?;
        let span0 = spans
            .get(0)
            .ok_or_else(|| KError::InvalidInput("missing first Schur block".into()))?;
        let span1 = spans
            .get(1)
            .ok_or_else(|| KError::InvalidInput("missing second Schur block".into()))?;

        let x1 = &x[span0.start..span0.end];
        let x2 = &x[span1.start..span1.end];

        let (left, right) = workspace.block_outputs.split_at_mut(1);
        let y1 = &mut left[0];
        y1.fill(S::zero());
        let y2 = &mut right[0];
        y2.fill(S::zero());

        match factorization {
            SchurFactorization::Diag => {
                self.apply_child(0, side, x1, y1, "schur_diag_a11")?;
                self.apply_child(1, side, x2, y2, "schur_diag_s")?;
            }
            SchurFactorization::Lower => {
                self.apply_child(0, side, x1, y1, "schur_lower_a11")?;
                let tmp2 = &mut workspace.schur_tmp1;
                tmp2.fill(S::zero());
                let exch = Instant::now();
                schur.a21.try_spmv(y1, tmp2)?;
                self.add_exchange_event(1, exch.elapsed());
                for i in 0..tmp2.len() {
                    tmp2[i] = x2[i] - tmp2[i];
                }
                if let Some(hook) = &self.schur_apply_hook {
                    hook(tmp2, y2)?;
                } else {
                    self.apply_child(1, side, tmp2, y2, "schur_lower_s")?;
                }
            }
            SchurFactorization::Upper => {
                self.apply_child(1, side, x2, y2, "schur_upper_s")?;
                let tmp1 = &mut workspace.schur_tmp0;
                tmp1.fill(S::zero());
                let exch = Instant::now();
                schur.a12.try_spmv(y2, tmp1)?;
                self.add_exchange_event(0, exch.elapsed());
                for i in 0..tmp1.len() {
                    tmp1[i] = x1[i] - tmp1[i];
                }
                self.apply_child(0, side, tmp1, y1, "schur_upper_a11")?;
            }
            SchurFactorization::Full => {
                self.apply_child(0, side, x1, y1, "schur_full_a11")?;
                let tmp2 = &mut workspace.schur_tmp1;
                tmp2.fill(S::zero());
                let exch = Instant::now();
                schur.a21.try_spmv(y1, tmp2)?;
                self.add_exchange_event(1, exch.elapsed());
                for i in 0..tmp2.len() {
                    tmp2[i] = x2[i] - tmp2[i];
                }
                if let Some(hook) = &self.schur_apply_hook {
                    hook(tmp2, y2)?;
                } else {
                    self.apply_child(1, side, tmp2, y2, "schur_full_s")?;
                }
                let tmp1 = &mut workspace.schur_tmp0;
                tmp1.fill(S::zero());
                let exch = Instant::now();
                schur.a12.try_spmv(y2, tmp1)?;
                self.add_exchange_event(0, exch.elapsed());
                let corr = &mut workspace.schur_corr;
                corr.fill(S::zero());
                self.apply_child(0, side, tmp1, corr, "schur_full_correction")?;
                for i in 0..y1.len() {
                    y1[i] -= corr[i];
                }
            }
        }

        y[span0.start..span0.end].copy_from_slice(y1.as_slice());
        y[span1.start..span1.end].copy_from_slice(y2.as_slice());
        Ok(())
    }
}

impl Preconditioner for FieldSplitPc {
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        if self.last_structure_id == Some(a.structure_id())
            && self.last_values_id == Some(a.values_id())
            && !self.block_spans.is_empty()
        {
            return Ok(());
        }
        let n = a.dims().0;
        let spans =
            Self::block_spans_from_sizes_with_layout(&self.block_sizes, n, a.dist_layout())?;
        if let Some(layout) = a.dist_layout()
            && spans.iter().map(BlockSpan::len).sum::<usize>() != n
            && self.block_sizes.iter().sum::<usize>() == layout.global_rows
        {
            return Err(KError::InvalidInput(
                "fieldsplit local block spans do not cover local rows; check global block sizes against ownership layout".into(),
            ));
        }
        let csr = Self::materialize_csr(a)?;
        let block_mats = self.extract_block_matrices(csr.as_ref(), &spans);
        let mut schur_precondition_matrix = None;
        let mut schur_apply_hook: Option<SchurApplyHook> = None;
        if let FieldSplitType::Schur { precondition, .. } = self.split_type {
            if precondition == SchurPrecondition::Diag && is_complex_scalar::<S>() {
                return Err(KError::InvalidInput(
                    "pc_fieldsplit_schur_precondition=diag is not supported for complex scalars"
                        .into(),
                ));
            }
            if precondition == SchurPrecondition::A11 && block_mats.len() >= 2 {
                let a11_dims = block_mats[0].dims();
                let a22_dims = block_mats[1].dims();
                if a11_dims != a22_dims {
                    return Err(KError::InvalidInput(format!(
                        "pc_fieldsplit_schur_precondition=a11 requires matching block sizes: A11={a11_dims:?}, A22={a22_dims:?}"
                    )));
                }
            }
            if precondition == SchurPrecondition::Diag {
                let schur = self
                    .extract_schur_blocks(csr.as_ref(), &spans)
                    .ok_or_else(|| KError::InvalidInput("missing Schur blocks".into()))?;
                let a11 = block_mats
                    .get(0)
                    .ok_or_else(|| KError::InvalidInput("missing A11 block".into()))?;
                let a22 = block_mats
                    .get(1)
                    .ok_or_else(|| KError::InvalidInput("missing A22 block".into()))?;
                let schur_mat = self.schur_diag_approx(a11.as_ref(), a22.as_ref(), &schur)?;
                schur_precondition_matrix = Some(Arc::new(schur_mat));
            } else if matches!(
                precondition,
                SchurPrecondition::Full
                    | SchurPrecondition::FullMatFree
                    | SchurPrecondition::User
                    | SchurPrecondition::Self_
                    | SchurPrecondition::SelfP
            ) {
                let schur = self
                    .extract_schur_blocks(csr.as_ref(), &spans)
                    .ok_or_else(|| KError::InvalidInput("missing Schur blocks".into()))?;
                let a22 = block_mats
                    .get(1)
                    .ok_or_else(|| KError::InvalidInput("missing A22 block".into()))?;
                let a11 = block_mats
                    .first()
                    .ok_or_else(|| KError::InvalidInput("missing A11 block".into()))?;
                let schur_mat = Arc::new(self.schur_full_approx(
                    a11.as_ref(),
                    a22.as_ref(),
                    &schur,
                    self.schur_approx_workflow.as_str(),
                )?);
                let (precondition_mat, apply_hook) =
                    self.complex_safe_schur_precondition(precondition, a22, schur_mat.clone());
                schur_precondition_matrix = precondition_mat;
                schur_apply_hook = apply_hook;
            }
        }
        for (idx, child) in self.children.iter_mut().enumerate() {
            let block = match self.split_type {
                FieldSplitType::Schur {
                    precondition: SchurPrecondition::A11,
                    ..
                } if idx == 1 => block_mats
                    .get(0)
                    .ok_or_else(|| KError::InvalidInput("missing A11 block".into()))?,
                FieldSplitType::Schur {
                    precondition: SchurPrecondition::Diag,
                    ..
                } if idx == 1 => schur_precondition_matrix
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("missing Schur approximation".into()))?,
                FieldSplitType::Schur {
                    precondition:
                        SchurPrecondition::Full
                        | SchurPrecondition::FullMatFree
                        | SchurPrecondition::User
                        | SchurPrecondition::Self_
                        | SchurPrecondition::SelfP,
                    ..
                } if idx == 1 && schur_precondition_matrix.is_some() => schur_precondition_matrix
                    .as_ref()
                    .ok_or_else(|| KError::InvalidInput("missing Schur approximation".into()))?,
                _ => block_mats
                    .get(idx)
                    .ok_or_else(|| KError::InvalidInput("missing fieldsplit block".into()))?,
            };
            child.setup(block.as_ref())?;
        }
        self.block_spans = spans;
        self.full_matrix = Some(csr);
        self.block_matrices = block_mats;
        self.schur_blocks = match self.split_type {
            FieldSplitType::Schur { .. } => {
                self.extract_schur_blocks(self.full_matrix.as_ref().unwrap(), &self.block_spans)
            }
            _ => None,
        };
        self.schur_precondition_matrix = schur_precondition_matrix;
        self.schur_apply_hook = schur_apply_hook;
        if let Ok(mut d) = self.diagnostics.lock() {
            *d = vec![SplitDiagnostics::default(); self.block_spans.len()];
        }
        if let Ok(mut pools) = self.apply_workspaces.lock() {
            pools.clear();
        }
        self.last_structure_id = Some(a.structure_id());
        self.last_values_id = Some(a.values_id());
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(
                "fieldsplit input/output length mismatch".into(),
            ));
        }
        let schedule = self.active_comm_schedule();
        let (workspace_key, mut workspace, allocation_free) = self.checkout_workspace(x.len());
        let start = Instant::now();
        let result = match self.split_type {
            FieldSplitType::Additive => self.apply_additive(side, x, y, &mut workspace),
            FieldSplitType::Multiplicative => self.apply_multiplicative(side, x, y, &mut workspace),
            FieldSplitType::Symmetric => self.apply_symmetric(side, x, y, &mut workspace),
            FieldSplitType::Schur { .. } => self.apply_schur(side, x, y, &mut workspace),
        };
        let elapsed = start.elapsed();
        self.checkin_workspace(workspace_key, workspace);
        self.add_apply_schedule_stats(schedule, elapsed, allocation_free);
        result
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        if self.all_children_local {
            PcDistributedSupport::LocalOnly
        } else {
            PcDistributedSupport::Distributed
        }
    }
}

#[cfg(all(test, feature = "backend-faer", not(feature = "complex")))]
mod tests {
    use super::*;
    use crate::config::options::KspOptions;
    use crate::context::ksp_context::{KspContext, SolverType};
    use crate::matrix::op::DenseOp;
    use faer::Mat;
    use std::sync::Arc;
    use std::time::Duration;

    fn tri_diag_2x2_blocks(n: usize) -> Arc<DenseOp<f64>> {
        let m = Mat::<f64>::from_fn(n, n, |i, j| {
            if i == j {
                3.5
            } else if (i as isize - j as isize).abs() == 1 {
                -1.0
            } else if (i as isize - j as isize).abs() == 2 {
                -0.2
            } else {
                0.0
            }
        });
        Arc::new(DenseOp::new(Arc::new(m)))
    }

    #[test]
    fn fieldsplit_schur_full_composition_solves_outer_system() {
        let a = tri_diag_2x2_blocks(12);
        let n = a.dims().0;
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];

        let mut ksp = KspContext::new();
        ksp.set_type(SolverType::Gmres).unwrap();
        let ksp_opts = KspOptions {
            maxits: Some(40),
            rtol: Some(1e-8),
            ..Default::default()
        };
        let pc_opts = PcOptions {
            pc_type: Some("fieldsplit".into()),
            pc_fieldsplit_block_sizes: Some(vec![6, 6]),
            pc_fieldsplit_type: Some("schur".into()),
            pc_fieldsplit_schur_fact_type: Some("full".into()),
            pc_fieldsplit_schur_precondition: Some("full".into()),
            pc_fieldsplit_prefixes: Some(vec![
                "pc_fieldsplit_0_".into(),
                "pc_fieldsplit_1_".into(),
            ]),
            scoped_children: vec![
                (
                    "pc_fieldsplit_0_".into(),
                    Box::new(PcOptions {
                        pc_type: Some("jacobi".into()),
                        ..Default::default()
                    }),
                ),
                (
                    "pc_fieldsplit_1_".into(),
                    Box::new(PcOptions {
                        pc_type: Some("none".into()),
                        ..Default::default()
                    }),
                ),
            ],
            ..Default::default()
        };

        ksp.set_from_all_options(&ksp_opts, &pc_opts).unwrap();
        ksp.set_operators(a, None);
        let stats = ksp.solve(&b, &mut x).unwrap();
        assert!(stats.reason.is_converged());
    }

    #[test]
    fn fieldsplit_distributed_support_is_local_when_children_local() {
        let pc = FieldSplitPc::new(vec![2, 2], Some("none".into()), PcOptions::default()).unwrap();
        assert_eq!(pc.distributed_support(), PcDistributedSupport::LocalOnly);
    }

    #[test]
    fn fieldsplit_distributed_support_is_distributed_for_mixed_children() {
        let opts = PcOptions {
            pc_fieldsplit_prefixes: Some(vec![
                "pc_fieldsplit_0_".into(),
                "pc_fieldsplit_1_".into(),
            ]),
            scoped_children: vec![
                (
                    "pc_fieldsplit_0_".into(),
                    Box::new(PcOptions {
                        pc_type: Some("ksp".into()),
                        ..Default::default()
                    }),
                ),
                (
                    "pc_fieldsplit_1_".into(),
                    Box::new(PcOptions {
                        pc_type: Some("jacobi".into()),
                        ..Default::default()
                    }),
                ),
            ],
            ..Default::default()
        };
        let pc = FieldSplitPc::new(vec![2, 2], Some("jacobi".into()), opts).unwrap();
        assert_eq!(pc.distributed_support(), PcDistributedSupport::Distributed);
    }

    #[test]
    fn fieldsplit_workspace_reuse_preserves_outputs() {
        let a = tri_diag_2x2_blocks(8);
        let mut pc = FieldSplitPc::new(
            vec![4, 4],
            Some("jacobi".into()),
            PcOptions {
                pc_fieldsplit_type: Some("symmetric".into()),
                pc_fieldsplit_comm_schedule: Some("local_first".into()),
                ..Default::default()
            },
        )
        .unwrap();
        pc.setup(a.as_ref()).unwrap();

        let rhs = vec![1.0; 8];
        let mut y_first = vec![0.0; 8];
        let mut y_second = vec![0.0; 8];
        pc.apply(PcSide::Left, &rhs, &mut y_first).unwrap();
        pc.apply(PcSide::Left, &rhs, &mut y_second).unwrap();
        assert_eq!(y_first, y_second);

        let diag = pc.split_diagnostics();
        assert!(diag.iter().all(|d| d.allocation_free_applies >= 1));
    }

    #[test]
    fn fieldsplit_diagnostics_report_schedule_delta() {
        let a = tri_diag_2x2_blocks(8);
        let rhs = vec![1.0; 8];

        let mut local_pc = FieldSplitPc::new(
            vec![4, 4],
            Some("jacobi".into()),
            PcOptions {
                pc_fieldsplit_type: Some("additive".into()),
                pc_fieldsplit_comm_schedule: Some("local_first".into()),
                ..Default::default()
            },
        )
        .unwrap();
        local_pc.setup(a.as_ref()).unwrap();
        let mut y_local = vec![0.0; 8];
        local_pc.apply(PcSide::Left, &rhs, &mut y_local).unwrap();

        let mut exchange_pc = FieldSplitPc::new(
            vec![4, 4],
            Some("jacobi".into()),
            PcOptions {
                pc_fieldsplit_type: Some("additive".into()),
                pc_fieldsplit_comm_schedule: Some("exchange_first".into()),
                ..Default::default()
            },
        )
        .unwrap();
        exchange_pc.setup(a.as_ref()).unwrap();
        let mut y_exchange = vec![0.0; 8];
        exchange_pc
            .apply(PcSide::Left, &rhs, &mut y_exchange)
            .unwrap();
        assert_eq!(y_local, y_exchange);

        let local_diag = local_pc.split_diagnostics();
        assert!(
            local_diag
                .iter()
                .all(|d| d.local_first_apply_time > Duration::ZERO)
        );
        assert!(
            local_diag
                .iter()
                .all(|d| d.comm_schedule_time_delta > Duration::ZERO)
        );

        let exchange_diag = exchange_pc.split_diagnostics();
        assert!(
            exchange_diag
                .iter()
                .all(|d| d.exchange_first_apply_time > Duration::ZERO)
        );
        assert!(
            exchange_diag
                .iter()
                .all(|d| d.comm_schedule_time_delta > Duration::ZERO)
        );
    }
}
