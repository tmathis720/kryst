use crate::algebra::scalar::{KrystScalar, S, is_complex_scalar};
use crate::config::options::PcOptions;
use crate::context::pc_context::{PcFactory, PcType};
use crate::core::traits::SubmatrixExtract;
use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::DistLayout;
use crate::matrix::op::LinOp;
use crate::matrix::op::{StructureId, ValuesId};
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::utils::spgemm_with_drop_tol_generic;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
use std::cmp::{max, min};
use std::str::FromStr;
use std::sync::Arc;

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

#[derive(Debug, Clone, Copy)]
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

impl FieldSplitPc {
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
        let kind = opts
            .pc_fieldsplit_type
            .as_deref()
            .unwrap_or("additive")
            .to_lowercase();
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
                let factorization = match opts
                    .pc_fieldsplit_schur_fact_type
                    .as_deref()
                    .unwrap_or("full")
                    .to_lowercase()
                    .as_str()
                {
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
                let precondition = match opts
                    .pc_fieldsplit_schur_precondition
                    .as_deref()
                    .unwrap_or("self")
                    .to_lowercase()
                    .as_str()
                {
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
    ) -> Result<CsrMatrix<S>, KError> {
        // Reuse the reusable block-factorization path with a higher-fidelity approximation
        // placeholder; currently this delegates to the diagonal-inverse A11 approximation.
        self.schur_diag_approx(a11, a22, schur)
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
        let mut scaled_vals = Vec::with_capacity(a12.values().len());
        for row in 0..a12.nrows() {
            let rs = a12.row_ptr()[row];
            let re = a12.row_ptr()[row + 1];
            for p in rs..re {
                scaled_vals.push(a12.values()[p] * diag_inv[row]);
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

    fn update_residual(&self, x: &[S], y: &[S], r: &mut [S]) -> Result<(), KError> {
        let a = self.full_matrix.as_ref().ok_or_else(|| {
            KError::InvalidInput("fieldsplit multiplicative requires CSR matrix".into())
        })?;
        let mut ay = vec![S::zero(); y.len()];
        a.try_spmv(y, &mut ay)?;
        for i in 0..r.len() {
            r[i] = x[i] - ay[i];
        }
        Ok(())
    }

    fn apply_additive(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        y.fill(S::zero());
        for (idx, (span, child)) in self
            .block_spans
            .iter()
            .zip(self.children.iter())
            .enumerate()
        {
            if span.len() == 0 {
                continue;
            }
            let mut zout = vec![S::zero(); span.len()];
            child
                .apply(side, self.restrict_rhs(x, *span), &mut zout)
                .map_err(|err| {
                    KError::PcFailed(format!(
                        "fieldsplit additive apply failed for block {idx} ({:?}): {err}",
                        self.split_type
                    ))
                })?;
            for (yi, zi) in y[span.start..span.end].iter_mut().zip(zout.iter()) {
                *yi += *zi;
            }
        }
        Ok(())
    }

    fn apply_multiplicative(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let n = x.len();
        let mut y_accum = vec![S::zero(); n];
        let mut residual = x.to_vec();
        for (idx, (span, child)) in self
            .block_spans
            .iter()
            .zip(self.children.iter())
            .enumerate()
        {
            if span.len() == 0 {
                continue;
            }
            let mut zout = vec![S::zero(); span.len()];
            child
                .apply(side, self.restrict_rhs(&residual, *span), &mut zout)
                .map_err(|err| {
                    KError::PcFailed(format!(
                        "fieldsplit multiplicative apply failed for block {idx} ({:?}): {err}",
                        self.split_type
                    ))
                })?;
            for (i, val) in zout.iter().enumerate() {
                y_accum[span.start + i] += *val;
            }
            self.update_residual(x, &y_accum, &mut residual)?;
        }
        y.copy_from_slice(&y_accum);
        Ok(())
    }

    fn apply_symmetric(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let n = x.len();
        let mut y_accum = vec![S::zero(); n];
        let mut residual = x.to_vec();
        for (idx, (span, child)) in self
            .block_spans
            .iter()
            .zip(self.children.iter())
            .enumerate()
        {
            if span.len() == 0 {
                continue;
            }
            let mut zout = vec![S::zero(); span.len()];
            child
                .apply(side, self.restrict_rhs(&residual, *span), &mut zout)
                .map_err(|err| {
                    KError::PcFailed(format!(
                        "fieldsplit symmetric-forward apply failed for block {idx} ({:?}): {err}",
                        self.split_type
                    ))
                })?;
            for (i, val) in zout.iter().enumerate() {
                y_accum[span.start + i] += *val;
            }
            self.update_residual(x, &y_accum, &mut residual)?;
        }
        for (idx, (span, child)) in self
            .block_spans
            .iter()
            .zip(self.children.iter())
            .enumerate()
            .rev()
        {
            if span.len() == 0 {
                continue;
            }
            let mut zout = vec![S::zero(); span.len()];
            child
                .apply(side, self.restrict_rhs(&residual, *span), &mut zout)
                .map_err(|err| {
                    KError::PcFailed(format!(
                        "fieldsplit symmetric-backward apply failed for block {idx} ({:?}): {err}",
                        self.split_type
                    ))
                })?;
            for (i, val) in zout.iter().enumerate() {
                y_accum[span.start + i] += *val;
            }
            self.update_residual(x, &y_accum, &mut residual)?;
        }
        y.copy_from_slice(&y_accum);
        Ok(())
    }

    fn apply_schur(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
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

        let mut y1 = vec![S::zero(); span0.len()];
        let mut y2 = vec![S::zero(); span1.len()];

        match factorization {
            SchurFactorization::Diag => {
                self.children[0].apply(side, x1, &mut y1)?;
                self.children[1].apply(side, x2, &mut y2)?;
            }
            SchurFactorization::Lower => {
                self.children[0].apply(side, x1, &mut y1)?;
                let mut tmp2 = vec![S::zero(); span1.len()];
                schur.a21.try_spmv(&y1, &mut tmp2)?;
                for i in 0..tmp2.len() {
                    tmp2[i] = x2[i] - tmp2[i];
                }
                if let Some(hook) = &self.schur_apply_hook {
                    hook(&tmp2, &mut y2)?;
                } else {
                    self.children[1].apply(side, &tmp2, &mut y2)?;
                }
            }
            SchurFactorization::Upper => {
                self.children[1].apply(side, x2, &mut y2)?;
                let mut tmp1 = vec![S::zero(); span0.len()];
                schur.a12.try_spmv(&y2, &mut tmp1)?;
                for i in 0..tmp1.len() {
                    tmp1[i] = x1[i] - tmp1[i];
                }
                self.children[0].apply(side, &tmp1, &mut y1)?;
            }
            SchurFactorization::Full => {
                self.children[0].apply(side, x1, &mut y1)?;
                let mut tmp2 = vec![S::zero(); span1.len()];
                schur.a21.try_spmv(&y1, &mut tmp2)?;
                for i in 0..tmp2.len() {
                    tmp2[i] = x2[i] - tmp2[i];
                }
                if let Some(hook) = &self.schur_apply_hook {
                    hook(&tmp2, &mut y2)?;
                } else {
                    self.children[1].apply(side, &tmp2, &mut y2)?;
                }
                let mut tmp1 = vec![S::zero(); span0.len()];
                schur.a12.try_spmv(&y2, &mut tmp1)?;
                let mut corr = vec![S::zero(); span0.len()];
                self.children[0].apply(side, &tmp1, &mut corr)?;
                for i in 0..y1.len() {
                    y1[i] -= corr[i];
                }
            }
        }

        y[span0.start..span0.end].copy_from_slice(&y1);
        y[span1.start..span1.end].copy_from_slice(&y2);
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
        let csr = csr_from_linop(a, 0.0)?;
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
                let schur_mat =
                    Arc::new(self.schur_full_approx(a11.as_ref(), a22.as_ref(), &schur)?);
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
        match self.split_type {
            FieldSplitType::Additive => self.apply_additive(side, x, y),
            FieldSplitType::Multiplicative => self.apply_multiplicative(side, x, y),
            FieldSplitType::Symmetric => self.apply_symmetric(side, x, y),
            FieldSplitType::Schur { .. } => self.apply_schur(side, x, y),
        }
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
        let pc =
            FieldSplitPc::new(vec![2, 2], Some("jacobi".into()), PcOptions::default()).unwrap();
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
}
