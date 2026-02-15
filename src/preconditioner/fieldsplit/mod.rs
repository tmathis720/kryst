use crate::algebra::scalar::{KrystScalar, S};
use crate::config::options::PcOptions;
use crate::context::pc_context::{PcFactory, PcType};
use crate::core::traits::SubmatrixExtract;
use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::DistLayout;
use crate::matrix::op::LinOp;
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
}

#[derive(Debug, Clone, Copy)]
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
    Diag,
    A11,
}

#[derive(Debug, Clone)]
struct SchurBlocks {
    a12: Arc<CsrMatrix<S>>,
    a21: Arc<CsrMatrix<S>>,
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
        let prefixes = opts.pc_fieldsplit_prefixes.clone().unwrap_or_default();
        for (i, _) in block_sizes.iter().enumerate() {
            let mut child_opts = opts.clone();
            if let Some(prefix) = prefixes.get(i)
                && let Some(scoped) = opts.scoped_child(prefix)
            {
                child_opts.overlay_from(scoped.clone());
            }
            children.push(PcFactory::create_preconditioner(
                child_type,
                Some(&child_opts),
            )?);
        }
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
        })
    }

    fn split_type_from_options(opts: &PcOptions) -> Result<FieldSplitType, KError> {
        let kind = opts
            .pc_fieldsplit_type
            .as_deref()
            .unwrap_or("additive")
            .to_lowercase();
        match kind.as_str() {
            "additive" | "diag" | "blockdiag" => Ok(FieldSplitType::Additive),
            "multiplicative" | "mul" => Ok(FieldSplitType::Multiplicative),
            "symmetric" | "sym" => Ok(FieldSplitType::Symmetric),
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
                    "diag" => SchurPrecondition::Diag,
                    "a11" => SchurPrecondition::A11,
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
        spans
            .iter()
            .map(|span| {
                let indices: Vec<usize> = (span.start..span.end).collect();
                Arc::new(csr.extract_submatrix(&indices, &indices))
            })
            .collect()
    }

    fn extract_schur_blocks(&self, csr: &CsrMatrix<S>, spans: &[BlockSpan]) -> Option<SchurBlocks> {
        if spans.len() != 2 {
            return None;
        }
        let rows_0: Vec<usize> = (spans[0].start..spans[0].end).collect();
        let rows_1: Vec<usize> = (spans[1].start..spans[1].end).collect();
        let a12 = Arc::new(csr.extract_submatrix(&rows_0, &rows_1));
        let a21 = Arc::new(csr.extract_submatrix(&rows_1, &rows_0));
        Some(SchurBlocks { a12, a21 })
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
        for (span, child) in self.block_spans.iter().zip(self.children.iter()) {
            if span.len() == 0 {
                continue;
            }
            let mut zout = vec![S::zero(); span.len()];
            child.apply(side, &x[span.start..span.end], &mut zout)?;
            y[span.start..span.end].copy_from_slice(&zout);
        }
        Ok(())
    }

    fn apply_multiplicative(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let n = x.len();
        let mut y_accum = vec![S::zero(); n];
        let mut residual = x.to_vec();
        for (span, child) in self.block_spans.iter().zip(self.children.iter()) {
            if span.len() == 0 {
                continue;
            }
            let mut zout = vec![S::zero(); span.len()];
            child.apply(side, &residual[span.start..span.end], &mut zout)?;
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
        for (span, child) in self.block_spans.iter().zip(self.children.iter()) {
            if span.len() == 0 {
                continue;
            }
            let mut zout = vec![S::zero(); span.len()];
            child.apply(side, &residual[span.start..span.end], &mut zout)?;
            for (i, val) in zout.iter().enumerate() {
                y_accum[span.start + i] += *val;
            }
            self.update_residual(x, &y_accum, &mut residual)?;
        }
        for (span, child) in self.block_spans.iter().zip(self.children.iter()).rev() {
            if span.len() == 0 {
                continue;
            }
            let mut zout = vec![S::zero(); span.len()];
            child.apply(side, &residual[span.start..span.end], &mut zout)?;
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
                self.children[1].apply(side, &tmp2, &mut y2)?;
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
                self.children[1].apply(side, &tmp2, &mut y2)?;
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
        let n = a.dims().0;
        let spans =
            Self::block_spans_from_sizes_with_layout(&self.block_sizes, n, a.dist_layout())?;
        let csr = csr_from_linop(a, 0.0)?;
        let block_mats = self.extract_block_matrices(csr.as_ref(), &spans);
        let mut schur_precondition_matrix = None;
        if let FieldSplitType::Schur { precondition, .. } = self.split_type {
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
        if self
            .children
            .iter()
            .any(|pc| pc.distributed_support() == PcDistributedSupport::Distributed)
        {
            PcDistributedSupport::Distributed
        } else {
            PcDistributedSupport::LocalOnly
        }
    }
}
