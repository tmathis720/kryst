use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::ilu_csr::{ReorderingKind, ReorderingOptions};
use crate::utils::conditioning::{ConditioningOptions, ScaleDirection, ScaleNorm};
use crate::utils::permutation::{
    Permutation, amd_csr, permute_csr_nonsymmetric, permute_csr_symmetric, rcm_csr,
};

#[derive(Clone, Debug)]
pub struct PreconditioningMetadata {
    pub left_perm: Permutation,
    pub right_perm: Permutation,
    pub row_scaling: Option<Vec<f64>>,
    pub col_scaling: Option<Vec<f64>>,
    pub matched_pairs: Option<Vec<(usize, usize)>>,
}

impl PreconditioningMetadata {
    pub fn identity(n: usize) -> Self {
        Self {
            left_perm: Permutation::identity(n),
            right_perm: Permutation::identity(n),
            row_scaling: None,
            col_scaling: None,
            matched_pairs: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreconditioningPipelineResult {
    pub matrix: CsrMatrix<f64>,
    pub metadata: PreconditioningMetadata,
}

pub fn apply_preconditioning_pipeline(
    a: &CsrMatrix<f64>,
    conditioning: &ConditioningOptions,
    reordering: &ReorderingOptions,
) -> Result<PreconditioningPipelineResult, KError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(KError::InvalidInput(
            "preconditioning pipeline requires square matrix".into(),
        ));
    }

    let mut work = a.clone();
    let mut meta = PreconditioningMetadata::identity(n);

    // 1) optional row scaling
    if matches!(
        conditioning.scale,
        Some(ScaleDirection::Row | ScaleDirection::Both)
    ) {
        let row = row_norms_csr(&work, conditioning.scale_norm);
        apply_row_scale(&mut work, &row);
        meta.row_scaling = Some(row);
    }

    // 2) optional column scaling
    if matches!(
        conditioning.scale,
        Some(ScaleDirection::Col | ScaleDirection::Both)
    ) {
        let col = col_norms_csr(&work, conditioning.scale_norm);
        apply_col_scale(&mut work, &col);
        meta.col_scaling = Some(col);
    }

    // 3) nonsymmetric matching/transversal permutation
    if !reordering.symmetric {
        let (row_p, col_p, pairs) = greedy_transversal_permutations(&work);
        work = permute_csr_nonsymmetric(&work, &row_p, &col_p);
        meta.left_perm = row_p;
        meta.right_perm = col_p;
        meta.matched_pairs = Some(pairs);
    }

    // 4) optional reorder
    let reorder = match reordering.kind {
        ReorderingKind::None => None,
        ReorderingKind::Rcm => Some(rcm_csr(&work)),
        ReorderingKind::Amd => Some(amd_csr(&work)),
    };
    if let Some(p) = reorder {
        if reordering.symmetric {
            work = permute_csr_symmetric(&work, &p);
            meta.left_perm = compose_perm(&p, &meta.left_perm);
            meta.right_perm = compose_perm(&p, &meta.right_perm);
        } else {
            work = permute_csr_nonsymmetric(&work, &p, &p);
            meta.left_perm = compose_perm(&p, &meta.left_perm);
            meta.right_perm = compose_perm(&p, &meta.right_perm);
        }
    }

    Ok(PreconditioningPipelineResult {
        matrix: work,
        metadata: meta,
    })
}

fn compose_perm(new_then_old: &Permutation, old: &Permutation) -> Permutation {
    let n = old.len();
    let p: Vec<usize> = (0..n).map(|i| old.p[new_then_old.p[i]]).collect();
    let mut pinv = vec![0usize; n];
    for (new_i, &old_i) in p.iter().enumerate() {
        pinv[old_i] = new_i;
    }
    Permutation { p, pinv }
}

fn greedy_transversal_permutations(
    a: &CsrMatrix<f64>,
) -> (Permutation, Permutation, Vec<(usize, usize)>) {
    let n = a.nrows();
    let mut used_cols = vec![false; n];
    let mut row_to_col = vec![None; n];
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    for i in 0..n {
        let mut best: Option<(usize, f64)> = None;
        for p in rp[i]..rp[i + 1] {
            let j = cj[p];
            if used_cols[j] {
                continue;
            }
            let score = vv[p].abs();
            if best.is_none_or(|(_, b)| score > b) {
                best = Some((j, score));
            }
        }
        if let Some((j, _)) = best {
            row_to_col[i] = Some(j);
            used_cols[j] = true;
        }
    }

    let mut free_cols: Vec<usize> = (0..n).filter(|&j| !used_cols[j]).collect();
    let mut pairs = Vec::with_capacity(n);
    let mut col_to_row = vec![usize::MAX; n];
    for i in 0..n {
        let j = row_to_col[i].unwrap_or_else(|| free_cols.pop().unwrap_or(i));
        col_to_row[j] = i;
        pairs.push((i, j));
    }

    let row_p: Vec<usize> = (0..n).collect();
    let mut row_pinv = vec![0usize; n];
    for i in 0..n {
        row_pinv[i] = i;
    }
    let mut col_p = vec![0usize; n];
    for (new_i, &(_, old_j)) in pairs.iter().enumerate() {
        col_p[new_i] = old_j;
    }
    let mut col_pinv = vec![0usize; n];
    for (new_i, &old_j) in col_p.iter().enumerate() {
        col_pinv[old_j] = new_i;
    }

    (
        Permutation {
            p: row_p,
            pinv: row_pinv,
        },
        Permutation {
            p: col_p,
            pinv: col_pinv,
        },
        pairs,
    )
}

fn row_norms_csr(a: &CsrMatrix<f64>, norm: ScaleNorm) -> Vec<f64> {
    let nrows = a.nrows();
    let mut out = vec![0.0; nrows];
    let rp = a.row_ptr();
    let values = a.values();
    for i in 0..nrows {
        let start = rp[i];
        let end = rp[i + 1];
        out[i] = match norm {
            ScaleNorm::One => values[start..end].iter().map(|v| v.abs()).sum(),
            ScaleNorm::Inf => values[start..end]
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f64::max),
        };
        if out[i] == 0.0 {
            out[i] = 1.0;
        }
    }
    out
}

fn col_norms_csr(a: &CsrMatrix<f64>, norm: ScaleNorm) -> Vec<f64> {
    let ncols = a.ncols();
    let mut one = vec![0.0; ncols];
    let mut inf = vec![0.0_f64; ncols];
    for (&j, &v) in a.col_idx().iter().zip(a.values().iter()) {
        let av = v.abs();
        one[j] += av;
        inf[j] = inf[j].max(av);
    }
    let mut out = match norm {
        ScaleNorm::One => one,
        ScaleNorm::Inf => inf,
    };
    for val in &mut out {
        if *val == 0.0 {
            *val = 1.0;
        }
    }
    out
}

fn apply_row_scale(a: &mut CsrMatrix<f64>, row: &[f64]) {
    let rp = a.row_ptr().to_vec();
    let nrows = a.nrows();
    let values = a.values_mut();
    for i in 0..nrows {
        let s = row[i];
        for p in rp[i]..rp[i + 1] {
            values[p] /= s;
        }
    }
}

fn apply_col_scale(a: &mut CsrMatrix<f64>, col: &[f64]) {
    let cols = a.col_idx().to_vec();
    let values = a.values_mut();
    for p in 0..values.len() {
        values[p] /= col[cols[p]];
    }
}
