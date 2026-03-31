use crate::matrix::sparse::CsrMatrix;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct NormStats {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

#[derive(Debug, Clone)]
pub struct MatrixDiagnostics {
    pub diagonal_nonzeros: usize,
    pub diagonal_zeros: usize,
    pub row_norm_1: NormStats,
}

#[derive(Debug, Clone)]
pub struct SystemTransform {
    pub row_perm: Vec<usize>,
    pub row_pinv: Vec<usize>,
    pub col_perm: Vec<usize>,
    pub col_pinv: Vec<usize>,
    pub row_scale: Option<Vec<f64>>,
    pub col_scale: Option<Vec<f64>>,
}

impl SystemTransform {
    pub fn identity(nrows: usize, ncols: usize) -> Self {
        Self {
            row_perm: (0..nrows).collect(),
            row_pinv: (0..nrows).collect(),
            col_perm: (0..ncols).collect(),
            col_pinv: (0..ncols).collect(),
            row_scale: None,
            col_scale: None,
        }
    }

    pub fn apply_rhs(&self, rhs: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; rhs.len()];
        for (new_i, &old_i) in self.row_perm.iter().enumerate() {
            out[new_i] = rhs[old_i];
        }
        if let Some(scale) = &self.row_scale {
            for (v, s) in out.iter_mut().zip(scale.iter()) {
                *v *= *s;
            }
        }
        out
    }

    pub fn recover_solution(&self, z: &[f64]) -> Vec<f64> {
        let mut x_col_perm = z.to_vec();
        if let Some(scale) = &self.col_scale {
            for (v, s) in x_col_perm.iter_mut().zip(scale.iter()) {
                *v *= *s;
            }
        }
        let mut x = vec![0.0; x_col_perm.len()];
        for old_j in 0..x.len() {
            x[old_j] = x_col_perm[self.col_pinv[old_j]];
        }
        x
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

pub fn matrix_diagnostics(a: &CsrMatrix<f64>) -> MatrixDiagnostics {
    let n = a.nrows().min(a.ncols());
    let mut diag_nonzeros = 0usize;
    for i in 0..n {
        if let Some(v) = a.diag_ref(i)
            && v.abs() > 0.0
        {
            diag_nonzeros += 1;
        }
    }

    let mut row_norms = vec![0.0; a.nrows()];
    for (i, row_norm) in row_norms.iter_mut().enumerate() {
        let (_, vals) = a.row(i);
        *row_norm = vals.iter().map(|v| v.abs()).sum::<f64>();
    }
    let min = row_norms.iter().copied().fold(f64::INFINITY, f64::min);
    let max = row_norms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let med = median(&mut row_norms);

    MatrixDiagnostics {
        diagonal_nonzeros: diag_nonzeros,
        diagonal_zeros: n - diag_nonzeros,
        row_norm_1: NormStats {
            min,
            median: med,
            max,
        },
    }
}

pub fn row_1_norm_scaling(a: &CsrMatrix<f64>, eps: f64) -> Vec<f64> {
    (0..a.nrows())
        .map(|i| {
            let (_, vals) = a.row(i);
            let norm = vals.iter().map(|v| v.abs()).sum::<f64>();
            if norm > eps { 1.0 / norm } else { 1.0 }
        })
        .collect()
}

pub fn col_1_norm_scaling(a: &CsrMatrix<f64>, eps: f64) -> Vec<f64> {
    let mut norms = vec![0.0; a.ncols()];
    for i in 0..a.nrows() {
        let (cols, vals) = a.row(i);
        for (&j, &v) in cols.iter().zip(vals.iter()) {
            norms[j] += v.abs();
        }
    }
    norms
        .into_iter()
        .map(|norm| if norm > eps { 1.0 / norm } else { 1.0 })
        .collect()
}

pub fn permute_rows_cols(
    a: &CsrMatrix<f64>,
    row_perm: &[usize],
    col_perm: &[usize],
) -> CsrMatrix<f64> {
    let nrows = a.nrows();
    let ncols = a.ncols();
    let mut col_pinv = vec![0usize; ncols];
    for (new_j, &old_j) in col_perm.iter().enumerate() {
        col_pinv[old_j] = new_j;
    }

    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_idx = Vec::with_capacity(a.nnz());
    let mut values = Vec::with_capacity(a.nnz());
    row_ptr.push(0);

    for &old_i in row_perm {
        let (cols, vals) = a.row(old_i);
        let mut entries: Vec<(usize, f64)> = cols
            .iter()
            .zip(vals.iter())
            .map(|(&old_j, &v)| (col_pinv[old_j], v))
            .collect();
        entries.sort_unstable_by_key(|e| e.0);
        for (j, v) in entries {
            col_idx.push(j);
            values.push(v);
        }
        row_ptr.push(col_idx.len());
    }

    CsrMatrix::from_csr(nrows, ncols, row_ptr, col_idx, values)
}

pub fn scale_rows(a: &CsrMatrix<f64>, scale: &[f64]) -> CsrMatrix<f64> {
    let mut out = a.clone();
    for (i, &s) in scale.iter().enumerate().take(out.nrows()) {
        for v in out.row_values_mut(i).iter_mut() {
            *v *= s;
        }
    }
    out
}

pub fn scale_cols(a: &CsrMatrix<f64>, scale: &[f64]) -> CsrMatrix<f64> {
    let mut out = a.clone();
    let col_idx = out.col_idx().to_vec();
    for (k, v) in out.values_mut().iter_mut().enumerate() {
        *v *= scale[col_idx[k]];
    }
    out
}

pub fn nonsymmetric_max_transversal_permutation(a: &CsrMatrix<f64>) -> (Vec<usize>, Vec<usize>) {
    let nrows = a.nrows();
    let ncols = a.ncols();
    let mut match_col_for_row: Vec<Option<usize>> = vec![None; nrows];
    let mut match_row_for_col: Vec<Option<usize>> = vec![None; ncols];

    for start_row in 0..nrows {
        let mut parent_col: Vec<Option<usize>> = vec![None; ncols];
        let mut seen_row = vec![false; nrows];
        let mut seen_col = vec![false; ncols];
        let mut q = VecDeque::new();
        q.push_back(start_row);
        seen_row[start_row] = true;

        let mut found_free_col = None;
        while let Some(r) = q.pop_front() {
            let (cols, _) = a.row(r);
            for &c in cols {
                if seen_col[c] {
                    continue;
                }
                seen_col[c] = true;
                parent_col[c] = Some(r);
                if match_row_for_col[c].is_none() {
                    found_free_col = Some(c);
                    break;
                }
                let next_r = match_row_for_col[c].expect("checked");
                if !seen_row[next_r] {
                    seen_row[next_r] = true;
                    q.push_back(next_r);
                }
            }
            if found_free_col.is_some() {
                break;
            }
        }

        if let Some(mut c) = found_free_col {
            while let Some(r) = parent_col[c] {
                let prev_c = match_col_for_row[r];
                match_col_for_row[r] = Some(c);
                match_row_for_col[c] = Some(r);
                if let Some(pc) = prev_c {
                    c = pc;
                } else {
                    break;
                }
            }
        }
    }

    let mut used_rows = vec![false; nrows];
    let mut used_cols = vec![false; ncols];
    let mut row_perm = Vec::with_capacity(nrows);
    let mut col_perm = Vec::with_capacity(ncols);

    for r in 0..nrows {
        if let Some(c) = match_col_for_row[r] {
            row_perm.push(r);
            col_perm.push(c);
            used_rows[r] = true;
            used_cols[c] = true;
        }
    }
    for (r, used) in used_rows.iter().enumerate().take(nrows) {
        if !*used {
            row_perm.push(r);
        }
    }
    for (c, used) in used_cols.iter().enumerate().take(ncols) {
        if !*used {
            col_perm.push(c);
        }
    }

    (row_perm, col_perm)
}

pub fn build_inverse(perm: &[usize]) -> Vec<usize> {
    let mut pinv = vec![0usize; perm.len()];
    for (new_i, &old_i) in perm.iter().enumerate() {
        pinv[old_i] = new_i;
    }
    pinv
}
