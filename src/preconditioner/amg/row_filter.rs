use std::cmp::Ordering;

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::{sparse::CsrMatrix, spmv::csr_spmm_dense};
use faer::{MatMut, MatRef};

#[derive(Clone, Copy)]
pub struct RowFilter {
    pub tau_abs: R,               // absolute drop (>= 0)
    pub tau_rel: R,               // relative truncation in [0,1)
    pub k_max: usize,             // cap (0 => unlimited)
    pub must_keep: Option<usize>, // column index to force-keep
}

/// In-place filter of a single row according to absolute drop, relative truncation, and cap.
/// `cols` and `vals` are parallel arrays. Upon return, they are sorted by column index.
pub fn filter_row_by_truncation<T: KrystScalar<Real = f64>>(
    cols: &mut Vec<usize>,
    vals: &mut Vec<T>,
    rf: RowFilter,
) {
    debug_assert_eq!(cols.len(), vals.len());

    // 1) Absolute drop
    if rf.tau_abs > 0.0 {
        let mut w = 0usize;
        for i in 0..cols.len() {
            let keep = vals[i].abs() >= rf.tau_abs || rf.must_keep.is_some_and(|c| c == cols[i]);
            if keep {
                cols[w] = cols[i];
                vals[w] = vals[i];
                w += 1;
            }
        }
        cols.truncate(w);
        vals.truncate(w);
    }

    // 2) Relative truncation
    if rf.tau_rel > 0.0 && !vals.is_empty() {
        // sort indices by ascending |v| with column tiebreaker for determinism
        let mut idx: Vec<usize> = (0..vals.len()).collect();
        idx.sort_unstable_by(|&i, &j| match vals[i].abs().total_cmp(&vals[j].abs()) {
            Ordering::Equal => cols[i].cmp(&cols[j]),
            o => o,
        });
        let total: f64 = vals.iter().map(|v| v.abs()).sum();
        let mut dropped_sum = 0.0f64;
        let mut drop_mask = vec![false; vals.len()];

        for &i in &idx {
            if rf.must_keep.is_some_and(|c| cols[i] == c) {
                continue;
            }
            let allow = rf.tau_rel * total;
            if dropped_sum + vals[i].abs() <= allow {
                drop_mask[i] = true;
                dropped_sum += vals[i].abs();
            } else {
                break;
            }
        }
        let mut w = 0usize;
        for i in 0..cols.len() {
            if !drop_mask[i] {
                cols[w] = cols[i];
                vals[w] = vals[i];
                w += 1;
            }
        }
        cols.truncate(w);
        vals.truncate(w);
    }

    // 3) Cap
    if rf.k_max > 0 && vals.len() > rf.k_max {
        let mut order: Vec<usize> = (0..vals.len()).collect();
        order.sort_unstable_by(|&i, &j| match vals[j].abs().total_cmp(&vals[i].abs()) {
            Ordering::Equal => cols[i].cmp(&cols[j]),
            o => o,
        });
        let mut keep = vec![false; vals.len()];
        for &idx in order.iter().take(rf.k_max) {
            keep[idx] = true;
        }
        if let Some(mk) = rf.must_keep
            && let Some(pos) = cols.iter().position(|&c| c == mk)
            && !keep[pos]
        {
            let mut replace: Option<usize> = None;
            for &idx in order.iter().take(rf.k_max) {
                if replace.is_none_or(|r| {
                    let cmp_mag = vals[idx].abs().total_cmp(&vals[r].abs());
                    cmp_mag == Ordering::Less || (cmp_mag == Ordering::Equal && cols[idx] > cols[r])
                }) {
                    replace = Some(idx);
                }
            }
            if let Some(ridx) = replace {
                keep[ridx] = false;
            }
            keep[pos] = true;
        }
        let mut w = 0usize;
        for i in 0..cols.len() {
            if keep[i] {
                cols[w] = cols[i];
                vals[w] = vals[i];
                w += 1;
            }
        }
        cols.truncate(w);
        vals.truncate(w);
    }

    // final sort by column index
    let mut pairs: Vec<(usize, T)> = cols.iter().cloned().zip(vals.iter().cloned()).collect();
    pairs.sort_unstable_by_key(|(c, _)| *c);
    for (i, (c, v)) in pairs.into_iter().enumerate() {
        cols[i] = c;
        vals[i] = v;
    }
}

/// Apply row-wise filtering to an existing CSR matrix values slice, zeroing dropped entries.
pub fn apply_filter_to_csr_values_in_place<T: KrystScalar<Real = f64>>(
    nrows: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &mut [T],
    mut rf_for_row: impl FnMut(usize) -> RowFilter,
) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let row_filters: Vec<RowFilter> = (0..nrows).map(&mut rf_for_row).collect();
        let filtered_rows: Vec<(usize, Vec<T>)> = (0..nrows)
            .into_par_iter()
            .map(|i| {
                let rs = row_ptr[i];
                let re = row_ptr[i + 1];
                if rs == re {
                    return (i, Vec::new());
                }
                let mut cols: Vec<usize> = col_idx[rs..re].to_vec();
                let mut vs: Vec<T> = vals[rs..re].to_vec();
                filter_row_by_truncation(&mut cols, &mut vs, row_filters[i]);
                let mut out = vec![T::zero(); re - rs];
                let mut keep_pos = 0usize;
                for (local, &c) in col_idx[rs..re].iter().enumerate() {
                    while keep_pos < cols.len() && cols[keep_pos] < c {
                        keep_pos += 1;
                    }
                    if keep_pos < cols.len() && cols[keep_pos] == c {
                        out[local] = vs[keep_pos];
                    }
                }
                (i, out)
            })
            .collect();
        for (i, out) in filtered_rows {
            let rs = row_ptr[i];
            let re = row_ptr[i + 1];
            if rs != re {
                vals[rs..re].copy_from_slice(&out);
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for i in 0..nrows {
            let rs = row_ptr[i];
            let re = row_ptr[i + 1];
            if rs == re {
                continue;
            }
            let mut cols: Vec<usize> = col_idx[rs..re].to_vec();
            let mut vs: Vec<T> = vals[rs..re].to_vec();
            let rf = rf_for_row(i);
            filter_row_by_truncation(&mut cols, &mut vs, rf);
            let mut keep_pos = 0usize;
            for p in rs..re {
                let c = col_idx[p];
                while keep_pos < cols.len() && cols[keep_pos] < c {
                    keep_pos += 1;
                }
                if keep_pos < cols.len() && cols[keep_pos] == c {
                    vals[p] = vs[keep_pos];
                } else {
                    vals[p] = T::zero();
                }
            }
        }
    }
}

pub fn restrict_trials<T: KrystScalar>(
    r: &CsrMatrix<T>,
    t_fine: MatRef<'_, T>,
    t_coarse: MatMut<'_, T>,
) -> Result<(), KError> {
    csr_spmm_dense(r, t_fine, t_coarse)
}

pub fn compensate_scalar_rows<T>(
    a: &mut CsrMatrix<T>,
    trials: MatRef<'_, T>,
    omega: f64,
    min_diag: Option<f64>,
) -> Result<(), KError>
where
    T: KrystScalar<Real = f64>,
{
    let n = a.nrows();
    if trials.nrows() != n {
        return Err(KError::InvalidInput(
            "trial matrix row count mismatch during scalar compensation".into(),
        ));
    }
    if trials.ncols() == 0 {
        return Ok(());
    }
    let eps = 1e-30;
    for i in 0..n {
        let (cols, vals) = a.row(i);
        let mut num = T::zero();
        let mut den = 0.0f64;
        for alpha in 0..trials.ncols() {
            let mut at = T::zero();
            for (&j, &v) in cols.iter().zip(vals.iter()) {
                at = at + v * trials[(j, alpha)];
            }
            let t = trials[(i, alpha)];
            num = num + t.conj() * at;
            den += t.abs2();
        }
        if den <= eps {
            continue;
        }
        let corr = num * T::from_real(omega / den);
        if let Some(diag) = a.diag_mut(i) {
            let new_val = *diag - corr;
            if let Some(min_allowed) = min_diag
                && new_val.real() <= min_allowed
            {
                continue;
            }
            *diag = new_val;
        } else {
            return Err(KError::InvalidInput(format!(
                "trial compensation requires structural diagonal at row {i}"
            )));
        }
    }
    Ok(())
}

pub fn compensate_nodal_diag<T: KrystScalar<Real = f64>>(
    a: &mut CsrMatrix<T>,
    trials: MatRef<'_, T>,
    block_size: usize,
    omega: f64,
    min_diag: Option<f64>,
) -> Result<(), KError> {
    if block_size <= 1 {
        return compensate_scalar_rows(a, trials, omega, min_diag);
    }
    let n = a.nrows();
    if trials.nrows() != n {
        return Err(KError::InvalidInput(
            "trial matrix row count mismatch during nodal compensation".into(),
        ));
    }
    if trials.ncols() == 0 {
        return Ok(());
    }
    if n % block_size != 0 {
        return Err(KError::InvalidInput(
            "nodal compensation requires matrix rows divisible by block size".into(),
        ));
    }
    let nodes = n / block_size;
    let eps = 1e-12;

    for node in 0..nodes {
        let row_start = node * block_size;
        let mut active = false;
        for q in 0..block_size {
            for alpha in 0..trials.ncols() {
                if trials[(row_start + q, alpha)].abs() > 0.0 {
                    active = true;
                    break;
                }
            }
            if active {
                break;
            }
        }
        if !active {
            continue;
        }
        let mut gram = vec![T::zero(); block_size * block_size];
        let mut rhs_diag = vec![T::zero(); block_size];
        for p in 0..block_size {
            for q in 0..block_size {
                let mut g = T::zero();
                for alpha in 0..trials.ncols() {
                    let tp = trials[(row_start + p, alpha)];
                    let tq = trials[(row_start + q, alpha)];
                    g = g + tp.conj() * tq;
                }
                if p == q {
                    g = g + T::from_real(eps);
                }
                gram[p * block_size + q] = g;
            }
        }
        for q in 0..block_size {
            let row = row_start + q;
            let (cols, vals) = a.row(row);
            for alpha in 0..trials.ncols() {
                let mut at = T::zero();
                for (&j, &v) in cols.iter().zip(vals.iter()) {
                    at = at + v * trials[(j, alpha)];
                }
                rhs_diag[q] = rhs_diag[q] + trials[(row, alpha)].conj() * at;
            }
        }
        let inv = invert_small_matrix_generic(&gram, block_size)?;
        let mut diag_updates = vec![T::zero(); block_size];
        for q in 0..block_size {
            let mut sum = T::zero();
            for k in 0..block_size {
                sum = sum + inv[q * block_size + k] * rhs_diag[k];
            }
            diag_updates[q] = T::from_real(omega) * sum;
        }
        for q in 0..block_size {
            let row = row_start + q;
            if let Some(diag) = a.diag_mut(row) {
                let new_val = *diag - diag_updates[q];
                if let Some(min_allowed) = min_diag
                    && new_val.real() <= min_allowed
                {
                    continue;
                }
                *diag = new_val;
            } else {
                return Err(KError::InvalidInput(format!(
                    "trial compensation requires structural diagonal at row {row}"
                )));
            }
        }
    }
    Ok(())
}

fn invert_small_matrix(mat: &faer::Mat<f64>) -> Result<Vec<f64>, KError> {
    let n = mat.nrows();
    debug_assert_eq!(mat.ncols(), n);
    let width = 2 * n;
    let mut aug = vec![0.0f64; n * width];
    for i in 0..n {
        for j in 0..n {
            aug[i * width + j] = mat[(i, j)];
        }
        for j in 0..n {
            aug[i * width + n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for col in 0..n {
        let mut pivot = col;
        let mut max_val = aug[col * width + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * width + col].abs();
            if val > max_val {
                max_val = val;
                pivot = row;
            }
        }
        if max_val <= 1e-18 {
            return Err(KError::InvalidInput(
                "trial compensation encountered singular Gram matrix".into(),
            ));
        }
        if pivot != col {
            for j in 0..width {
                aug.swap(col * width + j, pivot * width + j);
            }
        }
        let piv = aug[col * width + col];
        for j in 0..width {
            aug[col * width + j] /= piv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * width + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..width {
                aug[row * width + j] -= factor * aug[col * width + j];
            }
        }
    }
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * width + n + j];
        }
    }
    Ok(inv)
}

fn invert_small_matrix_generic<T: KrystScalar<Real = f64>>(
    mat: &[T],
    n: usize,
) -> Result<Vec<T>, KError> {
    if mat.len() != n * n {
        return Err(KError::InvalidInput(
            "trial compensation Gram matrix dimension mismatch".into(),
        ));
    }
    let width = 2 * n;
    let mut aug = vec![T::zero(); n * width];
    for i in 0..n {
        for j in 0..n {
            aug[i * width + j] = mat[i * n + j];
        }
        for j in 0..n {
            aug[i * width + n + j] = if i == j { T::one() } else { T::zero() };
        }
    }
    for col in 0..n {
        let mut pivot = col;
        let mut max_val = aug[col * width + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * width + col].abs();
            if val > max_val {
                max_val = val;
                pivot = row;
            }
        }
        if max_val <= 1e-18 {
            return Err(KError::InvalidInput(
                "trial compensation encountered singular Gram matrix".into(),
            ));
        }
        if pivot != col {
            for j in 0..width {
                aug.swap(col * width + j, pivot * width + j);
            }
        }
        let piv = aug[col * width + col];
        for j in 0..width {
            aug[col * width + j] = aug[col * width + j] / piv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * width + col];
            if factor == T::zero() {
                continue;
            }
            for j in 0..width {
                aug[row * width + j] = aug[row * width + j] - factor * aug[col * width + j];
            }
        }
    }
    let mut inv = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * width + n + j];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_drop_and_must_keep() {
        let mut cols = vec![0, 1];
        let mut vals = vec![1e-3, 2.0];
        let rf = RowFilter {
            tau_abs: 0.1,
            tau_rel: 0.0,
            k_max: 0,
            must_keep: Some(0),
        };
        filter_row_by_truncation(&mut cols, &mut vals, rf);
        assert_eq!(cols, vec![0, 1]);
    }

    #[test]
    fn relative_truncation() {
        let mut cols = vec![0, 1, 2];
        let mut vals = vec![0.2, 0.3, 0.5];
        let rf = RowFilter {
            tau_abs: 0.0,
            tau_rel: 0.25,
            k_max: 0,
            must_keep: None,
        };
        filter_row_by_truncation(&mut cols, &mut vals, rf);
        assert_eq!(cols, vec![1, 2]);
    }

    #[test]
    fn cap_with_must_keep() {
        let mut cols = vec![0, 1, 2];
        let mut vals = vec![1.0, 0.9, 0.8];
        let rf = RowFilter {
            tau_abs: 0.0,
            tau_rel: 0.0,
            k_max: 2,
            must_keep: Some(2),
        };
        filter_row_by_truncation(&mut cols, &mut vals, rf);
        assert_eq!(cols, vec![0, 2]);
    }

    #[test]
    fn apply_filter_zeroes_dropped() {
        let nrows = 2;
        let row_ptr = vec![0, 3, 5];
        let col_idx = vec![0, 1, 2, 0, 1];
        let mut vals = vec![10.0, 1e-3, 0.2, 3.0, 4.0];
        apply_filter_to_csr_values_in_place(nrows, &row_ptr, &col_idx, &mut vals, |row| {
            RowFilter {
                tau_abs: 0.5,
                tau_rel: 0.0,
                k_max: 0,
                must_keep: Some(row),
            }
        });
        assert_eq!(vals, vec![10.0, 0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn nodal_trial_compensation_updates_real_block_diagonal() {
        let mut a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0, 1.0, 1.0, 3.0],
        );
        let mut trials = faer::Mat::<f64>::zeros(2, 2);
        trials[(0, 0)] = 1.0;
        trials[(1, 1)] = 1.0;

        compensate_nodal_diag(&mut a, trials.as_ref(), 2, 0.5, None).unwrap();

        assert!((a.values()[0] - 2.0).abs() < 1e-10);
        assert_eq!(a.values()[1], 1.0);
        assert_eq!(a.values()[2], 1.0);
        assert!((a.values()[3] - 1.5).abs() < 1e-10);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn complex_filter_uses_magnitude_and_preserves_values() {
        let mut cols = vec![0, 1, 2];
        let mut vals = vec![
            crate::S::from_parts(0.01, 0.02),
            crate::S::from_parts(2.0, -1.0),
            crate::S::from_parts(0.2, 0.0),
        ];
        let rf = RowFilter {
            tau_abs: 0.1,
            tau_rel: 0.0,
            k_max: 0,
            must_keep: Some(0),
        };
        filter_row_by_truncation(&mut cols, &mut vals, rf);

        assert_eq!(cols, vec![0, 1, 2]);
        assert_eq!(vals[0], crate::S::from_parts(0.01, 0.02));
        assert_eq!(vals[1], crate::S::from_parts(2.0, -1.0));
        assert_eq!(vals[2], crate::S::from_parts(0.2, 0.0));

        let row_ptr = vec![0, 3];
        let col_idx = vec![0, 1, 2];
        apply_filter_to_csr_values_in_place(1, &row_ptr, &col_idx, &mut vals, |_| RowFilter {
            tau_abs: 0.25,
            tau_rel: 0.0,
            k_max: 0,
            must_keep: Some(0),
        });
        assert_eq!(vals[0], crate::S::from_parts(0.01, 0.02));
        assert_eq!(vals[1], crate::S::from_parts(2.0, -1.0));
        assert_eq!(vals[2], crate::S::zero());
    }

    #[cfg(feature = "complex")]
    #[test]
    fn nodal_trial_compensation_preserves_complex_block_diagonal_updates() {
        let mut a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![
                crate::S::from_parts(4.0, 1.0),
                crate::S::from_parts(1.0, 1.0),
                crate::S::from_parts(1.0, -1.0),
                crate::S::from_parts(3.0, -2.0),
            ],
        );
        let mut trials = faer::Mat::<crate::S>::zeros(2, 2);
        trials[(0, 0)] = crate::S::one();
        trials[(1, 1)] = crate::S::one();

        compensate_nodal_diag(&mut a, trials.as_ref(), 2, 0.5, None).unwrap();

        assert!((a.values()[0] - crate::S::from_parts(2.0, 0.5)).abs() < 1e-10);
        assert_eq!(a.values()[1], crate::S::from_parts(1.0, 1.0));
        assert_eq!(a.values()[2], crate::S::from_parts(1.0, -1.0));
        assert!((a.values()[3] - crate::S::from_parts(1.5, -1.0)).abs() < 1e-10);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn scalar_trial_compensation_uses_hermitian_products_for_complex_values() {
        let mut a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![
                crate::S::from_parts(4.0, 0.0),
                crate::S::from_parts(1.0, 1.0),
                crate::S::from_parts(1.0, -1.0),
                crate::S::from_parts(3.0, 0.0),
            ],
        );
        let mut trials = faer::Mat::<crate::S>::zeros(2, 1);
        trials[(0, 0)] = crate::S::from_parts(1.0, 1.0);
        trials[(1, 0)] = crate::S::from_parts(2.0, -1.0);

        compensate_scalar_rows(&mut a, trials.as_ref(), 0.5, None).unwrap();

        assert!((a.values()[0] - crate::S::from_parts(1.0, 0.5)).abs() < 1e-12);
        assert_eq!(a.values()[1], crate::S::from_parts(1.0, 1.0));
        assert_eq!(a.values()[2], crate::S::from_parts(1.0, -1.0));
        assert!((a.values()[3] - crate::S::from_parts(1.1, -0.2)).abs() < 1e-12);
    }
}
