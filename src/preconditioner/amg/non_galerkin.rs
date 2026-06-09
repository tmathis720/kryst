#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;

use super::{NgSymmetry, rap_ops::CsrPattern};

#[derive(Clone, Copy)]
pub(crate) struct NgRowFilter {
    pub tau_abs: f64,
    pub tau_rel: f64,
    pub k_max: usize,
    pub lump_diag: bool,
}

fn non_galerkin_row_keep<T: KrystScalar<Real = f64>>(
    i: usize,
    cols: &[usize],
    vals: &[T],
    rf: NgRowFilter,
) -> Vec<bool> {
    let mut keep = vec![false; cols.len()];
    if cols.is_empty() {
        return keep;
    }

    let mut idx: Vec<usize> = (0..cols.len()).collect();
    if rf.tau_abs > 0.0 || rf.tau_rel > 0.0 || (rf.k_max > 0 && cols.len() > rf.k_max) {
        idx.sort_unstable_by(|&u, &v| {
            let au = vals[u].abs();
            let av = vals[v].abs();
            au.total_cmp(&av).then_with(|| cols[u].cmp(&cols[v]))
        });
        let mut drop_mask = vec![false; cols.len()];
        let mut dropped_sum = 0.0f64;
        let l1 = cols
            .iter()
            .zip(vals)
            .filter_map(|(&col, val)| (col != i).then_some(val.abs()))
            .sum::<f64>();
        for &t in &idx {
            let col = cols[t];
            let v = vals[t];
            if col == i {
                continue;
            }
            let by_abs = v.abs() < rf.tau_abs;
            let allow = rf.tau_rel * l1;
            let by_rel = dropped_sum + v.abs() <= allow + 1e-300;
            if by_abs || (rf.tau_rel > 0.0 && by_rel) {
                drop_mask[t] = true;
                dropped_sum += v.abs();
            }
        }
        if rf.k_max > 0 {
            let mut order_keep: Vec<usize> = (0..cols.len()).collect();
            order_keep.sort_unstable_by(|&u, &v| {
                let au = vals[u].abs();
                let av = vals[v].abs();
                av.total_cmp(&au).then_with(|| cols[u].cmp(&cols[v]))
            });
            let mut kept_off = 0usize;
            for &t in &order_keep {
                if cols[t] == i {
                    keep[t] = true;
                    continue;
                }
                if kept_off < rf.k_max && !drop_mask[t] {
                    keep[t] = true;
                    kept_off += 1;
                }
            }
            for t in 0..cols.len() {
                if cols[t] == i {
                    keep[t] = true;
                } else if !drop_mask[t] && !keep[t] && kept_off < rf.k_max {
                    keep[t] = true;
                    kept_off += 1;
                }
            }
        } else {
            for t in 0..cols.len() {
                if cols[t] == i || !drop_mask[t] {
                    keep[t] = true;
                }
            }
        }
    } else {
        keep.fill(true);
    }

    let kept_off = (0..cols.len()).filter(|&t| keep[t] && cols[t] != i).count();
    let had_off = cols.iter().any(|&col| col != i);
    if kept_off == 0 && had_off {
        let mut best = None;
        let mut best_mag = 0.0;
        for t in 0..cols.len() {
            if cols[t] != i {
                let mag = vals[t].abs();
                if best.is_none() || mag > best_mag {
                    best = Some(t);
                    best_mag = mag;
                }
            }
        }
        keep[best.expect("had_off guarantees an off-diagonal entry")] = true;
    }
    if let Some(diag) = cols.iter().position(|&col| col == i) {
        keep[diag] = true;
    }
    keep
}

struct NgBuiltRow<T> {
    cols: Vec<usize>,
    vals: Vec<T>,
    full2local: Vec<Option<usize>>,
}

fn build_non_galerkin_row<T: KrystScalar<Real = f64>>(
    i: usize,
    cols: &[usize],
    vals: &[T],
    keep: &[bool],
    lump_diag: bool,
) -> NgBuiltRow<T> {
    let mut ng_cols = Vec::with_capacity(cols.len());
    let mut ng_vals = Vec::with_capacity(vals.len());
    let mut full2local = vec![None; cols.len()];
    let mut diag_add = T::zero();
    let mut has_lumped_entry = false;

    for t in 0..cols.len() {
        if keep[t] {
            full2local[t] = Some(ng_cols.len());
            ng_cols.push(cols[t]);
            ng_vals.push(vals[t]);
        } else if lump_diag && cols[t] != i {
            diag_add = diag_add + vals[t];
            has_lumped_entry = true;
        }
    }

    if has_lumped_entry {
        let diag_pos = match ng_cols.binary_search(&i) {
            Ok(pos) => {
                ng_vals[pos] = ng_vals[pos] + diag_add;
                pos
            }
            Err(pos) => {
                ng_cols.insert(pos, i);
                ng_vals.insert(pos, diag_add);
                for mapped in full2local.iter_mut().flatten() {
                    if *mapped >= pos {
                        *mapped += 1;
                    }
                }
                pos
            }
        };
        for t in 0..cols.len() {
            if !keep[t] && cols[t] != i {
                full2local[t] = Some(diag_pos);
            }
        }
    }

    NgBuiltRow {
        cols: ng_cols,
        vals: ng_vals,
        full2local,
    }
}

pub(crate) fn non_galerkin_filter_coarse<T: KrystScalar<Real = f64>>(
    pat: &CsrPattern,
    vals: &[T],
    symmetry: NgSymmetry,
    rf: NgRowFilter,
) -> (CsrPattern, Vec<T>, Vec<Option<usize>>) {
    let m = pat.nrows;
    let pr = &pat.row_ptr;
    let pc = &pat.col_idx;
    let nnz = pc.len();

    let mut keep = vec![false; nnz];

    let filter_row = |i: usize| {
        let rs = pr[i];
        let re = pr[i + 1];
        non_galerkin_row_keep(i, &pc[rs..re], &vals[rs..re], rf)
    };

    #[cfg(feature = "rayon")]
    let row_keep: Vec<Vec<bool>> = {
        use rayon::prelude::*;
        (0..m).into_par_iter().map(filter_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let row_keep: Vec<Vec<bool>> = (0..m).map(filter_row).collect();

    for (i, row) in row_keep.into_iter().enumerate() {
        keep[pr[i]..pr[i + 1]].copy_from_slice(&row);
    }

    // symmetry enforcement
    if let NgSymmetry::Symmetric = symmetry {
        for i in 0..m {
            let rs = pr[i];
            let re = pr[i + 1];
            for t in rs..re {
                let j = pc[t];
                if i == j {
                    continue;
                }
                let rjs = pr[j];
                let rje = pr[j + 1];
                if let Ok(pos) = pc[rjs..rje].binary_search(&i) {
                    let tj = rjs + pos;
                    if keep[t] ^ keep[tj] {
                        keep[t] = true;
                        keep[tj] = true;
                    }
                }
            }
        }
    }

    let build_row = |i: usize| {
        let rs = pr[i];
        let re = pr[i + 1];
        build_non_galerkin_row(i, &pc[rs..re], &vals[rs..re], &keep[rs..re], rf.lump_diag)
    };

    #[cfg(feature = "rayon")]
    let built_rows: Vec<NgBuiltRow<T>> = {
        use rayon::prelude::*;
        (0..m).into_par_iter().map(build_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let built_rows: Vec<NgBuiltRow<T>> = (0..m).map(build_row).collect();

    let mut ng_row_ptr = Vec::with_capacity(m + 1);
    let total_nnz = built_rows.iter().map(|row| row.cols.len()).sum();
    let mut ng_col_idx = Vec::with_capacity(total_nnz);
    let mut ng_vals = Vec::with_capacity(total_nnz);
    let mut full2ng = vec![None; nnz];
    ng_row_ptr.push(0);

    for (i, row) in built_rows.into_iter().enumerate() {
        let rs = pr[i];
        let row_start = ng_col_idx.len();
        for (local, mapped) in row.full2local.into_iter().enumerate() {
            if let Some(pos) = mapped {
                full2ng[rs + local] = Some(row_start + pos);
            }
        }
        ng_col_idx.extend(row.cols);
        ng_vals.extend(row.vals);
        ng_row_ptr.push(ng_col_idx.len());
    }

    let ng_pat = CsrPattern {
        nrows: pat.nrows,
        ncols: pat.ncols,
        row_ptr: ng_row_ptr,
        col_idx: ng_col_idx,
    };

    (ng_pat, ng_vals, full2ng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_drop() {
        // 3x3 dense pattern
        let pat = CsrPattern {
            nrows: 3,
            ncols: 3,
            row_ptr: vec![0, 3, 6, 9],
            col_idx: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        };
        let vals = vec![4.0, -0.1, 0.05, -0.1, 5.0, 0.02, 0.05, 0.02, 6.0];
        let rf = NgRowFilter {
            tau_abs: 0.1,
            tau_rel: 0.0,
            k_max: 0,
            lump_diag: true,
        };
        let (ng_pat, ng_vals, _) =
            non_galerkin_filter_coarse(&pat, &vals, NgSymmetry::Symmetric, rf);
        assert_eq!(ng_pat.row_ptr, vec![0, 3, 5, 7]);
        assert_eq!(ng_pat.col_idx, vec![0, 1, 2, 0, 1, 0, 2]);
        let expected_vals = vec![4.0, -0.1, 0.05, -0.1, 5.02, 0.05, 6.02];
        assert_eq!(ng_vals, expected_vals);
    }

    #[test]
    fn safety_keeps_a_zero_off_diagonal() {
        let keep = non_galerkin_row_keep(
            0,
            &[0, 1, 2],
            &[4.0, 0.0, 0.0],
            NgRowFilter {
                tau_abs: 1.0,
                tau_rel: 0.0,
                k_max: 0,
                lump_diag: false,
            },
        );
        assert_eq!(keep, vec![true, true, false]);
    }

    #[test]
    fn lumped_entries_map_to_inserted_diagonal_for_numeric_rebuild() {
        let pat = CsrPattern {
            nrows: 1,
            ncols: 3,
            row_ptr: vec![0, 2],
            col_idx: vec![1, 2],
        };
        let vals = vec![0.01, -0.02];
        let (ng_pat, ng_vals, full2ng) = non_galerkin_filter_coarse(
            &pat,
            &vals,
            NgSymmetry::None,
            NgRowFilter {
                tau_abs: 1.0,
                tau_rel: 0.0,
                k_max: 0,
                lump_diag: true,
            },
        );

        assert_eq!(ng_pat.row_ptr, vec![0, 2]);
        assert_eq!(ng_pat.col_idx, vec![0, 2]);
        assert_eq!(ng_vals, vec![0.01, -0.02]);
        assert_eq!(full2ng, vec![Some(0), Some(1)]);

        let rebuilt_full_vals = [2.0, 3.0];
        let mut rebuilt_ng_vals = vec![0.0; ng_pat.col_idx.len()];
        for (full, mapped) in rebuilt_full_vals.iter().zip(full2ng) {
            if let Some(ng) = mapped {
                rebuilt_ng_vals[ng] += full;
            }
        }
        assert_eq!(rebuilt_ng_vals, vec![2.0, 3.0]);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn complex_values_filter_by_magnitude_and_lump_complex_diag() {
        let pat = CsrPattern {
            nrows: 2,
            ncols: 3,
            row_ptr: vec![0, 3, 4],
            col_idx: vec![0, 1, 2, 1],
        };
        let vals = vec![
            crate::S::from_parts(4.0, 0.0),
            crate::S::from_parts(0.01, 0.02),
            crate::S::from_parts(-2.0, 1.0),
            crate::S::from_parts(3.0, 0.0),
        ];
        let rf = NgRowFilter {
            tau_abs: 0.1,
            tau_rel: 0.0,
            k_max: 0,
            lump_diag: true,
        };
        let (ng_pat, ng_vals, full2ng) =
            non_galerkin_filter_coarse(&pat, &vals, NgSymmetry::None, rf);

        assert_eq!(ng_pat.row_ptr, vec![0, 2, 3]);
        assert_eq!(ng_pat.col_idx, vec![0, 2, 1]);
        assert_eq!(full2ng, vec![Some(0), Some(0), Some(1), Some(2)]);
        assert_eq!(ng_vals[0], crate::S::from_parts(4.01, 0.02));
        assert_eq!(ng_vals[1], crate::S::from_parts(-2.0, 1.0));
        assert_eq!(ng_vals[2], crate::S::from_parts(3.0, 0.0));
    }
}
