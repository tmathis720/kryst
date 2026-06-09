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

    // build filtered pattern and values
    let mut ng_row_ptr = Vec::with_capacity(m + 1);
    let mut ng_col_idx = Vec::new();
    let mut ng_vals = Vec::new();
    let mut full2ng = vec![None; nnz];
    ng_row_ptr.push(0);

    for i in 0..m {
        let rs = pr[i];
        let re = pr[i + 1];
        let mut diag_add = T::zero();
        for t in rs..re {
            let j = pc[t];
            if keep[t] {
                full2ng[t] = Some(ng_col_idx.len());
                ng_col_idx.push(j);
                ng_vals.push(vals[t]);
            } else if rf.lump_diag && j != i {
                diag_add = diag_add + vals[t];
            }
        }
        if rf.lump_diag && diag_add != T::zero() {
            // find diagonal position
            let row_start = ng_row_ptr.last().copied().unwrap();
            if let Ok(pos) = ng_col_idx[row_start..].binary_search(&i) {
                let idx = row_start + pos;
                ng_vals[idx] = ng_vals[idx] + diag_add;
            } else {
                // insert diag
                let pos = match ng_col_idx[row_start..].binary_search(&i) {
                    Ok(p) => row_start + p,
                    Err(p) => row_start + p,
                };
                ng_col_idx.insert(pos, i);
                ng_vals.insert(pos, diag_add);
            }
        }
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
        assert_eq!(full2ng, vec![Some(0), None, Some(1), Some(2)]);
        assert_eq!(ng_vals[0], crate::S::from_parts(4.01, 0.02));
        assert_eq!(ng_vals[1], crate::S::from_parts(-2.0, 1.0));
        assert_eq!(ng_vals[2], crate::S::from_parts(3.0, 0.0));
    }
}
