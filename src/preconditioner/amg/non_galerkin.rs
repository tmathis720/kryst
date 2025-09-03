use super::{NgSymmetry, rap_ops::CsrPattern};

#[derive(Clone, Copy)]
pub(crate) struct NgRowFilter {
    pub tau_abs: f64,
    pub tau_rel: f64,
    pub k_max: usize,
    pub lump_diag: bool,
}

pub(crate) fn non_galerkin_filter_coarse(
    pat: &CsrPattern,
    vals: &[f64],
    symmetry: NgSymmetry,
    rf: NgRowFilter,
) -> (CsrPattern, Vec<f64>, Vec<Option<usize>>) {
    let m = pat.nrows;
    let pr = &pat.row_ptr;
    let pc = &pat.col_idx;
    let nnz = pc.len();

    let mut keep = vec![false; nnz];

    // First pass: row-wise filtering
    for i in 0..m {
        let rs = pr[i];
        let re = pr[i + 1];
        if rs == re {
            continue;
        }
        let mut idx: Vec<usize> = (rs..re).collect();
        // absolute and relative drops
        if rf.tau_abs > 0.0 || rf.tau_rel > 0.0 || (rf.k_max > 0 && re - rs > rf.k_max) {
            // sort indices by |v| asc then col
            idx.sort_unstable_by(|&u, &v| {
                let au = vals[u].abs();
                let av = vals[v].abs();
                au.total_cmp(&av).then_with(|| pc[u].cmp(&pc[v]))
            });
            let mut drop_mask = vec![false; re - rs];
            let mut dropped_sum = 0.0f64;
            let mut l1 = 0.0f64;
            for t in rs..re {
                if pc[t] != i {
                    l1 += vals[t].abs();
                }
            }
            for (order_pos, &t) in idx.iter().enumerate() {
                let col = pc[t];
                let v = vals[t];
                if col == i {
                    continue;
                }
                let by_abs = v.abs() < rf.tau_abs;
                let allow = rf.tau_rel * l1;
                let by_rel = dropped_sum + v.abs() <= allow + 1e-300;
                if by_abs || (rf.tau_rel > 0.0 && by_rel) {
                    drop_mask[t - rs] = true;
                    dropped_sum += v.abs();
                }
            }
            // cap
            if rf.k_max > 0 {
                let mut order_keep: Vec<usize> = (rs..re).collect();
                order_keep.sort_unstable_by(|&u, &v| {
                    let au = vals[u].abs();
                    let av = vals[v].abs();
                    av.total_cmp(&au).then_with(|| pc[u].cmp(&pc[v]))
                });
                let mut kept_off = 0usize;
                for &t in &order_keep {
                    if pc[t] == i {
                        keep[t] = true;
                        continue;
                    }
                    if kept_off < rf.k_max && !drop_mask[t - rs] {
                        keep[t] = true;
                        kept_off += 1;
                    }
                }
                // mark remaining non-dropped as keep
                for t in rs..re {
                    if pc[t] == i {
                        keep[t] = true;
                    } else if !drop_mask[t - rs] && !keep[t] && kept_off < rf.k_max {
                        keep[t] = true;
                        kept_off += 1;
                    }
                }
            } else {
                for t in rs..re {
                    if pc[t] == i || !drop_mask[t - rs] {
                        keep[t] = true;
                    }
                }
            }
        } else {
            for t in rs..re {
                keep[t] = true;
            }
        }

        // safety: ensure at least diag and one off-diagonal if row had any off-diagonals
        let kept_off = (rs..re).filter(|&t| keep[t] && pc[t] != i).count();
        let had_off = (rs..re).any(|t| pc[t] != i);
        if kept_off == 0 && had_off {
            // keep largest off-diagonal
            let mut best = rs;
            let mut best_mag = 0.0;
            for t in rs..re {
                if pc[t] != i {
                    let mag = vals[t].abs();
                    if mag > best_mag {
                        best = t;
                        best_mag = mag;
                    }
                }
            }
            keep[best] = true;
        }
        // always keep diagonal if present
        for t in rs..re {
            if pc[t] == i {
                keep[t] = true;
                break;
            }
        }
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
        let mut diag_add = 0.0;
        for t in rs..re {
            let j = pc[t];
            if keep[t] {
                full2ng[t] = Some(ng_col_idx.len());
                ng_col_idx.push(j);
                ng_vals.push(vals[t]);
            } else if rf.lump_diag && j != i {
                diag_add += vals[t];
            }
        }
        if rf.lump_diag && diag_add != 0.0 {
            // find diagonal position
            let row_start = ng_row_ptr.last().copied().unwrap();
            if let Ok(pos) = ng_col_idx[row_start..].binary_search(&i) {
                let idx = row_start + pos;
                ng_vals[idx] += diag_add;
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
}
