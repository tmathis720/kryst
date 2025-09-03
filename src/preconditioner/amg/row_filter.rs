use std::cmp::Ordering;

#[derive(Clone, Copy)]
pub struct RowFilter {
    pub tau_abs: f64,             // absolute drop (>= 0)
    pub tau_rel: f64,             // relative truncation in [0,1)
    pub k_max: usize,             // cap (0 => unlimited)
    pub must_keep: Option<usize>, // column index to force-keep
}

/// In-place filter of a single row according to absolute drop, relative truncation, and cap.
/// `cols` and `vals` are parallel arrays. Upon return, they are sorted by column index.
pub fn filter_row_by_truncation(cols: &mut Vec<usize>, vals: &mut Vec<f64>, rf: RowFilter) {
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
        let mut dropped_sum = 0.0;
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
                && !keep[pos] {
                    let mut replace: Option<usize> = None;
                    for &idx in order.iter().take(rf.k_max) {
                        if replace.is_none_or(|r| {
                            let cmp_mag = vals[idx].abs().total_cmp(&vals[r].abs());
                            cmp_mag == Ordering::Less
                                || (cmp_mag == Ordering::Equal && cols[idx] > cols[r])
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
    let mut pairs: Vec<(usize, f64)> = cols.iter().cloned().zip(vals.iter().cloned()).collect();
    pairs.sort_unstable_by_key(|(c, _)| *c);
    for (i, (c, v)) in pairs.into_iter().enumerate() {
        cols[i] = c;
        vals[i] = v;
    }
}

/// Apply row-wise filtering to an existing CSR matrix values slice, zeroing dropped entries.
pub fn apply_filter_to_csr_values_in_place(
    nrows: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &mut [f64],
    mut rf_for_row: impl FnMut(usize) -> RowFilter,
) {
    for i in 0..nrows {
        let rs = row_ptr[i];
        let re = row_ptr[i + 1];
        if rs == re {
            continue;
        }
        let mut cols: Vec<usize> = col_idx[rs..re].to_vec();
        let mut vs: Vec<f64> = vals[rs..re].to_vec();
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
                vals[p] = 0.0;
            }
        }
    }
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
}
