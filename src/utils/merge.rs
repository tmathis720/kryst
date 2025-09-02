use crate::algebra::scalar::Scalar;

/// Map a slice of value slots into an iterator over references into the
/// underlying value array.
#[inline]
pub fn map_vals<'a, S>(vals: &'a [S], slots: &'a [usize]) -> impl Iterator<Item = &'a S> {
    slots.iter().map(move |&p| &vals[p])
}

/// Lookup a value within a sorted sparse row returning `None` if the column is
/// absent.
#[inline]
pub fn lookup_in_row<S: Scalar>(cols: &[usize], vals: &[S], col: usize) -> Option<S> {
    match cols.binary_search(&col) {
        Ok(pos) => Some(vals[pos]),
        Err(_) => None,
    }
}

/// Compute the dot product of two sparse rows up to a column limit by
/// simultaneously walking both index arrays.
#[inline]
pub fn merged_dot_prefix<S: Scalar>(
    a_cols: &[usize],
    a_vals: &[S],
    b_cols: &[usize],
    b_vals: &[S],
    col_limit: usize,
) -> S {
    let mut i = 0;
    let mut j = 0;
    let mut acc = S::zero();
    while i < a_cols.len() && j < b_cols.len() {
        let ci = a_cols[i];
        if ci >= col_limit {
            break;
        }
        let cj = b_cols[j];
        if cj >= col_limit {
            break;
        }
        if ci == cj {
            acc = acc + a_vals[i] * b_vals[j];
            i += 1;
            j += 1;
        } else if ci < cj {
            i += 1;
        } else {
            j += 1;
        }
    }
    acc
}

/// Compute the dot product of the strictly upper part of two rows after a
/// given starting column. The walk skips all columns `<= start_col` in the
/// first row and only accumulates products where columns match and are greater
/// than `start_col`.
#[inline]
pub fn merged_dot_strict_upper<S: Scalar>(
    a_cols: &[usize],
    a_vals: &[S],
    start_col: usize,
    b_cols: &[usize],
    b_vals: &[S],
) -> S {
    let mut i = match a_cols.binary_search(&(start_col + 1)) {
        Ok(idx) => idx,
        Err(idx) => idx,
    };
    let mut j = 0;
    let mut acc = S::zero();
    while i < a_cols.len() && j < b_cols.len() {
        let ci = a_cols[i];
        let cj = b_cols[j];
        if ci == cj {
            acc = acc + a_vals[i] * b_vals[j];
            i += 1;
            j += 1;
        } else if ci < cj {
            i += 1;
        } else {
            j += 1;
        }
    }
    acc
}

/// Kahan compensated variant of [`merged_dot_prefix`] for improved numerical
/// reproducibility.
#[inline]
pub fn merged_dot_prefix_kahan<S: Scalar>(
    a_cols: &[usize],
    a_vals: &[S],
    b_cols: &[usize],
    b_vals: &[S],
    col_limit: usize,
) -> S {
    let mut i = 0;
    let mut j = 0;
    let mut sum = S::zero();
    let mut c = S::zero();
    while i < a_cols.len() && j < b_cols.len() {
        let ci = a_cols[i];
        if ci >= col_limit {
            break;
        }
        let cj = b_cols[j];
        if cj >= col_limit {
            break;
        }
        if ci == cj {
            let prod = a_vals[i] * b_vals[j];
            let y = prod - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
            i += 1;
            j += 1;
        } else if ci < cj {
            i += 1;
        } else {
            j += 1;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::{lookup_in_row, merged_dot_prefix, merged_dot_prefix_kahan, merged_dot_strict_upper};

    #[test]
    fn basic_prefix() {
        let a_cols = [0, 2, 5];
        let a_vals = [1.0, 2.0, 3.0];
        let b_cols = [1, 2, 4, 5];
        let b_vals = [4.0, 5.0, 6.0, 7.0];
        let res = merged_dot_prefix(&a_cols, &a_vals, &b_cols, &b_vals, 5);
        assert_eq!(res, 2.0 * 5.0);
    }

    #[test]
    fn kahan_matches_standard() {
        let a_cols = [0, 1, 2];
        let a_vals = [1e16, 1.0, -1e16];
        let b_cols = [0, 1, 2];
        let b_vals = [1.0, 1.0, 1.0];
        let res_std = merged_dot_prefix(&a_cols, &a_vals, &b_cols, &b_vals, 3);
        let res_kahan = merged_dot_prefix_kahan(&a_cols, &a_vals, &b_cols, &b_vals, 3);
        assert!((res_std - res_kahan).abs() <= 1e-10);
    }

    #[test]
    fn strict_upper() {
        let a_cols = [0, 2, 4, 7];
        let a_vals = [1.0, 2.0, 3.0, 4.0];
        let b_cols = [1, 2, 4, 6, 7];
        let b_vals = [5.0, 6.0, 7.0, 8.0, 9.0];
        let res = merged_dot_strict_upper(&a_cols, &a_vals, 2, &b_cols, &b_vals);
        // columns greater than 2 that match are 4 and 7
        assert_eq!(res, 3.0 * 7.0 + 4.0 * 9.0);
    }

    #[test]
    fn lookup_basic() {
        let cols = [0, 3, 5];
        let vals = [1.0, 2.0, 3.0];
        assert_eq!(lookup_in_row(&cols, &vals, 3), Some(2.0));
        assert!(lookup_in_row(&cols, &vals, 2).is_none());
    }
}
