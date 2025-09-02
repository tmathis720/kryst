use crate::algebra::scalar::Scalar;

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
    use super::{merged_dot_prefix, merged_dot_prefix_kahan};

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
}
