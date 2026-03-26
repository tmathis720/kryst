use crate::algebra::parallel_cfg::parallel_tune;
use crate::algebra::scalar::KrystScalar;
use crate::error::KError;

use super::{IluCsr, Real};

pub fn tri_solve_serial(pc: &IluCsr, x: &[Real], y: &mut [Real]) -> Result<(), KError> {
    let n = pc.n();
    let lr = pc.l_row();
    let lc = pc.l_col();
    let lv = pc.l_val();
    let ur = pc.u_row();
    let uc = pc.u_col();
    let uv = pc.u_val();
    let di = pc.u_diag_ix();

    if y.len() != n || x.len() != n {
        return Err(KError::InvalidInput("tri_solve: dimension mismatch".into()));
    }

    // forward: L y = x  (unit diagonal)
    // Reuse y as the solution vector
    for i in 0..n {
        let mut s = x[i];
        for p in lr[i]..lr[i + 1] {
            let j = lc[p]; // j < i
            s -= lv[p] * y[j];
        }
        y[i] = s;
    }

    // backward: U z = y -> y <- z (output)
    for i in (0..n).rev() {
        let mut s = y[i];
        for p in ur[i]..ur[i + 1] {
            let j = uc[p];
            if j > i {
                s -= uv[p] * y[j];
            }
        }
        let d = uv[di[i]];
        y[i] = s / d;
    }
    Ok(())
}

pub fn tri_solve_level_scheduled(pc: &IluCsr, x: &[Real], y: &mut [Real]) -> Result<(), KError> {
    // Fallback to serial if no levels computed
    if pc.buckets_fwd().is_empty() || pc.buckets_bwd().is_empty() {
        return tri_solve_serial(pc, x, y);
    }

    let n = pc.n();
    if x.len() != n || y.len() != n {
        return Err(KError::InvalidInput("tri_solve: dimension mismatch".into()));
    }

    let lr = pc.l_row();
    let lc = pc.l_col();
    let lv = pc.l_val();
    let ur = pc.u_row();
    let uc = pc.u_col();
    let uv = pc.u_val();
    let di = pc.u_diag_ix();

    let tune = parallel_tune();
    let min_parallel_rows = tune.min_rows_ilu_triangular_level_parallel.max(1);
    let min_bucket_coalesce = tune.min_rows_ilu_triangular_bucket_coalesce.max(1);

    // Forward: L y = x (unit diagonal). Level-group parallel with serial commit.
    y.fill(Real::zero());
    for group in coalesce_buckets(pc.buckets_fwd(), min_bucket_coalesce) {
        if group_row_count(group) < min_parallel_rows {
            solve_forward_group_serial(group, x, y, lr, lc, lv);
        } else {
            solve_forward_group_parallel(group, x, y, lr, lc, lv);
        }
    }

    // Backward: U z = y → write z into y. Level-group parallel with serial commit.
    for group in coalesce_buckets(pc.buckets_bwd(), min_bucket_coalesce) {
        if group_row_count(group) < min_parallel_rows {
            solve_backward_group_serial(group, y, ur, uc, uv, di);
        } else {
            solve_backward_group_parallel(group, y, ur, uc, uv, di);
        }
    }

    Ok(())
}

fn coalesce_buckets<'a>(
    buckets: &'a [Vec<usize>],
    min_bucket_coalesce: usize,
) -> Vec<&'a [Vec<usize>]> {
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < buckets.len() {
        let mut end = start + 1;
        let mut rows = buckets[start].len();
        while end < buckets.len() && rows < min_bucket_coalesce {
            rows += buckets[end].len();
            end += 1;
        }
        groups.push(&buckets[start..end]);
        start = end;
    }
    groups
}

#[inline]
fn group_row_count(group: &[Vec<usize>]) -> usize {
    group.iter().map(|bucket| bucket.len()).sum()
}

fn solve_forward_group_serial(
    group: &[Vec<usize>],
    x: &[Real],
    y: &mut [Real],
    lr: &[usize],
    lc: &[usize],
    lv: &[Real],
) {
    for bucket in group {
        for &i in bucket {
            let mut s = x[i];
            for p in lr[i]..lr[i + 1] {
                s -= lv[p] * y[lc[p]];
            }
            y[i] = s;
        }
    }
}

fn solve_forward_group_parallel(
    group: &[Vec<usize>],
    x: &[Real],
    y: &mut [Real],
    lr: &[usize],
    lc: &[usize],
    lv: &[Real],
) {
    for bucket in group {
        #[cfg(feature = "rayon")]
        let solved: Vec<(usize, Real)> = {
            use rayon::prelude::*;
            bucket
                .par_iter()
                .map(|&i| {
                    let mut s = x[i];
                    for p in lr[i]..lr[i + 1] {
                        s -= lv[p] * y[lc[p]];
                    }
                    (i, s)
                })
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let solved: Vec<(usize, Real)> = bucket
            .iter()
            .map(|&i| {
                let mut s = x[i];
                for p in lr[i]..lr[i + 1] {
                    s -= lv[p] * y[lc[p]];
                }
                (i, s)
            })
            .collect();
        for (i, v) in solved {
            y[i] = v;
        }
    }
}

fn solve_backward_group_serial(
    group: &[Vec<usize>],
    y: &mut [Real],
    ur: &[usize],
    uc: &[usize],
    uv: &[Real],
    di: &[usize],
) {
    for bucket in group {
        for &i in bucket {
            let mut s = y[i];
            for p in ur[i]..ur[i + 1] {
                let j = uc[p];
                if j > i {
                    s -= uv[p] * y[j];
                }
            }
            y[i] = s / uv[di[i]];
        }
    }
}

fn solve_backward_group_parallel(
    group: &[Vec<usize>],
    y: &mut [Real],
    ur: &[usize],
    uc: &[usize],
    uv: &[Real],
    di: &[usize],
) {
    for bucket in group {
        #[cfg(feature = "rayon")]
        let solved: Vec<(usize, Real)> = {
            use rayon::prelude::*;
            bucket
                .par_iter()
                .map(|&i| {
                    let mut s = y[i];
                    for p in ur[i]..ur[i + 1] {
                        let j = uc[p];
                        if j > i {
                            s -= uv[p] * y[j];
                        }
                    }
                    (i, s / uv[di[i]])
                })
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let solved: Vec<(usize, Real)> = bucket
            .iter()
            .map(|&i| {
                let mut s = y[i];
                for p in ur[i]..ur[i + 1] {
                    let j = uc[p];
                    if j > i {
                        s -= uv[p] * y[j];
                    }
                }
                (i, s / uv[di[i]])
            })
            .collect();
        for (i, v) in solved {
            y[i] = v;
        }
    }
}

pub fn tri_solve_transpose_serial(
    pc: &IluCsr,
    ut_row: &[usize],
    ut_col: &[usize],
    ut_val: &[Real],
    lt_row: &[usize],
    lt_col: &[usize],
    lt_val: &[Real],
    x: &[Real],
    y: &mut [Real],
) -> Result<(), KError> {
    let n = pc.n();
    if x.len() != n || y.len() != n {
        return Err(KError::InvalidInput("tri_solve: dimension mismatch".into()));
    }

    // Forward solve with U^T (lower with diag)
    for i in 0..n {
        let mut s = x[i];
        for p in ut_row[i]..ut_row[i + 1] {
            let j = ut_col[p];
            if j < i {
                s -= ut_val[p] * y[j];
            }
        }
        let d = pc.u_val()[pc.u_diag_ix()[i]];
        y[i] = s / d;
    }

    // Backward solve with L^T (upper unit)
    for i in (0..n).rev() {
        let mut s = y[i];
        for p in lt_row[i]..lt_row[i + 1] {
            let j = lt_col[p];
            if j > i {
                s -= lt_val[p] * y[j];
            }
        }
        y[i] = s;
    }

    Ok(())
}
