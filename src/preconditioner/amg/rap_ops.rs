#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;

#[derive(Clone, Debug)]
pub struct CsrPattern {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
}

/// Symbolic RAP = R * A * P returns the pattern of the coarse operator.
pub fn rap_symbolic<T: KrystScalar>(
    r: &CsrMatrix<T>,
    a: &CsrMatrix<T>,
    p: &CsrMatrix<T>,
) -> CsrPattern {
    let nc = r.nrows();
    let rp_r = r.row_ptr();
    let cj_r = r.col_idx();
    let rp_a = a.row_ptr();
    let cj_a = a.col_idx();
    let rp_p = p.row_ptr();
    let cj_p = p.col_idx();

    let mut row_ptr = Vec::with_capacity(nc + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    row_ptr.push(0);

    for i in 0..nc {
        let mut cols: Vec<usize> = Vec::new();
        let rs_r = rp_r[i];
        let re_r = rp_r[i + 1];
        for rpos in rs_r..re_r {
            let k = cj_r[rpos]; // fine row index
            let rs_a = rp_a[k];
            let re_a = rp_a[k + 1];
            for apos in rs_a..re_a {
                let j = cj_a[apos]; // fine column index
                // columns in coarse correspond to columns present in P[j, :]
                let rs_p = rp_p[j];
                let re_p = rp_p[j + 1];
                for ppos in rs_p..re_p {
                    cols.push(cj_p[ppos]);
                }
            }
        }
        // unique + sort
        if !cols.is_empty() {
            cols.sort_unstable();
            cols.dedup();
            col_idx.extend_from_slice(&cols);
        }
        row_ptr.push(col_idx.len());
    }

    CsrPattern {
        nrows: r.nrows(),
        ncols: p.ncols(),
        row_ptr,
        col_idx,
    }
}

/// Numeric RAP using fixed pattern, writing values into out_vals (nnz = pat.col_idx.len()).
fn rap_numeric_row<T: KrystScalar + std::ops::AddAssign>(
    i: usize,
    pat: &CsrPattern,
    r: &CsrMatrix<T>,
    a: &CsrMatrix<T>,
    p: &CsrMatrix<T>,
) -> Vec<T> {
    let rp_r = r.row_ptr();
    let cj_r = r.col_idx();
    let vv_r = r.values();
    let rp_a = a.row_ptr();
    let cj_a = a.col_idx();
    let vv_a = a.values();
    let rp_p = p.row_ptr();
    let cj_p = p.col_idx();
    let vv_p = p.values();
    let row_start = pat.row_ptr[i];
    let row_end = pat.row_ptr[i + 1];
    let len = row_end - row_start;
    let cols = &pat.col_idx[row_start..row_end];
    let mut vals = vec![T::zero(); len];

    let rs_r = rp_r[i];
    let re_r = rp_r[i + 1];
    for rpos in rs_r..re_r {
        let k = cj_r[rpos];
        let r_ik = vv_r[rpos];
        let rs_a = rp_a[k];
        let re_a = rp_a[k + 1];
        for apos in rs_a..re_a {
            let j = cj_a[apos];
            let a_kj = vv_a[apos];
            let rs_p = rp_p[j];
            let re_p = rp_p[j + 1];
            for ppos in rs_p..re_p {
                let c = cj_p[ppos];
                let v = r_ik * a_kj * vv_p[ppos];
                if let Ok(idx) = cols.binary_search(&c) {
                    vals[idx] += v;
                }
            }
        }
    }

    vals
}

pub fn rap_numeric<T: KrystScalar + std::ops::AddAssign>(
    pat: &CsrPattern,
    r: &CsrMatrix<T>,
    a: &CsrMatrix<T>,
    p: &CsrMatrix<T>,
    out_vals: &mut [T],
) {
    assert_eq!(out_vals.len(), pat.col_idx.len());
    out_vals.fill(T::zero());

    let pr = &pat.row_ptr;

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let rows: Vec<(usize, Vec<T>)> = (0..pat.nrows)
            .into_par_iter()
            .map(|i| (i, rap_numeric_row(i, pat, r, a, p)))
            .collect();
        for (i, vals) in rows {
            let row_start = pr[i];
            let row_end = pr[i + 1];
            out_vals[row_start..row_end].copy_from_slice(&vals);
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        // Row-wise accumulation respecting pattern order
        for i in 0..pat.nrows {
            let row_start = pr[i];
            let row_end = pr[i + 1];
            if row_start == row_end {
                continue;
            }
            let vals = rap_numeric_row(i, pat, r, a, p);
            out_vals[row_start..row_end].copy_from_slice(&vals);
        }
    }
}
