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

/// CSR adjoint transpose (`A^H`) with a forward-entry to transpose-entry map.
///
/// For real scalars this is the ordinary transpose. For complex scalars this
/// conjugates values, matching kryst's transpose SpMV convention.
pub fn adjoint_csr_with_pos<T: KrystScalar>(a: &CsrMatrix<T>) -> (CsrMatrix<T>, Vec<usize>) {
    let (m, n) = (a.nrows(), a.ncols());
    let nnz = a.nnz();
    let mut row_ptr = vec![0usize; n + 1];
    for &j in a.col_idx() {
        row_ptr[j + 1] += 1;
    }
    for i in 0..n {
        row_ptr[i + 1] += row_ptr[i];
    }

    let mut col_idx = vec![0usize; nnz];
    let mut values = vec![T::zero(); nnz];
    let mut next = row_ptr.clone();
    let mut a2ah_pos = vec![0usize; nnz];
    for i in 0..m {
        for p in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_idx()[p];
            let dest = next[j];
            col_idx[dest] = i;
            values[dest] = a.values()[p].conj();
            a2ah_pos[p] = dest;
            next[j] += 1;
        }
    }

    (
        CsrMatrix::from_csr(n, m, row_ptr, col_idx, values),
        a2ah_pos,
    )
}

/// Symbolic RAP = R * A * P returns the pattern of the coarse operator.
fn rap_symbolic_row<T: KrystScalar>(
    i: usize,
    r: &CsrMatrix<T>,
    a: &CsrMatrix<T>,
    p: &CsrMatrix<T>,
) -> Vec<usize> {
    let mut cols = Vec::new();
    for rpos in r.row_ptr()[i]..r.row_ptr()[i + 1] {
        let k = r.col_idx()[rpos];
        for apos in a.row_ptr()[k]..a.row_ptr()[k + 1] {
            let j = a.col_idx()[apos];
            for ppos in p.row_ptr()[j]..p.row_ptr()[j + 1] {
                cols.push(p.col_idx()[ppos]);
            }
        }
    }
    cols.sort_unstable();
    cols.dedup();
    cols
}

pub fn rap_symbolic<T: KrystScalar>(
    r: &CsrMatrix<T>,
    a: &CsrMatrix<T>,
    p: &CsrMatrix<T>,
) -> CsrPattern {
    let nc = r.nrows();

    #[cfg(feature = "rayon")]
    let rows: Vec<Vec<usize>> = {
        use rayon::prelude::*;
        (0..nc)
            .into_par_iter()
            .map(|i| rap_symbolic_row(i, r, a, p))
            .collect()
    };

    #[cfg(not(feature = "rayon"))]
    let rows: Vec<Vec<usize>> = (0..nc).map(|i| rap_symbolic_row(i, r, a, p)).collect();

    let total_nnz = rows.iter().map(Vec::len).sum();
    let mut row_ptr = Vec::with_capacity(nc + 1);
    let mut col_idx = Vec::with_capacity(total_nnz);
    row_ptr.push(0);
    for cols in rows {
        col_idx.extend_from_slice(&cols);
        row_ptr.push(col_idx.len());
    }

    CsrPattern {
        nrows: r.nrows(),
        ncols: p.ncols(),
        row_ptr,
        col_idx,
    }
}

/// Symbolic Galerkin coarse pattern `P^H A P`.
pub fn galerkin_symbolic<T: KrystScalar>(a: &CsrMatrix<T>, p: &CsrMatrix<T>) -> CsrPattern {
    let (r, _) = adjoint_csr_with_pos(p);
    rap_symbolic(&r, a, p)
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

/// Numeric Galerkin coarse values `P^H A P` for the supplied pattern.
pub fn galerkin_numeric<T: KrystScalar + std::ops::AddAssign>(
    pat: &CsrPattern,
    a: &CsrMatrix<T>,
    p: &CsrMatrix<T>,
    out_vals: &mut [T],
) {
    let (r, _) = adjoint_csr_with_pos(p);
    rap_numeric(pat, &r, a, p, out_vals);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjoint_transpose_conjugates_values() {
        let p = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![
                crate::S::from_parts(1.0, 2.0),
                crate::S::from_parts(3.0, -4.0),
                crate::S::from_parts(-2.0, 5.0),
            ],
        );
        let (ph, p2ph) = adjoint_csr_with_pos(&p);
        assert_eq!(ph.row_ptr(), &[0, 1, 3]);
        assert_eq!(ph.col_idx(), &[0, 0, 1]);
        assert_eq!(p2ph, vec![0, 1, 2]);
        assert_eq!(ph.values()[0], crate::S::from_parts(1.0, -2.0));
        assert_eq!(ph.values()[1], crate::S::from_parts(3.0, 4.0));
        assert_eq!(ph.values()[2], crate::S::from_parts(-2.0, -5.0));
    }

    #[test]
    fn rap_symbolic_rows_are_sorted_deduplicated_and_preserve_empty_rows() {
        let r = CsrMatrix::from_csr(
            3,
            3,
            vec![0, 2, 2, 3],
            vec![0, 2, 1],
            vec![crate::S::one(); 3],
        );
        let a = CsrMatrix::from_csr(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 0, 2, 1, 2],
            vec![crate::S::one(); 6],
        );
        let p = CsrMatrix::from_csr(
            3,
            3,
            vec![0, 1, 3, 4],
            vec![2, 0, 2, 1],
            vec![crate::S::one(); 4],
        );

        let pattern = rap_symbolic(&r, &a, &p);

        assert_eq!(pattern.row_ptr, vec![0, 3, 3, 5]);
        assert_eq!(pattern.col_idx, vec![0, 1, 2, 1, 2]);
    }

    #[test]
    fn galerkin_numeric_matches_dense_complex_reference() {
        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![
                crate::S::from_parts(4.0, 0.0),
                crate::S::from_parts(1.0, 2.0),
                crate::S::from_parts(1.0, -2.0),
                crate::S::from_parts(3.0, 0.0),
            ],
        );
        let p = CsrMatrix::from_csr(
            2,
            1,
            vec![0, 1, 2],
            vec![0, 0],
            vec![
                crate::S::from_parts(1.0, 1.0),
                crate::S::from_parts(2.0, -1.0),
            ],
        );
        let pat = galerkin_symbolic(&a, &p);
        let mut vals = vec![crate::S::zero(); pat.col_idx.len()];
        galerkin_numeric(&pat, &a, &p, &mut vals);

        let p0 = p.values()[0];
        let p1 = p.values()[1];
        let expected = p0.conj() * a.values()[0] * p0
            + p0.conj() * a.values()[1] * p1
            + p1.conj() * a.values()[2] * p0
            + p1.conj() * a.values()[3] * p1;
        assert_eq!(pat.row_ptr, vec![0, 1]);
        assert_eq!(pat.col_idx, vec![0]);
        assert!((vals[0] - expected).abs() < 1e-12);
        assert!(vals[0].imag().abs() < 1e-12);
    }
}
