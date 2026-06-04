#![allow(dead_code)]

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;

use super::row_filter::{RowFilter, filter_row_by_truncation};
use super::strength::Strength;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use faer::linalg::solvers::{FullPivLu, SolveCore};
use faer::{Conj, Mat, MatMut};

#[derive(Clone, Debug)]
pub struct TentativeP {
    pub agg_of: Vec<usize>,
    pub n_coarse: usize,
    pub num_functions: usize,
    pub nns: Option<Vec<Vec<f64>>>,
    pub comp_of: Option<Vec<usize>>,
}

pub fn tentative_from_aggregates(agg: Vec<usize>) -> TentativeP {
    let n_coarse = 1 + agg.iter().copied().max().unwrap_or(0);
    TentativeP {
        agg_of: agg,
        n_coarse,
        num_functions: 1,
        nns: None,
        comp_of: None,
    }
}

#[derive(Clone, Debug)]
pub struct TentativeNodal {
    pub agg_of: Vec<usize>,
    pub n_agg: usize,
    pub mfun: usize,
    pub row_basis: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Pcsr {
    pub m: usize,
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveWeight {
    None,
    Diag,
    RowNorm,
}

pub fn sample_low_modes(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    r: usize,
    nu: usize,
    omega: f64,
    seed: u64,
) -> Result<Mat<f64>, KError> {
    let n = a.nrows();
    let mut u_f = Mat::<f64>::zeros(n, r);
    if r == 0 {
        return Ok(u_f);
    }
    let mut v = vec![0.0f64; n];
    let mut u = vec![0.0f64; n];
    let mut tmp = vec![0.0f64; n];
    for k in 0..r {
        for i in 0..n {
            let t = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed.wrapping_add(k as u64));
            let bits = ((t >> 17) & 0xFFFF_FFFF) as u32;
            let rand = (bits as f64) * (1.0 / (u32::MAX as f64)) - 0.5;
            v[i] = rand;
        }
        u.copy_from_slice(&v);
        for _ in 0..nu {
            a.spmv_scaled(1.0, &u, 0.0, &mut tmp)?;
            for i in 0..n {
                u[i] -= omega * d_inv[i] * tmp[i];
            }
        }
        for prev in 0..k {
            let mut dot = 0.0;
            for i in 0..n {
                dot += u[i] * u_f[(i, prev)];
            }
            for i in 0..n {
                u[i] -= dot * u_f[(i, prev)];
            }
        }
        let mut norm_sq = 0.0f64;
        for &val in &u {
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt().max(1e-30);
        for i in 0..n {
            u_f[(i, k)] = u[i] / norm;
        }
    }
    Ok(u_f)
}

pub fn restrict_samples_to_coarse(
    a: &CsrMatrix<f64>,
    tp: &TentativeP,
    u_f: &Mat<f64>,
    mode: AdaptiveWeight,
) -> Mat<f64> {
    let n = a.nrows();
    let r = u_f.ncols();
    let n_coarse = tp.n_coarse;
    let mut u_c = Mat::<f64>::zeros(n_coarse, r);
    if n_coarse == 0 || r == 0 {
        return u_c;
    }
    let mut weights = vec![1.0f64; n];
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    match mode {
        AdaptiveWeight::None => {}
        AdaptiveWeight::Diag => {
            for i in 0..n {
                let mut diag = 0.0;
                for p in rp[i]..rp[i + 1] {
                    if cj[p] == i {
                        diag = vv[p].abs();
                        break;
                    }
                }
                weights[i] = diag.max(1e-30);
            }
        }
        AdaptiveWeight::RowNorm => {
            for i in 0..n {
                let mut sum = 0.0;
                for p in rp[i]..rp[i + 1] {
                    if cj[p] != i {
                        sum += vv[p].abs();
                    }
                }
                weights[i] = sum.max(1e-30);
            }
        }
    }
    let mut wsum = vec![0.0f64; n_coarse];
    for i in 0..n {
        let agg = tp.agg_of[i];
        let wi = weights[i];
        wsum[agg] += wi;
        for alpha in 0..r {
            u_c[(agg, alpha)] += wi * u_f[(i, alpha)];
        }
    }
    for c in 0..n_coarse {
        let denom = wsum[c].max(1e-30);
        for alpha in 0..r {
            u_c[(c, alpha)] /= denom;
        }
    }
    u_c
}

pub fn adaptive_fit_values_only<T: KrystScalar<Real = f64>>(
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    out_vals: &mut [T],
    tp: &TentativeP,
    u_f: &Mat<f64>,
    u_c: &Mat<f64>,
    lambda: f64,
    enforce_sum1: bool,
    trunc: f64,
) -> Result<(), KError> {
    if tp.num_functions != 1 {
        return Ok(());
    }
    let n = tp.agg_of.len();
    if p_row_ptr.len() != n + 1 {
        return Err(KError::InvalidInput("adaptive fit: row_ptr length".into()));
    }
    if u_f.nrows() != n {
        return Err(KError::InvalidInput(
            "adaptive fit: fine samples mismatch".into(),
        ));
    }
    if u_c.nrows() < tp.n_coarse {
        return Err(KError::InvalidInput(
            "adaptive fit: coarse samples mismatch".into(),
        ));
    }
    if p_col_idx.len() != out_vals.len() {
        return Err(KError::InvalidInput("adaptive fit: values length".into()));
    }
    let r = u_f.ncols();
    if r == 0 {
        return Ok(());
    }
    for i in 0..n {
        let rs = p_row_ptr[i];
        let re = p_row_ptr[i + 1];
        let m = re - rs;
        if m == 0 {
            continue;
        }
        if m == 1 {
            out_vals[rs] = T::one();
            continue;
        }
        let mut gram = Mat::<f64>::zeros(m, m);
        let mut rhs_base = vec![0.0f64; m];
        for row in 0..m {
            let c_row = p_col_idx[rs + row];
            let mut rhs_val = 0.0;
            for alpha in 0..r {
                rhs_val += u_c[(c_row, alpha)] * u_f[(i, alpha)];
            }
            rhs_base[row] = rhs_val;
            for col in row..m {
                let c_col = p_col_idx[rs + col];
                let mut val = 0.0;
                for alpha in 0..r {
                    val += u_c[(c_row, alpha)] * u_c[(c_col, alpha)];
                }
                gram[(row, col)] = val;
                if row != col {
                    gram[(col, row)] = val;
                }
            }
        }
        let gram_base = gram.clone();
        let mut lambda_cur = lambda.max(0.0);
        let mut solved = false;
        let mut best = vec![0.0f64; m];
        for _ in 0..3 {
            let mut gram = gram_base.clone();
            for d in 0..m {
                gram[(d, d)] += lambda_cur;
            }
            if enforce_sum1 {
                let mut kkt = Mat::<f64>::zeros(m + 1, m + 1);
                for row in 0..m {
                    for col in 0..m {
                        kkt[(row, col)] = gram[(row, col)];
                    }
                    kkt[(row, m)] = 1.0;
                    kkt[(m, row)] = 1.0;
                }
                let mut rhs = vec![0.0f64; m + 1];
                rhs[..m].copy_from_slice(&rhs_base[..m]);
                rhs[m] = 1.0;
                let lu = FullPivLu::new(kkt.as_ref());
                let rhs_mat = MatMut::from_column_major_slice_mut(&mut rhs, m + 1, 1);
                lu.solve_in_place_with_conj(Conj::No, rhs_mat);
                if rhs.iter().any(|x| !x.is_finite()) {
                    lambda_cur = if lambda_cur == 0.0 {
                        1e-10
                    } else {
                        lambda_cur * 10.0
                    };
                    continue;
                }
                best.copy_from_slice(&rhs[..m]);
                solved = true;
            } else {
                let lu = FullPivLu::new(gram.as_ref());
                let mut rhs = rhs_base.clone();
                let rhs_mat = MatMut::from_column_major_slice_mut(&mut rhs, m, 1);
                lu.solve_in_place_with_conj(Conj::No, rhs_mat);
                if rhs.iter().any(|x| !x.is_finite()) {
                    lambda_cur = if lambda_cur == 0.0 {
                        1e-10
                    } else {
                        lambda_cur * 10.0
                    };
                    continue;
                }
                best.copy_from_slice(&rhs);
                solved = true;
            }
            break;
        }
        if !solved {
            continue;
        }
        if trunc > 0.0 {
            for val in &mut best {
                if val.abs() < trunc {
                    *val = 0.0;
                }
            }
        }
        if enforce_sum1 {
            let sum = best.iter().copied().sum::<f64>();
            if sum.abs() < 1e-30 {
                continue;
            }
            for val in &mut best {
                *val /= sum;
            }
        }
        for (local, idx) in (rs..re).enumerate() {
            out_vals[idx] = T::from_real(best[local]);
        }
    }
    Ok(())
}

// ===== Classical interpolation family =======================================

/// Variants of classical interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassicalVariant {
    Direct,
    Standard,
    HE,
}

/// Coarse/fine bookkeeping produced by `classical_pattern`.
#[derive(Clone, Debug)]
pub struct CFInfo {
    pub is_c: Vec<bool>,
    pub coarse_of: Vec<Option<usize>>, // Some(k) if C, None if F
}

/// Parameters controlling value computation for the classical family.
#[derive(Clone, Debug)]
pub struct ClassicalParams {
    pub variant: ClassicalVariant,
    pub extended: bool,
    pub drop_abs: f64,
    pub trunc_rel: f64,
    pub cap_row: usize,
    pub keep_at_least_one: bool,
}

fn build_cf_info(is_c: &[bool]) -> CFInfo {
    let mut coarse_of = vec![None; is_c.len()];
    let mut k = 0usize;
    for (i, &c) in is_c.iter().enumerate() {
        if c {
            coarse_of[i] = Some(k);
            k += 1;
        }
    }
    CFInfo {
        is_c: is_c.to_vec(),
        coarse_of,
    }
}

/// Helper: get entry a[i,j] via binary search in row i.
fn csr_get(a: &CsrMatrix<f64>, i: usize, j: usize) -> Option<f64> {
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    let rs = rp[i];
    let re = rp[i + 1];
    cj[rs..re].binary_search(&j).ok().map(|p| vv[rs + p])
}

/// Build classical interpolation pattern.
pub fn classical_pattern(
    a: &CsrMatrix<f64>,
    s_sym: &Strength,
    is_c: &[bool],
    extended: bool,
) -> (Pcsr, CFInfo) {
    let n = a.nrows();
    let cf = build_cf_info(is_c);
    let ncoarse = cf
        .coarse_of
        .iter()
        .filter_map(|&x| x)
        .max()
        .map(|x| x + 1)
        .unwrap_or(0);

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::<usize>::new();
    let mut vals = Vec::<f64>::new();
    row_ptr.push(0);

    for i in 0..n {
        if cf.is_c[i] {
            let k = cf.coarse_of[i].unwrap();
            col_idx.push(k);
            vals.push(0.0);
            row_ptr.push(col_idx.len());
            continue;
        }

        let mut cols: Vec<usize> = Vec::new();
        let rs = s_sym.row_ptr[i];
        let re = s_sym.row_ptr[i + 1];
        for &j in &s_sym.col_idx[rs..re] {
            if cf.is_c[j] {
                cols.push(cf.coarse_of[j].unwrap());
            }
        }
        if extended {
            for &j in &s_sym.col_idx[rs..re] {
                if cf.is_c[j] {
                    continue;
                }
                let rj = s_sym.row_ptr[j];
                let ej = s_sym.row_ptr[j + 1];
                for &k in &s_sym.col_idx[rj..ej] {
                    if cf.is_c[k] {
                        cols.push(cf.coarse_of[k].unwrap());
                    }
                }
            }
        }
        cols.sort_unstable();
        cols.dedup();

        if cols.is_empty() {
            // weak search in A
            let rp = a.row_ptr();
            let cj = a.col_idx();
            let rs_a = rp[i];
            let re_a = rp[i + 1];
            let mut best: Option<usize> = None;
            let mut bestmag = 0.0;
            for p in rs_a..re_a {
                let j = cj[p];
                if cf.is_c[j] {
                    let k = cf.coarse_of[j].unwrap();
                    let v = a.values()[p].abs();
                    if v > bestmag {
                        bestmag = v;
                        best = Some(k);
                    }
                }
            }
            if let Some(k) = best {
                cols.push(k);
            } else if ncoarse > 0 {
                cols.push(0);
            }
        }

        for c in cols {
            col_idx.push(c);
            vals.push(0.0);
        }
        row_ptr.push(col_idx.len());
    }

    (
        Pcsr {
            m: n,
            n: ncoarse,
            row_ptr,
            col_idx,
            vals,
        },
        cf,
    )
}

/// For F-neighbor j, distribute influence over its strong C neighbors.
#[allow(non_snake_case)]
fn neighbor_distribution_over_C_of(
    j: usize,
    a: &CsrMatrix<f64>,
    s_sym: &Strength,
    cf: &CFInfo,
    cols: &mut Vec<usize>,
    wts: &mut Vec<f64>,
) {
    cols.clear();
    wts.clear();
    let rs = s_sym.row_ptr[j];
    let re = s_sym.row_ptr[j + 1];
    let mut tmp: Vec<(usize, f64)> = Vec::new();
    let mut sum_neg = 0.0;
    let mut sum_pos = 0.0;
    for &nbr in &s_sym.col_idx[rs..re] {
        if cf.is_c[nbr] {
            let c = cf.coarse_of[nbr].unwrap();
            if let Some(v) = csr_get(a, j, nbr) {
                tmp.push((c, v));
                if v < 0.0 {
                    sum_neg += -v;
                } else {
                    sum_pos += v;
                }
            }
        }
    }
    if tmp.is_empty() {
        return;
    }
    for (c, v) in tmp {
        let w = if v < 0.0 {
            if sum_neg > 0.0 { (-v) / sum_neg } else { 0.0 }
        } else if sum_pos > 0.0 {
            v / sum_pos
        } else {
            0.0
        };
        cols.push(c);
        wts.push(w);
    }
    let s: f64 = wts.iter().sum();
    if s > 0.0 {
        for w in wts.iter_mut() {
            *w /= s;
        }
    } else {
        let u = 1.0 / (wts.len() as f64);
        for w in wts.iter_mut() {
            *w = u;
        }
    }
}

/// Values-only refresh for classical interpolation.
pub fn classical_values_only<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<f64>,
    s_sym: &Strength,
    cf: &CFInfo,
    params: &ClassicalParams,
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    out_vals: &mut [T],
) -> Result<(), crate::error::KError> {
    let n = a.nrows();
    assert_eq!(p_row_ptr.len(), n + 1);
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    let mut buf_cols = Vec::<usize>::new();
    let mut buf_wts = Vec::<f64>::new();

    for i in 0..n {
        let rs_p = p_row_ptr[i];
        let re_p = p_row_ptr[i + 1];
        if cf.is_c[i] {
            for k in rs_p..re_p {
                out_vals[k] = T::zero();
            }
            if let Some(kc) = cf.coarse_of[i] {
                for k in rs_p..re_p {
                    if p_col_idx[k] == kc {
                        out_vals[k] = T::one();
                        break;
                    }
                }
            }
            continue;
        }

        let mut contrib_cols: Vec<usize> = Vec::new();
        let mut contrib_vals: Vec<f64> = Vec::new();

        let rs = s_sym.row_ptr[i];
        let re = s_sym.row_ptr[i + 1];

        // Direct part
        let mut sum_neg = 0.0;
        let mut sum_pos = 0.0;
        for &j in &s_sym.col_idx[rs..re] {
            if cf.is_c[j]
                && let Some(aij) = csr_get(a, i, j)
            {
                if params.variant == ClassicalVariant::Direct {
                    if aij < 0.0 {
                        sum_neg += -aij;
                    } else {
                        sum_pos += aij;
                    }
                } else {
                    let col = cf.coarse_of[j].unwrap();
                    contrib_cols.push(col);
                    contrib_vals.push(-aij);
                }
            }
        }
        if params.variant == ClassicalVariant::Direct {
            for &j in &s_sym.col_idx[rs..re] {
                if cf.is_c[j]
                    && let Some(aij) = csr_get(a, i, j)
                {
                    let col = cf.coarse_of[j].unwrap();
                    let w = if aij < 0.0 {
                        if sum_neg > 0.0 { (-aij) / sum_neg } else { 0.0 }
                    } else if sum_pos > 0.0 {
                        aij / sum_pos
                    } else {
                        0.0
                    };
                    contrib_cols.push(col);
                    contrib_vals.push(w);
                }
            }
        }

        if matches!(
            params.variant,
            ClassicalVariant::Standard | ClassicalVariant::HE
        ) {
            for &j in &s_sym.col_idx[rs..re] {
                if cf.is_c[j] {
                    continue;
                }
                let aij = match csr_get(a, i, j) {
                    Some(v) => v,
                    None => continue,
                };
                if aij == 0.0 {
                    continue;
                }
                neighbor_distribution_over_C_of(j, a, s_sym, cf, &mut buf_cols, &mut buf_wts);
                let scale = if matches!(params.variant, ClassicalVariant::HE) {
                    let mut rowsum = 0.0;
                    let mut ajj = 0.0;
                    let rj = rp[j];
                    let ej = rp[j + 1];
                    for p in rj..ej {
                        let v = vv[p];
                        if cj[p] == j {
                            ajj = v.abs();
                        } else {
                            rowsum += v.abs();
                        }
                    }
                    let denom = ajj.max(rowsum).max(1e-30);
                    (-aij) / denom
                } else {
                    -aij
                };
                for t in 0..buf_cols.len() {
                    contrib_cols.push(buf_cols[t]);
                    contrib_vals.push(scale * buf_wts[t]);
                }
            }

            // denominator
            let mut sum_neg_strong = 0.0;
            for &j in &s_sym.col_idx[rs..re] {
                if let Some(aij) = csr_get(a, i, j)
                    && aij < 0.0
                {
                    sum_neg_strong += -aij;
                }
            }
            let di = csr_get(a, i, i).unwrap_or(1.0);
            let di_eff = di - sum_neg_strong;
            let denom = if di_eff.abs() >= 1e-14 * di.abs().max(1.0) {
                di_eff
            } else {
                di
            };
            if denom.abs() < 1e-30 {
                let mut s = 0.0;
                for v in &contrib_vals {
                    s += v.abs();
                }
                if s > 0.0 {
                    for v in &mut contrib_vals {
                        *v /= s;
                    }
                }
            } else {
                for v in &mut contrib_vals {
                    *v /= denom;
                }
            }
        }

        if !contrib_cols.is_empty() {
            let mut idx: Vec<usize> = (0..contrib_cols.len()).collect();
            idx.sort_unstable_by(|&u, &v| contrib_cols[u].cmp(&contrib_cols[v]));
            let mut last = contrib_cols[idx[0]];
            let mut acc = 0.0;
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            for &id in &idx {
                let c = contrib_cols[id];
                if c == last {
                    acc += contrib_vals[id];
                } else {
                    if acc != 0.0 {
                        cols.push(last);
                        vals.push(acc);
                    }
                    last = c;
                    acc = contrib_vals[id];
                }
            }
            if acc != 0.0 {
                cols.push(last);
                vals.push(acc);
            }

            let kept_cols = cols.clone();
            let kept_vals = vals.clone();
            let rf = RowFilter {
                tau_abs: params.drop_abs,
                tau_rel: params.trunc_rel,
                k_max: params.cap_row,
                must_keep: None,
            };
            filter_row_by_truncation(&mut cols, &mut vals, rf);
            if params.keep_at_least_one && cols.is_empty() && !kept_cols.is_empty() {
                let mut best = 0usize;
                let mut bestmag = kept_vals[0].abs();
                for t in 1..kept_cols.len() {
                    let m = kept_vals[t].abs();
                    if m > bestmag {
                        bestmag = m;
                        best = t;
                    }
                }
                cols.push(kept_cols[best]);
                vals.push(kept_vals[best]);
            }
            for k in rs_p..re_p {
                let c = p_col_idx[k];
                match cols.binary_search(&c) {
                    Ok(pos) => out_vals[k] = T::from_real(vals[pos]),
                    Err(_) => out_vals[k] = T::zero(),
                }
            }
        } else {
            for k in rs_p..re_p {
                out_vals[k] = T::zero();
            }
            if rs_p < re_p {
                out_vals[rs_p] = T::one();
            }
        }
    }
    Ok(())
}

/// Build smoothed aggregation prolongator using one SA sweep.
/// Pattern is fixed by the set {agg(i)} ∪ {agg(j) for (i,j) in A} with
/// optional drop and cap. Values follow P = (I - ω D^{-1} A) P_t.
pub fn smooth_tentative_sa(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    tp: &TentativeP,
    omega: f64,
    drop_tol: f64,
    max_per_row: usize,
    trunc_rel: f64,
) -> Pcsr {
    let m = a.nrows();
    let ncoarse = tp.n_coarse;
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut marker: Vec<isize> = vec![-1; ncoarse.min(1024).max(512)];
    let mut acc_cols: Vec<usize> = Vec::new();
    let mut acc_vals: Vec<f64> = Vec::new();

    for i in 0..m {
        if marker.len() < ncoarse {
            marker.resize(ncoarse, -1);
        }
        acc_cols.clear();
        acc_vals.clear();

        // Start with 1.0 at own aggregate
        let myc = tp.agg_of[i];
        marker[myc] = 0;
        acc_cols.push(myc);
        acc_vals.push(1.0);

        // Accumulate -ω d_i a_ij into coarse columns of neighbors' aggregates
        let di = d_inv[i];
        let rs = rp[i];
        let re = rp[i + 1];
        for p in rs..re {
            let j = cj[p];
            if j == i {
                continue;
            }
            let cjg = tp.agg_of[j];
            let v = -omega * di * vv[p];
            let k = marker[cjg];
            if k >= 0 {
                acc_vals[k as usize] += v;
            } else {
                marker[cjg] = acc_cols.len() as isize;
                acc_cols.push(cjg);
                acc_vals.push(v);
            }
        }

        let mut cols: Vec<usize> = acc_cols.clone();
        let mut vs: Vec<f64> = acc_vals.clone();
        let rf = RowFilter {
            tau_abs: drop_tol,
            tau_rel: trunc_rel,
            k_max: max_per_row,
            must_keep: Some(myc),
        };
        filter_row_by_truncation(&mut cols, &mut vs, rf);
        if cols.is_empty() {
            cols.push(myc);
            vs.push(1.0);
        }
        for (c, v) in cols.into_iter().zip(vs.into_iter()) {
            col_idx.push(c);
            vals.push(v);
        }
        row_ptr.push(col_idx.len());

        // reset markers used
        for &c in &acc_cols {
            marker[c] = -1;
        }
    }

    Pcsr {
        m,
        n: ncoarse,
        row_ptr,
        col_idx,
        vals,
    }
}

/// Values-only refresh for P using fixed pattern in `p`.
pub fn smooth_sa_values_only<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    tp: &TentativeP,
    omega: f64,
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    out_vals: &mut [T],
) -> Result<(), crate::error::KError> {
    let m = a.nrows();
    assert_eq!(p_row_ptr.len(), m + 1);
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    let pr = p_row_ptr;
    let pc = p_col_idx;

    // row-local accumulator of values by coarse column
    let mut map_cols: Vec<usize> = Vec::new();
    let mut map_vals: Vec<f64> = Vec::new();

    for i in 0..m {
        map_cols.clear();
        map_vals.clear();
        // Start with 1 at own aggregate
        let myc = tp.agg_of[i];
        map_cols.push(myc);
        map_vals.push(1.0);
        // Add neighbors contributions
        let di = d_inv[i];
        let rs = rp[i];
        let re = rp[i + 1];
        for pidx in rs..re {
            let j = cj[pidx];
            if j == i {
                continue;
            }
            let cjg = tp.agg_of[j];
            let val = -omega * di * vv[pidx];
            // find or insert
            match map_cols.iter().position(|&c| c == cjg) {
                Some(pos) => {
                    map_vals[pos] += val;
                }
                None => {
                    map_cols.push(cjg);
                    map_vals.push(val);
                }
            }
        }
        // Scatter into existing pattern
        let rs_p = pr[i];
        let re_p = pr[i + 1];
        for k in rs_p..re_p {
            let c = pc[k];
            // find in map
            if let Some(pos) = map_cols.iter().position(|&cc| cc == c) {
                out_vals[k] = T::from_real(map_vals[pos]);
            } else {
                out_vals[k] = T::zero();
            }
        }
    }
    Ok(())
}

pub fn smooth_tentative_sa_multi(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    tp: &TentativeP,
    omega: f64,
    drop_tol: f64,
    max_per_row: usize,
    trunc_rel: f64,
) -> Pcsr {
    let m = a.nrows();
    let r = tp.num_functions;
    let ncoarse = tp.n_coarse * r;
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);
    let mut marker: Vec<isize> = vec![-1; ncoarse.min(1024).max(512)];
    let mut acc_cols: Vec<usize> = Vec::new();
    let mut acc_vals: Vec<f64> = Vec::new();
    for i in 0..m {
        if marker.len() < ncoarse {
            marker.resize(ncoarse, -1);
        }
        acc_cols.clear();
        acc_vals.clear();
        let base = tp.agg_of[i] * r;
        for alpha in 0..r {
            let c = base + alpha;
            marker[c] = acc_cols.len() as isize;
            acc_cols.push(c);
            let v0 = if let Some(ref nns) = tp.nns {
                nns[alpha][i]
            } else if let Some(ref comp) = tp.comp_of {
                if comp[i] == alpha { 1.0 } else { 0.0 }
            } else if alpha == 0 {
                1.0
            } else {
                0.0
            };
            acc_vals.push(v0);
        }
        let di = d_inv[i];
        let rs = rp[i];
        let re = rp[i + 1];
        for p in rs..re {
            let j = cj[p];
            if j == i {
                continue;
            }
            let gj = tp.agg_of[j] * r;
            let s = -omega * di * vv[p];
            for alpha in 0..r {
                let col = gj + alpha;
                let t = if let Some(ref nns) = tp.nns {
                    nns[alpha][j]
                } else if let Some(ref comp) = tp.comp_of {
                    if comp[j] == alpha { 1.0 } else { 0.0 }
                } else if alpha == 0 {
                    1.0
                } else {
                    0.0
                };
                let val = s * t;
                let k = marker[col];
                if k >= 0 {
                    acc_vals[k as usize] += val;
                } else {
                    marker[col] = acc_cols.len() as isize;
                    acc_cols.push(col);
                    acc_vals.push(val);
                }
            }
        }
        let mut cols = acc_cols.clone();
        let mut vs = acc_vals.clone();
        let rf = RowFilter {
            tau_abs: drop_tol,
            tau_rel: trunc_rel,
            k_max: max_per_row,
            must_keep: None,
        };
        filter_row_by_truncation(&mut cols, &mut vs, rf);
        for alpha in 0..r {
            let c = base + alpha;
            if !cols.contains(&c) {
                let v0 = if let Some(ref nns) = tp.nns {
                    nns[alpha][i]
                } else if let Some(ref comp) = tp.comp_of {
                    if comp[i] == alpha { 1.0 } else { 0.0 }
                } else if alpha == 0 {
                    1.0
                } else {
                    0.0
                };
                cols.push(c);
                vs.push(v0);
            }
        }
        for (c, v) in cols.into_iter().zip(vs.into_iter()) {
            col_idx.push(c);
            vals.push(v);
        }
        row_ptr.push(col_idx.len());
        for &c in &acc_cols {
            marker[c] = -1;
        }
    }
    Pcsr {
        m,
        n: ncoarse,
        row_ptr,
        col_idx,
        vals,
    }
}

pub fn smooth_tentative_sa_mf(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    tn: &TentativeNodal,
    omega: f64,
    drop_tol: f64,
    max_per_row: usize,
) -> Pcsr {
    let n = a.nrows();
    let mfun = tn.mfun;
    let ncoarse = tn.n_agg * mfun;
    debug_assert_eq!(tn.agg_of.len(), n);
    debug_assert_eq!(tn.row_basis.len(), n * mfun);

    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut marker: Vec<isize> = vec![-1; ncoarse.min(4096).max(512)];
    let mut acc_cols: Vec<usize> = Vec::new();
    let mut acc_vals: Vec<f64> = Vec::new();

    for i in 0..n {
        if marker.len() < ncoarse {
            marker.resize(ncoarse, -1);
        }
        acc_cols.clear();
        acc_vals.clear();

        let gi = tn.agg_of[i];
        let di = d_inv[i];

        for f in 0..mfun {
            let c = gi * mfun + f;
            marker[c] = acc_cols.len() as isize;
            acc_cols.push(c);
            acc_vals.push(tn.row_basis[i * mfun + f]);
        }

        let rs = rp[i];
        let re = rp[i + 1];
        for p in rs..re {
            let j = cj[p];
            if j == i {
                continue;
            }
            let gj = tn.agg_of[j];
            let scale = -omega * di * vv[p];
            for f in 0..mfun {
                let c = gj * mfun + f;
                let contrib = scale * tn.row_basis[j * mfun + f];
                let k = marker[c];
                if k >= 0 {
                    acc_vals[k as usize] += contrib;
                } else {
                    marker[c] = acc_cols.len() as isize;
                    acc_cols.push(c);
                    acc_vals.push(contrib);
                }
            }
        }

        let mut keep: Vec<(usize, f64)> = Vec::with_capacity(acc_cols.len());
        for (&c, &v) in acc_cols.iter().zip(acc_vals.iter()) {
            if v.abs() >= drop_tol {
                keep.push((c, v));
            }
        }

        if max_per_row > 0 && keep.len() > max_per_row {
            keep.select_nth_unstable_by(max_per_row, |a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            keep.truncate(max_per_row);
        }

        keep.sort_by(|a, b| a.0.cmp(&b.0));
        for (c, v) in keep {
            col_idx.push(c);
            vals.push(v);
        }
        row_ptr.push(col_idx.len());

        for &c in &acc_cols {
            marker[c] = -1;
        }
    }

    Pcsr {
        m: n,
        n: ncoarse,
        row_ptr,
        col_idx,
        vals,
    }
}

pub fn smooth_sa_values_only_multi<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    tp: &TentativeP,
    omega: f64,
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    out_vals: &mut [T],
) -> Result<(), crate::error::KError> {
    let m = a.nrows();
    let r = tp.num_functions;
    assert_eq!(p_row_ptr.len(), m + 1);
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    let pr = p_row_ptr;
    let pc = p_col_idx;
    let mut map_cols: Vec<usize> = Vec::new();
    let mut map_vals: Vec<f64> = Vec::new();
    for i in 0..m {
        map_cols.clear();
        map_vals.clear();
        let base = tp.agg_of[i] * r;
        for alpha in 0..r {
            map_cols.push(base + alpha);
            let v0 = if let Some(ref nns) = tp.nns {
                nns[alpha][i]
            } else if let Some(ref comp) = tp.comp_of {
                if comp[i] == alpha { 1.0 } else { 0.0 }
            } else if alpha == 0 {
                1.0
            } else {
                0.0
            };
            map_vals.push(v0);
        }
        let di = d_inv[i];
        let rs = rp[i];
        let re = rp[i + 1];
        for pidx in rs..re {
            let j = cj[pidx];
            if j == i {
                continue;
            }
            let gj = tp.agg_of[j] * r;
            let s = -omega * di * vv[pidx];
            for alpha in 0..r {
                let t = if let Some(ref nns) = tp.nns {
                    nns[alpha][j]
                } else if let Some(ref comp) = tp.comp_of {
                    if comp[j] == alpha { 1.0 } else { 0.0 }
                } else if alpha == 0 {
                    1.0
                } else {
                    0.0
                };
                let col = gj + alpha;
                match map_cols.iter().position(|&c| c == col) {
                    Some(pos) => {
                        map_vals[pos] += s * t;
                    }
                    None => {
                        map_cols.push(col);
                        map_vals.push(s * t);
                    }
                }
            }
        }
        let rs_p = pr[i];
        let re_p = pr[i + 1];
        for k in rs_p..re_p {
            let c = pc[k];
            if let Some(pos) = map_cols.iter().position(|&cc| cc == c) {
                out_vals[k] = T::from_real(map_vals[pos]);
            } else {
                out_vals[k] = T::zero();
            }
        }
    }
    Ok(())
}

pub fn smooth_sa_values_only_mf<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    tn: &TentativeNodal,
    omega: f64,
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    out_vals: &mut [T],
) -> Result<(), KError> {
    let n = a.nrows();
    let mfun = tn.mfun;
    if tn.agg_of.len() != n {
        return Err(KError::InvalidInput(
            "smooth_sa_values_only_mf: agg_of length mismatch".into(),
        ));
    }
    if tn.row_basis.len() != n * mfun {
        return Err(KError::InvalidInput(
            "smooth_sa_values_only_mf: row_basis length mismatch".into(),
        ));
    }
    if d_inv.len() != n {
        return Err(KError::InvalidInput(
            "smooth_sa_values_only_mf: d_inv length mismatch".into(),
        ));
    }
    if p_row_ptr.len() != n + 1 {
        return Err(KError::InvalidInput(
            "smooth_sa_values_only_mf: row_ptr length mismatch".into(),
        ));
    }
    if p_col_idx.len() != out_vals.len() {
        return Err(KError::InvalidInput(
            "smooth_sa_values_only_mf: values length mismatch".into(),
        ));
    }

    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    for i in 0..n {
        let gi = tn.agg_of[i];
        let di = d_inv[i];
        let rs = rp[i];
        let re = rp[i + 1];
        let prs = p_row_ptr[i];
        let pre = p_row_ptr[i + 1];
        for k in prs..pre {
            let c = p_col_idx[k];
            let g_col = c / mfun;
            let f_col = c % mfun;
            let mut value = if g_col == gi {
                tn.row_basis[i * mfun + f_col]
            } else {
                0.0
            };
            let mut sum = 0.0;
            for p in rs..re {
                let j = cj[p];
                if j != i && tn.agg_of[j] == g_col {
                    sum += vv[p] * tn.row_basis[j * mfun + f_col];
                }
            }
            value += -omega * di * sum;
            out_vals[k] = T::from_real(value);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::CsrMatrix;

    fn diag_inv(a: &CsrMatrix<f64>) -> Vec<f64> {
        let n = a.nrows();
        let mut out = vec![0.0; n];
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        for i in 0..n {
            let mut diag = 0.0;
            for p in rp[i]..rp[i + 1] {
                if cj[p] == i {
                    diag = vv[p];
                    break;
                }
            }
            out[i] = if diag.abs() > 0.0 { 1.0 / diag } else { 0.0 };
        }
        out
    }

    fn hetero_poisson_1d(n: usize) -> CsrMatrix<f64> {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        row_ptr.push(0);
        for i in 0..n {
            if i > 0 {
                col_idx.push(i - 1);
                vals.push(-1.0);
            }
            col_idx.push(i);
            let diag = if i % 2 == 0 { 4.0 } else { 8.0 };
            vals.push(diag);
            if i + 1 < n {
                col_idx.push(i + 1);
                vals.push(-1.5);
            }
            row_ptr.push(col_idx.len());
        }
        CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
    }

    fn row_fit_residuals(
        row_ptr: &[usize],
        col_idx: &[usize],
        vals: &[f64],
        u_f: &Mat<f64>,
        u_c: &Mat<f64>,
    ) -> Vec<f64> {
        let n = u_f.nrows();
        let r = u_f.ncols();
        let mut out = vec![0.0; n];
        for i in 0..n {
            let rs = row_ptr[i];
            let re = row_ptr[i + 1];
            if rs == re || r == 0 {
                continue;
            }
            let mut approx = vec![0.0f64; r];
            for k in rs..re {
                let w = vals[k];
                let c = col_idx[k];
                for alpha in 0..r {
                    approx[alpha] += w * u_c[(c, alpha)];
                }
            }
            let mut res = 0.0;
            for alpha in 0..r {
                let diff = approx[alpha] - u_f[(i, alpha)];
                res += diff * diff;
            }
            out[i] = res.sqrt();
        }
        out
    }

    #[test]
    fn own_aggregate_kept() {
        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 0.5, 0.5, 1.0],
        );
        let tp = TentativeP {
            agg_of: vec![0, 1],
            n_coarse: 2,
            num_functions: 1,
            nns: None,
            comp_of: None,
        };
        let d_inv = vec![1.0, 1.0];
        let p = smooth_tentative_sa_multi(&a, &d_inv, &tp, 1.0, 10.0, 0, 0.0);
        assert_eq!(p.col_idx, vec![0, 1]);
    }

    #[test]
    fn drop_tol_prunes_but_keeps_self() {
        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 0.5, 0.5, 1.0],
        );
        let tp = TentativeP {
            agg_of: vec![0, 1],
            n_coarse: 2,
            num_functions: 1,
            nns: None,
            comp_of: None,
        };
        let d_inv = vec![1.0, 1.0];
        // drop_tol=0 -> keep all
        let p_full = smooth_tentative_sa_multi(&a, &d_inv, &tp, 1.0, 0.0, 0, 0.0);
        assert_eq!(p_full.col_idx, vec![0, 1, 0, 1]);
        // drop_tol large -> only own aggregates
        let p_drop = smooth_tentative_sa_multi(&a, &d_inv, &tp, 1.0, 1.0, 0, 0.0);
        assert_eq!(p_drop.col_idx, vec![0, 1]);
    }

    #[test]
    fn adaptive_fit_reduces_residual() {
        let a = hetero_poisson_1d(8);
        let tp = TentativeP {
            agg_of: (0..8).map(|i| i / 2).collect(),
            n_coarse: 4,
            num_functions: 1,
            nns: None,
            comp_of: None,
        };
        let d_inv = diag_inv(&a);
        let p_csr = smooth_tentative_sa_multi(&a, &d_inv, &tp, 2.0 / 3.0, 0.0, 0, 0.0);
        let u_f = sample_low_modes(&a, &d_inv, 4, 3, 2.0 / 3.0, 0xCAFE).unwrap();
        let u_c = restrict_samples_to_coarse(&a, &tp, &u_f, AdaptiveWeight::Diag);
        let baseline = p_csr.vals.clone();
        let mut adapted = baseline.clone();
        adaptive_fit_values_only(
            &p_csr.row_ptr,
            &p_csr.col_idx,
            &mut adapted,
            &tp,
            &u_f,
            &u_c,
            1e-10,
            true,
            0.0,
        )
        .unwrap();
        let baseline_res = row_fit_residuals(&p_csr.row_ptr, &p_csr.col_idx, &baseline, &u_f, &u_c);
        let adapted_res = row_fit_residuals(&p_csr.row_ptr, &p_csr.col_idx, &adapted, &u_f, &u_c);
        let avg_baseline: f64 = baseline_res.iter().sum::<f64>() / (baseline_res.len() as f64);
        let avg_adapted: f64 = adapted_res.iter().sum::<f64>() / (adapted_res.len() as f64);
        assert!(avg_adapted < avg_baseline * 0.9);
    }

    #[test]
    fn adaptive_fit_preserves_sum_and_is_deterministic() {
        let a = hetero_poisson_1d(6);
        let tp = TentativeP {
            agg_of: (0..6).map(|i| i / 2).collect(),
            n_coarse: 3,
            num_functions: 1,
            nns: None,
            comp_of: None,
        };
        let d_inv = diag_inv(&a);
        let p_csr = smooth_tentative_sa_multi(&a, &d_inv, &tp, 2.0 / 3.0, 0.0, 0, 0.0);
        let u_f = sample_low_modes(&a, &d_inv, 3, 2, 2.0 / 3.0, 0x1234).unwrap();
        let u_c = restrict_samples_to_coarse(&a, &tp, &u_f, AdaptiveWeight::Diag);
        let mut first = p_csr.vals.clone();
        adaptive_fit_values_only(
            &p_csr.row_ptr,
            &p_csr.col_idx,
            &mut first,
            &tp,
            &u_f,
            &u_c,
            1e-10,
            true,
            0.0,
        )
        .unwrap();
        let mut second = p_csr.vals.clone();
        adaptive_fit_values_only(
            &p_csr.row_ptr,
            &p_csr.col_idx,
            &mut second,
            &tp,
            &u_f,
            &u_c,
            1e-10,
            true,
            0.0,
        )
        .unwrap();
        assert_eq!(first.len(), second.len());
        for i in 0..first.len() {
            assert!((first[i] - second[i]).abs() < 1e-12);
        }
        for row in 0..tp.agg_of.len() {
            let rs = p_csr.row_ptr[row];
            let re = p_csr.row_ptr[row + 1];
            if rs == re {
                continue;
            }
            let sum: f64 = first[rs..re].iter().sum();
            assert!((sum - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn smooth_tentative_sa_mf_preserves_row_basis_when_omega_zero() {
        let a = CsrMatrix::from_csr(
            4,
            4,
            vec![0, 1, 2, 3, 4],
            vec![0, 1, 2, 3],
            vec![2.0, 3.0, 4.0, 5.0],
        );
        let tn = TentativeNodal {
            agg_of: vec![0, 0, 1, 1],
            n_agg: 2,
            mfun: 2,
            row_basis: vec![
                1.0, 0.0, // row 0
                0.0, 1.0, // row 1
                1.0, 0.0, // row 2
                0.0, 1.0, // row 3
            ],
        };
        let d_inv = vec![1.0; 4];
        let p = smooth_tentative_sa_mf(&a, &d_inv, &tn, 0.0, 0.0, 0);
        assert_eq!(p.m, 4);
        assert_eq!(p.n, 4);
        assert_eq!(p.row_ptr, vec![0, 2, 4, 6, 8]);
        assert_eq!(p.col_idx, vec![0, 1, 0, 1, 2, 3, 2, 3]);
        assert_eq!(p.vals, vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn smooth_sa_values_only_mf_matches_builder() {
        let a = CsrMatrix::from_csr(
            4,
            4,
            vec![0, 3, 6, 9, 12],
            vec![0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3],
            vec![
                4.0, -1.0, 0.5, -1.0, 4.0, -0.5, 0.5, -1.0, 4.0, 0.5, -0.5, 4.0,
            ],
        );
        let tn = TentativeNodal {
            agg_of: vec![0, 0, 1, 1],
            n_agg: 2,
            mfun: 2,
            row_basis: vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        };
        let d_inv = vec![1.0 / 4.0; 4];
        let p = smooth_tentative_sa_mf(&a, &d_inv, &tn, 0.8, 0.0, 0);
        let mut refreshed = vec![0.0; p.vals.len()];
        smooth_sa_values_only_mf(&a, &d_inv, &tn, 0.8, &p.row_ptr, &p.col_idx, &mut refreshed)
            .unwrap();
        assert_eq!(refreshed.len(), p.vals.len());
        for (a, b) in refreshed.iter().zip(p.vals.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn values_only_refresh_accepts_complex_output_storage() {
        let a = CsrMatrix::from_csr(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 4.0, -0.5, -0.5, 3.0],
        );
        let tp = TentativeP {
            agg_of: vec![0, 0, 1],
            n_coarse: 2,
            num_functions: 1,
            nns: None,
            comp_of: None,
        };
        let d_inv = diag_inv(&a);
        let p = smooth_tentative_sa(&a, &d_inv, &tp, 0.75, 0.0, 0, 0.0);

        let mut real_vals = vec![0.0; p.vals.len()];
        smooth_sa_values_only(
            &a,
            &d_inv,
            &tp,
            0.75,
            &p.row_ptr,
            &p.col_idx,
            &mut real_vals,
        )
        .unwrap();
        let mut complex_vals = vec![S::zero(); p.vals.len()];
        smooth_sa_values_only(
            &a,
            &d_inv,
            &tp,
            0.75,
            &p.row_ptr,
            &p.col_idx,
            &mut complex_vals,
        )
        .unwrap();

        for (got, expected) in complex_vals.iter().zip(real_vals.iter()) {
            assert!((got.real() - expected).abs() < 1e-12);
            assert!(got.imag().abs() < 1e-12);
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn adaptive_fit_accepts_complex_output_storage() {
        let a = hetero_poisson_1d(6);
        let tp = TentativeP {
            agg_of: (0..6).map(|i| i / 2).collect(),
            n_coarse: 3,
            num_functions: 1,
            nns: None,
            comp_of: None,
        };
        let d_inv = diag_inv(&a);
        let p_csr = smooth_tentative_sa_multi(&a, &d_inv, &tp, 2.0 / 3.0, 0.0, 0, 0.0);
        let u_f = sample_low_modes(&a, &d_inv, 3, 2, 2.0 / 3.0, 0x4567).unwrap();
        let u_c = restrict_samples_to_coarse(&a, &tp, &u_f, AdaptiveWeight::Diag);

        let mut real_vals = p_csr.vals.clone();
        adaptive_fit_values_only(
            &p_csr.row_ptr,
            &p_csr.col_idx,
            &mut real_vals,
            &tp,
            &u_f,
            &u_c,
            1e-10,
            true,
            0.0,
        )
        .unwrap();

        let mut complex_vals: Vec<S> = p_csr.vals.iter().map(|&v| S::from_real(v)).collect();
        adaptive_fit_values_only(
            &p_csr.row_ptr,
            &p_csr.col_idx,
            &mut complex_vals,
            &tp,
            &u_f,
            &u_c,
            1e-10,
            true,
            0.0,
        )
        .unwrap();

        for (got, expected) in complex_vals.iter().zip(real_vals.iter()) {
            assert!((got.real() - expected).abs() < 1e-12);
            assert!(got.imag().abs() < 1e-12);
        }
    }
}
