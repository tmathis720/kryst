#![allow(dead_code)]

use crate::matrix::sparse::CsrMatrix;
use super::row_filter::{RowFilter, filter_row_by_truncation};
use super::strength::Strength;

#[derive(Clone, Debug)]
pub struct TentativeP {
    pub agg_of: Vec<usize>,
    pub n_coarse: usize,
}

pub fn tentative_from_aggregates(agg: Vec<usize>) -> TentativeP {
    let n_coarse = 1 + agg.iter().copied().max().unwrap_or(0);
    TentativeP { agg_of: agg, n_coarse }
}

#[derive(Clone, Debug)]
pub struct Pcsr {
    pub m: usize,
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<f64>,
}

// ===== Classical interpolation family =======================================

/// Variants of classical interpolation.
#[derive(Clone, Copy, Debug)]
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
    CFInfo { is_c: is_c.to_vec(), coarse_of }
}

/// Helper: get entry a[i,j] via binary search in row i.
fn csr_get(a: &CsrMatrix<f64>, i: usize, j: usize) -> Option<f64> {
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    let rs = rp[i];
    let re = rp[i + 1];
    cj[rs..re]
        .binary_search(&j)
        .ok()
        .map(|p| vv[rs + p])
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
    let ncoarse = cf.coarse_of.iter().filter_map(|&x| x).max().map(|x| x + 1).unwrap_or(0);

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
                if cf.is_c[j] { continue; }
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
                    if v > bestmag { bestmag = v; best = Some(k); }
                }
            }
            if let Some(k) = best { cols.push(k); }
            else if ncoarse > 0 { cols.push(0); }
        }

        for c in cols {
            col_idx.push(c);
            vals.push(0.0);
        }
        row_ptr.push(col_idx.len());
    }

    (Pcsr { m: n, n: ncoarse, row_ptr, col_idx, vals }, cf)
}

/// For F-neighbor j, distribute influence over its strong C neighbors.
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
                if v < 0.0 { sum_neg += -v; } else { sum_pos += v; }
            }
        }
    }
    if tmp.is_empty() { return; }
    for (c, v) in tmp {
        let w = if v < 0.0 { if sum_neg > 0.0 { (-v)/sum_neg } else { 0.0 } } else { if sum_pos > 0.0 { v/sum_pos } else { 0.0 } };
        cols.push(c);
        wts.push(w);
    }
    let s: f64 = wts.iter().sum();
    if s > 0.0 { for w in wts.iter_mut() { *w /= s; } }
    else {
        let u = 1.0 / (wts.len() as f64);
        for w in wts.iter_mut() { *w = u; }
    }
}

/// Values-only refresh for classical interpolation.
pub fn classical_values_only(
    a: &CsrMatrix<f64>,
    s_sym: &Strength,
    cf: &CFInfo,
    params: &ClassicalParams,
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    out_vals: &mut [f64],
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
            for k in rs_p..re_p { out_vals[k] = 0.0; }
            if let Some(kc) = cf.coarse_of[i] {
                for k in rs_p..re_p { if p_col_idx[k] == kc { out_vals[k] = 1.0; break; } }
            }
            continue;
        }

        let mut contrib_cols: Vec<usize> = Vec::new();
        let mut contrib_vals: Vec<f64> = Vec::new();

        let rs = s_sym.row_ptr[i];
        let re = s_sym.row_ptr[i + 1];

        // Direct part
        let mut sum_neg = 0.0; let mut sum_pos = 0.0;
        for &j in &s_sym.col_idx[rs..re] {
            if cf.is_c[j] {
                if let Some(aij) = csr_get(a, i, j) {
                    if params.variant == ClassicalVariant::Direct {
                        if aij < 0.0 { sum_neg += -aij; } else { sum_pos += aij; }
                    } else {
                        let col = cf.coarse_of[j].unwrap();
                        contrib_cols.push(col);
                        contrib_vals.push(-aij);
                    }
                }
            }
        }
        if params.variant == ClassicalVariant::Direct {
            for &j in &s_sym.col_idx[rs..re] {
                if cf.is_c[j] {
                    if let Some(aij) = csr_get(a, i, j) {
                        let col = cf.coarse_of[j].unwrap();
                        let w = if aij < 0.0 {
                            if sum_neg > 0.0 { (-aij)/sum_neg } else { 0.0 }
                        } else {
                            if sum_pos > 0.0 { aij/sum_pos } else { 0.0 }
                        };
                        contrib_cols.push(col);
                        contrib_vals.push(w);
                    }
                }
            }
        }

        if matches!(params.variant, ClassicalVariant::Standard | ClassicalVariant::HE) {
            for &j in &s_sym.col_idx[rs..re] {
                if cf.is_c[j] { continue; }
                let aij = match csr_get(a, i, j) { Some(v) => v, None => continue };
                if aij == 0.0 { continue; }
                neighbor_distribution_over_C_of(j, a, s_sym, cf, &mut buf_cols, &mut buf_wts);
                let scale = if matches!(params.variant, ClassicalVariant::HE) {
                    let mut rowsum = 0.0; let mut ajj = 0.0;
                    let rj = rp[j]; let ej = rp[j + 1];
                    for p in rj..ej {
                        let v = vv[p];
                        if cj[p] == j { ajj = v.abs(); } else { rowsum += v.abs(); }
                    }
                    let denom = ajj.max(rowsum).max(1e-30);
                    (-aij) / denom
                } else { -aij };
                for t in 0..buf_cols.len() {
                    contrib_cols.push(buf_cols[t]);
                    contrib_vals.push(scale * buf_wts[t]);
                }
            }

            // denominator
            let mut sum_neg_strong = 0.0;
            for &j in &s_sym.col_idx[rs..re] {
                if let Some(aij) = csr_get(a, i, j) { if aij < 0.0 { sum_neg_strong += -aij; } }
            }
            let mut di = csr_get(a, i, i).unwrap_or(1.0);
            let di_eff = di - sum_neg_strong;
            let denom = if di_eff.abs() >= 1e-14 * di.abs().max(1.0) { di_eff } else { di };
            if denom.abs() < 1e-30 {
                let mut s = 0.0; for v in &contrib_vals { s += v.abs(); }
                if s > 0.0 { for v in &mut contrib_vals { *v /= s; } }
            } else {
                for v in &mut contrib_vals { *v /= denom; }
            }
        }

        if !contrib_cols.is_empty() {
            let mut idx: Vec<usize> = (0..contrib_cols.len()).collect();
            idx.sort_unstable_by(|&u,&v| contrib_cols[u].cmp(&contrib_cols[v]));
            let mut last = contrib_cols[idx[0]];
            let mut acc = 0.0;
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            for &id in &idx {
                let c = contrib_cols[id];
                if c == last { acc += contrib_vals[id]; }
                else { if acc != 0.0 { cols.push(last); vals.push(acc); } last = c; acc = contrib_vals[id]; }
            }
            if acc != 0.0 { cols.push(last); vals.push(acc); }

            let kept_cols = cols.clone();
            let kept_vals = vals.clone();
            let mut rf = RowFilter { tau_abs: params.drop_abs, tau_rel: params.trunc_rel, k_max: params.cap_row, must_keep: None };
            filter_row_by_truncation(&mut cols, &mut vals, rf);
            if params.keep_at_least_one && cols.is_empty() && !kept_cols.is_empty() {
                let mut best = 0usize; let mut bestmag = kept_vals[0].abs();
                for t in 1..kept_cols.len() { let m = kept_vals[t].abs(); if m > bestmag { bestmag = m; best = t; } }
                cols.push(kept_cols[best]);
                vals.push(kept_vals[best]);
            }
            for k in rs_p..re_p {
                let c = p_col_idx[k];
                match cols.binary_search(&c) {
                    Ok(pos) => out_vals[k] = vals[pos],
                    Err(_) => out_vals[k] = 0.0,
                }
            }
        } else {
            for k in rs_p..re_p { out_vals[k] = 0.0; }
            if rs_p < re_p { out_vals[rs_p] = 1.0; }
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
        if marker.len() < ncoarse { marker.resize(ncoarse, -1); }
        acc_cols.clear(); acc_vals.clear();

        // Start with 1.0 at own aggregate
        let myc = tp.agg_of[i];
        marker[myc] = 0; acc_cols.push(myc); acc_vals.push(1.0);

        // Accumulate -ω d_i a_ij into coarse columns of neighbors' aggregates
        let di = d_inv[i];
        let rs = rp[i]; let re = rp[i + 1];
        for p in rs..re {
            let j = cj[p]; if j == i { continue; }
            let cjg = tp.agg_of[j];
            let v = -omega * di * vv[p];
            let k = marker[cjg];
            if k >= 0 {
                acc_vals[k as usize] += v;
            } else {
                marker[cjg] = acc_cols.len() as isize;
                acc_cols.push(cjg); acc_vals.push(v);
            }
        }

        let mut cols: Vec<usize> = acc_cols.clone();
        let mut vs: Vec<f64> = acc_vals.clone();
        let rf = RowFilter { tau_abs: drop_tol, tau_rel: trunc_rel, k_max: max_per_row, must_keep: Some(myc) };
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
        for &c in &acc_cols { marker[c] = -1; }
    }

    Pcsr { m, n: ncoarse, row_ptr, col_idx, vals }
}

/// Values-only refresh for P using fixed pattern in `p`.
pub fn smooth_sa_values_only(
    a: &CsrMatrix<f64>,
    d_inv: &[f64],
    tp: &TentativeP,
    omega: f64,
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    out_vals: &mut [f64],
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
        map_cols.clear(); map_vals.clear();
        // Start with 1 at own aggregate
        let myc = tp.agg_of[i];
        map_cols.push(myc); map_vals.push(1.0);
        // Add neighbors contributions
        let di = d_inv[i];
        let rs = rp[i]; let re = rp[i + 1];
        for pidx in rs..re {
            let j = cj[pidx]; if j == i { continue; }
            let cjg = tp.agg_of[j];
            let val = -omega * di * vv[pidx];
            // find or insert
            match map_cols.iter().position(|&c| c == cjg) {
                Some(pos) => { map_vals[pos] += val; }
                None => { map_cols.push(cjg); map_vals.push(val); }
            }
        }
        // Scatter into existing pattern
        let rs_p = pr[i]; let re_p = pr[i + 1];
        for k in rs_p..re_p {
            let c = pc[k];
            // find in map
            if let Some(pos) = map_cols.iter().position(|&cc| cc == c) {
                out_vals[k] = map_vals[pos];
            } else {
                out_vals[k] = 0.0;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::CsrMatrix;

    #[test]
    fn own_aggregate_kept() {
        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 0.5, 0.5, 1.0],
        );
        let tp = TentativeP { agg_of: vec![0, 1], n_coarse: 2 };
        let d_inv = vec![1.0, 1.0];
        let p = smooth_tentative_sa(&a, &d_inv, &tp, 1.0, 10.0, 0, 0.0);
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
        let tp = TentativeP { agg_of: vec![0, 1], n_coarse: 2 };
        let d_inv = vec![1.0, 1.0];
        // drop_tol=0 -> keep all
        let p_full = smooth_tentative_sa(&a, &d_inv, &tp, 1.0, 0.0, 0, 0.0);
        assert_eq!(p_full.col_idx, vec![0, 1, 0, 1]);
        // drop_tol large -> only own aggregates
        let p_drop = smooth_tentative_sa(&a, &d_inv, &tp, 1.0, 1.0, 0, 0.0);
        assert_eq!(p_drop.col_idx, vec![0, 1]);
    }
}
