use crate::matrix::sparse::CsrMatrix;

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

        // Drop small and cap per row
        let mut keep: Vec<(usize, f64)> = acc_cols.iter().zip(acc_vals.iter())
            .filter_map(|(&c, &v)| if v.abs() >= drop_tol { Some((c, v)) } else { None })
            .collect();
        if max_per_row > 0 && keep.len() > max_per_row {
            keep.select_nth_unstable_by(max_per_row, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            keep.truncate(max_per_row);
        }
        keep.sort_by_key(|(c, _)| *c);

        for (c, v) in keep {
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
