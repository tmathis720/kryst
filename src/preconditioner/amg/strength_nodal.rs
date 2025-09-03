use std::collections::BTreeMap;

use crate::matrix::sparse::CsrMatrix;

use super::strength::Strength;
use super::util::DofLayout;

pub fn strength_nodal(
    a: &CsrMatrix<f64>,
    layout: &DofLayout,
    theta: f64,
    normalize: bool,
) -> Strength {
    let n = layout.n_nodes;
    let mut diag = vec![0.0; n];
    let mut rows: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];

    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    for i_dof in 0..a.nrows() {
        let u = layout.node_of[i_dof];
        let rs = rp[i_dof];
        let re = rp[i_dof + 1];
        for p in rs..re {
            let j_dof = cj[p];
            let v = vv[p].abs();
            let w = layout.node_of[j_dof];
            if u == w {
                if v > diag[u] {
                    diag[u] = v;
                }
            } else {
                let e = rows[u].entry(w).or_insert(0.0);
                if v > *e {
                    *e = v;
                }
            }
        }
    }

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::<usize>::new();
    row_ptr.push(0);
    for u in 0..n {
        let mut count = 0usize;
        let mut max_off = 0.0;
        if !normalize {
            for &blk in rows[u].values() {
                if blk > max_off {
                    max_off = blk;
                }
            }
        }
        for (&w, &blk) in rows[u].iter() {
            let keep = if normalize {
                let denom = (diag[u] * diag[w]).sqrt();
                denom > 0.0 && blk / denom >= theta
            } else {
                max_off > 0.0 && blk >= theta * max_off
            };
            if keep {
                col_idx.push(w);
                count += 1;
            }
        }
        row_ptr.push(row_ptr.last().unwrap() + count);
    }
    Strength { row_ptr, col_idx }
}

