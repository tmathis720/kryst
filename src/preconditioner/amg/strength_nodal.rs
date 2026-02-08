use std::collections::BTreeMap;

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;

use super::strength::Strength;
use super::util::DofLayout;

#[derive(Clone, Debug)]
pub struct NodalStrength {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
}

pub fn strength_nodal_from_csr<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<T>,
    block_size: usize,
    theta: f64,
    normalize: bool,
) -> NodalStrength {
    assert!(block_size >= 1, "block_size must be positive");
    assert_eq!(a.nrows() % block_size, 0, "block_size must divide nrows");
    let n_nodes = a.nrows() / block_size;
    let mut diag_sq = vec![0.0; n_nodes];
    let mut rows: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n_nodes];

    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();
    for row in 0..a.nrows() {
        let u = row / block_size;
        let rs = rp[row];
        let re = rp[row + 1];
        for p in rs..re {
            let j = cj[p];
            let w = j / block_size;
            let val_sq = vv[p].abs2();
            if u == w {
                diag_sq[u] += val_sq;
            } else {
                *rows[u].entry(w).or_insert(0.0) += val_sq;
            }
        }
    }

    let diag_norm: Vec<f64> = diag_sq.iter().map(|&s| s.sqrt()).collect();

    let mut row_ptr = Vec::with_capacity(n_nodes + 1);
    let mut col_idx = Vec::<usize>::new();
    row_ptr.push(0);
    for u in 0..n_nodes {
        let mut max_off = 0.0;
        if !normalize {
            for &sq in rows[u].values() {
                let norm = sq.sqrt();
                if norm > max_off {
                    max_off = norm;
                }
            }
        }

        let start_len = col_idx.len();
        for (&w, &sq) in rows[u].iter() {
            let norm = sq.sqrt();
            let keep = if normalize {
                let denom = diag_norm[u] * diag_norm[w];
                denom > 0.0 && norm / denom >= theta
            } else {
                max_off > 0.0 && norm >= theta * max_off
            };
            if keep {
                col_idx.push(w);
            }
        }
        row_ptr.push(col_idx.len());
        debug_assert!(
            row_ptr.last().copied().unwrap() - row_ptr[row_ptr.len() - 2]
                == col_idx.len() - start_len
        );
    }

    NodalStrength { row_ptr, col_idx }
}

pub fn strength_nodal<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<T>,
    layout: &DofLayout,
    theta: f64,
    normalize: bool,
) -> Strength {
    let nodal = strength_nodal_from_csr(a, layout.block_size, theta, normalize);
    Strength {
        row_ptr: nodal.row_ptr,
        col_idx: nodal.col_idx,
    }
}
