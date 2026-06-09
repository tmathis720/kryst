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
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    let build_node_row = |u: usize| {
        let mut diag_sq = 0.0;
        let mut row_weights = BTreeMap::new();
        for row in u * block_size..(u + 1) * block_size {
            for p in rp[row]..rp[row + 1] {
                let w = cj[p] / block_size;
                let val_sq = vv[p].abs2();
                if u == w {
                    diag_sq += val_sq;
                } else {
                    *row_weights.entry(w).or_insert(0.0) += val_sq;
                }
            }
        }
        (diag_sq, row_weights)
    };

    #[cfg(feature = "rayon")]
    let node_rows: Vec<(f64, BTreeMap<usize, f64>)> = {
        use rayon::prelude::*;
        (0..n_nodes).into_par_iter().map(build_node_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let node_rows: Vec<(f64, BTreeMap<usize, f64>)> = (0..n_nodes).map(build_node_row).collect();

    let diag_norm: Vec<f64> = node_rows.iter().map(|(sq, _)| sq.sqrt()).collect();

    let filter_node_row = |u: usize| {
        let row_weights = &node_rows[u].1;
        let max_off = if normalize {
            0.0
        } else {
            row_weights.values().map(|sq| sq.sqrt()).fold(0.0, f64::max)
        };
        row_weights
            .iter()
            .filter_map(|(&w, &sq)| {
                let norm = sq.sqrt();
                let keep = if normalize {
                    let denom = diag_norm[u] * diag_norm[w];
                    denom > 0.0 && norm / denom >= theta
                } else {
                    max_off > 0.0 && norm >= theta * max_off
                };
                keep.then_some(w)
            })
            .collect::<Vec<_>>()
    };

    #[cfg(feature = "rayon")]
    let kept_rows: Vec<Vec<usize>> = {
        use rayon::prelude::*;
        (0..n_nodes).into_par_iter().map(filter_node_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let kept_rows: Vec<Vec<usize>> = (0..n_nodes).map(filter_node_row).collect();

    let mut row_ptr = Vec::with_capacity(n_nodes + 1);
    let mut col_idx = Vec::with_capacity(kept_rows.iter().map(Vec::len).sum());
    row_ptr.push(0);
    for cols in kept_rows {
        col_idx.extend_from_slice(&cols);
        row_ptr.push(col_idx.len());
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
