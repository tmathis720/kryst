#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;

#[derive(Clone, Debug)]
pub struct Strength {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
}

impl Strength {
    pub fn from_csr(a: &CsrMatrix<f64>, theta: f64, normalize: bool) -> Self {
        strength_csr(a, theta, normalize)
    }

    /// Optional per-row cap: keep the first `k` columns in ascending column order.
    pub fn capped(&self, k: usize) -> Strength {
        if k == 0 {
            return self.clone();
        }
        let n = self.row_ptr.len() - 1;
        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::with_capacity(self.col_idx.len());
        for i in 0..n {
            let rs = self.row_ptr[i];
            let re = self.row_ptr[i + 1];
            let take = (re - rs).min(k);
            col_idx.extend_from_slice(&self.col_idx[rs..rs + take]);
            row_ptr.push(col_idx.len());
        }
        Strength { row_ptr, col_idx }
    }

    /// Symmetrize the strength graph treating it as undirected.
    pub fn symmetrize(&self) -> Strength {
        let n = self.row_ptr.len() - 1;
        let mut deg = vec![0usize; n];
        for i in 0..n {
            for &j in &self.col_idx[self.row_ptr[i]..self.row_ptr[i + 1]] {
                deg[i] += 1;
                deg[j] += 1;
            }
        }
        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0);
        for i in 0..n {
            row_ptr.push(row_ptr[i] + deg[i]);
        }
        let mut col_idx = vec![usize::MAX; row_ptr[n]];
        let mut next = vec![0usize; n];
        for i in 0..n {
            next[i] = row_ptr[i];
        }
        for i in 0..n {
            for &j in &self.col_idx[self.row_ptr[i]..self.row_ptr[i + 1]] {
                let pi = next[i];
                col_idx[pi] = j;
                next[i] += 1;
                let pj = next[j];
                col_idx[pj] = i;
                next[j] += 1;
            }
        }
        for i in 0..n {
            let rs = row_ptr[i];
            let re = row_ptr[i + 1];
            col_idx[rs..re].sort_unstable();
            let mut write = rs;
            let mut last = None;
            for idx in rs..re {
                let v = col_idx[idx];
                if Some(v) != last {
                    col_idx[write] = v;
                    write += 1;
                    last = Some(v);
                }
            }
            let len = write - rs;
            col_idx.drain(write..re);
            let delta = (re - rs) - len;
            if delta > 0 {
                for r in (i + 1)..=n {
                    row_ptr[r] -= delta;
                }
            }
        }
        Strength { row_ptr, col_idx }
    }
}

pub fn strength_csr(a: &CsrMatrix<f64>, theta: f64, normalize: bool) -> Strength {
    let n = a.nrows();
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<usize> = Vec::with_capacity(a.nnz());
    row_ptr.push(0);

    if normalize {
        // |a_ij| / sqrt(|a_ii||a_jj|)
        let mut diag = vec![0.0f64; n];
        for i in 0..n {
            let rs = rp[i];
            let re = rp[i + 1];
            let mut aii = 0.0;
            for p in rs..re {
                if cj[p] == i {
                    aii = vv[p].abs();
                    break;
                }
            }
            diag[i] = aii;
        }
        for i in 0..n {
            let rs = rp[i];
            let re = rp[i + 1];
            let mut count = 0usize;
            for p in rs..re {
                let j = cj[p];
                if j == i {
                    continue;
                }
                let denom = (diag[i] * diag[j]).sqrt();
                if denom > 0.0 {
                    let s = vv[p].abs() / denom;
                    if s >= theta {
                        col_idx.push(j);
                        count += 1;
                    }
                }
            }
            row_ptr.push(row_ptr.last().unwrap() + count);
        }
    } else {
        // |a_ij| >= theta * max_off_i
        for i in 0..n {
            let rs = rp[i];
            let re = rp[i + 1];
            let mut max_off = 0.0f64;
            for p in rs..re {
                let j = cj[p];
                if j != i {
                    max_off = max_off.max(vv[p].abs());
                }
            }
            let thr = theta * max_off;
            let mut count = 0usize;
            for p in rs..re {
                let j = cj[p];
                if j != i && vv[p].abs() >= thr {
                    col_idx.push(j);
                    count += 1;
                }
            }
            row_ptr.push(row_ptr.last().unwrap() + count);
        }
    }

    Strength { row_ptr, col_idx }
}
