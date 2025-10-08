#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;

/// Permutation with cached inverse mapping.
#[derive(Clone, Debug)]
pub struct Permutation {
    /// Maps new index -> old index (p[i] = old index of new position i)
    pub p: Vec<usize>,
    /// Maps old index -> new index (pinv[old] = new position of old index)
    pub pinv: Vec<usize>,
}

impl Permutation {
    /// Identity permutation of length `n`.
    pub fn identity(n: usize) -> Self {
        let p: Vec<usize> = (0..n).collect();
        let pinv = p.clone();
        Self { p, pinv }
    }

    /// Length of permutation.
    #[inline]
    pub fn len(&self) -> usize {
        self.p.len()
    }

    /// Apply permutation to a vector: y(new) = x(old)[p[new]]
    pub fn apply_vec<S: KrystScalar>(&self, x_old: &[S], y_new: &mut [S]) {
        for (i, y) in y_new.iter_mut().enumerate() {
            *y = x_old[self.p[i]];
        }
    }

    /// Apply transpose permutation to a vector: y(old) = x(new)[pinv[old]]
    pub fn apply_vec_t<S: KrystScalar>(&self, x_new: &[S], y_old: &mut [S]) {
        for (i, y) in y_old.iter_mut().enumerate() {
            *y = x_new[self.pinv[i]];
        }
    }
}

/// Symmetric permutation of CSR matrix: A' = P A P^T
pub fn permute_csr_symmetric(a: &CsrMatrix<S>, perm: &Permutation) -> CsrMatrix<S> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    let rp = a.row_ptr();
    let cj = a.col_idx();
    let vv = a.values();

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(vv.len());
    let mut values = Vec::with_capacity(vv.len());
    row_ptr.push(0);

    for new_i in 0..n {
        let old_i = perm.p[new_i];
        let rs = rp[old_i];
        let re = rp[old_i + 1];
        let mut entries: Vec<(usize, S)> = Vec::with_capacity(re - rs);
        for k in rs..re {
            let old_j = cj[k];
            let new_j = perm.pinv[old_j];
            entries.push((new_j, vv[k]));
        }
        entries.sort_unstable_by_key(|e| e.0);
        for (j, v) in entries {
            col_idx.push(j);
            values.push(v);
        }
        row_ptr.push(col_idx.len());
    }

    CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}

/// Reverse Cuthill-McKee ordering for a symmetric graph given by CSR matrix.
pub fn rcm_csr(a: &CsrMatrix<S>) -> Permutation {
    let n = a.nrows();
    let rp = a.row_ptr();
    let cj = a.col_idx();
    // build symmetric adjacency list
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for k in rp[i]..rp[i + 1] {
            let j = cj[k];
            if i == j {
                continue;
            }
            adj[i].push(j);
            adj[j].push(i);
        }
    }
    for v in &mut adj {
        v.sort_unstable();
        v.dedup();
    }

    let degrees: Vec<usize> = adj.iter().map(|nbrs| nbrs.len()).collect();
    for i in 0..n {
        adj[i].sort_unstable_by(|&a, &b| degrees[a].cmp(&degrees[b]).then(a.cmp(&b)));
    }

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for start in 0..n {
        if visited[start] {
            continue;
        }
        // find unvisited node with smallest degree
        let mut s = start;
        let mut min_deg = degrees[start];
        for i in start..n {
            if !visited[i] && degrees[i] < min_deg {
                min_deg = degrees[i];
                s = i;
            }
        }
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);
        visited[s] = true;
        while let Some(i) = queue.pop_front() {
            order.push(i);
            for &j in &adj[i] {
                if !visited[j] {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }
    }
    order.reverse();

    let mut pinv = vec![0; n];
    for (new, &old) in order.iter().enumerate() {
        pinv[old] = new;
    }
    Permutation { p: order, pinv }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permute_csr_symmetric_matches_dense() {
        // 3x3 matrix
        // [1 2 0; 0 3 4; 5 0 6]
        let row_ptr = vec![0, 2, 4, 6];
        let col_idx = vec![0, 1, 1, 2, 0, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = CsrMatrix::from_csr(3, 3, row_ptr, col_idx, vals);
        let perm = Permutation {
            p: vec![2, 0, 1],
            pinv: vec![1, 2, 0],
        };
        let ap = permute_csr_symmetric(&a, &perm);
        let dense_ap = ap.to_dense();
        // compute dense reference
        let dense_a = a.to_dense();
        let mut ref_dense = faer::Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let old_i = perm.p[i];
                let old_j = perm.p[j];
                ref_dense[(i, j)] = dense_a[(old_i, old_j)];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!((dense_ap[(i, j)] - ref_dense[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn permute_csr_symmetric_complex_matches_dense() {
        use num_complex::Complex64;

        let row_ptr = vec![0, 2, 4, 6];
        let col_idx = vec![0, 1, 1, 2, 0, 2];
        let vals = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.25),
            Complex64::new(4.0, -0.75),
            Complex64::new(5.0, 1.5),
            Complex64::new(6.0, -2.0),
        ];
        let a = CsrMatrix::new(3, 3, row_ptr, col_idx, vals);
        let perm = Permutation {
            p: vec![2, 0, 1],
            pinv: vec![1, 2, 0],
        };
        let ap = permute_csr_symmetric(&a, &perm);
        let dense_ap = ap.to_dense();
        let dense_a = a.to_dense();
        let mut ref_dense = faer::Mat::<Complex64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let old_i = perm.p[i];
                let old_j = perm.p[j];
                ref_dense[(i, j)] = dense_a[(old_i, old_j)];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let diff = dense_ap[(i, j)] - ref_dense[(i, j)];
                assert!(diff.norm() < 1e-12);
            }
        }
    }
}
