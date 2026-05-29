use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;
use std::collections::VecDeque;

/// Permutation with cached inverse mapping.
#[derive(Clone, Debug)]
pub struct Permutation {
    /// Maps new index -> old index (p\[i\] = old index of new position i)
    pub p: Vec<usize>,
    /// Maps old index -> new index (pinv\[old\] = new position of old index)
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

    /// Apply permutation to a vector: `y(new) = x(old)[p[new]]`
    pub fn apply_vec<S: KrystScalar>(&self, x_old: &[S], y_new: &mut [S]) {
        for (i, y) in y_new.iter_mut().enumerate() {
            *y = x_old[self.p[i]];
        }
    }

    /// Apply transpose permutation to a vector: `y(old) = x(new)[pinv[old]]`
    pub fn apply_vec_t<S: KrystScalar>(&self, x_new: &[S], y_old: &mut [S]) {
        for (i, y) in y_old.iter_mut().enumerate() {
            *y = x_new[self.pinv[i]];
        }
    }
}

/// Symmetric permutation of CSR matrix: A' = P A P^T
pub fn permute_csr_symmetric<T: KrystScalar>(a: &CsrMatrix<T>, perm: &Permutation) -> CsrMatrix<T> {
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
        let mut entries: Vec<(usize, T)> = Vec::with_capacity(re - rs);
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

/// Nonsymmetric permutation of CSR matrix: A' = P_row A P_col^T
pub fn permute_csr_nonsymmetric<T: KrystScalar>(
    a: &CsrMatrix<T>,
    row_perm: &Permutation,
    col_perm: &Permutation,
) -> CsrMatrix<T> {
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
        let old_i = row_perm.p[new_i];
        let rs = rp[old_i];
        let re = rp[old_i + 1];
        let mut entries = Vec::with_capacity(re - rs);
        for k in rs..re {
            let old_j = cj[k];
            let new_j = col_perm.pinv[old_j];
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
pub fn rcm_csr<T>(a: &CsrMatrix<T>) -> Permutation {
    let n = a.nrows();
    let mut adj = build_symmetric_adj_from_csr(a);
    rcm_from_adj(&mut adj)
}

/// Approximate minimum degree ordering for a symmetric graph given by CSR matrix.
pub fn amd_csr<T>(a: &CsrMatrix<T>) -> Permutation {
    let mut adj = build_symmetric_adj_from_csr(a);
    amd_from_adj(&mut adj)
}

pub(crate) fn rcm_from_adj(adj: &mut [Vec<usize>]) -> Permutation {
    let mut order = cuthill_mckee_from_adj(adj);
    order.reverse();
    permutation_from_order(order)
}

pub(crate) fn amd_from_adj(adj: &mut [Vec<usize>]) -> Permutation {
    let order = minimum_degree_order(adj);
    permutation_from_order(order)
}

pub(crate) fn permutation_from_order(order: Vec<usize>) -> Permutation {
    let n = order.len();
    let mut pinv = vec![0; n];
    for (new, &old) in order.iter().enumerate() {
        pinv[old] = new;
    }
    Permutation { p: order, pinv }
}

pub(crate) fn build_symmetric_adj_from_csr<T>(a: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let rp = a.row_ptr();
    let cj = a.col_idx();
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
    adj
}

fn minimum_degree_order(adj: &mut [Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    for neighbors in adj.iter_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut active = vec![true; n];
    let mut order = Vec::with_capacity(n);

    for _ in 0..n {
        let mut min_node = None;
        let mut min_degree = usize::MAX;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            let degree = adj[i].iter().filter(|&&j| active[j]).count();
            if degree < min_degree {
                min_degree = degree;
                min_node = Some(i);
            }
        }
        let node = min_node.expect("minimum degree ordering requires at least one node");
        active[node] = false;
        order.push(node);
    }

    order
}

pub(crate) fn cuthill_mckee_from_adj(adj: &mut [Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    for neighbors in adj.iter_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let degrees: Vec<usize> = adj.iter().map(|nbrs| nbrs.len()).collect();
    for neighbors in adj.iter_mut() {
        neighbors.sort_unstable_by(|&a, &b| degrees[a].cmp(&degrees[b]).then_with(|| a.cmp(&b)));
    }

    let mut visited = vec![false; n];
    let mut permutation = Vec::with_capacity(n);

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut current_start = start;
        let mut min_degree = degrees[start];
        for i in start..n {
            if !visited[i] && degrees[i] < min_degree {
                min_degree = degrees[i];
                current_start = i;
            }
        }

        let mut queue = VecDeque::new();
        queue.push_back(current_start);
        visited[current_start] = true;

        while let Some(node) = queue.pop_front() {
            permutation.push(node);
            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }

    permutation
}

#[cfg(all(test, feature = "backend-faer"))]
mod tests {
    use super::*;
    use crate::matrix::sparse::CsrMatrix;
    use faer::Mat;
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
        let dense_ap = ap.to_dense().unwrap();
        // compute dense reference
        let dense_a = a.to_dense().unwrap();
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
        use crate::algebra::prelude::*; // gives you S, R, etc.

        let row_ptr = vec![0, 2, 4, 6];
        let col_idx = vec![0, 1, 1, 2, 0, 2];
        let vals: Vec<S> = vec![
            S::new(R::from(1.0), R::from(0.5)),
            S::new(R::from(2.0), R::from(-1.0)),
            S::new(R::from(3.0), R::from(0.25)),
            S::new(R::from(4.0), R::from(-0.75)),
            S::new(R::from(5.0), R::from(1.5)),
            S::new(R::from(6.0), R::from(-2.0)),
        ];
        let a = CsrMatrix::from_csr(3, 3, row_ptr, col_idx, vals);
        let perm = Permutation {
            p: vec![2, 0, 1],
            pinv: vec![1, 2, 0],
        };

        let ap = permute_csr_symmetric(&a, &perm);
        let err = ap.to_dense().unwrap_err();
        assert!(matches!(err, crate::error::KError::Unsupported(_)));
        let err = a.to_dense().unwrap_err();
        assert!(matches!(err, crate::error::KError::Unsupported(_)));
    }

    #[test]
    fn rcm_csr_matches_helper_reverse() {
        let row_ptr = vec![0, 3, 6, 9, 12];
        let col_idx = vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3];
        let vals = vec![2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 5.0];
        let a = CsrMatrix::from_csr(4, 4, row_ptr, col_idx, vals);
        let mut adj = build_symmetric_adj_from_csr(&a);
        let order = cuthill_mckee_from_adj(&mut adj);

        let perm = rcm_csr(&a);
        let mut expected = order.clone();
        expected.reverse();
        assert_eq!(perm.p, expected);

        for (new, &old) in perm.p.iter().enumerate() {
            assert_eq!(perm.pinv[old], new);
        }
    }
}
