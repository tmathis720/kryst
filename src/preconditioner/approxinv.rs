//! Sparse Approximate Inverse (SPAI) preconditioner

use crate::core::traits::MatVec;
use crate::error::KError;
use crate::preconditioner::SparsityPattern;
use crate::preconditioner::Preconditioner;
use num_traits::Float;
use std::any::TypeId;
use faer::prelude::SolveLstsq;
use faer::linalg::solvers::SolveCore;
use std::marker::PhantomData;

/// Sparse Approximate Inverse (SPAI) preconditioner
pub struct ApproxInv<M, V, T> {
    pub pattern:    SparsityPattern,
    pub tol:        T,
    pub max_iter:   usize,
    pub nbsteps:    usize,    // max number of improvement steps per row
    pub max_size:   usize,    // max dimensions of working arrays
    pub max_new:    usize,    // max new fill per step
    pub block_size: usize,    // block-size >1 support
    pub cache_size: usize,    // cache-level hint
    pub verbose:    bool,     // print timing/stats
    pub sp:         bool,     // symmetric pattern flag
    pub inv_rows:   Vec<Vec<(usize, T)>>, // CSR‐like: for each row i, (j, mij)
    pub a:          Option<M>,
    _phantom:       PhantomData<V>,
}

impl<M, V, T> ApproxInv<M, V, T>
where
    T: Float,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pattern: SparsityPattern,
        tol: T,
        max_iter: usize,
        nbsteps: usize,
        max_size: usize,
        max_new: usize,
        block_size: usize,
        cache_size: usize,
        verbose: bool,
        sp: bool,
    ) -> Self {
        Self {
            pattern,
            tol,
            max_iter,
            nbsteps,
            max_size,
            max_new,
            block_size,
            cache_size,
            verbose,
            sp,
            inv_rows: Vec::new(),
            a: None,
            _phantom: PhantomData,
        }
    }
}

impl<M: 'static + Sync, V: Sync, T> Preconditioner<M, V> for ApproxInv<M, V, T>
where
    M: MatVec<V>,
    V: From<Vec<T>> + AsRef<[T]> + AsMut<[T]> + Clone,
    T: Float + 'static + Send + Sync,
{
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        let n = match &self.pattern {
            SparsityPattern::Manual(pat) => pat.len(),
            SparsityPattern::Auto => {
                // Try to get n from a.nrows() if available
                if let Some(nrows) = get_nrows(a) {
                    nrows
                } else {
                    return Err(KError::Unsupported("SparsityPattern::Auto requires nrows() or row_indices() support"));
                }
            }
        };
        // Build SPAI by columns: for each column j, solve A m = e_j
        let mut cols = vec![vec![T::zero(); n]; n];
        for j in 0..n {
            let pattern: Vec<usize> = match &self.pattern {
                SparsityPattern::Auto => {
                    if let Some(rowpat) = get_row_pattern(a) {
                        rowpat.row_indices(j).to_vec()
                    } else {
                        (0..n).collect()
                    }
                }
                SparsityPattern::Manual(pat) => pat.get(j).cloned().unwrap_or_else(Vec::new),
            };
            let m = pattern.len();
            let mut b = vec![vec![T::zero(); n]; m];
            for (row_idx, &i) in pattern.iter().enumerate() {
                let mut ei = V::from(vec![T::zero(); n]);
                ei.as_mut()[i] = T::one();
                let mut col = V::from(vec![T::zero(); n]);
                a.matvec(&ei, &mut col);
                for k in 0..n {
                    b[row_idx][k] = col.as_ref()[k];
                }
            }
            let mut e_j = vec![T::zero(); n];
            e_j[j] = T::one();
            let m_vec: Vec<T> = if TypeId::of::<T>() == TypeId::of::<f64>() {
                use faer::{Mat, MatMut};
                use faer::linalg::solvers::{Qr, FullPivLu};
                // b_f64: n x m (column-major)
                let b_f64 = Mat::from_fn(n, m, |j, i| b[i][j].to_f64().unwrap());
                let rhs = Mat::from_fn(n, 1, |i, _| e_j[i].to_f64().unwrap());
                let sol: Vec<f64> = if m == n {
                    // square: FullPivLu
                    let lu = FullPivLu::new(b_f64.as_ref());
                    let mut x = rhs.col_as_slice(0).to_vec();
                    let x_mat = MatMut::from_column_major_slice_mut(&mut x, n, 1);
                    lu.solve_in_place_with_conj(faer::Conj::No, x_mat);
                    x
                } else {
                    // least squares: Qr, returns m x 1
                    let sol_mat = Qr::new(b_f64.as_ref()).solve_lstsq(rhs);
                    (0..m).map(|i| sol_mat[(i, 0)]).collect()
                };
                // Scatter m values into n-length vector
                let mut full = vec![T::zero(); n];
                for (k, &row_i) in pattern.iter().enumerate() {
                    full[row_i] = T::from(sol[k]).unwrap();
                }
                full
            } else {
                // Normal equations fallback: build m x m system
                let mut bt_b = vec![vec![T::zero(); m]; m];
                let mut bt_e = vec![T::zero(); m];
                for r in 0..m {
                    for c in 0..m {
                        for k in 0..n {
                            bt_b[r][c] = bt_b[r][c] + b[r][k] * b[c][k];
                        }
                    }
                    for k in 0..n {
                        bt_e[r] = bt_e[r] + b[r][k] * e_j[k];
                    }
                }
                let mut m_vec_pattern = bt_e.clone();
                for k in 0..m {
                    let mut max_row = k;
                    for r in (k+1)..m {
                        if bt_b[r][k].abs() > bt_b[max_row][k].abs() {
                            max_row = r;
                        }
                    }
                    if max_row != k {
                        bt_b.swap(k, max_row);
                        m_vec_pattern.swap(k, max_row);
                    }
                    let pivot = bt_b[k][k];
                    if pivot.abs() < self.tol {
                        continue;
                    }
                    for r in (k+1)..m {
                        let f = bt_b[r][k] / pivot;
                        for c in k..m {
                            bt_b[r][c] = bt_b[r][c] - f * bt_b[k][c];
                        }
                        m_vec_pattern[r] = m_vec_pattern[r] - f * m_vec_pattern[k];
                    }
                }
                for k in (0..m).rev() {
                    let mut sum = m_vec_pattern[k];
                    for c in (k+1)..m {
                        sum = sum - bt_b[k][c] * m_vec_pattern[c];
                    }
                    let pivot = bt_b[k][k];
                    if pivot.abs() < self.tol {
                        m_vec_pattern[k] = T::zero();
                    } else {
                        m_vec_pattern[k] = sum / pivot;
                    }
                }
                // Scatter m values into n-length vector
                let mut full = vec![T::zero(); n];
                for (k, &row_i) in pattern.iter().enumerate() {
                    full[row_i] = m_vec_pattern[k];
                }
                full
            };
            for i in 0..n {
                cols[j][i] = m_vec[i];
            }
        }
        // Transpose cols to row storage
        self.inv_rows = vec![vec![]; n];
        for i in 0..n {
            for j in 0..n {
                if cols[j][i].abs() > self.tol {
                    self.inv_rows[i].push((j, cols[j][i]));
                }
            }
        }
        Ok(())
    }
    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError> {
        // Zero-out y
        for yi in y.as_mut().iter_mut() {
            *yi = T::zero();
        }
        // Sparse matvec: y_i = sum_j M_ij x_j
        let x_ref = x.as_ref();
        let y_mut = y.as_mut();
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            y_mut.par_iter_mut().enumerate().for_each(|(i, yi)| {
                let mut sum = T::zero();
                for &(j, mij) in &self.inv_rows[i] {
                    sum = sum + mij * x_ref[j];
                }
                *yi = sum;
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for (i, row) in self.inv_rows.iter().enumerate() {
                let mut sum = T::zero();
                for &(j, mij) in row {
                    sum = sum + mij * x_ref[j];
                }
                y_mut[i] = sum;
            }
        }
        Ok(())
    }
}

// Helper: get nrows from a if possible
fn get_nrows<M: 'static>(a: &M) -> Option<usize> {
    use crate::core::traits::Indexing;
    let any = a as &dyn std::any::Any;
    if let Some(indexed) = any.downcast_ref::<&dyn Indexing>() {
        Some(indexed.nrows())
    } else if let Some(indexed) = any.downcast_ref::<&dyn crate::core::traits::MatShape>() {
        Some(indexed.nrows())
    } else {
        None
    }
}

// Helper: get RowPattern trait object if implemented
fn get_row_pattern<M: 'static>(a: &M) -> Option<&dyn crate::core::traits::RowPattern> {
    let any = a as &dyn std::any::Any;
    if let Some(rowpat) = any.downcast_ref::<&dyn crate::core::traits::RowPattern>() {
        Some(*rowpat)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;
    use crate::preconditioner::Preconditioner;
    use approx::assert_relative_eq;

    #[derive(Clone)]
    struct DenseMat<T> {
        data: Vec<Vec<T>>,
    }
    impl<T: Float> MatVec<Vec<T>> for DenseMat<T> {
        fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
            for (i, row) in self.data.iter().enumerate() {
                y[i] = row.iter().zip(x.iter()).map(|(a, b)| *a * *b).fold(T::zero(), |acc, v| acc + v);
            }
        }
    }
    impl<T: Float> crate::core::traits::RowPattern for DenseMat<T> {
        fn row_indices(&self, i: usize) -> &[usize] {
            thread_local! {
                static IDX: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::new());
            }
            let row = &self.data[i];
            IDX.with(|idx| {
                let mut idx = idx.borrow_mut();
                idx.clear();
                for (j, &val) in row.iter().enumerate() {
                    if val != T::zero() {
                        idx.push(j);
                    }
                }
                // SAFETY: only used in single-threaded test context
                unsafe { std::mem::transmute::<&[usize], &[usize]>(&*idx) }
            })
        }
    }

    fn eye<T: Float>(n: usize) -> DenseMat<T> {
        DenseMat {
            data: (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { T::one() } else { T::zero() })
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn approxinv_exact_inverse() {
        // 3x3 diagonal matrix
        let a = DenseMat { data: vec![vec![2.0, 0.0, 0.0], vec![0.0, 3.0, 0.0], vec![0.0, 0.0, 4.0]] };
        let pattern = SparsityPattern::Manual(vec![vec![0], vec![1], vec![2]]);
        let mut spai = ApproxInv::<_, Vec<f64>, f64>::new(
            pattern, 1e-12, 10, 1, 100, 8, 1, 0, false, false
        );
        spai.setup(&a).unwrap();
        // Should recover the exact inverse
        let inv = &spai.inv_rows;
        assert_relative_eq!(inv[0][0].1, 0.5, epsilon = 1e-12);
        assert_relative_eq!(inv[1][0].1, 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(inv[2][0].1, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn approxinv_apply_vector() {
        // 2x2 matrix
        let a = DenseMat { data: vec![vec![4.0, 1.0], vec![2.0, 3.0]] };
        let pattern = SparsityPattern::Manual(vec![vec![0, 1], vec![0, 1]]);
        let mut spai = ApproxInv::<_, Vec<f64>, f64>::new(
            pattern, 1e-12, 10, 1, 100, 8, 1, 0, false, false
        );
        spai.setup(&a).unwrap();
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0, 0.0];
        spai.apply(&x, &mut y).unwrap();
        // Compare to expected inverse * x using faer
        let a_inv = faer::Mat::<f64>::from_fn(2, 2, |i, j| {
            match (i, j) {
                (0, 0) => 0.375,
                (0, 1) => -0.125,
                (1, 0) => -0.25,
                (1, 1) => 0.5,
                _ => 0.0,
            }
        });
        let x_vec = faer::Mat::<f64>::from_fn(2, 1, |i, _| x[i]);
        let y_expected = &a_inv * &x_vec;
        assert_relative_eq!(y[0], y_expected[(0, 0)], epsilon = 2.5e-1);
        assert_relative_eq!(y[1], y_expected[(1, 0)], epsilon = 2.5e-1);
    }

    #[test]
    fn approxinv_identity() {
        // Identity matrix
        let a = eye::<f64>(4);
        let pattern = SparsityPattern::Manual(vec![vec![0], vec![1], vec![2], vec![3]]);
        let mut spai = ApproxInv::<_, Vec<f64>, f64>::new(
            pattern, 1e-12, 10, 1, 100, 8, 1, 0, false, false
        );
        spai.setup(&a).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];
        spai.apply(&x, &mut y).unwrap();
        assert_relative_eq!(x[0], y[0], epsilon = 1e-12);
        assert_relative_eq!(x[1], y[1], epsilon = 1e-12);
        assert_relative_eq!(x[2], y[2], epsilon = 1e-12);
        assert_relative_eq!(x[3], y[3], epsilon = 1e-12);
    }

    #[test]
    fn debug_faer_lu_inverse_rows() {
        use faer::{Mat, MatMut};
        use faer::linalg::solvers::FullPivLu;
        let a = Mat::from_fn(2, 2, |j, i| match (i, j) {
            (0, 0) => 4.0,
            (0, 1) => 1.0,
            (1, 0) => 2.0,
            (1, 1) => 3.0,
            _ => 0.0,
        });
        let lu = FullPivLu::new(a.as_ref());
        let mut inv = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            let mut e = vec![0.0; 2];
            e[i] = 1.0;
            let mut x = e.clone();
            let x_mat = MatMut::from_column_major_slice_mut(&mut x, 2, 1);
            lu.solve_in_place_with_conj(faer::Conj::No, x_mat);
            for j in 0..2 {
                inv[i][j] = x[j];
            }
        }
        println!("faer LU inverse rows: {:?}", inv);
    }
}
