//! ILUT preconditioner stub
//!
//! Implements ILUT (Incomplete LU with threshold and fill-in control) as a preconditioner.
//!
//! # Overview
//!
//! ILUT is an incomplete LU factorization with user-specified fill-in and drop tolerance.
//! It produces lower (L) and upper (U) triangular factors by dropping small entries and limiting
//! the number of nonzeros per row, making it suitable as a preconditioner for iterative solvers
//! on sparse matrices. The drop tolerance controls numerical dropping, and the fill parameter
//! limits the number of nonzeros per row.
//!
//! # Usage
//!
//! - Create an `Ilut` preconditioner with the desired fill and drop tolerance.
//! - Call `setup` with the system matrix to compute the factors.
//! - Use `apply` to solve M⁻¹r ≈ A⁻¹r using the computed factors.
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Section 10.3.

use crate::error::KError;
use crate::preconditioner::Preconditioner;
use crate::core::traits::MatShape;

/// Sparse row structure for storing L/U factors.
///
/// Each row stores the column indices and values of nonzero entries.
#[derive(Clone)]
pub struct SparseRow<T> {
    /// Column indices of nonzero entries
    pub cols: Vec<usize>,
    /// Values of nonzero entries
    pub vals: Vec<T>,
}
impl<T> SparseRow<T> {
    /// Create an empty sparse row
    pub fn new() -> Self {
        Self { cols: Vec::new(), vals: Vec::new() }
    }
}
impl<T> Default for SparseRow<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// ILUT preconditioner struct.
///
/// - `fill`: Maximum number of nonzeros per row (fill-in control)
/// - `droptol`: Drop tolerance (numerical dropping)
/// - `l`: Lower triangular factor (sparse row format)
/// - `u`: Upper triangular factor (sparse row format)
/// - `n`: Matrix size
pub struct Ilut<T> {
    pub fill: usize,
    pub droptol: T,
    pub l: Vec<SparseRow<T>>,
    pub u: Vec<SparseRow<T>>,
    pub n: usize,
}

impl<T: num_traits::Float + Clone + std::fmt::Debug> Ilut<T> {
    /// Create a new ILUT preconditioner with fill and drop tolerance.
    pub fn new(fill: usize, droptol: T) -> Self {
        Self { fill, droptol, l: Vec::new(), u: Vec::new(), n: 0 }
    }
}

impl<M, V, T> Preconditioner<M, V> for Ilut<T>
where
    T: num_traits::Float + Clone + std::fmt::Debug + PartialOrd,
    M: crate::core::traits::MatVec<V> + MatShape + std::ops::Index<(usize, usize), Output = T>,
    V: AsRef<[T]> + AsMut<[T]>,
{
    /// Setup ILUT factors from matrix `a`.
    ///
    /// For each row, keeps only the largest `fill` entries above the drop tolerance.
    /// Partitions each row into L (j < i) and U (j >= i).
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        let n = a.nrows();
        self.n = n;
        self.l = vec![SparseRow::new(); n];
        self.u = vec![SparseRow::new(); n];
        for i in 0..n {
            let mut row = vec![];
            // Gather all nonzero entries in row i
            for j in 0..n {
                let val = a[(i, j)];
                if !val.is_zero() {
                    row.push((j, val));
                }
            }
            // Apply dropping by magnitude (ILUT)
            row.retain(|&(_, v)| v.abs() >= self.droptol);
            // Keep only largest 'fill' entries by magnitude
            if row.len() > self.fill {
                row.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
                row.truncate(self.fill);
            }
            // Partition into L (j < i) and U (j >= i)
            let mut lrow = SparseRow::new();
            let mut urow = SparseRow::new();
            for (j, v) in row {
                if j < i {
                    lrow.cols.push(j);
                    lrow.vals.push(v);
                } else {
                    urow.cols.push(j);
                    urow.vals.push(v);
                }
            }
            self.l[i] = lrow;
            self.u[i] = urow;
        }
        Ok(())
    }
    /// Apply ILUT preconditioner: solve Ly = r, then Uz = y.
    ///
    /// Forward substitution for L, then backward substitution for U.
    fn apply(&self, r: &V, z: &mut V) -> Result<(), KError> {
        let n = self.n;
        let r = r.as_ref();
        let z = z.as_mut();
        let mut y = vec![T::zero(); n];
        // Forward substitution: solve L y = r
        for i in 0..n {
            let mut sum = r[i];
            for (j_idx, &j) in self.l[i].cols.iter().enumerate() {
                sum = sum - self.l[i].vals[j_idx] * y[j];
            }
            y[i] = sum;
        }
        // Backward substitution: solve U z = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            for (j_idx, &j) in self.u[i].cols.iter().enumerate() {
                if j > i {
                    sum = sum - self.u[i].vals[j_idx] * z[j];
                }
            }
            // Diagonal entry must exist for U(i,i)
            if let Some(idx) = self.u[i].cols.iter().position(|&col| col == i) {
                z[i] = sum / self.u[i].vals[idx];
            } else {
                z[i] = sum;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatShape;

    struct DenseMat<T> {
        data: Vec<Vec<T>>,
    }
    impl<T: Copy> DenseMat<T> {
        fn new(data: Vec<Vec<T>>) -> Self { Self { data } }
    }
    impl<T: Copy> MatShape for DenseMat<T> {
        fn nrows(&self) -> usize { self.data.len() }
        fn ncols(&self) -> usize { self.data[0].len() }
    }
    impl<T: Copy> std::ops::Index<(usize, usize)> for DenseMat<T> {
        type Output = T;
        fn index(&self, idx: (usize, usize)) -> &Self::Output {
            &self.data[idx.0][idx.1]
        }
    }
    impl<T> crate::core::traits::MatVec<Vec<T>> for DenseMat<T>
    where
        T: Copy + std::ops::Mul<Output = T> + num_traits::Zero + std::ops::Add<Output = T>,
    {
        fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
            for i in 0..self.nrows() {
                y[i] = (0..self.ncols()).map(|j| self[(i, j)] * x[j]).fold(T::zero(), |a, b| a + b);
            }
        }
    }

    #[test]
    fn ilut_identity() {
        type Mat = DenseMat<f64>;
        let a = Mat::new(vec![vec![1.0f64, 0.0], vec![0.0, 1.0]]);
        let mut pc: Ilut<f64> = Ilut::new(2, 1e-12);
        pc.setup(&a).unwrap();
        let r = vec![2.0f64, 3.0];
        let mut z = vec![0.0; 2];
        Preconditioner::<Mat, Vec<f64>>::apply(&pc, &r, &mut z).unwrap();
        assert!((z[0] - 2.0).abs() < 1e-12 && (z[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn ilut_tridiag() {
        type Mat = DenseMat<f64>;
        let a = Mat::new(vec![
            vec![2.0f64, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ]);
        let mut pc: Ilut<f64> = Ilut::new(3, 1e-12);
        pc.setup(&a).unwrap();
        let r = vec![1.0f64, 2.0, 3.0];
        let mut z = vec![0.0; 3];
        Preconditioner::<Mat, Vec<f64>>::apply(&pc, &r, &mut z).unwrap();
        assert!(z.iter().all(|&zi| zi.is_finite()));
    }
}
