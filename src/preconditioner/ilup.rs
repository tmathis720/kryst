//! ILU(p) preconditioner stub
//!
//! Implements classical ILU(p) factorization with level-of-fill control (Saad §10.3).
//! Used as a preconditioner for Krylov solvers.
//!
//! # Overview
//!
//! ILU(p) is an incomplete LU factorization with a user-specified level-of-fill `p`.
//! It produces lower (L) and upper (U) triangular factors with controlled fill-in, making it suitable
//! as a preconditioner for iterative solvers on sparse matrices. The fill level determines how much
//! additional nonzero structure is allowed beyond the original sparsity pattern.
//!
//! # Usage
//!
//! - Create an `Ilup` preconditioner with the desired fill level.
//! - Call `setup` with the system matrix to compute the factors.
//! - Use `apply` to solve M⁻¹r ≈ A⁻¹r using the computed factors.
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Section 10.3.

#[cfg(feature = "complex")]
use crate::algebra::bridge::{BridgeScratch, copy_real_into_scalar, copy_scalar_to_real_in};
#[cfg(feature = "complex")]
use crate::algebra::prelude::*;
use crate::core::traits::MatShape;
use crate::error::KError;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::legacy::Preconditioner;

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
        Self {
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }
}
impl<T> Default for SparseRow<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// ILU(p) preconditioner struct.
///
/// - `fill`: Level-of-fill parameter (maximum allowed fill-in)
/// - `l`: Lower triangular factor (sparse row format)
/// - `u`: Upper triangular factor (sparse row format)
/// - `n`: Matrix size
pub struct Ilup<T> {
    pub fill: usize,
    pub l: Vec<SparseRow<T>>,
    pub u: Vec<SparseRow<T>>,
    pub n: usize,
}

impl<T: num_traits::Float + Clone + std::fmt::Debug> Ilup<T> {
    /// Create a new ILU(p) preconditioner with given fill level.
    pub fn new(fill: usize) -> Self {
        Self {
            fill,
            l: Vec::new(),
            u: Vec::new(),
            n: 0,
        }
    }
}

impl<T> Ilup<T>
where
    T: num_traits::Float + Clone + std::fmt::Debug + PartialOrd,
{
    fn apply_slice(
        &self,
        _side: crate::preconditioner::PcSide,
        r: &[T],
        z: &mut [T],
    ) -> Result<(), KError> {
        let n = self.n;
        if r.len() != n || z.len() != n {
            return Err(KError::InvalidInput(format!(
                "Ilup::apply dimension mismatch: n={}, r.len()={}, z.len()={}",
                n,
                r.len(),
                z.len()
            )));
        }

        let mut y = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = r[i];
            for (j_idx, &j) in self.l[i].cols.iter().enumerate() {
                sum = sum - self.l[i].vals[j_idx] * y[j];
            }
            y[i] = sum;
        }

        for i in (0..n).rev() {
            let mut sum = y[i];
            for (j_idx, &j) in self.u[i].cols.iter().enumerate() {
                if j > i {
                    sum = sum - self.u[i].vals[j_idx] * z[j];
                }
            }
            if let Some(idx) = self.u[i].cols.iter().position(|&col| col == i) {
                z[i] = sum / self.u[i].vals[idx];
            } else {
                z[i] = sum;
            }
        }

        Ok(())
    }
}

impl<M, V, T> Preconditioner<M, V> for Ilup<T>
where
    T: num_traits::Float + Clone + std::fmt::Debug + PartialOrd,
    M: crate::core::traits::MatVec<V> + MatShape + std::ops::Index<(usize, usize), Output = T>,
    V: AsRef<[T]> + AsMut<[T]>,
{
    /// Setup ILU(p) factors from matrix `a`.
    ///
    /// Performs classical ILU(p) factorization (Saad §10.3) with level-of-fill bookkeeping.
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        // Classical ILU(p) factorization (Saad §10.3)
        let n = a.nrows();
        self.n = n;
        self.l = vec![SparseRow::new(); n];
        self.u = vec![SparseRow::new(); n];
        // Level-of-fill bookkeeping: level[i][j] = fill level of entry (i,j)
        let mut level = vec![vec![usize::MAX; n]; n];
        for i in 0..n {
            for j in 0..n {
                if !a[(i, j)].is_zero() {
                    level[i][j] = 0;
                }
            }
        }
        // Copy matrix to working array
        let mut a_work = vec![vec![T::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                a_work[i][j] = a[(i, j)];
            }
        }
        // Main ILU(p) factorization loop
        for i in 0..n {
            // Compute L(i,j) for j < i
            for j in 0..i {
                if !a_work[i][j].is_zero() && level[i][j] <= self.fill {
                    // Find U[j,j]
                    let u_jj = a_work[j][j];
                    if u_jj.is_zero() {
                        return Err(KError::SolveError(format!(
                            "ILUP: zero diagonal in U at row {j}"
                        )));
                    }
                    let lij = a_work[i][j] / u_jj;
                    self.l[i].cols.push(j);
                    self.l[i].vals.push(lij);
                    // Update fill levels and values for row i
                    for k in (j + 1)..n {
                        if !a_work[j][k].is_zero() {
                            let new_level =
                                level[i][j].saturating_add(level[j][k]).saturating_add(1);
                            if new_level <= self.fill {
                                let update = lij * a_work[j][k];
                                a_work[i][k] = a_work[i][k] - update;
                                level[i][k] = level[i][k].min(new_level);
                            }
                        }
                    }
                }
            }
            // Store U(i,*) for k >= i, with level <= fill
            for k in i..n {
                if !a_work[i][k].is_zero() && level[i][k] <= self.fill {
                    self.u[i].cols.push(k);
                    self.u[i].vals.push(a_work[i][k]);
                }
            }
        }
        Ok(())
    }
    /// Apply ILU(p) preconditioner: solve Ly = r, then Uz = y.
    ///
    /// Forward substitution for L, then backward substitution for U.
    fn apply(&self, side: crate::preconditioner::PcSide, r: &V, z: &mut V) -> Result<(), KError> {
        self.apply_slice(side, r.as_ref(), z.as_mut())
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for Ilup<f64> {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply_s(
        &self,
        side: crate::preconditioner::PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(format!(
                "Ilup::apply_s dimension mismatch: x.len()={}, y.len()={}",
                x.len(),
                y.len()
            )));
        }

        let n = x.len();
        scratch.with_pair(n, |xr, yr| {
            copy_scalar_to_real_in(x, xr);
            self.apply_slice(side, xr, yr)?;
            copy_real_into_scalar(yr, y);
            Ok(())
        })
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
        fn new(data: Vec<Vec<T>>) -> Self {
            Self { data }
        }
    }
    impl<T: Copy> MatShape for DenseMat<T> {
        fn nrows(&self) -> usize {
            self.data.len()
        }
        fn ncols(&self) -> usize {
            self.data[0].len()
        }
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
                y[i] = (0..self.ncols())
                    .map(|j| self[(i, j)] * x[j])
                    .fold(T::zero(), |a, b| a + b);
            }
        }
    }

    #[test]
    fn ilup_identity() {
        type Mat = DenseMat<f64>;
        let a = Mat::new(vec![vec![1.0f64, 0.0], vec![0.0, 1.0]]);
        let mut pc: Ilup<f64> = Ilup::new(0);
        pc.setup(&a).unwrap();
        let r = vec![2.0f64, 3.0];
        let mut z = vec![0.0; 2];
        Preconditioner::<Mat, Vec<f64>>::apply(
            &pc,
            crate::preconditioner::PcSide::Left,
            &r,
            &mut z,
        )
        .unwrap();
        assert!((z[0] - 2.0).abs() < 1e-12 && (z[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn ilup_tridiag() {
        type Mat = DenseMat<f64>;
        let a = Mat::new(vec![
            vec![2.0f64, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ]);
        let mut pc: Ilup<f64> = Ilup::new(0);
        pc.setup(&a).unwrap();
        let r = vec![1.0f64, 2.0, 3.0];
        let mut z = vec![0.0; 3];
        Preconditioner::<Mat, Vec<f64>>::apply(
            &pc,
            crate::preconditioner::PcSide::Left,
            &r,
            &mut z,
        )
        .unwrap();
        assert!(z.iter().all(|&zi| zi.is_finite()));
    }

    #[cfg(feature = "complex")]
    #[test]
    fn apply_s_matches_real_path() {
        use crate::algebra::bridge::BridgeScratch;
        use crate::algebra::prelude::*;
        use crate::ops::kpc::KPreconditioner;

        type Mat = DenseMat<f64>;
        let a = Mat::new(vec![vec![4.0f64, 1.0], vec![1.5, 3.5]]);
        let mut pc: Ilup<f64> = Ilup::new(1);
        pc.setup(&a).unwrap();

        let rhs_real = vec![6.0f64, 4.0];
        let mut out_real = vec![0.0; rhs_real.len()];
        Preconditioner::<Mat, Vec<f64>>::apply(
            &pc,
            crate::preconditioner::PcSide::Left,
            &rhs_real,
            &mut out_real,
        )
        .expect("ilup real apply");

        let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
        let mut out_s = vec![S::zero(); rhs_s.len()];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(
            crate::preconditioner::PcSide::Left,
            &rhs_s,
            &mut out_s,
            &mut scratch,
        )
        .expect("ilup apply_s");

        for (ys, yr) in out_s.iter().zip(out_real.iter()) {
            assert!((ys.real() - yr).abs() < 1e-10);
        }
    }
}
