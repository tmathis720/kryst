//! Classical ILU(p) preconditioner.
//!
//! Implements the textbook ILU(p) factorization with explicit level-of-fill control (Saad §10.3).
//! This preconditioner is useful for general `MatVec`/`MatShape` operators where the matrix is
//! not stored in a dense or CSR-friendly format.
//!
//! # Overview
//!
//! - Create an `Ilup` preconditioner with the desired level-of-fill.
//! - Call `setup` with the system matrix to compute the L/U factors.
//! - Use `apply` to perform the forward/backward sweeps (Ly = r, Uz = y).
//!
//! # Notes
//!
//! - For HYPRE-style ILUK or ILUT on `faer::Mat<f64>`, prefer the canonical `Ilu` implementation
//!   (`IluType::ILUK` / `IluType::ILUT`).
//! - `Ilup` remains useful when you need a generic matrix interface or want to embed the fill
//!   control logic in custom operators.
//! - It is generic over the Kryst scalar `S`, so complex problems can use this preconditioner directly.

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
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
pub struct SparseRow {
    /// Column indices of nonzero entries
    pub cols: Vec<usize>,
    /// Values of nonzero entries
    pub vals: Vec<S>,
}
impl SparseRow {
    /// Create an empty sparse row
    pub fn new() -> Self {
        Self {
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }
}
impl Default for SparseRow {
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
pub struct Ilup {
    pub fill: usize,
    pub l: Vec<SparseRow>,
    pub u: Vec<SparseRow>,
    pub n: usize,
}

impl Ilup {
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

impl Ilup {
    fn apply_slice(
        &self,
        _side: crate::preconditioner::PcSide,
        r: &[S],
        z: &mut [S],
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

        let mut y = vec![S::zero(); n];
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

impl<M, V> Preconditioner<M, V> for Ilup
where
    M: crate::core::traits::MatVec<V> + MatShape + std::ops::Index<(usize, usize), Output = S>,
    V: AsRef<[S]> + AsMut<[S]>,
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
                if a[(i, j)] != S::zero() {
                    level[i][j] = 0;
                }
            }
        }
        // Copy matrix to working array
        let mut a_work = vec![vec![S::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                a_work[i][j] = a[(i, j)];
            }
        }
        // Main ILU(p) factorization loop
        for i in 0..n {
            // Compute L(i,j) for j < i
            for j in 0..i {
                if a_work[i][j] != S::zero() && level[i][j] <= self.fill {
                    // Find U[j,j]
                    let u_jj = a_work[j][j];
                    if u_jj == S::zero() {
                        return Err(KError::SolveError(format!(
                            "ILUP: zero diagonal in U at row {j}"
                        )));
                    }
                    let lij = a_work[i][j] / u_jj;
                    self.l[i].cols.push(j);
                    self.l[i].vals.push(lij);
                    // Update fill levels and values for row i
                    for k in (j + 1)..n {
                        if a_work[j][k] != S::zero() {
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
                if a_work[i][k] != S::zero() && level[i][k] <= self.fill {
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
impl KPreconditioner for Ilup {
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
        _scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        self.apply_slice(side, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatShape;

    struct DenseMat {
        data: Vec<Vec<S>>,
    }
    impl DenseMat {
        fn new(data: Vec<Vec<S>>) -> Self {
            Self { data }
        }
    }
    impl MatShape for DenseMat {
        fn nrows(&self) -> usize {
            self.data.len()
        }
        fn ncols(&self) -> usize {
            self.data[0].len()
        }
    }
    impl std::ops::Index<(usize, usize)> for DenseMat {
        type Output = S;
        fn index(&self, idx: (usize, usize)) -> &Self::Output {
            &self.data[idx.0][idx.1]
        }
    }
    impl crate::core::traits::MatVec<Vec<S>> for DenseMat {
        fn matvec(&self, x: &Vec<S>, y: &mut Vec<S>) {
            for i in 0..self.nrows() {
                y[i] = (0..self.ncols())
                    .map(|j| self[(i, j)] * x[j])
                    .fold(S::zero(), |a, b| a + b);
            }
        }
    }

    #[test]
    fn ilup_identity() {
        type Mat = DenseMat;
        let a = Mat::new(vec![vec![S::one(), S::zero()], vec![S::zero(), S::one()]]);
        let mut pc = Ilup::new(0);
        pc.setup(&a).unwrap();
        let r = vec![S::from_real(2.0), S::from_real(3.0)];
        let mut z = vec![S::zero(); 2];
        Preconditioner::<Mat, Vec<S>>::apply(&pc, crate::preconditioner::PcSide::Left, &r, &mut z)
            .unwrap();
        assert!(
            (z[0] - S::from_real(2.0)).abs() < 1e-12 && (z[1] - S::from_real(3.0)).abs() < 1e-12
        );
    }

    #[test]
    fn ilup_tridiag() {
        type Mat = DenseMat;
        let a = Mat::new(vec![
            vec![S::from_real(2.0), S::from_real(-1.0), S::zero()],
            vec![S::from_real(-1.0), S::from_real(2.0), S::from_real(-1.0)],
            vec![S::zero(), S::from_real(-1.0), S::from_real(2.0)],
        ]);
        let mut pc = Ilup::new(0);
        pc.setup(&a).unwrap();
        let r = vec![S::from_real(1.0), S::from_real(2.0), S::from_real(3.0)];
        let mut z = vec![S::zero(); 3];
        Preconditioner::<Mat, Vec<S>>::apply(&pc, crate::preconditioner::PcSide::Left, &r, &mut z)
            .unwrap();
        assert!(z.iter().all(|&zi| zi.is_finite()));
    }

    #[cfg(feature = "complex")]
    #[test]
    fn apply_s_matches_real_path() {
        use crate::algebra::bridge::BridgeScratch;
        use crate::algebra::prelude::*;
        use crate::ops::kpc::KPreconditioner;

        type Mat = DenseMat;
        let a = Mat::new(vec![
            vec![S::from_real(4.0), S::from_real(1.0)],
            vec![S::from_real(1.5), S::from_real(3.5)],
        ]);
        let mut pc = Ilup::new(1);
        pc.setup(&a).unwrap();

        let rhs_real = vec![S::from_real(6.0), S::from_real(4.0)];
        let mut out_real = vec![S::zero(); rhs_real.len()];
        Preconditioner::<Mat, Vec<S>>::apply(
            &pc,
            crate::preconditioner::PcSide::Left,
            &rhs_real,
            &mut out_real,
        )
        .expect("ilup real apply");

        let rhs_s: Vec<S> = rhs_real.clone();
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
