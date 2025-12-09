//! Row-filtering preconditioner with threshold and fill-in control.
//!
//! This streams each row, drops entries below a magnitude threshold, keeps at most `fill`
//! entries, and splits the remaining entries into L (j < i) and U (j >= i) parts.
//! It does **not** perform an ILUT elimination.
//! It is implemented generically over the Kryst scalar `S`, so complex systems can call it
//! without requiring a real-to-complex bridge.
//!
//! For a true ILUT factorization, use `Ilu` with `IluType::ILUT`.

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
use crate::algebra::prelude::*;
use crate::core::traits::MatShape;
use crate::error::KError;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::{LocalPreconditioner, legacy::Preconditioner};
use std::sync::Mutex;

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

/// Workspace reused across RowFilterPreconditioner solves.
#[derive(Debug)]
pub struct RowFilterWorkspace {
    buf: Mutex<Vec<S>>,
    size: usize,
}

impl RowFilterWorkspace {
    pub fn new() -> Self {
        Self {
            buf: Mutex::new(Vec::new()),
            size: 0,
        }
    }

    pub fn ensure_size(&mut self, n: usize) {
        if n > self.size {
            let mut guard = self.buf.lock().unwrap();
            guard.resize(n, S::zero());
            self.size = n;
        }
    }

    #[inline]
    pub fn borrow_buf(&self, n: usize) -> std::sync::MutexGuard<'_, Vec<S>> {
        debug_assert!(self.size >= n, "workspace not sized via setup()");
        self.buf.lock().unwrap()
    }
}

/// Row-filtering preconditioner struct.
///
/// This preconditioner performs threshold-based dropping and row splitting without elimination.
/// - `fill`: Maximum number of nonzeros per row (fill-in control)
/// - `droptol`: Drop tolerance (numerical dropping)
/// - `l`: Lower triangular portion (sparse row format)
/// - `u`: Upper triangular portion (sparse row format)
/// - `n`: Matrix size
///
/// Implements [`LocalPreconditioner`] for use as a purely local block preconditioner that
/// assumes no MPI communication.
pub struct RowFilterPreconditioner {
    pub fill: usize,
    pub droptol: R,
    pub l: Vec<SparseRow>,
    pub u: Vec<SparseRow>,
    pub n: usize,
    workspace: RowFilterWorkspace,
}

/// Deprecated name; this type is *not* a true ILUT factorization.
/// Use `RowFilterPreconditioner` or `Ilu` with `IluType::ILUT` for a real ILUT factorization.
#[deprecated(
    note = "Ilut here is not a true ILUT factorization. Use Ilu (IluType::ILUT) or RowFilterPreconditioner instead."
)]
pub type Ilut = RowFilterPreconditioner;

impl RowFilterPreconditioner {
    /// Create a new ILUT preconditioner with fill and drop tolerance.
    pub fn new(fill: usize, droptol: R) -> Self {
        Self {
            fill,
            droptol,
            l: Vec::new(),
            u: Vec::new(),
            n: 0,
            workspace: RowFilterWorkspace::new(),
        }
    }
}

impl RowFilterPreconditioner {
    fn apply_slice(
        &self,
        _side: crate::preconditioner::PcSide,
        r: &[S],
        z: &mut [S],
    ) -> Result<(), KError> {
        let n = self.n;
        if r.len() != n || z.len() != n {
            return Err(KError::InvalidInput(format!(
                "RowFilterPreconditioner::apply dimension mismatch: n={}, r.len()={}, z.len()={}",
                n,
                r.len(),
                z.len()
            )));
        }

        let mut y_guard = self.workspace.borrow_buf(n);
        let y = &mut y_guard[..n];
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

impl<M, V> Preconditioner<M, V> for RowFilterPreconditioner
where
    M: crate::core::traits::MatVec<V> + MatShape + std::ops::Index<(usize, usize), Output = S>,
    V: AsRef<[S]> + AsMut<[S]>,
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
                if val != S::zero() {
                    row.push((j, val));
                }
            }
            // Apply dropping by magnitude (ILUT)
            row.retain(|&(_, v)| v.abs() >= self.droptol);
            // Keep only largest 'fill' entries by magnitude
            if row.len() > self.fill {
                row.sort_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
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
        self.workspace.ensure_size(n);
        Ok(())
    }
    /// Apply ILUT preconditioner: solve Ly = r, then Uz = y.
    ///
    /// Forward substitution for L, then backward substitution for U.
    fn apply(&self, side: crate::preconditioner::PcSide, r: &V, z: &mut V) -> Result<(), KError> {
        self.apply_slice(side, r.as_ref(), z.as_mut())
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for RowFilterPreconditioner {
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

impl LocalPreconditioner for RowFilterPreconditioner {
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply_local(&self, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let (n, _) = LocalPreconditioner::<S>::dims(self);
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        self.apply_slice(crate::preconditioner::PcSide::Left, x, y)
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
    impl crate::core::traits::MatVec<Vec<S>> for DenseMat<S> {
        fn matvec(&self, x: &Vec<S>, y: &mut Vec<S>) {
            for i in 0..self.nrows() {
                y[i] = (0..self.ncols())
                    .map(|j| self[(i, j)] * x[j])
                    .fold(S::zero(), |a, b| a + b);
            }
        }
    }

    #[test]
    fn ilut_identity() {
        type Mat = DenseMat<S>;
        let a = Mat::new(vec![
            vec![S::from_real(1.0), S::zero()],
            vec![S::zero(), S::from_real(1.0)],
        ]);
        let mut pc = RowFilterPreconditioner::new(2, R::from_real(1e-12));
        pc.setup(&a).unwrap();
        let r = vec![S::from_real(2.0), S::from_real(3.0)];
        let mut z = vec![S::zero(); 2];
        Preconditioner::<Mat, Vec<S>>::apply(&pc, crate::preconditioner::PcSide::Left, &r, &mut z)
            .unwrap();
        assert!(
            (z[0] - S::from_real(2.0)).abs() < R::from_real(1e-12)
                && (z[1] - S::from_real(3.0)).abs() < R::from_real(1e-12)
        );
    }

    #[test]
    fn ilut_tridiag() {
        type Mat = DenseMat<S>;
        let a = Mat::new(vec![
            vec![S::from_real(2.0), S::from_real(-1.0), S::zero()],
            vec![S::from_real(-1.0), S::from_real(2.0), S::from_real(-1.0)],
            vec![S::zero(), S::from_real(-1.0), S::from_real(2.0)],
        ]);
        let mut pc = RowFilterPreconditioner::new(3, R::from_real(1e-12));
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

        type Mat = DenseMat<S>;
        let a = Mat::new(vec![
            vec![S::from_real(4.0), S::from_real(1.0)],
            vec![S::from_real(2.0), S::from_real(3.0)],
        ]);
        let mut pc = RowFilterPreconditioner::new(2, R::from_real(1e-9));
        pc.setup(&a).unwrap();

        let rhs_real = vec![S::from_real(5.0), S::from_real(7.0)];
        let mut out_real = vec![S::zero(); rhs_real.len()];
        Preconditioner::<Mat, Vec<S>>::apply(
            &pc,
            crate::preconditioner::PcSide::Left,
            &rhs_real,
            &mut out_real,
        )
        .expect("ilut real apply");

        let rhs_s: Vec<S> = rhs_real.clone();
        let mut out_s = vec![S::zero(); rhs_s.len()];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(
            crate::preconditioner::PcSide::Left,
            &rhs_s,
            &mut out_s,
            &mut scratch,
        )
        .expect("ilut apply_s");

        for (ys, yr) in out_s.iter().zip(out_real.iter()) {
            assert!((*ys - *yr).abs() < R::from_real(1e-10));
        }
    }
}
