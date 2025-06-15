//! ILU(p) preconditioner stub

use crate::core::traits::MatShape;
use crate::error::KError;
use crate::preconditioner::Preconditioner;

#[derive(Clone)]
pub struct SparseRow<T> {
    pub cols: Vec<usize>,
    pub vals: Vec<T>,
}
impl<T> SparseRow<T> {
    pub fn new() -> Self {
        Self { cols: Vec::new(), vals: Vec::new() }
    }
}
impl<T> Default for SparseRow<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Ilup<T> {
    pub fill: usize,
    pub l: Vec<SparseRow<T>>,
    pub u: Vec<SparseRow<T>>,
    pub n: usize,
}

impl<T: num_traits::Float + Clone + std::fmt::Debug> Ilup<T> {
    pub fn new(fill: usize) -> Self {
        Self { fill, l: Vec::new(), u: Vec::new(), n: 0 }
    }
}

impl<M, V, T> Preconditioner<M, V> for Ilup<T>
where
    T: num_traits::Float + Clone + std::fmt::Debug + PartialOrd,
    M: crate::core::traits::MatVec<V> + MatShape + std::ops::Index<(usize, usize), Output = T>,
    V: AsRef<[T]> + AsMut<[T]>,
{
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        // Classical ILU(p) factorization (Saad §10.3)
        let n = a.nrows();
        self.n = n;
        self.l = vec![SparseRow::new(); n];
        self.u = vec![SparseRow::new(); n];
        // Level-of-fill bookkeeping
        let mut level = vec![vec![usize::MAX; n]; n];
        for i in 0..n {
            for j in 0..n {
                if !a[(i, j)].is_zero() {
                    level[i][j] = 0;
                }
            }
        }
        let mut a_work = vec![vec![T::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                a_work[i][j] = a[(i, j)];
            }
        }
        for i in 0..n {
            for j in 0..i {
                if !a_work[i][j].is_zero() && level[i][j] <= self.fill {
                    // Find U[j,j]
                    let u_jj = a_work[j][j];
                    if u_jj.is_zero() {
                        return Err(KError::SolveError(format!("ILUP: zero diagonal in U at row {}", j)));
                    }
                    let lij = a_work[i][j] / u_jj;
                    self.l[i].cols.push(j);
                    self.l[i].vals.push(lij);
                    for k in (j+1)..n {
                        if !a_work[j][k].is_zero() {
                            let new_level = level[i][j].saturating_add(level[j][k]).saturating_add(1);
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
    fn apply(&self, r: &V, z: &mut V) -> Result<(), KError> {
        let n = self.n;
        let r = r.as_ref();
        let z = z.as_mut();
        let mut y = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = r[i];
            for (j_idx, &j) in self.l[i].cols.iter().enumerate() {
                sum = sum - self.l[i].vals[j_idx] * y[j];
            }
            y[i] = sum;
        }
        // Backward substitution (sequential)
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
    fn ilup_identity() {
        type Mat = DenseMat<f64>;
        let a = Mat::new(vec![vec![1.0f64, 0.0], vec![0.0, 1.0]]);
        let mut pc: Ilup<f64> = Ilup::new(0);
        pc.setup(&a).unwrap();
        let r = vec![2.0f64, 3.0];
        let mut z = vec![0.0; 2];
        Preconditioner::<Mat, Vec<f64>>::apply(&pc, &r, &mut z).unwrap();
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
        Preconditioner::<Mat, Vec<f64>>::apply(&pc, &r, &mut z).unwrap();
        assert!(z.iter().all(|&zi| zi.is_finite()));
    }
}
