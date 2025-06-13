//! ILU(0) factorization with zero fill (Saad §10.3).

use crate::preconditioner::Preconditioner;
use crate::error::KError;
use num_traits::Float;
use faer::traits::ComplexField;
use faer::Mat;

pub struct Ilu0<T> {
    pub(crate) l: Mat<T>,
    pub(crate) u: Mat<T>,
}

impl<T: Float + Send + Sync + ComplexField> Ilu0<T> {
    pub fn new() -> Self {
        Self {
            l: Mat::zeros(0, 0),
            u: Mat::zeros(0, 0),
        }
    }
}

impl<T: Float + Send + Sync + ComplexField> Preconditioner<Mat<T>, Vec<T>> for Ilu0<T> {
    fn setup(&mut self, a: &Mat<T>) -> Result<(), KError> {
        let (n, _) = (a.nrows(), a.ncols());
        let mut l = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);

        for i in 0..n {
            u[(i, i)] = a[(i, i)];
            for j in (i+1)..n {
                if a[(i, j)] != T::zero() {
                    u[(i, j)] = a[(i, j)];
                }
            }
            l[(i, i)] = T::one();
            for j in (i+1)..n {
                if a[(j, i)] != T::zero() {
                    l[(j, i)] = a[(j, i)] / u[(i, i)];
                }
            }
            for j in (i+1)..n {
                for k in (i+1)..n {
                    if a[(j, k)] != T::zero() {
                        let v = a[(j, k)] - l[(j, i)] * u[(i, k)];
                        if v != T::zero() {
                            if k >= j {
                                u[(j, k)] = v;
                            } else {
                                l[(j, k)] = v;
                            }
                        }
                    }
                }
            }
        }
        self.l = l;
        self.u = u;
        Ok(())
    }

    fn apply(&self, x: &Vec<T>, y: &mut Vec<T>) -> Result<(), KError> {
        // solve L y1 = x
        let mut y1 = x.clone();
        let n = x.len();
        for i in 0..n {
            for j in 0..i {
                y1[i] = y1[i] - self.l[(i, j)] * y1[j];
            }
        }
        // solve U y = y1
        for i in (0..n).rev() {
            for j in (i+1)..n {
                y1[i] = y1[i] - self.u[(i, j)] * y1[j];
            }
            y1[i] = y1[i] / self.u[(i, i)];
        }
        *y = y1;
        Ok(())
    }
}
