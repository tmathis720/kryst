#![cfg(feature = "backend-faer")]

use crate::core::traits::MatVecOp;

impl MatVecOp<f64> for faer::Mat<f64> {
    fn mat_vec(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".into(),
            ));
        }

        if beta == 0.0 {
            for value in y.iter_mut() {
                *value = 0.0;
            }
        } else if beta != 1.0 {
            for value in y.iter_mut() {
                *value *= beta;
            }
        }

        let m = self.nrows();
        let n = self.ncols();
        for i in 0..m {
            let mut acc = 0.0;
            for j in 0..n {
                acc = f64::mul_add(self[(i, j)], x[j], acc);
            }
            y[i] += alpha * acc;
        }
        Ok(())
    }

    fn mat_vec_trans(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.nrows() || y.len() != self.ncols() {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".into(),
            ));
        }

        if beta == 0.0 {
            for value in y.iter_mut() {
                *value = 0.0;
            }
        } else if beta != 1.0 {
            for value in y.iter_mut() {
                *value *= beta;
            }
        }

        let m = self.nrows();
        let n = self.ncols();
        for j in 0..n {
            let mut acc = 0.0;
            for i in 0..m {
                acc = f64::mul_add(self[(i, j)], x[i], acc);
            }
            y[j] += alpha * acc;
        }
        Ok(())
    }

    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }
}
