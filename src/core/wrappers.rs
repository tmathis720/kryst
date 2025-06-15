// Wrappers for faer::Mat, faer::MatRef, sparse types

use crate::core::traits::{Indexing, InnerProduct, MatTransVec, MatVec};
use faer::{Mat, MatRef};
use num_traits::Float;

impl<T: Float> MatVec<Vec<T>> for Mat<T> {
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.nrows(), y.len());
        assert_eq!(self.ncols(), x.len());
        for i in 0..self.nrows() {
            y[i] = T::zero();
            for j in 0..self.ncols() {
                y[i] = y[i] + self[(i, j)] * x[j];
            }
        }
    }
}

impl<'a, T: Float> MatVec<Vec<T>> for MatRef<'a, T> {
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.nrows(), y.len());
        assert_eq!(self.ncols(), x.len());
        for i in 0..self.nrows() {
            y[i] = T::zero();
            for j in 0..self.ncols() {
                y[i] = y[i] + self[(i, j)] * x[j];
            }
        }
    }
}

impl<T: Float> MatTransVec<Vec<T>> for Mat<T> {
    fn mattransvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.ncols(), y.len());
        assert_eq!(self.nrows(), x.len());
        for j in 0..self.ncols() {
            y[j] = T::zero();
            for i in 0..self.nrows() {
                y[j] = y[j] + self[(i, j)] * x[i];
            }
        }
    }
}

impl<'a, T: Float> MatTransVec<Vec<T>> for MatRef<'a, T> {
    fn mattransvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.ncols(), y.len());
        assert_eq!(self.nrows(), x.len());
        for j in 0..self.ncols() {
            y[j] = T::zero();
            for i in 0..self.nrows() {
                y[j] = y[j] + self[(i, j)] * x[i];
            }
        }
    }
}

impl<T: Float + From<f64>> InnerProduct<Vec<T>> for () {
    type Scalar = T;
    fn dot(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| *xi * *yi)
            .fold(T::zero(), |acc, v| acc + v)
    }
    fn norm(&self, x: &Vec<T>) -> T {
        x.iter()
            .map(|xi| *xi * *xi)
            .fold(T::zero(), |acc, v| acc + v)
            .sqrt()
    }
}

impl<T> Indexing for Vec<T> {
    fn nrows(&self) -> usize {
        self.len()
    }
}

impl<T> Indexing for Mat<T> {
    fn nrows(&self) -> usize {
        self.nrows()
    }
}
