// Wrappers for faer::Mat, faer::MatRef, sparse types

use crate::core::traits::{Indexing, InnerProduct, MatTransVec, MatVec};
use faer::{Mat, MatRef};
use num_traits::{Float, FromPrimitive};

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

impl<T: Float + From<f64> + Send + Sync> InnerProduct<Vec<T>> for () {
    type Scalar = T;
    fn dot(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        assert_eq!(x.len(), y.len());
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            x.as_slice()
                .par_iter()
                .zip(y.as_slice().par_iter())
                .map(|(xi, yi)| *xi * *yi)
                .reduce(|| T::zero(), |acc, v| acc + v)
        }
        #[cfg(not(feature = "rayon"))]
        {
            x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| *xi * *yi)
                .fold(T::zero(), |acc, v| acc + v)
        }
    }
    fn norm(&self, x: &Vec<T>) -> T {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            x.as_slice()
                .par_iter()
                .map(|xi| *xi * *xi)
                .reduce(|| T::zero(), |acc, v| acc + v)
                .sqrt()
        }
        #[cfg(not(feature = "rayon"))]
        {
            x.iter()
                .map(|xi| *xi * *xi)
                .fold(T::zero(), |acc, v| acc + v)
                .sqrt()
        }
    }
}

#[cfg(feature = "mpi")]
pub struct DistributedInnerProduct<'a, C: crate::parallel::Comm> {
    pub comm: &'a C,
}

#[cfg(feature = "mpi")]
impl<'a, C: crate::parallel::Comm> DistributedInnerProduct<'a, C> {
    pub fn dot<T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero>(&self, x: &[T], y: &[T]) -> T {
        assert_eq!(x.len(), y.len());
        let local: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| (a * b).to_f64().unwrap_or(0.0)).sum();
        let global = self.comm.all_reduce(local);
        T::from_f64(global).unwrap_or(T::zero())
    }
    pub fn norm<T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::FromPrimitive + num_traits::ToPrimitive + num_traits::Zero + num_traits::Float>(&self, x: &[T]) -> T {
        let local: f64 = x.iter().map(|&a| (a * a).to_f64().unwrap_or(0.0)).sum();
        let global = self.comm.all_reduce(local);
        T::from_f64(global.sqrt()).unwrap_or(T::zero())
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
