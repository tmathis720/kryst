use crate::algebra::prelude::*;
use crate::algebra::scalar::{copy_real_to_scalar_in, copy_scalar_to_real_in};

/// Temporary buffers reused by solver bridges when converting between `S` and `f64`.
#[derive(Default, Clone, Debug)]
pub struct BridgeScratch {
    xr: Vec<f64>,
    yr: Vec<f64>,
    xs: Vec<S>,
    ys: Vec<S>,
}

impl BridgeScratch {
    fn ensure(&mut self, n: usize) {
        if self.xr.len() < n {
            self.xr.resize(n, 0.0);
        }
        if self.yr.len() < n {
            self.yr.resize(n, 0.0);
        }
        if self.xs.len() < n {
            self.xs.resize(n, S::zero());
        }
        if self.ys.len() < n {
            self.ys.resize(n, S::zero());
        }
    }

    #[inline]
    pub fn xr(&mut self, n: usize) -> &mut [f64] {
        self.ensure(n);
        &mut self.xr[..n]
    }

    #[inline]
    pub fn yr(&mut self, n: usize) -> &mut [f64] {
        self.ensure(n);
        &mut self.yr[..n]
    }

    #[inline]
    pub fn xs(&mut self, n: usize) -> &mut [S] {
        self.ensure(n);
        &mut self.xs[..n]
    }

    #[inline]
    pub fn ys(&mut self, n: usize) -> &mut [S] {
        self.ensure(n);
        &mut self.ys[..n]
    }

    #[inline]
    pub fn copy_scalar_into_real(&mut self, src: &[S]) -> &mut [f64] {
        let n = src.len();
        let dst = self.xr(n);
        copy_scalar_to_real_in(src, dst);
        dst
    }

    #[inline]
    pub fn copy_real_into_scalar(&mut self, src: &[f64], dst: &mut [S]) {
        copy_real_to_scalar_in(src, dst);
    }
}
