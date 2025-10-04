use crate::algebra::prelude::*;

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
        if self.xr.len() != n {
            self.xr.resize(n, 0.0);
        }
        if self.yr.len() != n {
            self.yr.resize(n, 0.0);
        }
        if self.xs.len() != n {
            self.xs.resize(n, S::zero());
        }
        if self.ys.len() != n {
            self.ys.resize(n, S::zero());
        }
    }

    #[inline]
    pub fn real_pair(&mut self, n: usize) -> (&mut [f64], &mut [f64]) {
        self.ensure(n);
        let xr = self.xr.as_mut_slice();
        let yr = self.yr.as_mut_slice();
        (&mut xr[..n], &mut yr[..n])
    }

    #[inline]
    pub fn scalar_pair(&mut self, n: usize) -> (&mut [S], &mut [S]) {
        self.ensure(n);
        let xs = self.xs.as_mut_slice();
        let ys = self.ys.as_mut_slice();
        (&mut xs[..n], &mut ys[..n])
    }
}
