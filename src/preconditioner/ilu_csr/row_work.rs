use super::Real;
use crate::algebra::scalar::KrystScalar;

/// Sparse row work array using epoch marking.
#[derive(Clone, Debug)]
pub struct RowWork {
    epoch: usize,
    mark: Vec<usize>,
    val: Vec<Real>,
    idx: Vec<usize>,
}

impl RowWork {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            mark: Vec::new(),
            val: Vec::new(),
            idx: Vec::new(),
        }
    }

    pub fn ensure_size(&mut self, n: usize) {
        if self.mark.len() < n {
            self.mark.resize(n, 0);
            self.val.resize(n, Real::zero());
        }
    }

    pub fn clear_row(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.idx.clear();
    }

    #[inline]
    pub fn get(&self, j: usize) -> Real {
        if self.mark.get(j).copied().unwrap_or(0) == self.epoch {
            self.val[j]
        } else {
            Real::zero()
        }
    }

    #[inline]
    pub fn set(&mut self, j: usize, x: Real) {
        if self.mark.get(j).copied().unwrap_or(0) != self.epoch {
            self.mark[j] = self.epoch;
            self.idx.push(j);
        }
        self.val[j] = x;
    }

    #[inline]
    pub fn add_to(&mut self, j: usize, delta: Real) {
        if self.mark.get(j).copied().unwrap_or(0) != self.epoch {
            self.mark[j] = self.epoch;
            self.val[j] = delta;
            self.idx.push(j);
        } else {
            self.val[j] += delta;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, Real)> + '_ {
        self.idx.iter().copied().map(|j| (j, self.val[j]))
    }
}
