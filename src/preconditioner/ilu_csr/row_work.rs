use num_traits::Zero;

/// Sparse row work array using epoch marking.
#[derive(Clone, Debug)]
pub struct RowWork<T> {
    epoch: usize,
    mark: Vec<usize>,
    val: Vec<T>,
    idx: Vec<usize>,
}

impl<T: Zero + Copy> RowWork<T> {
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
            self.val.resize(n, T::zero());
        }
    }

    pub fn clear_row(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.idx.clear();
    }

    #[inline]
    pub fn get(&self, j: usize) -> T {
        if self.mark.get(j).copied().unwrap_or(0) == self.epoch {
            self.val[j]
        } else {
            T::zero()
        }
    }

    #[inline]
    pub fn set(&mut self, j: usize, x: T) {
        if self.mark.get(j).copied().unwrap_or(0) != self.epoch {
            self.mark[j] = self.epoch;
            self.idx.push(j);
        }
        self.val[j] = x;
    }

    #[inline]
    pub fn add_to(&mut self, j: usize, delta: T)
    where
        T: std::ops::Add<Output = T>,
    {
        if self.mark.get(j).copied().unwrap_or(0) != self.epoch {
            self.mark[j] = self.epoch;
            self.val[j] = delta;
            self.idx.push(j);
        } else {
            self.val[j] = self.val[j] + delta;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, T)> + '_ {
        self.idx.iter().copied().map(|j| (j, self.val[j]))
    }
}
