use std::usize;

/// Position map for tracking column indices within a row during
/// sparse factorization.
///
/// This structure uses epoch-based marking to provide O(1) inserts
/// and lookups without clearing the entire map for each row.
#[derive(Clone, Debug)]
pub struct PosMap {
    pos: Vec<usize>,
    mark: Vec<u32>,
    epoch: u32,
}

impl PosMap {
    /// Create a new `PosMap` with capacity for `n` columns.
    pub fn new(n: usize) -> Self {
        Self {
            pos: vec![usize::MAX; n],
            mark: vec![0; n],
            epoch: 1,
        }
    }

    /// Advance to a new epoch, logically clearing the map.
    #[inline]
    pub fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Get the position associated with column `j` if it exists in the
    /// current epoch.
    #[inline]
    pub fn get(&self, j: usize) -> Option<usize> {
        if self.mark.get(j).copied().unwrap_or(0) == self.epoch {
            Some(self.pos[j])
        } else {
            None
        }
    }

    /// Set the position for column `j` to `idx` in the current epoch.
    #[inline]
    pub fn set(&mut self, j: usize, idx: usize) {
        if j >= self.pos.len() {
            let new_len = j + 1;
            self.pos.resize(new_len, usize::MAX);
            self.mark.resize(new_len, 0);
        }
        self.pos[j] = idx;
        self.mark[j] = self.epoch;
    }
}

#[cfg(test)]
mod tests {
    use super::PosMap;

    #[test]
    fn pos_map_basic() {
        let mut pm = PosMap::new(5);
        assert_eq!(pm.get(2), None);
        pm.set(2, 7);
        assert_eq!(pm.get(2), Some(7));
        pm.clear();
        assert_eq!(pm.get(2), None);
        pm.set(2, 3);
        assert_eq!(pm.get(2), Some(3));
    }
}
