use crate::algebra::prelude::*;
use crate::error::KError;

/// Column-major dense block vector storage used by block Krylov variants.
#[derive(Debug, Clone, Default)]
pub struct BlockVec {
    data: Vec<S>,
    n: usize,
    p: usize,
}

impl BlockVec {
    /// Create a new block vector with `n` rows and `p` columns.
    pub fn new(n: usize, p: usize) -> Self {
        Self {
            data: vec![S::zero(); n.saturating_mul(p)],
            n,
            p,
        }
    }

    /// Resize the block vector to `n` rows and `p` columns, zero-filling new entries.
    pub fn resize(&mut self, n: usize, p: usize) {
        if self.n != n || self.p != p {
            self.data.resize(n.saturating_mul(p), S::zero());
            self.n = n;
            self.p = p;
        } else {
            let needed = n.saturating_mul(p);
            if self.data.len() != needed {
                self.data.resize(needed, S::zero());
            }
        }
    }

    /// Number of rows in the block vector.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.n
    }

    /// Number of columns in the block vector.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.p
    }

    /// Immutable view into the `j`-th column.
    #[inline]
    pub fn col(&self, j: usize) -> &[S] {
        let offset = j * self.n;
        &self.data[offset..offset + self.n]
    }

    /// Mutable view into the `j`-th column.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [S] {
        let offset = j * self.n;
        &mut self.data[offset..offset + self.n]
    }

    /// Immutable view into the raw column-major storage.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }

    /// Mutable view into the raw column-major storage.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.data
    }
}

impl BlockVec {
    /// Fill the block vector with zeros.
    pub fn fill_zero(&mut self) {
        for v in &mut self.data {
            *v = S::zero();
        }
    }
}

/// Convenience helper for verifying block dimensions.
#[allow(dead_code)]
pub(crate) fn assert_block_dims(expected_rows: usize, vec: &BlockVec) -> Result<(), KError> {
    if vec.nrows() != expected_rows {
        return Err(KError::InvalidInput(format!(
            "BlockVec has {} rows but expected {}",
            vec.nrows(),
            expected_rows
        )));
    }
    Ok(())
}
