use std::mem::MaybeUninit;

/// Simple buffer pool that can be reused across calls to avoid allocations.
#[derive(Debug)]
pub struct BufferPool<T> {
    buf: Vec<MaybeUninit<T>>,
    len: usize,
}

impl<T> BufferPool<T> {
    /// Create a new pool with zero capacity.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a new pool with the given capacity.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            buf: Vec::with_capacity(n),
            len: 0,
        }
    }

    /// Returns the number of elements that the caller requested to populate.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the total reserved space in the pool.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Ensure the internal buffer can hold at least `n` elements.
    pub fn ensure_len(&mut self, n: usize) {
        if self.buf.len() < n {
            self.buf.resize_with(n, MaybeUninit::uninit);
        }
        self.len = n;
    }

    /// Gives mutable access to the uninitialized portion of the buffer.
    ///
    /// # Safety
    ///
    /// The caller must fill the first `self.len` entries before any call to
    /// [`initialized_slice`] or [`initialized_slice_mut`].
    pub unsafe fn uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
        &mut self.buf[..self.len]
    }

    /// Access the initialized slice.
    ///
    /// # Safety
    ///
    /// The caller must guarantee the first `self.len` entries are fully initialized.
    pub unsafe fn initialized_slice(&self) -> &[T] {
        let ptr = self.buf.as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Mutable access to the initialized slice.
    ///
    /// # Safety
    ///
    /// The caller must guarantee the first `self.len` entries are fully initialized.
    pub unsafe fn initialized_slice_mut(&mut self) -> &mut [T] {
        let ptr = self.buf.as_mut_ptr() as *mut T;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Mutable access for initializing, returning the raw uninitialized bytes.
    pub fn as_mut_slice_init(&mut self) -> &mut [MaybeUninit<T>] {
        &mut self.buf[..self.len]
    }

    /// Immutable view on the raw bytes previously validated as initialized.
    ///
    /// # Safety
    ///
    /// Use only if the caller has completed initialization for the first `self.len`
    /// entries.
    pub unsafe fn as_slice(&self) -> &[T] {
        unsafe { self.initialized_slice() }
    }
}

impl<T> Default for BufferPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for BufferPool<T> {
    fn clone(&self) -> Self {
        let mut buf = Vec::with_capacity(self.buf.len());
        buf.resize_with(self.buf.len(), MaybeUninit::uninit);
        Self { buf, len: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_pool_ensure_len_allows_initialization() {
        let mut pool = BufferPool::<u8>::with_capacity(2);
        pool.ensure_len(4);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.len(), 4);

        unsafe {
            let bytes = pool.uninit_slice_mut();
            for (idx, slot) in bytes.iter_mut().enumerate() {
                slot.write(idx as u8);
            }
            let filled = pool.initialized_slice();
            assert_eq!(filled, &[0, 1, 2, 3]);
        }
    }

    #[test]
    fn buffer_pool_clone_preserves_capacity_but_not_len() {
        let mut pool = BufferPool::<u8>::with_capacity(3);
        pool.ensure_len(2);
        let clone = pool.clone();
        assert_eq!(clone.capacity(), 3);
        assert_eq!(clone.len(), 0);
    }
}
