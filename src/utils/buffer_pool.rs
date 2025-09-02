use std::{cell::RefCell, mem::MaybeUninit};

/// Simple buffer pool that can be reused across calls to avoid allocations.
#[derive(Debug)]
pub struct BufferPool<T> {
    buf: RefCell<Vec<MaybeUninit<T>>>,
    len: usize,
}

impl<T> BufferPool<T> {
    /// Create a new pool with given capacity.
    pub fn with_capacity(n: usize) -> Self {
        Self { buf: RefCell::new(Vec::with_capacity(n)), len: 0 }
    }

    /// Ensure the internal buffer can hold at least `n` elements.
    pub fn ensure_len(&mut self, n: usize) {
        let mut b = self.buf.borrow_mut();
        if b.len() < n {
            b.resize_with(n, || MaybeUninit::uninit());
        }
        self.len = n;
    }

    /// Mutable slice view assuming caller initializes elements before reading.
    pub fn as_mut_slice_init(&self) -> &mut [T] {
        // SAFETY: Caller guarantees initialization of all elements before use.
        unsafe {
            &mut *(self.buf.borrow_mut().as_mut_slice() as *mut [MaybeUninit<T>] as *mut [T])
        }
    }

    /// Immutable slice view for already initialized data.
    pub fn as_slice(&self) -> &[T] {
        unsafe { &*(self.buf.borrow().as_slice() as *const [MaybeUninit<T>] as *const [T]) }
    }
}

impl<T> Default for BufferPool<T> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl<T> Clone for BufferPool<T> {
    fn clone(&self) -> Self {
        let len = self.buf.borrow().len();
        let mut vec = Vec::with_capacity(len);
        vec.resize_with(len, || MaybeUninit::uninit());
        Self { buf: RefCell::new(vec), len: self.len }
    }
}
