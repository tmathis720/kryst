use crate::algebra::prelude::*;

#[inline]
pub fn take_or_resize(buf: &mut Vec<S>, n: usize) {
    if buf.len() != n {
        buf.resize(n, S::zero());
    }
}
