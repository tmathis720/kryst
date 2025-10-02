use crate::algebra::scalar::{KrystScalar, R, S};

#[inline]
pub fn dot_conj(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = S::zero();
    for i in 0..x.len() {
        acc = x[i].conj().mul_add(y[i], acc);
    }
    acc
}

#[inline]
pub fn nrm2(x: &[S]) -> R {
    dot_conj(x, x).abs().sqrt()
}
