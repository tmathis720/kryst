use crate::algebra::parallel::par_sum_abs2_local;
#[allow(unused_imports)]
use crate::algebra::prelude::*;

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
    // Cheaper for complex scalars: avoids a complex multiply per element.
    par_sum_abs2_local(x).sqrt()
}
