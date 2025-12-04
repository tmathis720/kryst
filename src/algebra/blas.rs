use crate::algebra::parallel::{par_dot_conj_local, par_sum_abs2_local};
#[allow(unused_imports)]
use crate::algebra::prelude::*;

#[inline]
pub fn dot_conj(x: &[S], y: &[S]) -> S {
    par_dot_conj_local(x, y)
}

#[inline]
pub fn nrm2(x: &[S]) -> R {
    // Cheaper for complex scalars: avoids a complex multiply per element.
    par_sum_abs2_local(x).sqrt()
}
