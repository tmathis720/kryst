use super::scalar::{RealScalar, Scalar};

/// Compute the dot product \(x^\mathrm{H} y\).
#[inline]
pub fn dot<S>(x: &[S], y: &[S]) -> S
where
    S: Scalar,
{
    let mut acc = S::zero();
    for i in 0..x.len() {
        acc = acc + x[i].conj() * y[i];
    }
    acc
}

/// Compute the Euclidean norm of a vector.
#[inline]
pub fn nrm2<S>(x: &[S]) -> <S as Scalar>::Real
where
    S: Scalar,
{
    dot(x, x).real().sqrt()
}
