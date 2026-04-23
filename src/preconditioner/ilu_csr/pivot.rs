use super::Real;
use crate::algebra::scalar::{KrystScalar, R};
use crate::error::KError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PivotStrategy {
    Strict,
    Threshold,
    DiagonalPerturbation,
}

#[inline]
pub fn handle_pivot(
    raw_pivot: Real,
    strategy: PivotStrategy,
    thr: R,
    diag_perturb_factor: R,
    max_diag_abs: R,
) -> Result<Real, KError> {
    handle_pivot_scalar(raw_pivot, strategy, thr, diag_perturb_factor, max_diag_abs)
}

#[inline]
pub fn handle_pivot_scalar<T: KrystScalar<Real = R>>(
    raw_pivot: T,
    strategy: PivotStrategy,
    thr: R,
    diag_perturb_factor: R,
    max_diag_abs: R,
) -> Result<T, KError> {
    match strategy {
        PivotStrategy::Strict => {
            if raw_pivot.abs() < thr {
                Err(KError::ZeroPivot(0))
            } else {
                Ok(raw_pivot)
            }
        }
        PivotStrategy::Threshold => {
            let abs = raw_pivot.abs();
            if abs < thr {
                Ok(rescale_to_magnitude(raw_pivot, thr))
            } else {
                Ok(raw_pivot)
            }
        }
        PivotStrategy::DiagonalPerturbation => {
            let abs = raw_pivot.abs();
            if abs < thr {
                let base = if max_diag_abs.is_finite() && max_diag_abs > 0.0 {
                    max_diag_abs
                } else {
                    1.0
                };
                let delta = (diag_perturb_factor.abs().max(thr.min(1.0) * 1e-12)) * base;
                let delta_s = T::from_real(delta);
                let direction = if abs > 0.0 {
                    raw_pivot / T::from_real(abs)
                } else {
                    T::one()
                };
                Ok(direction * delta_s)
            } else {
                Ok(raw_pivot)
            }
        }
    }
}

#[inline]
fn rescale_to_magnitude<T: KrystScalar<Real = R>>(pivot: T, magnitude: R) -> T {
    if magnitude == 0.0 {
        return T::zero();
    }
    let abs = pivot.abs();
    if abs == 0.0 {
        T::from_real(magnitude)
    } else {
        pivot * T::from_real(magnitude / abs)
    }
}
