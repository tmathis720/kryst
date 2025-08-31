use crate::error::KError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PivotStrategy {
    Strict,
    Threshold,
    DiagonalPerturbation,
}

#[inline]
pub fn handle_pivot(
    raw_pivot: f64,
    strategy: PivotStrategy,
    thr: f64,
    diag_perturb_factor: f64,
    max_diag_abs: f64,
) -> Result<f64, KError> {
    match strategy {
        PivotStrategy::Strict => {
            if raw_pivot.abs() < thr {
                Err(KError::ZeroPivot(0))
            } else {
                Ok(raw_pivot)
            }
        }
        PivotStrategy::Threshold => {
            if raw_pivot.abs() < thr {
                Ok(if raw_pivot.is_sign_negative() { -thr } else { thr })
            } else {
                Ok(raw_pivot)
            }
        }
        PivotStrategy::DiagonalPerturbation => {
            if raw_pivot.abs() < thr {
                let base = if max_diag_abs.is_finite() && max_diag_abs > 0.0 {
                    max_diag_abs
                } else {
                    1.0
                };
                let delta = (diag_perturb_factor.abs().max(thr.min(1.0) * 1e-12)) * base;
                Ok(if raw_pivot.is_sign_negative() { -delta } else { delta })
            } else {
                Ok(raw_pivot)
            }
        }
    }
}

