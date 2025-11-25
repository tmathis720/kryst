#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
use faer::traits::{from_real as cf_from_real, mul_real as cf_mul_real, ComplexField};
use num_traits::{Float, Zero};

/// How to fix "too-small" pivots during ILU numerics.
#[derive(Clone, Debug)]
pub struct PivotPolicy {
    pub mode: PivotMode,
    pub tau: f64,
    pub scale: PivotScale,
    pub sign: PivotSignPolicy,
    pub deterministic: bool,
}

impl Default for PivotPolicy {
    fn default() -> Self {
        Self {
            mode: PivotMode::AdditiveFloor,
            tau: 1e-12,
            scale: PivotScale::RowGershgorin,
            sign: PivotSignPolicy::Positive,
            deterministic: true,
        }
    }
}

/// The action to take when |u_ii| is below the floor.
#[derive(Clone, Copy, Debug)]
pub enum PivotMode {
    /// Error out (legacy Strict).
    Strict,
    /// Add an additive shift to reach the floor (recommended).
    AdditiveFloor,
    /// Clamp toward the floor by minimum additive change (alias of AdditiveFloor).
    ThresholdFloor,
    /// (Optional later) ILUTP pivoting path will hook here.
    PivotingAllowed,
}

/// Scaling for the floor: floor_i = tau * scale_i
#[derive(Clone, Copy, Debug)]
pub enum PivotScale {
    /// s_i = max_j |A_jj| over the input (precomputed once).
    MaxDiagA,
    /// s_i = |A_ii| (local diag magnitude).
    LocalDiagA,
    /// s_i = ||A_{i,:}||_inf  (row inf-norm, precompute once).
    RowInfA,
    /// s_i = |A_ii| + sum_{j!=i} |A_ij|  (Gershgorin-based).
    RowGershgorin,
    /// s_i = running max of |U_kk| for k <= i (fill-growth aware).
    RunningMaxU,
}

/// Sign handling
#[derive(Clone, Copy, Debug)]
pub enum PivotSignPolicy {
    /// Keep current sign(u_ii); if zero, treat as +.
    Preserve,
    /// Force positive (good default for SPD or when using CG).
    Positive,
}

/// Statistics gathered during pivot handling.
#[derive(Clone, Debug)]
pub struct PivotStats {
    pub num_floors: usize,
    pub max_abs_shift: f64,
    pub sum_abs_shift: f64,
    pub num_strict_fail: usize,
    pub last_floor_value: f64,
}

impl Default for PivotStats {
    fn default() -> Self {
        Self {
            num_floors: 0,
            max_abs_shift: 0.0,
            sum_abs_shift: 0.0,
            num_strict_fail: 0,
            last_floor_value: 0.0,
        }
    }
}

fn record_pivot_shift<T>(stats: &mut PivotStats, old: T, new: T)
where
    T: ComplexField,
    T::Real: Float,
{
    let sh = (new - old).abs().to_f64().unwrap_or(0.0);
    stats.num_floors += 1;
    if sh > stats.max_abs_shift {
        stats.max_abs_shift = sh;
    }
    stats.sum_abs_shift += sh;
    stats.last_floor_value = new.abs().to_f64().unwrap_or(0.0);
}

/// Apply minimal additive shift to stabilize pivot.
pub fn stabilize_pivot_in_place<T>(
    u_ii: &mut T,
    s_i: T::Real,
    tau: T::Real,
    sign_policy: PivotSignPolicy,
    mode: PivotMode,
    stats: &mut PivotStats,
    row: usize,
) -> Result<(), KError>
where
    T: ComplexField + Copy,
    T::Real: Float,
{
    let floor = tau * s_i;
    let abs = u_ii.abs();

    match mode {
        PivotMode::Strict => {
            if !(abs >= floor) {
                stats.num_strict_fail += 1;
                stats.last_floor_value = floor.to_f64().unwrap_or(0.0);
                return Err(KError::ZeroPivot(row));
            }
        }
        PivotMode::AdditiveFloor | PivotMode::ThresholdFloor => {
            if abs >= floor {
                return Ok(());
            }
            // Determine target magnitude
            let new_value = match sign_policy {
                PivotSignPolicy::Preserve => {
                    if abs > T::Real::zero() {
                        let scale = floor / abs;
                        cf_mul_real(u_ii, &scale)
                    } else {
                        cf_from_real(&floor)
                    }
                }
                PivotSignPolicy::Positive => cf_from_real(&floor),
            };
            let old = *u_ii;
            *u_ii = new_value;
            record_pivot_shift(stats, old, *u_ii);
        }
        PivotMode::PivotingAllowed => {
            unimplemented!("Pivoting path not yet implemented");
        }
    }
    Ok(())
}
