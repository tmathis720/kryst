/// Parameters controlling ILUT factorization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IlutParams {
    /// Absolute drop tolerance.
    pub droptol_abs: f64,
    /// Relative drop tolerance multiplier (times row norm).
    pub droptol_rel: f64,
    /// Maximum number of L entries per row (strictly lower).
    pub p_l: usize,
    /// Maximum number of U entries per row (strictly upper, excluding diag).
    pub p_u: usize,
    /// Whether to drop during elimination.
    pub early_drop: bool,
    /// Pivot policy for guarding small diagonals.
    pub pivot: PivotPolicy,
    /// Threshold used by pivot policy.
    pub pivot_tau: f64,
    /// Deterministic ordering of kept entries.
    pub reproducible_order: bool,
    /// Placeholder for future pivoting strategies.
    pub pivoting: Pivoting,
}

impl Default for IlutParams {
    fn default() -> Self {
        Self {
            droptol_abs: 0.0,
            droptol_rel: 0.0,
            p_l: 0,
            p_u: 0,
            early_drop: true,
            pivot: PivotPolicy::DiagonalPerturbation,
            pivot_tau: 1e-12,
            reproducible_order: true,
            pivoting: Pivoting::None,
        }
    }
}

/// Pivot policy for ILUT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PivotPolicy {
    /// Return an error if pivot below threshold.
    Strict,
    /// Clamp pivot magnitude to threshold if too small.
    Threshold,
    /// Add threshold to diagonal when too small.
    DiagonalPerturbation,
}

/// Placeholder for future pivoting strategies (e.g., ILUTP).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pivoting {
    None,
    #[allow(dead_code)]
    ILUTP {
        band: usize,
        tau: f64,
    },
}
