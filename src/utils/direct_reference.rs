//! Reusable direct-reference policy and comparison helpers.

use crate::utils::verification::DirectReferenceLike;

/// Outcome of comparing an iterative solution against a direct reference.
#[derive(Clone, Debug)]
pub struct DirectReferenceComparison {
    pub abs_error_norm: f64,
    pub rel_error_norm: f64,
    pub matches_verified_answer: bool,
    pub reference_solve_executed: bool,
    pub elapsed_seconds: Option<f64>,
    pub note: String,
}

impl DirectReferenceLike for DirectReferenceComparison {
    fn matches_verified_answer(&self) -> bool {
        self.matches_verified_answer
    }

    fn policy_note(&self) -> &str {
        &self.note
    }
}

/// Inputs for the default direct-reference auto policy.
#[derive(Clone, Copy, Debug)]
pub struct DirectReferencePolicyInput {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
}

/// Compute whether a direct-reference solve should be attempted.
pub fn direct_reference_policy(input: DirectReferencePolicyInput) -> (bool, String) {
    const ALWAYS_DIRECT_MAX_N: usize = 512;
    const MODERATE_DENSITY_MAX_N: usize = 1024;
    const MODERATE_DENSITY_THRESHOLD: f64 = 0.05;

    let n = input.nrows.max(input.ncols);
    let density = input.nnz as f64 / (input.nrows * input.ncols) as f64;
    let env_override = std::env::var("KRYST_ENABLE_DIRECT_REFERENCE")
        .ok()
        .and_then(|v| parse_bool_override(&v));

    match env_override {
        Some(true) => (true, "env override: forced on".to_string()),
        Some(false) => (false, "env override: forced off".to_string()),
        None if n <= ALWAYS_DIRECT_MAX_N => (
            true,
            format!(
                "auto: n={} <= {} (always direct-reference band)",
                n, ALWAYS_DIRECT_MAX_N
            ),
        ),
        None if n <= MODERATE_DENSITY_MAX_N && density >= MODERATE_DENSITY_THRESHOLD => (
            true,
            format!(
                "auto: n={} in ({}, {}] and density {:.3e} >= {:.3e}",
                n, ALWAYS_DIRECT_MAX_N, MODERATE_DENSITY_MAX_N, density, MODERATE_DENSITY_THRESHOLD
            ),
        ),
        None if n <= MODERATE_DENSITY_MAX_N => (
            false,
            format!(
                "auto skip: n={} in ({}, {}] but density {:.3e} < {:.3e}",
                n, ALWAYS_DIRECT_MAX_N, MODERATE_DENSITY_MAX_N, density, MODERATE_DENSITY_THRESHOLD
            ),
        ),
        None => (
            false,
            format!(
                "auto skip: n={} > {} (explicit opt-in required for large matrices)",
                n, MODERATE_DENSITY_MAX_N
            ),
        ),
    }
}

/// Whether global policy allows direct-reference execution unless case-local policy disables it.
pub fn global_direct_reference_policy_allows() -> bool {
    !matches!(
        std::env::var("KRYST_ENABLE_DIRECT_REFERENCE").as_deref(),
        Ok("0" | "false" | "FALSE" | "no" | "NO")
    )
}

/// Compute absolute/relative L2 error against a known reference vector.
pub fn compare_solution_vectors(
    iterative_solution: &[f64],
    reference_solution: &[f64],
    rel_tol: f64,
    note: String,
) -> DirectReferenceComparison {
    let mut diff_sq = 0.0;
    let mut ref_sq = 0.0;
    for (&x_it, &x_ref) in iterative_solution.iter().zip(reference_solution.iter()) {
        let d = x_it - x_ref;
        diff_sq += d * d;
        ref_sq += x_ref * x_ref;
    }
    let abs_error_norm = diff_sq.sqrt();
    let rel_error_norm = abs_error_norm / ref_sq.sqrt().max(1e-32);
    let matches_verified_answer = rel_error_norm <= rel_tol;

    DirectReferenceComparison {
        abs_error_norm,
        rel_error_norm,
        matches_verified_answer,
        reference_solve_executed: true,
        elapsed_seconds: None,
        note,
    }
}

fn parse_bool_override(value: &str) -> Option<bool> {
    match value {
        "1" | "true" | "TRUE" | "yes" | "YES" => Some(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Some(false),
        _ => None,
    }
}
