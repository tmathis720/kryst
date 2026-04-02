//! Optimized solver demonstration using benchmark-proven configurations.
//!
//! This example analyzes matrices from the Matrix Market collection and applies
//! the best solver/preconditioner combinations based on extensive benchmark results.
//! The configurations are chosen to minimize solve time while maintaining accuracy.
//!
//! # Key Insights from Benchmarks
//!
//! ## Sherman3: Best with ILU preconditioning
//! - Trilinos GMRES + ILU: 29 iterations, 0.057s (fastest)
//! - Trilinos BiCGStab + ILU: 20 iterations, 0.049s
//! - ITL CG + ILU: 172 iterations, 0.150s
//!
//! ## Sherman5: Jacobi preconditioning works well
//! - Trilinos BiCGStab + Jacobi: 180 iterations, 0.045s
//! - Trilinos GMRES + ILU: 20 iterations, 0.036s (best)
//! - ITL BiCGStab + Jacobi: 179 iterations, 0.107s
//!
//! ## E05r0100: GMRES methods excel
//! - Hypre GMRES (no precond): 203 iterations, 0.027s (best)
//! - Trilinos GMRES (no precond): 203 iterations, 0.033s
//! - QQQ BiCGStab + ILU: 27 iterations, 0.025s
//!
//! ## E20r5000: ILU preconditioning critical
//! - QQQ BiCGStab + ILU: 45 iterations, 6.6s (best for accuracy)
//! - QQQ GMRES + ILU: 34 iterations, 4.6s
//! - Without ILU, solvers often fail or take 4000+ iterations
//!
//! ## Add20: ILU dramatically reduces iterations
//! - Trilinos CG + ILU: 5 iterations, 0.013s (best)
//! - Trilinos BiCGStab + ILU: 2 iterations, 0.013s
//! - Trilinos GMRES + ILU: 4 iterations, 0.013s
//!
//! ## Memplus: ILU essential for convergence
//! - Trilinos BiCGStab + ILU: 4 iterations, 0.324s (best)
//! - Trilinos GMRES + ILU: 7 iterations, 0.326s
//! - Without ILU: 17,000+ iterations, very slow
#[cfg(feature = "complex")]
fn main() {
    eprintln!("optimized_solver_demo is disabled when the complex feature is enabled.");
}

#[cfg(all(not(feature = "backend-faer"), not(feature = "complex")))]
#[cfg(not(feature = "complex"))]
fn main() {
    eprintln!("optimized_solver_demo requires the backend-faer feature.");
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::config::options::PcOptions;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::context::ksp_context::{KspContext, SolverType};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::context::pc_context::PcType;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::matrix::op::{CsrOp, LinOp};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::matrix::sparse::CsrMatrix;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::preconditioner::ilu_csr::{ReorderingKind, ReorderingOptions};
#[cfg(all(
    feature = "backend-faer",
    feature = "dense-direct",
    not(feature = "complex")
))]
use kryst::solver::dense_lu;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::conditioning::analyze_csr;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::conditioning::{ConditioningOptions, ScaleDirection, ScaleNorm};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::convergence::{AcceptanceStatus, ConvergedReason};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::matrix_market::{MatrixMarketSymmetry, read_matrix_market};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::matrix_screening::{
    SYMMETRY_MAX_ASYMMETRY_RATE, SymmetryAssessment, assess_symmetry, cg_compatibility_screen,
    repair_diagonal_csr,
};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::metrics::true_residual_norm;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::preconditioning_pipeline::{
    PreconditioningMetadata, apply_preconditioning_pipeline,
};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::solver_policy::benchmark_demo_gmres_profile;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::{
    DirectReferenceLike, DirectVerificationCapability, format_direct_verification_status,
};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::str::FromStr;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::time::Instant;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[path = "support/benchmark_catalog.rs"]
mod benchmark_catalog;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use benchmark_catalog::{ComparisonConfidence, compare_best_iterative, expectation_for};

/// Matrix-specific optimal solver configurations based on benchmark results
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct OptimalConfig {
    solver: &'static str,
    preconditioner: &'static str,
    _description: &'static str,
    expected_iterations: usize,
    fallback_solver: &'static str,
    amg_nonsymmetric_override: bool,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct ScreenReport {
    symmetry_hint: bool,
    spd_like_hint: bool,
    symmetry: SymmetryAssessment,
    diagonal_healthy: bool,
    density: f64,
    size_class: &'static str,
    condition_heuristic: f64,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct SelectionDecision {
    primary_solver: String,
    primary_pc: String,
    fallback_ladder: Vec<FallbackStep>,
    rationale: Vec<String>,
    contract_checks: Vec<String>,
    expected_iterations: usize,
    amg_mode: AmgMode,
    amg_status_label: String,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Debug)]
struct FallbackStep {
    solver: String,
    pc: String,
    rung: usize,
    note: String,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JacobiStrengthMode {
    Plain,
    FixDiagonal,
    RowL1OnDefect,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AmgMode {
    DefaultSpd,
    ExplicitNonSpd,
    Disabled,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct AmgDecision {
    mode: AmgMode,
    reason: String,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Debug)]
struct DirectReferenceComparison {
    abs_error_norm: f64,
    rel_error_norm: f64,
    matches_verified_answer: bool,
    reference_solve_executed: bool,
    elapsed_seconds: Option<f64>,
    note: String,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Debug)]
struct AttemptRecord {
    rung_id: usize,
    rung_label: String,
    solver: String,
    preconditioner: String,
    preprocessing_profile: String,
    iterations: usize,
    true_abs_residual: f64,
    true_rel_residual: f64,
    solver_reported_residual: f64,
    solver_reported_status: String,
    acceptance_status: String,
    acceptance_reason: String,
    elapsed_seconds: f64,
    accepted: bool,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Debug)]
struct TruthReference {
    selected_as_winner: bool,
    reference_solve_executed: bool,
    elapsed_seconds: Option<f64>,
    true_abs_residual: Option<f64>,
    true_rel_residual: Option<f64>,
    comparison: Option<DirectReferenceComparison>,
    note: String,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Debug)]
struct SolverTestResult {
    primary_method: String,
    chosen_method: String,
    solver_reason: String,
    outcome_code: String,
    diagnostics: String,
    converged: bool,
    attempts: Vec<AttemptRecord>,
    truth_reference: Option<TruthReference>,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl SolverTestResult {
    fn baseline_attempt(&self) -> Option<&AttemptRecord> {
        self.attempts.iter().find(|attempt| attempt.rung_id == 0)
    }

    fn best_iterative_attempt(&self) -> Option<&AttemptRecord> {
        self.attempts
            .iter()
            .filter(|attempt| attempt.accepted)
            .min_by(|a, b| {
                a.iterations
                    .cmp(&b.iterations)
                    .then_with(|| a.true_rel_residual.total_cmp(&b.true_rel_residual))
            })
    }

    fn best_verified_attempt(&self) -> Option<&AttemptRecord> {
        if self
            .truth_reference
            .as_ref()
            .is_some_and(|truth| truth.selected_as_winner)
        {
            return None;
        }
        self.best_iterative_attempt()
    }

    fn policy_rung_fidelity(&self) -> &'static str {
        let baseline_ok = self
            .attempts
            .iter()
            .any(|attempt| attempt.rung_id == 0 && attempt.accepted);
        let rescue_ok = self
            .attempts
            .iter()
            .any(|attempt| attempt.rung_id > 0 && attempt.accepted);
        match (baseline_ok, rescue_ok) {
            (true, _) => "baseline_ok",
            (false, true) => "rescue_only",
            (false, false) => "none_ok",
        }
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl DirectReferenceLike for DirectReferenceComparison {
    fn matches_verified_answer(&self) -> bool {
        self.matches_verified_answer
    }

    fn policy_note(&self) -> &str {
        &self.note
    }
}

/// Get the optimal solver configuration for a specific matrix
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn get_optimal_config(matrix_name: &str) -> OptimalConfig {
    match matrix_name {
        "fidap005" => OptimalConfig {
            solver: "cg",
            preconditioner: "amg",
            _description: "CG (AMG) - use default AMG only when SPD-like checks pass",
            expected_iterations: 100,
            fallback_solver: "gmres",
            amg_nonsymmetric_override: false,
        },
        "e05r0100" => OptimalConfig {
            solver: "gmres",
            preconditioner: "none",
            _description: "GMRES (no precond) - benchmark baseline for E05r0100",
            expected_iterations: 250,
            fallback_solver: "bicgstab",
            amg_nonsymmetric_override: false,
        },
        "fidap001" => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilut",
            _description: "GMRES (ILUT) - nonsymmetric-safe structural default",
            expected_iterations: 500,
            fallback_solver: "bicgstab",
            amg_nonsymmetric_override: false,
        },
        "sherman3" => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilu",
            _description: "GMRES (ILU) - benchmark-preferred for Sherman3",
            expected_iterations: 50,
            fallback_solver: "bicgstab",
            amg_nonsymmetric_override: false,
        },
        "add20" => OptimalConfig {
            solver: "cg",
            preconditioner: "ilu",
            _description: "CG (ILU) - benchmark-preferred for Add20",
            expected_iterations: 10,
            fallback_solver: "bicgstab",
            amg_nonsymmetric_override: false,
        },
        "memplus" => OptimalConfig {
            solver: "bicgstab",
            preconditioner: "ilu",
            _description: "BiCGStab (ILU) - benchmark-preferred for Memplus",
            expected_iterations: 10,
            fallback_solver: "gmres",
            amg_nonsymmetric_override: false,
        },
        _ => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilut",
            _description: "GMRES (ILUT) - general nonsymmetric-safe sparse default",
            expected_iterations: 500,
            fallback_solver: "bicgstab",
            amg_nonsymmetric_override: false,
        },
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn direct_reference_policy(a_mat: &CsrMatrix<f64>) -> (bool, String) {
    const ALWAYS_DIRECT_MAX_N: usize = 512;
    const MODERATE_DENSITY_MAX_N: usize = 1024;
    const MODERATE_DENSITY_THRESHOLD: f64 = 0.05;
    let n = a_mat.nrows().max(a_mat.ncols());
    let density = a_mat.nnz() as f64 / (a_mat.nrows() * a_mat.ncols()) as f64;
    let env_override = std::env::var("KRYST_ENABLE_DIRECT_REFERENCE")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
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

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn global_direct_reference_policy_allows() -> bool {
    !matches!(
        std::env::var("KRYST_ENABLE_DIRECT_REFERENCE").as_deref(),
        Ok("0" | "false" | "FALSE" | "no" | "NO")
    )
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn is_forced_direct_reference_matrix(matrix_name: &str) -> bool {
    matches!(matrix_name, "fidap005")
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DemoMode {
    Verify,
    IterativeBenchmark,
    Both,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl FromStr for DemoMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "verify" => Ok(Self::Verify),
            "iterative-benchmark" => Ok(Self::IterativeBenchmark),
            "both" => Ok(Self::Both),
            _ => Err(format!(
                "unsupported mode '{value}'. Use one of: verify, iterative-benchmark, both"
            )),
        }
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Copy, Debug)]
struct DirectTruthPolicy {
    allow_truth_path_winner: bool,
    mode_label: &'static str,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn mode_to_policy(mode: DemoMode) -> Vec<DirectTruthPolicy> {
    match mode {
        DemoMode::Verify => vec![DirectTruthPolicy {
            allow_truth_path_winner: true,
            mode_label: "verify",
        }],
        DemoMode::IterativeBenchmark => vec![DirectTruthPolicy {
            allow_truth_path_winner: false,
            mode_label: "iterative-benchmark",
        }],
        DemoMode::Both => vec![
            DirectTruthPolicy {
                allow_truth_path_winner: true,
                mode_label: "verify",
            },
            DirectTruthPolicy {
                allow_truth_path_winner: false,
                mode_label: "iterative-benchmark",
            },
        ],
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn compare_with_direct_reference(
    matrix_name: &str,
    a_mat: &CsrMatrix<f64>,
    rhs: &[f64],
    iterative_solution: &[f64],
) -> Result<Option<DirectReferenceComparison>, Box<dyn std::error::Error>> {
    let (enabled, mut policy_note) = direct_reference_policy(a_mat);
    let force_direct = is_forced_direct_reference_matrix(matrix_name);
    if force_direct {
        policy_note =
            "forced: fidap005 policy prioritizes direct reference path for side-by-side checks"
                .to_string();
    }
    if !enabled && !force_direct {
        return Ok(Some(DirectReferenceComparison {
            abs_error_norm: f64::NAN,
            rel_error_norm: f64::NAN,
            matches_verified_answer: false,
            reference_solve_executed: false,
            elapsed_seconds: None,
            note: policy_note,
        }));
    }

    #[cfg(feature = "dense-direct")]
    {
        let dense_mat = a_mat.to_dense()?;
        let mut reference_solution = vec![0.0; rhs.len()];
        let started = Instant::now();
        if let Err(err) = dense_lu::solve(&dense_mat, rhs, &mut reference_solution) {
            return Ok(Some(DirectReferenceComparison {
                abs_error_norm: f64::NAN,
                rel_error_norm: f64::NAN,
                matches_verified_answer: false,
                reference_solve_executed: true,
                elapsed_seconds: Some(started.elapsed().as_secs_f64()),
                note: format!("{}; direct LU failed ({err})", policy_note),
            }));
        }
        let reference_elapsed = started.elapsed().as_secs_f64();

        let mut diff_sq = 0.0;
        let mut ref_sq = 0.0;
        for (&x_it, &x_ref) in iterative_solution.iter().zip(reference_solution.iter()) {
            let d = x_it - x_ref;
            diff_sq += d * d;
            ref_sq += x_ref * x_ref;
        }
        let abs_error_norm = diff_sq.sqrt();
        let rel_error_norm = abs_error_norm / ref_sq.sqrt().max(1e-32);
        let matches_verified_answer = rel_error_norm <= 1e-6;
        return Ok(Some(DirectReferenceComparison {
            abs_error_norm,
            rel_error_norm,
            matches_verified_answer,
            reference_solve_executed: true,
            elapsed_seconds: Some(reference_elapsed),
            note: policy_note,
        }));
    }

    #[cfg(not(feature = "dense-direct"))]
    {
        let _ = (rhs, iterative_solution);
        Ok(Some(DirectReferenceComparison {
            abs_error_norm: f64::NAN,
            rel_error_norm: f64::NAN,
            matches_verified_answer: false,
            reference_solve_executed: false,
            elapsed_seconds: None,
            note: format!(
                "{}; direct LU failed (dense-direct feature is disabled)",
                policy_note
            ),
        }))
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Debug)]
struct PreconditionerPrepConfig {
    row_scaling: bool,
    col_scaling: bool,
    nonsymmetric_matching: bool,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl PreconditionerPrepConfig {
    fn baseline() -> Self {
        Self {
            row_scaling: true,
            col_scaling: true,
            nonsymmetric_matching: true,
        }
    }

    fn for_matrix(matrix_name: &str) -> Self {
        match matrix_name {
            "fidap001" | "fidap005" | "sherman3" | "e05r0100" => Self::baseline(),
            _ => Self {
                row_scaling: true,
                col_scaling: false,
                nonsymmetric_matching: false,
            },
        }
    }

    fn as_trace(&self) -> String {
        format!(
            "preprocess[row_scale={}, col_scale={}, nonsym_matching={}]",
            self.row_scaling, self.col_scaling, self.nonsymmetric_matching
        )
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn prep_config_for(matrix_name: &str) -> PreconditionerPrepConfig {
    PreconditionerPrepConfig::for_matrix(matrix_name)
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn select_reordering_for_preprocessing(
    prep_cfg: &PreconditionerPrepConfig,
    a_mat: &CsrMatrix<f64>,
) -> (ReorderingOptions, String, Option<String>) {
    if prep_cfg.nonsymmetric_matching {
        if a_mat.nrows() == a_mat.ncols() {
            let mode = ReorderingOptions {
                kind: ReorderingKind::Amd,
                symmetric: false,
                deterministic: true,
            };
            return (mode, "amd_nonsym(greedy_matching+amd)".to_string(), None);
        }
        let fallback = ReorderingOptions {
            kind: ReorderingKind::None,
            symmetric: true,
            deterministic: true,
        };
        return (
            fallback,
            "none".to_string(),
            Some(
                "requested nonsymmetric matching is unavailable for non-square matrices; falling back to none"
                    .to_string(),
            ),
        );
    }

    (
        ReorderingOptions {
            kind: ReorderingKind::None,
            symmetric: true,
            deterministic: true,
        },
        "none".to_string(),
        None,
    )
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn preprocessing_metadata_trace(meta: &PreconditioningMetadata) -> String {
    let matched_pairs = meta.matched_pairs.as_ref().map_or(0usize, Vec::len);
    format!(
        "meta[row_scaled={}, col_scaled={}, nonsym_pairs={}]",
        meta.row_scaling.is_some(),
        meta.col_scaling.is_some(),
        matched_pairs
    )
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn build_preconditioner_options(
    matrix_name: &str,
    solver: &str,
    pc: &str,
    amg_mode: AmgMode,
    fallback_rung: Option<usize>,
) -> Option<PcOptions> {
    if pc == "amg" {
        if amg_mode == AmgMode::ExplicitNonSpd {
            let mut opts = PcOptions::default();
            opts.amg_require_spd = Some(false);
            opts.amg_relax_type = Some("jacobi".into());
            opts.amg_smoother = Some("jacobi".into());
            opts.amg_coarse_solver = Some("ilu".into());
            return Some(opts);
        }
        let _ = solver;
        return None;
    }

    if pc != "ilu" && pc != "ilut" {
        return None;
    }

    let mut opts = PcOptions::default();
    let prep_cfg = prep_config_for(matrix_name);
    opts.ilu_type = Some(if pc == "ilut" { "ilut" } else { "iluk" }.to_string());
    opts.ilu_level_of_fill = Some(1);
    opts.ilu_max_fill_per_row = Some(24);
    opts.ilu_offdiag_drop_tolerance = Some(1e-4);
    opts.ilu_reordering_type = Some("rcm".to_string());
    opts.ilu_reordering = Some("rcm".to_string());
    opts.ilu_triangular_solve = Some("gauss-seidel".to_string());
    opts.ilu_pivot_threshold = Some(1e-10);
    opts.pc_scale = if prep_cfg.row_scaling && prep_cfg.col_scaling {
        Some("both".to_string())
    } else if prep_cfg.row_scaling {
        Some("row".to_string())
    } else if prep_cfg.col_scaling {
        Some("col".to_string())
    } else {
        None
    };
    opts.pc_scale_norm = Some("inf".to_string());
    if prep_cfg.nonsymmetric_matching {
        opts.ilu_reordering = Some("amd_nonsym".to_string());
    }

    match matrix_name {
        "e05r0100" if fallback_rung == Some(1) && pc == "ilut" => {
            opts.ilu_max_fill_per_row = Some(48);
            opts.ilu_offdiag_drop_tolerance = Some(1e-6);
            opts.ilu_reordering_type = Some("amd".to_string());
            opts.ilu_reordering = Some("amd_nonsym".to_string());
            opts.ilu_triangular_solve = Some("exact".to_string());
            opts.ilu_pivot_threshold = Some(1e-12);
        }
        "sherman3" => {
            if fallback_rung.is_some() {
                opts.ilu_type = Some("ilut".to_string());
                opts.ilu_max_fill_per_row = Some(64);
                opts.ilu_offdiag_drop_tolerance = Some(5e-7);
                opts.ilu_reordering_type = Some("amd".to_string());
                opts.ilu_reordering = Some("amd_nonsym".to_string());
                opts.ilu_triangular_solve = Some("exact".to_string());
                opts.ilu_pivot_threshold = Some(1e-12);
            } else {
                opts.ilu_type = Some("iluk".to_string());
                opts.ilu_level_of_fill = Some(2);
                opts.ilu_max_fill_per_row = Some(40);
                opts.ilu_offdiag_drop_tolerance = Some(5e-5);
            }
        }
        "fidap001" => {
            opts.ilu_type = Some("ilut".to_string());
            opts.ilu_max_fill_per_row = Some(72);
            opts.ilu_offdiag_drop_tolerance = Some(1e-7);
            opts.ilu_reordering_type = Some("amd".to_string());
            opts.ilu_reordering = Some("amd_nonsym".to_string());
            opts.ilu_triangular_solve = Some("exact".to_string());
            opts.ilu_pivot_threshold = Some(1e-12);
            opts.pc_scale = Some("both".to_string());
        }
        "fidap005" => {
            opts.ilu_type = Some("ilut".to_string());
            opts.ilu_max_fill_per_row = Some(56);
            opts.ilu_offdiag_drop_tolerance = Some(1e-6);
            opts.ilu_reordering_type = Some("amd".to_string());
            opts.ilu_reordering = Some("amd_nonsym".to_string());
            opts.ilu_triangular_solve = Some("exact".to_string());
            opts.ilu_pivot_threshold = Some(1e-12);
            opts.pc_scale_norm = Some("1".to_string());
        }
        _ => {}
    }
    if fallback_rung == Some(1) && matches!(solver, "gmres" | "fgmres") && pc == "ilut" {
        opts.ilu_type = Some("ilut".to_string());
        opts.ilu_max_fill_per_row = Some(opts.ilu_max_fill_per_row.unwrap_or(48).max(64));
        opts.ilu_offdiag_drop_tolerance =
            Some(opts.ilu_offdiag_drop_tolerance.unwrap_or(1e-6).min(5e-7));
        opts.ilu_reordering_type = Some("amd".to_string());
        opts.ilu_reordering = Some("amd_nonsym".to_string());
        opts.ilu_triangular_solve = Some("exact".to_string());
        opts.ilu_pivot_threshold = Some(1e-12);
    }

    Some(opts)
}

/// Test a solver configuration and return detailed results
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn test_optimal_solver(
    a_mat: &CsrMatrix<f64>,
    p_mat: Option<&CsrMatrix<f64>>,
    rhs: &[f64],
    decision: &SelectionDecision,
    screen: &ScreenReport,
    matrix_name: &str,
    prep_trace: &str,
    is_root_rank: bool,
    verbose_details: bool,
    direct_truth_policy: DirectTruthPolicy,
) -> Result<SolverTestResult, Box<dyn std::error::Error>> {
    let iterative_benchmark_mode = !direct_truth_policy.allow_truth_path_winner;
    let mut solution = vec![0.0; rhs.len()];
    let csr_a_matrix = Arc::new(a_mat.clone());
    let a_op: Arc<dyn LinOp<S = f64>> = Arc::new(CsrOp::new(Arc::clone(&csr_a_matrix)));
    let p_op: Option<Arc<dyn LinOp<S = f64>>> = match p_mat {
        Some(p) => Some(Arc::new(CsrOp::new(Arc::new(p.clone())))),
        None => None,
    };
    let rhs_vec = rhs.to_vec();

    let mut ksp = KspContext::new();
    let st = SolverType::from_str(&decision.primary_solver)?;
    let pct = PcType::from_str(&decision.primary_pc)?;
    let primary_opts = build_preconditioner_options(
        matrix_name,
        &decision.primary_solver,
        &decision.primary_pc,
        decision.amg_mode,
        None,
    );
    ksp.set_type(st)?
        .set_pc_type(pct, primary_opts.as_ref())?
        .set_tolerances(1e-6, 1e-12, 1e3, 1000);
    if matches!(decision.primary_solver.as_str(), "gmres" | "fgmres") {
        let difficult_nonsymmetric =
            !screen.symmetry_hint && screen.condition_heuristic >= 1.0e4 && !screen.spd_like_hint;
        let gmres_profile = benchmark_demo_gmres_profile(difficult_nonsymmetric);
        ksp.set_from_options(&gmres_profile)?;
    }
    ksp.set_operators(Arc::clone(&a_op), p_op.clone());
    ksp.setup()?;

    const CONTRACT_RTOL: f64 = 1e-6;
    const CONTRACT_ATOL: f64 = 1e-12;
    const CONTRACT_SLACK: f64 = 1.05;

    fn classify_failure(msg: &str) -> &'static str {
        let m = msg.to_ascii_lowercase();
        if m.contains("contract") || m.contains("cg rejected") || m.contains("wrong method") {
            "CONTRACT_MISMATCH"
        } else if m.contains("breakdown")
            || m.contains("nan")
            || m.contains("inf")
            || m.contains("zero pivot")
            || m.contains("singular")
        {
            "BREAKDOWN"
        } else if m.contains("stagnat")
            || m.contains("max iter")
            || m.contains("did not converge")
            || m.contains("no convergence")
        {
            "STAGNATED"
        } else {
            "FAILED"
        }
    }

    fn true_residual_metrics(
        a_op: &dyn LinOp<S = f64>,
        rhs: &[f64],
        solution: &[f64],
    ) -> (f64, f64) {
        let true_abs_res = true_residual_norm(a_op, rhs, solution);
        let b_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let true_rel_res = if b_norm > 0.0 {
            true_abs_res / b_norm
        } else if true_abs_res == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        (true_abs_res, true_rel_res)
    }

    fn classify_acceptance(
        reason: ConvergedReason,
        true_abs_res: f64,
        true_rel_res: f64,
        rtol: f64,
        atol: f64,
        slack: f64,
    ) -> AcceptanceStatus {
        let rtol_ok = true_rel_res.is_finite() && true_rel_res <= rtol * slack;
        let atol_ok = true_abs_res.is_finite() && true_abs_res <= atol * slack;
        let meets_any_contract = rtol_ok || atol_ok;
        match reason {
            ConvergedReason::ConvergedRtol => {
                if rtol_ok {
                    AcceptanceStatus::Ok
                } else if meets_any_contract {
                    AcceptanceStatus::OkWithWarning
                } else {
                    AcceptanceStatus::ContractMismatch
                }
            }
            ConvergedReason::ConvergedAtol => {
                if atol_ok {
                    AcceptanceStatus::Ok
                } else if meets_any_contract {
                    AcceptanceStatus::OkWithWarning
                } else {
                    AcceptanceStatus::ContractMismatch
                }
            }
            ConvergedReason::DivergedBreakdown
            | ConvergedReason::DivergedBreakdownBiCG
            | ConvergedReason::DivergedNan
            | ConvergedReason::DivergedInf
            | ConvergedReason::DivergedIndefiniteMatrix
            | ConvergedReason::DivergedIndefinitePC => {
                if meets_any_contract {
                    AcceptanceStatus::OkWithWarning
                } else {
                    AcceptanceStatus::Breakdown
                }
            }
            ConvergedReason::ConvergedTrustRegion | ConvergedReason::ConvergedHappyBreakdown => {
                if meets_any_contract {
                    AcceptanceStatus::OkWithWarning
                } else {
                    AcceptanceStatus::ContractMismatch
                }
            }
            ConvergedReason::DivergedDtol
            | ConvergedReason::DivergedMaxIts
            | ConvergedReason::StoppedByMonitor => {
                if meets_any_contract {
                    AcceptanceStatus::OkWithWarning
                } else {
                    AcceptanceStatus::Stagnated
                }
            }
            ConvergedReason::DivergedPcSetupFailed
            | ConvergedReason::DivergedPcFailed
            | ConvergedReason::Continued => {
                if meets_any_contract {
                    AcceptanceStatus::OkWithWarning
                } else {
                    AcceptanceStatus::Failed
                }
            }
        }
    }

    fn solver_reason_code(
        reason: ConvergedReason,
        acceptance_status: AcceptanceStatus,
    ) -> &'static str {
        match acceptance_status {
            AcceptanceStatus::Ok | AcceptanceStatus::OkWithWarning => match reason {
                ConvergedReason::ConvergedRtol => "RTOL_OK",
                ConvergedReason::ConvergedAtol => "ATOL_OK",
                ConvergedReason::ConvergedTrustRegion => "TRUST_REGION_OK",
                ConvergedReason::ConvergedHappyBreakdown => "HAPPY_BREAKDOWN_OK",
                ConvergedReason::DivergedDtol
                | ConvergedReason::DivergedMaxIts
                | ConvergedReason::StoppedByMonitor => "CONTRACT_OK_WARN",
                _ => "CONTRACT_OK_WARN",
            },
            AcceptanceStatus::ContractMismatch => "CONTRACT_MISMATCH",
            AcceptanceStatus::Breakdown => "BREAKDOWN",
            AcceptanceStatus::Stagnated => "STAGNATED",
            AcceptanceStatus::Failed => "FAILED",
        }
    }

    let primary_method = format!(
        "{} + {}",
        decision.primary_solver.to_uppercase(),
        decision.primary_pc.to_uppercase()
    );
    let mut best_iterative_solution: Option<Vec<f64>> = None;
    let mut best_iterative_method = "-".to_string();
    let mut best_iterative_reason = "NO_ACCEPTED_SOLVE".to_string();
    let mut best_iterative_outcome = "NO_ACCEPTED_SOLVE".to_string();
    let mut best_iterative_diagnostics = String::new();
    let mut best_iterative_iterations: Option<usize> = None;
    let mut best_iterative_rel_residual: Option<f64> = None;
    let mut attempts = Vec::<AttemptRecord>::new();
    let mut truth_reference: Option<TruthReference> = None;
    let solve_started = Instant::now();
    let result = ksp.solve(&rhs_vec, &mut solution);
    let solve_elapsed = solve_started.elapsed().as_secs_f64();

    let (primary_failure, primary_reason) = match result {
        Ok(stats) => {
            let (true_abs_res, true_rel_res) =
                true_residual_metrics(a_op.as_ref(), &rhs_vec, &solution);
            let acceptance_status = classify_acceptance(
                stats.reason,
                true_abs_res,
                true_rel_res,
                CONTRACT_RTOL,
                CONTRACT_ATOL,
                CONTRACT_SLACK,
            );
            let solver_reason = solver_reason_code(stats.reason, acceptance_status).to_string();
            let acceptance_reason = format!(
                "solver_reason={} | internal_classical_retry={}",
                stats.reason.petsc_reason(),
                stats.gmres_classical_retry
            );
            attempts.push(AttemptRecord {
                rung_id: 0,
                rung_label: "baseline".to_string(),
                solver: decision.primary_solver.to_uppercase(),
                preconditioner: decision.primary_pc.to_uppercase(),
                preprocessing_profile: prep_trace.to_string(),
                iterations: stats.iterations,
                true_abs_residual: true_abs_res,
                true_rel_residual: true_rel_res,
                solver_reported_residual: stats.final_residual,
                solver_reported_status: solver_reason.clone(),
                acceptance_status: acceptance_status.as_str().to_string(),
                acceptance_reason: acceptance_reason.clone(),
                elapsed_seconds: solve_elapsed,
                accepted: acceptance_status.is_accepted(),
            });
            if acceptance_status.is_accepted() {
                let reason = format!(
                    "solver_reason={} | rung=baseline | {} | internal_classical_retry={}",
                    stats.reason.petsc_reason(),
                    prep_trace,
                    stats.gmres_classical_retry
                );
                best_iterative_solution = Some(solution.clone());
                best_iterative_method = "-".to_string();
                best_iterative_reason = solver_reason.clone();
                best_iterative_outcome = "PRIMARY_ACCEPTED".to_string();
                best_iterative_diagnostics = reason.clone();
                best_iterative_iterations = Some(stats.iterations);
                best_iterative_rel_residual = Some(true_rel_res);
                if iterative_benchmark_mode {
                    // Continue through all fallback rungs and decide by the best iterative attempt.
                } else {
                    let comparison =
                        compare_with_direct_reference(matrix_name, a_mat, &rhs_vec, &solution)?;
                    let reference_solve_executed = comparison
                        .as_ref()
                        .is_some_and(|cmp| cmp.reference_solve_executed);
                    let reference_elapsed = comparison.as_ref().and_then(|cmp| cmp.elapsed_seconds);
                    truth_reference = Some(TruthReference {
                        selected_as_winner: false,
                        reference_solve_executed,
                        elapsed_seconds: reference_elapsed,
                        true_abs_residual: None,
                        true_rel_residual: None,
                        comparison,
                        note: "iterative winner with direct-reference side-channel check"
                            .to_string(),
                    });
                    return Ok(SolverTestResult {
                        primary_method,
                        chosen_method: "-".to_string(),
                        solver_reason,
                        outcome_code: "PRIMARY_ACCEPTED".to_string(),
                        diagnostics: reason,
                        converged: true,
                        attempts,
                        truth_reference,
                    });
                }
            }
            (
                acceptance_status.as_str().to_string(),
                format!(
                    "soft failure: reason={:?}, true_abs_res={:.3e}, true_rel_res={:.3e}, internal_classical_retry={}",
                    stats.reason, true_abs_res, true_rel_res, stats.gmres_classical_retry
                ),
            )
        }
        Err(primary_err) => {
            let msg = primary_err.to_string();
            (classify_failure(&msg).to_ascii_uppercase(), msg)
        }
    };

    if is_root_rank
        && verbose_details
        && !matches!(primary_failure.as_str(), "ok" | "ok_with_warning")
    {
        println!("    Primary rung failed, entering fallback ladder...");
    }

    let mut attempt_log = vec![format!(
        "rung=baseline [{}]: {} | {}",
        primary_failure, primary_reason, prep_trace
    )];

    let mut fallback_contract_unmet = false;
    for step in &decision.fallback_ladder {
        let mut sol_fb = vec![0.0; rhs.len()];
        let mut ksp_fallback = KspContext::new();
        let st_fb = SolverType::from_str(&step.solver)?;
        let pc_fb = PcType::from_str(&step.pc)?;
        let fallback_opts = build_preconditioner_options(
            matrix_name,
            &step.solver,
            &step.pc,
            AmgMode::Disabled,
            Some(step.rung),
        );
        ksp_fallback
            .set_type(st_fb)?
            .set_pc_type(pc_fb, fallback_opts.as_ref())?
            .set_tolerances(1e-6, 1e-12, 1e3, 1000);
        if matches!(step.solver.as_str(), "gmres" | "fgmres") {
            let difficult_nonsymmetric = !screen.symmetry_hint
                && screen.condition_heuristic >= 1.0e4
                && !screen.spd_like_hint;
            let mut gmres_profile = benchmark_demo_gmres_profile(difficult_nonsymmetric);
            gmres_profile.pc_side = Some("right".to_string());
            ksp_fallback.set_from_options(&gmres_profile)?;
        }
        ksp_fallback.set_operators(Arc::clone(&a_op), p_op.clone());
        ksp_fallback.setup()?;

        let fallback_started = Instant::now();
        match ksp_fallback.solve(&rhs_vec, &mut sol_fb) {
            Ok(stats_fallback) => {
                let fallback_elapsed = fallback_started.elapsed().as_secs_f64();
                let (true_abs_res, true_rel_res) =
                    true_residual_metrics(a_op.as_ref(), &rhs_vec, &sol_fb);
                let acceptance_status = classify_acceptance(
                    stats_fallback.reason,
                    true_abs_res,
                    true_rel_res,
                    CONTRACT_RTOL,
                    CONTRACT_ATOL,
                    CONTRACT_SLACK,
                );
                let fallback_method = format!(
                    "R{} {} + {}",
                    step.rung,
                    step.solver.to_uppercase(),
                    step.pc.to_uppercase()
                );
                let solver_reason =
                    solver_reason_code(stats_fallback.reason, acceptance_status).to_string();
                attempts.push(AttemptRecord {
                    rung_id: step.rung,
                    rung_label: format!("rescue_{}", step.rung),
                    solver: step.solver.to_uppercase(),
                    preconditioner: step.pc.to_uppercase(),
                    preprocessing_profile: prep_trace.to_string(),
                    iterations: stats_fallback.iterations,
                    true_abs_residual: true_abs_res,
                    true_rel_residual: true_rel_res,
                    solver_reported_residual: stats_fallback.final_residual,
                    solver_reported_status: solver_reason.clone(),
                    acceptance_status: acceptance_status.as_str().to_string(),
                    acceptance_reason: format!(
                        "solver_reason={} | {}",
                        stats_fallback.reason.petsc_reason(),
                        step.note
                    ),
                    elapsed_seconds: fallback_elapsed,
                    accepted: acceptance_status.is_accepted(),
                });
                if acceptance_status.is_accepted() {
                    let reason = format!(
                        "solver_reason={} | {} | attempted: {}",
                        stats_fallback.reason.petsc_reason(),
                        step.note,
                        attempt_log.join(" -> ")
                    );
                    let replace_best =
                        match (best_iterative_iterations, best_iterative_rel_residual) {
                            (None, _) => true,
                            (Some(best_iters), Some(best_rel)) => {
                                stats_fallback.iterations < best_iters
                                    || (stats_fallback.iterations == best_iters
                                        && true_rel_res.total_cmp(&best_rel).is_lt())
                            }
                            (Some(_), None) => true,
                        };
                    if replace_best {
                        best_iterative_solution = Some(sol_fb.clone());
                        best_iterative_method = fallback_method.clone();
                        best_iterative_reason = solver_reason.clone();
                        best_iterative_outcome = "FALLBACK_ACCEPTED".to_string();
                        best_iterative_diagnostics = reason.clone();
                        best_iterative_iterations = Some(stats_fallback.iterations);
                        best_iterative_rel_residual = Some(true_rel_res);
                    }
                    if iterative_benchmark_mode {
                        attempt_log.push(format!(
                            "rung={} [{}]: accepted reason={:?}, true_rel_res={:.3e}",
                            step.rung,
                            acceptance_status.as_str(),
                            stats_fallback.reason,
                            true_rel_res
                        ));
                        continue;
                    }
                    let comparison =
                        compare_with_direct_reference(matrix_name, a_mat, &rhs_vec, &sol_fb)?;
                    let reference_solve_executed = comparison
                        .as_ref()
                        .is_some_and(|cmp| cmp.reference_solve_executed);
                    let reference_elapsed = comparison.as_ref().and_then(|cmp| cmp.elapsed_seconds);
                    truth_reference = Some(TruthReference {
                        selected_as_winner: false,
                        reference_solve_executed,
                        elapsed_seconds: reference_elapsed,
                        true_abs_residual: None,
                        true_rel_residual: None,
                        comparison,
                        note: "iterative winner with direct-reference side-channel check"
                            .to_string(),
                    });
                    return Ok(SolverTestResult {
                        primary_method,
                        chosen_method: fallback_method,
                        solver_reason,
                        outcome_code: "FALLBACK_ACCEPTED".to_string(),
                        diagnostics: reason,
                        converged: true,
                        attempts,
                        truth_reference,
                    });
                }
                fallback_contract_unmet |=
                    matches!(acceptance_status, AcceptanceStatus::ContractMismatch);
                attempt_log.push(format!(
                    "rung={} [{}]: reason={:?}, true_rel_res={:.3e}",
                    step.rung,
                    acceptance_status.as_str(),
                    stats_fallback.reason,
                    true_rel_res
                ));
            }
            Err(fallback_err) => {
                let fallback_elapsed = fallback_started.elapsed().as_secs_f64();
                let failure = classify_failure(&fallback_err.to_string()).to_ascii_uppercase();
                attempts.push(AttemptRecord {
                    rung_id: step.rung,
                    rung_label: format!("rescue_{}", step.rung),
                    solver: step.solver.to_uppercase(),
                    preconditioner: step.pc.to_uppercase(),
                    preprocessing_profile: prep_trace.to_string(),
                    iterations: 0,
                    true_abs_residual: f64::NAN,
                    true_rel_residual: f64::NAN,
                    solver_reported_residual: f64::NAN,
                    solver_reported_status: failure.clone(),
                    acceptance_status: "failed".to_string(),
                    acceptance_reason: fallback_err.to_string(),
                    elapsed_seconds: fallback_elapsed,
                    accepted: false,
                });
                attempt_log.push(format!(
                    "rung={} [{}]: {}",
                    step.rung,
                    classify_failure(&fallback_err.to_string()),
                    fallback_err
                ));
            }
        }
    }

    let (allow_direct, direct_policy_reason) = direct_reference_policy(a_mat);
    if allow_direct && direct_truth_policy.allow_truth_path_winner {
        #[cfg(feature = "dense-direct")]
        {
            let dense_mat = a_mat.to_dense()?;
            let mut x_direct = vec![0.0; rhs_vec.len()];
            let direct_started = Instant::now();
            match dense_lu::solve(&dense_mat, &rhs_vec, &mut x_direct) {
                Ok(()) => {
                    let direct_elapsed = direct_started.elapsed().as_secs_f64();
                    let (true_abs_res, true_rel_res) =
                        true_residual_metrics(a_op.as_ref(), &rhs_vec, &x_direct);
                    let cmp = DirectReferenceComparison {
                        abs_error_norm: 0.0,
                        rel_error_norm: 0.0,
                        matches_verified_answer: true,
                        reference_solve_executed: true,
                        elapsed_seconds: Some(direct_elapsed),
                        note: format!("truth_path: {direct_policy_reason}"),
                    };
                    truth_reference = Some(TruthReference {
                        selected_as_winner: true,
                        reference_solve_executed: true,
                        elapsed_seconds: Some(direct_elapsed),
                        true_abs_residual: Some(true_abs_res),
                        true_rel_residual: Some(true_rel_res),
                        comparison: Some(cmp),
                        note: format!("truth_path: {direct_policy_reason}"),
                    });
                    return Ok(SolverTestResult {
                        primary_method,
                        chosen_method: "R3 DIRECT_LU (truth path)".to_string(),
                        solver_reason: "DIRECT_TRUTH_PATH".to_string(),
                        outcome_code: "DIRECT_TRUTH_PATH".to_string(),
                        diagnostics: format!(
                            "rung=3 direct truth path selected: {} | attempted: {}",
                            direct_policy_reason,
                            attempt_log.join(" -> ")
                        ),
                        converged: true,
                        attempts,
                        truth_reference,
                    });
                }
                Err(err) => {
                    attempt_log.push(format!("rung=3 [failed]: direct LU error: {err}"));
                    truth_reference = Some(TruthReference {
                        selected_as_winner: false,
                        reference_solve_executed: false,
                        elapsed_seconds: Some(direct_started.elapsed().as_secs_f64()),
                        true_abs_residual: None,
                        true_rel_residual: None,
                        comparison: None,
                        note: format!("truth_path attempted but failed: {err}"),
                    });
                }
            }
        }
        #[cfg(not(feature = "dense-direct"))]
        {
            attempt_log.push("rung=3 [failed]: dense-direct feature disabled".to_string());
            truth_reference = Some(TruthReference {
                selected_as_winner: false,
                reference_solve_executed: false,
                elapsed_seconds: None,
                true_abs_residual: None,
                true_rel_residual: None,
                comparison: None,
                note: "truth_path unavailable: dense-direct feature disabled".to_string(),
            });
        }
    } else if allow_direct && !direct_truth_policy.allow_truth_path_winner {
        let comparison = if let Some(solution) = best_iterative_solution.as_ref() {
            compare_with_direct_reference(matrix_name, a_mat, &rhs_vec, solution)?
        } else {
            None
        };
        let reference_solve_executed = comparison
            .as_ref()
            .is_some_and(|cmp| cmp.reference_solve_executed);
        let reference_elapsed = comparison.as_ref().and_then(|cmp| cmp.elapsed_seconds);
        attempt_log.push(format!(
            "rung=3 [suppressed]: mode={} keeps direct reference as side-channel only ({})",
            direct_truth_policy.mode_label, direct_policy_reason
        ));
        truth_reference = Some(TruthReference {
            selected_as_winner: false,
            reference_solve_executed,
            elapsed_seconds: reference_elapsed,
            true_abs_residual: None,
            true_rel_residual: None,
            comparison,
            note: format!(
                "truth_path suppressed by mode={} ({})",
                direct_truth_policy.mode_label, direct_policy_reason
            ),
        });
    } else {
        attempt_log.push(format!("rung=3 [skipped]: {}", direct_policy_reason));
        truth_reference = Some(TruthReference {
            selected_as_winner: false,
            reference_solve_executed: false,
            elapsed_seconds: None,
            true_abs_residual: None,
            true_rel_residual: None,
            comparison: None,
            note: format!("truth_path skipped: {direct_policy_reason}"),
        });
    }

    if iterative_benchmark_mode && best_iterative_solution.is_some() {
        return Ok(SolverTestResult {
            primary_method,
            chosen_method: best_iterative_method,
            solver_reason: best_iterative_reason,
            outcome_code: best_iterative_outcome,
            diagnostics: best_iterative_diagnostics,
            converged: true,
            attempts,
            truth_reference,
        });
    }

    let terminal_reason_code = if fallback_contract_unmet {
        "FALLBACK_CONTRACT_UNMET"
    } else {
        "FALLBACK_EXHAUSTED"
    };
    Ok(SolverTestResult {
        primary_method,
        chosen_method: "-".to_string(),
        solver_reason: "NO_ACCEPTED_SOLVE".to_string(),
        outcome_code: terminal_reason_code.to_string(),
        diagnostics: format!("all rungs exhausted: {}", attempt_log.join(" -> ")),
        converged: false,
        attempts,
        truth_reference,
    })
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn choose_amg_mode_with_override(
    diag_issues: bool,
    symmetry_hint: bool,
    spd_like_hint: bool,
    allow_nonsymmetric_override: bool,
) -> AmgDecision {
    if spd_like_hint {
        return AmgDecision {
            mode: AmgMode::DefaultSpd,
            reason: "accepted: SPD-like screen passed (conditioning symmetry + healthy diagonal)"
                .to_string(),
        };
    }

    if allow_nonsymmetric_override && !diag_issues {
        return AmgDecision {
            mode: AmgMode::ExplicitNonSpd,
            reason: "accepted: explicit nonsymmetric AMG override selected (require_spd=false, Jacobi smoother, ILU-like coarse solve)".to_string(),
        };
    }

    let mut causes = Vec::new();
    if !allow_nonsymmetric_override {
        causes.push("default AMG is SPD-only and no nonsymmetric override was selected");
    }
    if diag_issues {
        causes.push("diagonal issues");
    }
    if !symmetry_hint {
        causes.push("nonsymmetric structure");
    }
    AmgDecision {
        mode: AmgMode::Disabled,
        reason: format!(
            "rejected: {} (routed to ILU/ILUT fallback)",
            causes.join(", ")
        ),
    }
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn screen_matrix(matrix: &CsrMatrix<f64>) -> ScreenReport {
    let tiny_threshold = 1e-14;
    let stats = analyze_csr(matrix, tiny_threshold);
    let diag_issues = stats.diag_tiny_count > 0 || stats.diag_missing_count > 0;
    let symmetry = assess_symmetry(matrix, None, false);
    let approx_symmetric = symmetry.passes_threshold;
    let density = matrix.nnz() as f64 / (matrix.nrows() * matrix.ncols()) as f64;
    let condition_heuristic = if stats.diag_min_abs > 0.0 {
        stats.row_norm_1.max / stats.diag_min_abs
    } else {
        f64::INFINITY
    };
    let n = matrix.nrows().max(matrix.ncols());
    let size_class = if n <= 512 {
        "small"
    } else if n <= 4_096 {
        "medium"
    } else {
        "large"
    };
    let spd_like_hint = !diag_issues && approx_symmetric;
    ScreenReport {
        symmetry_hint: approx_symmetric,
        spd_like_hint,
        symmetry,
        diagonal_healthy: !diag_issues,
        density,
        size_class,
        condition_heuristic,
    }
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn jacobi_strength_mode_from_env() -> JacobiStrengthMode {
    match std::env::var("KRYST_DEMO_JACOBI_STRENGTH").ok().as_deref() {
        Some("fixdiag") | Some("fix_diagonal") => JacobiStrengthMode::FixDiagonal,
        Some("rowl1") | Some("row_l1") | Some("l1") => JacobiStrengthMode::RowL1OnDefect,
        _ => JacobiStrengthMode::Plain,
    }
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn compare_structural_cg_screen_enabled() -> bool {
    matches!(
        std::env::var("KRYST_DEMO_COMPARE_STRUCTURAL_CG_SCREEN")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn cg_reject_has_strong_evidence(
    symmetry_violation_count: usize,
    non_positive_diagonal_count: usize,
) -> bool {
    symmetry_violation_count > 0 || non_positive_diagonal_count > 0
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn default_fallback_ladder() -> Vec<FallbackStep> {
    vec![
        FallbackStep {
            solver: "fgmres".to_string(),
            pc: "ilut".to_string(),
            rung: 1,
            note: "rung=1 fallback: GMRES/FGMRES(right) + stronger ILUT profile".to_string(),
        },
        FallbackStep {
            solver: "bicgstab".to_string(),
            pc: "ilut".to_string(),
            rung: 2,
            note: "rung=2 fallback: BiCGStab + ILUT".to_string(),
        },
    ]
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn select_solver_policy(
    matrix_name: &str,
    screen: &ScreenReport,
    matrix: &CsrMatrix<f64>,
    structural_symmetry_hint: Option<bool>,
) -> SelectionDecision {
    let baseline = get_optimal_config(matrix_name);
    let mut primary_solver = baseline.solver.to_string();
    let mut primary_pc = baseline.preconditioner.to_string();
    let mut fallback_ladder = default_fallback_ladder();
    let mut rationale = vec![format!(
        "matrix hint '{}': {}",
        matrix_name, baseline._description
    )];
    let mut contract_checks = Vec::new();
    let mut amg_mode = AmgMode::Disabled;
    let jacobi_strength_mode = jacobi_strength_mode_from_env();

    let cg_screen = cg_compatibility_screen(matrix, !screen.diagonal_healthy, None, false);
    let base_strong_reject_evidence = cg_reject_has_strong_evidence(
        cg_screen.diagnostics.symmetry.symmetry_violation_count,
        cg_screen.diagnostics.non_positive_diagonal_count,
    );
    if primary_solver == "cg" && cg_screen.is_hard_reject && base_strong_reject_evidence {
        let safe_solver =
            if baseline.fallback_solver == "gmres" || baseline.fallback_solver == "bicgstab" {
                baseline.fallback_solver
            } else {
                "gmres"
            };
        primary_solver = safe_solver.to_string();
        primary_pc = "ilut".to_string();
        contract_checks.push(cg_screen.reason.clone());
        contract_checks.push(format!(
            "CG diagnostics (base): pairs={}, sym_violations={} ({:.2}%), non_positive_diag={}, weak_gershgorin={}, mm_structural_symmetry_hint={:?}",
            cg_screen.diagnostics.symmetry.sampled_pair_count,
            cg_screen.diagnostics.symmetry.symmetry_violation_count,
            100.0 * cg_screen.diagnostics.symmetry.symmetry_violation_rate,
            cg_screen.diagnostics.non_positive_diagonal_count,
            cg_screen.diagnostics.weak_gershgorin_count,
            cg_screen.diagnostics.symmetry.structural_symmetry_hint
        ));
        contract_checks.push(format!(
            "CG symmetry verdict (base): {}",
            cg_screen.diagnostics.symmetry.verdict
        ));
        rationale.push(format!(
            "CG screened out; switched primary to {} + ILUT",
            safe_solver.to_uppercase()
        ));
    } else {
        if primary_solver == "cg"
            && cg_screen.is_hard_reject
            && cg_screen.diagnostics.weak_gershgorin_count > 0
            && !base_strong_reject_evidence
        {
            contract_checks.push(
                "CG hard-reject downgraded to warning: weak Gershgorin evidence alone is treated as non-fatal in this demo"
                    .to_string(),
            );
            rationale.push(
                "Retained CG primary: Gershgorin weakness is warning-level unless reinforced by asymmetry/diagonal failures"
                    .to_string(),
            );
        }
        contract_checks.push(cg_screen.reason.clone());
        contract_checks.push(format!(
            "CG symmetry verdict (base): {}",
            cg_screen.diagnostics.symmetry.verdict
        ));
        if !cg_screen.warnings.is_empty() {
            contract_checks.push(format!(
                "CG soft warnings: {}",
                cg_screen.warnings.join("; ")
            ));
        }
    }

    let compare_structural = compare_structural_cg_screen_enabled() || matrix_name == "add20";
    if compare_structural {
        let cg_screen_structural = cg_compatibility_screen(
            matrix,
            !screen.diagonal_healthy,
            structural_symmetry_hint,
            true,
        );
        let structural_strong_reject_evidence = cg_reject_has_strong_evidence(
            cg_screen_structural
                .diagnostics
                .symmetry
                .symmetry_violation_count,
            cg_screen_structural.diagnostics.non_positive_diagonal_count,
        );
        contract_checks.push(format!(
            "CG structural-compare: base={} (strong_evidence={}) vs metadata-expanded={} (strong_evidence={})",
            if cg_screen.is_hard_reject {
                "hard-reject"
            } else {
                "accept"
            },
            base_strong_reject_evidence,
            if cg_screen_structural.is_hard_reject {
                "hard-reject"
            } else {
                "accept"
            },
            structural_strong_reject_evidence
        ));
        if matrix_name == "add20" {
            let asymmetry_interpretation = if cg_screen.is_hard_reject
                && !base_strong_reject_evidence
                && !cg_screen_structural.is_hard_reject
            {
                "likely structural metadata artifact (base asymmetry warning not confirmed after metadata expansion)"
            } else if cg_screen.is_hard_reject
                && cg_screen_structural.is_hard_reject
                && (base_strong_reject_evidence || structural_strong_reject_evidence)
            {
                "likely true hard reject (strong asymmetry/diagonal evidence persists)"
            } else {
                "inconclusive / warning-only asymmetry evidence"
            };
            contract_checks.push(format!(
                "add20 CG verdicts: base={} | metadata-expanded={} | asymmetry_interpretation={}",
                if cg_screen.is_hard_reject {
                    "hard-reject"
                } else {
                    "accept"
                },
                if cg_screen_structural.is_hard_reject {
                    "hard-reject"
                } else {
                    "accept"
                },
                asymmetry_interpretation
            ));
        }
        if cg_screen.is_hard_reject || cg_screen_structural.is_hard_reject {
            contract_checks.push(format!(
                "CG diagnostics (metadata-expanded): pairs={}, sym_violations={} ({:.2}%), non_positive_diag={}, weak_gershgorin={}, mm_structural_symmetry_hint={:?}",
                cg_screen_structural.diagnostics.symmetry.sampled_pair_count,
                cg_screen_structural.diagnostics.symmetry.symmetry_violation_count,
                100.0 * cg_screen_structural.diagnostics.symmetry.symmetry_violation_rate,
                cg_screen_structural.diagnostics.non_positive_diagonal_count,
                cg_screen_structural.diagnostics.weak_gershgorin_count,
                cg_screen_structural.diagnostics.symmetry.structural_symmetry_hint
            ));
            contract_checks.push(format!(
                "CG symmetry verdict (metadata-expanded): {}",
                cg_screen_structural.diagnostics.symmetry.verdict
            ));
        } else if !cg_screen_structural.warnings.is_empty() {
            contract_checks.push(format!(
                "CG structural soft warnings: {}",
                cg_screen_structural.warnings.join("; ")
            ));
        }
    }

    if primary_pc == "amg" {
        let amg_decision = choose_amg_mode_with_override(
            !screen.diagonal_healthy,
            screen.symmetry_hint,
            screen.spd_like_hint,
            baseline.amg_nonsymmetric_override,
        );
        amg_mode = amg_decision.mode;
        rationale.push(format!("AMG policy: {}", amg_decision.reason));
        match amg_decision.mode {
            AmgMode::DefaultSpd => {}
            AmgMode::ExplicitNonSpd => {
                if primary_solver == "cg" {
                    primary_solver = "gmres".to_string();
                    rationale.push("AMG non-SPD mode requires non-CG Krylov primary".to_string());
                }
            }
            AmgMode::Disabled => {
                primary_pc = if screen.diagonal_healthy {
                    "ilut".to_string()
                } else {
                    "ilu".to_string()
                };
                if primary_solver == "cg" {
                    primary_solver = "gmres".to_string();
                }
            }
        }
    }

    match matrix_name {
        "e05r0100" => {
            primary_solver = "gmres".to_string();
            primary_pc = "none".to_string();
            rationale.push(
                "policy override: benchmark-aligned primary plus full fallback ladder for hard nonsymmetric case"
                    .to_string(),
            );
        }
        "sherman3" => {
            if screen.spd_like_hint
                && screen.symmetry.symmetry_violation_rate > SYMMETRY_MAX_ASYMMETRY_RATE
            {
                panic!(
                    "sherman3 invariant violated: SPD-like hint cannot co-exist with high nonsymmetry (sampled asymmetry {:.2}% > {:.2}%)",
                    100.0 * screen.symmetry.symmetry_violation_rate,
                    100.0 * SYMMETRY_MAX_ASYMMETRY_RATE
                );
            }
            rationale.push(
                "policy override: hard nonsymmetric sequence = primary, rung1 FGMRES(right)+ILUT, rung2 BiCGStab+ILUT"
                    .to_string(),
            );
        }
        "fidap001" => {
            if cg_screen.is_hard_reject {
                primary_solver = "gmres".to_string();
                primary_pc = "ilut".to_string();
                rationale.push(
                    "policy override: CG hard reject; keep nonsymmetric fallback ladder"
                        .to_string(),
                );
            }
        }
        "fidap005" => {
            rationale.push(
                "policy override: prioritize direct-reference verification and retain iterative solve for comparison"
                    .to_string(),
            );
        }
        _ => {}
    }

    if !screen.diagonal_healthy && jacobi_strength_mode == JacobiStrengthMode::Plain {
        if primary_pc == "jacobi" {
            primary_pc = "ilut".to_string();
            rationale.push(
                "diagonal-bad screen: avoided plain Jacobi primary (enable KRYST_DEMO_JACOBI_STRENGTH=fixdiag|rowl1 to allow Jacobi)"
                    .to_string(),
            );
        }
        for step in &mut fallback_ladder {
            if step.pc == "jacobi" {
                step.pc = "ilut".to_string();
                rationale.push(
                    "diagonal-bad screen: avoided plain Jacobi fallback (enable KRYST_DEMO_JACOBI_STRENGTH=fixdiag|rowl1 to allow Jacobi)"
                        .to_string(),
                );
            }
        }
    } else if !screen.diagonal_healthy && jacobi_strength_mode != JacobiStrengthMode::Plain {
        rationale.push(format!(
            "diagonal-bad screen: Jacobi allowed due to strengthened mode {:?}",
            jacobi_strength_mode
        ));
    }

    SelectionDecision {
        primary_solver,
        primary_pc,
        fallback_ladder,
        rationale,
        contract_checks,
        expected_iterations: baseline.expected_iterations,
        amg_mode,
        amg_status_label: if baseline.preconditioner != "amg" {
            "AMG: not used".to_string()
        } else {
            match amg_mode {
                AmgMode::DefaultSpd => "AMG: SPD default mode".to_string(),
                AmgMode::ExplicitNonSpd => "AMG: nonsymmetric override mode".to_string(),
                AmgMode::Disabled => "AMG: disabled (fallback path)".to_string(),
            }
        },
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    fn classify_row_category(outcome_code: &str, fallback: &str, converged: bool) -> &'static str {
        if outcome_code == "DIRECT_TRUTH_PATH" || fallback.contains("DIRECT_LU (truth path)") {
            "verified_by_direct_reference"
        } else if converged {
            "iterative_accepted"
        } else {
            "failed"
        }
    }

    fn format_elapsed_ms(elapsed_seconds: Option<f64>) -> String {
        elapsed_seconds
            .map(|seconds| format!("{:.2} ms", seconds * 1_000.0))
            .unwrap_or_else(|| "-".to_string())
    }

    fn format_iters_time(iters: Option<usize>, elapsed_seconds: Option<f64>) -> String {
        match iters {
            Some(value) => format!("{value} / {}", format_elapsed_ms(elapsed_seconds)),
            None => "- / -".to_string(),
        }
    }

    fn is_root_rank() -> bool {
        // If MPI launcher environment variables are present, only rank 0 prints.
        // In regular (non-MPI) runs no rank variable is present, so default to true.
        let rank = std::env::var("OMPI_COMM_WORLD_RANK")
            .or_else(|_| std::env::var("PMI_RANK"))
            .or_else(|_| std::env::var("MV2_COMM_WORLD_RANK"))
            .or_else(|_| std::env::var("SLURM_PROCID"));

        match rank {
            Ok(value) => value.parse::<i32>() == Ok(0),
            Err(_) => true,
        }
    }

    let is_root_rank = is_root_rank();
    let mut selected_mode = DemoMode::Both;
    let mut verbose_details = false;
    let mut cli_iter = std::env::args().skip(1);
    while let Some(arg) = cli_iter.next() {
        if let Some(raw) = arg.strip_prefix("--mode=") {
            selected_mode =
                DemoMode::from_str(raw).map_err(|e| format!("invalid --mode argument: {e}"))?;
        } else if arg == "--mode" {
            let raw = cli_iter
                .next()
                .ok_or("--mode requires a value: verify | iterative-benchmark | both")?;
            selected_mode =
                DemoMode::from_str(&raw).map_err(|e| format!("invalid --mode argument: {e}"))?;
        } else if arg == "--verbose-details" {
            verbose_details = true;
        } else {
            return Err(format!("unknown argument '{arg}'").into());
        }
    }

    // Initialize logging if available
    #[cfg(feature = "logging")]
    env_logger::init();

    if is_root_rank {
        println!("Optimized Matrix Market Solver Demonstration");
        println!("===========================================");
        println!("Using benchmark-proven optimal configurations");
        println!(
            "Recommended structural-CG compare workflow: KRYST_DEMO_COMPARE_STRUCTURAL_CG_SCREEN=1 cargo run --example optimized_solver_demo --features backend-faer"
        );
        // Note: this demo does not perform distributed coordination unless explicitly added.
        println!();
        println!(
            "Mode selection: {:?} (supported: verify | iterative-benchmark | both)",
            selected_mode
        );
        println!("Verbose details: {}", verbose_details);
        println!();
    }

    let direct_verification_capability = DirectVerificationCapability {
        dense_direct_compiled: cfg!(feature = "dense-direct"),
        policy_allows_direct: global_direct_reference_policy_allows(),
    };

    // Test matrices with known optimal configurations
    let test_matrices = vec![
        ("fidap005", "FIDAP 27x27 structural problem"),
        ("e05r0100", "Driven cavity E05r0100 (236x236)"),
        ("fidap001", "FIDAP 216x216 structural problem"),
        ("sherman3", "Sherman3 sparse matrix (5005x5005)"),
        ("add20", "Add20 matrix (2395x2395)"),
        ("memplus", "Memplus matrix (if available)"),
    ];

    for policy in mode_to_policy(selected_mode) {
        if is_root_rank {
            println!("Reporting mode section: {}", policy.mode_label);
            println!(
                "Semantics: direct truth path {} become winning rung.",
                if policy.allow_truth_path_winner {
                    "can"
                } else {
                    "cannot"
                }
            );
            println!(
                "{:<12} {:<14} {:<20} {:<20} {:<12} {:<12} {:<12} {:<14} {:<26} {}",
                "Matrix",
                "Reference",
                "Baseline iters/time",
                "Best iter iters/time",
                "Ref time",
                "TrueRelRes",
                "RefDiff",
                "Policy rung",
                "Verdict",
                "Benchmark delta"
            );
            println!("{}", "=".repeat(144));
        }
        let mut total_rows = 0usize;
        let mut iterative_success_rows = 0usize;
        let mut verification_only_rows = 0usize;
        let mut failed_rows = 0usize;

        for (matrix_name, _description) in &test_matrices {
            let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let matrix_path = base_dir
                .join("examples")
                .join("mtx")
                .join(format!("{}.mtx", matrix_name));
            let rhs_path = base_dir
                .join("examples")
                .join("mtx")
                .join(format!("{}_rhs1.mtx", matrix_name));

            // Try to read the matrix and RHS
            let (matrix_data, rhs_data) = match (
                read_matrix_market(matrix_path.to_str().unwrap()),
                read_matrix_market(rhs_path.to_str().unwrap()),
            ) {
                (Ok(matrix), Ok(rhs)) => (matrix, rhs),
                _ => {
                    if is_root_rank {
                        println!(
                            "{:<12} {:<14} {:<20} {:<20} {:<12} {:<12} {:<12} {:<14} {:<26} {}",
                            matrix_name,
                            "unavailable",
                            "- / -",
                            "- / -",
                            "-",
                            "-",
                            "-",
                            "-",
                            "failed: files missing",
                            "-"
                        );
                    }
                    total_rows += 1;
                    failed_rows += 1;
                    continue;
                }
            };

            // Keep original operator matrix; build a preconditioner matrix with explicit preprocessing first,
            // then optional diagonal repair as a secondary safeguard.
            let a_op = matrix_data.to_csr_matrix()?;
            let prep_cfg = prep_config_for(matrix_name);
            let conditioning = ConditioningOptions {
                scale: match (prep_cfg.row_scaling, prep_cfg.col_scaling) {
                    (true, true) => Some(ScaleDirection::Both),
                    (true, false) => Some(ScaleDirection::Row),
                    (false, true) => Some(ScaleDirection::Col),
                    (false, false) => None,
                },
                scale_norm: ScaleNorm::Inf,
                ..ConditioningOptions::default()
            };
            let (reordering, reordering_trace, fallback_note) =
                select_reordering_for_preprocessing(&prep_cfg, &a_op);
            let preprocessed = apply_preconditioning_pipeline(&a_op, &conditioning, &reordering)?;
            let (p_candidate, repaired) = repair_diagonal_csr(&preprocessed.matrix, 1e-14, 1e-8);
            let p_mat = Some(p_candidate);
            let preprocessing_meta = preprocessing_metadata_trace(&preprocessed.metadata);
            let prep_trace = if repaired > 0 {
                format!(
                    "{} -> reorder[kind={}, symmetric={}] -> {}{} -> diag_repair[count={}, tol=1e-14, tau=1e-8]",
                    prep_cfg.as_trace(),
                    reordering_trace,
                    reordering.symmetric,
                    preprocessing_meta,
                    fallback_note
                        .as_ref()
                        .map(|note| format!(" -> note[{note}]"))
                        .unwrap_or_default(),
                    repaired
                )
            } else {
                format!(
                    "{} -> reorder[kind={}, symmetric={}] -> {}{} -> diag_repair[count=0]",
                    prep_cfg.as_trace(),
                    reordering_trace,
                    reordering.symmetric,
                    preprocessing_meta,
                    fallback_note
                        .as_ref()
                        .map(|note| format!(" -> note[{note}]"))
                        .unwrap_or_default()
                )
            };
            if is_root_rank && verbose_details {
                println!("    → Preconditioner prep: {prep_trace}.");
            }
            let rhs = rhs_data.to_vector()?;
            let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
            let acceptance_threshold = f64::max(1e-6 * rhs_norm, 1e-12);

            let screen = screen_matrix(&a_op);
            let structural_symmetry_hint = Some(matches!(
                matrix_data.symmetry,
                MatrixMarketSymmetry::Symmetric | MatrixMarketSymmetry::Hermitian
            ));
            let decision =
                select_solver_policy(matrix_name, &screen, &a_op, structural_symmetry_hint);

            // Skip very large matrices for this demo
            if a_op.nrows() > 6000 {
                if is_root_rank {
                    println!(
                        "{:<12} {:<14} {:<20} {:<20} {:<12} {:<12} {:<12} {:<14} {:<26} {}",
                        matrix_name,
                        "skipped",
                        "- / -",
                        "- / -",
                        "-",
                        "-",
                        "-",
                        "-",
                        "failed: too large",
                        "-"
                    );
                }
                total_rows += 1;
                failed_rows += 1;
                continue;
            }

            // Test with optimal configuration
            match test_optimal_solver(
                &a_op,
                p_mat.as_ref(),
                &rhs,
                &decision,
                &screen,
                matrix_name,
                &prep_trace,
                is_root_rank,
                verbose_details,
                policy,
            ) {
                Ok(test_result) => {
                    let primary = test_result.primary_method.as_str();
                    let fallback = test_result.chosen_method.as_str();
                    let solver_reason = test_result.solver_reason.as_str();
                    let outcome_code = test_result.outcome_code.as_str();
                    let converged = test_result.converged;
                    let direct_comparison = test_result
                        .truth_reference
                        .as_ref()
                        .and_then(|truth| truth.comparison.as_ref());
                    let best_iterative = test_result.best_iterative_attempt();
                    let row_category = classify_row_category(outcome_code, fallback, converged);
                    let iterative_row = row_category == "iterative_accepted";
                    let best_iter_label = if iterative_row {
                        best_iterative
                            .map(|attempt| format!("{}+{}", attempt.solver, attempt.preconditioner))
                            .unwrap_or_else(|| "-".to_string())
                    } else {
                        "N/A".to_string()
                    };
                    let baseline_attempt = test_result.baseline_attempt();
                    let baseline_iters_time = format_iters_time(
                        baseline_attempt.map(|attempt| attempt.iterations),
                        baseline_attempt.map(|attempt| attempt.elapsed_seconds),
                    );
                    let best_iters_time = format_iters_time(
                        best_iterative.map(|attempt| attempt.iterations),
                        best_iterative.map(|attempt| attempt.elapsed_seconds),
                    );
                    let reference_time_text = format_elapsed_ms(
                        test_result
                            .truth_reference
                            .as_ref()
                            .and_then(|truth| truth.elapsed_seconds),
                    );
                    let true_rel_res_text = if iterative_row {
                        best_iterative
                            .map(|attempt| format!("{:.2e}", attempt.true_rel_residual))
                            .unwrap_or_else(|| "-".to_string())
                    } else {
                        "N/A".to_string()
                    };
                    let reference = test_result
                        .truth_reference
                        .as_ref()
                        .map(|truth| {
                            if truth.reference_solve_executed {
                                "executed"
                            } else {
                                "side-channel"
                            }
                        })
                        .unwrap_or("none");
                    let ref_diff_text = direct_comparison
                        .map(|cmp| format!("{:.2e}", cmp.rel_error_norm))
                        .unwrap_or_else(|| "-".to_string());
                    let verdict = match row_category {
                        "iterative_accepted" => "iterative accepted".to_string(),
                        "verified_by_direct_reference" => {
                            "verified by direct reference".to_string()
                        }
                        _ => "failed".to_string(),
                    };
                    let benchmark_delta = if iterative_row {
                        best_iterative
                            .map(|attempt| {
                                if let Some(catalog_delta) = compare_best_iterative(
                                    matrix_name,
                                    &attempt.solver,
                                    &attempt.preconditioner,
                                    attempt.iterations,
                                    attempt.accepted,
                                ) {
                                    let expectation = expectation_for(matrix_name)
                                        .expect("benchmark catalog entry should exist");
                                    let method = if catalog_delta.method_family_match {
                                        "method=match"
                                    } else {
                                        "method=mismatch"
                                    };
                                    let range_status = if attempt.iterations
                                        < expectation.iteration_range.min
                                    {
                                        format!(
                                            "iter=below({:+})",
                                            catalog_delta.iteration_delta_low
                                        )
                                    } else if attempt.iterations > expectation.iteration_range.max
                                    {
                                        format!(
                                            "iter=above({:+})",
                                            catalog_delta.iteration_delta_high
                                        )
                                    } else {
                                        "iter=in-range(0)".to_string()
                                    };
                                    let tolerance = if catalog_delta.tolerance_pass {
                                        "tol=pass"
                                    } else {
                                        "tol=fail"
                                    };
                                    let confidence = match expectation.confidence {
                                        ComparisonConfidence::Exact => "exact",
                                        ComparisonConfidence::Approximate => "approx",
                                    };
                                    let time_note = expectation
                                        .time_note
                                        .map(|note| format!("; note={note}"))
                                        .unwrap_or_default();
                                    format!(
                                        "{method}; {range_status}; {tolerance}; conf={confidence}{time_note}"
                                    )
                                } else {
                                    "catalog=missing".to_string()
                                }
                            })
                            .unwrap_or_else(|| "-".to_string())
                    } else {
                        "N/A".to_string()
                    };

                    let verified = format_direct_verification_status(
                        direct_comparison,
                        direct_verification_capability,
                    );
                    let policy_rung = test_result.policy_rung_fidelity();

                    if is_root_rank {
                        println!(
                            "{:<12} {:<14} {:<20} {:<20} {:<12} {:<12} {:<12} {:<14} {:<26} {}",
                            matrix_name,
                            reference,
                            baseline_iters_time,
                            best_iters_time,
                            reference_time_text,
                            true_rel_res_text,
                            ref_diff_text,
                            policy_rung,
                            verdict,
                            benchmark_delta
                        );
                        if verbose_details {
                            println!(
                                "    ↳ diagnostics: primary={primary}, fallback={fallback}, solver_code={solver_reason}, outcome={outcome_code}, ||b||={:.2e}, acceptance_threshold={:.2e}, verified={verified}",
                                rhs_norm, acceptance_threshold
                            );
                            println!(
                                "    → Timing summary: baseline={}, best_iterative={} ({best_iter_label}), reference={}",
                                baseline_iters_time, best_iters_time, reference_time_text
                            );
                            println!("    → Preprocessing trace: {prep_trace}");
                            println!(
                                "    → Attempted rungs: {}",
                                test_result
                                    .attempts
                                    .iter()
                                    .map(|a| format!(
                                        "{}:{}+{}:{}(iters={}, rel={:.2e}, time={})",
                                        a.rung_label,
                                        a.solver,
                                        a.preconditioner,
                                        a.acceptance_status,
                                        a.iterations,
                                        a.true_rel_residual,
                                        format_elapsed_ms(Some(a.elapsed_seconds))
                                    ))
                                    .collect::<Vec<_>>()
                                    .join(" | ")
                            );
                            if let Some(cmp) = direct_comparison {
                                let reference_phase = if cmp.reference_solve_executed {
                                    "executed"
                                } else {
                                    "skipped"
                                };
                                println!(
                                    "    → Direct reference check: status={}, abs_err_norm={:.3e}, rel_diff={:.3e}, matches_verified_answer={}, policy={}",
                                    reference_phase,
                                    cmp.abs_error_norm,
                                    cmp.rel_error_norm,
                                    cmp.matches_verified_answer,
                                    cmp.note
                                );
                                println!(
                                    "    → Direct reference timing: {}",
                                    format_elapsed_ms(cmp.elapsed_seconds)
                                );
                            }
                            println!(
                                "    → Screening details: symmetry_hint={}, spd_like_hint={}, diagonal_healthy={}, density={:.3e}, size_class={}, cond_heuristic={:.2e}, symmetry_verdict={}",
                                screen.symmetry_hint,
                                screen.spd_like_hint,
                                screen.diagonal_healthy,
                                screen.density,
                                screen.size_class,
                                screen.condition_heuristic,
                                screen.symmetry.verdict
                            );
                            println!("    → {}", decision.amg_status_label);
                            println!(
                                "    → Contract checks: {}",
                                decision.contract_checks.join(" | ")
                            );
                        }
                    }

                    total_rows += 1;
                    match row_category {
                        "iterative_accepted" => iterative_success_rows += 1,
                        "verified_by_direct_reference" => verification_only_rows += 1,
                        _ => failed_rows += 1,
                    }

                    // Additional diagnostics for interesting cases
                    if is_root_rank && verbose_details && screen.condition_heuristic >= 100.0 {
                        println!(
                            "    → Ill-conditioned matrix (est. cond. ≈ {:.1e})",
                            screen.condition_heuristic
                        );
                    }
                    if is_root_rank && verbose_details && screen.density > 0.1 {
                        println!(
                            "    → Dense matrix ({:.1}% fill) - direct methods may be preferred",
                            screen.density * 100.0
                        );
                    }
                    if is_root_rank
                        && verbose_details
                        && iterative_row
                        && converged
                        && best_iterative.is_some_and(|attempt| {
                            attempt.iterations < decision.expected_iterations / 2
                        })
                    {
                        println!(
                            "    → Excellent performance: {} iterations vs {} expected",
                            best_iterative
                                .map(|attempt| attempt.iterations)
                                .unwrap_or(0),
                            decision.expected_iterations
                        );
                    }
                }
                Err(e) => {
                    if is_root_rank {
                        println!(
                            "{:<12} {:<14} {:<20} {:<20} {:<12} {:<12} {:<12} {:<14} {:<26} {}",
                            matrix_name,
                            "none",
                            "- / -",
                            "- / -",
                            "-",
                            "-",
                            "-",
                            "none_ok",
                            "failed",
                            e
                        );
                    }
                    total_rows += 1;
                    failed_rows += 1;
                }
            }
        }

        if is_root_rank {
            println!();
            println!("Acceptance Summary [{}]:", policy.mode_label);
            println!("===================");
            println!("• Total rows: {total_rows}");
            println!("• Iterative accepted rows: {iterative_success_rows}");
            println!("• Verification-only rows (direct truth path): {verification_only_rows}");
            println!("• Failed rows: {failed_rows}");
            println!();
        }
    }

    if is_root_rank {
        println!("Benchmark Insights Summary:");
        println!("==========================");
        println!("• Sherman matrices: ILU preconditioning essential for fast convergence");
        println!("• Driven cavity (E-series): GMRES often best, ILU critical for difficult cases");
        println!("• Add20: CG with ILU gives near-optimal 5-iteration convergence");
        println!("• Memplus: BiCGStab + ILU reduces 17,000+ iterations to just 4");
        println!("• General rule: ILU preconditioning dramatically improves robustness");
        println!("• Fallback: GMRES + ILU is reliable for unknown matrices");
        println!();
        println!("Key Performance Factors:");
        println!("• Matrix conditioning: Well-conditioned matrices converge faster");
        println!("• Sparsity pattern: Diagonal dominance helps preconditioner stability");
        println!("• Problem physics: Fluid flow problems often need robust iterative methods");
        println!("• Preconditioner quality: ILU factorization quality affects convergence rate");
    }

    Ok(())
}
