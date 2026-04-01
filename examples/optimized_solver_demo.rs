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
use std::str::FromStr;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::time::Instant;

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
#[cfg(all(
    feature = "backend-faer",
    feature = "dense-direct",
    not(feature = "complex")
))]
use kryst::solver::dense_lu;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::classify_acceptance_status;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::conditioning::analyze_csr;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::matrix_market::read_matrix_market;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::matrix_screening::{lookup_csr, repair_diagonal_csr};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::metrics::true_residual_norm;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::{
    DirectReferenceLike, DirectVerificationCapability, format_direct_verification_status,
};

/// Matrix-specific optimal solver configurations based on benchmark results
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct OptimalConfig {
    solver: &'static str,
    preconditioner: &'static str,
    _description: &'static str,
    expected_iterations: usize,
    fallback_solver: &'static str,
    fallback_pc: &'static str,
    amg_nonsymmetric_override: bool,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct ScreenReport {
    symmetry_hint: bool,
    spd_like_hint: bool,
    diagonal_healthy: bool,
    density: f64,
    size_class: &'static str,
    condition_heuristic: f64,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct SelectionDecision {
    primary_solver: String,
    primary_pc: String,
    fallback_solver: String,
    fallback_pc: String,
    rationale: Vec<String>,
    contract_checks: Vec<String>,
    expected_iterations: usize,
    amg_mode: AmgMode,
    amg_status_label: String,
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
struct CgCompatibility {
    cg_safe: bool,
    reason: String,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct DirectReferenceComparison {
    abs_error_norm: f64,
    rel_error_norm: f64,
    matches_verified_answer: bool,
    note: String,
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
            fallback_pc: "ilu",
            amg_nonsymmetric_override: false,
        },
        "e05r0100" => OptimalConfig {
            solver: "gmres",
            preconditioner: "none",
            _description: "GMRES (no precond) - benchmark baseline for E05r0100",
            expected_iterations: 250,
            fallback_solver: "bicgstab",
            fallback_pc: "ilut",
            amg_nonsymmetric_override: false,
        },
        "fidap001" => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilut",
            _description: "GMRES (ILUT) - nonsymmetric-safe structural default",
            expected_iterations: 500,
            fallback_solver: "bicgstab",
            fallback_pc: "jacobi",
            amg_nonsymmetric_override: false,
        },
        "sherman3" => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilu",
            _description: "GMRES (ILU) - benchmark-preferred for Sherman3",
            expected_iterations: 50,
            fallback_solver: "bicgstab",
            fallback_pc: "ilut",
            amg_nonsymmetric_override: false,
        },
        "add20" => OptimalConfig {
            solver: "cg",
            preconditioner: "ilu",
            _description: "CG (ILU) - benchmark-preferred for Add20",
            expected_iterations: 10,
            fallback_solver: "bicgstab",
            fallback_pc: "ilu",
            amg_nonsymmetric_override: false,
        },
        "memplus" => OptimalConfig {
            solver: "bicgstab",
            preconditioner: "ilu",
            _description: "BiCGStab (ILU) - benchmark-preferred for Memplus",
            expected_iterations: 10,
            fallback_solver: "gmres",
            fallback_pc: "ilu",
            amg_nonsymmetric_override: false,
        },
        _ => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilut",
            _description: "GMRES (ILUT) - general nonsymmetric-safe sparse default",
            expected_iterations: 500,
            fallback_solver: "bicgstab",
            fallback_pc: "jacobi",
            amg_nonsymmetric_override: false,
        },
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn direct_reference_policy(a_mat: &CsrMatrix<f64>) -> (bool, String) {
    let small_matrix_gate = a_mat.nrows() <= 1024 && a_mat.ncols() <= 1024;
    if !small_matrix_gate {
        return (
            false,
            format!(
                "skip: size gate failed ({}x{})",
                a_mat.nrows(),
                a_mat.ncols()
            ),
        );
    }
    let density = a_mat.nnz() as f64 / (a_mat.nrows() * a_mat.ncols()) as f64;
    let dense_threshold = 0.10;
    let env_override = std::env::var("KRYST_ENABLE_DIRECT_REFERENCE")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
    match env_override {
        Some(true) => (true, "env override: forced on".to_string()),
        Some(false) => (false, "env override: forced off".to_string()),
        None if density >= dense_threshold => (
            true,
            format!(
                "auto: density {:.3e} >= {:.3e} and size gate passed",
                density, dense_threshold
            ),
        ),
        None => (
            false,
            format!(
                "auto skip: density {:.3e} < {:.3e} (size gate passed)",
                density, dense_threshold
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
fn compare_with_direct_reference(
    matrix_name: &str,
    a_mat: &CsrMatrix<f64>,
    rhs: &[f64],
    iterative_solution: &[f64],
) -> Result<Option<DirectReferenceComparison>, Box<dyn std::error::Error>> {
    let (enabled, mut policy_note) = direct_reference_policy(a_mat);
    let force_direct = matrix_name == "fidap005";
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
            note: policy_note,
        }));
    }

    #[cfg(feature = "dense-direct")]
    {
        let dense_mat = a_mat.to_dense()?;
        let mut reference_solution = vec![0.0; rhs.len()];
        if let Err(err) = dense_lu::solve(&dense_mat, rhs, &mut reference_solution) {
            return Ok(Some(DirectReferenceComparison {
                abs_error_norm: f64::NAN,
                rel_error_norm: f64::NAN,
                matches_verified_answer: false,
                note: format!("{}; direct LU failed ({err})", policy_note),
            }));
        }

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
            note: format!(
                "{}; direct LU failed (dense-direct feature is disabled)",
                policy_note
            ),
        }))
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn build_preconditioner_options(
    matrix_name: &str,
    solver: &str,
    pc: &str,
    amg_mode: AmgMode,
    is_fallback: bool,
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
    opts.ilu_type = Some(if pc == "ilut" { "ilut" } else { "iluk" }.to_string());
    opts.ilu_level_of_fill = Some(1);
    opts.ilu_max_fill_per_row = Some(24);
    opts.ilu_offdiag_drop_tolerance = Some(1e-4);
    opts.ilu_reordering_type = Some("rcm".to_string());
    opts.ilu_reordering = Some("rcm".to_string());
    opts.ilu_triangular_solve = Some("gauss-seidel".to_string());
    opts.ilu_pivot_threshold = Some(1e-10);
    opts.pc_scale = Some("both".to_string());
    opts.pc_scale_norm = Some("inf".to_string());

    match matrix_name {
        "e05r0100" if is_fallback && pc == "ilut" => {
            opts.ilu_max_fill_per_row = Some(48);
            opts.ilu_offdiag_drop_tolerance = Some(1e-6);
            opts.ilu_reordering_type = Some("amd".to_string());
            opts.ilu_reordering = Some("amd_nonsym".to_string());
            opts.ilu_triangular_solve = Some("exact".to_string());
            opts.ilu_pivot_threshold = Some(1e-12);
        }
        "sherman3" => {
            if is_fallback {
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
            opts.pc_scale = Some("row".to_string());
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
    matrix_name: &str,
    is_root_rank: bool,
) -> Result<
    (
        String,
        String,
        String,
        String,
        f64,
        f64,
        usize,
        String,
        bool,
        Option<DirectReferenceComparison>,
    ),
    Box<dyn std::error::Error>,
> {
    let _ = matrix_name;
    let mut solution = vec![0.0; rhs.len()];

    // Sparse-first path: keep A/P in CSR form so AMG/ILU/Jacobi comparisons reflect
    // actual sparse operator semantics used by iterative methods.
    let csr_a_matrix = Arc::new(a_mat.clone());
    let a_op: Arc<dyn LinOp<S = f64>> = Arc::new(CsrOp::new(Arc::clone(&csr_a_matrix)));
    // Optional repaired/scaled matrix for preconditioner construction only (also sparse).
    let p_op: Option<Arc<dyn LinOp<S = f64>>> = match p_mat {
        Some(p) => Some(Arc::new(CsrOp::new(Arc::new(p.clone())))),
        None => None,
    };
    let rhs_vec = rhs.to_vec();

    // Try primary configuration
    let mut ksp = KspContext::new();
    let st = SolverType::from_str(&decision.primary_solver)?;
    let pct = PcType::from_str(&decision.primary_pc)?;
    let primary_opts = build_preconditioner_options(
        matrix_name,
        &decision.primary_solver,
        &decision.primary_pc,
        decision.amg_mode,
        false,
    );
    let primary_opts_ref = primary_opts.as_ref();
    ksp.set_type(st)?
        .set_pc_type(pct, primary_opts_ref)?
        .set_tolerances(1e-6, 1e-12, 1e3, 1000);

    // Solve and residual checks both use the same A_op.
    ksp.set_operators(Arc::clone(&a_op), p_op.clone());
    ksp.setup()?;

    let result = ksp.solve(&rhs_vec, &mut solution);

    fn classify_failure(msg: &str) -> &'static str {
        let m = msg.to_ascii_lowercase();
        if m.contains("contract") || m.contains("cg rejected") || m.contains("wrong method") {
            "contract_mismatch"
        } else if m.contains("breakdown")
            || m.contains("nan")
            || m.contains("inf")
            || m.contains("zero pivot")
            || m.contains("singular")
        {
            "breakdown"
        } else if m.contains("stagnat")
            || m.contains("max iter")
            || m.contains("did not converge")
            || m.contains("no convergence")
        {
            "stagnated"
        } else {
            "failed"
        }
    }

    fn needs_fallback(
        stats: &kryst::utils::convergence::SolveStats<f64>,
        true_residual: f64,
        tol: f64,
    ) -> bool {
        !classify_acceptance_status(stats.reason, true_residual, tol).is_accepted()
    }

    match result {
        Ok(stats) => {
            let true_residual = true_residual_norm(a_op.as_ref(), &rhs_vec, &solution);
            let primary_method = format!(
                "{} + {}",
                decision.primary_solver.to_uppercase(),
                decision.primary_pc.to_uppercase()
            );
            if needs_fallback(&stats, true_residual, 1e-6) {
                let primary_reason = format!(
                    "soft failure: reason={:?}, true_residual={:.3e} (tol=1.000e-6), internal_classical_retry={}",
                    stats.reason, true_residual, stats.gmres_classical_retry
                );
                let primary_failure =
                    classify_acceptance_status(stats.reason, true_residual, 1e-6).as_str();
                if is_root_rank {
                    println!(
                        "    Primary method did not meet convergence contract, trying fallback..."
                    );
                }
                let mut solution_fallback = vec![0.0; rhs.len()];

                let mut ksp_fallback = KspContext::new();
                let st_fb = SolverType::from_str(&decision.fallback_solver)?;
                let pc_fb = PcType::from_str(&decision.fallback_pc)?;
                let fallback_opts = build_preconditioner_options(
                    matrix_name,
                    &decision.fallback_solver,
                    &decision.fallback_pc,
                    AmgMode::Disabled,
                    true,
                );
                ksp_fallback
                    .set_type(st_fb)?
                    .set_pc_type(pc_fb, fallback_opts.as_ref())?
                    .set_tolerances(1e-6, 1e-12, 1e3, 1000);
                ksp_fallback.set_operators(Arc::clone(&a_op), p_op.clone());
                ksp_fallback.setup()?;

                let start_fallback = Instant::now();
                let fallback_solve = ksp_fallback.solve(&rhs_vec, &mut solution_fallback);
                let solve_time_fallback = start_fallback.elapsed().as_secs_f64();
                let fallback_method = format!(
                    "{} + {}",
                    decision.fallback_solver.to_uppercase(),
                    decision.fallback_pc.to_uppercase()
                );
                match fallback_solve {
                    Ok(stats_fallback) => {
                        let true_residual =
                            true_residual_norm(a_op.as_ref(), &rhs_vec, &solution_fallback);
                        let acceptance_status =
                            classify_acceptance_status(stats_fallback.reason, true_residual, 1e-6);
                        let status = acceptance_status.as_str().to_string();
                        let converged = acceptance_status.is_accepted();
                        let fallback_outcome = if acceptance_status.is_accepted() {
                            "succeeded"
                        } else {
                            "completed"
                        };
                        let solver_reason = stats_fallback.reason.petsc_reason().to_string();
                        let reason = format!(
                            "solver_reason={} | primary {}: {}; fallback {} with {} (internal_classical_retry={})",
                            solver_reason,
                            primary_failure,
                            primary_reason,
                            fallback_outcome,
                            fallback_method,
                            stats_fallback.gmres_classical_retry
                        );
                        let direct_comparison = compare_with_direct_reference(
                            matrix_name,
                            a_mat,
                            &rhs_vec,
                            &solution_fallback,
                        )?;
                        let _ = solve_time_fallback;
                        Ok((
                            primary_method,
                            fallback_method,
                            solver_reason,
                            reason,
                            true_residual,
                            stats_fallback.final_residual,
                            stats_fallback.iterations,
                            status,
                            converged,
                            direct_comparison,
                        ))
                    }
                    Err(fallback_err) => {
                        let fallback_reason = fallback_err.to_string();
                        let fallback_failure = classify_failure(&fallback_reason);
                        let status = if primary_failure == "contract_mismatch" {
                            "contract_mismatch".to_string()
                        } else {
                            fallback_failure.to_string()
                        };
                        Ok((
                            primary_method,
                            fallback_method,
                            "N/A".to_string(),
                            format!(
                                "primary {}: {}; fallback {}: {}",
                                primary_failure, primary_reason, fallback_failure, fallback_reason
                            ),
                            f64::NAN,
                            f64::NAN,
                            0,
                            status,
                            false,
                            None,
                        ))
                    }
                }
            } else {
                let acceptance_status =
                    classify_acceptance_status(stats.reason, true_residual, 1e-6);
                let status = acceptance_status.as_str().to_string();
                let solver_reason = stats.reason.petsc_reason().to_string();
                let reason = format!(
                    "solver_reason={} | internal_classical_retry={}",
                    solver_reason, stats.gmres_classical_retry
                );
                let direct_comparison =
                    compare_with_direct_reference(matrix_name, a_mat, &rhs_vec, &solution)?;
                Ok((
                    primary_method,
                    "-".to_string(),
                    solver_reason,
                    reason,
                    true_residual,
                    stats.final_residual,
                    stats.iterations,
                    status,
                    acceptance_status.is_accepted(),
                    direct_comparison,
                ))
            }
        }
        Err(primary_err) => {
            let primary_reason = primary_err.to_string();
            let primary_failure = classify_failure(&primary_reason).to_string();
            let primary_method = format!(
                "{} + {}",
                decision.primary_solver.to_uppercase(),
                decision.primary_pc.to_uppercase()
            );
            // Try fallback configuration
            if is_root_rank {
                println!("    Primary method failed, trying fallback...");
            }
            let mut solution_fallback = vec![0.0; rhs.len()];

            let mut ksp_fallback = KspContext::new();
            let st_fb = SolverType::from_str(&decision.fallback_solver)?;
            let pc_fb = PcType::from_str(&decision.fallback_pc)?;
            let fallback_opts = build_preconditioner_options(
                matrix_name,
                &decision.fallback_solver,
                &decision.fallback_pc,
                AmgMode::Disabled,
                true,
            );
            ksp_fallback
                .set_type(st_fb)?
                .set_pc_type(pc_fb, fallback_opts.as_ref())?
                .set_tolerances(1e-6, 1e-12, 1e3, 1000);
            ksp_fallback.set_operators(Arc::clone(&a_op), p_op);
            ksp_fallback.setup()?;

            let start_fallback = Instant::now();
            let fallback_solve = ksp_fallback.solve(&rhs_vec, &mut solution_fallback);
            let solve_time_fallback = start_fallback.elapsed().as_secs_f64();
            let fallback_method = format!(
                "{} + {}",
                decision.fallback_solver.to_uppercase(),
                decision.fallback_pc.to_uppercase()
            );
            match fallback_solve {
                Ok(stats_fallback) => {
                    let true_residual =
                        true_residual_norm(a_op.as_ref(), &rhs_vec, &solution_fallback);
                    let acceptance_status =
                        classify_acceptance_status(stats_fallback.reason, true_residual, 1e-6);
                    let status = acceptance_status.as_str().to_string();
                    let converged = acceptance_status.is_accepted();
                    let fallback_outcome = if acceptance_status.is_accepted() {
                        "succeeded"
                    } else {
                        "completed"
                    };
                    let solver_reason = stats_fallback.reason.petsc_reason().to_string();
                    let reason = format!(
                        "solver_reason={} | primary {}: {}; fallback {} with {} (internal_classical_retry={})",
                        solver_reason,
                        primary_failure,
                        primary_reason,
                        fallback_outcome,
                        fallback_method,
                        stats_fallback.gmres_classical_retry
                    );
                    let direct_comparison = compare_with_direct_reference(
                        matrix_name,
                        a_mat,
                        &rhs_vec,
                        &solution_fallback,
                    )?;
                    let _ = solve_time_fallback;
                    Ok((
                        primary_method,
                        fallback_method,
                        solver_reason,
                        reason,
                        true_residual,
                        stats_fallback.final_residual,
                        stats_fallback.iterations,
                        status,
                        converged,
                        direct_comparison,
                    ))
                }
                Err(fallback_err) => {
                    let fallback_reason = fallback_err.to_string();
                    let fallback_failure = classify_failure(&fallback_reason);
                    let status = if primary_failure == "contract_mismatch" {
                        "contract_mismatch".to_string()
                    } else {
                        fallback_failure.to_string()
                    };
                    Ok((
                        primary_method,
                        fallback_method,
                        "N/A".to_string(),
                        format!(
                            "primary {}: {}; fallback {}: {}",
                            primary_failure, primary_reason, fallback_failure, fallback_reason
                        ),
                        f64::NAN,
                        f64::NAN,
                        0,
                        status,
                        false,
                        None,
                    ))
                }
            }
        }
    }
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn cg_compatibility_screen(matrix: &CsrMatrix<f64>, diag_issues: bool) -> CgCompatibility {
    let n = matrix.nrows().min(matrix.ncols());
    let sample_rows = n.min(1024);
    let symmetry_tol = 1e-7;

    let mut sampled_pairs = 0usize;
    let mut symmetry_violations = 0usize;
    let mut non_positive_diagonal = 0usize;
    let mut weak_gershgorin_rows = 0usize;

    for i in 0..sample_rows {
        let (cols, vals) = matrix.row(i);
        let mut row_abs_offdiag_sum = 0.0;
        let mut diag = None;

        for (&j, &a_ij) in cols.iter().zip(vals.iter()) {
            if j == i {
                diag = Some(a_ij);
                continue;
            }
            row_abs_offdiag_sum += a_ij.abs();
            if j < sample_rows {
                sampled_pairs += 1;
                let a_ji = lookup_csr(matrix, j, i).unwrap_or(0.0);
                if (a_ij - a_ji).abs() > symmetry_tol {
                    symmetry_violations += 1;
                }
            }
        }

        let d = diag.unwrap_or(0.0);
        if d <= 0.0 {
            non_positive_diagonal += 1;
        }
        if d <= row_abs_offdiag_sum {
            weak_gershgorin_rows += 1;
        }
    }

    let symmetry_ok = sampled_pairs == 0 || symmetry_violations * 100 <= sampled_pairs;
    let diag_ok = non_positive_diagonal == 0 && !diag_issues;
    let gershgorin_ok = weak_gershgorin_rows * 5 <= sample_rows.max(1); // <=20% weak rows
    let cg_safe = symmetry_ok && diag_ok && gershgorin_ok;

    if cg_safe {
        return CgCompatibility {
            cg_safe: true,
            reason: format!(
                "CG contract accepted: sampled symmetry/SPD heuristic passed (sym diff {:.2}%, weak Gershgorin rows {}/{}, diag issues: {})",
                if sampled_pairs > 0 {
                    100.0 * symmetry_violations as f64 / sampled_pairs as f64
                } else {
                    0.0
                },
                weak_gershgorin_rows,
                sample_rows,
                if diag_issues { "yes" } else { "no" }
            ),
        };
    }

    let mut causes = Vec::new();
    if !symmetry_ok {
        causes.push(format!(
            "sampled nonsymmetry {:.2}%>{:.2}%",
            100.0 * symmetry_violations as f64 / sampled_pairs.max(1) as f64,
            1.0
        ));
    }
    if !diag_ok {
        causes.push(format!(
            "non-positive/missing diagonal rows {} + diag_issues={}",
            non_positive_diagonal, diag_issues
        ));
    }
    if !gershgorin_ok {
        causes.push(format!(
            "weak Gershgorin rows {}/{}",
            weak_gershgorin_rows, sample_rows
        ));
    }

    CgCompatibility {
        cg_safe: false,
        reason: format!(
            "wrong method for matrix contract: CG rejected by compatibility screen ({})",
            causes.join("; ")
        ),
    }
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
    let approx_symmetric = stats
        .symmetry_estimate
        .map(|sym| sym >= 0.99)
        .unwrap_or(false);
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
fn select_solver_policy(
    matrix_name: &str,
    screen: &ScreenReport,
    matrix: &CsrMatrix<f64>,
) -> SelectionDecision {
    let baseline = get_optimal_config(matrix_name);
    let mut primary_solver = baseline.solver.to_string();
    let mut primary_pc = baseline.preconditioner.to_string();
    let mut fallback_solver = baseline.fallback_solver.to_string();
    let mut fallback_pc = baseline.fallback_pc.to_string();
    let mut rationale = vec![format!(
        "matrix hint '{}': {}",
        matrix_name, baseline._description
    )];
    let mut contract_checks = Vec::new();
    let mut amg_mode = AmgMode::Disabled;
    let jacobi_strength_mode = jacobi_strength_mode_from_env();

    let cg_screen = cg_compatibility_screen(matrix, !screen.diagonal_healthy);
    if primary_solver == "cg" && !cg_screen.cg_safe {
        let safe_solver =
            if baseline.fallback_solver == "gmres" || baseline.fallback_solver == "bicgstab" {
                baseline.fallback_solver
            } else {
                "gmres"
            };
        primary_solver = safe_solver.to_string();
        primary_pc = "ilut".to_string();
        contract_checks.push(cg_screen.reason);
        rationale.push(format!(
            "CG screened out; switched primary to {} + ILUT",
            safe_solver.to_uppercase()
        ));
    } else {
        contract_checks.push(cg_screen.reason);
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
            fallback_solver = "bicgstab".to_string();
            fallback_pc = "ilut".to_string();
            rationale.push(
                "policy override: keep GMRES+NONE baseline and route fallback to tuned BiCGStab+ILUT"
                    .to_string(),
            );
        }
        "sherman3" => {
            fallback_solver = "bicgstab".to_string();
            fallback_pc = "ilut".to_string();
            rationale.push(
                "policy override: on stagnation promote from baseline ILU to stronger ILUT profile"
                    .to_string(),
            );
        }
        "fidap001" => {
            if !cg_screen.cg_safe {
                primary_solver = "gmres".to_string();
                primary_pc = "ilut".to_string();
                fallback_solver = "bicgstab".to_string();
                fallback_pc = "ilut".to_string();
                rationale.push(
                    "policy override: CG screen failed; go directly to tuned nonsymmetric ILUT path"
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
        if fallback_pc == "jacobi" {
            fallback_pc = "ilut".to_string();
            rationale.push(
                "diagonal-bad screen: avoided plain Jacobi fallback (enable KRYST_DEMO_JACOBI_STRENGTH=fixdiag|rowl1 to allow Jacobi)"
                    .to_string(),
            );
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
        fallback_solver,
        fallback_pc,
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

    // Initialize logging if available
    #[cfg(feature = "logging")]
    env_logger::init();

    if is_root_rank {
        println!("Optimized Matrix Market Solver Demonstration");
        println!("===========================================");
        println!("Using benchmark-proven optimal configurations");
        // Note: this demo does not perform distributed coordination unless explicitly added.
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

    if is_root_rank {
        println!(
            "{:<15} {:<22} {:<22} {:<28} {:<28} {:<12} {:<12} {:<10} {:<18} {:<10} {}",
            "Matrix",
            "Primary",
            "Fallback",
            "SolverReason",
            "Reason",
            "TrueRes",
            "StatsRes",
            "Iterations",
            "AcceptanceStatus",
            "Verified?",
            "Performance vs Benchmark"
        );
        println!("{}", "=".repeat(210));
    }

    for (matrix_name, _description) in test_matrices {
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
                        "{:<15} {:<22} {:<22} {:<28} {:<28} {:<12} {:<12} {:<10} {:<18} {:<10} {}",
                        matrix_name,
                        "-",
                        "-",
                        "-",
                        "files_missing",
                        "N/A",
                        "N/A",
                        "N/A",
                        "failed",
                        "N/A",
                        "-"
                    );
                }
                continue;
            }
        };

        // Keep original operator matrix; optionally build repaired matrix for preconditioner setup only.
        let a_op = matrix_data.to_csr_matrix()?;
        let (p_candidate, repaired) = repair_diagonal_csr(&a_op, 1e-14, 1e-8);
        let p_mat = if repaired > 0 {
            Some(p_candidate)
        } else {
            None
        };
        if is_root_rank && repaired > 0 {
            println!(
                "    → Preconditioner matrix repaired: {repaired} diagonal entries (|diag|<=1e-14 or missing)."
            );
        }
        let rhs = rhs_data.to_vector()?;

        let screen = screen_matrix(&a_op);
        let decision = select_solver_policy(matrix_name, &screen, &a_op);

        // Skip very large matrices for this demo
        if a_op.nrows() > 6000 {
            if is_root_rank {
                println!(
                    "{:<15} {:<22} {:<22} {:<28} {:<28} {:<12} {:<12} {:<10} {:<18} {:<10} {}",
                    matrix_name,
                    "-",
                    "-",
                    "-",
                    "too_large",
                    "N/A",
                    "N/A",
                    "SKIP",
                    "failed",
                    "N/A",
                    "-"
                );
            }
            continue;
        }

        // Test with optimal configuration
        match test_optimal_solver(
            &a_op,
            p_mat.as_ref(),
            &rhs,
            &decision,
            matrix_name,
            is_root_rank,
        ) {
            Ok((
                primary,
                fallback,
                solver_reason,
                reason,
                true_residual,
                prec_residual,
                iters,
                status,
                converged,
                direct_comparison,
            )) => {
                // Compare with benchmark expectations
                let iter_performance = if converged && iters <= decision.expected_iterations {
                    "Better than expected"
                } else if converged && iters <= decision.expected_iterations * 2 {
                    "Within expected range"
                } else if converged {
                    "Slower than expected"
                } else {
                    "-"
                };

                let verified = format_direct_verification_status(
                    direct_comparison.as_ref(),
                    direct_verification_capability,
                );

                if is_root_rank {
                    println!(
                        "{:<15} {:<22} {:<22} {:<28} {:<28} {:<12.2e} {:<12.2e} {:<10} {:<18} {:<10} {}",
                        matrix_name,
                        primary,
                        fallback,
                        solver_reason,
                        reason,
                        true_residual,
                        prec_residual,
                        iters,
                        status,
                        verified,
                        iter_performance
                    );
                    if let Some(cmp) = direct_comparison.as_ref() {
                        println!(
                            "    → Direct reference check: abs_err_norm={:.3e}, rel_diff={:.3e}, matches_verified_answer={}, policy={}",
                            cmp.abs_error_norm,
                            cmp.rel_error_norm,
                            cmp.matches_verified_answer,
                            cmp.note
                        );
                    }
                    println!(
                        "    → Screen: symmetry_hint={}, spd_like_hint={}, diagonal_healthy={}, density={:.3e}, size_class={}, cond_heuristic={:.2e}",
                        screen.symmetry_hint,
                        screen.spd_like_hint,
                        screen.diagonal_healthy,
                        screen.density,
                        screen.size_class,
                        screen.condition_heuristic
                    );
                    println!("    → {}", decision.amg_status_label);
                    println!(
                        "    → Decision rationale: {}",
                        decision.rationale.join(" | ")
                    );
                    println!(
                        "    → Contract checks: {}",
                        decision.contract_checks.join(" | ")
                    );
                }

                // Additional diagnostics for interesting cases
                if is_root_rank && screen.condition_heuristic >= 100.0 {
                    println!(
                        "    → Ill-conditioned matrix (est. cond. ≈ {:.1e})",
                        screen.condition_heuristic
                    );
                }
                if is_root_rank && screen.density > 0.1 {
                    println!(
                        "    → Dense matrix ({:.1}% fill) - direct methods may be preferred",
                        screen.density * 100.0
                    );
                }
                if is_root_rank && converged && iters < decision.expected_iterations / 2 {
                    println!(
                        "    → Excellent performance: {} iterations vs {} expected",
                        iters, decision.expected_iterations
                    );
                }
            }
            Err(e) => {
                if is_root_rank {
                    println!(
                        "{:<15} {:<22} {:<22} {:<28} {:<28} {:<12} {:<12} {:<10} {:<18} {:<10} {}",
                        matrix_name,
                        "-",
                        "-",
                        "-",
                        "failed",
                        "N/A",
                        "N/A",
                        "N/A",
                        "failed",
                        "N/A",
                        e
                    );
                }
            }
        }
    }

    if is_root_rank {
        println!();
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
