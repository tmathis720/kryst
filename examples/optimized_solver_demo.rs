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
use kryst::matrix::op::{CsrOp, DenseOp, LinOp};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::matrix::sparse::CsrMatrix;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use kryst::utils::matrix_market::read_matrix_market;

/// Matrix-specific optimal solver configurations based on benchmark results
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
struct OptimalConfig {
    solver: &'static str,
    preconditioner: &'static str,
    _description: &'static str,
    expected_iterations: usize,
    fallback_solver: &'static str,
    fallback_pc: &'static str,
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
        },
        "e05r0100" => OptimalConfig {
            solver: "gmres",
            preconditioner: "none",
            _description: "GMRES (no precond) - benchmark baseline for E05r0100",
            expected_iterations: 250,
            fallback_solver: "bicgstab",
            fallback_pc: "ilut",
        },
        "fidap001" => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilut",
            _description: "GMRES (ILUT) - nonsymmetric-safe structural default",
            expected_iterations: 500,
            fallback_solver: "bicgstab",
            fallback_pc: "jacobi",
        },
        "sherman3" => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilu",
            _description: "GMRES (ILU) - benchmark-preferred for Sherman3",
            expected_iterations: 50,
            fallback_solver: "bicgstab",
            fallback_pc: "ilu",
        },
        "add20" => OptimalConfig {
            solver: "cg",
            preconditioner: "ilu",
            _description: "CG (ILU) - benchmark-preferred for Add20",
            expected_iterations: 10,
            fallback_solver: "bicgstab",
            fallback_pc: "ilu",
        },
        "memplus" => OptimalConfig {
            solver: "bicgstab",
            preconditioner: "ilu",
            _description: "BiCGStab (ILU) - benchmark-preferred for Memplus",
            expected_iterations: 10,
            fallback_solver: "gmres",
            fallback_pc: "ilu",
        },
        _ => OptimalConfig {
            solver: "gmres",
            preconditioner: "ilut",
            _description: "GMRES (ILUT) - general nonsymmetric-safe sparse default",
            expected_iterations: 500,
            fallback_solver: "bicgstab",
            fallback_pc: "jacobi",
        },
    }
}

/// Test a solver configuration and return detailed results
#[cfg(not(feature = "complex"))]
// Use the true residual to gauge accuracy in the original system rather than preconditioned space.
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn true_residual_norm(op: &dyn LinOp<S = f64>, rhs: &[f64], solution: &[f64]) -> f64 {
    let mut ax = vec![0.0; rhs.len()];
    op.matvec(solution, &mut ax);
    let mut norm_sq = 0.0;
    for (b, ax_i) in rhs.iter().zip(ax.iter()) {
        let r = b - ax_i;
        norm_sq += r * r;
    }
    norm_sq.sqrt()
}

/// Test a solver configuration and return detailed results
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn test_optimal_solver(
    a_mat: &CsrMatrix<f64>,
    p_mat: Option<&CsrMatrix<f64>>,
    rhs: &[f64],
    decision: &SelectionDecision,
    _matrix_name: &str,
    is_root_rank: bool,
) -> Result<(String, String, String, f64, f64, usize, String, bool), Box<dyn std::error::Error>> {
    let mut solution = vec![0.0; rhs.len()];

    // Sparse-first path: keep A/P in CSR form so AMG/ILU/Jacobi comparisons reflect
    // actual sparse operator semantics used by iterative methods.
    let csr_a_matrix = Arc::new(a_mat.clone());
    let mut a_op: Arc<dyn LinOp<S = f64>> = Arc::new(CsrOp::new(Arc::clone(&csr_a_matrix)));
    // Optional repaired/scaled matrix for preconditioner construction only (also sparse).
    let p_op: Option<Arc<dyn LinOp<S = f64>>> = match p_mat {
        Some(p) => Some(Arc::new(CsrOp::new(Arc::new(p.clone())))),
        None => None,
    };
    // Dense/direct reference path is intentionally opt-in and size-gated to avoid
    // forcing densification in the normal iterative benchmark workflow.
    let allow_direct_reference = std::env::var("KRYST_ENABLE_DIRECT_REFERENCE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    let small_matrix_for_direct = a_mat.nrows() <= 1024 && a_mat.ncols() <= 1024;
    if allow_direct_reference && small_matrix_for_direct {
        let dense_a_matrix = Arc::new(a_mat.to_dense()?);
        a_op = Arc::new(DenseOp::<f64>::new(dense_a_matrix));
    }
    let rhs_vec = rhs.to_vec();

    // Try primary configuration
    let mut ksp = KspContext::new();
    let st = SolverType::from_str(&decision.primary_solver)?;
    let pct = PcType::from_str(&decision.primary_pc)?;
    let amg_opts = if decision.primary_pc == "amg" && decision.primary_solver != "cg" {
        let mut opts = PcOptions::default();
        opts.amg_require_spd = Some(false);
        opts.amg_relax_type = Some("chebyshev".into());
        Some(opts)
    } else {
        None
    };
    let amg_opts_ref = amg_opts.as_ref();
    ksp.set_type(st)?
        .set_pc_type(pct, amg_opts_ref)?
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

    match result {
        Ok(stats) => {
            let true_residual = true_residual_norm(a_op.as_ref(), &rhs_vec, &solution);
            let converged = true_residual < 1e-6;
            let primary_method = format!(
                "{} + {}",
                decision.primary_solver.to_uppercase(),
                decision.primary_pc.to_uppercase()
            );
            let status = if converged { "ok" } else { "stagnated" }.to_string();
            Ok((
                primary_method,
                "-".to_string(),
                "-".to_string(),
                true_residual,
                stats.final_residual,
                stats.iterations,
                status,
                converged,
            ))
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
            ksp_fallback
                .set_type(st_fb)?
                .set_pc_type(pc_fb, None)?
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
                    let converged = true_residual < 1e-6;
                    let status = if converged { "ok" } else { "stagnated" }.to_string();
                    let reason = format!(
                        "primary {}: {}; fallback succeeded with {}",
                        primary_failure, primary_reason, fallback_method
                    );
                    let _ = solve_time_fallback;
                    Ok((
                        primary_method,
                        fallback_method,
                        reason,
                        true_residual,
                        stats_fallback.final_residual,
                        stats_fallback.iterations,
                        status,
                        converged,
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
                        format!(
                            "primary {}: {}; fallback {}: {}",
                            primary_failure, primary_reason, fallback_failure, fallback_reason
                        ),
                        f64::NAN,
                        f64::NAN,
                        0,
                        status,
                        false,
                    ))
                }
            }
        }
    }
}

/// Analyze matrix properties for diagnostics
#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn analyze_matrix_properties(matrix: &CsrMatrix<f64>) -> (f64, f64, bool) {
    let n = matrix.nrows();
    let nnz = matrix.nnz();
    let density = nnz as f64 / (n * n) as f64;

    // Estimate condition number from diagonal dominance (rough heuristic)
    let mut min_diag = f64::INFINITY;
    let mut max_off_diag_sum: f64 = 0.0;

    if n < 2000 {
        // Only for reasonably sized matrices
        if let Ok(dense) = matrix.to_dense() {
            for i in 0..n {
                let diag_val = dense[(i, i)].abs();
                if diag_val > 0.0 {
                    min_diag = min_diag.min(diag_val);
                }

                let mut off_diag_sum = 0.0;
                for j in 0..n {
                    if i != j {
                        off_diag_sum += dense[(i, j)].abs();
                    }
                }
                max_off_diag_sum = max_off_diag_sum.max(off_diag_sum);
            }
        }
    }

    let condition_estimate = if min_diag > 0.0 && min_diag.is_finite() {
        max_off_diag_sum / min_diag
    } else {
        f64::INFINITY
    };

    let is_well_conditioned = condition_estimate < 100.0;

    (density, condition_estimate, is_well_conditioned)
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
                let a_ji = lookup(matrix, j, i).unwrap_or(0.0);
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
fn repair_diagonal_csr(a: &CsrMatrix<f64>, tol: f64, tau: f64) -> (CsrMatrix<f64>, usize) {
    let nrows = a.nrows();
    let ncols = a.ncols();

    let mut rp: Vec<usize> = Vec::with_capacity(nrows + 1);
    let mut ci: Vec<usize> = Vec::with_capacity(a.nnz() + nrows);
    let mut vv: Vec<f64> = Vec::with_capacity(a.nnz() + nrows);

    rp.push(0);
    let mut fixed = 0usize;

    for i in 0..nrows {
        let (cols, vals) = a.row(i);
        let row_abs_sum: f64 = vals.iter().map(|x| x.abs()).sum();
        let repl = (tau * row_abs_sum).max(tol);
        let mut diag_handled = false;

        for (&c, &v) in cols.iter().zip(vals.iter()) {
            if !diag_handled && i < ncols && c > i {
                ci.push(i);
                vv.push(repl);
                fixed += 1;
                diag_handled = true;
            }

            if c == i {
                let new_v = if v.abs() <= tol {
                    fixed += 1;
                    repl
                } else {
                    v
                };
                ci.push(c);
                vv.push(new_v);
                diag_handled = true;
            } else {
                ci.push(c);
                vv.push(v);
            }
        }

        if !diag_handled && i < ncols {
            ci.push(i);
            vv.push(repl);
            fixed += 1;
        }

        rp.push(ci.len());
    }

    (CsrMatrix::from_csr(nrows, ncols, rp, ci, vv), fixed)
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn detect_diag_issues(matrix: &CsrMatrix<f64>, tol: f64, max_rows: usize) -> bool {
    let n = matrix.nrows().min(matrix.ncols());
    let limit = n.min(max_rows);
    for i in 0..limit {
        match lookup(matrix, i, i) {
            Some(val) if val.abs() > tol => continue,
            _ => return true,
        }
    }
    false
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn is_approximately_symmetric(matrix: &CsrMatrix<f64>, tol: f64, max_rows: usize) -> bool {
    let n = matrix.nrows().min(matrix.ncols());
    let limit = n.min(max_rows);
    for i in 0..limit {
        let (cols, vals) = matrix.row(i);
        for (&j, &a_ij) in cols.iter().zip(vals.iter()) {
            if j >= limit {
                continue;
            }
            let a_ji = lookup(matrix, j, i).unwrap_or(0.0);
            if (a_ij - a_ji).abs() > tol {
                return false;
            }
        }
    }
    true
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn has_positive_diagonal(matrix: &CsrMatrix<f64>, tol: f64, max_rows: usize) -> bool {
    let n = matrix.nrows().min(matrix.ncols());
    let limit = n.min(max_rows);
    for i in 0..limit {
        match lookup(matrix, i, i) {
            Some(val) if val > tol => continue,
            _ => return false,
        }
    }
    true
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn choose_amg_mode(matrix: &CsrMatrix<f64>, diag_issues: bool) -> AmgDecision {
    let approx_symmetric = is_approximately_symmetric(matrix, 1e-8, 4_000);
    let positive_diag = has_positive_diagonal(matrix, 1e-14, 20_000);
    let spd_like = !diag_issues && approx_symmetric && positive_diag;
    if spd_like {
        return AmgDecision {
            mode: AmgMode::DefaultSpd,
            reason: "accepted: SPD-like screen passed (symmetric sample + positive diagonal + no diagonal issues)".to_string(),
        };
    }

    if !diag_issues && approx_symmetric {
        return AmgDecision {
            mode: AmgMode::ExplicitNonSpd,
            reason: "accepted with explicit non-SPD AMG options (relaxed SPD requirement for nearly symmetric matrix)".to_string(),
        };
    }

    let mut causes = Vec::new();
    if diag_issues {
        causes.push("diagonal issues");
    }
    if !approx_symmetric {
        causes.push("nonsymmetric structure");
    }
    if !positive_diag {
        causes.push("non-positive diagonal");
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
    let diag_issues = detect_diag_issues(matrix, 1e-14, 20_000);
    let approx_symmetric = is_approximately_symmetric(matrix, 1e-8, 4_000);
    let positive_diag = has_positive_diagonal(matrix, 1e-14, 20_000);
    let (density, condition_heuristic, _is_well_conditioned) = analyze_matrix_properties(matrix);
    let n = matrix.nrows().max(matrix.ncols());
    let size_class = if n <= 512 {
        "small"
    } else if n <= 4_096 {
        "medium"
    } else {
        "large"
    };
    let spd_like_hint = !diag_issues && approx_symmetric && positive_diag;
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
fn select_solver_policy(
    matrix_name: &str,
    screen: &ScreenReport,
    matrix: &CsrMatrix<f64>,
) -> SelectionDecision {
    let baseline = get_optimal_config(matrix_name);
    let mut primary_solver = baseline.solver.to_string();
    let mut primary_pc = baseline.preconditioner.to_string();
    let fallback_solver = baseline.fallback_solver.to_string();
    let fallback_pc = baseline.fallback_pc.to_string();
    let mut rationale = vec![format!(
        "matrix hint '{}': {}",
        matrix_name, baseline._description
    )];
    let mut contract_checks = Vec::new();

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
        let amg_decision = choose_amg_mode(matrix, !screen.diagonal_healthy);
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

    SelectionDecision {
        primary_solver,
        primary_pc,
        fallback_solver,
        fallback_pc,
        rationale,
        contract_checks,
        expected_iterations: baseline.expected_iterations,
    }
}

#[cfg(not(feature = "complex"))]
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn lookup(matrix: &CsrMatrix<f64>, row: usize, col: usize) -> Option<f64> {
    if row >= matrix.nrows() {
        return None;
    }
    let (cols, vals) = matrix.row(row);
    for (&c, &v) in cols.iter().zip(vals.iter()) {
        if c == col {
            return Some(v);
        }
        if c > col {
            break;
        }
    }
    None
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
            "{:<15} {:<22} {:<22} {:<18} {:<12} {:<12} {:<10} {:<14} {}",
            "Matrix",
            "Primary",
            "Fallback",
            "Reason",
            "TrueRes",
            "StatsRes",
            "Iterations",
            "Status",
            "Performance vs Benchmark"
        );
        println!("{}", "=".repeat(150));
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
                        "{:<15} {:<22} {:<22} {:<18} {:<12} {:<12} {:<10} {:<14} {}",
                        matrix_name, "-", "-", "files_missing", "N/A", "N/A", "N/A", "failed", "-"
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
                    "{:<15} {:<22} {:<22} {:<18} {:<12} {:<12} {:<10} {:<14} {}",
                    matrix_name, "-", "-", "too_large", "N/A", "N/A", "SKIP", "failed", "-"
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
                reason,
                true_residual,
                prec_residual,
                iters,
                status,
                converged,
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

                if is_root_rank {
                    println!(
                        "{:<15} {:<22} {:<22} {:<18} {:<12.2e} {:<12.2e} {:<10} {:<14} {}",
                        matrix_name,
                        primary,
                        fallback,
                        reason,
                        true_residual,
                        prec_residual,
                        status,
                        iters,
                        iter_performance
                    );
                    println!(
                        "    → Screen: symmetry_hint={}, spd_like_hint={}, diagonal_healthy={}, density={:.3e}, size_class={}, cond_heuristic={:.2e}",
                        screen.symmetry_hint,
                        screen.spd_like_hint,
                        screen.diagonal_healthy,
                        screen.density,
                        screen.size_class,
                        screen.condition_heuristic
                    );
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
                        "{:<15} {:<22} {:<22} {:<18} {:<12} {:<12} {:<10} {:<14} {}",
                        matrix_name, "-", "-", "failed", "N/A", "N/A", "N/A", "failed", e
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
