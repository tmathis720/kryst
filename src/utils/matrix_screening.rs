//! Shared CSR matrix diagnostics and repair helpers used by demos.

use crate::matrix::sparse::CsrMatrix;

#[derive(Clone, Debug)]
pub struct CgCompatibilityDiagnostics {
    pub sampled_pair_count: usize,
    pub symmetry_violation_count: usize,
    pub symmetry_violation_rate: f64,
    pub non_positive_diagonal_count: usize,
    pub weak_gershgorin_count: usize,
    pub conditioning_ratio_estimate: Option<f64>,
    pub structural_symmetry_hint: Option<bool>,
    pub used_structural_symmetry_expansion: bool,
}

#[derive(Clone, Debug)]
pub struct CgCompatibility {
    pub is_hard_reject: bool,
    pub warnings: Vec<String>,
    pub hard_reject_reasons: Vec<String>,
    pub reason: String,
    pub diagnostics: CgCompatibilityDiagnostics,
}

pub fn lookup_csr(a: &CsrMatrix<f64>, row: usize, col: usize) -> Option<f64> {
    if row >= a.nrows() {
        return None;
    }
    let (cols, vals) = a.row(row);
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

pub fn repair_diagonal_csr(a: &CsrMatrix<f64>, tol: f64, tau: f64) -> (CsrMatrix<f64>, usize) {
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

pub fn detect_diag_issues(a: &CsrMatrix<f64>, tol: f64, max_rows: usize) -> bool {
    let limit = a.nrows().min(a.ncols()).min(max_rows);
    for i in 0..limit {
        match lookup_csr(a, i, i) {
            Some(v) if v.abs() > tol => {}
            _ => return true,
        }
    }
    false
}

pub fn is_approximately_symmetric(a: &CsrMatrix<f64>, tol: f64, max_rows: usize) -> bool {
    let limit = a.nrows().min(a.ncols()).min(max_rows);
    for i in 0..limit {
        let (cols, vals) = a.row(i);
        for (&j, &a_ij) in cols.iter().zip(vals.iter()) {
            if j >= limit {
                continue;
            }
            let a_ji = lookup_csr(a, j, i).unwrap_or(0.0);
            if (a_ij - a_ji).abs() > tol {
                return false;
            }
        }
    }
    true
}

pub fn has_positive_diagonal(a: &CsrMatrix<f64>, tol: f64, max_rows: usize) -> bool {
    let limit = a.nrows().min(a.ncols()).min(max_rows);
    for i in 0..limit {
        match lookup_csr(a, i, i) {
            Some(v) if v > tol => {}
            _ => return false,
        }
    }
    true
}

pub fn cg_compatibility_screen(
    matrix: &CsrMatrix<f64>,
    diag_issues: bool,
    structural_symmetry_hint: Option<bool>,
    use_structural_symmetry_expansion: bool,
) -> CgCompatibility {
    let n = matrix.nrows().min(matrix.ncols());
    let sample_rows = n.min(1024);
    let symmetry_tol = 1e-7;

    let mut sampled_pairs = 0usize;
    let mut symmetry_violations = 0usize;
    let mut non_positive_diagonal = 0usize;
    let mut weak_gershgorin_rows = 0usize;
    let mut min_pos_diag = f64::INFINITY;
    let mut max_abs_diag = 0.0f64;
    let mut negative_diagonal = 0usize;

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
            if d < 0.0 {
                negative_diagonal += 1;
            }
        } else {
            min_pos_diag = min_pos_diag.min(d);
        }
        max_abs_diag = max_abs_diag.max(d.abs());
        if d <= row_abs_offdiag_sum {
            weak_gershgorin_rows += 1;
        }
    }

    let sampled_symmetry_ok = sampled_pairs == 0 || symmetry_violations * 100 <= sampled_pairs;
    let symmetry_ok = if use_structural_symmetry_expansion {
        structural_symmetry_hint.unwrap_or(sampled_symmetry_ok) || sampled_symmetry_ok
    } else {
        sampled_symmetry_ok
    };
    let diag_ok = non_positive_diagonal == 0 && !diag_issues;
    let gershgorin_ok = weak_gershgorin_rows * 5 <= sample_rows.max(1); // <=20% weak rows
    let conditioning_ratio_estimate = if min_pos_diag.is_finite() && min_pos_diag > 0.0 {
        Some(max_abs_diag / min_pos_diag)
    } else {
        None
    };
    let symmetry_rate = if sampled_pairs > 0 {
        symmetry_violations as f64 / sampled_pairs as f64
    } else {
        0.0
    };

    let mut hard_reject_reasons = Vec::new();
    if !symmetry_ok {
        hard_reject_reasons.push(format!(
            "sampled asymmetry {:.2}% exceeds {:.2}% threshold",
            100.0 * symmetry_rate,
            1.0
        ));
    }
    if !diag_ok {
        hard_reject_reasons.push(format!(
            "non-positive/missing diagonal evidence: rows={} diag_issues={}",
            non_positive_diagonal, diag_issues
        ));
    }
    if negative_diagonal > 0 {
        hard_reject_reasons.push(format!(
            "explicit indefiniteness evidence: negative diagonal rows={}",
            negative_diagonal
        ));
    }

    let mut warnings = Vec::new();
    if !gershgorin_ok {
        warnings.push(format!(
            "weak Gershgorin dominance rows {}/{}",
            weak_gershgorin_rows, sample_rows
        ));
    }
    if let Some(ratio) = conditioning_ratio_estimate {
        if ratio >= 1.0e8 {
            warnings.push(format!(
                "conditioning heuristic: diagonal spread ratio {:.2e} suggests ill-conditioning",
                ratio
            ));
        }
    }
    if structural_symmetry_hint.is_none() {
        warnings.push("missing symmetry metadata hint".to_string());
    }

    let diagnostics = CgCompatibilityDiagnostics {
        sampled_pair_count: sampled_pairs,
        symmetry_violation_count: symmetry_violations,
        symmetry_violation_rate: symmetry_rate,
        non_positive_diagonal_count: non_positive_diagonal,
        weak_gershgorin_count: weak_gershgorin_rows,
        conditioning_ratio_estimate,
        structural_symmetry_hint,
        used_structural_symmetry_expansion: use_structural_symmetry_expansion,
    };

    if hard_reject_reasons.is_empty() {
        let warning_suffix = if warnings.is_empty() {
            "none".to_string()
        } else {
            warnings.join("; ")
        };
        return CgCompatibility {
            is_hard_reject: false,
            warnings,
            hard_reject_reasons,
            reason: format!(
                "CG contract accepted: sampled symmetry/SPD hard checks passed (sym diff {:.2}%, weak Gershgorin rows {}/{}, diag issues: {}, structural expansion: {}, warnings: {})",
                100.0 * symmetry_rate,
                weak_gershgorin_rows,
                sample_rows,
                if diag_issues { "yes" } else { "no" },
                if use_structural_symmetry_expansion {
                    "on"
                } else {
                    "off"
                },
                warning_suffix
            ),
            diagnostics,
        };
    }

    CgCompatibility {
        is_hard_reject: true,
        warnings: warnings.clone(),
        hard_reject_reasons: hard_reject_reasons.clone(),
        reason: format!(
            "wrong method for matrix contract: CG rejected by compatibility screen ({}){}",
            hard_reject_reasons.join("; "),
            if warnings.is_empty() {
                String::new()
            } else {
                format!(" [warnings: {}]", warnings.join("; "))
            }
        ),
        diagnostics,
    }
}
