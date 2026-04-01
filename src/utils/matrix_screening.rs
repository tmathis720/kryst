//! Shared CSR matrix diagnostics and repair helpers used by demos.

use crate::matrix::sparse::CsrMatrix;

#[derive(Clone, Debug)]
pub struct CgCompatibilityDiagnostics {
    pub sampled_pair_count: usize,
    pub symmetry_violation_count: usize,
    pub symmetry_violation_rate: f64,
    pub non_positive_diagonal_count: usize,
    pub weak_gershgorin_count: usize,
    pub structural_symmetry_hint: Option<bool>,
    pub used_structural_symmetry_expansion: bool,
}

#[derive(Clone, Debug)]
pub struct CgCompatibility {
    pub cg_safe: bool,
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

    let sampled_symmetry_ok = sampled_pairs == 0 || symmetry_violations * 100 <= sampled_pairs;
    let symmetry_ok = if use_structural_symmetry_expansion {
        structural_symmetry_hint.unwrap_or(sampled_symmetry_ok) || sampled_symmetry_ok
    } else {
        sampled_symmetry_ok
    };
    let diag_ok = non_positive_diagonal == 0 && !diag_issues;
    let gershgorin_ok = weak_gershgorin_rows * 5 <= sample_rows.max(1); // <=20% weak rows
    let cg_safe = symmetry_ok && diag_ok && gershgorin_ok;
    let symmetry_rate = if sampled_pairs > 0 {
        symmetry_violations as f64 / sampled_pairs as f64
    } else {
        0.0
    };

    let diagnostics = CgCompatibilityDiagnostics {
        sampled_pair_count: sampled_pairs,
        symmetry_violation_count: symmetry_violations,
        symmetry_violation_rate: symmetry_rate,
        non_positive_diagonal_count: non_positive_diagonal,
        weak_gershgorin_count: weak_gershgorin_rows,
        structural_symmetry_hint,
        used_structural_symmetry_expansion: use_structural_symmetry_expansion,
    };

    if cg_safe {
        return CgCompatibility {
            cg_safe: true,
            reason: format!(
                "CG contract accepted: sampled symmetry/SPD heuristic passed (sym diff {:.2}%, weak Gershgorin rows {}/{}, diag issues: {}, structural expansion: {})",
                100.0 * symmetry_rate,
                weak_gershgorin_rows,
                sample_rows,
                if diag_issues { "yes" } else { "no" },
                if use_structural_symmetry_expansion {
                    "on"
                } else {
                    "off"
                }
            ),
            diagnostics,
        };
    }

    let mut causes = Vec::new();
    if !symmetry_ok {
        causes.push(format!(
            "sampled nonsymmetry {:.2}%>{:.2}%",
            100.0 * symmetry_rate,
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
        diagnostics,
    }
}
