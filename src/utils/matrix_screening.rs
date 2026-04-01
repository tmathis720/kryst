//! Shared CSR matrix diagnostics and repair helpers used by demos.

use crate::matrix::sparse::CsrMatrix;

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
