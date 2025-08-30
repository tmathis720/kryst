//! Matrix utility functions that can be used across different modules.
//!
//! This module contains generic matrix operations, analysis functions, and
//! utilities that are useful beyond just the AMG preconditioner.

use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use faer::Mat;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Matrix analysis helper function that computes basic properties
///
/// Returns (nnz, diagonal_dominance, diagonal_sum)
pub fn analyze_matrix_properties(matrix: &Mat<f64>) -> (usize, f64, f64) {
    let mut nnz = 0;
    let mut diagonal_sum = 0.0;
    let mut off_diagonal_sum = 0.0;

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix[(i, j)];
            if val.abs() > 1e-15 {
                nnz += 1;
                if i == j {
                    diagonal_sum += val.abs();
                } else {
                    off_diagonal_sum += val.abs();
                }
            }
        }
    }

    let diagonal_dominance = if off_diagonal_sum > 0.0 {
        diagonal_sum / off_diagonal_sum
    } else {
        f64::INFINITY
    };

    (nnz, diagonal_dominance, diagonal_sum)
}

/// Check for numerical issues in the matrix (NaN, Inf, very large condition numbers)
pub fn has_numerical_issues(matrix: &Mat<f64>) -> bool {
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix[(i, j)];
            if !val.is_finite() {
                return true;
            }
        }
    }
    false
}

/// IEEE safety check for matrix values
pub fn check_ieee_values(matrix: &Mat<f64>) -> Result<(), KError> {
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix[(i, j)];
            if val.is_nan() {
                return Err(KError::InvalidInput(format!(
                    "NaN detected at position ({}, {})",
                    i, j
                )));
            }
            if val.is_infinite() {
                return Err(KError::InvalidInput(format!(
                    "Infinite value detected at position ({}, {})",
                    i, j
                )));
            }
        }
    }
    Ok(())
}

/// Extract the inverse of the diagonal of a matrix, with zero for near-singular entries.
pub fn extract_diagonal_inverse(m: &Mat<f64>) -> Vec<f64> {
    let n = m.nrows();
    let mut diag_inv = vec![0.0; n];
    for i in 0..n {
        let diag_val = m[(i, i)];
        if diag_val.abs() > 1e-14 {
            diag_inv[i] = 1.0 / diag_val;
        }
    }
    diag_inv
}

/// Convert dense matrix to sparse format with drop tolerance
/// This is the foundation for sparse Galerkin products
pub fn to_sparse_with_tolerance(matrix: &Mat<f64>, drop_tol: f64) -> CsrMatrix<f64> {
    CsrMatrix::from_dense(matrix, drop_tol)
}

/// Sparse C = A * B using Gustavson's algorithm on CSR arrays.
/// Returns a CSR with sorted columns per row and optional dropping.
///
/// `drop_tol`: entries with |v| <= drop_tol are removed.
pub fn spgemm_with_drop_tol(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    drop_tol: f64,
) -> Result<CsrMatrix<f64>, KError> {
    if a.ncols() != b.nrows() {
        return Err(KError::InvalidInput(format!(
            "spgemm: dimension mismatch A is {}x{}, B is {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }

    let m = a.nrows();
    let n = b.ncols();

    let ap = a.row_ptr();
    let aj = a.col_idx();
    let av = a.values();

    let bp = b.row_ptr();
    let bj = b.col_idx();
    let bv = b.values();

    let mut row_ptr = Vec::with_capacity(m + 1);
    row_ptr.push(0usize);
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Marker and accumulator per column of C's row.
    // `mark[j] == i` means column j is present in row i's structure.
    let mut mark = vec![usize::MAX; n];
    let mut acc = vec![0.0f64; n];

    for i in 0..m {
        let row_head = cols.len();

        // Row i of A
        for kk in ap[i]..ap[i + 1] {
            let k = aj[kk];
            let aik = av[kk];

            // Row k of B (since B is CSR, this is B[k,:])
            for jj in bp[k]..bp[k + 1] {
                let j = bj[jj];
                let inc = aik * bv[jj];

                if mark[j] != i {
                    mark[j] = i;
                    acc[j] = inc;
                    cols.push(j);
                } else {
                    acc[j] += inc;
                }
            }
        }

        // Sort column indices in this row's slice
        let row_tail = cols.len();
        cols[row_head..row_tail].sort_unstable();

        // Compact in-place while merging duplicates and applying drop tolerance.
        // Because cols[row_head..row_tail] is sorted, equal columns are grouped.
        let mut write = row_head;
        let mut read = row_head;
        while read < row_tail {
            let j0 = cols[read];
            // The accumulator already holds the full sum for j0; skip any duplicates
            let sum = acc[j0];
            // advance read past all instances of j0
            while read < row_tail && cols[read] == j0 { read += 1; }
            // reset for next row
            acc[j0] = 0.0;
            mark[j0] = usize::MAX;
            if sum.abs() > drop_tol {
                cols[write] = j0;
                vals.push(sum);
                write += 1;
            }
            // else: drop this column entirely
        }
        // Remove dropped columns from `cols` for this row
        cols.truncate(write);

        row_ptr.push(vals.len());
    }

    Ok(CsrMatrix::from_csr(m, n, row_ptr, cols, vals))
}

/// Convenience wrapper with a default numerical drop tolerance.
#[inline]
pub fn spgemm(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> Result<CsrMatrix<f64>, KError> {
    spgemm_with_drop_tol(a, b, 1e-12)
}

/// Baseline Sparse C = A * B using per-row BTreeMap accumulation.
///
/// This implementation is intentionally simple and allocation-heavy to serve
/// as a comparison baseline for optimized kernels in benchmarks.
pub fn spgemm_btree(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> Result<CsrMatrix<f64>, KError> {
    use std::collections::BTreeMap;

    if a.ncols() != b.nrows() {
        return Err(KError::InvalidInput(format!(
            "spgemm_btree: dimension mismatch A is {}x{}, B is {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }

    let m = a.nrows();
    let n = b.ncols();
    let ap = a.row_ptr();
    let aj = a.col_idx();
    let av = a.values();
    let bp = b.row_ptr();
    let bj = b.col_idx();
    let bv = b.values();

    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);

    for i in 0..m {
        let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
        for kk in ap[i]..ap[i + 1] {
            let k = aj[kk];
            let aik = av[kk];
            for jj in bp[k]..bp[k + 1] {
                let j = bj[jj];
                *acc.entry(j).or_insert(0.0) += aik * bv[jj];
            }
        }
        for (j, v) in acc.into_iter() {
            if v != 0.0 {
                col_idx.push(j);
                vals.push(v);
            }
        }
        row_ptr.push(col_idx.len());
    }

    Ok(CsrMatrix::from_csr(m, n, row_ptr, col_idx, vals))
}

/// Sparse Galerkin product: C = R * A * P
/// with all inputs CSR, via two SpGEMMs.
pub fn sparse_galerkin_product(
    restriction: &CsrMatrix<f64>,   // R
    matrix: &CsrMatrix<f64>,        // A
    interpolation: &CsrMatrix<f64>, // P
) -> Result<CsrMatrix<f64>, KError> {
    // Step 1: T = A * P
    let ap = spgemm(matrix, interpolation)?;
    // Step 2: C = R * T
    spgemm(restriction, &ap)
}

/// Baseline RAP (triple product) using the `spgemm_btree` baseline twice.
pub fn rap_btree(
    restriction: &CsrMatrix<f64>,
    matrix: &CsrMatrix<f64>,
    interpolation: &CsrMatrix<f64>,
) -> Result<CsrMatrix<f64>, KError> {
    let ap = spgemm_btree(matrix, interpolation)?;
    spgemm_btree(restriction, &ap)
}

/// Optimized RAP wrapper to keep a stable API for benchmarks.
/// Currently composes two calls to the optimized SpGEMM.
#[inline]
pub fn rap_opt(
    restriction: &CsrMatrix<f64>,
    matrix: &CsrMatrix<f64>,
    interpolation: &CsrMatrix<f64>,
) -> Result<CsrMatrix<f64>, KError> {
    sparse_galerkin_product(restriction, matrix, interpolation)
}

/// Apply HYPRE-style truncation to interpolation matrix
/// Removes weak connections based on threshold-based dropping
///
/// This function performs row-wise truncation of the interpolation operator,
/// keeping only the strongest connections to improve operator complexity.
pub fn apply_truncation(interpolation: &mut Mat<f64>, truncation_factor: f64) {
    if truncation_factor <= 0.0 || truncation_factor >= 1.0 {
        return; // Invalid truncation factor
    }

    let nrows = interpolation.nrows();
    let ncols = interpolation.ncols();

    for i in 0..nrows {
        // Collect (magnitude, column, original_value) tuples for this row
        let mut row_entries: Vec<(f64, usize, f64)> = Vec::new();

        for j in 0..ncols {
            let val = interpolation[(i, j)];
            if val.abs() > 1e-15 {
                row_entries.push((val.abs(), j, val));
            }
        }

        if row_entries.is_empty() {
            continue;
        }

        // Sort by magnitude (largest first)
        row_entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Determine how many entries to keep
        let max_row_nnz =
            ((row_entries.len() as f64) * (1.0 - truncation_factor)).max(1.0) as usize;
        let keep_count = max_row_nnz.min(row_entries.len());

        // Zero out all entries first
        for j in 0..ncols {
            interpolation[(i, j)] = 0.0;
        }

        // Keep only the strongest entries with their original values
        for k in 0..keep_count {
            let (_magnitude, j, original_val) = row_entries[k];
            interpolation[(i, j)] = original_val;
        }
    }
}

/// Parallel matrix-vector operation for dense matrices
/// Returns Result for consistency with sparse operations
pub fn parallel_mat_vec(a: &Mat<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    if x.len() != a.ncols() || y.len() != a.nrows() {
        return Err(KError::InvalidInput(format!(
            "Dimension mismatch: A={}x{}, x.len()={}, y.len()={}",
            a.nrows(),
            a.ncols(),
            x.len(),
            y.len()
        )));
    }

    let _rows = a.nrows();
    let cols = a.ncols();

    #[cfg(feature = "rayon")]
    {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = (0..cols).map(|j| a[(i, j)] * x[j]).sum();
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = (0..cols).map(|j| a[(i, j)] * x[j]).sum();
        }
    }

    Ok(())
}

/// Parallel matrix-vector operation for sparse matrices
/// Returns Result for error handling
pub fn parallel_mat_vec_sparse(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
    a.spmv_scaled(1.0, x, 0.0, y)
}

/// Count non-zeros in a dense matrix
pub fn count_nnz(matrix: &Mat<f64>) -> usize {
    let mut nnz = 0;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            if matrix[(i, j)].abs() > 1e-15 {
                nnz += 1;
            }
        }
    }
    nnz
}

/// Compute anisotropy for each row of the matrix.
/// Anisotropy is defined as the ratio max_off_diag/diag.
pub fn compute_anisotropy(a: &Mat<f64>) -> Vec<f64> {
    let n = a.nrows();
    let mut anisotropy = vec![1.0; n];

    for i in 0..n {
        let diag = a[(i, i)].abs();
        if diag < 1e-14 {
            anisotropy[i] = f64::INFINITY;
            continue;
        }

        let mut max_off_diag = 0.0f64;
        for j in 0..n {
            if i != j {
                max_off_diag = max_off_diag.max(a[(i, j)].abs());
            }
        }

        anisotropy[i] = max_off_diag / diag;
    }

    anisotropy
}

/// Compute an adaptive threshold based on global anisotropy indicators.
///
/// The threshold is scaled by the average anisotropy to improve coarsening for highly anisotropic problems.
pub fn compute_adaptive_threshold(a: &Mat<f64>, base_threshold: f64) -> f64 {
    let anisotropy = compute_anisotropy(a);
    let avg_anisotropy = anisotropy.iter().sum::<f64>() / anisotropy.len() as f64;

    // Scale threshold based on anisotropy: higher anisotropy -> lower threshold
    let scaling_factor = (1.0 + avg_anisotropy.log10()).max(0.5).min(2.0);
    base_threshold * scaling_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::CsrMatrix;
    use faer::Mat;

    #[test]
    fn test_analyze_matrix_properties() {
        let matrix = Mat::from_fn(3, 3, |i, j| {
            if i == j {
                2.0
            } else if (i + j) % 2 == 0 {
                1.0
            } else {
                0.0
            }
        });

        let (nnz, diag_dominance, diag_sum) = analyze_matrix_properties(&matrix);

        assert_eq!(nnz, 5); // 3 diagonal + 2 off-diagonal
        assert!((diag_sum - 6.0).abs() < 1e-12); // 3 * 2.0
        assert!(diag_dominance > 1.0); // Should be diagonally dominant
    }

    #[test]
    fn test_extract_diagonal_inverse() {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });

        let diag_inv = extract_diagonal_inverse(&matrix);

        assert_eq!(diag_inv.len(), 3);
        assert!((diag_inv[0] - 1.0).abs() < 1e-12);
        assert!((diag_inv[1] - 0.5).abs() < 1e-12);
        assert!((diag_inv[2] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_check_ieee_values() {
        let good_matrix = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
        assert!(check_ieee_values(&good_matrix).is_ok());

        let bad_matrix = Mat::from_fn(2, 2, |i, j| {
            if i == 0 && j == 0 {
                f64::NAN
            } else {
                (i + j) as f64
            }
        });
        assert!(check_ieee_values(&bad_matrix).is_err());
    }

    #[test]
    fn spgemm_identity_left() {
        let i3 = CsrMatrix::from_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]);
        // A = [[1,2,0],[0,3,4],[0,0,5]]
        let a = CsrMatrix::from_csr(
            3,
            3,
            vec![0, 2, 4, 5],
            vec![0, 1, 1, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        );
        let c = spgemm(&i3, &a).unwrap();
        assert_eq!(c.row_ptr(), a.row_ptr());
        assert_eq!(c.col_idx(), a.col_idx());
        assert_eq!(c.values(), a.values());
    }

    #[test]
    fn spgemm_simple() {
        // A: 2x3 [[1,2,0],[0,3,4]]
        let a = CsrMatrix::from_csr(
            2,
            3,
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        );
        // B: 3x2 [[5,0],[0,6],[7,8]]
        let b = CsrMatrix::from_csr(
            3,
            2,
            vec![0, 1, 2, 4],
            vec![0, 1, 0, 1],
            vec![5.0, 6.0, 7.0, 8.0],
        );
        // A*B should be:
        // row0: [1*5 + 2*0 + 0*7, 1*0 + 2*6 + 0*8] = [5, 12]
        // row1: [0*5 + 3*0 + 4*7, 0*0 + 3*6 + 4*8] = [28, 50]
        let c = spgemm(&a, &b).unwrap();
        assert_eq!(c.row_ptr(), &[0, 2, 4]);
        assert_eq!(c.col_idx(), &[0, 1, 0, 1]);
        assert_eq!(c.values(), &[5.0, 12.0, 28.0, 50.0]);
    }

    #[test]
    fn galerkin_triple() {
        // R = I, so RAP = A
        let i3 = CsrMatrix::from_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]);
        let a = CsrMatrix::from_csr(
            3,
            3,
            vec![0, 2, 4, 5],
            vec![0, 1, 1, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        );
        let c = sparse_galerkin_product(&i3, &a, &i3).unwrap();
        assert_eq!(c.row_ptr(), a.row_ptr());
        assert_eq!(c.col_idx(), a.col_idx());
        assert_eq!(c.values(), a.values());
    }
}
