//! Matrix utility functions that can be used across different modules.
//!
//! This module contains generic matrix operations, analysis functions, and
//! utilities that are useful beyond just the AMG preconditioner.

use crate::KError;
use crate::matrix::sparse::CsrMatrix;
use faer::Mat;

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};

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
                return Err(KError::InvalidInput(
                    format!("NaN detected at position ({}, {})", i, j)
                ));
            }
            if val.is_infinite() {
                return Err(KError::InvalidInput(
                    format!("Infinite value detected at position ({}, {})", i, j)
                ));
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

/// Sparse matrix-matrix multiplication: C = A * B
/// Uses Gustavson's algorithm for efficient CSR x CSR multiplication
/// Avoids HashMap overhead with sorted merge and scratch buffers
pub fn sparse_matrix_multiply(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>
) -> Result<CsrMatrix<f64>, KError> {
    if a.ncols() != b.nrows() {
        return Err(KError::InvalidInput(format!(
            "Matrix dimension mismatch: A is {}x{}, B is {}x{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()
        )));
    }
    
    let m = a.nrows();
    let n = b.ncols();
    let mut row_ptr = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    
    // Gustavson's algorithm: use scratch arrays for efficient accumulation
    let mut x = vec![0.0; n]; // Accumulator array
    let mut w = vec![usize::MAX; n]; // Marker array (MAX means "not seen yet")
    
    // For each row i in A
    for i in 0..m {
        let row_start = col_indices.len();
        
        // Get non-zeros in row i of A
        let a_row_start = a.row_ptrs()[i];
        let a_row_end = a.row_ptrs()[i + 1];
        
        for a_idx in a_row_start..a_row_end {
            let a_col = a.col_indices()[a_idx];
            let a_val = a.values()[a_idx];
            
            // For each non-zero A[i,k], add A[i,k] * B[k,:] to result row
            let b_row_start = b.row_ptrs()[a_col];
            let b_row_end = b.row_ptrs()[a_col + 1];
            
            for b_idx in b_row_start..b_row_end {
                let b_col = b.col_indices()[b_idx];
                let b_val = b.values()[b_idx];
                
                if w[b_col] != i {
                    // First time seeing this column in row i
                    w[b_col] = i;
                    x[b_col] = a_val * b_val;
                    col_indices.push(b_col);
                } else {
                    // Accumulate into existing entry
                    x[b_col] += a_val * b_val;
                }
            }
        }
        
        // Sort columns for this row and extract values
        let row_end = col_indices.len();
        col_indices[row_start..row_end].sort_unstable();
        
        // Extract values in sorted order and apply drop tolerance
        let mut kept_cols = Vec::new();
        let mut kept_vals = Vec::new();
        
        for &col in &col_indices[row_start..row_end] {
            let val = x[col];
            if val.abs() > 1e-12 { // Drop tolerance for numerical stability
                kept_cols.push(col);
                kept_vals.push(val);
            }
        }
        
        // Replace the range with kept columns
        col_indices.truncate(row_start);
        col_indices.extend(kept_cols);
        values.extend(kept_vals);
        
        row_ptr.push(col_indices.len());
    }
    
    Ok(CsrMatrix::from_csr(m, n, row_ptr, col_indices, values))
}

/// Sparse Galerkin product: C = R * A * P (all sparse)
/// This replaces the O(n³) dense triple matrix product with O(nnz) sparse operations
pub fn sparse_galerkin_product(
    restriction: &CsrMatrix<f64>,
    matrix: &CsrMatrix<f64>, 
    interpolation: &CsrMatrix<f64>
) -> Result<CsrMatrix<f64>, KError> {
    // Implement true sparse triple product: R * A * P
    // This is the key optimization that makes AMG scalable
    
    // Step 1: Compute A * P (sparse matrix-matrix multiply)
    let ap = sparse_matrix_multiply(matrix, interpolation)?;
    
    // Step 2: Compute R * (A * P) (sparse matrix-matrix multiply)  
    let coarse_matrix = sparse_matrix_multiply(restriction, &ap)?;
    
    Ok(coarse_matrix)
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
        let max_row_nnz = ((row_entries.len() as f64) * (1.0 - truncation_factor)).max(1.0) as usize;
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
            a.nrows(), a.ncols(), x.len(), y.len()
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
        let matrix = Mat::from_fn(3, 3, |i, j| {
            if i == j {
                (i + 1) as f64
            } else {
                0.0
            }
        });
        
        let diag_inv = extract_diagonal_inverse(&matrix);
        
        assert_eq!(diag_inv.len(), 3);
        assert!((diag_inv[0] - 1.0).abs() < 1e-12);
        assert!((diag_inv[1] - 0.5).abs() < 1e-12);
        assert!((diag_inv[2] - 1.0/3.0).abs() < 1e-12);
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
}
