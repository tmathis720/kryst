//! Matrix reordering and scaling for numerical robustness.
//!
//! This module implements various matrix preprocessing techniques to improve
//! solver convergence and numerical stability:
//!
//! - **Reordering**: RCM (Reverse Cuthill-McKee), natural ordering
//! - **Scaling**: Diagonal scaling, symmetric scaling
//!
//! These transformations are applied during the setup phase when the matrix
//! is available, supporting the Phase II numerical robustness goals.
//!
//! # References
//! - Cuthill, E. and McKee, J. (1969). Reducing the bandwidth of sparse symmetric matrices
//! - Duff, I.S. and Koster, J. (2001). On algorithms for permuting large entries to the diagonal

use crate::error::KError;
use faer::Mat;
use std::collections::VecDeque;

/// Matrix reordering algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReorderingMethod {
    /// No reordering (natural ordering)
    None,
    /// Reverse Cuthill-McKee (RCM) for bandwidth reduction
    Rcm,
    /// Cuthill-McKee (natural order)
    CuthillMckee,
    /// Column Approximate Minimum Degree (COLAMD) - placeholder
    Colamd,
    /// Approximate Minimum Degree (AMD) - placeholder
    Amd,
}

/// Matrix scaling algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMethod {
    /// No scaling
    None,
    /// Diagonal scaling: D^{-1/2} A D^{-1/2} where D = diag(A)
    Diagonal,
    /// Symmetric scaling (equilibration)
    Symmetric,
}

/// Result of matrix preprocessing (reordering + scaling).
#[derive(Debug, Clone)]
pub struct MatrixPreprocessing {
    /// Permutation array: new_index = perm[old_index]
    pub permutation: Vec<usize>,
    /// Inverse permutation: old_index = inv_perm[new_index]
    pub inv_permutation: Vec<usize>,
    /// Left scaling factors (if any)
    pub left_scaling: Option<Vec<f64>>,
    /// Right scaling factors (if any)
    pub right_scaling: Option<Vec<f64>>,
    /// Whether any transformation was applied
    pub is_identity: bool,
}

impl MatrixPreprocessing {
    /// Create identity preprocessing (no changes).
    pub fn identity(n: usize) -> Self {
        Self {
            permutation: (0..n).collect(),
            inv_permutation: (0..n).collect(),
            left_scaling: None,
            right_scaling: None,
            is_identity: true,
        }
    }

    /// Apply preprocessing to transform matrix: P D_L A D_R P^T.
    pub fn apply_to_matrix(&self, a: &Mat<f64>) -> Result<Mat<f64>, KError> {
        let n = a.nrows();
        if self.is_identity {
            return Ok(a.clone());
        }

        // Start with permuted matrix: P A P^T
        let mut result = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let orig_i = self.inv_permutation[i];
                let orig_j = self.inv_permutation[j];
                result[(i, j)] = a[(orig_i, orig_j)];
            }
        }

        // Apply scaling: D_L (P A P^T) D_R
        if let (Some(left), Some(right)) = (&self.left_scaling, &self.right_scaling) {
            for i in 0..n {
                for j in 0..n {
                    result[(i, j)] = left[i] * result[(i, j)] * right[j];
                }
            }
        } else if let Some(diag) = &self.left_scaling {
            // Symmetric diagonal scaling: D^{-1/2} A D^{-1/2}
            for i in 0..n {
                for j in 0..n {
                    result[(i, j)] = diag[i] * result[(i, j)] * diag[j];
                }
            }
        }

        Ok(result)
    }

    /// Transform a vector from original to preprocessed ordering: y = P D x.
    pub fn transform_vector(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut result = vec![0.0; n];

        // Apply permutation
        for i in 0..n {
            result[i] = x[self.inv_permutation[i]];
        }

        // Apply left scaling (if any)
        if let Some(left) = &self.left_scaling {
            for i in 0..n {
                result[i] *= left[i];
            }
        }

        result
    }

    /// Transform a vector from preprocessed back to original ordering: x = D^{-1} P^T y.
    pub fn untransform_vector(&self, y: &[f64]) -> Vec<f64> {
        let n = y.len();
        let mut temp = y.to_vec();

        // Undo left scaling (if any)
        if let Some(left) = &self.left_scaling {
            for i in 0..n {
                temp[i] /= left[i];
            }
        }

        // Undo permutation
        let mut result = vec![0.0; n];
        for i in 0..n {
            result[self.inv_permutation[i]] = temp[i];
        }

        result
    }
}

/// Apply matrix reordering and scaling preprocessing.
///
/// # Arguments
/// * `a` - Input matrix
/// * `reorder_method` - Reordering algorithm to apply
/// * `scaling_method` - Scaling algorithm to apply
///
/// # Returns
/// * `Ok((preprocessed_matrix, preprocessing_info))` - Transformed matrix and transformation info
/// * `Err(KError)` - If preprocessing fails
pub fn preprocess_matrix(
    a: &Mat<f64>,
    reorder_method: ReorderingMethod,
    scaling_method: ScalingMethod,
) -> Result<(Mat<f64>, MatrixPreprocessing), KError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(KError::SolveError(
            "Matrix must be square for preprocessing".to_string(),
        ));
    }

    // Step 1: Compute reordering permutation
    let permutation = match reorder_method {
        ReorderingMethod::None => (0..n).collect(),
        ReorderingMethod::Rcm => reverse_cuthill_mckee(a)?,
        ReorderingMethod::CuthillMckee => cuthill_mckee(a)?,
        ReorderingMethod::Colamd | ReorderingMethod::Amd => {
            return Err(KError::NotImplemented(format!(
                "{reorder_method:?} reordering not yet implemented"
            )));
        }
    };

    // Compute inverse permutation
    let mut inv_permutation = vec![0; n];
    for (new_idx, &old_idx) in permutation.iter().enumerate() {
        inv_permutation[old_idx] = new_idx;
    }

    // Step 2: Apply permutation to get P A P^T
    let mut permuted = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let orig_i = inv_permutation[i];
            let orig_j = inv_permutation[j];
            permuted[(i, j)] = a[(orig_i, orig_j)];
        }
    }

    // Step 3: Compute scaling factors
    let (left_scaling, right_scaling) = match scaling_method {
        ScalingMethod::None => (None, None),
        ScalingMethod::Diagonal => {
            let scaling: Vec<f64> = (0..n)
                .map(|i| {
                    let diag_val = permuted[(i, i)].abs();
                    if diag_val > 1e-15 {
                        1.0 / diag_val.sqrt()
                    } else {
                        1.0 // Don't scale zero diagonal entries
                    }
                })
                .collect();
            (Some(scaling.clone()), Some(scaling))
        }
        ScalingMethod::Symmetric => {
            return Err(KError::NotImplemented(
                "Symmetric scaling not yet implemented".to_string(),
            ));
        }
    };

    // Step 4: Apply scaling to get final matrix
    let mut result = permuted;
    if let (Some(left), Some(right)) = (&left_scaling, &right_scaling) {
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] = left[i] * result[(i, j)] * right[j];
            }
        }
    }

    let preprocessing = MatrixPreprocessing {
        permutation,
        inv_permutation,
        left_scaling,
        right_scaling,
        is_identity: reorder_method == ReorderingMethod::None
            && scaling_method == ScalingMethod::None,
    };

    Ok((result, preprocessing))
}

/// Reverse Cuthill-McKee (RCM) reordering for bandwidth reduction.
///
/// The RCM algorithm finds a permutation that reduces the bandwidth of the matrix,
/// which can improve cache locality and numerical stability.
///
/// # Arguments
/// * `a` - Input symmetric matrix
///
/// # Returns
/// * `Ok(permutation)` - Reordering permutation
/// * `Err(KError)` - If algorithm fails
fn reverse_cuthill_mckee(a: &Mat<f64>) -> Result<Vec<usize>, KError> {
    let perm = cuthill_mckee(a)?;
    // Reverse the permutation
    Ok(perm.into_iter().rev().collect())
}

/// Cuthill-McKee reordering algorithm.
///
/// Implements the classic Cuthill-McKee algorithm for bandwidth reduction.
/// The algorithm performs a breadth-first search ordering vertices by increasing degree.
///
/// # Arguments
/// * `a` - Input symmetric matrix
///
/// # Returns
/// * `Ok(permutation)` - Reordering permutation
/// * `Err(KError)` - If algorithm fails
fn cuthill_mckee(a: &Mat<f64>) -> Result<Vec<usize>, KError> {
    let n = a.nrows();
    let tol = 1e-15; // Zero tolerance

    // Build adjacency list (considering matrix symmetry)
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i != j && (a[(i, j)].abs() > tol || a[(j, i)].abs() > tol) {
                adj[i].push(j);
            }
        }
    }

    // Precompute degrees once and sort neighbors by degree (ascending)
    let degrees: Vec<usize> = adj.iter().map(|nbrs| nbrs.len()).collect();
    for i in 0..n {
        // Stable order not required for CM; use unstable for speed
        adj[i].sort_unstable_by_key(|&nbr| degrees[nbr]);
        // For deterministic tie-breaks, consider:
        // adj[i].sort_unstable_by(|&a, &b| degrees[a].cmp(&degrees[b]).then(a.cmp(&b)));
    }

    let mut visited = vec![false; n];
    let mut permutation = Vec::new();

    // Process all connected components
    for start in 0..n {
        if visited[start] {
            continue;
        }

        // Find peripheral node (node with minimum degree in unvisited component)
        let mut current_start = start;
        let mut min_degree = degrees[start];
        for i in start..n {
            if !visited[i] && degrees[i] < min_degree {
                min_degree = degrees[i];
                current_start = i;
            }
        }

        // BFS from peripheral node
        let mut queue = VecDeque::new();
        queue.push_back(current_start);
        visited[current_start] = true;

        while let Some(node) = queue.pop_front() {
            permutation.push(node);

            // Neighbors are already sorted by degree; enqueue unvisited
            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }

    Ok(permutation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn create_test_matrix() -> Mat<f64> {
        // 4x4 symmetric matrix with specific sparsity pattern
        // [2  1  0  1]
        // [1  3  1  0]
        // [0  1  4  1]
        // [1  0  1  5]
        Mat::from_fn(4, 4, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (0, 1) => 1.0,
            (0, 3) => 1.0,
            (1, 0) => 1.0,
            (1, 1) => 3.0,
            (1, 2) => 1.0,
            (2, 1) => 1.0,
            (2, 2) => 4.0,
            (2, 3) => 1.0,
            (3, 0) => 1.0,
            (3, 2) => 1.0,
            (3, 3) => 5.0,
            _ => 0.0,
        })
    }

    #[test]
    fn test_identity_preprocessing() {
        let a = create_test_matrix();
        let (result, info) =
            preprocess_matrix(&a, ReorderingMethod::None, ScalingMethod::None).unwrap();

        assert!(info.is_identity);
        assert_eq!(info.permutation, vec![0, 1, 2, 3]);
        assert_eq!(info.inv_permutation, vec![0, 1, 2, 3]);
        assert!(info.left_scaling.is_none());
        assert!(info.right_scaling.is_none());

        // Result should be identical to input
        for i in 0..4 {
            for j in 0..4 {
                assert!((result[(i, j)] - a[(i, j)]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_cuthill_mckee_reordering() {
        let a = create_test_matrix();
        let (_result, info) =
            preprocess_matrix(&a, ReorderingMethod::CuthillMckee, ScalingMethod::None).unwrap();

        assert!(!info.is_identity);
        assert_eq!(info.permutation.len(), 4);
        assert_eq!(info.inv_permutation.len(), 4);

        // Verify permutation is valid
        let mut check = vec![false; 4];
        for &p in &info.permutation {
            assert!(p < 4);
            assert!(!check[p]); // No duplicates
            check[p] = true;
        }
        assert!(check.iter().all(|&x| x)); // All indices covered

        // Verify inverse permutation
        for (new_idx, &old_idx) in info.permutation.iter().enumerate() {
            assert_eq!(info.inv_permutation[old_idx], new_idx);
        }
    }

    #[test]
    fn test_diagonal_scaling() {
        let a = create_test_matrix();
        let (_result, info) =
            preprocess_matrix(&a, ReorderingMethod::None, ScalingMethod::Diagonal).unwrap();

        assert!(!info.is_identity);
        assert!(info.left_scaling.is_some());
        assert!(info.right_scaling.is_some());

        let scaling = info.left_scaling.as_ref().unwrap();

        // Check that diagonal entries become 1 after scaling
        for i in 0..4 {
            let scaled_diag = scaling[i] * a[(i, i)] * scaling[i];
            assert!(
                (scaled_diag - 1.0).abs() < 1e-12,
                "Diagonal entry {} not scaled to 1: {}",
                i,
                scaled_diag
            );
        }
    }

    #[test]
    fn test_vector_transformation() {
        let a = create_test_matrix();
        let x = vec![1.0, 2.0, 3.0, 4.0];

        let (_, info) =
            preprocess_matrix(&a, ReorderingMethod::CuthillMckee, ScalingMethod::Diagonal).unwrap();

        // Transform and untransform should be inverse operations
        let transformed = info.transform_vector(&x);
        let recovered = info.untransform_vector(&transformed);

        for i in 0..4 {
            assert!(
                (recovered[i] - x[i]).abs() < 1e-12,
                "Vector transformation not invertible"
            );
        }
    }
}
