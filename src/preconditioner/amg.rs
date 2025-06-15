// Adapted AMG implementation for kryst
// use crate::core::traits::{InnerProduct, MatVec}; // Remove unused imports
use crate::preconditioner::Preconditioner;
use faer::Mat;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub struct AMG {
    levels: Vec<AMGLevel>,
    nu_pre: usize,
    nu_post: usize,
    matrix: Mat<f64>, // Store the system matrix
}

struct AMGLevel {
    interpolation: Mat<f64>,
    restriction: Mat<f64>,
    coarse_matrix: Mat<f64>,
    diag_inv: Vec<f64>,
}

impl AMG {
    pub fn new(a: &Mat<f64>, max_levels: usize, base_threshold: f64) -> Self {
        let mut levels = Vec::new();
        let mut current_matrix = a.clone();
        let mut current_diag = Self::extract_diagonal_inverse(&current_matrix);
        for _level_idx in 0..max_levels {
            let n = current_matrix.nrows();
            if n <= 10 {
                break;
            }
            let adaptive_threshold = compute_adaptive_threshold(&current_matrix, base_threshold);
            let (mut interpolation, restriction) = AMG::generate_operators(
                &current_matrix,
                adaptive_threshold,
                true,
            );
            smooth_interpolation(&mut interpolation, &current_matrix, 0.5);
            minimize_energy(&mut interpolation, &current_matrix);
            let coarse_matrix = &restriction * &current_matrix * &interpolation;
            let coarse_diag = Self::extract_diagonal_inverse(&coarse_matrix);
            levels.push(AMGLevel {
                interpolation,
                restriction,
                coarse_matrix: current_matrix.clone(),
                diag_inv: current_diag,
            });
            current_matrix = coarse_matrix.clone();
            current_diag = coarse_diag;
        }
        let diag_inv_final = Self::extract_diagonal_inverse(&current_matrix);
        levels.push(AMGLevel {
            interpolation: Mat::identity(current_matrix.nrows(), current_matrix.nrows()),
            restriction: Mat::identity(current_matrix.nrows(), current_matrix.nrows()),
            coarse_matrix: current_matrix.clone(),
            diag_inv: diag_inv_final,
        });
        AMG {
            levels,
            nu_pre: 1,
            nu_post: 1,
            matrix: a.clone(),
        }
    }
    fn generate_operators(
        a: &Mat<f64>,
        threshold: f64,
        double_pairwise: bool,
    ) -> (Mat<f64>, Mat<f64>) {
        let strength_matrix = compute_strength_matrix(a, threshold);
        let aggregates = if double_pairwise {
            double_pairwise_aggregation(&strength_matrix)
        } else {
            greedy_aggregation(&strength_matrix)
        };
        let prolongation = construct_prolongation(a, &aggregates);
        let restriction = prolongation.transpose().to_owned();
        (prolongation, restriction)
    }
    fn extract_diagonal_inverse(m: &Mat<f64>) -> Vec<f64> {
        assert_eq!(m.nrows(), m.ncols());
        let n = m.nrows();
        (0..n)
            .into_par_iter()
            .map(|i| {
                let d = m[(i, i)];
                if d.abs() < 1e-14 {
                    0.0
                } else {
                    1.0 / d
                }
            })
            .collect()
    }
    fn smooth_jacobi_parallel(a: &Mat<f64>, diag_inv: &[f64], r: &[f64], z: &mut [f64], iterations: usize) {
        let n = r.len();
        let mut z_vec = z.to_vec();
        let mut temp = vec![0.0; n];
        for _ in 0..iterations {
            parallel_mat_vec(a, &z_vec, &mut temp);
            temp.par_iter_mut().enumerate().for_each(|(i, val)| {
                *val = r[i] - *val;
            });
            z_vec.par_iter_mut().enumerate().for_each(|(i, val)| *val += diag_inv[i] * temp[i]);
        }
        z.copy_from_slice(&z_vec);
    }
    fn apply_recursive(&self, level: usize, r: &[f64], z: &mut [f64]) {
        if level + 1 == self.levels.len() {
            AMG::solve_direct(&self.levels[level].coarse_matrix, r, z);
            return;
        }
        let a = &self.levels[level].coarse_matrix;
        let diag_inv = &self.levels[level].diag_inv;
        let restriction = &self.levels[level].restriction;
        let interpolation = &self.levels[level].interpolation;
        let coarse_matrix = &self.levels[level + 1].coarse_matrix;
        AMG::smooth_jacobi_parallel(a, diag_inv, r, z, self.nu_pre);
        let mut az = vec![0.0; a.nrows()];
        parallel_mat_vec(a, z, &mut az);
        for i in 0..az.len() {
            az[i] = r[i] - az[i];
        }
        let mut coarse_residual = vec![0.0; coarse_matrix.nrows()];
        parallel_mat_vec(restriction, &az, &mut coarse_residual);
        let mut coarse_solution = vec![0.0; coarse_matrix.nrows()];
        self.apply_recursive(
            level + 1,
            &coarse_residual,
            &mut coarse_solution,
        );
        let mut fine_correction = vec![0.0; a.nrows()];
        parallel_mat_vec(interpolation, &coarse_solution, &mut fine_correction);
        for i in 0..z.len() {
            z[i] += fine_correction[i];
        }
        AMG::smooth_jacobi_parallel(a, diag_inv, r, z, self.nu_post);
    }
    fn solve_direct(a: &Mat<f64>, r: &[f64], z: &mut [f64]) {
        let n = r.len();
        assert_eq!(a.ncols(), n);
        assert_eq!(a.nrows(), n);
        assert_eq!(z.len(), n);
        let mut x = vec![0.0; n];
        let mut residual = r.to_vec();
        let mut p = residual.clone();
        let mut ap = vec![0.0; n];
        let mut alpha;
        let mut beta;
        let mut rr_new = residual.iter().map(|&val| val * val).sum::<f64>();
        let mut rr_old;
        for _ in 0..n {
            parallel_mat_vec(a, &p, &mut ap);
            let denominator = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum::<f64>();
            alpha = rr_new / denominator;
            for (xi, &pi) in x.iter_mut().zip(p.iter()) {
                *xi += alpha * pi;
            }
            for (ri, &api) in residual.iter_mut().zip(ap.iter()) {
                *ri -= alpha * api;
            }
            rr_old = rr_new;
            rr_new = residual.iter().map(|&val| val * val).sum::<f64>();
            if rr_new.sqrt() < 1e-10 {
                break;
            }
            beta = rr_new / rr_old;
            for (pi, &ri) in p.iter_mut().zip(residual.iter()) {
                *pi = ri + beta * *pi;
            }
        }
        z.copy_from_slice(&x);
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for AMG {
    fn apply(&self, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), crate::error::KError> {
        let residual = r.clone();
        let mut solution = vec![0.0; residual.len()];
        if self.levels.is_empty() {
            let diag_inv = AMG::extract_diagonal_inverse(&self.matrix);
            AMG::smooth_jacobi_parallel(&self.matrix, &diag_inv, &residual, &mut solution, 10);
        } else {
            assert_eq!(self.levels[0].coarse_matrix.nrows(), self.matrix.nrows());
            self.apply_recursive(0, &residual, &mut solution);
        }
        z.copy_from_slice(&solution);
        Ok(())
    }
}

// ------------------- Additional Functions for Improvements -------------------

/// Compute anisotropy for each row of the matrix.
/// Anisotropy is defined as the ratio max_off_diag/diag.
fn compute_anisotropy(a: &Mat<f64>) -> Vec<f64> {
    let n = a.nrows();

    (0..n)
        .into_par_iter() // Parallel iterator
        .map(|i| {
            let diag = a[(i, i)];
            let max_off_diag = (0..n)
                .filter(|&j| i != j) // Exclude the diagonal element
                .map(|j| a[(i, j)].abs()) // Compute absolute value of off-diagonal elements
                .fold(0.0, f64::max); // Find the maximum off-diagonal element

            if diag.abs() > 1e-14 {
                max_off_diag / diag.abs()
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute an adaptive threshold based on global anisotropy indicators.
fn compute_adaptive_threshold(a: &Mat<f64>, base_threshold: f64) -> f64 {
    let anis = compute_anisotropy(a);
    let avg_anis = if anis.is_empty() {
        1.0
    } else {
        anis.iter().sum::<f64>() / (anis.len() as f64)
    };
    base_threshold * (1.0 + avg_anis.max(0.5))
}

/// Smooth the interpolation matrix to improve prolongation accuracy.
fn smooth_interpolation(interpolation: &mut Mat<f64>, matrix: &Mat<f64>, weight: f64) {
    let row_count = interpolation.nrows().min(matrix.nrows());
    let col_count = interpolation.ncols().min(matrix.ncols());

    let interpolation = Arc::new(Mutex::new(interpolation));
    (0..col_count).into_par_iter().for_each(|j| {
        for i in 0..row_count {
            let mut interpolation = interpolation.lock().unwrap();
            let old_val = interpolation[(i, j)];
            let diff = weight * matrix[(i, j)];
            interpolation[(i, j)] = old_val - diff;
        }
    });
    let _interpolation = Arc::try_unwrap(interpolation).expect("Failed to unwrap Arc").into_inner().unwrap();
}


/// Normalize rows of the interpolation matrix to minimize energy.
fn minimize_energy(interpolation: &mut Mat<f64>, _matrix: &Mat<f64>) {
    let rows = interpolation.nrows();
    let cols = interpolation.ncols();
    // Step 1: In parallel, compute normalized rows
    let normalized_rows: Vec<Vec<f64>> = (0..rows).into_par_iter().map(|i| {
        let mut row_vec: Vec<f64> = (0..cols).map(|j| interpolation[(i, j)]).collect();
        let row_sum: f64 = row_vec.iter().map(|&val| val * val).sum();
        let norm_factor = if row_sum.abs() > 1e-14 {
            row_sum.sqrt()
        } else {
            1.0
        };
        for val in row_vec.iter_mut() {
            *val /= norm_factor;
        }
        row_vec
    }).collect();
    // Step 2: Sequentially write back
    for i in 0..rows {
        for j in 0..cols {
            interpolation[(i, j)] = normalized_rows[i][j];
        }
    }
}

/// Parallel mat-vec multiplication using rayon.
fn parallel_mat_vec(mat: &Mat<f64>, vec: &[f64], result: &mut [f64]) {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    let (vlen, rlen) = (vec.len(), result.len());
    assert_eq!(cols, vlen, "Dimension mismatch in parallel_mat_vec!\n \
         Matrix is {}x{}, but input vector length is {}.\n \
         (Matrix columns must match vector length.)", rows, cols, vlen);
    assert_eq!(rows, rlen, "Dimension mismatch in parallel_mat_vec!\n \
         Matrix is {}x{}, but result length is {}.\n \
         (Matrix rows must match result length.)", rows, cols, rlen);
    result
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, res)| {
            *res = (0..cols)
                .map(|j| mat[(i, j)] * vec[j])
                .sum();
        });
}


// ------------------- Helper Functions for Enhanced Coarsening -------------------

/// Compute strength of connection matrix S, based on the definition:
/// S(i, j) = |A_ij| / sqrt(|A_ii| * |A_jj|) if > threshold, else 0.
fn compute_strength_matrix(a: &Mat<f64>, threshold: f64) -> Mat<f64> {
    let n = a.nrows();
    let mut s = Mat::<f64>::zeros(n, n);

    let updates: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let a_ii = a[(i, i)].abs();
            (0..n)
                .filter_map(move |j| {
                    if i == j {
                        return Some((i, j, 0.0));
                    }
                    let val = a[(i, j)];
                    let a_jj = a[(j, j)].abs();
                    if a_ii > 1e-14 && a_jj > 1e-14 {
                        let strength = val.abs() / (a_ii * a_jj).sqrt();
                        if strength > threshold {
                            return Some((i, j, strength));
                        }
                    }
                    None
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for (i, j, value) in updates {
        s[(i, j)] = value;
    }

    s
}



/// Perform double-pairwise aggregation:
/// 1. Pairwise aggregate the graph to form coarse nodes.
/// 2. On the coarse graph, perform another round of pairing to form larger aggregates.
///    This function returns a vector where `aggregates[i]` = aggregate index of node i.
fn double_pairwise_aggregation(s: &Mat<f64>) -> Vec<usize> {
    // First pass: pairwise aggregation
    let first_pass = pairwise_aggregation(s);

    // Construct a coarse-level graph and apply pairwise aggregation again
    let coarse_graph = build_coarse_graph(s, &first_pass);
    let second_pass = pairwise_aggregation(&coarse_graph);

    // Map the second pass results back to the fine level
    remap_aggregates(&first_pass, &second_pass)
}

/// Greedy aggregation based on strength of connection:
/// Each node finds its strongest neighbor and they form an aggregate.
/// If a node is already aggregated, skip it.
fn greedy_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let n = s.nrows();
    let mut aggregates = vec![usize::MAX; n];
    let mut next_agg_id = 0;

    for i in 0..n {
        if aggregates[i] == usize::MAX {
            let mut max_strength = 0.0;
            let mut strongest = i;
            for j in 0..n {
                let strength = s[(i, j)];
                if strength > max_strength && aggregates[j] == usize::MAX && i != j {
                    max_strength = strength;
                    strongest = j;
                }
            }
            aggregates[i] = next_agg_id;
            if strongest != i {
                aggregates[strongest] = next_agg_id;
            }
            next_agg_id += 1;
        }
    }

    aggregates
}

/// Pairwise aggregate a given strength matrix. This is a helper for double_pairwise_aggregation.
fn pairwise_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let n = s.nrows();
    let mut aggregates = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    let mut aggregate_id = 0;

    for i in 0..n {
        if visited[i] {
            continue;
        }

        // Find the strongest unvisited neighbor
        let mut max_strength = 0.0;
        let mut strongest_neighbor = None;
        for j in 0..n {
            if i != j && !visited[j] {
                let strength = s[(i, j)];
                if strength > max_strength {
                    max_strength = strength;
                    strongest_neighbor = Some(j);
                }
            }
        }

        // Form an aggregate with the strongest neighbor
        if let Some(j) = strongest_neighbor {
            aggregates[i] = aggregate_id;
            aggregates[j] = aggregate_id;
            visited[i] = true;
            visited[j] = true;
            aggregate_id += 1;
        } else {
            // No neighbor found, form a singleton aggregate
            aggregates[i] = aggregate_id;
            visited[i] = true;
            aggregate_id += 1;
        }
    }

    aggregates
}

/// Build a coarse graph from fine-level aggregates.
/// Each aggregate forms a node in the coarse graph.
/// The weights of edges between coarse nodes can be inherited or averaged from the fine graph.
fn build_coarse_graph(s: &Mat<f64>, aggregates: &[usize]) -> Mat<f64> {
    let max_agg_id = *aggregates.iter().max().unwrap_or(&0);
    let coarse_n = max_agg_id + 1;
    let mut coarse_mat = Mat::<f64>::zeros(coarse_n, coarse_n);
    let n = s.nrows();
    // Use a sequential loop for correctness
    for fine_node_i in 0..n {
        for fine_node_j in 0..s.ncols() {
            let agg_i = aggregates[fine_node_i];
            let agg_j = aggregates[fine_node_j];
            if agg_j < usize::MAX {
                let val = s[(fine_node_i, fine_node_j)];
                if val != 0.0 {
                    coarse_mat[(agg_i, agg_j)] += val;
                }
            }
        }
    }
    coarse_mat
}

/// Remap second pass aggregates to fine-level nodes.
fn remap_aggregates(first_pass: &[usize], second_pass: &[usize]) -> Vec<usize> {
    // Preallocate memory and use parallel iterator for improved performance
    first_pass
        .par_iter()
        .map(|&coarse_agg| second_pass[coarse_agg])
        .collect()
}

/// Construct the prolongation matrix P from the aggregate assignments.
/// For piecewise constant interpolation:
/// P_{ij} = 1 if node i is in aggregate j, else 0.
fn construct_prolongation(a: &Mat<f64>, aggregates: &[usize]) -> Mat<f64> {
    let n = a.nrows();
    let max_agg_id = *aggregates.iter().max().unwrap();
    let coarse_n = max_agg_id + 1;

    let p = Arc::new(Mutex::new(Mat::<f64>::zeros(n, coarse_n)));

    // Use parallel iterator to populate the matrix
    aggregates.par_iter().enumerate().for_each(|(i, &agg_id)| {
        let mut p = p.lock().unwrap();
        p[(i, agg_id)] = 1.0;
    });

    Arc::try_unwrap(p).expect("Failed to unwrap Arc").into_inner().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::MatVec;
    use faer::mat;

    #[test]
    fn test_amg_preconditioner_simple() {
        let matrix = mat![
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0]
        ];
        let r = vec![5.0, 5.0, 3.0];
        let mut z = vec![0.0; 3];

        let max_levels = 2;
        let coarsening_threshold = 0.1;
        let amg_preconditioner = AMG::new(&matrix, max_levels, coarsening_threshold);

        amg_preconditioner.apply(&r, &mut z).unwrap();

        let mut residual = vec![0.0; 3];
        matrix.matvec(&z, &mut residual);
        for i in 0..3 {
            residual[i] = r[i] - residual[i];
        }
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(residual_norm < 1.0, "Residual norm too high: {}", residual_norm);
    }

    #[test]
    fn test_amg_preconditioner_odd_size() {
        let matrix = mat![
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 4.0]
        ];
        let r = vec![5.0, 5.0, 3.0, 1.0];
        let mut z = vec![0.0; 4];

        let max_levels = 2;
        let coarsening_threshold = 0.1;
        let amg_preconditioner = AMG::new(&matrix, max_levels, coarsening_threshold);

        amg_preconditioner.apply(&r, &mut z).unwrap();

        let mut residual = vec![0.0; 4];
        matrix.matvec(&z, &mut residual);
        for i in 0..4 {
            residual[i] = r[i] - residual[i];
        }
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(residual_norm < 1.0, "Residual norm too high: {}", residual_norm);
    }

    #[test]
    fn test_smooth_interpolation_basic() {
        // Input matrices
        let mut interpolation = mat![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        let matrix = mat![
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5]
        ];
        let weight = 0.5;

        // Apply the function
        smooth_interpolation(&mut interpolation, &matrix, weight);

        // Expected result
        let expected = mat![
            [0.75, 1.75, 2.75],
            [3.5, 4.5, 5.5],
            [6.25, 7.25, 8.25]
        ];

        // Assertions
        assert_eq!(interpolation, expected);
    }

    #[test]
    fn test_smooth_interpolation_partial_overlap() {
        // Matrix has fewer columns than interpolation
        let mut interpolation = mat![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        let matrix = mat![
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5]
        ];
        let weight = 1.0;

        // Apply the function
        smooth_interpolation(&mut interpolation, &matrix, weight);

        // Expected result
        let expected = mat![
            [0.5, 1.5, 3.0, 4.0],
            [4.0, 5.0, 7.0, 8.0],
            [7.5, 8.5, 11.0, 12.0]
        ];

        // Assertions
        assert_eq!(interpolation, expected);
    }
}
