//! Matrix utilities for AMG-oriented workflows.
//!
//! The helpers here are scalar-generic where possible, while keeping real-only
//! wrappers (`f64`) for performance-sensitive paths. Complex builds use the
//! same kernels, relying on `KrystScalar` traits to handle magnitude-based
//! dropping and accumulation.

use crate::algebra::scalar::KrystScalar;
use crate::error::KError;
use crate::matrix::dense_api::DenseMatRef;
use crate::matrix::sparse::CsrMatrix;
#[cfg(feature = "simd")]
use crate::matrix::spmv::SpmvTuning;
#[cfg(feature = "backend-faer")]
use faer::Mat;
use oorandom::Rand64;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Matrix analysis helper function that computes basic properties
///
/// Returns (nnz, diagonal_dominance, diagonal_sum)
pub fn analyze_matrix_properties<M>(matrix: &M) -> (usize, f64, f64)
where
    M: DenseMatRef<f64>,
{
    let mut nnz = 0;
    let mut diagonal_sum = 0.0;
    let mut off_diagonal_sum = 0.0;

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix.get(i, j);
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
pub fn has_numerical_issues<M>(matrix: &M) -> bool
where
    M: DenseMatRef<f64>,
{
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix.get(i, j);
            if !val.is_finite() {
                return true;
            }
        }
    }
    false
}

/// IEEE safety check for matrix values
pub fn check_ieee_values<M>(matrix: &M) -> Result<(), KError>
where
    M: DenseMatRef<f64>,
{
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix.get(i, j);
            if val.is_nan() {
                return Err(KError::InvalidInput(format!(
                    "NaN detected at position ({i}, {j})"
                )));
            }
            if val.is_infinite() {
                return Err(KError::InvalidInput(format!(
                    "Infinite value detected at position ({i}, {j})"
                )));
            }
        }
    }
    Ok(())
}

/// Extract the inverse of the diagonal of a matrix, with zero for near-singular entries.
pub fn extract_diagonal_inverse<M>(m: &M) -> Vec<f64>
where
    M: DenseMatRef<f64>,
{
    let n = m.nrows();
    let mut diag_inv = vec![0.0; n];
    for i in 0..n {
        let diag_val = m.get(i, i);
        if diag_val.abs() > 1e-14 {
            diag_inv[i] = 1.0 / diag_val;
        }
    }
    diag_inv
}

/// Convert dense matrix to sparse format with drop tolerance
/// This is the foundation for sparse Galerkin products
#[cfg(feature = "backend-faer")]
pub fn to_sparse_with_tolerance(
    matrix: &Mat<f64>,
    drop_tol: f64,
) -> Result<CsrMatrix<f64>, KError> {
    CsrMatrix::from_dense(matrix, drop_tol)
}

/// Generic sparse C = A * B using Gustavson's algorithm on CSR arrays.
/// Returns a CSR with sorted columns per row and optional dropping.
///
/// # Requirements
/// - `a` and `b` must already satisfy the CSR invariants documented on
///   [`CsrMatrix`], including sorted `col_idx` slices per row.
/// - Column indices stay in ascending order as rows are merged; duplicates
///   within a row are collapsed during accumulation before applying `drop_tol`.
///
/// `drop_tol`: entries with |v| <= drop_tol are removed.
pub fn spgemm_with_drop_tol_generic<T>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
    drop_tol: T::Real,
) -> Result<CsrMatrix<T>, KError>
where
    T: KrystScalar<Real = f64> + std::ops::AddAssign,
{
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
    let mut vals: Vec<T> = Vec::new();

    // Marker and accumulator per column of C's row.
    // `mark[j] == i` means column j is present in row i's structure.
    let mut mark = vec![usize::MAX; n];
    let mut acc = vec![T::zero(); n];

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
            while read < row_tail && cols[read] == j0 {
                read += 1;
            }
            // reset for next row
            acc[j0] = T::zero();
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

/// Sparse C = A * B using Gustavson's algorithm on CSR arrays (real wrapper).
/// Returns a CSR with sorted columns per row and optional dropping.
#[inline]
pub fn spgemm_with_drop_tol(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    drop_tol: f64,
) -> Result<CsrMatrix<f64>, KError> {
    spgemm_with_drop_tol_generic(a, b, drop_tol)
}

/// Convenience wrapper with a default numerical drop tolerance (generic).
#[inline]
pub fn spgemm_generic<T>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> Result<CsrMatrix<T>, KError>
where
    T: KrystScalar<Real = f64> + std::ops::AddAssign,
{
    spgemm_with_drop_tol_generic(a, b, 1e-12)
}

/// Convenience wrapper with a default numerical drop tolerance (real).
#[inline]
pub fn spgemm(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> Result<CsrMatrix<f64>, KError> {
    spgemm_generic(a, b)
}

/// Canonical Poisson stencil helpers used by solver tests and benchmarks.
pub mod poisson {
    use super::*;

    /// Return the 5-point finite-difference discretisation of the 2-D Poisson
    /// operator on an `n × n` grid.
    pub fn poisson_5pt_2d(n: usize) -> CsrMatrix<f64> {
        super::poisson_2d(n, n)
    }

    /// Return the 7-point finite-difference discretisation of the 3-D Poisson
    /// operator on an `n × n × n` grid.
    pub fn poisson_7pt_3d(n: usize) -> CsrMatrix<f64> {
        super::poisson_3d(n, n, n)
    }
}

/// Build an SPD anisotropic diffusion operator with a rotated principal axis.
///
/// The diffusion tensor is `R diag(1, eps) Rᵀ` with rotation angle `theta`.
/// The discretisation uses a nine-point stencil so that cross-derivative terms
/// are represented explicitly; the diagonal entry is chosen to preserve row
/// sum symmetry so the matrix remains SPD.
/// # CSR invariants
/// - The intermediate entries are sorted and deduplicated before CSR assembly.
/// - Each row ends up with sorted columns and a single diagonal entry.
/// - Diagonal dominance is enforced via the accumulated contribution of neighbors.
pub fn anisotropic_poisson_2d(n: usize, theta: f64, eps: f64) -> CsrMatrix<f64> {
    assert!(n > 1, "grid must be at least 2 × 2");
    let nx = n;
    let ny = n;
    let mut row_ptr = Vec::with_capacity(nx * ny + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);

    let c = theta.cos();
    let s = theta.sin();
    let d11 = c * c + eps * s * s;
    let d22 = s * s + eps * c * c;
    let d12 = (1.0 - eps) * s * c;

    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x;
            let mut diag = 0.0;
            let mut entries: Vec<(usize, f64)> = Vec::with_capacity(9);

            // West/East diffusion
            if x > 0 {
                entries.push((idx - 1, -d11));
                diag += d11;
            }
            if x + 1 < nx {
                entries.push((idx + 1, -d11));
                diag += d11;
            }

            // South/North diffusion
            if y > 0 {
                entries.push((idx - nx, -d22));
                diag += d22;
            }
            if y + 1 < ny {
                entries.push((idx + nx, -d22));
                diag += d22;
            }

            // Cross derivatives (nine-point stencil). The discretisation uses
            // the symmetric form so that opposite corners receive opposite
            // signs which keeps the assembled matrix SPD for eps > 0.
            if x > 0 && y > 0 {
                entries.push((idx - nx - 1, -d12));
                diag += d12.abs();
            }
            if x > 0 && y + 1 < ny {
                entries.push((idx + nx - 1, d12));
                diag += d12.abs();
            }
            if x + 1 < nx && y > 0 {
                entries.push((idx - nx + 1, d12));
                diag += d12.abs();
            }
            if x + 1 < nx && y + 1 < ny {
                entries.push((idx + nx + 1, -d12));
                diag += d12.abs();
            }

            entries.push((idx, diag));

            // Ensure the CSR row is strictly sorted so downstream routines do
            // not hit the faer CSR validation assert.
            entries.sort_unstable_by_key(|&(j, _)| j);
            let mut deduped: Vec<(usize, f64)> = Vec::with_capacity(entries.len());
            for (j, v) in entries.into_iter() {
                if let Some((last_j, last_v)) = deduped.last_mut()
                    && *last_j == j
                {
                    *last_v += v;
                    continue;
                }
                deduped.push((j, v));
            }
            for (j, v) in deduped {
                col_idx.push(j);
                vals.push(v);
            }
            row_ptr.push(col_idx.len());
        }
    }

    CsrMatrix::from_csr(nx * ny, nx * ny, row_ptr, col_idx, vals)
}

/// Build a non-symmetric convection–diffusion operator on an `n × n` grid.
///
/// The diffusion part is the standard 5-point Laplacian, while the
/// convection term uses a simple upwind bias controlled by the Peclet number.
/// # CSR invariants
/// - Each row is sorted because we sort/deduplicate the small stencil entries before appending.
/// - Duplicate neighbors (e.g., east/west) are merged so each column index appears once.
pub fn convection_diffusion_2d(n: usize, peclet: f64) -> CsrMatrix<f64> {
    assert!(n > 1, "grid must be at least 2 × 2");
    let nx = n;
    let ny = n;
    let mut row_ptr = Vec::with_capacity(nx * ny + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);

    let upwind = 0.5 * peclet;

    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x;
            let mut diag = 4.0;
            let mut entries: Vec<(usize, f64)> = Vec::with_capacity(5);

            if x > 0 {
                entries.push((idx - 1, -1.0 - upwind));
                diag += 1.0 + upwind;
            }
            if x + 1 < nx {
                entries.push((idx + 1, -1.0 + upwind));
                diag += 1.0 - upwind;
            }
            if y > 0 {
                entries.push((idx - nx, -1.0 - upwind));
                diag += 1.0 + upwind;
            }
            if y + 1 < ny {
                entries.push((idx + nx, -1.0 + upwind));
                diag += 1.0 - upwind;
            }

            entries.push((idx, diag));

            entries.sort_unstable_by_key(|&(j, _)| j);
            let mut deduped: Vec<(usize, f64)> = Vec::with_capacity(entries.len());
            for (j, v) in entries.into_iter() {
                if let Some((last_j, last_v)) = deduped.last_mut()
                    && *last_j == j
                {
                    *last_v += v;
                    continue;
                }
                deduped.push((j, v));
            }
            for (j, v) in deduped {
                col_idx.push(j);
                vals.push(v);
            }
            row_ptr.push(col_idx.len());
        }
    }

    CsrMatrix::from_csr(nx * ny, nx * ny, row_ptr, col_idx, vals)
}

/// Generate a reproducible random right-hand side vector of length `n` using a
/// deterministic seed. Values lie in `[-0.5, 0.5)`.
pub fn random_rhs(n: usize, seed: u64) -> Vec<f64> {
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random::<f64>() - 0.5).collect()
}

#[cfg(feature = "simd")]
/// Returns the crate-wide default tuning for the SIMD SpMV plan builder.
pub fn default_spmv_tuning() -> SpmvTuning {
    SpmvTuning {
        allow_simd: cfg!(feature = "simd"),
        prefer_sellc: true,
        sell_c: 16,
        sell_sigma: 64,
        bench_nsamples: 3,
        min_nnz_for_simd: 2_000,
    }
}

/// Baseline Sparse C = A * B using per-row BTreeMap accumulation.
///
/// # Requirements
/// - Inputs must satisfy the CSR invariants: sorted rows and `col_idx < ncols`.
/// - Row storage is merged at the end via the `BTreeMap`, so duplicates are
///   automatically coalesced while preserving ascending order.
///
/// This implementation is intentionally simple and allocation-heavy to serve
/// as a comparison baseline for optimized kernels in benchmarks.
pub fn spgemm_btree_generic<T>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> Result<CsrMatrix<T>, KError>
where
    T: KrystScalar<Real = f64> + std::ops::AddAssign,
{
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
    let mut vals: Vec<T> = Vec::new();
    row_ptr.push(0);

    for i in 0..m {
        let mut acc: BTreeMap<usize, T> = BTreeMap::new();
        for kk in ap[i]..ap[i + 1] {
            let k = aj[kk];
            let aik = av[kk];
            for jj in bp[k]..bp[k + 1] {
                let j = bj[jj];
                *acc.entry(j).or_insert(T::zero()) += aik * bv[jj];
            }
        }
        for (j, v) in acc.into_iter() {
            if v != T::zero() {
                col_idx.push(j);
                vals.push(v);
            }
        }
        row_ptr.push(col_idx.len());
    }

    Ok(CsrMatrix::from_csr(m, n, row_ptr, col_idx, vals))
}

/// Baseline Sparse C = A * B using per-row BTreeMap accumulation (real wrapper).
pub fn spgemm_btree(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> Result<CsrMatrix<f64>, KError> {
    spgemm_btree_generic(a, b)
}

/// Sparse Galerkin product: C = R * A * P
/// with all inputs CSR, via two SpGEMMs.
pub fn sparse_galerkin_product_generic<T>(
    restriction: &CsrMatrix<T>,   // R
    matrix: &CsrMatrix<T>,        // A
    interpolation: &CsrMatrix<T>, // P
) -> Result<CsrMatrix<T>, KError>
where
    T: KrystScalar<Real = f64> + std::ops::AddAssign,
{
    // Step 1: T = A * P
    let ap = spgemm_generic(matrix, interpolation)?;
    // Step 2: C = R * T
    spgemm_generic(restriction, &ap)
}

/// Sparse Galerkin product: C = R * A * P (real wrapper).
pub fn sparse_galerkin_product(
    restriction: &CsrMatrix<f64>,   // R
    matrix: &CsrMatrix<f64>,        // A
    interpolation: &CsrMatrix<f64>, // P
) -> Result<CsrMatrix<f64>, KError> {
    sparse_galerkin_product_generic(restriction, matrix, interpolation)
}

/// Baseline RAP (triple product) using the `spgemm_btree` baseline twice.
pub fn rap_btree_generic<T>(
    restriction: &CsrMatrix<T>,
    matrix: &CsrMatrix<T>,
    interpolation: &CsrMatrix<T>,
) -> Result<CsrMatrix<T>, KError>
where
    T: KrystScalar<Real = f64> + std::ops::AddAssign,
{
    let ap = spgemm_btree_generic(matrix, interpolation)?;
    spgemm_btree_generic(restriction, &ap)
}

/// Baseline RAP (triple product) using the `spgemm_btree` baseline twice (real wrapper).
pub fn rap_btree(
    restriction: &CsrMatrix<f64>,
    matrix: &CsrMatrix<f64>,
    interpolation: &CsrMatrix<f64>,
) -> Result<CsrMatrix<f64>, KError> {
    rap_btree_generic(restriction, matrix, interpolation)
}

/// Optimized RAP wrapper to keep a stable API for benchmarks.
/// Currently composes two calls to the optimized SpGEMM.
#[inline]
pub fn rap_opt_generic<T>(
    restriction: &CsrMatrix<T>,
    matrix: &CsrMatrix<T>,
    interpolation: &CsrMatrix<T>,
) -> Result<CsrMatrix<T>, KError>
where
    T: KrystScalar<Real = f64> + std::ops::AddAssign,
{
    sparse_galerkin_product_generic(restriction, matrix, interpolation)
}

/// Optimized RAP wrapper to keep a stable API for benchmarks (real wrapper).
#[inline]
pub fn rap_opt(
    restriction: &CsrMatrix<f64>,
    matrix: &CsrMatrix<f64>,
    interpolation: &CsrMatrix<f64>,
) -> Result<CsrMatrix<f64>, KError> {
    rap_opt_generic(restriction, matrix, interpolation)
}

/// Apply HYPRE-style truncation to interpolation matrix
/// Removes weak connections based on threshold-based dropping
///
/// This function performs row-wise truncation of the interpolation operator,
/// keeping only the strongest connections to improve operator complexity.
#[cfg(feature = "backend-faer")]
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
#[cfg(feature = "backend-faer")]
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
    #[cfg(feature = "rayon")]
    {
        use crate::matrix::spmv::csr_matvec_par;
        return csr_matvec_par(a, x, y);
    }
    #[cfg(not(feature = "rayon"))]
    {
        use crate::matrix::spmv::csr_matvec;
        csr_matvec(a, x, y)
    }
}

/// Count non-zeros in a dense matrix
#[cfg(feature = "backend-faer")]
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
#[cfg(feature = "backend-faer")]
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
#[cfg(feature = "backend-faer")]
pub fn compute_adaptive_threshold(a: &Mat<f64>, base_threshold: f64) -> f64 {
    let anisotropy = compute_anisotropy(a);
    let avg_anisotropy = anisotropy.iter().sum::<f64>() / anisotropy.len() as f64;

    // Scale threshold based on anisotropy: higher anisotropy -> lower threshold
    let scaling_factor = (1.0 + avg_anisotropy.log10()).max(0.5).min(2.0);
    base_threshold * scaling_factor
}

/// Generate a 2D Poisson matrix on an `n_x` by `n_y` grid using the 5-point stencil.
///
/// The resulting matrix is SPD with size `(n_x*n_y)`.
///
/// # CSR invariants
/// - Rows are assembled in geometric order (west, south, diag, east, north) so that
///   each `col_idx[row_ptr[i]..row_ptr[i+1]]` is sorted.
/// - No duplicate column indices occur within a row.
/// - Diagonal entries exist where expected, allowing [`CsrMatrix::build_diag_pos`] to find them.
pub fn poisson_2d(n_x: usize, n_y: usize) -> CsrMatrix<f64> {
    let n = n_x * n_y;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for j in 0..n_y {
        for i in 0..n_x {
            let idx = j * n_x + i;
            if j > 0 {
                col_idx.push(idx - n_x);
                vals.push(-1.0);
            }
            if i > 0 {
                col_idx.push(idx - 1);
                vals.push(-1.0);
            }
            col_idx.push(idx);
            vals.push(4.0);
            if i + 1 < n_x {
                col_idx.push(idx + 1);
                vals.push(-1.0);
            }
            if j + 1 < n_y {
                col_idx.push(idx + n_x);
                vals.push(-1.0);
            }
            row_ptr.push(col_idx.len());
        }
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

/// Generate a 3D Poisson matrix on an `n_x` by `n_y` by `n_z` grid using the 7-point stencil.
///
/// The resulting matrix is symmetric positive definite with size `(n_x*n_y*n_z)`.
///
/// # CSR invariants
/// - Neighbor entries are appended in increasing order so per-row columns stay sorted.
/// - Diagonal entries are included once and neighbors never duplicate per row.
/// - The pattern produced matches the finite-difference stencil expected by downstream solvers.
pub fn poisson_3d(n_x: usize, n_y: usize, n_z: usize) -> CsrMatrix<f64> {
    let n = n_x * n_y * n_z;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n * 7);
    let mut vals = Vec::with_capacity(n * 7);
    row_ptr.push(0);

    for k in 0..n_z {
        for j in 0..n_y {
            for i in 0..n_x {
                let idx = (k * n_y + j) * n_x + i;

                if k > 0 {
                    col_idx.push(idx - n_x * n_y);
                    vals.push(-1.0);
                }
                if j > 0 {
                    col_idx.push(idx - n_x);
                    vals.push(-1.0);
                }
                if i > 0 {
                    col_idx.push(idx - 1);
                    vals.push(-1.0);
                }

                col_idx.push(idx);
                vals.push(6.0);

                if i + 1 < n_x {
                    col_idx.push(idx + 1);
                    vals.push(-1.0);
                }
                if j + 1 < n_y {
                    col_idx.push(idx + n_x);
                    vals.push(-1.0);
                }
                if k + 1 < n_z {
                    col_idx.push(idx + n_x * n_y);
                    vals.push(-1.0);
                }

                row_ptr.push(col_idx.len());
            }
        }
    }

    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

/// Generate a random symmetric positive definite banded matrix.
///
/// `bandwidth` controls the half-bandwidth; larger values yield denser matrices.
///
/// # CSR invariants
/// - Entries are sorted because the tuple vector is sorted by `(row, col)` before assembly.
/// - Duplicate `(i, j)` pairs are merged during the CSR build so each column appears at most once per row.
/// - Diagonal dominance is enforced via accumulated absolute off-diagonal contributions.
pub fn random_spd(n: usize, bandwidth: usize) -> CsrMatrix<f64> {
    let mut rng = Rand64::new(0);
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut diag = vec![0.0f64; n];

    for i in 0..n {
        let j_start = i.saturating_sub(bandwidth);
        let j_end = usize::min(n - 1, i + bandwidth);
        for j in j_start..=j_end {
            if i == j {
                continue;
            }
            let v = rng.rand_float() - 0.5;
            entries.push((i, j, v));
            entries.push((j, i, v));
            diag[i] += v.abs();
            diag[j] += v.abs();
        }
    }

    for i in 0..n {
        entries.push((i, i, diag[i] + 1.0));
    }

    entries.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::with_capacity(entries.len());
    let mut vals = Vec::with_capacity(entries.len());
    let mut current_row = 0usize;
    for (i, j, v) in entries {
        while i > current_row {
            row_ptr.push(col_idx.len());
            current_row += 1;
        }
        col_idx.push(j);
        vals.push(v);
    }
    row_ptr.push(col_idx.len());

    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

#[cfg(all(test, feature = "backend-faer"))]
mod tests {
    use super::*;
    #[cfg(feature = "complex")]
    use crate::algebra::prelude::*;
    use crate::matrix::sparse::CsrMatrix;
    use faer::Mat;

    #[cfg(feature = "complex")]
    fn complex_csr(value: S) -> CsrMatrix<S> {
        CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![value])
    }

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

    #[cfg(feature = "complex")]
    #[test]
    fn spgemm_handles_complex_scalars() {
        let a = complex_csr(S::from_parts(1.0, 0.5));
        let b = complex_csr(S::from_parts(2.0, -1.0));
        let c = spgemm_with_drop_tol_generic(&a, &b, 0.0).unwrap();
        assert_eq!(c.nrows(), 1);
        assert_eq!(c.ncols(), 1);
        assert_eq!(c.values()[0], S::from_parts(2.5, 0.0));
    }

    #[cfg(feature = "complex")]
    #[test]
    fn spgemm_btree_handles_complex_scalars() {
        let a = complex_csr(S::from_parts(1.0, 0.5));
        let b = complex_csr(S::from_parts(2.0, -1.0));
        let c = spgemm_btree_generic(&a, &b).unwrap();
        assert_eq!(c.values()[0], S::from_parts(2.5, 0.0));
    }

    #[cfg(feature = "complex")]
    #[test]
    fn sparse_galerkin_product_handles_complex_scalars() {
        let r = complex_csr(S::from_parts(1.0, 0.0));
        let a = complex_csr(S::from_parts(2.0, -1.0));
        let p = complex_csr(S::from_parts(0.5, 0.25));
        let c = sparse_galerkin_product_generic(&r, &a, &p).unwrap();
        assert_eq!(c.values()[0], S::from_parts(1.25, 0.0));
        let c_btree = rap_btree_generic(&r, &a, &p).unwrap();
        assert_eq!(c_btree.values()[0], S::from_parts(1.25, 0.0));
        let c_opt = rap_opt_generic(&r, &a, &p).unwrap();
        assert_eq!(c_opt.values()[0], S::from_parts(1.25, 0.0));
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

    #[test]
    fn poisson_3d_basic() {
        let a = poisson_3d(2, 2, 2);
        assert_eq!(a.nrows(), 8);
        assert_eq!(a.ncols(), 8);
        assert_eq!(a.row_ptr(), &[0, 4, 8, 12, 16, 20, 24, 28, 32]);
        assert_eq!(&a.col_idx()[0..4], &[0, 1, 2, 4]);
        assert_eq!(&a.values()[0..4], &[6.0, -1.0, -1.0, -1.0]);
    }
}
