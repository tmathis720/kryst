#![cfg(feature = "backend-faer")]

//! Incomplete LU factorization with threshold dropping plus partial pivoting (ILUTP).
//!
//! ILUTP is typically used for nonsymmetric, possibly indefinite problems (e.g., Navier–Stokes)
//! where simple ILU0/ILUT may fail due to unstable pivots. This implementation combines ILUT-style
//! dropping with partial pivoting in `U`.
//! It uses:
//! - a dense working matrix during factorization,
//! - explicit row permutations recorded in `row_perm`,
//! - threshold-based dropping controlled by `drop_tol` and `max_fill`,
//! - pivot tolerance `perm_tol` to steer stability.
//! The factors remain real (`f64`), but complex solvers reuse them via the `KPreconditioner`
//! bridge (`BridgeScratch`).
//!
//! # Real vs complex
//! Factorization is always performed in real arithmetic (`f64`), and complex Krylov solvers depend
//! on the [`KPreconditioner`] bridge that copies into real scratch buffers, applies ILUTP, and
//! copies back.
//!
//! For non-pivoting ILUT on `faer::Mat<f64>`, prefer [`Ilu`] with [`IluType::ILUT`].
//!
//! The algorithm follows Saad's *Iterative Methods for Sparse Linear Systems* with stability
//! safeguards tailored for pragmatic use.
//! See `examples/convection_diffusion_ilutp.rs` for a convection–diffusion demonstration.

#[cfg(feature = "complex")]
use crate::algebra::bridge::{BridgeScratch, copy_real_into_scalar, copy_scalar_to_real_in};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::{LocalPreconditioner, legacy::Preconditioner};
use crate::preconditioner::ilu_csr::{ReorderingKind, ReorderingOptions};
use crate::utils::permutation::{Permutation, amd_from_adj, rcm_from_adj};
use crate::utils::conditioning::{apply_dense_transforms, ConditioningOptions};
use crate::utils::metrics::{Counters, SolveTimer};
use faer::Mat;
use std::cmp::Ordering;
use std::sync::Mutex;
use std::time::Instant;

/// ILUTP preconditioner with threshold control and partial pivoting.
///
/// This preconditioner is particularly effective for nonsymmetric, indefinite
/// matrices arising from discretized Navier-Stokes equations.
/// It implements [`LocalPreconditioner`] and is designed to stay on the local rank
/// so that MPI routines can wrap it without additional communication.
#[derive(Debug)]
pub struct Ilutp {
    /// Lower triangular factor
    l_factor: Mat<f64>,
    /// Upper triangular factor
    u_factor: Mat<f64>,
    /// Row permutation vector
    row_perm: Vec<usize>,
    /// Maximum fill-in per row
    max_fill: usize,
    /// Drop tolerance for small elements
    drop_tol: f64,
    /// Pivot tolerance (threshold for pivoting)
    perm_tol: f64,
    workspace: IlutpWorkspace,
    /// Time spent in the most recent setup
    setup_time: f64,
    /// Solve timing counters
    solve_ctrs: Counters,
    /// Optional conditioning transforms applied before factorization.
    conditioning: ConditioningOptions,
    /// Reordering configuration for local factorization.
    reordering: ReorderingOptions,
    /// Permutation applied before factorization.
    perm: Permutation,
}

#[derive(Debug, Clone)]
pub struct IlutpStats {
    pub setup_time: f64,
    pub solve_time: f64,
    pub solve_count: usize,
}

#[derive(Debug)]
struct IlutpWorkspace {
    buf: Mutex<Vec<f64>>,
    size: usize,
}

impl IlutpWorkspace {
    fn new() -> Self {
        Self {
            buf: Mutex::new(Vec::new()),
            size: 0,
        }
    }

    fn ensure_size(&mut self, n: usize) {
        if n > self.size {
            let mut guard = self.buf.lock().unwrap();
            guard.resize(n, 0.0);
            self.size = n;
        }
    }

    #[inline]
    fn borrow_buf(&self, n: usize) -> std::sync::MutexGuard<'_, Vec<f64>> {
        debug_assert!(self.size >= n, "workspace not sized via setup()");
        self.buf.lock().unwrap()
    }
}

fn apply_reordering(
    matrix: &Mat<f64>,
    reordering: ReorderingOptions,
) -> Result<(Mat<f64>, Permutation), KError> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(KError::InvalidInput(
            "Matrix must be square for ILUTP".to_string(),
        ));
    }
    if !reordering.symmetric {
        return Err(KError::Unsupported(
            "Ilutp only supports symmetric reordering".to_string(),
        ));
    }

    let perm = match reordering.kind {
        ReorderingKind::None => Permutation::identity(n),
        ReorderingKind::Rcm => {
            let mut adj = dense_adj_from_matrix(matrix);
            rcm_from_adj(&mut adj)
        }
        ReorderingKind::Amd => {
            let mut adj = dense_adj_from_matrix(matrix);
            amd_from_adj(&mut adj)
        }
    };

    if matches!(reordering.kind, ReorderingKind::None) {
        Ok((matrix.clone(), perm))
    } else {
        Ok((permute_dense_symmetric(matrix, &perm), perm))
    }
}

fn dense_adj_from_matrix(matrix: &Mat<f64>) -> Vec<Vec<usize>> {
    let n = matrix.nrows();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if matrix[(i, j)] != 0.0 || matrix[(j, i)] != 0.0 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

fn permute_dense_symmetric(matrix: &Mat<f64>, perm: &Permutation) -> Mat<f64> {
    let n = matrix.nrows();
    let mut result = Mat::zeros(n, n);
    for i in 0..n {
        let old_i = perm.p[i];
        for j in 0..n {
            let old_j = perm.p[j];
            result[(i, j)] = matrix[(old_i, old_j)];
        }
    }
    result
}

impl Ilutp {
    /// Create a new ILUTP preconditioner with default parameters.
    pub fn new() -> Self {
        Self {
            l_factor: Mat::zeros(0, 0),
            u_factor: Mat::zeros(0, 0),
            row_perm: Vec::new(),
            max_fill: 10,
            drop_tol: 1e-4,
            perm_tol: 0.1,
            workspace: IlutpWorkspace::new(),
            setup_time: 0.0,
            solve_ctrs: Counters::new(),
            conditioning: ConditioningOptions::default(),
            reordering: ReorderingOptions::default(),
            perm: Permutation::identity(0),
        }
    }

    /// Create a new ILUTP preconditioner with custom parameters.
    pub fn with_params(max_fill: usize, drop_tol: f64, perm_tol: f64) -> Self {
        Self {
            l_factor: Mat::zeros(0, 0),
            u_factor: Mat::zeros(0, 0),
            row_perm: Vec::new(),
            max_fill,
            drop_tol,
            perm_tol,
            workspace: IlutpWorkspace::new(),
            setup_time: 0.0,
            solve_ctrs: Counters::new(),
            conditioning: ConditioningOptions::default(),
            reordering: ReorderingOptions::default(),
            perm: Permutation::identity(0),
        }
    }

    pub fn set_conditioning(&mut self, conditioning: ConditioningOptions) {
        self.conditioning = conditioning;
    }

    pub fn set_reordering(&mut self, reordering: ReorderingOptions) {
        self.reordering = reordering;
    }

    /// Set the maximum fill-in per row.
    pub fn set_max_fill(&mut self, max_fill: usize) {
        self.max_fill = max_fill;
    }

    /// Set the drop tolerance.
    pub fn set_drop_tolerance(&mut self, drop_tol: f64) {
        self.drop_tol = drop_tol;
    }

    /// Set the pivot tolerance.
    pub fn set_pivot_tolerance(&mut self, perm_tol: f64) {
        self.perm_tol = perm_tol;
    }

    /// Compute ILUTP factorization with simplified algorithm.
    fn compute_factorization(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(KError::InvalidInput(
                "Matrix must be square for ILUTP".to_string(),
            ));
        }

        // Initialize working matrix as a copy
        let mut work_matrix = matrix.clone();

        // Initialize permutation as identity
        self.row_perm = (0..n).collect();

        // Initialize L and U factors
        let mut l_factor = Mat::zeros(n, n);
        let mut u_factor = Mat::zeros(n, n);

        // Set diagonal of L to 1
        for i in 0..n {
            l_factor[(i, i)] = 1.0;
        }

        // Simplified ILUTP factorization
        for k in 0..n {
            // Find pivot row (simplified pivoting with zero avoidance)
            let mut pivot_row = k;
            let mut max_val = work_matrix[(k, k)].abs();

            // Look for a better pivot if current is too small
            if max_val < 1e-12 {
                for i in (k + 1)..n {
                    let val = work_matrix[(i, k)].abs();
                    if val > max_val {
                        max_val = val;
                        pivot_row = i;
                    }
                }
            } else {
                // Standard partial pivoting
                for i in (k + 1)..n {
                    let val = work_matrix[(i, k)].abs();
                    if val > max_val && val > self.perm_tol * max_val {
                        max_val = val;
                        pivot_row = i;
                    }
                }
            }

            // Swap rows if needed
            if pivot_row != k {
                self.row_perm.swap(k, pivot_row);
                for j in 0..n {
                    let temp = work_matrix[(k, j)];
                    work_matrix[(k, j)] = work_matrix[(pivot_row, j)];
                    work_matrix[(pivot_row, j)] = temp;
                }
            }

            let pivot = work_matrix[(k, k)];
            if pivot.abs() < 1e-12 {
                // TODO: consider reusing `preconditioner::pivot` handling (e.g., `PivotPolicy`)
                // for consistent stabilization instead of this ad-hoc regularization.
                // Handle near-zero pivot by regularization
                work_matrix[(k, k)] = if pivot >= 0.0 { 1e-12 } else { -1e-12 };
                eprintln!(
                    "Warning: Near-zero pivot at row {} regularized from {} to {}",
                    k,
                    pivot,
                    work_matrix[(k, k)]
                );
            }

            // Store U factor
            for j in k..n {
                u_factor[(k, j)] = work_matrix[(k, j)];
            }

            // Elimination with threshold dropping
            for i in (k + 1)..n {
                if work_matrix[(i, k)].abs() > self.drop_tol {
                    let multiplier = work_matrix[(i, k)] / pivot;
                    l_factor[(i, k)] = multiplier;

                    // Collect row elements for fill-in control
                    let mut row_elements = Vec::new();
                    for j in (k + 1)..n {
                        let new_val = work_matrix[(i, j)] - multiplier * work_matrix[(k, j)];
                        if new_val.abs() > self.drop_tol {
                            row_elements.push((j, new_val));
                        }
                    }

                    // Apply fill-in control
                    if row_elements.len() > self.max_fill {
                        row_elements.sort_by(|a, b| {
                            b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal)
                        });
                        row_elements.truncate(self.max_fill);
                    }

                    // Clear row and set selected elements
                    for j in (k + 1)..n {
                        work_matrix[(i, j)] = 0.0;
                    }
                    for (j, val) in row_elements {
                        work_matrix[(i, j)] = val;
                    }
                }
            }
        }

        self.l_factor = l_factor;
        self.u_factor = u_factor;
        self.workspace.ensure_size(n);
        Ok(())
    }

    /// Forward substitution: solve Ly = Pb
    fn forward_solve(&self, b: &[f64], y: &mut [f64]) -> Result<(), KError> {
        let n = self.l_factor.nrows();

        // Apply row permutation
        for i in 0..n {
            y[i] = b[self.row_perm[i]];
        }

        // Forward substitution: Ly = Pb
        for i in 0..n {
            for j in 0..i {
                y[i] -= self.l_factor[(i, j)] * y[j];
            }
            // L has unit diagonal, so no division needed
        }

        Ok(())
    }

    /// Backward substitution: solve Ux = y
    fn backward_solve(&self, y: &[f64], x: &mut [f64]) -> Result<(), KError> {
        let n = self.u_factor.nrows();
        x.copy_from_slice(y);

        // Backward substitution: Ux = y
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= self.u_factor[(i, j)] * x[j];
            }
            let diag = self.u_factor[(i, i)];
            if diag.abs() < 1e-14 {
                return Err(KError::ZeroPivot(i));
            }
            x[i] /= diag;
        }

        Ok(())
    }

    fn apply_slice(&self, input: &[f64], output: &mut [f64]) -> Result<(), KError> {
        let n = input.len();
        let expected = self.l_factor.nrows();
        if output.len() != n || expected != n {
            return Err(KError::InvalidInput(format!(
                "Ilutp::apply dimension mismatch: input.len()={}, output.len()={}, n={}",
                n,
                output.len(),
                expected
            )));
        }

        let _timer = SolveTimer::start(&self.solve_ctrs);
        if matches!(self.reordering.kind, ReorderingKind::None) {
            let mut temp_guard = self.workspace.borrow_buf(n);
            let temp = &mut temp_guard[..n];
            self.forward_solve(input, temp)?;
            self.backward_solve(temp, output)?;
        } else {
            let mut perm_input = vec![0.0; n];
            self.perm.apply_vec(input, &mut perm_input);
            let mut temp_guard = self.workspace.borrow_buf(n);
            let temp = &mut temp_guard[..n];
            self.forward_solve(&perm_input, temp)?;
            let mut perm_output = vec![0.0; n];
            self.backward_solve(temp, &mut perm_output)?;
            self.perm.apply_vec_t(&perm_output, output);
        }
        Ok(())
    }
}

impl Default for Ilutp {
    fn default() -> Self {
        Self::new()
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for Ilutp {
    /// Setup the ILUTP preconditioner by computing the factorization.
    fn setup(&mut self, matrix: &Mat<f64>) -> Result<(), KError> {
        let start = Instant::now();
        self.solve_ctrs = Counters::new();
        let mut conditioned = None;
        let matrix = if self.conditioning.is_active() {
            let mut local = matrix.clone();
            apply_dense_transforms("ILUTP", &mut local, &self.conditioning)?;
            conditioned = Some(local);
            conditioned.as_ref().unwrap()
        } else {
            matrix
        };
        let (matrix, perm) = apply_reordering(matrix, self.reordering)?;
        self.perm = perm;
        self.compute_factorization(&matrix)?;
        self.setup_time = start.elapsed().as_secs_f64();
        Ok(())
    }

    /// Apply the ILUTP preconditioner: solve M⁻¹x = b where M ≈ A.
    fn apply(
        &self,
        _side: crate::preconditioner::PcSide,
        input: &Vec<f64>,
        output: &mut Vec<f64>,
    ) -> Result<(), KError> {
        self.apply_slice(input, output)
    }
}

impl LocalPreconditioner<f64> for Ilutp {
    fn dims(&self) -> (usize, usize) {
        let n = self.l_factor.nrows();
        (n, n)
    }

    fn apply_local(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        let (n, _) = LocalPreconditioner::<f64>::dims(self);
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        self.apply_slice(x, y)
    }
}

impl Ilutp {
    pub fn get_stats(&self) -> IlutpStats {
        let (total_ns, count, _) = self.solve_ctrs.snapshot();
        IlutpStats {
            setup_time: self.setup_time,
            solve_time: if count == 0 {
                0.0
            } else {
                total_ns as f64 / count as f64 / 1e9
            },
            solve_count: count as usize,
        }
    }
}

#[cfg(test)]
impl Ilutp {
    pub fn test_max_fill(&self) -> usize {
        self.max_fill
    }

    pub fn test_drop_tolerance(&self) -> f64 {
        self.drop_tol
    }

    pub fn test_perm_tolerance(&self) -> f64 {
        self.perm_tol
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for Ilutp {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        let n = self.l_factor.nrows();
        (n, n)
    }

    fn apply_s(
        &self,
        _side: crate::preconditioner::PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        let (rows, cols) = LocalPreconditioner::<f64>::dims(self);
        let n = x.len();
        if x.len() != y.len() || rows != n || cols != n {
            return Err(KError::InvalidInput(format!(
                "Ilutp::apply_s dimension mismatch: expected {}x{}, got x.len()={} y.len()={}",
                rows,
                cols,
                x.len(),
                y.len()
            )));
        }
        scratch.with_pair(n, |xr, yr| {
            copy_scalar_to_real_in(x, xr);
            self.apply_slice(xr, yr)?;
            copy_real_into_scalar(yr, y);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn test_ilutp_basic() {
        let mut ilutp = Ilutp::new();

        // Test matrix: simple 3x3 diagonally dominant
        let mut matrix = Mat::zeros(3, 3);
        matrix[(0, 0)] = 4.0;
        matrix[(0, 1)] = -1.0;
        matrix[(1, 0)] = -1.0;
        matrix[(1, 1)] = 4.0;
        matrix[(1, 2)] = -1.0;
        matrix[(2, 1)] = -1.0;
        matrix[(2, 2)] = 4.0;

        // Setup should succeed
        assert!(ilutp.setup(&matrix).is_ok());

        // Test application
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        assert!(
            ilutp
                .apply(crate::preconditioner::PcSide::Left, &input, &mut output)
                .is_ok()
        );
    }

    #[test]
    fn test_ilutp_with_custom_params() {
        let mut ilutp = Ilutp::with_params(5, 1e-6, 0.01);

        // Test parameter setting
        ilutp.set_max_fill(15);
        ilutp.set_drop_tolerance(1e-8);
        ilutp.set_pivot_tolerance(0.1);

        assert_eq!(ilutp.max_fill, 15);
        assert_eq!(ilutp.drop_tol, 1e-8);
        assert_eq!(ilutp.perm_tol, 0.1);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn apply_s_matches_real_path() {
        use crate::algebra::bridge::BridgeScratch;
        use crate::algebra::prelude::*;
        use crate::ops::kpc::KPreconditioner;

        let mut ilutp = Ilutp::with_params(4, 1e-6, 0.05);

        let mut matrix = Mat::zeros(2, 2);
        matrix[(0, 0)] = 5.0;
        matrix[(0, 1)] = -1.0;
        matrix[(1, 0)] = -2.0;
        matrix[(1, 1)] = 6.0;
        ilutp.setup(&matrix).unwrap();

        let rhs_real = vec![3.0f64, -1.0];
        let mut out_real = vec![0.0; rhs_real.len()];
        ilutp
            .apply(
                crate::preconditioner::PcSide::Left,
                &rhs_real,
                &mut out_real,
            )
            .expect("ilutp real apply");

        let rhs_s: Vec<S> = rhs_real.iter().copied().map(S::from_real).collect();
        let mut out_s = vec![S::zero(); rhs_s.len()];
        let mut scratch = BridgeScratch::default();
        ilutp
            .apply_s(
                crate::preconditioner::PcSide::Left,
                &rhs_s,
                &mut out_s,
                &mut scratch,
            )
            .expect("ilutp apply_s");

        for (ys, yr) in out_s.iter().zip(out_real.iter()) {
            assert!((ys.real() - yr).abs() < 1e-10);
        }
    }
}
