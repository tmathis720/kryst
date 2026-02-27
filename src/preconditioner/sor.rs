//! SOR/SSOR preconditioner implementation.
//!
//! Implements Successive Over-Relaxation (SOR) and Symmetric SOR (SSOR) as a preconditioner for iterative solvers.
//!
//! # Overview
//!
//! SOR is an iterative method and preconditioner that generalizes Gauss–Seidel by introducing a relaxation parameter ω.
//! SSOR applies both forward and backward sweeps for improved convergence. This implementation supports various sweep types
//! and options via bitflags, and can be used as a preconditioner for Krylov solvers.
//!
//! Note: multi-color sweeps execute per-color in parallel and are an approximation to classical Gauss–Seidel; they may
//! converge differently from sequential sweeps.
//!
//! # Usage
//!
//! - Create a `Sor` preconditioner with the desired parameters (ω, sweeps, etc).
//! - Call `setup` with the system matrix to extract the diagonal and store its inverse.
//! - Use `apply` to apply the preconditioner to a vector.
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Section 10.2.
//! - https://en.wikipedia.org/wiki/Successive_over-relaxation

#[cfg(feature = "complex")]
use crate::algebra::bridge::BridgeScratch;
use crate::algebra::parallel;
use crate::algebra::prelude::*;
use crate::core::traits::{Indexing, MatVec};
use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
#[cfg(feature = "complex")]
use crate::ops::kpc::KPreconditioner;
use crate::preconditioner::Preconditioner as ObjPreconditioner;
#[cfg(feature = "complex")]
use crate::preconditioner::bridge::{
    apply_pc_mut_s as bridge_apply_pc_mut_s, apply_pc_s as bridge_apply_pc_s,
};
use crate::preconditioner::{PcDistributedSupport, PcSide, legacy::Preconditioner};
use crate::utils::coloring::{build_blocks_from_colors, csr_distance2_coloring};
use bitflags::bitflags;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Complex arithmetic capability for this preconditioner implementation.
pub const COMPLEX_SUPPORT: &str = "native_complex";

#[cfg(feature = "complex")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SorComplexKernelMode {
    Native,
    DegradedSplitRealImag,
}

bitflags! {
    /// Bitflags for SOR sweep types and options.
    ///
    /// Allows selection of forward, backward, symmetric, and Eisenstat sweeps.
    #[derive(Copy, Clone, Debug)]
    pub struct MatSorType: u32 {
        const ZERO_INITIAL_GUESS       = 0b000_00001;
        const APPLY_LOWER              = 0b000_00010; // forward Gauss–Seidel
        const APPLY_UPPER              = 0b000_00100; // backward
        const SYMMETRIC_SWEEP          = Self::APPLY_LOWER.bits() | Self::APPLY_UPPER.bits();
        const LOCAL_FORWARD_SWEEP      = 0b000_01000;
        const LOCAL_BACKWARD_SWEEP     = 0b000_10000;
        const LOCAL_SYMMETRIC_SWEEP    = Self::LOCAL_FORWARD_SWEEP.bits() | Self::LOCAL_BACKWARD_SWEEP.bits();
        const EISENSTAT                = 0b0010_0000;
        const COLOR_SWEEP              = 0b0100_0000;
    }
}

/// SOR/SSOR preconditioner struct.
///
/// - `its`: Number of outer SOR iterations
/// - `lits`: Number of local iterations (unused)
/// - `sym`: Sweep type (forward, backward, symmetric, etc)
/// - `omega`: Relaxation parameter (ω)
/// - `fshift`: Diagonal shift (for regularization)
/// - `inv_diag`: Inverse diagonal entries
/// - `a`: Matrix reference (after setup)
pub struct Sor<M, V> {
    pub its: usize,       // Number of outer SOR iterations
    pub lits: usize,      // Number of local iterations (unused)
    pub sym: MatSorType,  // Sweep type (forward, backward, symmetric)
    pub omega: S,         // Relaxation parameter
    pub fshift: S,        // Diagonal shift
    pub inv_diag: Vec<S>, // Inverse diagonal entries
    pub a: Option<M>,     // Matrix reference (after setup)
    _phantom: PhantomData<V>,
}

impl<M, V> Sor<M, V> {
    /// Create a new SOR preconditioner with the given parameters.
    pub fn new(omega: S, its: usize, lits: usize, sym: MatSorType, fshift: S) -> Self {
        Self {
            its,
            lits,
            sym,
            omega,
            fshift,
            inv_diag: Vec::new(),
            a: None,
            _phantom: PhantomData,
        }
    }
    // Setters and getters for parameters
    pub fn set_omega(&mut self, omega: S) {
        self.omega = omega;
    }
    pub fn omega(&self) -> S {
        self.omega
    }
    pub fn set_its(&mut self, its: usize) {
        self.its = its;
    }
    pub fn its(&self) -> usize {
        self.its
    }
    pub fn set_lits(&mut self, lits: usize) {
        self.lits = lits;
    }
    pub fn lits(&self) -> usize {
        self.lits
    }
    pub fn set_sym(&mut self, sym: MatSorType) {
        self.sym = sym;
    }
    pub fn sym(&self) -> MatSorType {
        self.sym
    }
    pub fn set_fshift(&mut self, fshift: S) {
        self.fshift = fshift;
    }
    pub fn fshift(&self) -> S {
        self.fshift
    }
}

impl<M, V> fmt::Display for Sor<M, V>
where
    S: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SOR(omega={}, its={}, lits={}, sym={:?}, fshift={})",
            self.omega, self.its, self.lits, self.sym, self.fshift
        )
    }
}

impl<M, V> Preconditioner<M, V> for Sor<M, V>
where
    M: MatVec<V> + Indexing + Clone + std::ops::Index<(usize, usize), Output = S>,
    V: AsRef<[S]> + AsMut<[S]> + From<Vec<S>>,
{
    /// Setup SOR: extract diagonal and store inverse.
    ///
    /// Stores a reference to the matrix and computes the inverse of the diagonal (with optional shift).
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        self.a = Some(a.clone());
        let n = a.nrows();
        self.inv_diag.resize(n, S::zero());
        for i in 0..n {
            let aii = a[(i, i)] + self.fshift;
            if aii == S::zero() {
                return Err(KError::ZeroPivot(i));
            }
            self.inv_diag[i] = S::one() / aii;
        }
        Ok(())
    }

    /// Apply SOR/SSOR preconditioner: `y = M⁻¹ x`.
    ///
    /// The solver decides whether this result lands on the left or right based
    /// on `side`; the actual sweeps are identical. For symmetric SOR, a forward
    /// sweep is followed by a backward sweep using the forward result as input.
    fn apply(&self, side: PcSide, x: &V, y: &mut V) -> Result<(), KError> {
        let a = self.a.as_ref().expect("SOR not setup");
        let x = x.as_ref();
        let y_mut = y.as_mut();
        y_mut.fill(S::zero());

        for _ in 0..self.its {
            match (side, self.sym) {
                (_, s) if s.contains(MatSorType::SYMMETRIC_SWEEP) => {
                    self.forward_sweep(a, x, y_mut);
                    let tmp = y_mut.to_vec();
                    y_mut.fill(S::zero());
                    self.backward_sweep(a, &tmp, y_mut);
                }
                (PcSide::Left, s) | (PcSide::Right, s) if s.contains(MatSorType::APPLY_LOWER) => {
                    self.forward_sweep(a, x, y_mut);
                }
                (PcSide::Left, s) | (PcSide::Right, s) if s.contains(MatSorType::APPLY_UPPER) => {
                    self.backward_sweep(a, x, y_mut);
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl<M, V> Sor<M, V>
where
    M: MatVec<V> + Indexing + std::ops::Index<(usize, usize), Output = S>,
    V: AsRef<[S]> + AsMut<[S]> + From<Vec<S>>,
{
    fn forward_sweep(&self, a: &M, x: &[S], y: &mut [S]) {
        let n = x.len();
        for i in 0..n {
            let mut sigma = S::zero();
            for j in 0..i {
                sigma = sigma + a[(i, j)] * y[j];
            }
            if !self.sym.contains(MatSorType::EISENSTAT) {
                for j in (i + 1)..n {
                    sigma = sigma + a[(i, j)] * x[j];
                }
            }
            let xi = x[i];
            let yi = (xi - sigma) * self.inv_diag[i];
            y[i] = (S::one() - self.omega) * xi + self.omega * yi;
        }
    }

    fn backward_sweep(&self, a: &M, x: &[S], y: &mut [S]) {
        let n = x.len();
        for ii in (0..n).rev() {
            let mut sigma = S::zero();
            for j in (ii + 1)..n {
                sigma = sigma + a[(ii, j)] * y[j];
            }
            if !self.sym.contains(MatSorType::EISENSTAT) {
                for j in 0..ii {
                    sigma = sigma + a[(ii, j)] * y[j];
                }
            }
            let xi = x[ii];
            let yi = (xi - sigma) * self.inv_diag[ii];
            y[ii] = (S::one() - self.omega) * xi + self.omega * yi;
        }
    }
}

#[cfg(all(test, feature = "legacy-pc-bridge"))]
mod tests_symmetric;

// -----------------------------------------------------------------------------
// Object-safe SOR preconditioner over LinOp + CSR
// -----------------------------------------------------------------------------

/// Object-safe SOR/SSOR preconditioner operating on `&dyn LinOp<S=f64>`.
pub struct SorPc {
    omega: f64,
    sweeps: usize,
    mat_side: MatSorType,
    fshift: f64,
    a_csr: Option<Arc<CsrMatrix<f64>>>,
    inv_diag: Vec<R>,
    #[cfg(feature = "complex")]
    a_csr_complex: Option<Arc<CsrMatrix<S>>>,
    #[cfg(feature = "complex")]
    inv_diag_complex: Vec<S>,
    #[cfg(feature = "complex")]
    complex_force_split_fallback: bool,
    #[cfg(feature = "complex")]
    complex_fallback_diagnostics: Mutex<Option<String>>,
    color_of: Vec<usize>,
    color_blocks: Vec<Vec<usize>>,
    n: usize,
    scratch: Mutex<Vec<R>>, // reuse for symmetric sweep without heap activity
}

impl SorPc {
    pub fn new(omega: f64, sweeps: usize, mat_side: MatSorType, fshift: f64) -> Self {
        Self {
            omega,
            sweeps,
            mat_side,
            fshift,
            a_csr: None,
            inv_diag: Vec::new(),
            #[cfg(feature = "complex")]
            a_csr_complex: None,
            #[cfg(feature = "complex")]
            inv_diag_complex: Vec::new(),
            #[cfg(feature = "complex")]
            complex_force_split_fallback: false,
            #[cfg(feature = "complex")]
            complex_fallback_diagnostics: Mutex::new(None),
            color_of: Vec::new(),
            color_blocks: Vec::new(),
            n: 0,
            scratch: Mutex::new(Vec::new()),
        }
    }

    #[cfg(feature = "complex")]
    pub fn set_complex_force_split_fallback(&mut self, on: bool) {
        self.complex_force_split_fallback = on;
    }

    #[cfg(feature = "complex")]
    pub fn complex_kernel_mode(&self) -> SorComplexKernelMode {
        if self.complex_force_split_fallback {
            SorComplexKernelMode::DegradedSplitRealImag
        } else {
            SorComplexKernelMode::Native
        }
    }

    #[cfg(feature = "complex")]
    pub fn complex_fallback_diagnostics(&self) -> Option<String> {
        self.complex_fallback_diagnostics
            .lock()
            .ok()
            .and_then(|msg| msg.clone())
    }

    fn ensure_inv_diag(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        let n = a.nrows().min(a.ncols());
        self.inv_diag.resize(n, R::zero());
        for i in 0..n {
            let rs = a.row_ptr()[i];
            let re = a.row_ptr()[i + 1];
            let mut aii = 0.0;
            for p in rs..re {
                if a.col_idx()[p] == i {
                    aii = a.values()[p];
                    break;
                }
            }
            let aii_shift = aii + self.fshift;
            if aii_shift == 0.0 {
                return Err(KError::ZeroPivot(i));
            }
            self.inv_diag[i] = 1.0 / aii_shift;
        }
        self.n = n;
        // resize scratch once
        let mut s = self.scratch.lock().unwrap();
        if s.len() != n {
            s.resize(n, R::zero());
        }
        Ok(())
    }

    #[cfg(feature = "complex")]
    fn ensure_inv_diag_complex(&mut self, a: &CsrMatrix<S>) -> Result<(), KError> {
        let n = a.nrows().min(a.ncols());
        self.inv_diag_complex.resize(n, S::zero());
        for i in 0..n {
            let rs = a.row_ptr()[i];
            let re = a.row_ptr()[i + 1];
            let mut aii = S::zero();
            for p in rs..re {
                if a.col_idx()[p] == i {
                    aii = a.values()[p];
                    break;
                }
            }
            let aii_shift = aii + S::from_real(self.fshift);
            if aii_shift == S::zero() {
                return Err(KError::ZeroPivot(i));
            }
            self.inv_diag_complex[i] = S::one() / aii_shift;
        }
        self.n = n;
        Ok(())
    }

    fn ensure_coloring(&mut self, a: &CsrMatrix<f64>) {
        if self.mat_side.contains(MatSorType::COLOR_SWEEP) {
            self.color_of = csr_distance2_coloring(a);
            self.color_blocks = build_blocks_from_colors(&self.color_of);
        } else {
            self.color_of.clear();
            self.color_blocks.clear();
        }
    }

    #[inline]
    fn forward_sweep(&self, a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let eisenstat = self.mat_side.contains(MatSorType::EISENSTAT);
        for i in 0..n {
            let mut sigma = 0.0;
            let rs = rp[i];
            let re = rp[i + 1];
            for p in rs..re {
                let j = cj[p];
                if j < i {
                    sigma = f64::mul_add(vv[p], y[j], sigma);
                } else if !eisenstat && j > i {
                    sigma = f64::mul_add(vv[p], x[j], sigma);
                }
            }
            let xi = x[i];
            let yi = (xi - sigma) * self.inv_diag[i];
            y[i] = yi;
        }
    }

    #[inline]
    fn forward_sweep_color(&self, a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let eisenstat = self.mat_side.contains(MatSorType::EISENSTAT);
        let color_of = &self.color_of;
        for (color, bucket) in self.color_blocks.iter().enumerate() {
            let y_ptr = AtomicPtr::new(y.as_mut_ptr());
            parallel::par_for_each_index(bucket.len(), |k| unsafe {
                let i = *bucket.get_unchecked(k);
                let mut sigma = 0.0;
                let rs = *rp.get_unchecked(i);
                let re = *rp.get_unchecked(i + 1);
                let y_ptr = y_ptr.load(Ordering::Relaxed);
                for p in rs..re {
                    let j = *cj.get_unchecked(p);
                    if j >= n || j == i {
                        continue;
                    }
                    let color_j = *color_of.get_unchecked(j);
                    if color_j < color {
                        sigma = f64::mul_add(*vv.get_unchecked(p), *y_ptr.add(j), sigma);
                    } else if !eisenstat && color_j > color {
                        sigma = f64::mul_add(*vv.get_unchecked(p), *x.get_unchecked(j), sigma);
                    }
                }
                let xi = *x.get_unchecked(i);
                let yi = (xi - sigma) * *self.inv_diag.get_unchecked(i);
                *y_ptr.add(i) = yi;
            });
        }
    }

    #[inline]
    fn backward_sweep(&self, a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let eisenstat = self.mat_side.contains(MatSorType::EISENSTAT);
        for ii in (0..n).rev() {
            let mut sigma = 0.0;
            let rs = rp[ii];
            let re = rp[ii + 1];
            for p in rs..re {
                let j = cj[p];
                if j > ii {
                    sigma = f64::mul_add(vv[p], y[j], sigma);
                } else if !eisenstat && j < ii {
                    sigma = f64::mul_add(vv[p], y[j], sigma);
                }
            }
            let xi = x[ii];
            let yi = (xi - sigma) * self.inv_diag[ii];
            y[ii] = (1.0 - self.omega) * xi + self.omega * yi;
        }
    }

    #[inline]
    fn backward_sweep_color(&self, a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let eisenstat = self.mat_side.contains(MatSorType::EISENSTAT);
        let color_of = &self.color_of;
        for color in (0..self.color_blocks.len()).rev() {
            let bucket = &self.color_blocks[color];
            let y_ptr = AtomicPtr::new(y.as_mut_ptr());
            parallel::par_for_each_index(bucket.len(), |k| unsafe {
                let i = *bucket.get_unchecked(k);
                let mut sigma = 0.0;
                let rs = *rp.get_unchecked(i);
                let re = *rp.get_unchecked(i + 1);
                let y_ptr = y_ptr.load(Ordering::Relaxed);
                for p in rs..re {
                    let j = *cj.get_unchecked(p);
                    if j >= n || j == i {
                        continue;
                    }
                    let color_j = *color_of.get_unchecked(j);
                    if color_j > color || (!eisenstat && color_j < color) {
                        sigma = f64::mul_add(*vv.get_unchecked(p), *y_ptr.add(j), sigma);
                    }
                }
                let xi = *x.get_unchecked(i);
                let yi = (xi - sigma) * *self.inv_diag.get_unchecked(i);
                *y_ptr.add(i) = (1.0 - self.omega) * xi + self.omega * yi;
            });
        }
    }
    fn apply_real(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        let a = self
            .a_csr
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("SOR not setup".into()))?;
        if x.len() != self.n || y.len() != self.n {
            return Err(KError::InvalidInput(
                "dimension mismatch in SorPc::apply".into(),
            ));
        }
        for _ in 0..self.sweeps {
            match (side, self.mat_side) {
                (_, s) if s.contains(MatSorType::SYMMETRIC_SWEEP) => {
                    if s.contains(MatSorType::COLOR_SWEEP) && !self.color_blocks.is_empty() {
                        self.forward_sweep_color(a, x, y);
                    } else {
                        self.forward_sweep(a, x, y);
                    }
                    let mut s = self.scratch.lock().unwrap();
                    s.copy_from_slice(y);
                    if self.mat_side.contains(MatSorType::COLOR_SWEEP)
                        && !self.color_blocks.is_empty()
                    {
                        self.backward_sweep_color(a, &s, y);
                    } else {
                        self.backward_sweep(a, &s, y);
                    }
                }
                (PcSide::Left, s) | (PcSide::Right, s) if s.contains(MatSorType::APPLY_LOWER) => {
                    if s.contains(MatSorType::COLOR_SWEEP) && !self.color_blocks.is_empty() {
                        self.forward_sweep_color(a, x, y);
                    } else {
                        self.forward_sweep(a, x, y);
                    }
                }
                (PcSide::Left, s) | (PcSide::Right, s) if s.contains(MatSorType::APPLY_UPPER) => {
                    if s.contains(MatSorType::COLOR_SWEEP) && !self.color_blocks.is_empty() {
                        self.backward_sweep_color(a, x, y);
                    } else {
                        self.backward_sweep(a, x, y);
                    }
                }
                _ => {
                    if self.mat_side.contains(MatSorType::COLOR_SWEEP)
                        && !self.color_blocks.is_empty()
                    {
                        self.forward_sweep_color(a, x, y);
                    } else {
                        self.forward_sweep(a, x, y);
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "complex")]
    fn apply_complex(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if self.complex_force_split_fallback {
            if let Ok(mut msg) = self.complex_fallback_diagnostics.lock() {
                *msg = Some("sor_complex_degraded_split_real_imag: forced fallback enabled; disable it to use native complex sweep".to_string());
            }
            let mut xr = vec![0.0; x.len()];
            let mut xi = vec![0.0; x.len()];
            let mut yr = vec![0.0; y.len()];
            let mut yi = vec![0.0; y.len()];
            for i in 0..x.len() {
                xr[i] = x[i].real();
                xi[i] = x[i].imag();
            }
            self.apply_real(side, &xr, &mut yr)?;
            self.apply_real(side, &xi, &mut yi)?;
            for i in 0..y.len() {
                y[i] = S::from_parts(yr[i], yi[i]);
            }
            return Ok(());
        }
        if let Ok(mut msg) = self.complex_fallback_diagnostics.lock() {
            *msg = None;
        }
        let a = self
            .a_csr_complex
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("SOR not setup".into()))?;
        if x.len() != self.n || y.len() != self.n {
            return Err(KError::InvalidInput(
                "dimension mismatch in SorPc::apply".into(),
            ));
        }
        y.fill(S::zero());
        let rp = a.row_ptr();
        let cj = a.col_idx();
        let vv = a.values();
        let omega = S::from_real(self.omega);
        for _ in 0..self.sweeps {
            for i in 0..self.n {
                let mut sigma = S::zero();
                for p in rp[i]..rp[i + 1] {
                    let j = cj[p];
                    if j < i {
                        sigma += vv[p] * y[j];
                    } else if j > i {
                        sigma += vv[p] * x[j];
                    }
                }
                let yi = (x[i] - sigma) * self.inv_diag_complex[i];
                y[i] = (S::one() - omega) * x[i] + omega * yi;
            }
            if self.mat_side.contains(MatSorType::SYMMETRIC_SWEEP) {
                for ii in (0..self.n).rev() {
                    let mut sigma = S::zero();
                    for p in rp[ii]..rp[ii + 1] {
                        let j = cj[p];
                        if j != ii {
                            sigma += vv[p] * y[j];
                        }
                    }
                    let yi = (x[ii] - sigma) * self.inv_diag_complex[ii];
                    y[ii] = (S::one() - omega) * x[ii] + omega * yi;
                }
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "complex"))]
impl ObjPreconditioner for SorPc {
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let csr = csr_from_linop(op, 0.0)?;
        self.a_csr = Some(csr.clone());
        self.ensure_inv_diag(&csr)?;
        self.ensure_coloring(&csr);
        Ok(())
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        // Re-extract CSR (values may have changed) and recompute inverse diagonal
        let csr = csr_from_linop(op, 0.0)?;
        self.a_csr = Some(csr.clone());
        self.ensure_inv_diag(&csr)?;
        self.ensure_coloring(&csr);
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        self.apply_real(side, x, y)
    }

    fn required_format(&self) -> crate::matrix::format::OpFormat {
        crate::matrix::format::OpFormat::Csr
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::LocalOnly
    }
}

#[cfg(feature = "complex")]
impl ObjPreconditioner for SorPc {
    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        let csr = if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<S>>() {
            Arc::new(csr.clone())
        } else {
            return Err(KError::Unsupported(
                "SOR complex setup currently requires a CSR operator".into(),
            ));
        };
        self.a_csr_complex = Some(csr.clone());
        self.ensure_inv_diag_complex(&csr)?;
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply_complex(side, x, y)
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::LocalOnly
    }
}

#[cfg(feature = "complex")]
impl KPreconditioner for SorPc {
    type Scalar = S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        bridge_apply_pc_mut_s(self, side, x, y, scratch)
    }

    fn on_restart_s(&mut self, outer_iter: usize, residual_norm: R) -> Result<(), KError> {
        ObjPreconditioner::on_restart(self, outer_iter, residual_norm)
    }
}

#[cfg(all(test, feature = "complex"))]
mod tests {
    use super::*;
    use crate::algebra::bridge::BridgeScratch;
    use crate::matrix::sparse::CsrMatrix;
    use crate::ops::kpc::KPreconditioner;
    use crate::preconditioner::PcSide;

    #[test]
    fn apply_s_runs_for_complex_rhs() {
        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![
                S::from_real(4.0),
                S::from_real(-1.0),
                S::from_real(-1.0),
                S::from_real(4.0),
            ],
        );
        let mut pc = SorPc::new(1.0, 1, MatSorType::APPLY_LOWER, 0.0);
        pc.setup(&a).unwrap();
        let rhs = vec![S::from_parts(1.0, 0.5); 2];
        let mut out = vec![S::zero(); rhs.len()];
        let mut scratch = BridgeScratch::default();
        pc.apply_s(PcSide::Left, &rhs, &mut out, &mut scratch)
            .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn complex_fallback_diagnostics_reports_forced_degradation() {
        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![
                S::from_parts(4.0, 0.0),
                S::from_parts(-1.0, 0.0),
                S::from_parts(-1.0, 0.0),
                S::from_parts(4.0, 0.0),
            ],
        );
        let mut pc = SorPc::new(1.0, 1, MatSorType::APPLY_LOWER, 0.0);
        pc.set_complex_force_split_fallback(true);
        pc.setup(&a).unwrap();
        let rhs = vec![S::from_parts(1.0, 0.25); 2];
        let mut out = vec![S::zero(); 2];
        pc.apply(PcSide::Left, &rhs, &mut out).unwrap();
        let diag = pc.complex_fallback_diagnostics();
        assert!(
            diag.as_deref()
                .unwrap_or("")
                .contains("degraded_split_real_imag")
        );
    }

    #[test]
    fn complex_fallback_mode_switches_to_split_real_imag() {
        let a = CsrMatrix::from_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![
                S::from_parts(4.0, 0.1),
                S::from_parts(-1.0, 0.0),
                S::from_parts(-1.0, 0.0),
                S::from_parts(4.0, -0.1),
            ],
        );
        let mut pc = SorPc::new(1.0, 1, MatSorType::APPLY_LOWER, 0.0);
        pc.set_complex_force_split_fallback(true);
        pc.setup(&a).unwrap();
        assert_eq!(
            pc.complex_kernel_mode(),
            SorComplexKernelMode::DegradedSplitRealImag
        );
        let rhs = vec![S::from_parts(1.0, 0.5); 2];
        let mut out = vec![S::zero(); rhs.len()];
        pc.apply(PcSide::Left, &rhs, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

#[cfg(all(test, not(feature = "complex")))]
mod tests_color_sweep {
    use super::*;
    use crate::matrix::op::LinOp;
    use crate::matrix::sparse::CsrMatrix;

    #[test]
    fn color_sweep_reduces_residual() {
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::from_csr(3, 3, row_ptr, col_idx, values);
        let mut pc = SorPc::new(
            1.0,
            1,
            MatSorType::APPLY_LOWER | MatSorType::COLOR_SWEEP,
            0.0,
        );
        pc.setup(&a).unwrap();

        let b = vec![1.0; 3];
        let mut y = vec![0.0; 3];
        pc.apply(PcSide::Left, &b, &mut y).unwrap();

        let mut ay = vec![0.0; 3];
        LinOp::matvec(&a, &y, &mut ay);
        let mut r_norm = 0.0;
        let mut b_norm = 0.0;
        for i in 0..3 {
            let ri = b[i] - ay[i];
            r_norm += ri * ri;
            b_norm += b[i] * b[i];
        }
        assert!(r_norm.sqrt() < b_norm.sqrt());
    }
}
