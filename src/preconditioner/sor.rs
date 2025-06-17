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
//! # Usage
//!
//! - Create a `Sor` preconditioner with the desired parameters (ω, sweeps, etc).
//! - Call `setup` with the system matrix to extract the diagonal and store its inverse.
//! - Use `apply` to apply the preconditioner to a vector.
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Section 10.2.
//! - https://en.wikipedia.org/wiki/Successive_over-relaxation

use std::marker::PhantomData;
use std::fmt;
use bitflags::bitflags;
use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, Indexing};
use crate::error::KError;
use num_traits::Float;

/// Bitflags for SOR sweep types and options.
///
/// Allows selection of forward, backward, symmetric, and Eisenstat sweeps.
bitflags! {
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
pub struct Sor<M, V, T> {
    pub its:      usize,      // Number of outer SOR iterations
    pub lits:     usize,      // Number of local iterations (unused)
    pub sym:      MatSorType, // Sweep type (forward, backward, symmetric)
    pub omega:    T,          // Relaxation parameter
    pub fshift:   T,          // Diagonal shift
    pub inv_diag: Vec<T>,     // Inverse diagonal entries
    pub a:        Option<M>,  // Matrix reference (after setup)
    _phantom:     PhantomData<V>,
}

impl<M, V, T> Sor<M, V, T>
where
    T: Float,
{
    /// Create a new SOR preconditioner with the given parameters.
    pub fn new(omega: T, its: usize, lits: usize, sym: MatSorType, fshift: T) -> Self {
        Self { its, lits, sym, omega, fshift, inv_diag: Vec::new(), a: None, _phantom: PhantomData }
    }
    // Setters and getters for parameters
    pub fn set_omega(&mut self, omega: T) { self.omega = omega; }
    pub fn omega(&self) -> T { self.omega }
    pub fn set_its(&mut self, its: usize) { self.its = its; }
    pub fn its(&self) -> usize { self.its }
    pub fn set_lits(&mut self, lits: usize) { self.lits = lits; }
    pub fn lits(&self) -> usize { self.lits }
    pub fn set_sym(&mut self, sym: MatSorType) { self.sym = sym; }
    pub fn sym(&self) -> MatSorType { self.sym }
    pub fn set_fshift(&mut self, fshift: T) { self.fshift = fshift; }
    pub fn fshift(&self) -> T { self.fshift }
}

impl<M, V, T> fmt::Display for Sor<M, V, T>
where
    T: Float + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SOR(omega={}, its={}, lits={}, sym={:?}, fshift={})",
            self.omega, self.its, self.lits, self.sym, self.fshift)
    }
}

impl<M, V, T> Preconditioner<M, V> for Sor<M, V, T>
where
    M: MatVec<V> + Indexing + Clone + std::ops::Index<(usize, usize), Output = T>,
    V: AsRef<[T]> + AsMut<[T]> + From<Vec<T>>,
    T: Float + Copy,
{
    /// Setup SOR: extract diagonal and store inverse.
    ///
    /// Stores a reference to the matrix and computes the inverse of the diagonal (with optional shift).
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        self.a = Some(a.clone());
        let n = a.nrows();
        self.inv_diag.resize(n, T::zero());
        for i in 0..n {
            let aii = a[(i, i)] + self.fshift;
            if aii == T::zero() {
                return Err(KError::ZeroPivot(i));
            }
            self.inv_diag[i] = T::one() / aii;
        }
        Ok(())
    }

    /// Apply SOR/SSOR preconditioner: y = M⁻¹ x.
    ///
    /// Performs the specified number of forward and/or backward sweeps, depending on the `sym` flags.
    /// Each sweep updates the solution vector in-place using the SOR formula.
    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError> {
        let a = self.a.as_ref().expect("SOR not setup");
        let n = x.as_ref().len();
        y.as_mut().fill(T::zero());
        for _ in 0..self.its {
            // FORWARD sweep (Gauss–Seidel or SOR)
            if self.sym.intersects(MatSorType::APPLY_LOWER) {
                for i in 0..n {
                    let mut sigma = T::zero();
                    // Lower triangle: sum a[i,j] * y[j] for j < i
                    for j in 0..i {
                        sigma = sigma + a[(i, j)] * y.as_ref()[j];
                    }
                    // Optionally include upper triangle (Eisenstat trick)
                    if !self.sym.contains(MatSorType::EISENSTAT) {
                        for j in (i+1)..n {
                            sigma = sigma + a[(i, j)] * x.as_ref()[j];
                        }
                    }
                    let xi = x.as_ref()[i];
                    let yi = (xi - sigma) * self.inv_diag[i];
                    y.as_mut()[i] = yi;
                }
            }
            // BACKWARD sweep (reverse Gauss–Seidel or SOR)
            if self.sym.intersects(MatSorType::APPLY_UPPER) {
                for ii in (0..n).rev() {
                    let mut sigma = T::zero();
                    // Upper triangle: sum a[ii,j] * y[j] for j > ii
                    for j in (ii+1)..n {
                        sigma = sigma + a[(ii, j)] * y.as_ref()[j];
                    }
                    // Optionally include lower triangle (Eisenstat trick)
                    if !self.sym.contains(MatSorType::EISENSTAT) {
                        for j in 0..ii {
                            sigma = sigma + a[(ii, j)] * y.as_ref()[j];
                        }
                    }
                    let xi = x.as_ref()[ii];
                    let yi = (xi - sigma) * self.inv_diag[ii];
                    // Weighted update: (1-omega)*xi + omega*yi
                    y.as_mut()[ii] = (T::one()-self.omega)*xi + self.omega*yi;
                }
            }
        }
        Ok(())
    }
}
