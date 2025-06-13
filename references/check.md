Review the below source code for the `kryst` package, which is intended to provide a PETSc-like interface for preconditioners and linear and direct system solvers (e.g. pick a PC + pick a Solver). The reference text from Saad, 2001 provides clear description of the underlying theory for iterative solvers and preconditioners. Audit the code base for completeness and adherence to the spirit of `kryst`, which is to provide a flexible interface for PCs and Solvers inclusive of those described in Saad 2001.

Here is `src/lib.rs`

```rust
//! kryst: PETSc-style PC/KSP interface over Faer

#[cfg(feature = "parallel")]
pub mod parallel;

pub mod config;
pub mod context;
pub mod core;
pub mod error;
pub mod matrix;
pub mod preconditioner;
pub mod solver;
pub mod utils;

// Re-exports for convenience
pub use config::*;
pub use context::*;
pub use core::*;
pub use error::*;
pub use matrix::*;
pub use preconditioner::*;
pub use solver::*;
pub use utils::*;
```

`src/error.rs`

```rust
use thiserror::Error;

// Unified error type for kryst

#[derive(Error, Debug)]
pub enum KError {
    #[error("factorization error: {0}")]
    FactorError(String),
    #[error("solve error: {0}")]
    SolveError(String),
}
```

`src/config/mod.rs`

```rust
pub mod options;
pub use options::PcOptions;
```

`src/config/options.rs`

```rust
//! Command-line or API options for preconditioners.
//!
//! This module provides the `PcOptions` struct, which is used to specify
//! options for various types of preconditioners via command-line arguments
//! or API calls. The available preconditioners include Jacobi, SSOR, and
//! ILU(0). Additionally, parameters such as the relaxation factor for SSOR
//! and the drop tolerance for ILU(p) can be set.

/// Preconditioner types & parameters.
#[derive(Debug)]
pub struct PcOptions {
    /// Type of preconditioner (none, jacobi, ssor, ilu0)
    pub pc_type: String,

    /// Relaxation factor ω for SSOR
    pub omega: f64,

    /// Drop tolerance for ILU(p)
    pub drop_tol: f64,
}
```

`src/context/mod.rs`

```rust
pub mod ksp_context;
pub use ksp_context::{KspContext, KspType};
pub mod pc_context;
```

`src/context/ksp_context.rs`

```rust
//! Factory for Krylov methods (KSP).

use crate::error::KError;
use crate::solver::{CgSolver, GmresSolver, LinearSolver};

pub enum KspType {
    Cg { tol: f64, max_iters: usize },
    Gmres { restart: usize, tol: f64, max_iters: usize },
}

pub struct KspContext {
    pub kind: KspType,
}

impl KspContext {
    pub fn new(kind: KspType) -> Self {
        Self { kind }
    }

    /// Build an instance of LinearSolver<M,V>.
    pub fn build<M, V, T>(&self) -> Box<dyn LinearSolver<M, V, Error = KError>>
    where
        M: 'static + crate::core::traits::MatVec<V>,
        (): crate::core::traits::InnerProduct<V, Scalar = T>,
        V: 'static + AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
        T: 'static + num_traits::Float + Clone + From<f64>,
    {
        match self.kind {
            KspType::Cg { tol, max_iters } => Box::new(CgSolver::new(tol.into(), max_iters)),
            KspType::Gmres { restart, tol, max_iters } => {
                Box::new(GmresSolver::new(restart, tol.into(), max_iters))
            }
        }
    }
}
```

`src/context/pc_context.rs`

```rust
// PCContext struct and logic
```

`src/core/mod.rs`

```rust
pub mod traits;
pub mod wrappers;
```

`src/core/traits.rs`

```rust
//! Core linear-algebra traits for kryst.

/// Matrix–vector product: y ← A x.
pub trait MatVec<V> {
    /// Compute y = A · x.
    fn matvec(&self, x: &V, y: &mut V);
}

/// Inner products & norms.
pub trait InnerProduct<V> {
    /// Associated scalar type.
    type Scalar: Copy + PartialOrd + From<f64>;
    /// Compute dot(x, y).
    fn dot(&self, x: &V, y: &V) -> Self::Scalar;
    /// Compute ‖x‖₂.
    fn norm(&self, x: &V) -> Self::Scalar;
}

/// Uniform indexing into vectors (dense or sparse).
pub trait Indexing {
    /// Number of rows (or length for a vector).
    fn nrows(&self) -> usize;
}
```

`src/core/wrappers.rs`

```rust
// Wrappers for faer::Mat, faer::MatRef, sparse types

use crate::core::traits::{Indexing, InnerProduct, MatVec};
use faer::{Mat, MatRef};
use num_traits::Float;

impl<T: Float> MatVec<Vec<T>> for Mat<T> {
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.nrows(), y.len());
        assert_eq!(self.ncols(), x.len());
        for i in 0..self.nrows() {
            y[i] = T::zero();
            for j in 0..self.ncols() {
                y[i] = y[i] + self[(i, j)] * x[j];
            }
        }
    }
}

impl<'a, T: Float> MatVec<Vec<T>> for MatRef<'a, T> {
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(self.nrows(), y.len());
        assert_eq!(self.ncols(), x.len());
        for i in 0..self.nrows() {
            y[i] = T::zero();
            for j in 0..self.ncols() {
                y[i] = y[i] + self[(i, j)] * x[j];
            }
        }
    }
}

impl<T: Float + From<f64>> InnerProduct<Vec<T>> for () {
    type Scalar = T;
    fn dot(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| *xi * *yi)
            .fold(T::zero(), |acc, v| acc + v)
    }
    fn norm(&self, x: &Vec<T>) -> T {
        x.iter()
            .map(|xi| *xi * *xi)
            .fold(T::zero(), |acc, v| acc + v)
            .sqrt()
    }
}

impl<T> Indexing for Vec<T> {
    fn nrows(&self) -> usize {
        self.len()
    }
}

impl<T> Indexing for Mat<T> {
    fn nrows(&self) -> usize {
        self.nrows()
    }
}
```

`src/matrix/mod.rs`

```rust
pub mod dense;
pub use dense::DenseMatrix;
pub mod sparse;
```

`src/matrix/dense.rs`

```rust
//! Dense‐matrix API on top of Faer.
//!
//! This module provides the `DenseMatrix` trait and its implementation for the `faer::Mat<T>` type,
//! enabling construction from raw column-major storage.

use crate::core::traits::{Indexing, MatVec};
use faer::Mat;

/// Blanket impl so any Faer Mat<T> is a DenseMatrix.
pub trait DenseMatrix<T>: MatVec<Vec<T>> + Indexing {
    /// Construct from raw column-major storage.
    fn from_raw(nrows: usize, ncols: usize, data: Vec<T>) -> Self;
}

impl<T: Copy + num_traits::Float> DenseMatrix<T> for Mat<T> {
    fn from_raw(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        Mat::from_fn(nrows, ncols, |i, j| data[j * nrows + i])
    }
}
```

`src/matrix/sparse.rs`

```rust
// SparseMatrix trait and implementations (CSR, CSC)

/// A read‐only sparse matrix supporting y = A * x.
pub trait SparseMatrix<T> {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
    /// Compute y = A * x.  `x.len() == ncols()`, `y.len() == nrows()`.
    fn spmv(&self, x: &[T], y: &mut [T]);
}

use faer::sparse::{
    SymbolicSparseRowMat,    // owning symbolic CSR alias
    SparseRowMat,            // owning numeric CSR alias
    CreationError,           // error type for builders
};
use faer::traits::ComplexField;
use faer::sparse::linalg::matmul::sparse_dense_matmul;

pub struct CsrMatrix<T> {
    inner: SparseRowMat<usize, T>,
}

impl<T: ComplexField + Copy> CsrMatrix<T> {
    /// Build a CSR from raw row‐ptr, col‐idx, and values.
    pub fn from_csr(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        // Build symbolic structure; second argument `None` means “no separate row_nnz”:
        let symbolic = SymbolicSparseRowMat::new_checked(
            nrows,
            ncols,
            row_ptr,
            None,      // optional row_nnz: Option<Vec<usize>>
            col_idx,
        );
        // Attach the numerical values:
        let inner = SparseRowMat::new(symbolic, values);
        Self { inner }
    }
}

impl<T: ComplexField + Copy + num_traits::One> SparseMatrix<T> for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        let x_mat = faer::Mat::<T>::from_fn(self.ncols(), 1, |i, _| x[i]);
        let mut y_mat = faer::Mat::<T>::zeros(self.nrows(), 1);
        // Fallback: convert to dense and multiply (since sparse_dense_matmul expects CSC)
        let dense = self.inner.to_dense();
        y_mat.copy_from(&dense * &x_mat);
        for i in 0..y.len() {
            y[i] = y_mat[(i, 0)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_spmv() {
        // 3×3 identity in CSR: row_ptr=[0,1,2,3], col_idx=[0,1,2], vals=[1,1,1]
        let m = CsrMatrix::from_csr(3, 3, vec![0,1,2,3], vec![0,1,2], vec![1.0,1.0,1.0]);
        let x = vec![2.0, 3.0, 5.0];
        let mut y = vec![0.0; 3];
        m.spmv(&x, &mut y);
        assert_eq!(y, x);
    }

    #[test]
    fn simple_pattern() {
        // 2×3 matrix [[1,2,0],[0,3,4]]
        let m = CsrMatrix::from_csr(
            2, 3,
            vec![0,2,4],
            vec![0,1,1,2],
            vec![1.0,2.0,3.0,4.0],
        );
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];
        m.spmv(&x, &mut y);
        assert_eq!(y, vec![3.0, 7.0]);
    }
}
```

`src/parallel/mod.rs`

```rust
pub mod mpi_comm;
pub mod rayon_comm;
```

`src/parallel/mpi_comm.rs`

```rust
// MPI-based parallel communication (mpi)
```

`src/parallel/rayon_comm.rs`

```rust
// rayon-based parallel communication
```

`src/preconditioner/mod.rs`

```rust
//! Preconditioners for linear solvers.

use crate::error::KError;

/// A preconditioner M ≈ A⁻¹.
/// - `setup` is called once to factor or build M.
/// - `apply` applies M to a vector: y ← M x.
pub trait Preconditioner<M, V>: Send + Sync {
    /// Factor or initialize the preconditioner from A.
    fn setup(&mut self, a: &M) -> Result<(), KError>;

    /// Apply preconditioner: y ← M x.
    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError>;
}

pub mod block_jacobi;
pub mod ilu;
pub mod jacobi;
pub mod ssor;
pub use jacobi::Jacobi;
pub use ssor::Ssor;
pub use ilu::Ilu0;
```

`src/preconditioner/block_jacobi.rs`

```rust
// Block-Jacobi preconditioner implementation
```

`src/preconditioner/ilu.rs`

```rust
//! ILU(0) factorization with zero fill (Saad §10.3).

use crate::preconditioner::Preconditioner;
use crate::error::KError;
use num_traits::Float;
use faer::traits::ComplexField;
use faer::Mat;

pub struct Ilu0<T> {
    pub(crate) l: Mat<T>,
    pub(crate) u: Mat<T>,
}

impl<T: Float + Send + Sync + ComplexField> Ilu0<T> {
    pub fn new() -> Self {
        Self {
            l: Mat::zeros(0, 0),
            u: Mat::zeros(0, 0),
        }
    }
}

impl<T: Float + Send + Sync + ComplexField> Preconditioner<Mat<T>, Vec<T>> for Ilu0<T> {
    fn setup(&mut self, a: &Mat<T>) -> Result<(), KError> {
        let (n, _) = (a.nrows(), a.ncols());
        let mut l = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);

        for i in 0..n {
            u[(i, i)] = a[(i, i)];
            for j in (i+1)..n {
                if a[(i, j)] != T::zero() {
                    u[(i, j)] = a[(i, j)];
                }
            }
            l[(i, i)] = T::one();
            for j in (i+1)..n {
                if a[(j, i)] != T::zero() {
                    l[(j, i)] = a[(j, i)] / u[(i, i)];
                }
            }
            for j in (i+1)..n {
                for k in (i+1)..n {
                    if a[(j, k)] != T::zero() {
                        let v = a[(j, k)] - l[(j, i)] * u[(i, k)];
                        if v != T::zero() {
                            if k >= j {
                                u[(j, k)] = v;
                            } else {
                                l[(j, k)] = v;
                            }
                        }
                    }
                }
            }
        }
        self.l = l;
        self.u = u;
        Ok(())
    }

    fn apply(&self, x: &Vec<T>, y: &mut Vec<T>) -> Result<(), KError> {
        // solve L y1 = x
        let mut y1 = x.clone();
        let n = x.len();
        for i in 0..n {
            for j in 0..i {
                y1[i] = y1[i] - self.l[(i, j)] * y1[j];
            }
        }
        // solve U y = y1
        for i in (0..n).rev() {
            for j in (i+1)..n {
                y1[i] = y1[i] - self.u[(i, j)] * y1[j];
            }
            y1[i] = y1[i] / self.u[(i, i)];
        }
        *y = y1;
        Ok(())
    }
}
```

`src/preconditioner/jacobi.rs`

```rust
// Jacobi preconditioner implementation

use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, Indexing};
use crate::error::KError;
use num_traits::Float;

/// Jacobi preconditioner: M⁻¹ = D⁻¹
pub struct Jacobi<T> {
    pub(crate) inv_diag: Vec<T>,
}

impl<T: Float> Jacobi<T> {
    /// new with empty state; user must call `setup`.
    pub fn new() -> Self {
        Self { inv_diag: Vec::new() }
    }
}

impl<M, V, T> Preconditioner<M, V> for Jacobi<T>
where
    M: MatVec<V> + Indexing,
    V: AsRef<[T]> + AsMut<[T]> + From<Vec<T>>,
    T: Float + Send + Sync,
{
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        let n = a.nrows();
        let mut diag = vec![T::zero(); n];
        let mut e = vec![T::zero(); n];
        let mut col = vec![T::zero(); n];
        for i in 0..n {
            e.iter_mut().for_each(|x| *x = T::zero());
            e[i] = T::one();
            let e_v = V::from(e.clone());
            let mut col_v = V::from(col.clone());
            a.matvec(&e_v, &mut col_v);
            col = col_v.as_ref().to_vec();
            diag[i] = col[i];
        }
        self.inv_diag = diag.into_iter()
            .map(|d| if d != T::zero() { T::one() / d } else { T::zero() })
            .collect();
        Ok(())
    }

    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError> {
        let x_ref = x.as_ref();
        let y_mut = y.as_mut();
        for i in 0..x_ref.len() {
            y_mut[i] = self.inv_diag[i] * x_ref[i];
        }
        Ok(())
    }
}
```

`src/preconditioner/ssor.rs`

```rust
use crate::preconditioner::Preconditioner;
use crate::core::traits::{MatVec, Indexing};
use crate::error::KError;
use num_traits::Float;

/// Symmetric Successive Over-Relaxation.
/// M = (D/ω + L) D⁻¹ (D/ω + U)
pub struct Ssor<T> {
    omega: T,
    inv_diag: Vec<T>,
}

impl<T: Float> Ssor<T> {
    pub fn new(omega: T) -> Self {
        Self { omega, inv_diag: Vec::new() }
    }
}

impl<M, V, T> Preconditioner<M, V> for Ssor<T>
where
    M: MatVec<V> + Indexing,
    V: AsRef<[T]> + AsMut<[T]> + From<Vec<T>>,
    T: num_traits::Float + Send + Sync + std::ops::Mul<Output = T> + Copy,
{
    fn setup(&mut self, a: &M) -> Result<(), KError> {
        // reuse Jacobi logic for inv_diag
        let mut jac = crate::preconditioner::Jacobi::<T>::new();
        jac.setup(a)?;
        self.inv_diag = jac.inv_diag;
        Ok(())
    }

    fn apply(&self, x: &V, y: &mut V) -> Result<(), KError> {
        let n = x.as_ref().len();
        let mut tmp = vec![T::zero(); n];
        // forward Gauss-Seidel: (D/ω + L)⁻¹ x
        for i in 0..n {
            let sum = x.as_ref()[i];
            // subtract lower part: a_ij * tmp[j]
            // we approximate a_ij via matvec of basis vectors (expensive, but OK for demo)
            // TODO: optimize by storing L
            tmp[i] = sum * self.inv_diag[i] * self.omega;
        }
        // backward: (D/ω + U)⁻¹ tmp → y
        for i in (0..n).rev() {
            y.as_mut()[i] = tmp[i] * self.inv_diag[i] * self.omega;
        }
        Ok(())
    }
}
```

`src/solver/mod.rs`

```rust
//! Krylov & direct solver interfaces.

/// Common interface for any direct or iterative solver.
pub trait LinearSolver<M, V> {
    type Error;
    /// Solve A·x = b, writing result into `x`.
    fn solve(
        &mut self,
        a: &M,
        b: &V,
        x: &mut V
    ) -> Result<(), Self::Error>;
}

pub mod direct_lu;
pub use direct_lu::{LuSolver, QrSolver};

pub mod cg;
pub use cg::CgSolver;

pub mod gmres;
pub use gmres::GmresSolver;
```

`src/solver/bicgstab.rs`

```rust
// BiCGStab solver implementation
```

`src/solver/cg.rs`

```rust
//! Conjugate Gradient (unpreconditioned) per Saad §6.1.

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;

pub struct CgSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: Copy + num_traits::Float> CgSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for CgSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone,
{
    type Error = KError;

    fn solve(&mut self, a: &M, b: &V, x: &mut V) -> Result<(), KError> {
        let n = b.as_ref().len();
        let mut x_vec = x.as_ref().to_vec();
        let ip = ();
        let mut r = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(x_vec.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let mut p = r.clone();
        let mut rsq = ip.dot(&r, &r);
        let res0 = rsq.sqrt();
        let _stats = SolveStats { iterations: 0, final_residual: res0, converged: false };

        for i in 1..=self.conv.max_iters {
            let mut ap = V::from(vec![T::zero(); n]);
            a.matvec(&p.clone(), &mut ap);
            let alpha = rsq / ip.dot(&p, &ap);
            for (xj, pj) in x_vec.iter_mut().zip(p.as_ref()) {
                *xj = *xj + alpha * *pj;
            }
            for (rj, apj) in r.as_mut().iter_mut().zip(ap.as_ref()) {
                *rj = *rj - alpha * *apj;
            }
            let rsq_new = ip.dot(&r, &r);
            let res_norm = rsq_new.sqrt();
            let (stop, s) = self.conv.check(res_norm, res0, i);
            if stop && s.converged {
                *x = V::from(x_vec.clone());
                return Ok(());
            }
            let beta = rsq_new / rsq;
            let p_old = p.clone();
            for ((pj, rj), old_pj) in p.as_mut().iter_mut().zip(r.as_ref()).zip(p_old.as_ref()) {
                *pj = *rj + beta * *old_pj;
            }
            rsq = rsq_new;
        }
        *x = V::from(x_vec);
        Ok(())
    }
}
```

`src/solver/direct_lu.rs`

```rust
//! Direct dense solves via Faer.

use crate::error::KError;
use crate::solver::LinearSolver;
use faer::linalg::solvers::{FullPivLu, Qr, SolveCore};
use faer::{Mat, MatMut, Conj};
use faer::traits::{ComplexField, RealField};

// LU solver
pub struct LuSolver<T> {
    factor: Option<FullPivLu<T>>,
}

impl<T: ComplexField + RealField> LuSolver<T> {
    pub fn new() -> Self {
        LuSolver { factor: None }
    }
}

impl<T: ComplexField + RealField> LinearSolver<Mat<T>, Vec<T>> for LuSolver<T> {
    type Error = KError;

    fn solve(&mut self, a: &Mat<T>, b: &Vec<T>, x: &mut Vec<T>) -> Result<(), KError> {
        // Factorize if needed
        let factor = FullPivLu::new(a.as_ref());
        self.factor = Some(factor);
        // Copy b into x
        x.clone_from(b);
        // Avoid double borrow: get len first
        let n = x.len();
        let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
        self.factor
            .as_ref()
            .unwrap()
            .solve_in_place_with_conj(Conj::No, x_mat);
        Ok(())
    }
}

// QR solver
pub struct QrSolver;

impl QrSolver {
    pub fn new() -> Self {
        QrSolver
    }
}

impl<T: ComplexField + RealField> LinearSolver<Mat<T>, Vec<T>> for QrSolver {
    type Error = KError;

    fn solve(&mut self, a: &Mat<T>, b: &Vec<T>, x: &mut Vec<T>) -> Result<(), KError> {
        let factor = Qr::new(a.as_ref());
        x.clone_from(b);
        let n = x.len();
        let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
        factor.solve_in_place_with_conj(Conj::No, x_mat);
        Ok(())
    }
}
```

`src/solver/gmres.rs`

```rust
//! GMRES with fixed restart per Saad §6.4.

use crate::core::traits::{InnerProduct, MatVec};
use crate::solver::LinearSolver;
use crate::utils::convergence::{Convergence, SolveStats};
use crate::error::KError;
use num_traits::Float;

pub struct GmresSolver<T> {
    pub restart: usize,
    pub conv: Convergence<T>,
}

impl<T: Copy + Float> GmresSolver<T> {
    pub fn new(restart: usize, tol: T, max_iters: usize) -> Self {
        Self {
            restart,
            conv: Convergence { tol, max_iters },
        }
    }

    // --- Arnoldi process with double orthogonalization and happy breakdown ---
    fn arnoldi<M, V>(
        a: &M,
        ip: &(),
        v_basis: &mut Vec<V>,
        h: &mut [Vec<T>],
        j: usize,
        epsilon: T,
    ) -> bool
    where
        M: MatVec<V>,
        (): InnerProduct<V, Scalar = T>,
        V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
        T: num_traits::Float + Clone,
    {
        let n = v_basis[0].as_ref().len();
        let mut w = V::from(vec![T::zero(); n]);
        a.matvec(&v_basis[j].clone(), &mut w);
        // Modified Gram-Schmidt
        for i in 0..=j {
            h[i][j] = ip.dot(&w, &v_basis[i]);
            for (wk, vik) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                *wk = *wk - h[i][j] * *vik;
            }
        }
        // Iterative refinement (second orthogonalization)
        for i in 0..=j {
            let tmp = ip.dot(&w, &v_basis[i]);
            h[i][j] = h[i][j] + tmp;
            for (wk, vik) in w.as_mut().iter_mut().zip(v_basis[i].as_ref()) {
                *wk = *wk - tmp * *vik;
            }
        }
        h[j + 1][j] = ip.norm(&w);
        // Happy breakdown: if norm is very small, return true
        if h[j + 1][j].abs() < epsilon {
            return true;
        }
        let vj1 = V::from(w.as_ref().iter().map(|&wi| wi / h[j + 1][j]).collect::<Vec<_>>());
        v_basis.push(vj1);
        false
    }

    // --- Apply Givens rotation and update g together ---
    fn apply_givens_and_update_g(h: &mut [Vec<T>], g: &mut [T], cs: &mut [T], sn: &mut [T], j: usize, epsilon: T) {
        for i in 0..j {
            let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
            h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
            h[i][j] = temp;
        }
        let h_kk = h[j][j];
        let h_k1k = h[j + 1][j];
        let r = (h_kk * h_kk + h_k1k * h_k1k).sqrt();
        if r.abs() < epsilon {
            cs[j] = T::one();
            sn[j] = T::zero();
        } else {
            cs[j] = h_kk / r;
            sn[j] = h_k1k / r;
        }
        h[j][j] = cs[j] * h_kk + sn[j] * h_k1k;
        h[j + 1][j] = T::zero();
        // Update g
        let temp = cs[j] * g[j] + sn[j] * g[j + 1];
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
        g[j] = temp;
    }

    // --- Back-substitution for least squares with zero-pivot protection ---
    fn back_substitution(h: &[Vec<T>], g: &[T], y: &mut [T], m: usize, epsilon: T) {
        for i in (0..m).rev() {
            y[i] = g[i];
            for j in (i + 1)..m {
                y[i] = y[i] - h[i][j] * y[j];
            }
            if h[i][i].abs() > epsilon {
                y[i] = y[i] / h[i][i];
            } else {
                y[i] = T::zero();
            }
        }
    }
}

impl<M, V, T> LinearSolver<M, V> for GmresSolver<T>
where
    M: MatVec<V>,
    (): InnerProduct<V, Scalar = T>,
    V: AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
    T: num_traits::Float + Clone,
{
    type Error = KError;

    fn solve(&mut self, a: &M, b: &V, x: &mut V) -> Result<(), KError> {
        let n = b.as_ref().len();
        let ip = ();
        let mut xk = x.as_ref().to_vec();
        let mut r0 = {
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            V::from(r_vec)
        };
        let mut beta = ip.norm(&r0);
        let res0 = beta;
        let mut stats = SolveStats { iterations: 0, final_residual: beta, converged: false };

        let max_outer = (self.conv.max_iters + self.restart - 1) / self.restart;
        let mut iteration = 0;
        let epsilon = T::from(1e-14).unwrap();
        for _ in 0..max_outer {
            let mut v_basis: Vec<V> = Vec::with_capacity(self.restart + 1);
            // Normalize r0 for the first basis vector
            let r0_norm = beta;
            let v0 = r0.clone().as_ref().iter().map(|&ri| ri / r0_norm).collect::<Vec<_>>();
            v_basis.push(V::from(v0));
            let mut h = vec![vec![T::zero(); self.restart]; self.restart + 1];
            let mut g = vec![T::zero(); self.restart + 1];
            g[0] = beta;
            let mut cs = vec![T::zero(); self.restart];
            let mut sn = vec![T::zero(); self.restart];
            let mut m = 0;
            let mut happy_breakdown = false;
            for j in 0..self.restart {
                iteration += 1;
                happy_breakdown = Self::arnoldi(a, &ip, &mut v_basis, &mut h, j, epsilon);
                Self::apply_givens_and_update_g(&mut h, &mut g, &mut cs, &mut sn, j, epsilon);
                let res_norm = g[j + 1].abs();
                let (stop, s) = self.conv.check(res_norm, res0, iteration);
                stats = s.clone();
                m = j + 1;
                if stop && s.converged || happy_breakdown {
                    break;
                }
            }
            // Solve least squares Hy = g (only use m x m upper part)
            let mut y = vec![T::zero(); m];
            let h_upper: Vec<Vec<T>> = h.iter().take(m).map(|row| row[..m].to_vec()).collect();
            let g_upper = &g[..m];
            Self::back_substitution(&h_upper, g_upper, &mut y, m, epsilon);
            // Update xk = xk + sum_{j=0}^{m-1} y[j] * v_basis[j]
            for j in 0..m {
                for (xk_i, vj_i) in xk.iter_mut().zip(v_basis[j].as_ref()) {
                    *xk_i = *xk_i + y[j] * *vj_i;
                }
            }
            // Recompute true residual after restart
            let mut tmp = V::from(vec![T::zero(); n]);
            a.matvec(&V::from(xk.clone()), &mut tmp);
            let r_vec = tmp.as_ref().iter().zip(b.as_ref()).map(|(&ax, &bi)| bi - ax).collect::<Vec<_>>();
            r0 = V::from(r_vec);
            beta = ip.norm(&r0);
            if stats.converged || iteration >= self.conv.max_iters {
                break;
            }
        }
        *x = V::from(xk);
        Ok(())
    }
}
```

`src/utils/mod.rs`

```rust
// Logging, convergence checks, norms

pub mod convergence;
```

`src/utils/convergence.rs`

```rust
//! Convergence tracking & tolerance checks for iterative solvers.

/// Stopping criteria & stats.
pub struct Convergence<T> {
    pub tol: T,
    pub max_iters: usize,
}

#[derive(Clone)]
pub struct SolveStats<T> {
    pub iterations: usize,
    pub final_residual: T,
    pub converged: bool,
}

impl<T: Copy + num_traits::Float> Convergence<T> {
    /// Returns (should_stop, stats) given current `res_norm` and iteration `i`.
    pub fn check(
        &self,
        res_norm: T,
        res0_norm: T,
        i: usize,
    ) -> (bool, SolveStats<T>) {
        let rel = res_norm / res0_norm;
        let converged = rel <= self.tol || i >= self.max_iters;
        (
            converged,
            SolveStats {
                iterations: i,
                final_residual: res_norm,
                converged,
            },
        )
    }
}
```

`examples/dense_direct.rs`

```rust
use kryst::solver::{LuSolver, QrSolver, LinearSolver};
use faer::Mat;
use rand::Rng;

fn main() {
    let n = 10;
    // build a random SPD matrix: A = MᵀM + I
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n*n).map(|_| rng.gen()).collect();
    let m = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let m_t = m.transpose();
    // a = m^T * m
    let mut a = &m_t * &m;
    // a = a + I
    for i in 0..n { a[(i,i)] = a[(i,i)] + 1.0; }

    // rhs
    let b: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    let mut x = vec![0.0; n];

    // LU solve
    let mut lus = LuSolver::new();
    lus.solve(&a, &b, &mut x).unwrap();
    println!("LU x = {:?}", x);

    // QR solve
    let mut qrs = QrSolver::new();
    qrs.solve(&a, &b, &mut x).unwrap();
    println!("QR x = {:?}", x);
}
```

`tests/core_dense.rs`

```rust
use approx::assert_abs_diff_eq;
use faer::Mat;
use kryst::core::traits::{InnerProduct, MatVec};
use rand::Rng;

#[test]
fn matvec_random_small() {
    let n = 5;
    let mut rng = rand::thread_rng();
    let vals: Vec<f64> = (0..n * n).map(|_| rng.gen()).collect();
    // Use from_fn to build a column-major matrix
    let a = Mat::from_fn(n, n, |i, j| vals[j * n + i]);
    let x: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    let mut y = vec![0.0; n];
    a.matvec(&x, &mut y);

    // check y[i] == sum_j A[i,j]*x[j]
    for i in 0..n {
        let expected = (0..n).map(|j| vals[j * n + i] * x[j]).sum::<f64>();
        assert_abs_diff_eq!(y[i], expected, epsilon = 1e-12);
    }
}

#[test]
fn dot_and_norm() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, -5.0, 6.0];
    let ip = ();
    let dot = ip.dot(&x, &y);
    assert_abs_diff_eq!(dot, 1.0 * 4.0 + 2.0 * (-5.0) + 3.0 * 6.0, epsilon = 1e-12);
    let norm_x = ip.norm(&x);
    let expected_norm = ((1.0f64).powi(2) + 2.0f64.powi(2) + 3.0f64.powi(2)).sqrt();
    assert_abs_diff_eq!(norm_x, expected_norm, epsilon = 1e-12);
}
```

`tests/preconditioner_integration.rs`

```rust
use kryst::preconditioner::Preconditioner;
use kryst::preconditioner::{Jacobi, Ilu0};
use kryst::solver::{CgSolver, GmresSolver, LinearSolver};
use faer::Mat;

/// Build a badly conditioned diagonal matrix
fn ill_cond(n: usize, kappa: f64) -> (Mat<f64>, Vec<f64>) {
    let mut diag = vec![1.0; n];
    diag[n-1] = kappa;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = diag[i];
    }
    let b = vec![1.0; n];
    (a, b)
}

#[test]
fn cg_with_jacobi() {
    let (a, b) = ill_cond(5, 1e6);
    let mut pc = Jacobi::new();
    <Jacobi<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::setup(&mut pc, &a).unwrap();
    let mut solver = CgSolver::new(1e-6, 1000);
    let mut x = vec![0.0; 5];
    // wrap A and pc into a PCG solver if you have one, else manually:
    let r_in = b.clone();
    let mut r_out = vec![0.0; b.len()];
    <Jacobi<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::apply(&pc, &r_in, &mut r_out).unwrap();
    solver.solve(&a, &b, &mut x).unwrap();
    // No stats to check; just ensure it runs without panic
}

#[test]
fn gmres_with_ilu0() {
    let (a, b) = ill_cond(5, 1e4);
    let mut pc = Ilu0::new();
    <Ilu0<f64> as Preconditioner<Mat<f64>, Vec<f64>>>::setup(&mut pc, &a).unwrap();
    let mut solver = GmresSolver::new(4, 1e-6, 1000);
    let mut x = vec![0.0; 5];
    solver.solve(&a, &b, &mut x).unwrap();
    // No stats to check; just ensure it runs without panic
}
```

`tests/solver_iterative.rs`

```rust
use faer::Mat;
use faer::linalg::solvers::SolveCore;
use kryst::solver::{CgSolver, GmresSolver, LinearSolver};
use rand::Rng;
use approx::assert_abs_diff_eq;

// Helper: small SPD matrix A = Mᵀ M + I
fn random_spd(n: usize) -> (faer::Mat<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n*n).map(|_| rng.gen()).collect();
    let m = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let m_t = m.transpose();
    let a = &m_t * &m + Mat::<f64>::identity(n, n);
    let b: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    (a, b)
}

#[test]
fn cg_vs_direct_on_spd() {
    let n = 10;
    let (a, b) = random_spd(n);
    let mut x_cg = vec![0.0; n];
    let mut solver = CgSolver::new(1e-8, 1000);
    assert!(solver.solve(&a, &b, &mut x_cg).is_ok());
    // direct solve
    let mut x_direct = b.clone();
    let lus = faer::linalg::solvers::FullPivLu::new(a.as_ref());
    let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_direct, n, 1);
    lus.solve_in_place_with_conj(faer::Conj::No, x_mat);
    for i in 0..n {
        assert_abs_diff_eq!(x_cg[i], x_direct[i], epsilon = 1e-6);
    }
}

#[test]
fn gmres_vs_direct_on_nonsymmetric() {
    let n = 10;
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n*n).map(|_| rng.gen()).collect();
    let a = Mat::from_fn(n, n, |i, j| data[j * n + i]);
    let b: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    let mut x_gmres = vec![0.0; n];
    let mut solver = GmresSolver::new(100, 1e-8, 1000);
    assert!(solver.solve(&a, &b, &mut x_gmres).is_ok());
    // direct solve
    let mut x_direct = b.clone();
    let qr = faer::linalg::solvers::Qr::new(a.as_ref());
    let x_mat = faer::MatMut::from_column_major_slice_mut(&mut x_direct, n, 1);
    qr.solve_in_place_with_conj(faer::Conj::No, x_mat);
    for i in 0..n {
        assert_abs_diff_eq!(x_gmres[i], x_direct[i], epsilon = 1e-6);
    }
}
```