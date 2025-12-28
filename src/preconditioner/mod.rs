//! # Preconditioners
//!
//! Most users select a [`PcType`](crate::context::pc_context::PcType) through
//! [`KspContext`](crate::context::KspContext). Implement the [`Preconditioner`]
//! trait only when you need a custom operator.
//!
//! ## Contract
//! - [`Preconditioner::apply`] must compute **`y = M^{-1} x`** regardless of [`PcSide`].
//!   Side is forwarded so PCs *with internal sweep order* (e.g. SOR/SSOR) can choose
//!   the appropriate triangular traversal. Most PCs ignore side.
//! - [`Preconditioner::apply_mut`] is used by flexible solvers (FGMRES) to allow
//!   iteration-varying behavior. Default forwards to [`apply`].
//! - [`Preconditioner::on_restart`] is an optional hook invoked at solver restarts
//!   (e.g., by FGMRES) to allow adaptive tuning. Default is a no-op.
//! - Direct methods may implement [`direct_solve`] and will be used by `PREONLY`.
//!
//! ## Reuse semantics
//! Callers may invoke [`update_numeric`] when structure is unchanged; otherwise
//! use [`update_symbolic`]. Report truthfully via [`supports_numeric_update`].
//!
//! ## Parallelism
//! Preconditioners obtain the communicator via [`LinOp::comm()`] from the operator
//! provided to [`setup`]. **Do not** thread communicators manually.
//!
//! ## Side semantics (solver-enforced)
//! Solvers place `M^{-1}`:
//! - Left: build on `M^{-1} A` and monitor `||M^{-1} r||`
//! - Right: build on `A M^{-1}` and monitor `||r||`
//!
//! ## Examples
//! ```no_run
//! # #[cfg(feature = "backend-faer")] {
//! # use kryst::context::ksp_context::{KspContext, SolverType};
//! # use kryst::context::pc_context::PcType;
//! # use kryst::matrix::op::DenseOp;
//! # use faer::Mat;
//! # use std::sync::Arc;
//! let a = Mat::<f64>::from_fn(100,100, |i,j| if i==j {4.0} else if (i as isize-j as isize).abs()==1 {-1.0} else {0.0});
//! let a = Arc::new(DenseOp::<f64>::new(Arc::new(a)));
//! # use kryst::algebra::prelude::{KrystScalar, S};
//! let b = vec![S::from_real(1.0); 100];
//! let mut x = vec![S::zero(); 100];
//! let mut ksp = KspContext::new();
//! ksp.set_type(SolverType::Gmres).unwrap()
//!    .set_pc_type(PcType::Jacobi, None).unwrap()
//!    .set_operators(a, None);
//! let _stats = ksp.solve(&b, &mut x).unwrap();
//! # }
//! ```

#[cfg(all(feature = "legacy-pc-bridge", feature = "backend-faer"))]
use crate::algebra::prelude::*;
use crate::algebra::scalar::{KrystScalar, R, S};
use crate::error::KError;
pub use crate::matrix::format::OpFormat;
use crate::matrix::op::LinOp;

pub mod bridge;
pub mod pc_bridge;
#[cfg(all(feature = "legacy-pc-bridge", feature = "backend-faer"))]
use faer::Mat;
use std::str::FromStr;

/// Which side to apply M⁻¹ on in preconditioning.
///
/// For the linear system Ax = b with preconditioner M ≈ A:
/// - Left: Solve M⁻¹Ax = M⁻¹b
/// - Right: Solve AM⁻¹y = b, then x = M⁻¹y  
/// - Symmetric: Apply both left and right preconditioning (M₁⁻¹AM₂⁻¹)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PcSide {
    /// Left preconditioning: M⁻¹Ax = M⁻¹b
    #[default]
    Left,
    /// Right preconditioning: AM⁻¹y = b, x = M⁻¹y
    Right,
    /// Symmetric preconditioning: M₁⁻¹AM₂⁻¹y = M₁⁻¹b, x = M₂⁻¹y
    Symmetric,
}

impl FromStr for PcSide {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "left" => Ok(PcSide::Left),
            "right" => Ok(PcSide::Right),
            "symmetric" => Ok(PcSide::Symmetric),
            _ => Err(KError::UnrecognizedPcSide(s.to_string())),
        }
    }
}

/// Matrix operation selector for preconditioner application.
///
/// `Op::NoTrans` applies the factor as-is, while `Op::Trans` applies the
/// transpose. `Op::ConjTrans` is reserved for future complex-number support and
/// maps to `Op::Trans` for real-valued matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Use the factors as stored.
    NoTrans,
    /// Apply the transpose of the factors.
    Trans,
    /// Apply the conjugate-transpose of the factors. Equivalent to `Trans` for
    /// real data.
    ConjTrans,
}

/// Capability flags advertised by a preconditioner so that solvers can
/// negotiate required operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct PcCaps {
    /// Whether the preconditioner supports applying the transpose inverse.
    pub supports_transpose: bool,
    /// Whether the preconditioner supports a conjugate-transpose operation.
    pub supports_conj_trans: bool,
    /// Whether the preconditioner is symmetric positive definite when applied on the left.
    pub is_spd: bool,
    /// Restrict application to a specific side (left/right) if needed.
    pub side_restriction: Option<PcSide>,
}

#[derive(Clone, Copy, Debug)]
pub enum PcReusePolicy {
    Never,
    ReuseNumeric,
    Auto,
}
impl PcReusePolicy {
    pub fn allow_numeric(self) -> bool {
        matches!(self, PcReusePolicy::ReuseNumeric | PcReusePolicy::Auto)
    }
}

/// Object-safe preconditioner operating on `S` slices and [`LinOp`] matrices.
///
/// Preconditioners may optionally implement [`direct_solve`], allowing the
/// preconditioner to act as a stand-alone direct solver (e.g. LU, QR). Only
/// implementations that are true direct methods should override it. The default
/// implementation returns a clear [`KError`], so existing preconditioners
/// continue to work unchanged.
///
/// # SPD contract
///
/// Solvers such as Conjugate Gradient and MINRES **require** that left
/// preconditioners behave like a symmetric positive definite operator so that
/// `z = M^{-1} r` preserves symmetry.  These solvers check `rᵀz > 0` at runtime
/// and may return [`KError::IndefinitePreconditioner`] if the contract is
/// violated.
pub trait Preconditioner: Send + Sync {
    /// Dimensions `(nrows, ncols)` of the preconditioner application.
    ///
    /// Defaults to `(0, 0)` when the implementation cannot provide the size.
    /// Implementations should override this once the information is available
    /// so solvers can pre-size workspaces.
    fn dims(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Build any factorization/hierarchy once from the system matrix.
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError>;

    /// Apply `M^{-op}` to input vector, writing result to output slice.
    ///
    /// When used with CG and [`PcSide::Left`], this operation must represent an
    /// SPD preconditioner `M` so that `rᵀ M⁻¹ r > 0` holds.
    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError>;

    /// Apply `M^{-op}` where the operation (`op`) specifies transpose handling.
    ///
    /// The default implementation routes `Op::NoTrans` to [`apply`]; transpose
    /// requests return [`KError::Unsupported`]. Implementations overriding this
    /// method should ignore the [`PcSide`] argument entirely and instead allow
    /// callers to decide left/right placement by where they invoke the method.
    fn apply_op(&self, op: Op, x: &[S], y: &mut [S]) -> Result<(), KError> {
        match op {
            Op::NoTrans => self.apply(PcSide::Left, x, y),
            Op::Trans | Op::ConjTrans => {
                Err(KError::Unsupported("transpose application not supported"))
            }
        }
    }

    /// Convenience helper for in-place application (y := M^{-op} y).
    ///
    /// The default implementation allocates a temporary buffer; performance-
    /// sensitive implementations should override this.
    fn apply_op_inplace(&self, op: Op, y: &mut [S]) -> Result<(), KError> {
        let tmp = y.to_vec();
        self.apply_op(op, &tmp, y)
    }

    /// Capabilities advertised by this preconditioner.
    fn capabilities(&self) -> PcCaps {
        PcCaps::default()
    }

    /// Mutable application (flexible/nonlinear preconditioners).
    ///
    /// By default, delegates to [`apply`], so existing preconditioners
    /// remain immutable unless they explicitly override this method.
    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply(side, x, y)
    }

    /// Optional hook called at solver restarts (e.g., by FGMRES).
    ///
    /// The `outer_iter` is the total number of iterations completed so far,
    /// and `residual_norm` is the norm of the current (true) residual. The
    /// default implementation does nothing.
    fn on_restart(&mut self, _outer_iter: usize, _residual_norm: R) -> Result<(), KError> {
        Ok(())
    }

    /// Attempt to solve `op * x = b` directly using the preconditioner.
    ///
    /// The default implementation returns [`KError::SolveError`] indicating
    /// that direct solves are not supported by this preconditioner.
    fn direct_solve(
        &mut self,
        _op: &dyn LinOp<S = S>,
        _b: &[S],
        _x: &mut [S],
    ) -> Result<(), KError> {
        Err(KError::SolveError(
            "direct_solve not supported by this preconditioner".into(),
        ))
    }

    /// True if we can keep the symbolic structure and only refresh numeric values.
    fn supports_numeric_update(&self) -> bool {
        false
    }

    /// Pattern unchanged: re-use hierarchy/structure, BUT refresh all numeric data.
    fn update_numeric(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
        Err(KError::Unsupported("numeric update not supported"))
    }

    /// Pattern may have changed: rebuild structure (potentially expensive).
    fn update_symbolic(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        // By default, fall back to a full setup
        self.setup(a)
    }

    /// Preferred matrix format for `setup`/`update_*` calls.
    ///
    /// Callers will materialize a stable view of the operator in this format once per
    /// setup/update, preserving the original communicator. Defaults to Any.
    fn required_format(&self) -> OpFormat {
        OpFormat::Any
    }

    /// Optional numerical drop tolerance to use when creating the preferred format.
    ///
    /// Useful for threshold-based sparsification during conversion (e.g., ILUT).
    /// Return `None` to keep all values (treated as 0.0).
    fn preferred_drop_tol_for_format(&self) -> Option<R> {
        None
    }
}

/// Local, communicator-agnostic preconditioner used by kernel abstractions.
pub trait LocalPreconditioner<T: KrystScalar = S>: Send + Sync {
    /// Dimensions `(nrows, ncols)` of the operator.
    fn dims(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Apply `M^{-1}` locally: `y = M^{-1} x`.
    ///
    /// Contract:
    /// - `dims()` must return `(n_local, n_local)` = the number of rows handled
    ///   on the calling rank.
    /// - `x` and `y` are slices of length `n_local`, owned by the caller, and the
    ///   implementor must not assume they are contiguous with any global vector.
    /// - The implementation must perform no MPI communication; any halo exchanges
    ///   or gathers belong in the global wrapper.
    /// - After `setup` completes, repeated calls to `apply_local` should be
    ///   allocation-free (same spirit as ILU workspaces).
    fn apply_local(&self, x: &[T], y: &mut [T]) -> Result<(), crate::error::KError>;

    /// Convenience in-place application using a temporary buffer.
    fn apply_local_in_place(&self, x: &mut [T]) -> Result<(), crate::error::KError> {
        let tmp = x.to_vec();
        self.apply_local(&tmp, x)
    }
}

/// Marker trait: any [`Preconditioner`] can be treated as flexible via [`apply_mut`].
pub trait FlexiblePreconditioner: Preconditioner {}
impl<T: Preconditioner + ?Sized> FlexiblePreconditioner for T {}

pub mod stats;

/// Legacy generic preconditioner traits retained for transitional adapters.
pub mod legacy {
    use super::PcSide;
    use crate::error::KError;

    /// Generic preconditioner operating on matrix and vector types.
    pub trait Preconditioner<M: ?Sized, V> {
        fn setup(&mut self, a: &M) -> Result<(), KError>;
        fn apply(&self, side: PcSide, r: &V, z: &mut V) -> Result<(), KError>;
    }

    /// Flexible preconditioner for FGMRES-style solvers.
    pub trait FlexiblePreconditioner<M: ?Sized, V> {
        fn setup(&mut self, a: &M) -> Result<(), KError>;
        fn apply(&mut self, r: &V, z: &mut V) -> Result<(), KError>;
    }
}

#[cfg(feature = "legacy-pc-bridge")]
use std::sync::Mutex;

#[cfg(feature = "legacy-pc-bridge")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy-pc-bridge")))]
pub struct LegacyOpPreconditioner {
    inner: Box<dyn legacy::Preconditioner<Mat<f64>, Vec<R>> + Send + Sync>,
    scratch: Mutex<Scratch>,
}

#[cfg(feature = "legacy-pc-bridge")]
#[derive(Default)]
struct Scratch {
    x: Vec<R>,
    y: Vec<R>,
}

#[cfg(feature = "legacy-pc-bridge")]
impl LegacyOpPreconditioner {
    pub fn new(inner: Box<dyn legacy::Preconditioner<Mat<f64>, Vec<R>> + Send + Sync>) -> Self {
        Self {
            inner,
            scratch: Mutex::new(Scratch::default()),
        }
    }

    #[inline]
    fn ensure_scratch(s: &mut Scratch, n: usize) {
        if s.x.len() != n {
            s.x.resize(n, R::default());
        }
        if s.y.len() != n {
            s.y.resize(n, R::default());
        }
    }
}

#[cfg(feature = "legacy-pc-bridge")]
impl Preconditioner for LegacyOpPreconditioner {
    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        use crate::error::KError;
        let m = a
            .as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("expected faer::Mat<f64>".into()))?;
        self.inner.setup(m)
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        use crate::error::KError;
        if x.len() != y.len() {
            return Err(KError::InvalidInput(format!(
                "x.len()={} != y.len()={}",
                x.len(),
                y.len()
            )));
        }
        let mut s = self.scratch.lock().unwrap();
        Self::ensure_scratch(&mut s, x.len());
        crate::algebra::scalar::copy_scalar_to_real_in(x, &mut s.x);
        let Scratch { x: x_buf, y: y_buf } = &mut *s;
        self.inner.apply(side, &*x_buf, y_buf)?;
        crate::algebra::scalar::copy_real_to_scalar_in(&s.y, y);
        Ok(())
    }

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply(side, x, y)
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.setup(a)
    }

    fn required_format(&self) -> OpFormat {
        // Legacy bridge adapters expect dense Mat<f64>
        OpFormat::Dense
    }
}

#[cfg(all(feature = "legacy-pc-bridge", feature = "complex"))]
impl crate::ops::kpc::KPreconditioner for LegacyOpPreconditioner {
    type Scalar = crate::S;

    #[inline]
    fn dims(&self) -> (usize, usize) {
        let guard = self.scratch.lock().unwrap();
        let n = guard.x.len();
        if n == 0 { (0, 0) } else { (n, n) }
    }

    fn apply_s(
        &self,
        side: PcSide,
        x: &[crate::S],
        y: &mut [crate::S],
        scratch: &mut crate::algebra::bridge::BridgeScratch,
    ) -> Result<(), KError> {
        crate::preconditioner::bridge::apply_pc_s(self, side, x, y, scratch)
    }

    fn apply_mut_s(
        &mut self,
        side: PcSide,
        x: &[crate::S],
        y: &mut [crate::S],
        scratch: &mut crate::algebra::bridge::BridgeScratch,
    ) -> Result<(), KError> {
        crate::preconditioner::bridge::apply_pc_mut_s(self, side, x, y, scratch)
    }

    fn on_restart_s(
        &mut self,
        outer_iter: usize,
        residual_norm: <Self::Scalar as crate::algebra::prelude::KrystScalar>::Real,
    ) -> Result<(), KError> {
        self.on_restart(outer_iter, residual_norm)
    }
}

#[cfg(all(not(feature = "legacy-pc-bridge"), feature = "backend-faer"))]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy-pc-bridge")))]
pub struct LegacyOpPreconditioner {
    _private: (),
}

#[cfg(all(not(feature = "legacy-pc-bridge"), feature = "backend-faer"))]
impl LegacyOpPreconditioner {
    pub fn new(_: Box<dyn legacy::Preconditioner<faer::Mat<f64>, Vec<f64>> + Send + Sync>) -> Self {
        panic!("legacy-pc-bridge feature is disabled")
    }
}

#[cfg(all(not(feature = "legacy-pc-bridge"), feature = "backend-faer"))]
impl Preconditioner for LegacyOpPreconditioner {
    fn setup(&mut self, _: &dyn LinOp<S = S>) -> Result<(), KError> {
        Err(KError::Unsupported("legacy-pc-bridge feature is disabled"))
    }
    fn apply(&self, _: PcSide, _: &[S], _: &mut [S]) -> Result<(), KError> {
        Err(KError::Unsupported("legacy-pc-bridge feature is disabled"))
    }
}

#[cfg(not(feature = "backend-faer"))]
pub struct LegacyOpPreconditioner {
    _private: (),
}

#[cfg(not(feature = "backend-faer"))]
impl LegacyOpPreconditioner {
    pub fn new<T>(_: T) -> Self {
        panic!("backend-faer feature is disabled")
    }
}

#[cfg(not(feature = "backend-faer"))]
impl Preconditioner for LegacyOpPreconditioner {
    fn setup(&mut self, _: &dyn LinOp<S = S>) -> Result<(), KError> {
        Err(KError::Unsupported("backend-faer feature is disabled"))
    }
    fn apply(&self, _: PcSide, _: &[S], _: &mut [S]) -> Result<(), KError> {
        Err(KError::Unsupported("backend-faer feature is disabled"))
    }
}

// Submodules for various preconditioners
#[cfg(feature = "backend-faer")]
pub mod amg;
#[cfg(feature = "backend-faer")]
pub mod approxinv;
#[cfg(feature = "backend-faer")]
pub mod approxinv_csr;
#[cfg(feature = "backend-faer")]
pub mod asm;
#[cfg(feature = "backend-faer")]
pub mod asm_amg;
#[cfg(feature = "backend-faer")]
pub mod block_jacobi;
#[cfg(feature = "backend-faer")]
pub mod builders;
pub mod builders_none;
#[cfg(feature = "backend-faer")]
pub mod builders_faer;
#[cfg(feature = "backend-nalgebra")]
pub mod builders_nalgebra;
pub mod chain;
#[cfg(feature = "backend-faer")]
pub mod chebyshev;
#[cfg(feature = "backend-faer")]
pub mod deflation;
#[cfg(feature = "backend-faer")]
pub mod direct;
#[cfg(feature = "backend-faer")]
pub mod dist;
#[cfg(feature = "backend-nalgebra")]
pub mod nalgebra_direct;
#[cfg(feature = "backend-faer")]
pub mod ilu;
#[cfg(feature = "backend-faer")]
pub mod ilu_csr;
pub mod ilu_options;
#[cfg(feature = "backend-faer")]
pub mod ilup;
#[cfg(feature = "backend-faer")]
pub mod ilut;
#[cfg(feature = "backend-faer")]
pub mod ilutp;
pub mod jacobi;
pub mod pivot;
#[cfg(feature = "backend-faer")]
pub mod sor;
pub mod tri_solve;

// Re-exports for convenience
#[cfg(feature = "backend-faer")]
pub use self::sor::MatSorType;
#[cfg(feature = "backend-faer")]
pub use amg::AMG;
#[cfg(feature = "backend-faer")]
pub use approxinv::ApproxInv;
#[cfg(feature = "backend-faer")]
pub use approxinv_csr::{ApproxInvBuilder, ApproxInvKind, ApproxInvParams, FsaiCsr, SpaiCsr};
#[cfg(feature = "backend-faer")]
pub use asm::{AdditiveSchwarz, Asm, AsmBuilder, AsmCombine, AsmConfig, AsmLocalSolver};
#[cfg(feature = "backend-faer")]
pub use asm_amg::{AsmAmg, AsmAmgBuilder, TwoLevelConfig, TwoLevelMode};
pub use chain::PcChain;
#[cfg(feature = "backend-faer")]
pub use chebyshev::{Chebyshev, ChebyshevPre};
#[cfg(feature = "backend-faer")]
pub use deflation::{AmgCoarseSpace, DeflationOptions, DeflationPC, ZSource, with_amg_deflation};
#[cfg(feature = "backend-faer")]
pub use direct::SuperLuDistPc;
#[cfg(all(feature = "dense-direct", feature = "backend-faer"))]
pub use direct::{LuPc, QrPc};
#[cfg(feature = "backend-faer")]
pub use ilu::Ilu0;
#[cfg(feature = "backend-faer")]
pub use ilup::Ilup;
#[cfg(feature = "backend-faer")]
#[allow(deprecated)]
pub use ilut::{Ilut, RowFilterPreconditioner};
#[cfg(feature = "backend-faer")]
pub use ilutp::Ilutp;
pub use jacobi::Jacobi;
#[cfg(feature = "backend-faer")]
pub use sor::Sor;

pub use pivot::{PivotMode, PivotPolicy, PivotScale, PivotSignPolicy, PivotStats};
pub use tri_solve::TriangularSolve;

/// Unified preconditioner enum for all supported types.
pub use crate::context::pc_context::SparsityPattern;

#[cfg(test)]
mod tests;
