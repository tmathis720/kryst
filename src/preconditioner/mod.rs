//! Preconditioners for linear solvers.
//!
//! This module defines the Preconditioner trait and includes implementations such as Jacobi, ILU, SOR, AMG, Additive Schwarz, and more.
//!
//! ## Flexible preconditioners
//!
//! Flexible Krylov methods call [`Preconditioner::apply_mut`], allowing
//! preconditioners to update internal state between applications.
//! Non-flexible solvers continue to invoke [`Preconditioner::apply`].
//! The default `apply_mut` simply forwards to `apply`, so existing
//! implementations remain unchanged unless they opt in to mutation.
//!
//! Preconditioners obtain the parallel communicator from the operator via
//! [`LinOp::comm()`], eliminating the need to thread communicator handles
//! through solver interfaces.

use crate::error::KError;
use crate::matrix::op::LinOp;
#[cfg(feature = "legacy-pc-bridge")]
use faer::Mat;
use std::str::FromStr;

/// Which side to apply M⁻¹ on in preconditioning.
///
/// For the linear system Ax = b with preconditioner M ≈ A:
/// - Left: Solve M⁻¹Ax = M⁻¹b
/// - Right: Solve AM⁻¹y = b, then x = M⁻¹y  
/// - Symmetric: Apply both left and right preconditioning (M₁⁻¹AM₂⁻¹)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcSide {
    /// Left preconditioning: M⁻¹Ax = M⁻¹b
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

impl Default for PcSide {
    fn default() -> Self {
        PcSide::Left
    }
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

/// Object-safe preconditioner operating on `f64` slices and [`LinOp`] matrices.
///
/// Preconditioners may optionally implement [`direct_solve`], allowing the
/// preconditioner to act as a stand-alone direct solver (e.g. LU, QR). Only
/// implementations that are true direct methods should override it. The default
/// implementation returns a clear [`KError`], so existing preconditioners
/// continue to work unchanged.
pub trait Preconditioner: Send + Sync {
    /// Build any factorization/hierarchy once from the system matrix.
    fn setup(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError>;

    /// Apply M⁻¹ to input vector, writing result to output slice.
    fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError>;

    /// Mutable application (flexible/nonlinear preconditioners).
    ///
    /// By default, delegates to [`apply`], so existing preconditioners
    /// remain immutable unless they explicitly override this method.
    fn apply_mut(
        &mut self,
        side: PcSide,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), KError> {
        self.apply(side, x, y)
    }

    /// Attempt to solve `op * x = b` directly using the preconditioner.
    ///
    /// The default implementation returns [`KError::SolveError`] indicating
    /// that direct solves are not supported by this preconditioner.
    fn direct_solve(
        &mut self,
        _op: &dyn LinOp<S = f64>,
        _b: &[f64],
        _x: &mut [f64],
    ) -> Result<(), KError> {
        Err(KError::SolveError(
            "direct_solve not supported by this preconditioner".into(),
        ))
    }

    /// True if we can keep the symbolic structure and only refresh numeric values.
    fn supports_numeric_update(&self) -> bool { false }

    /// Pattern unchanged: re-use hierarchy/structure, BUT refresh all numeric data.
    fn update_numeric(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Err(KError::Unsupported("numeric update not supported"))
    }

    /// Pattern may have changed: rebuild structure (potentially expensive).
    fn update_symbolic(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        // By default, fall back to a full setup
        self.setup(a)
    }
}

/// Marker trait: any [`Preconditioner`] can be treated as flexible via [`apply_mut`].
pub trait FlexiblePreconditioner: Preconditioner {}
impl<T: Preconditioner + ?Sized> FlexiblePreconditioner for T {}

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
pub struct LegacyOpPreconditioner {
    inner: Box<dyn legacy::Preconditioner<Mat<f64>, Vec<f64>> + Send + Sync>,
    scratch: Mutex<Scratch>,
}

#[cfg(feature = "legacy-pc-bridge")]
#[derive(Default)]
struct Scratch {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[cfg(feature = "legacy-pc-bridge")]
impl LegacyOpPreconditioner {
    pub fn new(inner: Box<dyn legacy::Preconditioner<Mat<f64>, Vec<f64>> + Send + Sync>) -> Self {
        Self { inner, scratch: Mutex::new(Scratch::default()) }
    }

    #[inline]
    fn ensure_scratch(s: &mut Scratch, n: usize) {
        if s.x.len() != n { s.x.resize(n, 0.0); }
        if s.y.len() != n { s.y.resize(n, 0.0); }
    }
}

#[cfg(feature = "legacy-pc-bridge")]
impl Preconditioner for LegacyOpPreconditioner {
    fn setup(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        use crate::error::KError;
        let m = a
            .as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("expected faer::Mat<f64>".into()))?;
        self.inner.setup(m)
    }

    fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        use crate::error::KError;
        if x.len() != y.len() {
            return Err(KError::InvalidInput(format!("x.len()={} != y.len()={}", x.len(), y.len())));
        }
        let mut s = self.scratch.lock().unwrap();
        Self::ensure_scratch(&mut s, x.len());
        s.x.copy_from_slice(x);
        let Scratch { x: x_buf, y: y_buf } = &mut *s;
        self.inner.apply(side, &*x_buf, y_buf)?;
        y.copy_from_slice(&s.y);
        Ok(())
    }

    fn apply_mut(&mut self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        self.apply(side, x, y)
    }
}

#[cfg(not(feature = "legacy-pc-bridge"))]
pub struct LegacyOpPreconditioner {
    _private: (),
}

#[cfg(not(feature = "legacy-pc-bridge"))]
impl LegacyOpPreconditioner {
    pub fn new(_: Box<dyn legacy::Preconditioner<faer::Mat<f64>, Vec<f64>> + Send + Sync>) -> Self {
        panic!("legacy-pc-bridge feature is disabled")
    }
}

#[cfg(not(feature = "legacy-pc-bridge"))]
impl Preconditioner for LegacyOpPreconditioner {
    fn setup(&mut self, _: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Err(KError::Unsupported("legacy-pc-bridge feature is disabled"))
    }
    fn apply(&self, _: PcSide, _: &[f64], _: &mut [f64]) -> Result<(), KError> {
        Err(KError::Unsupported("legacy-pc-bridge feature is disabled"))
    }
}


// Submodules for various preconditioners
pub mod block_jacobi;
pub mod ilu;
pub mod jacobi;
pub mod sor;
pub mod amg;
pub mod asm;
pub mod ilut;
pub mod ilutp;
pub mod ilup;
pub mod chebyshev;
pub mod approxinv;
pub mod chain;
pub mod direct;
pub mod builders;

// Re-exports for convenience
pub use jacobi::Jacobi;
pub use sor::Sor;
pub use ilu::Ilu0;
pub use amg::AMG;
pub use asm::AdditiveSchwarz;
pub use ilut::Ilut;
pub use ilutp::Ilutp;
pub use ilup::Ilup;
pub use chebyshev::{Chebyshev, ChebyshevPre};
pub use approxinv::ApproxInv;
pub use chain::PcChain;
pub use direct::{LuPc, QrPc, SuperLuDistPc};
pub use self::sor::MatSorType;

/// Unified preconditioner enum for all supported types.
pub use crate::context::pc_context::{PC, SparsityPattern};

#[cfg(test)]
mod tests;
