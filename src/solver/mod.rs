//! Krylov & direct solver interfaces.

use crate::utils::convergence::SolveStats;
use crate::preconditioner::Preconditioner;

/// Common interface for any direct or iterative solver.
pub trait LinearSolver<M, V> {
    type Error;
    /// Solve A·x = b, optionally with preconditioner M⁻¹, writing result into `x`.
    /// Returns iteration stats (including convergence info).
    fn solve(
        &mut self,
        a: &M,
        pc: Option<&dyn Preconditioner<M, V>>,
        b: &V,
        x: &mut V
    ) -> Result<SolveStats<<Self as LinearSolver<M, V>>::Scalar>, Self::Error>;
    type Scalar: Copy + PartialOrd + From<f64>;
}

pub mod direct_lu;
pub use direct_lu::{LuSolver, QrSolver};

pub mod cg;
pub use cg::CgSolver;

pub mod gmres;
pub use gmres::GmresSolver;

pub mod bicgstab;
pub use bicgstab::BiCgStabSolver;

pub mod cgs;
pub use cgs::CgsSolver;

pub mod qmr;
pub use qmr::QmrSolver;

pub mod minres;
pub use minres::MinresSolver;

pub mod tfqmr;
pub use tfqmr::TfqmrSolver;

pub mod cgnr;
pub use cgnr::{CgnrSolver, CgneSolver};

pub mod pcg;
pub use self::pcg::PcgSolver;

pub mod fgmres;
