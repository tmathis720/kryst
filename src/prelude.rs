//! Daily-driver re-exports for kryst.

pub use crate::algebra::prelude::{KrystScalar, R, S};
pub use crate::config::options::{KspOptions, PcOptions};
pub use crate::context::KspContext;
pub use crate::context::ksp_context::SolverType;
pub use crate::context::pc_context::PcType;
pub use crate::matrix::LinOp;
pub use crate::matrix::OpFormat;
pub use crate::parallel::{Comm, UniverseComm};
pub use crate::preconditioner::{PcSide, Preconditioner};
pub use crate::utils::convergence::{ConvergedReason, SolveStats};
