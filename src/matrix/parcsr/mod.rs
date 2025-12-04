//! Legacy AMG-focused distributed CSR (diag/off splitting).
//!
//! NOTE: `ParCsrMatrix` is maintained for backward compatibility but is
//! considered secondary to `matrix::DistCsrOp`. New distributed code should
//! prefer `DistCsrOp`, and ParCsrMatrix will be gradually reworked to build on
//! the same halo infrastructure.

pub type Global = u64;
pub type Local = usize;

pub mod builder;
pub mod halo;
pub mod mat;

pub use halo::HaloPlan;
pub use mat::{ParCsrMatrix, ParCsrOp};
