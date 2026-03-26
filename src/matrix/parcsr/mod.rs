//! Legacy AMG-focused distributed CSR (diag/off splitting).
//!
//! NOTE: `ParCsrMatrix` is maintained for backward compatibility but is
//! considered secondary to `matrix::DistCsrOp`. New distributed code should
//! prefer `DistCsrOp`, and ParCsrMatrix will be gradually reworked to build on
//! the same halo infrastructure.
//!
//! Migration plan (active):
//! 1. `ParCsrMatrix` matvec now delegates to an internal canonical
//!    `DistCsrOp` adapter.
//! 2. Halo exchange and ownership semantics are unified through
//!    `matrix::dist::halo::HaloPlan` used by `DistCsrOp`.
//! 3. Solver/PC entry points should consume `LinOp::dist_layout()` /
//!    `LinOp::comm()` metadata; `ParCsrOp` forwards these from the canonical
//!    operator.
//! 4. Legacy diag/off data structures remain only as compatibility adapters and
//!    are marked for deprecation in favor of canonical conversion helpers.
//! 5. Test parity gates validate that legacy `ParCsrMatrix` SpMV behavior
//!    matches canonical distributed SpMV before removing legacy routes.

pub type Global = u64;
pub type Local = usize;

pub mod builder;
pub mod halo;
pub mod mat;

pub use halo::HaloPlan;
pub use mat::{ParCsrMatrix, ParCsrOp};
