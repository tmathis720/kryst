//! The solver-facing “ops” layer.
//!
//! * `KLinOp` / `KPreconditioner` are the canonical, scalar-generic interfaces solvers
//!   manipulate (typically via `Arc<dyn ...>` or borrowed trait objects). Each trait is object
//!   safe and keeps backend details hidden behind `KrystScalar` and `BridgeScratch`.
//! * `matrix::*` / `preconditioner::*` provide concrete `f64` implementations that expose
//!   `LinOpF64` / `PreconditionerF64`.
//! * `wrap` translates those real backends into the solver-facing world (`F64AsSOp`,
//!   `F64AsSPc`, etc.), so complex scalars and backend-specific details never leak into solvers.
pub mod klinop;
pub mod kpc;
pub mod wrap;
