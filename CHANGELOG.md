# Changelog

## [Unreleased]
- `Preconditioner::direct_solve` signature change
- `LinearSolver::solve` now receives `PcSide`
- Removal of `MatOpPreconditioner`; feature-gated `LegacyOpPreconditioner`
- Options → Builders wiring; deferred PCs; chains
- Standardized error path to `crate::error::KError` across the crate. Examples/tests updated for consistency. Root re-export remains for compatibility.
