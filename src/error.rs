use thiserror::Error;

// Unified error type for kryst

#[derive(Error, Debug)]
pub enum KError {
    #[error("breakdown or indefinite preconditioner detected (beta < 0)")]
    DivergedIndefinitePC,
    #[error("breakdown or indefinite situation detected (beta < 0 or other)")]
    BreakdownOrIndefinite,
    #[error("factorization error: {0}")]
    FactorError(String),
    #[error("solve error: {0}")]
    SolveError(String),
    #[error("indefinite matrix detected (p^T A p <= 0)")]
    IndefiniteMatrix,
    #[error("indefinite preconditioner detected (beta < 0)")]
    IndefinitePreconditioner,
    #[error("zero pivot at row {0}")]
    ZeroPivot(usize),
    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),
    #[error("unrecognized solver type: {0}")]
    UnrecognizedSolverType(String),
    #[error("unrecognized preconditioner type: {0}")]
    UnrecognizedPcType(String),
    #[error("unrecognized preconditioner side: {0}")]
    UnrecognizedPcSide(String),
}
