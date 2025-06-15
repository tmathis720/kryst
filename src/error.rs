use thiserror::Error;

// Unified error type for kryst

#[derive(Error, Debug)]
pub enum KError {
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
}
