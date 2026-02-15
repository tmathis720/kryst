use thiserror::Error;

use crate::utils::convergence::NestedPcFailure;

// Unified error type for kryst

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum KError {
    #[error("help requested")]
    HelpRequested(String),
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
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("preconditioner failed: {0}")]
    PcFailed(String),
    #[error("nested preconditioner failed ({0:?})")]
    NestedPcFailed(NestedPcFailure),
    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),
    #[error("unrecognized solver type: {0}")]
    UnrecognizedSolverType(String),
    #[error("unrecognized preconditioner type: {0}")]
    UnrecognizedPcType(String),
    #[error("unrecognized preconditioner side: {0}")]
    UnrecognizedPcSide(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_messages() {
        assert_eq!(
            format!("{}", KError::DivergedIndefinitePC),
            "breakdown or indefinite preconditioner detected (beta < 0)"
        );

        assert_eq!(
            format!("{}", KError::BreakdownOrIndefinite),
            "breakdown or indefinite situation detected (beta < 0 or other)"
        );

        assert_eq!(
            format!("{}", KError::FactorError("singular matrix".to_string())),
            "factorization error: singular matrix"
        );

        assert_eq!(
            format!(
                "{}",
                KError::SolveError("maximum iterations reached".to_string())
            ),
            "solve error: maximum iterations reached"
        );

        assert_eq!(
            format!("{}", KError::IndefiniteMatrix),
            "indefinite matrix detected (p^T A p <= 0)"
        );

        assert_eq!(
            format!("{}", KError::IndefinitePreconditioner),
            "indefinite preconditioner detected (beta < 0)"
        );

        assert_eq!(
            format!("{}", KError::PcFailed("pc shell".to_string())),
            "preconditioner failed: pc shell"
        );

        assert_eq!(format!("{}", KError::ZeroPivot(42)), "zero pivot at row 42");

        assert_eq!(
            format!("{}", KError::Unsupported("complex arithmetic")),
            "unsupported operation: complex arithmetic"
        );

        assert_eq!(
            format!("{}", KError::UnrecognizedSolverType("unknown".to_string())),
            "unrecognized solver type: unknown"
        );

        assert_eq!(
            format!("{}", KError::UnrecognizedPcType("mystery".to_string())),
            "unrecognized preconditioner type: mystery"
        );

        assert_eq!(
            format!("{}", KError::UnrecognizedPcSide("diagonal".to_string())),
            "unrecognized preconditioner side: diagonal"
        );
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = KError::ZeroPivot(10);
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("ZeroPivot"));
        assert!(debug_str.contains("10"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<KError>();
        assert_sync::<KError>();
    }

    #[test]
    fn test_error_equality() {
        assert_eq!(KError::DivergedIndefinitePC, KError::DivergedIndefinitePC);
        assert_eq!(KError::IndefiniteMatrix, KError::IndefiniteMatrix);
        assert_eq!(KError::ZeroPivot(5), KError::ZeroPivot(5));
        assert_ne!(KError::ZeroPivot(5), KError::ZeroPivot(6));

        let error1 = KError::FactorError("test".to_string());
        let error2 = KError::FactorError("test".to_string());
        assert_eq!(error1, error2);

        let error3 = KError::FactorError("different".to_string());
        assert_ne!(error1, error3);
    }

    #[test]
    fn test_error_clone() {
        let original = KError::SolveError("convergence failed".to_string());
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_specific_error_variants() {
        // Test construction of each variant
        let _e1 = KError::DivergedIndefinitePC;
        let _e2 = KError::BreakdownOrIndefinite;
        let _e3 = KError::FactorError("test".to_string());
        let _e4 = KError::SolveError("test".to_string());
        let _e5 = KError::IndefiniteMatrix;
        let _e6 = KError::IndefinitePreconditioner;
        let _e7 = KError::ZeroPivot(0);
        let _e8 = KError::PcFailed("test".to_string());
        let _e9 = KError::Unsupported("test");
        let _e10 = KError::UnrecognizedSolverType("test".to_string());
        let _e11 = KError::UnrecognizedPcType("test".to_string());
        let _e12 = KError::UnrecognizedPcSide("test".to_string());

        // All variants should be constructible
        assert!(true);
    }
}
