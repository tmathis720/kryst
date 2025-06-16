// PCContext struct and logic

/// Unified preconditioner enum for all supported types.
pub enum PC<T> {
    Jacobi,
    Ssor,
    Ilu0,
    Ilup { fill: usize },
    Ilut { fill: usize, droptol: T },
    Chebyshev { degree: usize, emin: Option<T>, emax: Option<T> },
    ApproxInv { pattern: SparsityPattern, tol: T, max_iter: usize },
    BlockJacobi { blocks: Vec<Vec<usize>> },
    Multicolor { colors: Vec<usize> },
    AMG,
    AdditiveSchwarz,
}

/// Sparsity pattern for approximate inverse preconditioners.
pub enum SparsityPattern {
    Auto,
    Manual(Vec<Vec<usize>>), // for each row, the list of column indices
}
