//! Matrix module: dense and sparse matrix types and traits.

pub mod dense;
pub use dense::DenseMatrix;
pub mod sparse;
pub mod utils;
pub mod op;
pub mod op_shell;

pub use op::LinOp;
pub use op_shell::MatShell;
