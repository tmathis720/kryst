//! Matrix module: dense and sparse matrix types and traits.

pub mod dense;
pub use dense::DenseMatrix;
pub mod sparse;
pub mod utils;
pub mod op;
pub mod op_shell;
pub mod format;
mod format_impls;
pub mod convert;

pub use op::LinOp;
pub use op_shell::MatShell;
pub use convert::{try_as_csr, to_csr_cached};
