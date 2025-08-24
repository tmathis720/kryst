//! Matrix module: dense and sparse matrix types and traits.

pub mod dense;
pub use dense::DenseMatrix;
pub mod convert;
pub mod format;
mod format_impls;
pub mod op;
pub mod op_shell;
pub mod sparse;
pub mod utils;

pub use convert::{to_csr_cached, try_as_csr};
pub use op::{ChangeIds, CsrOp, DenseOp, LinOp, StructureId, ValuesId};
pub use op_shell::MatShell;
