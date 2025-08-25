//! Matrix module: dense and sparse matrix types and traits.

pub mod dense;
pub use dense::DenseMatrix;
pub mod convert;
pub mod csc;
pub mod dist_csr;
pub mod format;
mod format_impls;
pub mod op;
pub mod op_shell;
pub mod sparse;
pub mod utils;

pub use convert::owned_from_mat;
pub use convert::{
    csc_from_linop, csr_from_linop, dense_from_linop, to_csc_cached, to_csr_cached, try_as_csc,
    try_as_csr,
};
pub use csc::CscMatrix;
pub use sparse::CsrMatrix;

pub use dist_csr::DistCsrOp;
pub use op::{ChangeIds, CsrOp, DenseOp, LinOp, StructureId, ValuesId};
pub use op_shell::MatShell;
