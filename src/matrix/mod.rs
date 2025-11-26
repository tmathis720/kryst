//! Matrix module: dense and sparse matrix types and traits.

#[cfg(feature = "backend-faer")]
pub mod dense;
#[cfg(feature = "backend-faer")]
pub use dense::DenseMatrix;
#[cfg(feature = "backend-faer")]
pub mod convert;
#[cfg(feature = "backend-faer")]
pub mod csc;
pub mod csr;
pub mod dist;
pub mod dist_csr;
pub mod format;
#[cfg(feature = "backend-faer")]
mod format_impls;
pub mod op;
pub mod op_bridge;
pub mod op_shell;
pub mod parcsr;
pub mod sparse;
#[cfg(feature = "backend-faer")]
pub mod spmv;
pub mod utils;

#[cfg(feature = "backend-faer")]
pub use convert::owned_from_mat;
#[cfg(feature = "backend-faer")]
pub use convert::{
    csc_from_linop, csr_from_linop, dense_from_linop, to_csc_cached, to_csr_cached, try_as_csc,
    try_as_csr,
};
#[cfg(feature = "backend-faer")]
pub use csc::CscMatrix;
pub use sparse::CsrMatrix;

#[allow(unused_imports)]
use crate::algebra::prelude::*;

pub type Csr = crate::matrix::sparse::CsrMatrix<S>;
#[cfg(feature = "backend-faer")]
pub type Csc = crate::matrix::csc::CscMatrix<S>;

pub use dist_csr::DistCsrOp;
pub use op::{ChangeIds, LinOp, LinOpF64, StructureId, ValuesId};
#[cfg(feature = "backend-faer")]
pub use op::{CsrOp, DenseOp, GenericCsrOp};
pub use op_shell::MatShell;
