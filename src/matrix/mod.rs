//! Matrix module: dense and sparse matrix types and traits.
//!
//! `LinOp` is the core operator trait used by [`KspContext`](crate::context::KspContext).
//! Its `StructureId` / `ValuesId` are used to decide when setup can be reused.
//!
//! # Real-only helpers vs generic APIs
//!
//! - The core `LinOp` trait and most sparse structures are generic over
//!   `S: KrystScalar`.
//! - AMG-oriented helpers (e.g. `matrix::convert`, `matrix::format`,
//!   `matrix::utils`, SIMD SpMV plans) are intentionally **restricted to
//!   `S = f64`**.
//!
//! This split lets the solvers remain scalar-general while making real-valued
//! workflows (AMG, ILU, etc.) fast and ergonomic.

pub mod backend;
#[cfg(feature = "backend-faer")]
pub mod dense;
pub mod dense_api;
pub mod sparse_api;
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
#[cfg(feature = "backend-nalgebra")]
pub mod op_nalgebra;
pub mod op_shell;
pub mod parcsr;
pub mod preprocess;
pub mod sparse;
pub mod spmv;
pub mod utils;

#[cfg(feature = "backend-faer")]
pub mod real_amg {
    //! Re-exports of AMG-oriented helpers that assume `S = f64`.
    pub use super::convert;
    pub use super::format;
    pub use super::spmv;
    pub use super::utils;
}

#[cfg(feature = "backend-faer")]
pub use backend::DefaultBackend;
#[cfg(feature = "backend-faer")]
pub use backend::faer::{DefaultCscMat, DefaultCsrMat, DefaultDenseMat, FaerBackend};
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
pub use sparse_api::{CscMatMut, CscMatRef, CsrMatMut, CsrMatRef};

#[allow(unused_imports)]
use crate::algebra::prelude::*;

pub type Csr = crate::matrix::sparse::CsrMatrix<S>;
#[cfg(feature = "backend-faer")]
pub type Csc = crate::matrix::csc::CscMatrix<S>;

pub use dist::{DistRowCsr, LocalSquareCsr};
pub use dist_csr::DistCsrOp;
pub use format::OpFormat;
pub use op::{ChangeIds, LinOp, LinOpF64, StructureId, ValuesId};
#[cfg(feature = "backend-faer")]
pub use op::{CsrOp, DenseOp, GenericCsrOp};
#[cfg(feature = "backend-nalgebra")]
pub use op_nalgebra::NalgebraDenseOp;
pub use op_shell::MatShell;
