//! Optional CUDA execution support.
//!
//! CUDA objects are explicit: constructing a [`CudaRuntime`] selects a device,
//! and [`CudaVector`] values remain on that device until copied by the caller.
//! The ordinary host-slice [`crate::context::KspContext`] API is unchanged.

mod dense;
mod kernels;
mod operator;
mod preconditioner;
mod runtime;
mod solver;
mod triangular;
mod vector;

mod dist;

pub use dense::CudaDenseOp;
pub use operator::{CudaCsrOp, CudaLinOp, CudaOperation};
pub use preconditioner::{
    CudaAmg, CudaAmgOptions, CudaBlockJacobi, CudaChebyshev, CudaIlu0, CudaJacobi, CudaNone,
    CudaPreconditioner,
};
pub use runtime::{
    CudaDiagnosticsSnapshot, CudaMpiTransport, CudaOptions, CudaRuntime, CudaSpmvAlgorithm,
};
pub use solver::{CudaCgVariant, CudaGmresVariant, CudaKspContext};
pub use vector::CudaVector;

pub use dist::CudaDistCsrOp;
