//! Block vector helpers for block Krylov solvers.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
pub use crate::context::ksp_context::BlockVec;
