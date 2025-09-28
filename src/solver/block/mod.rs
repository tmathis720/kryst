//! Block Krylov solver infrastructure.

use crate::context::ksp_context::ReorthPolicy;

pub mod arnoldi;
pub mod bicgstab;
pub mod block_vec;
pub mod gmres;
pub mod kernels;

pub use arnoldi::{ArnoldiOutput, block_arnoldi_step};

/// Configuration options shared across block Krylov solvers.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockKrylovOptions {
    /// Number of right-hand sides processed together.
    pub block_size: usize,
    /// Number of block Arnoldi steps between restarts.
    pub restart_blocks: usize,
    /// Reorthogonalisation policy for the block Arnoldi process.
    pub reorth: ReorthPolicy,
    /// Conditioning guard for the Cholesky/QR factorisations.
    pub max_cond: f64,
    /// Selected solver variant.
    pub variant: BlockVariant,
}

impl Default for BlockKrylovOptions {
    fn default() -> Self {
        Self {
            block_size: 1,
            restart_blocks: 10,
            reorth: ReorthPolicy::IfNeeded,
            max_cond: 1.0e8,
            variant: BlockVariant::Gmres,
        }
    }
}

/// Supported block Krylov variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockVariant {
    Gmres,
    FgmresRight,
    Bicgstab,
}

pub use block_vec::BlockVec;
