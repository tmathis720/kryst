//! Distributed preconditioner helpers and wrappers.
//!
//! Local preconditioners operate purely on the data owned by the current rank.
//! This module exposes a small trait for global wrappers and the distributed
//! vector helper that carries only the local slice.

use crate::error::KError;
use crate::parallel::UniverseComm;
use crate::preconditioner::PcSide;
use crate::preconditioner::ilu::IluConfig;
use std::str::FromStr;

/// Global distributed preconditioner modes exposed to CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalPcKind {
    /// No MPI-level preconditioning.
    None,
    /// Block-Jacobi wrapping a local ILU-like solver.
    BlockJacobi,
    /// Additive Schwarz (not implemented yet).
    Asm,
}

impl FromStr for GlobalPcKind {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(GlobalPcKind::None),
            "block-jacobi" | "blockjacobi" | "block_jacobi" => Ok(GlobalPcKind::BlockJacobi),
            "asm" => Ok(GlobalPcKind::Asm),
            other => Err(KError::InvalidInput(format!(
                "Unknown pc_global value: {other}"
            ))),
        }
    }
}

/// Local block preconditioner family used within block-Jacobi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalPcKind {
    Ilu,
    Ilut,
    Ilutp,
}

impl FromStr for LocalPcKind {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ilu" => Ok(LocalPcKind::Ilu),
            "ilut" => Ok(LocalPcKind::Ilut),
            "ilutp" => Ok(LocalPcKind::Ilutp),
            other => Err(KError::InvalidInput(format!(
                "Unknown pc_local value: {other}"
            ))),
        }
    }
}

/// Parsed MPI-specific PC options.
#[derive(Clone, Debug)]
pub struct MpiPcOptions {
    pub global_pc: GlobalPcKind,
    pub local_pc: LocalPcKind,
    pub ilu_config: IluConfig,
    pub ilut_fill: usize,
    pub ilut_drop_tol: f64,
    pub ilut_perm_tol: f64,
    pub ilutp_max_fill: usize,
    pub ilutp_drop_tol: f64,
    pub ilutp_perm_tol: f64,
}

impl Default for MpiPcOptions {
    fn default() -> Self {
        Self {
            global_pc: GlobalPcKind::None,
            local_pc: LocalPcKind::Ilu,
            ilu_config: IluConfig::default(),
            ilut_fill: 10,
            ilut_drop_tol: 1e-4,
            ilut_perm_tol: 0.1,
            ilutp_max_fill: 10,
            ilutp_drop_tol: 1e-4,
            ilutp_perm_tol: 0.1,
        }
    }
}

/// Distributed preconditioners expose an MPI-friendly apply API.
pub trait DistributedPreconditioner {
    type Scalar;

    /// Apply the distributed preconditioner to a global vector.
    fn apply_global(&self, side: PcSide, x: &mut DistVec) -> Result<(), KError>;
}

/// Simple distributed vector carrying only the owned local slice.
#[derive(Debug)]
pub struct DistVec {
    comm: UniverseComm,
    row_offset: usize,
    global_len: usize,
    local: Vec<f64>,
}

impl DistVec {
    /// Construct a distributed vector for the current rank.
    pub fn new(comm: UniverseComm, row_offset: usize, global_len: usize, local: Vec<f64>) -> Self {
        if row_offset > global_len {
            panic!("row_offset ({row_offset}) must be < global_len ({global_len})");
        }
        if row_offset + local.len() > global_len {
            panic!(
                "local slice length ({}) at offset {} exceeds global length {}",
                local.len(),
                row_offset,
                global_len
            );
        }
        Self {
            comm,
            row_offset,
            global_len,
            local,
        }
    }

    /// Communicator owning this vector.
    pub fn comm(&self) -> &UniverseComm {
        &self.comm
    }

    /// Global length of the distributed vector.
    pub fn global_len(&self) -> usize {
        self.global_len
    }

    /// Row offset of the local slice within the global vector.
    pub fn row_offset(&self) -> usize {
        self.row_offset
    }

    /// Local slice owned by this rank.
    pub fn local_view(&self) -> &[f64] {
        &self.local
    }

    /// Mutable local slice owned by this rank.
    pub fn local_view_mut(&mut self) -> &mut [f64] {
        &mut self.local
    }

    /// Number of entries owned by this rank.
    pub fn local_len(&self) -> usize {
        self.local.len()
    }
}

#[cfg(feature = "backend-faer")]
pub mod block_jacobi_ilu;

#[cfg(feature = "backend-faer")]
pub use block_jacobi_ilu::BlockJacobiLocalPc;

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub use block_jacobi_ilu::{
    build_block_jacobi_ilu_pc, build_block_jacobi_ilut_pc, build_block_jacobi_ilutp_pc,
    build_block_jacobi_pc,
};
