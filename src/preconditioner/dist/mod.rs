//! Distributed preconditioner helpers and wrappers.
//!
//! Local preconditioners operate purely on the data owned by the current rank.
//! This module exposes a small trait for global wrappers and the distributed
//! vector helper that carries only the local slice.

mod coarse;
mod native_plan;

pub use coarse::{DistCoarseRepartition, DistCoarseSolverRoute, DistCoarseStrategy};
pub use native_plan::{
    DistLocalApplyMode, DistRouteDecision, DistRouteDecisionReason, DistRouteFallbackReason,
    DistRoutePolicy, DistRouteResolveInput, DistRouteSelection, resolve_dist_route,
};

use crate::error::KError;
use crate::parallel::UniverseComm;
use crate::preconditioner::PcSide;
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use crate::preconditioner::Preconditioner;
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use crate::preconditioner::asm::{AsmBlockSolver, AsmInnerPc, AsmMode, DistributedAsm, Weighting};
use crate::preconditioner::ilu::IluConfig;
use crate::utils::conditioning::ConditioningOptions;
use std::str::FromStr;
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use std::sync::Mutex;

#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use crate::algebra::scalar::S;
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use crate::matrix::DistCsrOp;
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
use crate::matrix::op::LinOp;

/// Global distributed preconditioner modes exposed to CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalPcKind {
    /// No MPI-level preconditioning.
    None,
    /// Block-Jacobi wrapping a local ILU-like solver.
    BlockJacobi,
    /// Additive Schwarz.
    Asm,
    /// Restricted additive Schwarz (RAS).
    Ras,
}

impl FromStr for GlobalPcKind {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(GlobalPcKind::None),
            "block-jacobi" | "blockjacobi" | "block_jacobi" => Ok(GlobalPcKind::BlockJacobi),
            "asm" => Ok(GlobalPcKind::Asm),
            "ras" => Ok(GlobalPcKind::Ras),
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
    Jacobi,
    Sor,
    Chebyshev,
    Fsai,
    Spai,
}

impl FromStr for LocalPcKind {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ilu" => Ok(LocalPcKind::Ilu),
            "ilut" => Ok(LocalPcKind::Ilut),
            "ilutp" => Ok(LocalPcKind::Ilutp),
            "jacobi" => Ok(LocalPcKind::Jacobi),
            "sor" => Ok(LocalPcKind::Sor),
            "chebyshev" | "cheby" => Ok(LocalPcKind::Chebyshev),
            "fsai" => Ok(LocalPcKind::Fsai),
            "spai" | "approxinv" => Ok(LocalPcKind::Spai),
            other => Err(KError::InvalidInput(format!(
                "Unknown pc_local value: {other}"
            ))),
        }
    }
}

/// Parsed MPI-specific PC options.
///
/// Default policy favors DistCsr-native distributed kernels whenever
/// communicator and operator constraints are satisfied. Adapter routes are
/// treated as explicit fallback paths via `pc_dist_route=adapted`.
#[derive(Clone, Debug)]
pub struct MpiPcOptions {
    pub global_pc: GlobalPcKind,
    pub local_pc: LocalPcKind,
    pub ilu_config: IluConfig,
    pub conditioning: ConditioningOptions,
    pub ilut_fill: usize,
    pub ilut_drop_tol: f64,
    pub ilut_perm_tol: f64,
    pub ilutp_max_fill: usize,
    pub ilutp_drop_tol: f64,
    pub ilutp_perm_tol: f64,
    pub local_apply_mode: DistLocalApplyMode,
    pub route_policy: DistRoutePolicy,
}

impl Default for MpiPcOptions {
    fn default() -> Self {
        Self {
            global_pc: GlobalPcKind::None,
            local_pc: LocalPcKind::Ilu,
            ilu_config: IluConfig::default(),
            conditioning: ConditioningOptions::default(),
            ilut_fill: 10,
            ilut_drop_tol: 1e-4,
            ilut_perm_tol: 0.1,
            ilutp_max_fill: 10,
            ilutp_drop_tol: 1e-4,
            ilutp_perm_tol: 0.1,
            local_apply_mode: DistLocalApplyMode::NativeLocalHalo,
            route_policy: DistRoutePolicy::Native,
        }
    }
}

/// Distributed preconditioners expose an MPI-friendly apply API.
pub trait DistributedPreconditioner: Send + Sync {
    type Scalar;

    /// Apply the distributed preconditioner to a global vector.
    fn apply_global(&self, side: PcSide, x: &mut DistVec<'_>) -> Result<(), KError>;
}

/// Simple distributed vector carrying only the owned local slice.
#[derive(Debug)]
pub struct DistVec<'a> {
    comm: UniverseComm,
    row_offset: usize,
    global_len: usize,
    local: DistVecLocal<'a>,
    scratch: DistVecScratch<'a>,
}

#[derive(Debug)]
enum DistVecLocal<'a> {
    Owned(Vec<f64>),
    Borrowed(&'a mut [f64]),
}

#[derive(Debug)]
enum DistVecScratch<'a> {
    Owned(Vec<f64>),
    Borrowed(&'a mut Vec<f64>),
}

impl DistVec<'_> {
    /// Construct a distributed vector for the current rank.
    pub fn new(comm: UniverseComm, row_offset: usize, global_len: usize, local: Vec<f64>) -> Self {
        Self::with_scratch(comm, row_offset, global_len, local, Vec::new())
    }

    /// Construct a distributed vector with reusable owned work buffers.
    pub fn with_scratch(
        comm: UniverseComm,
        row_offset: usize,
        global_len: usize,
        local: Vec<f64>,
        scratch: Vec<f64>,
    ) -> Self {
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
            local: DistVecLocal::Owned(local),
            scratch: DistVecScratch::Owned(scratch),
        }
    }

    /// Construct a distributed vector that writes directly into a borrowed local slice.
    pub fn from_local_slice<'a>(
        comm: UniverseComm,
        row_offset: usize,
        global_len: usize,
        local: &'a mut [f64],
        scratch: &'a mut Vec<f64>,
    ) -> DistVec<'a> {
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
        DistVec {
            comm,
            row_offset,
            global_len,
            local: DistVecLocal::Borrowed(local),
            scratch: DistVecScratch::Borrowed(scratch),
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
        match &self.local {
            DistVecLocal::Owned(local) => local,
            DistVecLocal::Borrowed(local) => local,
        }
    }

    /// Mutable local slice owned by this rank.
    pub fn local_view_mut(&mut self) -> &mut [f64] {
        match &mut self.local {
            DistVecLocal::Owned(local) => local,
            DistVecLocal::Borrowed(local) => local,
        }
    }

    /// Number of entries owned by this rank.
    pub fn local_len(&self) -> usize {
        self.local_view().len()
    }

    /// Return a mutable scratch slice sized for local operations.
    pub fn scratch_mut(&mut self) -> &mut [f64] {
        let local_len = self.local_len();
        let scratch = match &mut self.scratch {
            DistVecScratch::Owned(scratch) => scratch,
            DistVecScratch::Borrowed(scratch) => scratch,
        };
        if scratch.len() != local_len {
            scratch.resize(local_len, 0.0);
        }
        scratch.as_mut_slice()
    }

    /// Current scratch slice.
    pub fn scratch_view(&self) -> &[f64] {
        match &self.scratch {
            DistVecScratch::Owned(scratch) => scratch,
            DistVecScratch::Borrowed(scratch) => scratch,
        }
    }

    /// Copy local data into the reusable scratch buffer.
    pub fn copy_local_to_scratch(&mut self) {
        let local_len = self.local_len();
        let local_ptr = self.local_view().as_ptr();
        let scratch = self.scratch_mut();
        debug_assert_eq!(scratch.len(), local_len);
        // SAFETY: local and scratch refer to independent buffers managed by this type.
        unsafe {
            std::ptr::copy_nonoverlapping(local_ptr, scratch.as_mut_ptr(), local_len);
        }
    }

    /// Provide immutable scratch input and mutable local output slices.
    pub fn with_scratch_input_local_output<R>(
        &mut self,
        f: impl FnOnce(&[f64], &mut [f64]) -> R,
    ) -> R {
        self.copy_local_to_scratch();
        let len = self.local_len();
        let in_ptr = self.scratch_mut().as_ptr();
        let out_ptr = self.local_view_mut().as_mut_ptr();
        // SAFETY: scratch and local are distinct storage owned by separate fields.
        unsafe {
            let x_local = std::slice::from_raw_parts(in_ptr, len);
            let y_local = std::slice::from_raw_parts_mut(out_ptr, len);
            f(x_local, y_local)
        }
    }
}

/// Builder configuration for distributed preconditioner adapters.
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
#[derive(Clone, Debug)]
pub enum DistPcBuilder {
    BlockJacobi {
        opts: MpiPcOptions,
    },
    Asm {
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        inner_pc: AsmInnerPc,
        weighting: Weighting,
    },
    Ras {
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        inner_pc: AsmInnerPc,
        weighting: Weighting,
    },
}

#[cfg(all(not(feature = "complex"), feature = "mpi"))]
#[derive(Debug)]
struct DistAsmPc {
    inner: DistributedAsm,
}

#[cfg(all(not(feature = "complex"), feature = "mpi"))]
impl DistAsmPc {
    fn new(mut inner: DistributedAsm, dist_op: &DistCsrOp) -> Result<Self, KError> {
        inner.setup(dist_op)?;
        Ok(Self { inner })
    }
}

#[cfg(all(not(feature = "complex"), feature = "mpi"))]
impl DistributedPreconditioner for DistAsmPc {
    type Scalar = f64;

    fn apply_global(&self, side: PcSide, x: &mut DistVec<'_>) -> Result<(), KError> {
        x.with_scratch_input_local_output(|x_local, y_local| {
            self.inner.apply(side, x_local, y_local)
        })?;
        Ok(())
    }
}

/// Adapter bridging a `DistributedPreconditioner` to the `Preconditioner` trait.
#[cfg(all(not(feature = "complex"), feature = "mpi"))]
pub struct DistPcAdapter {
    comm: UniverseComm,
    row_offset: usize,
    global_len: usize,
    local_len: usize,
    inner: Box<dyn DistributedPreconditioner<Scalar = f64>>,
    builder: DistPcBuilder,
    workspace: Mutex<Vec<f64>>,
}

#[cfg(all(not(feature = "complex"), feature = "mpi"))]
impl DistPcAdapter {
    pub fn build(dist_op: &DistCsrOp, builder: DistPcBuilder) -> Result<Self, KError> {
        let inner = build_dist_pc(dist_op, &builder)?;
        Ok(Self::new(dist_op, inner, builder))
    }

    fn new(
        dist_op: &DistCsrOp,
        inner: Box<dyn DistributedPreconditioner<Scalar = f64>>,
        builder: DistPcBuilder,
    ) -> Self {
        let comm = dist_op.comm();
        let row_offset = dist_op.local_row_offset();
        let global_len = dist_op.n_global;
        let local_len = dist_op.local_nrows();
        Self {
            comm,
            row_offset,
            global_len,
            local_len,
            inner,
            builder,
            workspace: Mutex::new(Vec::new()),
        }
    }

    fn rebuild(&mut self, dist_op: &DistCsrOp) -> Result<(), KError> {
        self.inner = build_dist_pc(dist_op, &self.builder)?;
        self.comm = dist_op.comm();
        self.row_offset = dist_op.local_row_offset();
        self.global_len = dist_op.n_global;
        self.local_len = dist_op.local_nrows();
        Ok(())
    }
}

#[cfg(all(not(feature = "complex"), feature = "mpi"))]
impl Preconditioner for DistPcAdapter {
    fn dims(&self) -> (usize, usize) {
        (self.local_len, self.local_len)
    }

    fn setup(&mut self, op: &dyn crate::matrix::op::LinOp<S = S>) -> Result<(), KError> {
        let dist_op = op
            .as_any()
            .downcast_ref::<DistCsrOp>()
            .ok_or_else(|| KError::InvalidInput("distributed PC requires a DistCsrOp".into()))?;
        self.rebuild(dist_op)
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != self.local_len || y.len() != self.local_len {
            return Err(KError::InvalidInput(
                "distributed PC apply length mismatch".into(),
            ));
        }
        y.copy_from_slice(x);
        let mut workspace = self
            .workspace
            .lock()
            .expect("distributed PC workspace mutex poisoned");
        let mut dist_vec = DistVec::from_local_slice(
            self.comm.clone(),
            self.row_offset,
            self.global_len,
            y,
            &mut workspace,
        );
        self.inner.apply_global(side, &mut dist_vec)?;
        Ok(())
    }

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply(side, x, y)
    }

    fn distributed_support(&self) -> crate::preconditioner::PcDistributedSupport {
        crate::preconditioner::PcDistributedSupport::Distributed
    }
}

#[cfg(all(not(feature = "complex"), feature = "mpi"))]
fn build_dist_pc(
    dist_op: &DistCsrOp,
    builder: &DistPcBuilder,
) -> Result<Box<dyn DistributedPreconditioner<Scalar = f64>>, KError> {
    match builder {
        DistPcBuilder::BlockJacobi { opts } => {
            #[cfg(feature = "backend-faer")]
            {
                let pc = build_block_jacobi_pc(dist_op, opts)?.ok_or_else(|| {
                    KError::InvalidInput("block-Jacobi PC not constructed".into())
                })?;
                Ok(pc)
            }
            #[cfg(not(feature = "backend-faer"))]
            {
                let _ = dist_op;
                let _ = opts;
                Err(KError::Unsupported(
                    "block-Jacobi distributed PC requires backend-faer".into(),
                ))
            }
        }
        DistPcBuilder::Asm {
            overlap,
            subdomain_hint,
            block_solver,
            inner_pc,
            weighting,
        } => {
            let asm = DistributedAsm::new(
                *overlap,
                *subdomain_hint,
                *block_solver,
                *inner_pc,
                AsmMode::ASM,
                *weighting,
                DistCoarseStrategy::None,
            );
            Ok(Box::new(DistAsmPc::new(asm, dist_op)?))
        }
        DistPcBuilder::Ras {
            overlap,
            subdomain_hint,
            block_solver,
            inner_pc,
            weighting,
        } => {
            let asm = DistributedAsm::new_ras(
                *overlap,
                *subdomain_hint,
                *block_solver,
                *inner_pc,
                *weighting,
            );
            Ok(Box::new(DistAsmPc::new(asm, dist_op)?))
        }
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
