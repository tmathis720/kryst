#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::algebra::scalar::{KrystScalar, S};
use crate::error::KError;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::dist::halo::HaloPlan;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::dist_csr::DistCsrOp;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::approxinv_csr::{ApproxInvKind, ApproxInvParams, FsaiCsr, SpaiCsr};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::chebyshev::ChebyshevPc;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::ilu::{Ilu, IluConfig};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::ilut::RowFilterPreconditioner;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::ilutp::Ilutp;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::jacobi::Jacobi;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::legacy::Preconditioner as LegacyPreconditioner;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::sor::{MatSorType, SorPc};
use crate::preconditioner::{LocalPreconditioner, PcSide, Preconditioner as ObjPreconditioner};
use crate::utils::conditioning::ConditioningOptions;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::collections::BTreeMap;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::op::CsrOp;

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use super::{DistLocalApplyMode, GlobalPcKind, LocalPcKind, MpiPcOptions};
use super::{DistVec, DistributedPreconditioner};

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone)]
struct NativeCouplingPlan {
    halo: Arc<HaloPlan>,
    remote_entries_by_row: Vec<Vec<(usize, f64)>>,
    diag_inv: Vec<f64>,
    omega: f64,
    coarse_weight: f64,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
#[derive(Clone, Copy, Debug)]
enum NativePlanBuildReason {
    ReusedOperatorHalo,
    BuiltDedicatedHalo,
    IncompatibleHaloIndex,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl NativePlanBuildReason {
    fn code(self) -> &'static str {
        match self {
            Self::ReusedOperatorHalo => "reused_operator_halo",
            Self::BuiltDedicatedHalo => "built_dedicated_halo",
            Self::IncompatibleHaloIndex => "incompatible_halo_index",
        }
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl NativeCouplingPlan {
    fn from_dist_op(
        dist_op: &DistCsrOp,
        omega: f64,
        local_apply_mode: DistLocalApplyMode,
    ) -> Result<Self, KError> {
        let local = dist_op.local_matrix();
        let row_start = dist_op.local_row_offset();
        let row_end = row_start + dist_op.local_nrows();
        let part = dist_op.row_partition();
        let rank = dist_op.comm().rank();

        let mut recv_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut remote_entries_by_row = vec![Vec::new(); local.nrows()];
        let mut diag_inv = vec![1.0; local.nrows()];

        for row in 0..local.nrows() {
            let mut diag = None;
            for idx in local.row_ptr()[row]..local.row_ptr()[row + 1] {
                let gcol = local.col_idx()[idx];
                if gcol == row_start + row {
                    diag = Some(local.values()[idx]);
                    continue;
                }
                if gcol < row_start || gcol >= row_end {
                    let owner = owner_of(gcol, part.as_ref());
                    if owner == rank {
                        continue;
                    }
                    recv_map.entry(owner).or_default().push(gcol);
                    remote_entries_by_row[row].push((gcol, local.values()[idx]));
                }
            }
            if let Some(d) = diag
                && d.abs() > 1e-14
            {
                diag_inv[row] = 1.0 / d;
            }
        }

        for cols in recv_map.values_mut() {
            cols.sort_unstable();
            cols.dedup();
        }

        let (halo, reason) = Self::build_halo_plan(dist_op, &recv_map, row_start, row_end, part)?;
        log::debug!("native coupling halo plan reason_code={}", reason.code());

        Ok(Self {
            halo: Arc::new(halo),
            remote_entries_by_row,
            diag_inv,
            omega,
            coarse_weight: if matches!(local_apply_mode, DistLocalApplyMode::NativeHybrid) {
                0.1
            } else {
                0.0
            },
        })
    }

    fn build_halo_plan(
        dist_op: &DistCsrOp,
        recv_map: &BTreeMap<usize, Vec<usize>>,
        row_start: usize,
        row_end: usize,
        part: Arc<Vec<usize>>,
    ) -> Result<(HaloPlan, NativePlanBuildReason), KError> {
        let op_index = dist_op.halo_index();
        if Self::halo_index_compatible(op_index.as_ref(), recv_map) {
            return Ok((
                HaloPlan::from_shared_index(op_index),
                NativePlanBuildReason::ReusedOperatorHalo,
            ));
        }

        let reason = NativePlanBuildReason::IncompatibleHaloIndex;
        log::debug!(
            "native coupling halo reuse skipped reason_code={}",
            reason.code()
        );

        Ok((
            HaloPlan::new(dist_op.comm(), part, row_start, row_end, recv_map.clone())?,
            NativePlanBuildReason::BuiltDedicatedHalo,
        ))
    }

    fn halo_index_compatible(
        op_index: &crate::matrix::dist::halo::HaloIndexPlan,
        recv_map: &BTreeMap<usize, Vec<usize>>,
    ) -> bool {
        if op_index.recv_map.len() != recv_map.len() {
            return false;
        }
        recv_map.iter().all(|(nbr, cols)| {
            if let Some(op_cols) = op_index.recv_map.get(nbr) {
                let mut lhs = cols.clone();
                let mut rhs = op_cols.clone();
                lhs.sort_unstable();
                lhs.dedup();
                rhs.sort_unstable();
                rhs.dedup();
                lhs == rhs
            } else {
                false
            }
        })
    }

    fn apply_remote_correction(&self, x_local: &[f64], y_local: &mut [f64]) {
        if self.halo.index.n_ghost == 0 {
            return;
        }
        let req = self.halo.post_halo(x_local);
        self.halo.complete_halo(req);
        let ghost = self.halo.ghost_slice_ref();
        for (row, entries) in self.remote_entries_by_row.iter().enumerate() {
            if entries.is_empty() {
                continue;
            }
            let mut remote_sum = 0.0;
            for &(gcol, value) in entries {
                if let Some(&gidx) = self.halo.index.ghost_index_of.get(&gcol) {
                    remote_sum += value * ghost[gidx];
                } else {
                    log::error!(
                        "native coupling halo mismatch reason_code=missing_ghost_column gcol={gcol}"
                    );
                }
            }
            y_local[row] -= self.omega * self.diag_inv[row] * remote_sum;
        }

        if self.coarse_weight != 0.0 && !x_local.is_empty() {
            let comm = &self.halo.index.comm;
            let local_sum: f64 = x_local.iter().copied().sum();
            let global_sum = comm.all_reduce_f64(local_sum);
            let global_n = comm.all_reduce_f64(x_local.len() as f64).max(1.0);
            let coarse_avg = global_sum / global_n;
            for yi in y_local.iter_mut() {
                *yi += self.coarse_weight * coarse_avg;
            }
        }
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn maybe_native_plan(
    dist_op: &DistCsrOp,
    local_apply_mode: DistLocalApplyMode,
    supports_native: bool,
    local_pc_name: &str,
) -> Result<Option<NativeCouplingPlan>, KError> {
    if !local_apply_mode.is_distributed_native() {
        return Ok(None);
    }
    if !supports_native {
        if local_apply_mode.requires_native() {
            return Err(KError::InvalidInput(format!(
                "pc_dist_local_apply=strict requested but pc_local={local_pc_name} only supports wrapped_local mode"
            )));
        }
        log::warn!(
            "Distributed native route unavailable for pc_local={local_pc_name}; falling back to wrapped_local compatibility adapter"
        );
        return Ok(None);
    }
    Ok(Some(NativeCouplingPlan::from_dist_op(
        dist_op,
        1.0,
        local_apply_mode,
    )?))
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn strict_native_unavailable(local_pc_name: &str) -> KError {
    KError::InvalidInput(format!(
        "pc_dist_local_apply=strict requested but pc_local={local_pc_name} only supports wrapped_local mode"
    ))
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
fn build_obj_block_pc<L>(
    dist_op: &DistCsrOp,
    mut local_pc: L,
    local_apply_mode: DistLocalApplyMode,
    supports_native: bool,
    local_pc_name: &str,
) -> Result<BlockJacobiObjPc, KError>
where
    L: ObjPreconditioner + 'static,
{
    let local = Arc::new(dist_op.local_block_csr());
    let op = CsrOp::new(local);
    ObjPreconditioner::setup(&mut local_pc, &op)?;
    let native_plan = if supports_native {
        maybe_native_plan(dist_op, local_apply_mode, true, local_pc_name)?
    } else if local_apply_mode.requires_native() {
        return Err(strict_native_unavailable(local_pc_name));
    } else {
        None
    };
    Ok(BlockJacobiObjPc::new(
        dist_op.comm(),
        Box::new(local_pc),
        dist_op.local_row_offset(),
        dist_op.local_nrows(),
        native_plan,
    ))
}

fn owner_of(gcol: usize, row_part: &[usize]) -> usize {
    let mut lo = 0usize;
    let mut hi = row_part.len().saturating_sub(2);
    while lo <= hi {
        let mid = (lo + hi) / 2;
        if gcol < row_part[mid + 1] {
            if gcol >= row_part[mid] {
                return mid;
            }
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Distributed block-Jacobi wrapper around a local ILU-like preconditioner.
///
/// Each rank applies the local preconditioner to its owned slice without doing
/// any MPI communication. The surrounding solver is responsible for distributing
/// the vector consistently.
pub struct BlockJacobiLocalPc<LPC>
where
    LPC: LocalPreconditioner<f64>,
{
    comm: UniverseComm,
    local_pc: LPC,
    row_offset: usize,
    n_local: usize,
    native_plan: Option<NativeCouplingPlan>,
}

impl<LPC> BlockJacobiLocalPc<LPC>
where
    LPC: LocalPreconditioner<f64>,
{
    /// Construct a new distributed block-Jacobi preconditioner.
    fn new(
        comm: UniverseComm,
        local_pc: LPC,
        row_offset: usize,
        native_plan: Option<NativeCouplingPlan>,
    ) -> Self {
        let (n_local, _) = local_pc.dims();
        Self {
            comm,
            local_pc,
            row_offset,
            n_local,
            native_plan,
        }
    }

    /// Communicator used by this preconditioner.
    pub fn comm(&self) -> &UniverseComm {
        &self.comm
    }

    /// Global row offset for this block.
    pub fn row_offset(&self) -> usize {
        self.row_offset
    }

    /// Number of rows owned locally.
    pub fn n_local(&self) -> usize {
        self.n_local
    }
}

impl<LPC> DistributedPreconditioner for BlockJacobiLocalPc<LPC>
where
    LPC: LocalPreconditioner<f64>,
{
    type Scalar = f64;

    fn apply_global(&self, side: PcSide, x: &mut DistVec<'_>) -> Result<(), KError> {
        debug_assert_eq!(x.row_offset(), self.row_offset);
        debug_assert_eq!(x.local_len(), self.n_local);
        if self.n_local == 0 {
            return Ok(());
        }

        debug_assert!(matches!(side, PcSide::Left));
        x.with_scratch_input_local_output(|x_local, y_local| {
            self.local_pc.apply_local(x_local, y_local)?;
            if let Some(plan) = &self.native_plan {
                plan.apply_remote_correction(x_local, y_local);
            }
            Ok::<(), KError>(())
        })?;
        Ok(())
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub struct BlockJacobiObjPc {
    comm: UniverseComm,
    local_pc: Box<dyn ObjPreconditioner>,
    row_offset: usize,
    n_local: usize,
    native_plan: Option<NativeCouplingPlan>,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl BlockJacobiObjPc {
    fn new(
        comm: UniverseComm,
        local_pc: Box<dyn ObjPreconditioner>,
        row_offset: usize,
        n_local: usize,
        native_plan: Option<NativeCouplingPlan>,
    ) -> Self {
        Self {
            comm,
            local_pc,
            row_offset,
            n_local,
            native_plan,
        }
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl DistributedPreconditioner for BlockJacobiObjPc {
    type Scalar = f64;

    fn apply_global(&self, side: PcSide, x: &mut DistVec<'_>) -> Result<(), KError> {
        let _ = &self.comm;
        debug_assert_eq!(x.row_offset(), self.row_offset);
        debug_assert_eq!(x.local_len(), self.n_local);
        if self.n_local == 0 {
            return Ok(());
        }
        x.with_scratch_input_local_output(|x_local, y_local| {
            self.local_pc.apply(side, x_local, y_local)?;
            if let Some(plan) = &self.native_plan {
                plan.apply_remote_correction(x_local, y_local);
            }
            Ok::<(), KError>(())
        })?;
        Ok(())
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub fn build_block_jacobi_ilu_pc(
    dist_op: &DistCsrOp,
    config: &IluConfig,
    local_apply_mode: DistLocalApplyMode,
) -> Result<BlockJacobiLocalPc<Ilu>, KError> {
    let mut ilu = Ilu::new_with_config(config.clone())?;
    let local = dist_op.local_block_dense();
    LegacyPreconditioner::setup(&mut ilu, &local)?;
    Ok(BlockJacobiLocalPc::new(
        dist_op.comm(),
        ilu,
        dist_op.local_row_offset(),
        maybe_native_plan(dist_op, local_apply_mode, true, "ilu")?,
    ))
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub fn build_block_jacobi_ilut_pc(
    dist_op: &DistCsrOp,
    fill: usize,
    drop_tol: f64,
    conditioning: ConditioningOptions,
    local_apply_mode: DistLocalApplyMode,
) -> Result<BlockJacobiLocalPc<RowFilterPreconditioner>, KError> {
    let mut pc = RowFilterPreconditioner::new(fill, S::from_real(drop_tol));
    pc.set_conditioning(conditioning);
    let local = dist_op.local_block_dense();
    LegacyPreconditioner::setup(&mut pc, &local)?;
    Ok(BlockJacobiLocalPc::new(
        dist_op.comm(),
        pc,
        dist_op.local_row_offset(),
        maybe_native_plan(dist_op, local_apply_mode, true, "ilut")?,
    ))
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub fn build_block_jacobi_ilutp_pc(
    dist_op: &DistCsrOp,
    max_fill: usize,
    drop_tol: f64,
    perm_tol: f64,
    conditioning: ConditioningOptions,
    local_apply_mode: DistLocalApplyMode,
) -> Result<BlockJacobiLocalPc<Ilutp>, KError> {
    let mut pc = Ilutp::with_params(max_fill, drop_tol, perm_tol);
    pc.set_conditioning(conditioning);
    let local = dist_op.local_block_dense();
    LegacyPreconditioner::setup(&mut pc, &local)?;
    Ok(BlockJacobiLocalPc::new(
        dist_op.comm(),
        pc,
        dist_op.local_row_offset(),
        maybe_native_plan(dist_op, local_apply_mode, true, "ilutp")?,
    ))
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub fn build_block_jacobi_pc(
    dist_op: &DistCsrOp,
    opts: &MpiPcOptions,
) -> Result<Option<Box<dyn DistributedPreconditioner<Scalar = f64>>>, KError> {
    match opts.global_pc {
        GlobalPcKind::None => Ok(None),
        GlobalPcKind::BlockJacobi => {
            let wrapper: Box<dyn DistributedPreconditioner<Scalar = f64>> = match opts.local_pc {
                LocalPcKind::Ilu => Box::new(build_block_jacobi_ilu_pc(
                    dist_op,
                    &opts.ilu_config,
                    opts.local_apply_mode,
                )?),
                LocalPcKind::Ilut => Box::new(build_block_jacobi_ilut_pc(
                    dist_op,
                    opts.ilut_fill,
                    opts.ilut_drop_tol,
                    opts.conditioning.clone(),
                    opts.local_apply_mode,
                )?),
                LocalPcKind::Ilutp => Box::new(build_block_jacobi_ilutp_pc(
                    dist_op,
                    opts.ilutp_max_fill,
                    opts.ilutp_drop_tol,
                    opts.ilutp_perm_tol,
                    opts.conditioning.clone(),
                    opts.local_apply_mode,
                )?),
                LocalPcKind::Sor => Box::new(build_obj_block_pc(
                    dist_op,
                    SorPc::new(1.0, 1, MatSorType::SYMMETRIC_SWEEP, 0.0),
                    opts.local_apply_mode,
                    true,
                    "sor",
                )?),
                LocalPcKind::Chebyshev => Box::new(build_obj_block_pc(
                    dist_op,
                    ChebyshevPc::new(2, 1e-2, 1.0),
                    opts.local_apply_mode,
                    true,
                    "chebyshev",
                )?),
                LocalPcKind::Jacobi => {
                    let mut jacobi = Jacobi::new();
                    let local = Arc::new(dist_op.local_block_csr());
                    let op = CsrOp::new(local);
                    ObjPreconditioner::setup(&mut jacobi, &op)?;
                    Box::new(BlockJacobiLocalPc::new(
                        dist_op.comm(),
                        jacobi,
                        dist_op.local_row_offset(),
                        maybe_native_plan(dist_op, opts.local_apply_mode, true, "jacobi")?,
                    ))
                }
                LocalPcKind::Fsai => Box::new(build_obj_block_pc(
                    dist_op,
                    FsaiCsr::new_with_params(ApproxInvParams {
                        kind: ApproxInvKind::FSAI,
                        ..ApproxInvParams::default()
                    }),
                    opts.local_apply_mode,
                    false,
                    "fsai",
                )?),
                LocalPcKind::Spai => Box::new(build_obj_block_pc(
                    dist_op,
                    SpaiCsr::new_with_params(ApproxInvParams {
                        kind: ApproxInvKind::SPAI,
                        ..ApproxInvParams::default()
                    }),
                    opts.local_apply_mode,
                    false,
                    "spai",
                )?),
            };
            Ok(Some(wrapper))
        }
        GlobalPcKind::Asm | GlobalPcKind::Ras => Err(KError::NotImplemented(
            "block-Jacobi builder does not apply to ASM/RAS distributed preconditioners".into(),
        )),
    }
}

// Builder helpers and distributed wrappers are defined below.
