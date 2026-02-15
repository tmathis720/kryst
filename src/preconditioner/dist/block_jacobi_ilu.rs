#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::algebra::scalar::{KrystScalar, S};
use crate::error::KError;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::dist_csr::DistCsrOp;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
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
use crate::preconditioner::legacy::Preconditioner as LegacyPreconditioner;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::sor::{MatSorType, SorPc};
use crate::preconditioner::{LocalPreconditioner, PcSide, Preconditioner as ObjPreconditioner};
use crate::utils::conditioning::ConditioningOptions;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use std::sync::Arc;

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::op::CsrOp;

use super::{DistVec, DistributedPreconditioner};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use super::{GlobalPcKind, LocalPcKind, MpiPcOptions};

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
}

impl<LPC> BlockJacobiLocalPc<LPC>
where
    LPC: LocalPreconditioner<f64>,
{
    /// Construct a new distributed block-Jacobi preconditioner.
    pub fn new(comm: UniverseComm, local_pc: LPC, row_offset: usize) -> Self {
        let (n_local, _) = local_pc.dims();
        Self {
            comm,
            local_pc,
            row_offset,
            n_local,
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

    fn apply_global(&self, side: PcSide, x: &mut DistVec) -> Result<(), KError> {
        debug_assert_eq!(x.row_offset(), self.row_offset);
        debug_assert_eq!(x.local_len(), self.n_local);
        if self.n_local == 0 {
            return Ok(());
        }

        debug_assert!(matches!(side, PcSide::Left));
        let x_local = x.local_view();
        let mut y_local = vec![0.0; self.n_local];
        self.local_pc.apply_local(x_local, &mut y_local)?;
        x.local_view_mut().copy_from_slice(&y_local);
        Ok(())
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub struct BlockJacobiObjPc {
    comm: UniverseComm,
    local_pc: Box<dyn ObjPreconditioner>,
    row_offset: usize,
    n_local: usize,
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl BlockJacobiObjPc {
    pub fn new(
        comm: UniverseComm,
        local_pc: Box<dyn ObjPreconditioner>,
        row_offset: usize,
        n_local: usize,
    ) -> Self {
        Self {
            comm,
            local_pc,
            row_offset,
            n_local,
        }
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
impl DistributedPreconditioner for BlockJacobiObjPc {
    type Scalar = f64;

    fn apply_global(&self, side: PcSide, x: &mut DistVec) -> Result<(), KError> {
        let _ = &self.comm;
        debug_assert_eq!(x.row_offset(), self.row_offset);
        debug_assert_eq!(x.local_len(), self.n_local);
        if self.n_local == 0 {
            return Ok(());
        }
        let mut y_local = vec![0.0; self.n_local];
        self.local_pc.apply(side, x.local_view(), &mut y_local)?;
        x.local_view_mut().copy_from_slice(&y_local);
        Ok(())
    }
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub fn build_block_jacobi_ilu_pc(
    dist_op: &DistCsrOp,
    config: &IluConfig,
) -> Result<BlockJacobiLocalPc<Ilu>, KError> {
    let mut ilu = Ilu::new_with_config(config.clone())?;
    let local = dist_op.local_block_dense();
    LegacyPreconditioner::setup(&mut ilu, &local)?;
    Ok(BlockJacobiLocalPc::new(
        dist_op.comm(),
        ilu,
        dist_op.local_row_offset(),
    ))
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub fn build_block_jacobi_ilut_pc(
    dist_op: &DistCsrOp,
    fill: usize,
    drop_tol: f64,
    conditioning: ConditioningOptions,
) -> Result<BlockJacobiLocalPc<RowFilterPreconditioner>, KError> {
    let mut pc = RowFilterPreconditioner::new(fill, S::from_real(drop_tol));
    pc.set_conditioning(conditioning);
    let local = dist_op.local_block_dense();
    LegacyPreconditioner::setup(&mut pc, &local)?;
    Ok(BlockJacobiLocalPc::new(
        dist_op.comm(),
        pc,
        dist_op.local_row_offset(),
    ))
}

#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
pub fn build_block_jacobi_ilutp_pc(
    dist_op: &DistCsrOp,
    max_fill: usize,
    drop_tol: f64,
    perm_tol: f64,
    conditioning: ConditioningOptions,
) -> Result<BlockJacobiLocalPc<Ilutp>, KError> {
    let mut pc = Ilutp::with_params(max_fill, drop_tol, perm_tol);
    pc.set_conditioning(conditioning);
    let local = dist_op.local_block_dense();
    LegacyPreconditioner::setup(&mut pc, &local)?;
    Ok(BlockJacobiLocalPc::new(
        dist_op.comm(),
        pc,
        dist_op.local_row_offset(),
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
                LocalPcKind::Ilu => Box::new(build_block_jacobi_ilu_pc(dist_op, &opts.ilu_config)?),
                LocalPcKind::Ilut => Box::new(build_block_jacobi_ilut_pc(
                    dist_op,
                    opts.ilut_fill,
                    opts.ilut_drop_tol,
                    opts.conditioning.clone(),
                )?),
                LocalPcKind::Ilutp => Box::new(build_block_jacobi_ilutp_pc(
                    dist_op,
                    opts.ilutp_max_fill,
                    opts.ilutp_drop_tol,
                    opts.ilutp_perm_tol,
                    opts.conditioning.clone(),
                )?),
                LocalPcKind::Sor => {
                    let mut pc = SorPc::new(1.0, 1, MatSorType::SYMMETRIC_SWEEP, 0.0);
                    let local = Arc::new(dist_op.local_block_csr());
                    let op = CsrOp::new(local);
                    ObjPreconditioner::setup(&mut pc, &op)?;
                    Box::new(BlockJacobiObjPc::new(
                        dist_op.comm(),
                        Box::new(pc),
                        dist_op.local_row_offset(),
                        dist_op.local_nrows(),
                    ))
                }
                LocalPcKind::Chebyshev => {
                    let mut pc = ChebyshevPc::new(2, 1e-2, 1.0);
                    let local = Arc::new(dist_op.local_block_csr());
                    let op = CsrOp::new(local);
                    ObjPreconditioner::setup(&mut pc, &op)?;
                    Box::new(BlockJacobiObjPc::new(
                        dist_op.comm(),
                        Box::new(pc),
                        dist_op.local_row_offset(),
                        dist_op.local_nrows(),
                    ))
                }
                LocalPcKind::Fsai => {
                    let mut pc = FsaiCsr::new_with_params(ApproxInvParams {
                        kind: ApproxInvKind::FSAI,
                        ..ApproxInvParams::default()
                    });
                    let local = Arc::new(dist_op.local_block_csr());
                    let op = CsrOp::new(local);
                    ObjPreconditioner::setup(&mut pc, &op)?;
                    Box::new(BlockJacobiObjPc::new(
                        dist_op.comm(),
                        Box::new(pc),
                        dist_op.local_row_offset(),
                        dist_op.local_nrows(),
                    ))
                }
                LocalPcKind::Spai => {
                    let mut pc = SpaiCsr::new_with_params(ApproxInvParams {
                        kind: ApproxInvKind::SPAI,
                        ..ApproxInvParams::default()
                    });
                    let local = Arc::new(dist_op.local_block_csr());
                    let op = CsrOp::new(local);
                    ObjPreconditioner::setup(&mut pc, &op)?;
                    Box::new(BlockJacobiObjPc::new(
                        dist_op.comm(),
                        Box::new(pc),
                        dist_op.local_row_offset(),
                        dist_op.local_nrows(),
                    ))
                }
            };
            Ok(Some(wrapper))
        }
        GlobalPcKind::Asm | GlobalPcKind::Ras => Err(KError::NotImplemented(
            "block-Jacobi builder does not apply to ASM/RAS distributed preconditioners".into(),
        )),
    }
}

// Builder helpers and distributed wrappers are defined below.
