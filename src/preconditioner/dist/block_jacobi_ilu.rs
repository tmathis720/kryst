#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::algebra::scalar::{KrystScalar, S};
use crate::error::KError;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::dist_csr::DistCsrOp;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::ilu::{Ilu, IluConfig};
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::ilut::RowFilterPreconditioner;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::ilutp::Ilutp;
#[cfg(all(feature = "backend-faer", not(feature = "complex")))]
use crate::preconditioner::legacy::Preconditioner as LegacyPreconditioner;
use crate::preconditioner::{LocalPreconditioner, PcSide};
use crate::utils::conditioning::ConditioningOptions;

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
            };
            Ok(Some(wrapper))
        }
        GlobalPcKind::Asm => Err(KError::NotImplemented(
            "ASM distributed preconditioner is not implemented".into(),
        )),
    }
}

// Builder helpers and distributed wrappers are defined below.
