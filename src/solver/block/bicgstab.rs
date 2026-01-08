//! Block BiCGSTAB solver.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::solver::MonitorCallback;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::ops::wrap::as_s_op;
use crate::solver::LinearSolver;
use crate::solver::bicgstab::BiCgStabSolver;
use crate::utils::convergence::{ConvergedReason, SolveStats, SolverCounters};
use std::any::Any;

use super::BlockKrylovOptions;
use super::block_vec::BlockVec;

/// Block BiCGSTAB solver for multiple right-hand sides.
pub struct BlockBicgstabSolver {
    pub options: BlockKrylovOptions,
}

impl BlockBicgstabSolver {
    pub fn new(options: BlockKrylovOptions) -> Self {
        Self { options }
    }
}

impl LinearSolver for BlockBicgstabSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn solve(
        &mut self,
        a: &dyn crate::matrix::op::LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        if pc.is_some() {
            return Err(KError::Unsupported(
                "block BiCGSTAB preconditioning is not implemented",
            ));
        }
        let (nrows, ncols) = a.dims();
        if nrows != ncols {
            return Err(KError::InvalidInput(
                "block BiCGSTAB requires a square operator".into(),
            ));
        }
        let p = self.options.block_size;
        if p == 0 {
            return Err(KError::InvalidInput(
                "block BiCGSTAB requires a positive block size".into(),
            ));
        }
        let expected_len = ncols.saturating_mul(p);
        if !(b.len() == expected_len || (p == 1 && b.len() == ncols)) {
            return Err(KError::InvalidInput(
                "block BiCGSTAB expects b to be column-major with block_size columns".into(),
            ));
        }
        if !(x.len() == expected_len || (p == 1 && x.len() == ncols)) {
            return Err(KError::InvalidInput(
                "block BiCGSTAB expects x to be column-major with block_size columns".into(),
            ));
        }

        let mut local_ws = Workspace::default();
        let work = work.unwrap_or(&mut local_ws);

        // Bridge dyn LinOp<S=f64> -> KLinOp via F64AsSOp wrapper.
        let op = as_s_op(a);

        let mut b_block = BlockVec::new(ncols, p);
        fill_block_from_slice(&mut b_block, b)?;
        let mut x_block = BlockVec::new(ncols, p);
        fill_block_from_slice(&mut x_block, x)?;

        let mut max_iters = 0usize;
        let mut max_residual: f64 = 0.0;
        let mut reason = ConvergedReason::ConvergedRtol;
        let mut counters = SolverCounters::default();

        for col in 0..p {
            let mut solver = BiCgStabSolver::new(self.options.rtol, self.options.max_iters);
            solver.atol = self.options.atol;
            solver.dtol = self.options.dtol;
            let stats = solver.solve(
                &op,
                None,
                b_block.col(col),
                x_block.col_mut(col),
                pc_side,
                comm,
                monitors,
                Some(work),
            )?;
            max_iters = max_iters.max(stats.iterations);
            max_residual = max_residual.max(stats.final_residual);
            counters.num_global_reductions += stats.counters.num_global_reductions;
            counters.residual_replacements += stats.counters.residual_replacements;
            reason = combine_reason(reason, stats.reason);
        }

        write_block_to_slice(&x_block, x)?;
        Ok(SolveStats::new(max_iters, max_residual, reason).with_counters(counters))
    }
}

fn fill_block_from_slice(block: &mut BlockVec, data: &[f64]) -> Result<(), KError> {
    let n = block.nrows();
    let p = block.ncols();
    if data.len() == n {
        if p != 1 {
            return Err(KError::InvalidInput(
                "block BiCGSTAB expects a full block for block_size > 1".into(),
            ));
        }
        copy_real_block(data, block.as_mut_slice());
        return Ok(());
    }
    if data.len() != n * p {
        return Err(KError::InvalidInput(
            "block BiCGSTAB expects column-major block storage".into(),
        ));
    }
    copy_real_block(data, block.as_mut_slice());
    Ok(())
}

fn write_block_to_slice(block: &BlockVec, data: &mut [f64]) -> Result<(), KError> {
    let n = block.nrows();
    let p = block.ncols();
    if data.len() == n && p == 1 {
        copy_block_to_real(block.as_slice(), data);
        return Ok(());
    }
    if data.len() != n * p {
        return Err(KError::InvalidInput(
            "block BiCGSTAB expects column-major block storage".into(),
        ));
    }
    copy_block_to_real(block.as_slice(), data);
    Ok(())
}

#[cfg(feature = "complex")]
fn copy_real_block(src: &[f64], dst: &mut [S]) {
    crate::algebra::bridge::copy_real_into_scalar(src, dst);
}

#[cfg(not(feature = "complex"))]
fn copy_real_block(src: &[f64], dst: &mut [S]) {
    dst.copy_from_slice(src);
}

#[cfg(feature = "complex")]
fn copy_block_to_real(src: &[S], dst: &mut [f64]) {
    crate::algebra::bridge::copy_scalar_to_real_in(src, dst);
}

#[cfg(not(feature = "complex"))]
fn copy_block_to_real(src: &[S], dst: &mut [f64]) {
    dst.copy_from_slice(src);
}

fn combine_reason(current: ConvergedReason, next: ConvergedReason) -> ConvergedReason {
    use ConvergedReason::*;
    match (current, next) {
        (DivergedDtol, _) | (_, DivergedDtol) => DivergedDtol,
        (DivergedMaxIts, _) | (_, DivergedMaxIts) => DivergedMaxIts,
        (ConvergedAtol, ConvergedAtol) => ConvergedAtol,
        (ConvergedAtol, ConvergedRtol) | (ConvergedRtol, ConvergedAtol) => ConvergedRtol,
        (ConvergedRtol, ConvergedRtol) => ConvergedRtol,
        (Continued, other) => other,
        (other, Continued) => other,
        _ => ConvergedRtol,
    }
}
