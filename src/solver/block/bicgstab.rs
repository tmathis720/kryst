//! Block BiCGSTAB solver (placeholder).

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::SolveStats;
use std::any::Any;

use super::BlockKrylovOptions;

/// Temporary stub for the block BiCGSTAB solver.
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
        _a: &dyn crate::matrix::op::LinOp<S = f64>,
        _pc: Option<&mut dyn Preconditioner>,
        _b: &[f64],
        _x: &mut [f64],
        _pc_side: PcSide,
        _comm: &UniverseComm,
        _monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        _work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        Err(KError::NotImplemented(
            "block BiCGSTAB solver is not yet implemented".into(),
        ))
    }
}
