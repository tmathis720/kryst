//! Additive Schwarz preconditioner (ASM) module.

mod serial;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
mod comm_plan;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
mod subdomain;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
mod distributed;

pub use serial::*;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
pub use distributed::DistributedAsm;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::LinOp;
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::parallel::Comm;
use crate::preconditioner::{PcSide, Preconditioner};
#[cfg(all(feature = "mpi", not(feature = "complex")))]
use crate::matrix::DistCsrOp;

/// High-level ASM preconditioner that dispatches to serial or distributed implementations.
pub struct AsmPc {
    overlap: usize,
    subdomain_hint: Option<usize>,
    block_solver: AsmBlockSolver,
    mode: AsmMode,
    weighting: Weighting,
    inner: Option<AsmImpl>,
}

enum AsmImpl {
    Serial(AdditiveSchwarz<faer::Mat<f64>, Vec<f64>, f64>),
    #[cfg(all(feature = "mpi", not(feature = "complex")))]
    Distributed(DistributedAsm),
}

#[derive(Clone, Copy, Debug)]
pub enum AsmBlockSolver {
    LuDense,
    Csr,
}

impl AsmPc {
    pub fn new(
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        mode: AsmMode,
        weighting: Weighting,
    ) -> Self {
        Self {
            overlap,
            subdomain_hint,
            block_solver,
            mode,
            weighting,
            inner: None,
        }
    }

    pub fn ras(
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        weighting: Weighting,
    ) -> Self {
        Self::new(overlap, subdomain_hint, block_solver, AsmMode::RAS, weighting)
    }

    fn build_serial(&self) -> AdditiveSchwarz<faer::Mat<f64>, Vec<f64>, f64> {
        let factory = match self.block_solver {
            AsmBlockSolver::LuDense => BlockSolverFactory::LuDense,
            AsmBlockSolver::Csr => BlockSolverFactory::CsrSolver,
        };
        let mut asm =
            AdditiveSchwarz::<faer::Mat<f64>, Vec<f64>, f64>::new(self.overlap, Vec::new(), factory);
        asm.set_mode(self.mode);
        asm.set_weighting(self.weighting);
        if let Some(hint) = self.subdomain_hint {
            asm.set_num_parts(hint);
        }
        asm
    }
}

impl Preconditioner for AsmPc {
    fn dims(&self) -> (usize, usize) {
        match &self.inner {
            Some(AsmImpl::Serial(pc)) => pc.dims(),
            #[cfg(all(feature = "mpi", not(feature = "complex")))]
            Some(AsmImpl::Distributed(pc)) => pc.dims(),
            None => (0, 0),
        }
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        #[cfg(all(feature = "mpi", not(feature = "complex")))]
        {
            let has_layout = op.dist_layout().is_some()
                || op.as_any().downcast_ref::<DistCsrOp>().is_some();
            if op.comm().size() > 1 && has_layout {
                let mut dist = DistributedAsm::new(
                    self.overlap,
                    self.subdomain_hint,
                    self.block_solver,
                    self.mode,
                    self.weighting,
                );
                match dist.setup(op) {
                    Ok(()) => {
                        self.inner = Some(AsmImpl::Distributed(dist));
                        return Ok(());
                    }
                    Err(err) => return Err(err),
                }
            }
        }

        let mut serial = self.build_serial();
        Preconditioner::setup(&mut serial, op)?;
        self.inner = Some(AsmImpl::Serial(serial));
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        match &self.inner {
            Some(AsmImpl::Serial(pc)) => pc.apply(side, x, y),
            #[cfg(all(feature = "mpi", not(feature = "complex")))]
            Some(AsmImpl::Distributed(pc)) => pc.apply(side, x, y),
            None => Err(KError::InvalidInput("ASM preconditioner not setup".into())),
        }
    }

    fn supports_numeric_update(&self) -> bool {
        match &self.inner {
            Some(AsmImpl::Serial(pc)) => pc.supports_numeric_update(),
            #[cfg(all(feature = "mpi", not(feature = "complex")))]
            Some(AsmImpl::Distributed(pc)) => pc.supports_numeric_update(),
            None => false,
        }
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        match &mut self.inner {
            Some(AsmImpl::Serial(pc)) => pc.update_numeric(op),
            #[cfg(all(feature = "mpi", not(feature = "complex")))]
            Some(AsmImpl::Distributed(pc)) => pc.update_numeric(op),
            None => Err(KError::InvalidInput("ASM preconditioner not setup".into())),
        }
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        match &mut self.inner {
            Some(AsmImpl::Serial(pc)) => pc.update_symbolic(op),
            #[cfg(all(feature = "mpi", not(feature = "complex")))]
            Some(AsmImpl::Distributed(pc)) => pc.update_symbolic(op),
            None => self.setup(op),
        }
    }
}
