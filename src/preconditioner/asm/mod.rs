//! Additive Schwarz preconditioner (ASM) module.

#[cfg(feature = "mpi")]
mod comm_plan;
#[cfg(feature = "mpi")]
mod distributed;
mod serial;
#[cfg(feature = "mpi")]
mod subdomain;

#[cfg(feature = "mpi")]
pub use distributed::DistributedAsm;
pub use serial::*;

use crate::algebra::prelude::*;
use crate::error::KError;
#[cfg(feature = "mpi")]
use crate::matrix::DistCsrOp;
use crate::matrix::op::LinOp;
#[cfg(feature = "mpi")]
use crate::parallel::Comm;
use crate::preconditioner::{PcSide, Preconditioner};

/// High-level ASM preconditioner that dispatches to serial or distributed implementations.
pub struct AsmPc {
    overlap: usize,
    subdomain_hint: Option<usize>,
    block_solver: AsmBlockSolver,
    inner_pc: AsmInnerPc,
    mode: AsmMode,
    weighting: Weighting,
    inner: Option<AsmImpl>,
}

enum AsmImpl {
    Serial(AdditiveSchwarz<faer::Mat<f64>, Vec<f64>, f64>),
    #[cfg(feature = "mpi")]
    Distributed(DistributedAsm),
}

#[derive(Clone, Copy, Debug)]
pub enum AsmBlockSolver {
    LuDense,
    Csr,
}

#[derive(Clone, Copy, Debug)]
pub enum AsmInnerPc {
    Jacobi,
    Ilu0,
    Ilut {
        drop_tol: f64,
        max_fill: usize,
    },
    Ilutp {
        drop_tol: f64,
        max_fill: usize,
        perm_tol: f64,
    },
}

impl AsmPc {
    pub fn new(
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        inner_pc: AsmInnerPc,
        mode: AsmMode,
        weighting: Weighting,
    ) -> Self {
        Self {
            overlap,
            subdomain_hint,
            block_solver,
            inner_pc,
            mode,
            weighting,
            inner: None,
        }
    }

    pub fn ras(
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: AsmBlockSolver,
        inner_pc: AsmInnerPc,
        weighting: Weighting,
    ) -> Self {
        Self::new(
            overlap,
            subdomain_hint,
            block_solver,
            inner_pc,
            AsmMode::RAS,
            weighting,
        )
    }

    fn build_serial(&self) -> AdditiveSchwarz<faer::Mat<f64>, Vec<f64>, f64> {
        let factory = match self.block_solver {
            AsmBlockSolver::LuDense => BlockSolverFactory::LuDense,
            AsmBlockSolver::Csr => BlockSolverFactory::CsrSolver,
        };
        let mut asm = AdditiveSchwarz::<faer::Mat<f64>, Vec<f64>, f64>::new(
            self.overlap,
            Vec::new(),
            factory,
        );
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
            #[cfg(feature = "mpi")]
            Some(AsmImpl::Distributed(pc)) => pc.dims(),
            None => (0, 0),
        }
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        #[cfg(feature = "mpi")]
        {
            let has_layout =
                op.dist_layout().is_some() || op.as_any().downcast_ref::<DistCsrOp>().is_some();
            if op.comm().size() > 1 && has_layout {
                let mut dist = DistributedAsm::new(
                    self.overlap,
                    self.subdomain_hint,
                    self.block_solver,
                    self.inner_pc,
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
            #[cfg(feature = "mpi")]
            Some(AsmImpl::Distributed(pc)) => pc.apply(side, x, y),
            None => Err(KError::InvalidInput("ASM preconditioner not setup".into())),
        }
    }

    fn supports_numeric_update(&self) -> bool {
        match &self.inner {
            Some(AsmImpl::Serial(pc)) => pc.supports_numeric_update(),
            #[cfg(feature = "mpi")]
            Some(AsmImpl::Distributed(pc)) => pc.supports_numeric_update(),
            None => false,
        }
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        match &mut self.inner {
            Some(AsmImpl::Serial(pc)) => pc.update_numeric(op),
            #[cfg(feature = "mpi")]
            Some(AsmImpl::Distributed(pc)) => pc.update_numeric(op),
            None => Err(KError::InvalidInput("ASM preconditioner not setup".into())),
        }
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        match &mut self.inner {
            Some(AsmImpl::Serial(pc)) => pc.update_symbolic(op),
            #[cfg(feature = "mpi")]
            Some(AsmImpl::Distributed(pc)) => pc.update_symbolic(op),
            None => self.setup(op),
        }
    }
}
