use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::{LinearSolver, MonitorCallback};
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

pub struct RichardsonSolver {
    conv: Convergence,
    omega: f64,
}

impl RichardsonSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            conv: Convergence::new(rtol, 1e-12, 1e3, maxits),
            omega: 1.0,
        }
    }

    pub fn set_omega(&mut self, omega: f64) {
        self.omega = omega;
    }
}

impl LinearSolver for RichardsonSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        mut pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        _work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let n = b.len();
        if x.len() != n {
            return Err(KError::InvalidInput(
                "Richardson: vector length mismatch".into(),
            ));
        }
        let mut ax = vec![0.0; n];
        let mut r = vec![0.0; n];
        let mut z = vec![0.0; n];
        a.matvec(x, &mut ax);
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }
        let bnorm = comm.norm2(b);
        let mut rnorm = comm.norm2(&r);
        for m in monitors.unwrap_or(&[]) {
            let _ = m(0, rnorm, 0);
        }
        let (reason0, s0) = self.conv.check(rnorm, bnorm, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            return Ok(SolveStats::new(0, rnorm, s0.reason));
        }

        for k in 1..=self.conv.max_iters {
            if let Some(pc_ref) = pc.as_deref_mut() {
                pc_ref.apply(pc_side, &r, &mut z)?;
            } else {
                z.copy_from_slice(&r);
            }
            for i in 0..n {
                x[i] += self.omega * z[i];
            }
            a.matvec(x, &mut ax);
            for i in 0..n {
                r[i] = b[i] - ax[i];
            }
            rnorm = comm.norm2(&r);
            for m in monitors.unwrap_or(&[]) {
                let _ = m(k, rnorm, 0);
            }
            let (reason, s) = self.conv.check(rnorm, bnorm, k);
            if !matches!(reason, ConvergedReason::Continued) {
                return Ok(SolveStats::new(k, rnorm, s.reason));
            }
        }
        Ok(SolveStats::new(
            self.conv.max_iters,
            rnorm,
            ConvergedReason::DivergedMaxIts,
        ))
    }
}
