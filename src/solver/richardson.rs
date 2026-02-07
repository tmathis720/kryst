use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
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

    #[allow(clippy::too_many_arguments)]
    pub fn solve<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        let n = b.len();
        if x.len() != n {
            return Err(KError::InvalidInput(
                "Richardson: vector length mismatch".into(),
            ));
        }

        let mut local_scratch = BridgeScratch::default();
        let scratch = work
            .map(|w| &mut w.bridge)
            .unwrap_or(&mut local_scratch);

        let mut ax = vec![S::zero(); n];
        let mut r = vec![S::zero(); n];
        let mut z = vec![S::zero(); n];
        a.matvec_s(x, &mut ax, scratch);
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }

        let bnorm = comm.norm2(b);
        let mut rnorm = comm.norm2(&r);
        let monitors = monitors.unwrap_or(&[]);
        for m in monitors {
            let _ = m(0, rnorm, 0);
        }
        let (reason0, s0) = self.conv.check(rnorm, bnorm, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            return Ok(SolveStats::new(0, rnorm, s0.reason));
        }

        let omega = S::from_real(self.omega);
        for k in 1..=self.conv.max_iters {
            if let Some(pc_ref) = pc {
                pc_ref.apply_s(pc_side, &r, &mut z, scratch)?;
            } else {
                z.copy_from_slice(&r);
            }
            for i in 0..n {
                x[i] += omega * z[i];
            }
            a.matvec_s(x, &mut ax, scratch);
            for i in 0..n {
                r[i] = b[i] - ax[i];
            }
            rnorm = comm.norm2(&r);
            for m in monitors {
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

    #[allow(clippy::too_many_arguments)]
    pub fn solve_f64<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError>
    where
        A: LinOpF64 + LinOp<S = f64> + Send + Sync + ?Sized,
    {
        let op = as_s_op(a);
        let pc_wrapper = pc.map(as_s_pc);
        let pc_ref = pc_wrapper
            .as_ref()
            .map(|w| w as &dyn KPreconditioner<Scalar = S>);

        #[cfg(not(feature = "complex"))]
        {
            let b_s: &[S] = unsafe { &*(b as *const [f64] as *const [S]) };
            let x_s: &mut [S] = unsafe { &mut *(x as *mut [f64] as *mut [S]) };
            self.solve(&op, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
        }
        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result = self.solve(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work);
            if result.is_ok() {
                for (dst, src) in x.iter_mut().zip(x_s.iter()) {
                    *dst = src.real();
                }
            }
            result
        }
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
        self.solve_f64(
            a,
            pc.as_deref_mut().map(|p| &*p),
            b,
            x,
            pc_side,
            comm,
            monitors,
            _work,
        )
    }
}
