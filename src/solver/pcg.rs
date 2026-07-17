use crate::algebra::parallel::{dot_conj_local_with_mode, sum_abs2_local_with_mode};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::config::options::CgVariant;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::ops::wrap::{as_s_op, as_s_pc_mut};
use crate::parallel::{Comm, NoComm, ReduceHandle, ReductionEngine, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::reduction::{ReductionOptions, ReproMode};
use crate::solver::LinearSolver;
use crate::solver::MonitorCallback;
use crate::solver::cg::CgSolver;
use crate::utils::convergence::{Convergence, ReductionModel, SolveStats};
use crate::utils::reduction::{ReductOptions, record_reduction};
use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// A synchronous reduction engine for caller-provided communicator wrappers.
///
/// `PcgSolver::solve_with_comm` accepts `Comm + Clone` implementations.
/// The solver delegates its iteration to `CgSolver`, which needs a
/// `UniverseComm` for its execution policy, so this engine preserves the
/// caller's communicator for the actual reductions.
struct CallerCommReductionEngine<C> {
    comm: C,
    mode: ReproMode,
}

impl<C> CallerCommReductionEngine<C> {
    fn new(comm: C, mode: ReproMode) -> Self {
        Self { comm, mode }
    }
}

impl<C> fmt::Debug for CallerCommReductionEngine<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallerCommReductionEngine")
            .finish_non_exhaustive()
    }
}

impl<C: Comm> ReductionEngine for CallerCommReductionEngine<C> {
    fn supports_async(&self) -> bool {
        false
    }

    fn allreduce_sum_r(&self, x: R) -> R {
        self.comm.allreduce_sum(x)
    }

    fn allreduce_sum_s(&self, x: S) -> S {
        self.comm.allreduce_sum_scalar(x)
    }

    fn norm2_s(&self, x: &[S]) -> R {
        self.allreduce_sum_r(sum_abs2_local_with_mode(x, self.mode))
            .max(R::zero())
            .sqrt()
    }

    fn dot_s(&self, x: &[S], y: &[S]) -> S {
        self.allreduce_sum_s(dot_conj_local_with_mode(x, y, self.mode))
    }

    fn sum_vec_r(&self, mut values: Vec<R>) -> Vec<R> {
        record_reduction(values.len());
        self.comm.allreduce_sum_slice(&mut values);
        values
    }

    fn iallreduce_sum_r(&self, x: R) -> ReduceHandle<R> {
        record_reduction(1);
        ReduceHandle::ready(self.allreduce_sum_r(x))
    }

    fn iallreduce_sum_s(&self, x: S) -> ReduceHandle<S> {
        #[cfg(feature = "complex")]
        record_reduction(2);
        #[cfg(not(feature = "complex"))]
        record_reduction(1);
        ReduceHandle::ready(self.allreduce_sum_s(x))
    }

    fn iallreduce_sum_vec_r(&self, values: Vec<R>) -> ReduceHandle<Vec<R>> {
        ReduceHandle::ready(self.sum_vec_r(values))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CgNormType {
    /// Monitor the preconditioned residual `sqrt(rᵀz)` (default)
    Preconditioned,
    /// Monitor the unpreconditioned residual `||r||₂`
    Unpreconditioned,
    /// Monitor the "natural" norm of the preconditioned residual vector
    /// `||z||₂`, matching PETSc's `-ksp_norm_type natural` semantics.
    Natural,
    /// Do not compute or report a residual norm
    None,
}

/// Supported PCG algorithmic variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcgVariant {
    /// Classic Hestenes-Stiefel style PCG.
    Classic,
    /// Pipelined CG with asynchronous reductions.
    Pipelined {
        /// Residual replacement interval (0 disables replacement).
        replace_every: usize,
    },
}

/// Default replacement interval for pipelined CG.
///
pub const PCG_PIPELINED_DEFAULT_REPLACE_EVERY: usize = 50;

pub struct PcgSolver {
    pub(crate) conv: Convergence,
    norm_type: CgNormType,
    reduction: ReductionOptions,
    true_residual_monitor: Option<Box<MonitorCallback<f64>>>,
    /// Whether the initial guess in `x` should be treated as nonzero.
    ///
    /// PETSc zeroes the initial guess by default unless told otherwise via
    /// `KSPSetInitialGuessNonzero`. We follow the same policy; when this flag is
    /// `false` and the provided `x` is exactly the zero vector, the solver
    /// skips the initial matvec and assumes a zero guess.
    initial_guess_nonzero: bool,
    variant: PcgVariant,
    async_reduction: ReductOptions,
    async_enabled: bool,
    async_min_n: usize,
}

impl PcgSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            conv: Convergence {
                rtol,
                atol: 1e-50,
                dtol: 1e5,
                max_iters: maxits,
            },
            norm_type: CgNormType::Preconditioned,
            reduction: ReductionOptions::default(),
            true_residual_monitor: None,
            initial_guess_nonzero: false,
            variant: PcgVariant::Classic,
            async_reduction: ReductOptions::default(),
            async_enabled: true,
            async_min_n: 10_000,
        }
    }

    /// Optional runtime update of solver tolerances
    pub fn set_tolerances(&mut self, rtol: f64, atol: f64, dtol: f64, maxits: usize) {
        self.conv.rtol = rtol;
        self.conv.atol = atol;
        self.conv.dtol = dtol;
        self.conv.max_iters = maxits;
    }

    pub fn with_norm(mut self, norm_type: CgNormType) -> Self {
        self.norm_type = norm_type;
        self
    }

    /// Enable a more reproducible (but slightly slower) local dot product using
    /// Kahan summation. When combined with a deterministic MPI reduction this
    /// yields bitwise-identical results across runs.
    pub fn with_reproducible_dot(mut self, f: bool) -> Self {
        self.reduction.mode = if f {
            ReproMode::Deterministic
        } else {
            ReproMode::Fast
        };
        self
    }

    /// Install a monitor that receives the true residual norm `||b - A x||₂`
    /// at each iteration. This uses the already available residual and is
    /// intended for debugging.
    pub fn with_true_residual_monitor(mut self, m: Box<MonitorCallback<f64>>) -> Self {
        self.true_residual_monitor = Some(m);
        self
    }

    #[must_use = "with_variant returns an updated solver; assign it before continuing"]
    pub fn with_variant(mut self, variant: PcgVariant) -> Self {
        self.variant = variant;
        self
    }

    pub fn set_variant(&mut self, variant: PcgVariant) {
        self.variant = variant;
    }

    pub fn variant(&self) -> PcgVariant {
        self.variant
    }

    pub fn pipelined_residual_refresh_every(&self) -> Option<usize> {
        match self.variant {
            PcgVariant::Pipelined { replace_every } if replace_every > 0 => Some(replace_every),
            _ => None,
        }
    }

    pub fn set_async_reduction_options(&mut self, opt: ReductOptions) {
        self.async_reduction = opt;
    }

    pub fn set_async_enabled(&mut self, enabled: bool) {
        self.async_enabled = enabled;
    }

    pub fn async_enabled(&self) -> bool {
        self.async_enabled
    }

    pub fn set_async_min_n(&mut self, n: usize) {
        self.async_min_n = n;
    }

    pub fn async_min_n(&self) -> usize {
        self.async_min_n
    }

    fn reduction_model(&self) -> ReductionModel {
        match self.variant {
            PcgVariant::Classic => ReductionModel {
                variant: "pcg-classic",
                startup: 2,
                per_iteration: 2.0,
                tail: 0,
            },
            PcgVariant::Pipelined { .. } => ReductionModel {
                variant: "pcg-pipelined",
                startup: 1,
                per_iteration: 1.0,
                tail: 0,
            },
        }
    }

    fn async_options(&self) -> ReductOptions {
        let mut opt = self.async_reduction.clone();
        opt.mode = self.reduction.mode;
        opt
    }

    fn configured_cg(&self) -> CgSolver {
        let mut cg = CgSolver::new(self.conv.rtol, self.conv.max_iters);
        cg.conv.atol = self.conv.atol;
        cg.conv.dtol = self.conv.dtol;
        cg.set_nonzero_guess(self.initial_guess_nonzero);
        cg.set_async_enabled(self.async_enabled);
        cg.set_async_min_n(self.async_min_n);
        cg.set_variant(match self.variant {
            PcgVariant::Classic => CgVariant::Classic,
            PcgVariant::Pipelined { .. } => CgVariant::Pipelined,
        });
        if let PcgVariant::Pipelined { replace_every } = self.variant {
            cg.set_pipelined_residual_refresh_every((replace_every > 0).then_some(replace_every));
        }
        cg.set_norm(match self.norm_type {
            CgNormType::Preconditioned => crate::solver::cg::CgNormType::Preconditioned,
            CgNormType::Unpreconditioned => crate::solver::cg::CgNormType::Unpreconditioned,
            CgNormType::Natural => crate::solver::cg::CgNormType::Natural,
            CgNormType::None => crate::solver::cg::CgNormType::None,
        });
        cg
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_k_via_cg<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn crate::ops::kpc::KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
        reduction_engine: Option<Arc<dyn ReductionEngine>>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: crate::ops::klinop::KLinOp<Scalar = S> + ?Sized,
    {
        let mut owned_workspace;
        let work = match work {
            Some(work) => work,
            None => {
                owned_workspace = Workspace::new(b.len());
                &mut owned_workspace
            }
        };

        let saved_reduction = work.reduction_options().clone();
        let saved_engine = work.reduction_engine().cloned();
        let reduction = self.async_options();
        work.set_reduction_options(reduction.clone());
        work.set_reduction_engine(
            reduction_engine.unwrap_or_else(|| comm.reduction_engine(&reduction)),
        );

        let mut cg = self.configured_cg();
        cg.set_true_residual_monitor(self.true_residual_monitor.take());
        cg.setup_workspace(work);

        let result = cg
            .solve_with_comm(a, pc, b, x, pc_side, comm, monitors, Some(work))
            .map(|stats| stats.with_reduction_model(self.reduction_model()));

        self.true_residual_monitor = cg.take_true_residual_monitor();
        work.set_reduction_options(saved_reduction);
        if let Some(engine) = saved_engine {
            work.set_reduction_engine(engine);
        } else {
            work.clear_reduction_engine();
        }

        result
    }

    /// Indicate whether the supplied initial guess is nonzero.
    ///
    /// By default `x` is assumed to be zero, which avoids an extra matvec on
    /// entry. Calling this with `true` forces the solver to compute the initial
    /// residual `b - A x` even if `x` happens to be the zero vector.
    pub fn with_nonzero_guess(mut self, f: bool) -> Self {
        self.initial_guess_nonzero = f;
        self
    }

    /// Set the nonzero initial guess flag after construction.
    pub fn set_nonzero_guess(&mut self, f: bool) {
        self.initial_guess_nonzero = f;
    }

    /// Toggle reproducible local dot products after construction.
    pub fn set_reproducible_dot(&mut self, f: bool) {
        self.reduction.mode = if f {
            ReproMode::Deterministic
        } else {
            ReproMode::Fast
        };
    }

    /// Set or clear the true residual monitor after construction.
    pub fn set_true_residual_monitor(&mut self, m: Option<Box<MonitorCallback<f64>>>) {
        self.true_residual_monitor = m;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_k<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn crate::ops::kpc::KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: crate::ops::klinop::KLinOp<Scalar = S> + ?Sized,
    {
        self.solve_k_via_cg(a, pc, b, x, pc_side, comm, monitors, work, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_comm<C: Comm + Clone>(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &C,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let universe = (comm as &dyn Any)
            .downcast_ref::<UniverseComm>()
            .cloned()
            .or_else(|| {
                (comm as &dyn Any)
                    .downcast_ref::<NoComm>()
                    .map(|_| UniverseComm::NoComm(NoComm))
            })
            .or_else(|| {
                #[cfg(feature = "rayon")]
                {
                    (comm as &dyn Any)
                        .downcast_ref::<crate::parallel::RayonComm>()
                        .map(|c| UniverseComm::Rayon(c.clone()))
                }
                #[cfg(not(feature = "rayon"))]
                {
                    None
                }
            })
            .or_else(|| {
                #[cfg(feature = "mpi")]
                {
                    (comm as &dyn Any)
                        .downcast_ref::<crate::parallel::MpiComm>()
                        .map(|c| UniverseComm::Mpi(std::sync::Arc::new(c.dup())))
                }
                #[cfg(not(feature = "mpi"))]
                {
                    None
                }
            });
        let caller_reduction_engine = universe.is_none().then(|| {
            Arc::new(CallerCommReductionEngine::new(
                comm.clone(),
                self.async_options().effective_mode(),
            )) as Arc<dyn ReductionEngine>
        });
        let universe = universe.unwrap_or_else(|| comm.split(0, comm.rank() as i32));
        self.solve_impl(
            a,
            pc,
            b,
            x,
            pc_side,
            &universe,
            monitors,
            work,
            caller_reduction_engine,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_impl(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
        reduction_engine: Option<Arc<dyn ReductionEngine>>,
    ) -> Result<SolveStats<f64>, KError> {
        let op = as_s_op(a);
        let pc_wrapper = pc.map(as_s_pc_mut);
        let pc_ref = pc_wrapper
            .as_ref()
            .map(|pc| pc as &dyn crate::ops::kpc::KPreconditioner<Scalar = S>);

        let mut owned_workspace;
        let work = match work {
            Some(work) => Some(work),
            None => {
                owned_workspace = Workspace::new(b.len());
                Some(&mut owned_workspace)
            }
        };

        #[cfg(not(feature = "complex"))]
        {
            let b_s: &[S] = unsafe { &*(b as *const [f64] as *const [S]) };
            let x_s: &mut [S] = unsafe { &mut *(x as *mut [f64] as *mut [S]) };
            self.solve_k_via_cg(
                &op,
                pc_ref,
                b_s,
                x_s,
                pc_side,
                comm,
                monitors,
                work,
                reduction_engine,
            )
        }

        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result = self.solve_k_via_cg(
                &op,
                pc_ref,
                &b_s,
                &mut x_s,
                pc_side,
                comm,
                monitors,
                work,
                reduction_engine,
            );
            if result.is_ok() {
                for (dst, src) in x.iter_mut().zip(x_s.iter()) {
                    *dst = src.real();
                }
            }
            result
        }
    }
}

impl LinearSolver for PcgSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q.len() < 2 {
            work.q.resize(2, Vec::new());
        }
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        self.solve_impl(a, pc, b, x, pc_side, comm, monitors, work, None)
    }
}
