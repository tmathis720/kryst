#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::config::options::CgVariant;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::op_bridge::matvec_s;
use crate::ops::wrap::{as_s_op, as_s_pc_mut};
use crate::parallel::{Comm, NoComm, UniverseComm};
use crate::preconditioner::bridge::apply_pc_s;
use crate::preconditioner::{PcSide, Preconditioner};
#[cfg(not(feature = "complex"))]
use crate::reduction::DotEngine;
#[cfg(feature = "complex")]
use crate::reduction::Packet;
use crate::reduction::{CommDeterministic, ReductionOptions, ReproMode};
use crate::solver::LinearSolver;
use crate::solver::MonitorCallback;
use crate::solver::cg::CgSolver;
use crate::solver::common::call_monitors;
#[cfg(feature = "complex")]
use crate::solver::common::dot_result_to_real;
use crate::solver::common::{dot1_async_s, nrm2_async_s};
use crate::utils::convergence::{
    ConvergedReason, Convergence, ReductionModel, SolveStats, SolverCounters,
};
use crate::utils::reduction::{AllreduceHandle, AllreduceOps, ReductOptions};
use smallvec::SmallVec;
use std::any::Any;

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

struct ClassicWorkspace<'a> {
    r: &'a mut [S],
    z: &'a mut [S],
    p: &'a mut [S],
    ap: &'a mut [S],
    scratch: &'a mut BridgeScratch,
}

impl<'a> ClassicWorkspace<'a> {
    fn acquire(work: &'a mut Workspace, n: usize) -> Self {
        if work.tmp1.len() != n {
            work.tmp1.resize(n, S::zero());
        }
        if work.tmp2.len() != n {
            work.tmp2.resize(n, S::zero());
        }
        while work.q_s.len() < 2 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..2] {
            if buf.len() != n {
                buf.resize(n, S::zero());
            }
        }
        let (pbuf, rest) = work.q_s.split_at_mut(1);
        let (apbuf, _) = rest.split_at_mut(1);
        Self {
            r: &mut work.tmp1[..n],
            z: &mut work.tmp2[..n],
            p: &mut pbuf[0][..n],
            ap: &mut apbuf[0][..n],
            scratch: &mut work.bridge,
        }
    }
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
        work.set_reduction_engine(comm.reduction_engine(&reduction));

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

    #[inline]
    #[cfg(not(feature = "complex"))]
    fn dot<C: Comm + CommDeterministic>(&self, u: &[f64], v: &[f64], comm: &C) -> f64 {
        let engine = DotEngine {
            opts: self.reduction,
        };
        engine.dot(u, v, comm)
    }

    #[inline]
    fn dot_scalar<C: Comm + CommDeterministic>(&self, u: &[S], v: &[S], comm: &C) -> R {
        #[cfg(not(feature = "complex"))]
        {
            let ur: &[f64] = unsafe { &*(u as *const [S] as *const [f64]) };
            let vr: &[f64] = unsafe { &*(v as *const [S] as *const [f64]) };
            self.dot(ur, vr, comm)
        }

        #[cfg(feature = "complex")]
        {
            let local = crate::algebra::blas::dot_conj(u, v);
            if matches!(self.reduction.mode, ReproMode::Fast) {
                return dot_result_to_real(comm.allreduce_sum_scalar(local));
            }

            let packet = Packet::<2> {
                v: [local.real(), local.imag()],
            };
            let reduced = comm.allreduce_det(&packet, self.reduction.mode);
            reduced.v[0]
        }
    }

    #[inline]
    fn dot_scalar_many<C: Comm + CommDeterministic>(
        &self,
        pairs: &[(&[S], &[S])],
        comm: &C,
        out: &mut [R],
    ) {
        assert_eq!(pairs.len(), out.len());
        if pairs.is_empty() {
            return;
        }

        #[cfg(not(feature = "complex"))]
        {
            let engine = DotEngine {
                opts: self.reduction,
            };
            let mut real_pairs: SmallVec<[(&[f64], &[f64]); 8]> =
                SmallVec::with_capacity(pairs.len());
            for &(u, v) in pairs {
                let ur: &[f64] = unsafe { &*(u as *const [S] as *const [f64]) };
                let vr: &[f64] = unsafe { &*(v as *const [S] as *const [f64]) };
                real_pairs.push((ur, vr));
            }
            engine.dot_many_into(real_pairs.as_slice(), out, comm);
        }

        #[cfg(feature = "complex")]
        {
            for ((u, v), slot) in pairs.iter().zip(out.iter_mut()) {
                let local = crate::algebra::blas::dot_conj(u, v);
                if matches!(self.reduction.mode, ReproMode::Fast) {
                    *slot = dot_result_to_real(comm.allreduce_sum_scalar(local));
                } else {
                    let packet = Packet::<2> {
                        v: [local.real(), local.imag()],
                    };
                    let reduced = comm.allreduce_det(&packet, self.reduction.mode);
                    *slot = reduced.v[0];
                }
            }
        }
    }

    #[inline]
    fn nrm2_scalar<C: Comm + CommDeterministic>(&self, u: &[S], comm: &C) -> R {
        let val = self.dot_scalar(u, u, comm);
        val.abs().sqrt()
    }

    #[inline]
    fn ensure_norm<C: Comm + CommDeterministic>(
        &self,
        vec: &[S],
        comm: &C,
        cache: &mut Option<R>,
    ) -> R {
        if let Some(val) = *cache {
            val
        } else {
            let val = self.nrm2_scalar(vec, comm);
            *cache = Some(val);
            val
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_classic_scalar<C: Comm + CommDeterministic>(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&dyn Preconditioner>,
        b: &[S],
        x: &mut [S],
        comm: &C,
        monitors: &[Box<MonitorCallback<f64>>],
        work: &mut Workspace,
    ) -> Result<SolveStats<f64>, KError> {
        let n = b.len();
        let mut buffers = ClassicWorkspace::acquire(work, n);
        let ClassicWorkspace {
            r,
            z,
            p,
            ap,
            scratch,
        } = &mut buffers;
        let (r, z, p, ap, scratch) = (&mut **r, &mut **z, &mut **p, &mut **ap, &mut **scratch);

        let zero_guess =
            !self.initial_guess_nonzero && x.iter().all(|&xi| xi.abs() <= R::default());
        if zero_guess {
            r.copy_from_slice(b);
        } else {
            matvec_s(a, x, &mut ap[..], scratch);
            for i in 0..n {
                r[i] = b[i] - ap[i];
            }
        }

        if let Some(pc) = pc {
            apply_pc_s(pc, PcSide::Left, r, &mut z[..], scratch)?;
        } else {
            z.copy_from_slice(r);
        }

        let need_r_norm_res = matches!(
            self.norm_type,
            CgNormType::Unpreconditioned | CgNormType::None
        );
        let need_z_norm_res = matches!(self.norm_type, CgNormType::Natural);
        let need_r_norm_monitor = self.true_residual_monitor.is_some();

        let mut initial_pairs: SmallVec<[(&[S], &[S]); 3]> = SmallVec::new();
        initial_pairs.push((&r[..], &z[..]));
        if need_r_norm_res || need_r_norm_monitor {
            initial_pairs.push((&r[..], &r[..]));
        }
        if need_z_norm_res {
            initial_pairs.push((&z[..], &z[..]));
        }
        let mut reductions: SmallVec<[R; 3]> = SmallVec::new();
        reductions.resize(initial_pairs.len(), R::zero());
        self.dot_scalar_many(initial_pairs.as_slice(), comm, reductions.as_mut_slice());

        let mut idx = 0;
        let mut rho = reductions[idx];
        idx += 1;
        if rho <= R::default() || !rho.is_finite() {
            return Err(KError::IndefinitePreconditioner);
        }

        let mut cached_r_norm = if need_r_norm_res || need_r_norm_monitor {
            let value = reductions[idx];
            idx += 1;
            Some(
                if value < R::default() {
                    R::default()
                } else {
                    value
                }
                .sqrt(),
            )
        } else {
            None
        };
        let cached_z_norm = if need_z_norm_res {
            let value = reductions[idx];
            Some(
                if value < R::default() {
                    R::default()
                } else {
                    value
                }
                .sqrt(),
            )
        } else {
            None
        };
        let mut rho_prev = rho;

        drop(initial_pairs);
        drop(reductions);

        let mut res = match self.norm_type {
            CgNormType::Preconditioned => rho.abs().sqrt(),
            CgNormType::Unpreconditioned => cached_r_norm.unwrap(),
            CgNormType::Natural => cached_z_norm.unwrap(),
            CgNormType::None => cached_r_norm.unwrap(),
        };
        let res0 = res;

        if call_monitors(monitors, 0, res, 0) {
            return Ok(SolveStats::new(0, res, ConvergedReason::StoppedByMonitor));
        }
        if let Some(m) = &self.true_residual_monitor {
            let value = self.ensure_norm(r, comm, &mut cached_r_norm);
            m(0, value, 0);
        }

        p.copy_from_slice(z);

        let (reason0, mut stats0) = self.conv.check(res, res0, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            stats0.final_residual = self.ensure_norm(r, comm, &mut cached_r_norm);
            return Ok(stats0);
        }

        for k in 1..=self.conv.max_iters {
            if k > 1 {
                let beta = rho / rho_prev;
                let beta_s = S::from_real(beta);
                for i in 0..n {
                    p[i] = z[i] + beta_s * p[i];
                }
            }

            matvec_s(a, p, &mut ap[..], scratch);
            let p_ap = self.dot_scalar(p, ap, comm);
            if !p_ap.is_finite() || p_ap <= R::default() {
                return Err(KError::IndefiniteMatrix);
            }

            let alpha = rho / p_ap;
            let alpha_s = S::from_real(alpha);
            for i in 0..n {
                x[i] += alpha_s * p[i];
                r[i] -= alpha_s * ap[i];
            }

            if let Some(pc) = pc {
                apply_pc_s(pc, PcSide::Left, r, &mut z[..], scratch)?;
            } else {
                z.copy_from_slice(r);
            }

            let need_r_norm_res = matches!(
                self.norm_type,
                CgNormType::Unpreconditioned | CgNormType::None
            );
            let need_z_norm_res = matches!(self.norm_type, CgNormType::Natural);
            let need_r_norm_monitor = self.true_residual_monitor.is_some();

            let mut dot_pairs: SmallVec<[(&[S], &[S]); 3]> = SmallVec::new();
            dot_pairs.push((&r[..], &z[..]));
            if need_r_norm_res || need_r_norm_monitor {
                dot_pairs.push((&r[..], &r[..]));
            }
            if need_z_norm_res {
                dot_pairs.push((&z[..], &z[..]));
            }
            let mut dot_results: SmallVec<[R; 3]> = SmallVec::new();
            dot_results.resize(dot_pairs.len(), R::default());
            self.dot_scalar_many(dot_pairs.as_slice(), comm, dot_results.as_mut_slice());

            let mut idx = 0;
            let mut rho_new = dot_results[idx];
            idx += 1;
            if !rho_new.is_finite() || rho_new < R::default() {
                return Err(KError::IndefinitePreconditioner);
            }
            if rho_new < 1e-300 {
                rho_new = R::default();
            }

            let mut r_norm = if need_r_norm_res || need_r_norm_monitor {
                let value = dot_results[idx];
                idx += 1;
                Some(
                    if value < R::default() {
                        R::default()
                    } else {
                        value
                    }
                    .sqrt(),
                )
            } else {
                None
            };
            let mut z_norm = if need_z_norm_res {
                let value = dot_results[idx];
                Some(
                    if value < R::default() {
                        R::default()
                    } else {
                        value
                    }
                    .sqrt(),
                )
            } else {
                None
            };

            drop(dot_pairs);
            drop(dot_results);

            match self.norm_type {
                CgNormType::Preconditioned => {
                    res = rho_new.abs().sqrt();
                }
                CgNormType::Unpreconditioned => {
                    res = r_norm.unwrap();
                }
                CgNormType::Natural => {
                    res = z_norm.unwrap();
                }
                CgNormType::None => {}
            }

            if call_monitors(monitors, k, res, 0) {
                return Ok(SolveStats::new(k, res, ConvergedReason::StoppedByMonitor));
            }
            if let Some(m) = &self.true_residual_monitor {
                let value = self.ensure_norm(r, comm, &mut r_norm);
                m(k, value, 0);
            }

            let res_check = match self.norm_type {
                CgNormType::Preconditioned => rho_new.abs().sqrt(),
                CgNormType::Unpreconditioned => self.ensure_norm(r, comm, &mut r_norm),
                CgNormType::Natural => self.ensure_norm(z, comm, &mut z_norm),
                CgNormType::None => self.ensure_norm(r, comm, &mut r_norm),
            };
            let (reason, mut stats) = self.conv.check(res_check, res0, k);
            if !matches!(reason, ConvergedReason::Continued) {
                stats.final_residual = self.ensure_norm(r, comm, &mut r_norm);
                return Ok(stats);
            }

            rho_prev = rho;
            rho = rho_new;
            cached_r_norm = r_norm;
        }

        let final_res = self.ensure_norm(r, comm, &mut cached_r_norm);
        Ok(SolveStats::new(
            self.conv.max_iters,
            final_res,
            ConvergedReason::DivergedMaxIts,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_pipelined_scalar<C: Comm + CommDeterministic + AllreduceOps>(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&dyn Preconditioner>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &C,
        monitors: &[Box<MonitorCallback<f64>>],
        work: &mut Workspace,
    ) -> Result<SolveStats<f64>, KError> {
        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "Pipelined PCG requires left preconditioning with HPD M; choose PcSide::Left or use MINRES (Hermitian) / GMRES (general) instead".into(),
            ));
        }

        let n = b.len();
        if x.len() != n {
            return Err(KError::InvalidInput("dimension mismatch: x,b".into()));
        }

        let mut counters = SolverCounters::default();
        let mut buffers = ClassicWorkspace::acquire(work, n);
        let ClassicWorkspace {
            r,
            z,
            p,
            ap,
            scratch,
        } = &mut buffers;

        let zero_guess =
            !self.initial_guess_nonzero && x.iter().all(|&xi| xi.abs() <= R::default());
        if zero_guess {
            r.copy_from_slice(b);
        } else {
            matvec_s(a, x, &mut ap[..], scratch);
            for i in 0..n {
                r[i] = b[i] - ap[i];
            }
        }

        if let Some(pc) = pc {
            apply_pc_s(pc, PcSide::Left, r, &mut z[..], scratch)?;
        } else {
            z.copy_from_slice(r);
        }

        p.copy_from_slice(z);

        let mut opt = self.async_options();
        if opt.max_inflight == 0 {
            opt.max_inflight = 1;
        }

        let (h_rho0, _) = dot1_async_s(comm, r, z, &opt)?;
        let rho0 = {
            counters.num_global_reductions += 1;
            <C as AllreduceOps>::wait_pair(h_rho0).0
        };
        if !rho0.is_finite() || rho0 < R::default() {
            return Err(KError::IndefinitePreconditioner);
        }

        let (h_rnorm0, _) = nrm2_async_s(comm, r, &opt);
        let rnorm0_sq = {
            counters.num_global_reductions += 1;
            <C as AllreduceOps>::wait_pair(h_rnorm0).0
        };
        let rnorm0 = rnorm0_sq.sqrt();
        if rnorm0 == R::default() {
            return Ok(
                SolveStats::new(0, R::default(), ConvergedReason::ConvergedRtol).with_counters({
                    counters.overlap_global_reductions = counters.num_global_reductions;
                    counters
                }),
            );
        }

        let _ = match self.norm_type {
            CgNormType::Preconditioned => rho0.sqrt(),
            CgNormType::Unpreconditioned | CgNormType::None => rnorm0,
            CgNormType::Natural => {
                let (h_z, _) = nrm2_async_s(comm, z, &opt);
                counters.num_global_reductions += 1;
                <C as AllreduceOps>::wait_pair(h_z).0.sqrt()
            }
        };

        let actual_res0 = self.nrm2_scalar(r, comm);
        counters.num_global_reductions += 1;
        if call_monitors(monitors, 0, actual_res0, counters.num_global_reductions) {
            return Ok(
                SolveStats::new(0, actual_res0, ConvergedReason::StoppedByMonitor).with_counters({
                    counters.overlap_global_reductions = counters.num_global_reductions;
                    counters
                }),
            );
        }
        if let Some(m) = &self.true_residual_monitor {
            m(0, actual_res0, 0);
        }

        let (reason0, mut stats0) = self.conv.check(actual_res0, rnorm0, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            stats0.final_residual = actual_res0;
            counters.residual_replacements = 0;
            counters.overlap_global_reductions = counters.num_global_reductions;
            stats0.counters = counters;
            return Ok(stats0);
        }

        let replace_every = match self.variant {
            PcgVariant::Pipelined { replace_every } => replace_every,
            PcgVariant::Classic => 0,
        };

        let mut rho_curr = rho0;
        let mut rho_prev = rho0;
        let mut pending_rho: Option<AllreduceHandle<(R, R)>> = None;
        let mut iterations = 0usize;
        let mut residual_replacements = 0usize;
        let mut force_restart = false;

        'solve: loop {
            while iterations < self.conv.max_iters {
                if iterations > 0 {
                    let handle = pending_rho
                        .take()
                        .expect("pipelined PCG pending rho handle");
                    rho_curr = {
                        counters.num_global_reductions += 1;
                        <C as AllreduceOps>::wait_pair(handle).0
                    };
                    if !rho_curr.is_finite() || rho_curr < R::default() {
                        return Err(KError::IndefinitePreconditioner);
                    }

                    let _ = match self.norm_type {
                        CgNormType::Preconditioned => rho_curr.sqrt(),
                        CgNormType::Unpreconditioned | CgNormType::None => {
                            let (h_nr, _) = nrm2_async_s(comm, r, &opt);
                            counters.num_global_reductions += 1;
                            <C as AllreduceOps>::wait_pair(h_nr).0.sqrt()
                        }
                        CgNormType::Natural => {
                            let (h_z, _) = nrm2_async_s(comm, z, &opt);
                            counters.num_global_reductions += 1;
                            <C as AllreduceOps>::wait_pair(h_z).0.sqrt()
                        }
                    };

                    let actual_res = self.nrm2_scalar(r, comm);
                    counters.num_global_reductions += 1;

                    if call_monitors(
                        monitors,
                        iterations,
                        actual_res,
                        counters.num_global_reductions,
                    ) {
                        return Ok(SolveStats::new(
                            iterations,
                            actual_res,
                            ConvergedReason::StoppedByMonitor,
                        )
                        .with_counters({
                            counters.overlap_global_reductions = counters.num_global_reductions;
                            counters
                        }));
                    }
                    if let Some(m) = &self.true_residual_monitor {
                        m(iterations, actual_res, 0);
                    }

                    let (reason, mut stats) = self.conv.check(actual_res, rnorm0, iterations);
                    if !matches!(reason, ConvergedReason::Continued) {
                        stats.final_residual = actual_res;
                        let tol = self.conv.atol.max(self.conv.rtol * rnorm0);
                        if actual_res > tol {
                            matvec_s(a, x, &mut ap[..], scratch);
                            for i in 0..n {
                                r[i] = b[i] - ap[i];
                            }
                            if let Some(pc) = pc {
                                apply_pc_s(pc, PcSide::Left, r, &mut z[..], scratch)?;
                            } else {
                                z.copy_from_slice(r);
                            }
                            residual_replacements += 1;
                            p.copy_from_slice(z);
                            rho_prev = R::default();

                            rho_curr = self.dot_scalar(r, z, comm);
                            counters.num_global_reductions += 1;
                            if !rho_curr.is_finite() || rho_curr < R::default() {
                                return Err(KError::IndefinitePreconditioner);
                            }

                            let _ = match self.norm_type {
                                CgNormType::Preconditioned => rho_curr.sqrt(),
                                CgNormType::Unpreconditioned | CgNormType::None => {
                                    counters.num_global_reductions += 1;
                                    self.nrm2_scalar(r, comm)
                                }
                                CgNormType::Natural => {
                                    counters.num_global_reductions += 1;
                                    self.nrm2_scalar(z, comm)
                                }
                            };

                            let (h_rho_next, _) = dot1_async_s(comm, r, z, &opt)?;
                            pending_rho = Some(h_rho_next);
                            counters.residual_replacements = residual_replacements;
                            continue;
                        }

                        counters.residual_replacements = residual_replacements;
                        stats.counters = counters;
                        return Ok(stats);
                    }
                }

                if iterations >= self.conv.max_iters {
                    break;
                }

                if iterations > 0 {
                    let beta = if rho_prev == R::default() {
                        R::default()
                    } else {
                        rho_curr / rho_prev
                    };
                    let beta_s = S::from_real(beta);
                    for i in 0..n {
                        p[i] = z[i] + beta_s * p[i];
                    }
                }

                matvec_s(a, p, &mut ap[..], scratch);

                #[cfg(not(feature = "complex"))]
                let pp_ap_local: R = p
                    .iter()
                    .zip(ap.iter())
                    .fold(R::default(), |acc, (&pi, &api)| acc + pi * api);

                #[cfg(feature = "complex")]
                let pp_ap_local: R = dot_result_to_real(dot_conj(p, ap));

                let (h_ppap, _) = comm.allreduce2_async(pp_ap_local, R::default(), &opt)?;
                let pp_ap = {
                    counters.num_global_reductions += 1;
                    <C as AllreduceOps>::wait_pair(h_ppap).0
                };
                if !pp_ap.is_finite() {
                    return Err(KError::IndefiniteMatrix);
                }
                if pp_ap.abs() <= f64::EPSILON {
                    return Ok(SolveStats::new(
                        iterations,
                        self.nrm2_scalar(r, comm),
                        ConvergedReason::ConvergedHappyBreakdown,
                    )
                    .with_counters({
                        counters.residual_replacements = residual_replacements;
                        counters
                    }));
                }

                let alpha = rho_curr / pp_ap;
                let alpha_s = S::from_real(alpha);
                for i in 0..n {
                    x[i] += alpha_s * p[i];
                    r[i] -= alpha_s * ap[i];
                }

                if let Some(pc) = pc {
                    apply_pc_s(pc, PcSide::Left, r, &mut z[..], scratch)?;
                } else {
                    z.copy_from_slice(r);
                }

                if replace_every > 0 && ((iterations + 1) % replace_every == 0) {
                    matvec_s(a, x, &mut ap[..], scratch);
                    for i in 0..n {
                        r[i] = b[i] - ap[i];
                    }
                    if let Some(pc) = pc {
                        apply_pc_s(pc, PcSide::Left, r, &mut z[..], scratch)?;
                    } else {
                        z.copy_from_slice(r);
                    }
                    residual_replacements += 1;
                    p.copy_from_slice(z);
                    force_restart = true;
                }

                let (h_rho_next, _) = dot1_async_s(comm, r, z, &opt)?;
                pending_rho = Some(h_rho_next);

                if force_restart {
                    rho_prev = R::default();
                    force_restart = false;
                } else {
                    rho_prev = rho_curr;
                }
                iterations += 1;
            }

            if let Some(handle) = pending_rho.take() {
                counters.num_global_reductions += 1;
                rho_curr = <C as AllreduceOps>::wait_pair(handle).0;
                if !rho_curr.is_finite() || rho_curr < R::default() {
                    return Err(KError::IndefinitePreconditioner);
                }

                let _ = match self.norm_type {
                    CgNormType::Preconditioned => rho_curr.sqrt(),
                    CgNormType::Unpreconditioned | CgNormType::None => {
                        let (h_nr, _) = nrm2_async_s(comm, r, &opt);
                        counters.num_global_reductions += 1;
                        <C as AllreduceOps>::wait_pair(h_nr).0.sqrt()
                    }
                    CgNormType::Natural => {
                        let (h_z, _) = nrm2_async_s(comm, z, &opt);
                        counters.num_global_reductions += 1;
                        <C as AllreduceOps>::wait_pair(h_z).0.sqrt()
                    }
                };

                let actual_res = self.nrm2_scalar(r, comm);
                counters.num_global_reductions += 1;

                if call_monitors(
                    monitors,
                    iterations,
                    actual_res,
                    counters.num_global_reductions,
                ) {
                    return Ok(SolveStats::new(
                        iterations,
                        actual_res,
                        ConvergedReason::StoppedByMonitor,
                    )
                    .with_counters({
                        counters.overlap_global_reductions = counters.num_global_reductions;
                        counters
                    }));
                }
                if let Some(m) = &self.true_residual_monitor {
                    m(iterations, actual_res, 0);
                }

                let (reason, mut stats) = self.conv.check(actual_res, rnorm0, iterations);
                if !matches!(reason, ConvergedReason::Continued) {
                    stats.final_residual = actual_res;
                    let tol = self.conv.atol.max(self.conv.rtol * rnorm0);
                    if actual_res > tol {
                        if iterations >= self.conv.max_iters {
                            counters.residual_replacements = residual_replacements;
                            break 'solve;
                        }

                        matvec_s(a, x, &mut ap[..], scratch);
                        for i in 0..n {
                            r[i] = b[i] - ap[i];
                        }
                        if let Some(pc) = pc {
                            apply_pc_s(pc, PcSide::Left, r, &mut z[..], scratch)?;
                        } else {
                            z.copy_from_slice(r);
                        }
                        residual_replacements += 1;
                        p.copy_from_slice(z);
                        rho_prev = R::default();

                        rho_curr = self.dot_scalar(r, z, comm);
                        counters.num_global_reductions += 1;
                        if !rho_curr.is_finite() || rho_curr < R::default() {
                            return Err(KError::IndefinitePreconditioner);
                        }

                        let _ = match self.norm_type {
                            CgNormType::Preconditioned => rho_curr.sqrt(),
                            CgNormType::Unpreconditioned | CgNormType::None => {
                                counters.num_global_reductions += 1;
                                self.nrm2_scalar(r, comm)
                            }
                            CgNormType::Natural => {
                                counters.num_global_reductions += 1;
                                self.nrm2_scalar(z, comm)
                            }
                        };

                        let (h_rho_next, _) = dot1_async_s(comm, r, z, &opt)?;
                        pending_rho = Some(h_rho_next);
                        if iterations < self.conv.max_iters {
                            force_restart = true;
                        }
                        counters.residual_replacements = residual_replacements;
                        continue 'solve;
                    }

                    counters.residual_replacements = residual_replacements;
                    counters.overlap_global_reductions = counters.num_global_reductions;
                    stats.counters = counters;
                    return Ok(stats);
                }
            }

            break 'solve;
        }

        counters.residual_replacements = residual_replacements;
        counters.overlap_global_reductions = counters.num_global_reductions;
        let final_res = self.nrm2_scalar(r, comm);
        Ok(
            SolveStats::new(iterations, final_res, ConvergedReason::DivergedMaxIts).with_counters(
                {
                    counters.overlap_global_reductions = counters.num_global_reductions;
                    counters
                },
            ),
        )
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
        self.solve_k_via_cg(a, pc, b, x, pc_side, comm, monitors, work)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_comm<C: Comm + CommDeterministic + AllreduceOps>(
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
            });
        if let Some(universe) = universe {
            return self.solve_impl(a, pc, b, x, pc_side, &universe, monitors, work);
        }

        match self.variant {
            PcgVariant::Classic => self.solve_classic(a, pc, b, x, pc_side, comm, monitors, work),
            PcgVariant::Pipelined { .. } => {
                self.solve_pipelined(a, pc, b, x, pc_side, comm, monitors, work)
            }
        }
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
            self.solve_k_via_cg(&op, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
        }

        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result =
                self.solve_k_via_cg(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work);
            if result.is_ok() {
                for (dst, src) in x.iter_mut().zip(x_s.iter()) {
                    *dst = src.real();
                }
            }
            result
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_classic<C: Comm + CommDeterministic>(
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
        let pc_ref = pc.as_deref();
        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "CG/PCG requires left preconditioning with HPD M; choose PcSide::Left or use MINRES (Hermitian) / GMRES (general) instead".into(),
            ));
        }

        let n = b.len();
        if x.len() != n {
            return Err(KError::InvalidInput("dimension mismatch: x,b".into()));
        }

        let monitors = monitors.unwrap_or(&[]);

        let mut owned_workspace;
        let work = match work {
            Some(ws) => ws,
            None => {
                owned_workspace = Workspace::new(n);
                &mut owned_workspace
            }
        };

        #[cfg(not(feature = "complex"))]
        let b_slice: &[S] = unsafe { &*(b as *const [f64] as *const [S]) };
        #[cfg(not(feature = "complex"))]
        let x_slice: &mut [S] = unsafe { &mut *(x as *mut [f64] as *mut [S]) };

        #[cfg(feature = "complex")]
        let b_owned: Vec<S> = b.iter().copied().map(S::from_real).collect();
        #[cfg(feature = "complex")]
        let mut x_owned: Vec<S> = x.iter().copied().map(S::from_real).collect();
        #[cfg(feature = "complex")]
        let b_slice: &[S] = &b_owned;
        #[cfg(feature = "complex")]
        let x_slice: &mut [S] = &mut x_owned;

        let stats = self.solve_classic_scalar(a, pc_ref, b_slice, x_slice, comm, monitors, work)?;

        #[cfg(feature = "complex")]
        {
            for (dst, src) in x.iter_mut().zip(x_slice.iter()) {
                *dst = src.real();
            }
        }

        Ok(stats.with_reduction_model(self.reduction_model()))
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_pipelined<C: Comm + CommDeterministic + AllreduceOps>(
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
        let pc_ref = pc.as_deref();
        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "Pipelined PCG requires left preconditioning with HPD M; choose PcSide::Left or use MINRES (Hermitian) / GMRES (general) instead".into(),
            ));
        }

        let n = b.len();
        if x.len() != n {
            return Err(KError::InvalidInput("dimension mismatch: x,b".into()));
        }

        let monitors = monitors.unwrap_or(&[]);

        let mut owned_workspace;
        let work = match work {
            Some(ws) => ws,
            None => {
                owned_workspace = Workspace::new(n);
                &mut owned_workspace
            }
        };

        #[cfg(not(feature = "complex"))]
        let b_slice: &[S] = unsafe { &*(b as *const [f64] as *const [S]) };
        #[cfg(not(feature = "complex"))]
        let x_slice: &mut [S] = unsafe { &mut *(x as *mut [f64] as *mut [S]) };

        #[cfg(feature = "complex")]
        let b_owned: Vec<S> = b.iter().copied().map(S::from_real).collect();
        #[cfg(feature = "complex")]
        let mut x_owned: Vec<S> = x.iter().copied().map(S::from_real).collect();
        #[cfg(feature = "complex")]
        let b_slice: &[S] = &b_owned;
        #[cfg(feature = "complex")]
        let x_slice: &mut [S] = &mut x_owned;

        let stats = self
            .solve_pipelined_scalar(a, pc_ref, b_slice, x_slice, pc_side, comm, monitors, work)?;

        #[cfg(feature = "complex")]
        {
            for (dst, src) in x.iter_mut().zip(x_slice.iter()) {
                *dst = src.real();
            }
        }

        Ok(stats.with_reduction_model(self.reduction_model()))
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
        self.solve_impl(a, pc, b, x, pc_side, comm, monitors, work)
    }
}
