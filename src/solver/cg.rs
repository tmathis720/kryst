//! # CG side semantics
//!
//! * CG/PCG requires **Left** preconditioning with **HPD** `M`.
//! * `PcSide::Left` is enforced even if callers attempt right preconditioning; the solver
//!   returns [`KError::InvalidInput`] otherwise.
//! * In complex builds both `A` and the (optional) preconditioner must be Hermitian
//!   positive definite.
//! * Inner products use conjugation on the first argument. Tiny imaginary drift introduced by
//!   floating-point or MPI reductions is discarded via [`dot_result_to_real`] after a scaled check.
//! * The reported residual norm defaults to the preconditioned norm `sqrt(⟨r,z⟩)`; final stats
//!   include the true `||r||`.
//! * Violations fail fast with structured errors:
//!   - [`KError::InvalidInput`] for wrong `PcSide`,
//!   - [`KError::IndefinitePreconditioner`] if `⟨r,z⟩ ≤ 0`,
//!   - [`KError::IndefiniteMatrix`] if `⟨p,Ap⟩ ≤ 0`.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
use crate::algebra::parallel::{
    dot_conj_local_with_mode, par_axpby, par_axpy, par_copy, sum_abs2_local_with_mode,
};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::config::options::CgVariant;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::Comm;
use crate::parallel::{ReduceHandle, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::MonitorCallback;
use crate::solver::common::{ReductCtx, dot_result_to_real};
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use smallvec::SmallVec;
use std::any::Any;

pub mod debug {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct IterEvent {
        pub iteration: usize,
        pub alpha: R,
        pub beta: Option<R>,
        pub rho: R,
        pub rho_prev: Option<R>,
        pub rho_new: R,
        pub p_ap: R,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DotKind {
        InitialRho,
        PAp,
        Rho,
        RNorm,
        ZNorm,
        Other,
    }

    impl DotKind {
        pub const COUNT: usize = 6;

        #[inline]
        pub fn index(self) -> usize {
            self as usize
        }
    }

    type IterHook = dyn Fn(IterEvent) + Send + Sync + 'static;

    static ITER_HOOK: Mutex<Option<Box<IterHook>>> = Mutex::new(None);
    static ITER_HOOK_SET: AtomicBool = AtomicBool::new(false);
    static COUNTS: [AtomicUsize; DotKind::COUNT] = [
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
    ];
    static MAX_RATIO_X32: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub(crate) fn emit_iter(event: IterEvent) {
        if ITER_HOOK_SET.load(Ordering::Relaxed) {
            if let Some(hook) = ITER_HOOK.lock().unwrap().as_ref() {
                hook(event);
            }
        }
    }

    pub fn set_iter_hook(hook: Option<Box<IterHook>>) {
        let mut guard = ITER_HOOK.lock().unwrap();
        *guard = hook;
        ITER_HOOK_SET.store(guard.is_some(), Ordering::Release);
    }

    pub fn clear_iter_hook() {
        set_iter_hook(None);
    }

    #[inline]
    pub(crate) fn record_dot(kind: DotKind, value: S) {
        #[cfg(feature = "complex")]
        {
            let imag = value.imag().abs();
            let scale = 1.0 + value.abs();
            let ratio = if scale > 0.0 { imag / scale } else { 0.0 };
            if ratio > 128.0 * f64::EPSILON {
                COUNTS[kind.index()].fetch_add(1, Ordering::Relaxed);
            }
            let scaled = (ratio * (u32::MAX as f64)) as u64;
            loop {
                let current = MAX_RATIO_X32.load(Ordering::Relaxed);
                if scaled <= current {
                    break;
                }
                if MAX_RATIO_X32
                    .compare_exchange(current, scaled, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
            }
        }
        #[cfg(not(feature = "complex"))]
        {
            let _ = value;
            let _ = kind;
        }
    }

    pub fn reset_counters() {
        for counter in &COUNTS {
            counter.store(0, Ordering::Relaxed);
        }
        MAX_RATIO_X32.store(0, Ordering::Relaxed);
    }

    pub fn large_imag_count() -> usize {
        COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    pub fn snapshot() -> (usize, [usize; DotKind::COUNT], f64) {
        let mut per_kind = [0usize; DotKind::COUNT];
        for (idx, slot) in per_kind.iter_mut().enumerate() {
            *slot = COUNTS[idx].load(Ordering::Relaxed);
        }
        let total = per_kind.iter().copied().sum();
        let max_ratio = (MAX_RATIO_X32.load(Ordering::Relaxed) as f64) / (u32::MAX as f64);
        (total, per_kind, max_ratio)
    }
}

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::{trace, warn};

#[inline]
fn has_nontrivial_guess(x: &[S]) -> bool {
    let mut max_abs: R = R::zero();
    for &xi in x {
        let v = xi.abs();
        if v > max_abs {
            max_abs = v;
        }
    }
    max_abs > 64.0 * f64::EPSILON
}

struct CgWorkspace<'a> {
    r: &'a mut [S],
    z: &'a mut [S],
    p: &'a mut [S],
    ap: &'a mut [S],
    tmp: &'a mut [S],
    scratch: &'a mut BridgeScratch,
}

impl<'a> CgWorkspace<'a> {
    fn acquire(n: usize, work: &'a mut Workspace) -> Self {
        while work.q_s.len() < 4 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..4] {
            if buf.len() != n {
                buf.resize(n, S::zero());
            }
        }
        if work.tmp1.len() != n {
            work.tmp1.resize(n, S::zero());
        }
        let (q0, rest) = work.q_s.split_at_mut(1);
        let (q1, rest) = rest.split_at_mut(1);
        let (q2, rest) = rest.split_at_mut(1);
        let (q3, _) = rest.split_at_mut(1);
        Self {
            r: &mut q0[0][..n],
            z: &mut q1[0][..n],
            p: &mut q2[0][..n],
            ap: &mut q3[0][..n],
            tmp: &mut work.tmp1[..n],
            scratch: &mut work.bridge,
        }
    }
}

struct CgPipeWorkspace<'a> {
    r: &'a mut [S],
    u: &'a mut [S],
    p: &'a mut [S],
    s: &'a mut [S],
    w: &'a mut [S],
    tmp: &'a mut [S],
    scratch: &'a mut BridgeScratch,
}

impl<'a> CgPipeWorkspace<'a> {
    fn acquire(n: usize, work: &'a mut Workspace) -> Self {
        while work.q_s.len() < 5 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..5] {
            if buf.len() != n {
                buf.resize(n, S::zero());
            }
        }
        if work.tmp1.len() != n {
            work.tmp1.resize(n, S::zero());
        }
        let (q0, rest) = work.q_s.split_at_mut(1);
        let (q1, rest) = rest.split_at_mut(1);
        let (q2, rest) = rest.split_at_mut(1);
        let (q3, rest) = rest.split_at_mut(1);
        let (q4, _) = rest.split_at_mut(1);
        Self {
            r: &mut q0[0][..n],
            u: &mut q1[0][..n],
            p: &mut q2[0][..n],
            s: &mut q3[0][..n],
            w: &mut q4[0][..n],
            tmp: &mut work.tmp1[..n],
            scratch: &mut work.bridge,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CgNormType {
    /// Monitor the preconditioned residual `sqrt(rᵀz)` (default for left PCG)
    Preconditioned,
    /// Monitor the unpreconditioned residual `||r||₂`
    Unpreconditioned,
    /// Monitor the "natural" norm of the preconditioned residual vector
    /// `||z||₂`, matching PETSc's `-ksp_norm_type natural` semantics.
    Natural,
    /// Do not compute or report a residual norm
    None,
}

pub struct CgSolver {
    pub(crate) conv: Convergence,
    norm_type: CgNormType,
    trust_region: Option<R>,
    true_residual_monitor: Option<Box<MonitorCallback<R>>>,
    /// Whether the supplied initial guess in `x` should be treated as
    /// nonzero. When `false` (the default) and `x` is numerically the zero
    /// vector, the solver skips the initial matvec and assumes `x = 0`.
    /// This mirrors PETSc's `KSPSetInitialGuessNonzero` semantics.
    initial_guess_nonzero: bool,
    async_enabled: bool,
    async_min_n: usize,
    variant: CgVariant,
}

impl CgSolver {
    pub fn new(rtol: R, maxits: usize) -> Self {
        Self {
            conv: Convergence {
                rtol,
                atol: 1e-50,
                dtol: 1e5,
                max_iters: maxits,
            },
            // Default monitors use preconditioned norm per policy
            norm_type: CgNormType::Preconditioned,
            trust_region: None,
            true_residual_monitor: None,
            initial_guess_nonzero: false,
            async_enabled: true,
            async_min_n: 10_000,
            variant: CgVariant::Classic,
        }
    }

    pub fn with_norm(mut self, n: CgNormType) -> Self {
        self.norm_type = n;
        self
    }
    pub fn with_trust_region(mut self, r: R) -> Self {
        self.trust_region = Some(r);
        self
    }
    /// Indicate whether the supplied initial guess is nonzero.
    ///
    /// By default `x` is assumed to be the zero vector, avoiding an extra
    /// matvec on entry. Calling this with `true` forces the solver to compute
    /// the initial residual `b - A x` even if `x` happens to be zero.
    pub fn with_nonzero_guess(mut self, f: bool) -> Self {
        self.initial_guess_nonzero = f;
        self
    }
    pub fn with_true_residual_monitor(mut self, m: Box<MonitorCallback<R>>) -> Self {
        self.true_residual_monitor = Some(m);
        self
    }

    pub fn with_variant(mut self, variant: CgVariant) -> Self {
        self.variant = variant;
        self
    }

    pub fn set_norm(&mut self, n: CgNormType) {
        self.norm_type = n;
    }
    pub fn set_trust_region(&mut self, r: R) {
        self.trust_region = Some(r);
    }
    /// Select the CG algorithm variant.
    pub fn set_variant(&mut self, variant: CgVariant) {
        self.variant = variant;
    }
    /// Return the active CG variant.
    pub fn variant(&self) -> CgVariant {
        self.variant
    }
    #[inline]
    fn attach_drift_stats(mut stats: SolveStats<R>) -> SolveStats<R> {
        let (total, per_kind, max_ratio) = debug::snapshot();
        stats.complex_drift_events = total;
        stats.complex_drift_counts = per_kind;
        stats.complex_drift_max_rel = max_ratio;
        #[cfg(feature = "logging")]
        if total > 0 {
            warn!(
                "CG: complex drift observed: total={}, per_kind={:?}, max_rel_imag={:.3e}",
                total, per_kind, max_ratio
            );
        }
        stats
    }

    #[inline(always)]
    fn prefetch_like<T>(slice: &[T]) {
        let _ = core::hint::black_box(slice.as_ptr());
    }
    pub fn set_async_enabled(&mut self, enabled: bool) {
        self.async_enabled = enabled;
    }
    pub fn set_async_min_n(&mut self, n: usize) {
        self.async_min_n = n;
    }
    /// Set the nonzero initial guess flag after construction.
    pub fn set_nonzero_guess(&mut self, f: bool) {
        self.initial_guess_nonzero = f;
    }
    pub fn set_true_residual_monitor(&mut self, m: Option<Box<MonitorCallback<R>>>) {
        self.true_residual_monitor = m;
    }

    #[inline]
    fn should_use_async(&self, comm: &UniverseComm, n: usize) -> bool {
        self.async_enabled && comm.size() > 1 && n >= self.async_min_n
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_comm<A>(
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
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CG");

        enum PapRequest {
            None {
                p_ap: S,
                pnorm_sq: Option<R>,
            },
            Handle {
                handle: ReduceHandle<Vec<R>>,
                need_pnorm: bool,
            },
        }

        enum RhoRequest {
            None,
            Handle {
                handle: ReduceHandle<Vec<R>>,
                count: usize,
            },
        }

        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "CG/PCG requires left preconditioning with HPD M; choose PcSide::Left or use MINRES (Hermitian) / GMRES (general) instead".into(),
            ));
        }

        let (nrows, ncols) = a.dims();
        if nrows != ncols || b.len() != nrows || x.len() != ncols {
            return Err(KError::InvalidInput("dimension mismatch x,b".into()));
        }

        let work = work.ok_or_else(|| {
            KError::InvalidInput("CG requires a Workspace; use KSP or Workspace::new(n)".into())
        })?;
        let red = ReductCtx::new(comm, Some(&*work));
        let red_mode = red.mode();

        if b.is_empty() {
            return Ok(Self::attach_drift_stats(SolveStats::new(
                0,
                R::zero(),
                ConvergedReason::ConvergedAtol,
            )));
        }

        if matches!(self.variant, CgVariant::Pipelined) {
            return self.solve_pipelined_with_comm(a, pc, b, x, comm, monitors, work, nrows);
        }

        let mut buffers = CgWorkspace::acquire(nrows, work);
        let CgWorkspace {
            r,
            z,
            p,
            ap,
            tmp,
            scratch,
        } = &mut buffers;

        let guess_nonzero = self.initial_guess_nonzero || has_nontrivial_guess(x);
        if guess_nonzero {
            a.matvec_s(x, &mut tmp[..], scratch);
            for i in 0..nrows {
                r[i] = b[i] - tmp[i];
            }
        } else {
            par_copy(b, r);
        }

        if let Some(pc) = pc {
            pc.apply_s(PcSide::Left, r, &mut z[..], scratch)?;
        } else {
            par_copy(r, z);
        }

        let want_unpre = matches!(self.norm_type, CgNormType::Unpreconditioned);
        let want_natural = matches!(self.norm_type, CgNormType::Natural);

        let (mut rho, rsq, znorm) = {
            let mut dot_pairs: SmallVec<[(&[S], &[S]); 3]> = SmallVec::new();
            dot_pairs.push((&r[..], &z[..]));
            if want_unpre {
                dot_pairs.push((&r[..], &r[..]));
            }
            if want_natural {
                dot_pairs.push((&z[..], &z[..]));
            }

            let mut dot_results: SmallVec<[S; 3]> = SmallVec::new();
            dot_results.resize(dot_pairs.len(), S::zero());
            red.dot_many_into(dot_pairs.as_slice(), dot_results.as_mut_slice());

            let mut result_idx = 0usize;
            let rho_scalar = dot_results[result_idx];
            debug::record_dot(debug::DotKind::InitialRho, rho_scalar);
            let rho: R = dot_result_to_real(rho_scalar);
            result_idx += 1;

            let rsq = if want_unpre {
                let value = dot_results[result_idx];
                debug::record_dot(debug::DotKind::RNorm, value);
                result_idx += 1;
                Some(dot_result_to_real(value))
            } else {
                None
            };
            let znorm = if want_natural {
                let value = dot_results[result_idx];
                debug::record_dot(debug::DotKind::ZNorm, value);
                Some(dot_result_to_real(value))
            } else {
                None
            };
            (rho, rsq, znorm)
        };
        let mut rho_prev: R = rho;
        if rho < R::zero() || !rho.is_finite() {
            return Err(KError::IndefinitePreconditioner);
        }
        let mut xnorm = if self.trust_region.is_some() {
            red.norm2(x)
        } else {
            R::zero()
        };

        let res0_reported: R = match self.norm_type {
            CgNormType::Preconditioned => rho.abs().sqrt(),
            CgNormType::Unpreconditioned => rsq.unwrap().abs().sqrt(),
            CgNormType::Natural => znorm.unwrap().abs().sqrt(),
            CgNormType::None => R::zero(),
        };
        let zero_floor = self.conv.atol.max(self.conv.rtol * res0_reported) * R::from(1e-5);

        if let Some(ms) = monitors {
            for m in ms {
                let _ = m(0, res0_reported, 0);
            }
        }
        if let Some(m) = &self.true_residual_monitor {
            let true_res = red.norm2(r);
            let _ = m(0, true_res, 0);
        }
        #[cfg(feature = "logging")]
        trace!("CG initial residual: {res0_reported:.3e}");

        par_copy(z, p);

        let mut stats = SolveStats::new(0, res0_reported, ConvergedReason::Continued);

        let (reason0, s0) = self.conv.check(res0_reported, res0_reported, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let mut s = s0;
            s.final_residual = red.norm2(r);
            if s.final_residual <= zero_floor {
                s.final_residual = R::zero();
            }
            return Ok(Self::attach_drift_stats(s));
        }

        for k in 1..=self.conv.max_iters {
            let beta_value = if k > 1 { Some(rho / rho_prev) } else { None };

            if let Some(beta) = beta_value {
                let beta_s: S = S::from_real(beta);
                par_axpby(z, S::one(), p, beta_s);
            } else {
                par_copy(z, p);
            }

            a.matvec_s(p, &mut ap[..], scratch);

            let async_ok = self.should_use_async(comm, nrows);
            let need_pnorm = self.trust_region.is_some();
            let local_pap = dot_conj_local_with_mode(p, ap, red_mode);
            let local_pnorm_sq = if need_pnorm {
                sum_abs2_local_with_mode(p, red_mode)
            } else {
                R::zero()
            };

            let pap_req = if async_ok {
                #[cfg(feature = "complex")]
                let mut payload = Vec::with_capacity(if need_pnorm { 3 } else { 2 });
                #[cfg(not(feature = "complex"))]
                let mut payload = Vec::with_capacity(if need_pnorm { 2 } else { 1 });
                #[cfg(feature = "complex")]
                {
                    payload.push(local_pap.real());
                    payload.push(local_pap.imag());
                }
                #[cfg(not(feature = "complex"))]
                {
                    payload.push(local_pap.real());
                }
                if need_pnorm {
                    payload.push(local_pnorm_sq);
                }
                PapRequest::Handle {
                    handle: red.engine().iallreduce_sum_vec_r(payload),
                    need_pnorm,
                }
            } else {
                let p_ap = red.engine().allreduce_sum_s(local_pap);
                let pnorm_sq = if need_pnorm {
                    Some(red.engine().allreduce_sum_r(local_pnorm_sq))
                } else {
                    None
                };
                PapRequest::None { p_ap, pnorm_sq }
            };

            Self::prefetch_like(&z[..]);
            Self::prefetch_like(&r[..]);

            let (p_ap_scalar, pnorm_sq_opt) = match pap_req {
                PapRequest::None { p_ap, pnorm_sq } => (p_ap, pnorm_sq),
                PapRequest::Handle { handle, need_pnorm } => {
                    let reduced = handle.wait();
                    #[cfg(feature = "complex")]
                    let (p_ap, offset) = (S::from_parts(reduced[0], reduced[1]), 2);
                    #[cfg(not(feature = "complex"))]
                    let (p_ap, offset) = (S::from_real(reduced[0]), 1);
                    let pnorm_sq = if need_pnorm {
                        Some(reduced[offset])
                    } else {
                        None
                    };
                    (p_ap, pnorm_sq)
                }
            };

            debug::record_dot(debug::DotKind::PAp, p_ap_scalar);
            let p_ap: R = dot_result_to_real(p_ap_scalar);
            if p_ap <= R::zero() || !p_ap.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }

            let pnorm_opt = pnorm_sq_opt.map(|v| v.max(R::zero()).sqrt());
            let alpha: R = rho / p_ap;
            let alpha_s: S = S::from_real(alpha);

            if let Some(rmax) = self.trust_region {
                let pnorm = pnorm_opt.unwrap_or_else(|| {
                    red.engine()
                        .allreduce_sum_r(local_pnorm_sq)
                        .max(R::zero())
                        .sqrt()
                });
                if xnorm + alpha.abs() * pnorm > rmax {
                    let step: R = (rmax - xnorm) / (pnorm + 1e-300);
                    let step_s: S = S::from_real(step);
                    par_axpy(p, step_s, x);
                    par_axpy(ap, -step_s, r);
                    stats.iterations = k;
                    stats.reason = ConvergedReason::ConvergedTrustRegion;
                    stats.final_residual = red.norm2(r);
                    return Ok(Self::attach_drift_stats(stats));
                }
            }

            par_axpy(p, alpha_s, x);
            par_axpy(ap, -alpha_s, r);
            if self.trust_region.is_some() {
                xnorm = red.norm2(x);
            }

            if let Some(pc) = pc {
                pc.apply_s(PcSide::Left, r, &mut z[..], scratch)?;
            } else {
                par_copy(r, z);
            }

            let want_unpre = matches!(self.norm_type, CgNormType::Unpreconditioned);
            let want_natural = matches!(self.norm_type, CgNormType::Natural);
            let local_rz = dot_conj_local_with_mode(r, z, red_mode);

            let mut dot_results: SmallVec<[S; 3]> = SmallVec::new();
            dot_results.push(local_rz);
            let rho_idx = 0usize;
            let rsq_idx = if want_unpre {
                dot_results.push(dot_conj_local_with_mode(r, r, red_mode));
                Some(dot_results.len() - 1)
            } else {
                None
            };
            let znorm_idx = if want_natural {
                dot_results.push(dot_conj_local_with_mode(z, z, red_mode));
                Some(dot_results.len() - 1)
            } else {
                None
            };

            let rho_req = if async_ok {
                #[cfg(feature = "complex")]
                let mut payload = Vec::with_capacity(dot_results.len() * 2);
                #[cfg(not(feature = "complex"))]
                let mut payload = Vec::with_capacity(dot_results.len());
                for value in dot_results.iter().copied() {
                    #[cfg(feature = "complex")]
                    {
                        payload.push(value.real());
                        payload.push(value.imag());
                    }
                    #[cfg(not(feature = "complex"))]
                    {
                        payload.push(value.real());
                    }
                }
                RhoRequest::Handle {
                    handle: red.engine().iallreduce_sum_vec_r(payload),
                    count: dot_results.len(),
                }
            } else if want_unpre || want_natural {
                let mut dot_pairs: SmallVec<[(&[S], &[S]); 3]> = SmallVec::new();
                dot_pairs.push((&r[..], &z[..]));
                if want_unpre {
                    dot_pairs.push((&r[..], &r[..]));
                }
                if want_natural {
                    dot_pairs.push((&z[..], &z[..]));
                }
                red.dot_many_into(dot_pairs.as_slice(), dot_results.as_mut_slice());
                RhoRequest::None
            } else {
                dot_results[rho_idx] = red.dot(r, z);
                RhoRequest::None
            };

            Self::prefetch_like(&p[..]);
            Self::prefetch_like(&x[..]);

            if let RhoRequest::Handle { handle, count } = rho_req {
                let reduced = handle.wait();
                #[cfg(feature = "complex")]
                {
                    let _ = count;
                    for (slot, chunk) in dot_results.iter_mut().zip(reduced.chunks_exact(2)) {
                        *slot = S::from_parts(chunk[0], chunk[1]);
                    }
                }
                #[cfg(not(feature = "complex"))]
                {
                    debug_assert_eq!(reduced.len(), count);
                    for (slot, value) in dot_results.iter_mut().zip(reduced.into_iter()) {
                        *slot = S::from_real(value);
                    }
                }
            }

            let rho_scalar = dot_results[rho_idx];
            debug::record_dot(debug::DotKind::Rho, rho_scalar);
            let rho_new: R = dot_result_to_real(rho_scalar);
            if rho_new < R::zero() || !rho_new.is_finite() {
                return Err(KError::IndefinitePreconditioner);
            }

            let rsq_new = if let Some(idx) = rsq_idx {
                let value = dot_results[idx];
                debug::record_dot(debug::DotKind::RNorm, value);
                Some(dot_result_to_real(value))
            } else {
                None
            };
            let znorm_new = if let Some(idx) = znorm_idx {
                let value = dot_results[idx];
                debug::record_dot(debug::DotKind::ZNorm, value);
                Some(dot_result_to_real(value))
            } else {
                None
            };

            debug::emit_iter(debug::IterEvent {
                iteration: k,
                alpha,
                beta: beta_value,
                rho,
                rho_prev: if k > 1 { Some(rho_prev) } else { None },
                rho_new,
                p_ap,
            });

            let res_reported: R = match self.norm_type {
                CgNormType::Preconditioned => rho_new.abs().sqrt(),
                CgNormType::Unpreconditioned => rsq_new.unwrap().abs().sqrt(),
                CgNormType::Natural => znorm_new.unwrap().abs().sqrt(),
                CgNormType::None => R::zero(),
            };

            if let Some(ms) = monitors {
                for m in ms {
                    let _ = m(k, res_reported, 0);
                }
            }
            if let Some(m) = &self.true_residual_monitor {
                let true_res = red.norm2(r);
                let _ = m(k, true_res, 0);
            }

            let (reason, mut s) = self.conv.check(res_reported, res0_reported, k);
            if !matches!(reason, ConvergedReason::Continued) {
                s.final_residual = red.norm2(r);
                if s.final_residual <= zero_floor {
                    s.final_residual = R::zero();
                }
                return Ok(Self::attach_drift_stats(s));
            }

            rho_prev = rho;
            rho = rho_new;
            stats.iterations = k;
            stats.final_residual = res_reported;
        }

        let true_res = red.norm2(r);
        Ok(Self::attach_drift_stats(SolveStats::new(
            self.conv.max_iters,
            true_res,
            ConvergedReason::DivergedMaxIts,
        )))
    }

    fn solve_pipelined_with_comm<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: &mut Workspace,
        nrows: usize,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        let red = ReductCtx::new(comm, Some(&*work));
        let red_mode = red.mode();
        let mut buffers = CgPipeWorkspace::acquire(nrows, work);
        let CgPipeWorkspace {
            r,
            u,
            p,
            s,
            w,
            tmp,
            scratch,
        } = &mut buffers;

        let guess_nonzero = self.initial_guess_nonzero || has_nontrivial_guess(x);
        if guess_nonzero {
            a.matvec_s(x, &mut tmp[..], scratch);
            for i in 0..nrows {
                r[i] = b[i] - tmp[i];
            }
        } else {
            par_copy(b, r);
        }

        if let Some(pc) = pc {
            pc.apply_s(PcSide::Left, r, &mut u[..], scratch)?;
        } else {
            par_copy(r, u);
        }
        a.matvec_s(u, &mut w[..], scratch);

        let want_unpre = matches!(self.norm_type, CgNormType::Unpreconditioned);
        let want_natural = matches!(self.norm_type, CgNormType::Natural);

        let (gamma_scalar, delta_scalar, rsq, znorm) = {
            let mut pairs: SmallVec<[(&[S], &[S]); 4]> = SmallVec::new();
            pairs.push((r, u));
            let gamma_idx = 0usize;
            pairs.push((u, w));
            let delta_idx = 1usize;
            let rsq_idx = if want_unpre {
                pairs.push((r, r));
                Some(pairs.len() - 1)
            } else {
                None
            };
            let znorm_idx = if want_natural {
                pairs.push((u, u));
                Some(pairs.len() - 1)
            } else {
                None
            };
            let mut scalars: SmallVec<[S; 4]> = SmallVec::new();
            scalars.resize(pairs.len(), S::zero());
            red.dot_many_into(pairs.as_slice(), scalars.as_mut_slice());

            let gamma_scalar = scalars[gamma_idx];
            debug::record_dot(debug::DotKind::InitialRho, gamma_scalar);

            let delta_scalar = scalars[delta_idx];
            debug::record_dot(debug::DotKind::PAp, delta_scalar);

            let rsq = rsq_idx.map(|idx| {
                let value = scalars[idx];
                debug::record_dot(debug::DotKind::RNorm, value);
                dot_result_to_real(value)
            });
            let znorm = znorm_idx.map(|idx| {
                let value = scalars[idx];
                debug::record_dot(debug::DotKind::ZNorm, value);
                dot_result_to_real(value)
            });
            (gamma_scalar, delta_scalar, rsq, znorm)
        };

        let mut rho: R = dot_result_to_real(gamma_scalar);
        if rho <= R::zero() || !rho.is_finite() {
            return Err(KError::IndefinitePreconditioner);
        }

        let mut delta: R = dot_result_to_real(delta_scalar);
        if delta <= R::zero() || !delta.is_finite() {
            return Err(KError::IndefiniteMatrix);
        }
        let mut alpha: R = rho / delta;

        par_copy(u, p);
        par_copy(w, s);

        let mut xnorm = if self.trust_region.is_some() {
            red.norm2(x)
        } else {
            R::zero()
        };

        let res0_reported: R = match self.norm_type {
            CgNormType::Preconditioned => rho.abs().sqrt(),
            CgNormType::Unpreconditioned => rsq.unwrap().abs().sqrt(),
            CgNormType::Natural => znorm.unwrap().abs().sqrt(),
            CgNormType::None => R::zero(),
        };
        let zero_floor = self.conv.atol.max(self.conv.rtol * res0_reported) * R::from(1e-5);

        if let Some(ms) = monitors {
            for m in ms {
                let _ = m(0, res0_reported, 0);
            }
        }
        if let Some(m) = &self.true_residual_monitor {
            let true_res = red.norm2(r);
            let _ = m(0, true_res, 0);
        }
        #[cfg(feature = "logging")]
        trace!("CG initial residual: {res0_reported:.3e}");

        let mut stats = SolveStats::new(0, res0_reported, ConvergedReason::Continued);

        let (reason0, s0) = self.conv.check(res0_reported, res0_reported, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let mut s_out = s0;
            s_out.final_residual = red.norm2(r);
            if s_out.final_residual <= zero_floor {
                s_out.final_residual = R::zero();
            }
            return Ok(Self::attach_drift_stats(s_out));
        }

        let mut rho_prev = rho;

        for k in 1..=self.conv.max_iters {
            let alpha_s: S = S::from_real(alpha);
            let local_pnorm_sq = if self.trust_region.is_some() {
                sum_abs2_local_with_mode(p, red_mode)
            } else {
                R::zero()
            };

            if let Some(rmax) = self.trust_region {
                let pnorm = red
                    .engine()
                    .allreduce_sum_r(local_pnorm_sq)
                    .max(R::zero())
                    .sqrt();
                if xnorm + alpha.abs() * pnorm > rmax {
                    let step: R = (rmax - xnorm) / (pnorm + 1e-300);
                    let step_s: S = S::from_real(step);
                    par_axpy(p, step_s, x);
                    par_axpy(s, -step_s, r);
                    stats.iterations = k;
                    stats.reason = ConvergedReason::ConvergedTrustRegion;
                    stats.final_residual = red.norm2(r);
                    return Ok(Self::attach_drift_stats(stats));
                }
            }

            par_axpy(p, alpha_s, x);
            par_axpy(s, -alpha_s, r);
            if self.trust_region.is_some() {
                xnorm = red.norm2(x);
            }

            if let Some(pc) = pc {
                pc.apply_s(PcSide::Left, r, &mut u[..], scratch)?;
            } else {
                par_copy(r, u);
            }
            a.matvec_s(u, &mut w[..], scratch);

            let mut tuple: SmallVec<[S; 4]> = SmallVec::new();
            tuple.push(dot_conj_local_with_mode(r, u, red_mode));
            let rho_idx = 0usize;
            tuple.push(dot_conj_local_with_mode(u, w, red_mode));
            let delta_idx = 1usize;
            let rsq_idx = if want_unpre {
                tuple.push(dot_conj_local_with_mode(r, r, red_mode));
                Some(tuple.len() - 1)
            } else {
                None
            };
            let znorm_idx = if want_natural {
                tuple.push(dot_conj_local_with_mode(u, u, red_mode));
                Some(tuple.len() - 1)
            } else {
                None
            };

            let async_ok = self.should_use_async(comm, nrows);
            #[cfg(feature = "complex")]
            let mut payload = Vec::with_capacity(tuple.len() * 2);
            #[cfg(not(feature = "complex"))]
            let mut payload = Vec::with_capacity(tuple.len());
            for value in tuple.iter().copied() {
                #[cfg(feature = "complex")]
                {
                    payload.push(value.real());
                    payload.push(value.imag());
                }
                #[cfg(not(feature = "complex"))]
                {
                    payload.push(value.real());
                }
            }

            let reduced = if async_ok {
                let handle = red.engine().iallreduce_sum_vec_r(payload);
                Self::prefetch_like(&p[..]);
                Self::prefetch_like(&x[..]);
                handle.wait()
            } else {
                red.engine().sum_vec_r(payload)
            };

            #[cfg(feature = "complex")]
            {
                for (slot, chunk) in tuple.iter_mut().zip(reduced.chunks_exact(2)) {
                    *slot = S::from_parts(chunk[0], chunk[1]);
                }
            }
            #[cfg(not(feature = "complex"))]
            {
                for (slot, value) in tuple.iter_mut().zip(reduced.into_iter()) {
                    *slot = S::from_real(value);
                }
            }

            let rho_scalar = tuple[rho_idx];
            debug::record_dot(debug::DotKind::Rho, rho_scalar);
            let rho_new: R = dot_result_to_real(rho_scalar);
            if rho_new < R::zero() || !rho_new.is_finite() {
                return Err(KError::IndefinitePreconditioner);
            }

            let delta_scalar = tuple[delta_idx];
            debug::record_dot(debug::DotKind::PAp, delta_scalar);
            let delta_new: R = dot_result_to_real(delta_scalar);

            let rsq_new = if let Some(idx) = rsq_idx {
                let value = tuple[idx];
                debug::record_dot(debug::DotKind::RNorm, value);
                Some(dot_result_to_real(value))
            } else {
                None
            };
            let znorm_new = if let Some(idx) = znorm_idx {
                let value = tuple[idx];
                debug::record_dot(debug::DotKind::ZNorm, value);
                Some(dot_result_to_real(value))
            } else {
                None
            };

            let beta: R = rho_new / rho;
            let beta_s: S = S::from_real(beta);
            par_axpy(p, beta_s, u);
            par_copy(u, p);
            par_axpy(s, beta_s, w);
            par_copy(w, s);

            debug::emit_iter(debug::IterEvent {
                iteration: k,
                alpha,
                beta: Some(beta),
                rho,
                rho_prev: if k > 1 { Some(rho_prev) } else { None },
                rho_new,
                p_ap: delta,
            });

            let res_reported: R = match self.norm_type {
                CgNormType::Preconditioned => rho_new.abs().sqrt(),
                CgNormType::Unpreconditioned => rsq_new.unwrap().abs().sqrt(),
                CgNormType::Natural => znorm_new.unwrap().abs().sqrt(),
                CgNormType::None => R::zero(),
            };

            if let Some(ms) = monitors {
                for m in ms {
                    let _ = m(k, res_reported, 0);
                }
            }
            if let Some(m) = &self.true_residual_monitor {
                let true_res = red.norm2(r);
                let _ = m(k, true_res, 0);
            }

            let (reason, mut s_out) = self.conv.check(res_reported, res0_reported, k);
            if !matches!(reason, ConvergedReason::Continued) {
                s_out.final_residual = red.norm2(r);
                if s_out.final_residual <= zero_floor {
                    s_out.final_residual = R::zero();
                }
                return Ok(Self::attach_drift_stats(s_out));
            }

            if delta_new <= R::zero() || !delta_new.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }

            let denom: R = delta_new - (beta / alpha) * rho_new;
            if denom <= R::zero() || !denom.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }

            rho_prev = rho;
            rho = rho_new;
            delta = delta_new;
            alpha = rho / denom;
            stats.iterations = k;
            stats.final_residual = res_reported;
        }

        let true_res = red.norm2(r);
        Ok(Self::attach_drift_stats(SolveStats::new(
            self.conv.max_iters,
            true_res,
            ConvergedReason::DivergedMaxIts,
        )))
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
        self.solve_with_comm(a, pc, b, x, pc_side, comm, monitors, work)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_f64<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn PreconditionerF64>,
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

impl LinearSolver for CgSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        let required = if matches!(self.variant, CgVariant::Pipelined) {
            5
        } else {
            4
        };
        if work.q_s.len() < required {
            work.q_s.resize(required, Vec::new());
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        self.solve_f64(a, pc.as_deref(), b, x, pc_side, comm, monitors, work)
    }
}
