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
use crate::algebra::parallel::{par_axpby, par_axpy, par_copy};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::{UniverseComm, global_dot_conj, global_dot_conj_many_into, global_nrm2};
use crate::preconditioner::{PcSide, Preconditioner, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::common::dot_result_to_real;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use smallvec::SmallVec;
use std::any::Any;

pub mod debug {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    }

    type IterHook = dyn Fn(IterEvent) + Send + Sync + 'static;

    static ITER_HOOK: Mutex<Option<Box<IterHook>>> = Mutex::new(None);
    static ITER_HOOK_SET: AtomicBool = AtomicBool::new(false);
    static LARGE_IMAG_COUNT: AtomicUsize = AtomicUsize::new(0);

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
    pub(crate) fn record_dot(kind: DotKind, _iteration: usize, value: S) {
        #[cfg(feature = "complex")]
        {
            let imag = value.imag().abs();
            let scale = 1.0 + value.abs();
            if imag > 128.0 * f64::EPSILON * scale {
                LARGE_IMAG_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }
        #[cfg(not(feature = "complex"))]
        let _ = value;
        let _ = kind;
    }

    pub fn reset_counters() {
        LARGE_IMAG_COUNT.store(0, Ordering::Relaxed);
    }

    pub fn large_imag_count() -> usize {
        LARGE_IMAG_COUNT.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::trace;

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
    true_residual_monitor: Option<Box<dyn Fn(usize, R) + Send + Sync>>,
    /// Whether the supplied initial guess in `x` should be treated as
    /// nonzero. When `false` (the default) and `x` is numerically the zero
    /// vector, the solver skips the initial matvec and assumes `x = 0`.
    /// This mirrors PETSc's `KSPSetInitialGuessNonzero` semantics.
    initial_guess_nonzero: bool,
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
    pub fn with_true_residual_monitor(mut self, m: Box<dyn Fn(usize, R) + Send + Sync>) -> Self {
        self.true_residual_monitor = Some(m);
        self
    }

    pub fn set_norm(&mut self, n: CgNormType) {
        self.norm_type = n;
    }
    pub fn set_trust_region(&mut self, r: R) {
        self.trust_region = Some(r);
    }
    /// Set the nonzero initial guess flag after construction.
    pub fn set_nonzero_guess(&mut self, f: bool) {
        self.initial_guess_nonzero = f;
    }
    pub fn set_true_residual_monitor(&mut self, m: Option<Box<dyn Fn(usize, R) + Send + Sync>>) {
        self.true_residual_monitor = m;
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
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CG");

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

        if b.is_empty() {
            return Ok(SolveStats::new(
                0,
                R::zero(),
                ConvergedReason::ConvergedAtol,
            ));
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
            global_dot_conj_many_into(comm, dot_pairs.as_slice(), dot_results.as_mut_slice());

            let mut result_idx = 0usize;
            let rho_scalar = dot_results[result_idx];
            debug::record_dot(debug::DotKind::InitialRho, 0, rho_scalar);
            let rho: R = dot_result_to_real(rho_scalar);
            result_idx += 1;

            let rsq = if want_unpre {
                let value = dot_results[result_idx];
                debug::record_dot(debug::DotKind::RNorm, 0, value);
                result_idx += 1;
                Some(dot_result_to_real(value))
            } else {
                None
            };
            let znorm = if want_natural {
                let value = dot_results[result_idx];
                debug::record_dot(debug::DotKind::ZNorm, 0, value);
                Some(dot_result_to_real(value))
            } else {
                None
            };
            (rho, rsq, znorm)
        };
        let mut rho_prev: R = rho;
        if rho <= R::zero() || !rho.is_finite() {
            return Err(KError::IndefinitePreconditioner);
        }
        let mut xnorm = if self.trust_region.is_some() {
            global_nrm2(comm, x)
        } else {
            R::zero()
        };

        let res0_reported: R = match self.norm_type {
            CgNormType::Preconditioned => rho.abs().sqrt(),
            CgNormType::Unpreconditioned => rsq.unwrap().abs().sqrt(),
            CgNormType::Natural => znorm.unwrap().abs().sqrt(),
            CgNormType::None => R::zero(),
        };

        if let Some(ms) = monitors {
            for m in ms {
                m(0, res0_reported);
            }
        }
        if let Some(m) = &self.true_residual_monitor {
            let true_res = global_nrm2(comm, r);
            m(0, true_res);
        }
        #[cfg(feature = "logging")]
        trace!("CG initial residual: {res0_reported:.3e}");

        par_copy(z, p);

        let mut stats = SolveStats::new(0, res0_reported, ConvergedReason::Continued);

        let (reason0, s0) = self.conv.check(res0_reported, res0_reported, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let mut s = s0;
            s.final_residual = global_nrm2(comm, r);
            return Ok(s);
        }

        for k in 1..=self.conv.max_iters {
            let beta_value = if k > 1 { Some(rho / rho_prev) } else { None };

            if let Some(beta) = beta_value {
                let beta_s: S = S::from_real(beta);
                par_axpby(z, S::one(), p, beta_s);
            }

            a.matvec_s(p, &mut ap[..], scratch);

            let p_ap_scalar = global_dot_conj(comm, p, ap);
            debug::record_dot(debug::DotKind::PAp, k, p_ap_scalar);
            let p_ap: R = dot_result_to_real(p_ap_scalar);
            if p_ap <= R::zero() || !p_ap.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }

            let alpha: R = rho / p_ap;
            let alpha_s: S = S::from_real(alpha);

            if let Some(rmax) = self.trust_region {
                let pnorm = global_nrm2(comm, p);
                if xnorm + alpha.abs() * pnorm > rmax {
                    let step: R = (rmax - xnorm) / (pnorm + 1e-300);
                    let step_s: S = S::from_real(step);
                    par_axpy(p, step_s, x);
                    par_axpy(ap, -step_s, r);
                    stats.iterations = k;
                    stats.reason = ConvergedReason::ConvergedTrustRegion;
                    stats.final_residual = global_nrm2(comm, r);
                    return Ok(stats);
                }
            }

            par_axpy(p, alpha_s, x);
            par_axpy(ap, -alpha_s, r);
            if self.trust_region.is_some() {
                xnorm = global_nrm2(comm, x);
            }

            if let Some(pc) = pc {
                pc.apply_s(PcSide::Left, r, &mut z[..], scratch)?;
            } else {
                par_copy(r, z);
            }

            let (rho_new, rsq_new, znorm_new) = {
                let want_unpre = matches!(self.norm_type, CgNormType::Unpreconditioned);
                let want_natural = matches!(self.norm_type, CgNormType::Natural);

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
                global_dot_conj_many_into(comm, dot_pairs.as_slice(), dot_results.as_mut_slice());

                let mut result_idx = 0usize;
                let rho_scalar = dot_results[result_idx];
                debug::record_dot(debug::DotKind::Rho, k, rho_scalar);
                let rho_new: R = dot_result_to_real(rho_scalar);
                result_idx += 1;
                if rho_new <= R::zero() || !rho_new.is_finite() {
                    return Err(KError::IndefinitePreconditioner);
                }

                let rsq_new = if want_unpre {
                    let value = dot_results[result_idx];
                    debug::record_dot(debug::DotKind::RNorm, k, value);
                    result_idx += 1;
                    Some(dot_result_to_real(value))
                } else {
                    None
                };
                let znorm_new = if want_natural {
                    let value = dot_results[result_idx];
                    debug::record_dot(debug::DotKind::ZNorm, k, value);
                    Some(dot_result_to_real(value))
                } else {
                    None
                };
                (rho_new, rsq_new, znorm_new)
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
                    m(k, res_reported);
                }
            }
            if let Some(m) = &self.true_residual_monitor {
                let true_res = global_nrm2(comm, r);
                m(k, true_res);
            }

            let (reason, mut s) = self.conv.check(res_reported, res0_reported, k);
            if !matches!(reason, ConvergedReason::Continued) {
                s.final_residual = global_nrm2(comm, r);
                return Ok(s);
            }

            rho_prev = rho;
            rho = rho_new;
            stats.iterations = k;
            stats.final_residual = res_reported;
        }

        let true_res = global_nrm2(comm, r);
        Ok(SolveStats::new(
            self.conv.max_iters,
            true_res,
            ConvergedReason::DivergedMaxIts,
        ))
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
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
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
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
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
        if work.q_s.len() < 4 {
            work.q_s.resize(4, Vec::new());
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
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        self.solve_f64(a, pc.as_deref(), b, x, pc_side, comm, monitors, work)
    }
}
