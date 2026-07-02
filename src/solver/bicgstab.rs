//!
//! Robust breakdown checks:
//!   - |rho|   <= eps_rho      → DivergedDtol
//!   - |alpha_den| <= eps_alpha → DivergedDtol
//!   - |omega_den| <= eps_omega → DivergedDtol
//!   - |omega| <= eps_omega     → DivergedDtol
//!
//! Notes for later (PC integration):
//!   Left  : r̃ = M^{-1} r,    p = r̃ + β (p − ω ṽ), ṽ = M^{-1} v = M^{-1} A p
//!   Right : keep r (true),    use z = M^{-1} p in A·z, and x ← x + α z + ω z_s, etc.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
#[cfg(not(feature = "complex"))]
use crate::matrix::format::OpFormat;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::MonitorCallback;
#[cfg(not(feature = "complex"))]
use crate::solver::common::dot2_async_s;
use crate::solver::common::exit_checks::{
    reconcile_reason_with_true_residual, true_residual_converged_reason, true_residual_norm,
};
use crate::solver::common::{ReductCtx, call_monitors};
use crate::utils::convergence::{
    AcceptanceStatus, ConvergedReason, ReasonEmitter, ReductionModel, SolveStats, SolverCounters,
};
#[cfg(not(feature = "complex"))]
use crate::utils::reduction::{AllreduceHandle, AllreduceOps, ReductOptions};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::trace;

#[cfg(not(feature = "complex"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RealDispatchRoute {
    DistCsrBorrowed,
    CsrBorrowed,
    #[cfg(feature = "backend-faer")]
    CsrMaterialized,
    Generic,
}

struct BiCgWorkspace<'a> {
    r: &'a mut [S],
    r_hat: &'a mut [S],
    v: &'a mut [S],
    p: &'a mut [S],
    s: &'a mut [S],
    t: &'a mut [S],
    z_p: Option<&'a mut [S]>,
    z_s: Option<&'a mut [S]>,
    v_raw: Option<&'a mut [S]>,
    scratch: &'a mut BridgeScratch,
}

impl<'a> BiCgWorkspace<'a> {
    fn acquire(work: &'a mut Workspace, n: usize, need_z: bool, need_v_raw: bool) -> Self {
        if work.tmp1.len() != n {
            work.tmp1.resize(n, S::zero());
        }
        if work.tmp2.len() != n {
            work.tmp2.resize(n, S::zero());
        }
        let need_q = if need_v_raw { 5 } else { 4 };
        while work.q_s.len() < need_q {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..need_q] {
            if buf.len() != n {
                buf.resize(n, S::zero());
            }
        }
        let mut z_p = None;
        let mut z_s = None;
        if need_z {
            while work.z_s.len() < 2 {
                work.z_s.push(Vec::new());
            }
            for buf in &mut work.z_s[..2] {
                if buf.len() != n {
                    buf.resize(n, S::zero());
                }
            }
            let (z0, rest) = work.z_s.split_at_mut(1);
            let (z1, _) = rest.split_at_mut(1);
            z_p = Some(&mut z0[0][..n]);
            z_s = Some(&mut z1[0][..n]);
        }
        let (q0, rest) = work.q_s.split_at_mut(1);
        let (q1, rest) = rest.split_at_mut(1);
        let (q2, rest) = rest.split_at_mut(1);
        let (q3, rest) = rest.split_at_mut(1);
        let v_raw = if need_v_raw {
            Some(&mut rest[0][..n])
        } else {
            None
        };
        Self {
            r: &mut work.tmp1[..n],
            r_hat: &mut work.tmp2[..n],
            v: &mut q0[0][..n],
            p: &mut q1[0][..n],
            s: &mut q2[0][..n],
            t: &mut q3[0][..n],
            z_p,
            z_s,
            v_raw,
            scratch: &mut work.bridge,
        }
    }
}

#[inline]
fn bicgstab_update_p_from_r_or_s(p: &mut [S], r_or_s: &[S], v: &[S], beta: S, omega: S) {
    debug_assert_eq!(p.len(), r_or_s.len());
    debug_assert_eq!(p.len(), v.len());
    #[cfg(feature = "rayon")]
    {
        let n = p.len();
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
            p.par_iter_mut()
                .zip(r_or_s.par_iter().copied())
                .zip(v.par_iter().copied())
                .for_each(|((pi, ri), vi)| {
                    *pi = ri + beta * (*pi - omega * vi);
                });
            return;
        }
    }
    for i in 0..p.len() {
        p[i] = r_or_s[i] + beta * (p[i] - omega * v[i]);
    }
}

#[inline]
fn bicgstab_update_s_from_r_alpha_v(s: &mut [S], r: &[S], v: &[S], alpha: S) {
    debug_assert_eq!(s.len(), r.len());
    debug_assert_eq!(s.len(), v.len());
    #[cfg(feature = "rayon")]
    {
        let n = s.len();
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
            s.par_iter_mut()
                .zip(r.par_iter().copied())
                .zip(v.par_iter().copied())
                .for_each(|((si, ri), vi)| {
                    *si = ri - alpha * vi;
                });
            return;
        }
    }
    for i in 0..s.len() {
        s[i] = r[i] - alpha * v[i];
    }
}

#[inline]
fn bicgstab_update_x_alpha_y_omega_z(x: &mut [S], y: &[S], z: &[S], alpha: S, omega: S) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
            x.par_iter_mut()
                .zip(y.par_iter().copied())
                .zip(z.par_iter().copied())
                .for_each(|((xi, yi), zi)| {
                    *xi = *xi + alpha * yi + omega * zi;
                });
            return;
        }
    }
    for i in 0..x.len() {
        x[i] += alpha * y[i] + omega * z[i];
    }
}

#[inline]
fn bicgstab_update_r_from_s_omega_t(r: &mut [S], s: &[S], t: &[S], omega: S) {
    debug_assert_eq!(r.len(), s.len());
    debug_assert_eq!(r.len(), t.len());
    #[cfg(feature = "rayon")]
    {
        let n = r.len();
        let min_len = crate::algebra::parallel_cfg::parallel_tune().min_len_vec;
        if n >= min_len && !crate::algebra::parallel_cfg::force_serial() {
            r.par_iter_mut()
                .zip(s.par_iter().copied())
                .zip(t.par_iter().copied())
                .for_each(|((ri, si), ti)| {
                    *ri = si - omega * ti;
                });
            return;
        }
    }
    for i in 0..r.len() {
        r[i] = s[i] - omega * t[i];
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_true_residual_exit<A: KLinOp<Scalar = S> + ?Sized>(
    a: &A,
    b: &[S],
    x: &[S],
    red: &ReductCtx,
    tmp: &mut [S],
    scratch: &mut BridgeScratch,
    bnorm: R,
    atol: R,
    rtol: R,
    iterations: usize,
    nominal_reason: ConvergedReason,
) -> SolveStats<R> {
    let true_residual = true_residual_norm(a, b, x, red, tmp, scratch);
    let reason =
        reconcile_reason_with_true_residual(nominal_reason, true_residual, bnorm, atol, rtol);
    SolveStats::new(iterations, true_residual, reason)
}

#[allow(clippy::too_many_arguments)]
fn finalize_breakdown_salvage_exit<A: KLinOp<Scalar = S> + ?Sized>(
    a: &A,
    b: &[S],
    x: &[S],
    red: &ReductCtx,
    tmp: &mut [S],
    scratch: &mut BridgeScratch,
    bnorm: R,
    atol: R,
    rtol: R,
    iterations: usize,
    breakdown_reason: ConvergedReason,
    breakdown_detail: &str,
) -> SolveStats<R> {
    let true_residual = true_residual_norm(a, b, x, red, tmp, scratch);
    let contract_tol = atol.max(rtol * bnorm);
    let rel_residual = true_residual / bnorm;

    if true_residual.is_finite() && true_residual <= contract_tol {
        let mut stats = SolveStats::new(
            iterations,
            true_residual,
            ConvergedReason::ConvergedHappyBreakdown,
        );
        stats.acceptance_status = AcceptanceStatus::OkWithWarning;
        stats.breakdown_reason = Some(breakdown_reason);
        stats.residual_override_note = Some(format!(
            "BiCGStab breakdown ({breakdown_detail}) salvaged at iter {iterations}: ||r||={true_residual:.6e}, ||r||/||b||={rel_residual:.6e}, tol=max(atol, rtol*||b||)={contract_tol:.6e}"
        ));
        return stats;
    }

    SolveStats::new(iterations, true_residual, breakdown_reason)
}

/// BiCGStab solver (scalar-generic internal variant)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiCgStabVariant {
    Classic,
    FewerChecks,
    Reliable { residual_replace_every: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiCgStabBreakdownPolicy {
    Strict,
    RefreshShadow { max_refreshes: usize },
}

pub struct BiCgStabSolver {
    pub rtol: R,
    pub atol: R,
    pub dtol: R,
    pub maxits: usize,
    pub variant: BiCgStabVariant,
    pub breakdown_policy: BiCgStabBreakdownPolicy,
}

impl BiCgStabSolver {
    #[cfg(not(feature = "complex"))]
    fn select_real_dispatch<A>(a: &A, pc: Option<&dyn PreconditionerF64>) -> RealDispatchRoute
    where
        A: LinOpF64 + LinOp<S = f64> + ?Sized,
    {
        let any_op = a.as_any();
        if any_op.downcast_ref::<crate::matrix::DistCsrOp>().is_some() {
            return RealDispatchRoute::DistCsrBorrowed;
        }
        if any_op
            .downcast_ref::<crate::matrix::sparse::CsrMatrix<f64>>()
            .is_some()
        {
            return RealDispatchRoute::CsrBorrowed;
        }
        let pc_wants_csr = pc
            .map(|p| p.required_format() == OpFormat::Csr)
            .unwrap_or(false);
        #[cfg(feature = "backend-faer")]
        if a.format() == OpFormat::Csr
            && pc_wants_csr
            && any_op
                .downcast_ref::<crate::matrix::op::GenericCsrOp<f64>>()
                .is_some()
        {
            return RealDispatchRoute::CsrMaterialized;
        }
        RealDispatchRoute::Generic
    }

    pub fn new(rtol: R, maxits: usize) -> Self {
        Self {
            rtol,
            atol: 1e-12,
            dtol: 1e3,
            maxits,
            variant: BiCgStabVariant::Classic,
            breakdown_policy: BiCgStabBreakdownPolicy::Strict,
        }
    }

    pub fn set_variant(&mut self, variant: BiCgStabVariant) {
        self.variant = variant;
    }

    pub fn set_breakdown_policy(&mut self, policy: BiCgStabBreakdownPolicy) {
        self.breakdown_policy = policy;
    }

    pub fn reduction_model(&self) -> ReductionModel {
        match self.variant {
            BiCgStabVariant::Classic => ReductionModel {
                variant: "bicgstab-classic",
                startup: 2,
                per_iteration: 5.0,
                tail: 1,
            },
            BiCgStabVariant::FewerChecks => ReductionModel {
                variant: "bicgstab-fewer-checks",
                startup: 2,
                per_iteration: 4.0,
                tail: 1,
            },
            BiCgStabVariant::Reliable { .. } => ReductionModel {
                variant: "bicgstab-reliable",
                startup: 2,
                per_iteration: 5.0,
                tail: 1,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_k<A>(
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
        let _guard = StageGuard::new("BiCGStab");

        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "BiCGStab: square operator and matching b,x required".into(),
            ));
        }
        let mons = monitors.unwrap_or(&[]);

        if matches!(pc_side, PcSide::Symmetric) {
            return Err(KError::InvalidInput(
                "BiCGStab: PcSide::Symmetric is unsupported; use PcSide::Left or PcSide::Right"
                    .into(),
            ));
        }
        let pc_apply_side = pc_side;
        let side = pc_side;
        let work =
            work.ok_or_else(|| KError::InvalidInput("BiCGStab requires a Workspace".into()))?;
        let red = ReductCtx::new(comm, Some(&*work));
        let need_right = matches!(side, PcSide::Right) && pc.is_some();
        let need_left = matches!(side, PcSide::Left) && pc.is_some();
        let need_z = need_right || need_left;
        let need_v_raw = need_left;
        let BiCgWorkspace {
            r,
            r_hat,
            v,
            p,
            s,
            t,
            z_p,
            z_s,
            v_raw,
            scratch,
        } = BiCgWorkspace::acquire(work, n, need_z, need_v_raw);
        let mut z_p = z_p;
        let mut z_s = z_s;
        let mut v_raw = v_raw;
        let scratch = scratch;

        if x.iter().any(|&xi| xi.abs() > R::default()) {
            a.matvec_s(x, &mut v[..], &mut *scratch);
            for i in 0..n {
                r[i] = b[i] - v[i];
            }
        } else {
            r.copy_from_slice(b);
        }

        let res0 = if need_left {
            if let Some(zs) = z_s.as_deref_mut() {
                if let Some(pc) = pc {
                    pc.apply_s(pc_apply_side, r, zs, &mut *scratch)?;
                } else {
                    zs.copy_from_slice(r);
                }
                r_hat.copy_from_slice(zs);
                s.copy_from_slice(zs);
                p.copy_from_slice(zs);
                red.norm2(s)
            } else {
                r_hat.copy_from_slice(r);
                p.copy_from_slice(r);
                red.norm2(r)
            }
        } else {
            r_hat.copy_from_slice(r);
            p.copy_from_slice(r);
            red.norm2(r)
        };

        let bnorm = red.norm2(b).max(1e-32);
        let thr = self.atol.max(self.rtol * bnorm);

        if let Some(reason) = ReasonEmitter::non_finite(res0) {
            return Ok(
                SolveStats::new(0, res0, reason).with_counters(SolverCounters {
                    num_global_reductions: 2,
                    overlap_global_reductions: 0,
                    residual_replacements: 0,
                }),
            );
        }
        if call_monitors(mons, 0, res0, 0) {
            return Ok(
                SolveStats::new(0, res0, ConvergedReason::StoppedByMonitor).with_counters(
                    SolverCounters {
                        num_global_reductions: 2,
                        overlap_global_reductions: 0,
                        residual_replacements: 0,
                    },
                ),
            );
        }
        if res0 <= thr {
            let reason = if res0 <= self.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            return Ok(
                SolveStats::new(0, res0, reason).with_counters(SolverCounters {
                    num_global_reductions: 2,
                    overlap_global_reductions: 0,
                    residual_replacements: 0,
                }),
            );
        }

        let mut rho_prev = S::one();
        let mut alpha = S::one();
        let mut omega_prev = S::one();

        let eps_rho = 1e-30;
        let eps_alpha = 1e-30;
        let eps_omega = 1e-30;
        let eps_t_norm = eps_omega;
        let eps_s_norm = eps_omega;

        let mut sync_reductions = 2usize;
        #[cfg(not(feature = "complex"))]
        let mut async_reduction_waits = 0usize;
        #[cfg(feature = "complex")]
        let async_reduction_waits = 0usize;
        let mut residual_replacements = 0usize;
        let mut rho_refreshes_remaining = match self.breakdown_policy {
            BiCgStabBreakdownPolicy::Strict => 0,
            BiCgStabBreakdownPolicy::RefreshShadow { max_refreshes } => max_refreshes,
        };
        let mut refreshed_shadow = false;

        let (check_s_norm, replace_every) = match self.variant {
            BiCgStabVariant::Classic => (true, None),
            BiCgStabVariant::FewerChecks => (false, None),
            BiCgStabVariant::Reliable {
                residual_replace_every,
            } => (true, Some(residual_replace_every.max(1))),
        };

        for k in 1..=self.maxits {
            let rho = if need_left {
                red.dot(r_hat, s)
            } else {
                red.dot(r_hat, r)
            };
            sync_reductions += 1;
            if rho.abs() <= eps_rho || !rho.is_finite() {
                if rho_refreshes_remaining > 0 {
                    rho_refreshes_remaining -= 1;
                    residual_replacements += 1;

                    a.matvec_s(x, &mut v[..], &mut *scratch);
                    for i in 0..n {
                        r[i] = b[i] - v[i];
                    }
                    if need_left {
                        if let Some(zs) = z_s.as_deref_mut() {
                            if let Some(pc) = pc {
                                pc.apply_s(pc_apply_side, r, zs, &mut *scratch)?;
                            } else {
                                zs.copy_from_slice(r);
                            }
                            s.copy_from_slice(zs);
                        } else {
                            s.copy_from_slice(r);
                        }
                        r_hat.copy_from_slice(s);
                    } else {
                        r_hat.copy_from_slice(r);
                    }

                    rho_prev = S::one();
                    alpha = S::one();
                    omega_prev = S::one();
                    refreshed_shadow = true;
                    continue;
                }
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: rho ~ 0 at iter {k}");
                let stats = finalize_breakdown_salvage_exit(
                    a,
                    b,
                    x,
                    &red,
                    r,
                    &mut *scratch,
                    bnorm,
                    self.atol,
                    self.rtol,
                    k - 1,
                    ReasonEmitter::breakdown_bicg(),
                    "rho near-zero/non-finite",
                );
                return Ok(stats.with_counters(SolverCounters {
                    num_global_reductions: sync_reductions + async_reduction_waits,
                    overlap_global_reductions: async_reduction_waits,
                    residual_replacements,
                }));
            }

            let beta = if k == 1 || refreshed_shadow {
                refreshed_shadow = false;
                S::zero()
            } else {
                (rho / rho_prev) * (alpha / omega_prev)
            };
            if need_left {
                bicgstab_update_p_from_r_or_s(p, s, v, beta, omega_prev);
            } else {
                bicgstab_update_p_from_r_or_s(p, r, v, beta, omega_prev);
            }

            if need_left {
                let vr = v_raw
                    .as_deref_mut()
                    .expect("workspace: missing v_raw buffer");
                a.matvec_s(p, vr, &mut *scratch);
                if let Some(pc) = pc {
                    pc.apply_s(pc_apply_side, vr, v, &mut *scratch)?;
                } else {
                    v.copy_from_slice(vr);
                }
            } else {
                match (side, pc, z_p.as_deref_mut()) {
                    (PcSide::Right, Some(pc), Some(zp)) => {
                        pc.apply_s(pc_apply_side, p, zp, &mut *scratch)?;
                        a.matvec_s(zp, &mut v[..], &mut *scratch);
                    }
                    _ => {
                        a.matvec_s(p, &mut v[..], &mut *scratch);
                    }
                }
            }

            let alpha_den = red.dot(r_hat, v);
            sync_reductions += 1;
            if alpha_den.abs() <= eps_alpha || !alpha_den.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: alpha_den ~ 0 at iter {k}");
                let stats = finalize_breakdown_salvage_exit(
                    a,
                    b,
                    x,
                    &red,
                    r,
                    &mut *scratch,
                    bnorm,
                    self.atol,
                    self.rtol,
                    k - 1,
                    ReasonEmitter::breakdown_bicg(),
                    "alpha denominator near-zero/non-finite",
                );
                return Ok(stats.with_counters(SolverCounters {
                    num_global_reductions: sync_reductions + async_reduction_waits,
                    overlap_global_reductions: async_reduction_waits,
                    residual_replacements,
                }));
            }
            alpha = rho / alpha_den;

            if need_left {
                for i in 0..n {
                    s[i] -= alpha * v[i];
                }
            } else {
                bicgstab_update_s_from_r_alpha_v(s, r, v, alpha);
            }

            if check_s_norm {
                let s_norm = red.norm2(s);
                sync_reductions += 1;
                if let Some(reason) = ReasonEmitter::non_finite(s_norm) {
                    let stats = finalize_true_residual_exit(
                        a,
                        b,
                        x,
                        &red,
                        r,
                        &mut *scratch,
                        bnorm,
                        self.atol,
                        self.rtol,
                        k,
                        reason,
                    );
                    return Ok(stats.with_counters(SolverCounters {
                        num_global_reductions: sync_reductions + async_reduction_waits,
                        overlap_global_reductions: async_reduction_waits,
                        residual_replacements,
                    }));
                }
                if call_monitors(mons, k, s_norm, 0) {
                    return Ok(
                        SolveStats::new(k, s_norm, ConvergedReason::StoppedByMonitor)
                            .with_counters(SolverCounters {
                                num_global_reductions: sync_reductions + async_reduction_waits,
                                overlap_global_reductions: async_reduction_waits,
                                residual_replacements,
                            }),
                    );
                }
                if need_left && s_norm <= thr {
                    let x_trial = z_p.as_deref_mut().expect("workspace: missing z_p buffer");
                    for i in 0..n {
                        x_trial[i] = x[i] + alpha * p[i];
                    }

                    let true_residual = true_residual_norm(a, b, x_trial, &red, r, &mut *scratch);
                    if let Some(reason) = ReasonEmitter::non_finite(true_residual) {
                        return Ok(SolveStats::new(k, true_residual, reason).with_counters(
                            SolverCounters {
                                num_global_reductions: sync_reductions + async_reduction_waits,
                                overlap_global_reductions: async_reduction_waits,
                                residual_replacements,
                            },
                        ));
                    }

                    x.copy_from_slice(x_trial);
                    if let Some(reason) =
                        true_residual_converged_reason(true_residual, bnorm, self.atol, self.rtol)
                    {
                        return Ok(SolveStats::new(k, true_residual, reason).with_counters(
                            SolverCounters {
                                num_global_reductions: sync_reductions + async_reduction_waits,
                                overlap_global_reductions: async_reduction_waits,
                                residual_replacements,
                            },
                        ));
                    }

                    if let Some(zs) = z_s.as_deref_mut() {
                        if let Some(pc) = pc {
                            pc.apply_s(pc_apply_side, r, zs, &mut *scratch)?;
                        } else {
                            zs.copy_from_slice(r);
                        }
                        s.copy_from_slice(zs);
                    } else {
                        s.copy_from_slice(r);
                    }
                    r_hat.copy_from_slice(s);
                    rho_prev = S::one();
                    alpha = S::one();
                    omega_prev = S::one();
                    refreshed_shadow = true;
                    continue;
                }

                if !need_left && s_norm <= thr {
                    match (side, pc, z_p.as_deref()) {
                        (PcSide::Right, Some(_), Some(zp)) => {
                            for i in 0..n {
                                x[i] += alpha * zp[i];
                            }
                        }
                        _ => {
                            for i in 0..n {
                                x[i] += alpha * p[i];
                            }
                        }
                    }
                    let nominal = if s_norm <= self.atol {
                        ConvergedReason::ConvergedAtol
                    } else {
                        ConvergedReason::ConvergedRtol
                    };
                    let stats = finalize_true_residual_exit(
                        a,
                        b,
                        x,
                        &red,
                        r,
                        &mut *scratch,
                        bnorm,
                        self.atol,
                        self.rtol,
                        k,
                        nominal,
                    );
                    return Ok(stats.with_counters(SolverCounters {
                        num_global_reductions: sync_reductions + async_reduction_waits,
                        overlap_global_reductions: async_reduction_waits,
                        residual_replacements,
                    }));
                }
            }

            if need_left {
                let zs = z_s.as_deref_mut().expect("workspace: missing z_s buffer");
                a.matvec_s(s, zs, &mut *scratch);
                if let Some(pc) = pc {
                    pc.apply_s(pc_apply_side, zs, t, &mut *scratch)?;
                } else {
                    t.copy_from_slice(zs);
                }
            } else {
                match (side, pc, z_s.as_deref_mut()) {
                    (PcSide::Right, Some(pc), Some(zs)) => {
                        pc.apply_s(pc_apply_side, s, zs, &mut *scratch)?;
                        a.matvec_s(zs, &mut t[..], &mut *scratch);
                    }
                    _ => {
                        a.matvec_s(s, &mut t[..], &mut *scratch);
                    }
                }
            }

            #[cfg(not(feature = "complex"))]
            let (omega_den, omega_num, t_norm) = {
                let async_opt = ReductOptions {
                    mode: red.mode(),
                    ..ReductOptions::default()
                };
                let (omega_req, _omega_local) =
                    dot2_async_s(comm, &t[..], &t[..], &t[..], &s[..], &async_opt)?;
                let overlap_reduction = !matches!(omega_req, AllreduceHandle::Ready(_));
                let omega_reds = <UniverseComm as AllreduceOps>::wait_pair(omega_req);
                if overlap_reduction {
                    async_reduction_waits += 1;
                }
                (
                    S::from_real(omega_reds.0),
                    S::from_real(omega_reds.1),
                    omega_reds.0.abs().sqrt(),
                )
            };

            #[cfg(feature = "complex")]
            let (omega_den, omega_num, t_norm) = {
                let mut omega_dots = [S::zero(); 2];
                red.dot_many_into(&[(&t[..], &t[..]), (&t[..], &s[..])], &mut omega_dots);
                sync_reductions += 1;
                let den = omega_dots[0].real();
                (S::from_real(den), omega_dots[1], den.abs().sqrt())
            };

            if t_norm <= eps_t_norm {
                let s_norm = red.norm2(s);
                sync_reductions += 1;
                if let Some(reason) = ReasonEmitter::non_finite(s_norm) {
                    let stats = finalize_true_residual_exit(
                        a,
                        b,
                        x,
                        &red,
                        r,
                        &mut *scratch,
                        bnorm,
                        self.atol,
                        self.rtol,
                        k,
                        reason,
                    );
                    return Ok(stats.with_counters(SolverCounters {
                        num_global_reductions: sync_reductions + async_reduction_waits,
                        overlap_global_reductions: async_reduction_waits,
                        residual_replacements,
                    }));
                }
                if s_norm <= eps_s_norm {
                    if need_left {
                        for i in 0..n {
                            x[i] += alpha * p[i];
                        }
                    } else {
                        match (side, pc, z_p.as_deref()) {
                            (PcSide::Right, Some(_), Some(zp)) => {
                                for i in 0..n {
                                    x[i] += alpha * zp[i];
                                }
                            }
                            _ => {
                                for i in 0..n {
                                    x[i] += alpha * p[i];
                                }
                            }
                        }
                    }
                    let nominal = if s_norm <= self.atol {
                        ConvergedReason::ConvergedAtol
                    } else {
                        ConvergedReason::ConvergedRtol
                    };
                    let stats = finalize_true_residual_exit(
                        a,
                        b,
                        x,
                        &red,
                        r,
                        &mut *scratch,
                        bnorm,
                        self.atol,
                        self.rtol,
                        k,
                        nominal,
                    );
                    return Ok(stats.with_counters(SolverCounters {
                        num_global_reductions: sync_reductions + async_reduction_waits,
                        overlap_global_reductions: async_reduction_waits,
                        residual_replacements,
                    }));
                }
                let stats = finalize_breakdown_salvage_exit(
                    a,
                    b,
                    x,
                    &red,
                    r,
                    &mut *scratch,
                    bnorm,
                    self.atol,
                    self.rtol,
                    k,
                    ReasonEmitter::breakdown_bicg(),
                    "singular preconditioned operator: ||t|| near-zero while ||s|| is not",
                );
                return Ok(stats.with_counters(SolverCounters {
                    num_global_reductions: sync_reductions + async_reduction_waits,
                    overlap_global_reductions: async_reduction_waits,
                    residual_replacements,
                }));
            }
            if omega_den.abs() <= eps_omega || !omega_den.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: omega_den ~ 0 at iter {k}");
                let stats = finalize_breakdown_salvage_exit(
                    a,
                    b,
                    x,
                    &red,
                    r,
                    &mut *scratch,
                    bnorm,
                    self.atol,
                    self.rtol,
                    k,
                    ReasonEmitter::breakdown_bicg(),
                    "omega denominator near-zero/non-finite",
                );
                return Ok(stats.with_counters(SolverCounters {
                    num_global_reductions: sync_reductions + async_reduction_waits,
                    overlap_global_reductions: async_reduction_waits,
                    residual_replacements,
                }));
            }
            let omega = omega_num / omega_den;
            if omega.abs() <= eps_omega || !omega.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: omega ~ 0 at iter {k}");
                let stats = finalize_breakdown_salvage_exit(
                    a,
                    b,
                    x,
                    &red,
                    r,
                    &mut *scratch,
                    bnorm,
                    self.atol,
                    self.rtol,
                    k,
                    ReasonEmitter::breakdown_bicg(),
                    "omega near-zero/non-finite",
                );
                return Ok(stats.with_counters(SolverCounters {
                    num_global_reductions: sync_reductions + async_reduction_waits,
                    overlap_global_reductions: async_reduction_waits,
                    residual_replacements,
                }));
            }

            if need_left {
                bicgstab_update_x_alpha_y_omega_z(x, p, s, alpha, omega);
            } else {
                match (side, pc, z_p.as_deref(), z_s.as_deref()) {
                    (PcSide::Right, Some(_), Some(zp), Some(zs)) => {
                        bicgstab_update_x_alpha_y_omega_z(x, zp, zs, alpha, omega);
                    }
                    _ => {
                        bicgstab_update_x_alpha_y_omega_z(x, p, s, alpha, omega);
                    }
                }
            }

            if need_left {
                let vr = v_raw.as_deref().expect("workspace: missing v_raw buffer");
                let as_raw = z_s.as_deref().expect("workspace: missing z_s buffer");
                for i in 0..n {
                    r[i] -= alpha * vr[i] + omega * as_raw[i];
                }
                if let Some(zs) = z_s.as_deref_mut() {
                    if let Some(pc) = pc {
                        pc.apply_s(pc_apply_side, r, zs, &mut *scratch)?;
                    } else {
                        zs.copy_from_slice(r);
                    }
                    s.copy_from_slice(zs);
                } else {
                    s.copy_from_slice(r);
                }
            } else {
                bicgstab_update_r_from_s_omega_t(r, s, t, omega);
            }

            if let Some(every) = replace_every {
                if k % every == 0 {
                    a.matvec_s(x, &mut v[..], &mut *scratch);
                    for i in 0..n {
                        r[i] = b[i] - v[i];
                    }
                    residual_replacements += 1;
                    if need_left {
                        if let Some(zs) = z_s.as_deref_mut() {
                            if let Some(pc) = pc {
                                pc.apply_s(pc_apply_side, r, zs, &mut *scratch)?;
                            } else {
                                zs.copy_from_slice(r);
                            }
                            s.copy_from_slice(zs);
                        } else {
                            s.copy_from_slice(r);
                        }
                    }
                }
            }

            let r_norm = if need_left {
                red.norm2(s)
            } else {
                red.norm2(r)
            };
            sync_reductions += 1;
            if let Some(reason) = ReasonEmitter::non_finite(r_norm) {
                let stats = finalize_true_residual_exit(
                    a,
                    b,
                    x,
                    &red,
                    r,
                    &mut *scratch,
                    bnorm,
                    self.atol,
                    self.rtol,
                    k,
                    reason,
                );
                return Ok(stats.with_counters(SolverCounters {
                    num_global_reductions: sync_reductions + async_reduction_waits,
                    overlap_global_reductions: async_reduction_waits,
                    residual_replacements,
                }));
            }
            if call_monitors(mons, k, r_norm, 0) {
                return Ok(
                    SolveStats::new(k, r_norm, ConvergedReason::StoppedByMonitor).with_counters(
                        SolverCounters {
                            num_global_reductions: sync_reductions + async_reduction_waits,
                            overlap_global_reductions: async_reduction_waits,
                            residual_replacements,
                        },
                    ),
                );
            }

            if r_norm <= thr {
                let nominal = if r_norm <= self.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                };
                if need_left {
                    let true_residual = true_residual_norm(a, b, x, &red, r, &mut *scratch);
                    if let Some(reason) = ReasonEmitter::non_finite(true_residual) {
                        return Ok(SolveStats::new(k, true_residual, reason).with_counters(
                            SolverCounters {
                                num_global_reductions: sync_reductions + async_reduction_waits,
                                overlap_global_reductions: async_reduction_waits,
                                residual_replacements,
                            },
                        ));
                    }
                    if let Some(reason) =
                        true_residual_converged_reason(true_residual, bnorm, self.atol, self.rtol)
                    {
                        return Ok(SolveStats::new(k, true_residual, reason).with_counters(
                            SolverCounters {
                                num_global_reductions: sync_reductions + async_reduction_waits,
                                overlap_global_reductions: async_reduction_waits,
                                residual_replacements,
                            },
                        ));
                    }

                    if let Some(zs) = z_s.as_deref_mut() {
                        if let Some(pc) = pc {
                            pc.apply_s(pc_apply_side, r, zs, &mut *scratch)?;
                        } else {
                            zs.copy_from_slice(r);
                        }
                        s.copy_from_slice(zs);
                    } else {
                        s.copy_from_slice(r);
                    }
                    r_hat.copy_from_slice(s);
                    rho_prev = S::one();
                    alpha = S::one();
                    omega_prev = S::one();
                    refreshed_shadow = true;
                    residual_replacements += 1;
                    continue;
                }
                let stats = finalize_true_residual_exit(
                    a,
                    b,
                    x,
                    &red,
                    r,
                    &mut *scratch,
                    bnorm,
                    self.atol,
                    self.rtol,
                    k,
                    nominal,
                );
                return Ok(stats.with_counters(SolverCounters {
                    num_global_reductions: sync_reductions + async_reduction_waits,
                    overlap_global_reductions: async_reduction_waits,
                    residual_replacements,
                }));
            }
            if r_norm >= self.dtol * bnorm {
                let stats = finalize_true_residual_exit(
                    a,
                    b,
                    x,
                    &red,
                    r,
                    &mut *scratch,
                    bnorm,
                    self.atol,
                    self.rtol,
                    k,
                    ConvergedReason::DivergedDtol,
                );
                return Ok(stats.with_counters(SolverCounters {
                    num_global_reductions: sync_reductions + async_reduction_waits,
                    overlap_global_reductions: async_reduction_waits,
                    residual_replacements,
                }));
            }

            rho_prev = rho;
            omega_prev = omega;
        }

        let r_norm = red.norm2(r);
        sync_reductions += 1;
        let _ = r_norm;
        let stats = finalize_true_residual_exit(
            a,
            b,
            x,
            &red,
            r,
            &mut *scratch,
            bnorm,
            self.atol,
            self.rtol,
            self.maxits,
            ConvergedReason::DivergedMaxIts,
        );
        Ok(stats.with_counters(SolverCounters {
            num_global_reductions: sync_reductions + async_reduction_waits,
            overlap_global_reductions: async_reduction_waits,
            residual_replacements,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(not(feature = "complex"))]
    fn solve_csr(
        &mut self,
        a: &crate::matrix::sparse::CsrMatrix<f64>,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError> {
        self.solve_k(a, pc, b, x, pc_side, comm, monitors, work)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(not(feature = "complex"))]
    fn solve_dist_csr(
        &mut self,
        a: &crate::matrix::DistCsrOp,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError> {
        self.solve_k(a, pc, b, x, pc_side, comm, monitors, work)
    }

    #[allow(clippy::too_many_arguments)]
    /// Solve using real (`f64`) operator/preconditioner interfaces.
    ///
    /// In `feature = "complex"` builds this bridge maps `f64 -> S` and projects
    /// the final iterate back to `f64` (real part). For genuine complex solves,
    /// prefer `Self::solve_k` or `Self::solve_c64`.
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
            match Self::select_real_dispatch(a, pc) {
                RealDispatchRoute::DistCsrBorrowed => {
                    let dist = a
                        .as_any()
                        .downcast_ref::<crate::matrix::DistCsrOp>()
                        .expect("dist dispatch requires DistCsrOp");
                    return self
                        .solve_dist_csr(dist, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
                        .map(|stats| stats.with_reduction_model(self.reduction_model()));
                }
                RealDispatchRoute::CsrBorrowed => {
                    let csr = a
                        .as_any()
                        .downcast_ref::<crate::matrix::sparse::CsrMatrix<f64>>()
                        .expect("csr dispatch requires CsrMatrix");
                    return self
                        .solve_csr(csr, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
                        .map(|stats| stats.with_reduction_model(self.reduction_model()));
                }
                #[cfg(feature = "backend-faer")]
                RealDispatchRoute::CsrMaterialized => {
                    let generic = a
                        .as_any()
                        .downcast_ref::<crate::matrix::op::GenericCsrOp<f64>>()
                        .expect("materialized CSR dispatch requires GenericCsrOp<f64>");
                    let matrix = generic.matrix();
                    let csr = crate::matrix::sparse::CsrMatrix::from_csr(
                        matrix.nrows(),
                        matrix.ncols(),
                        matrix.row_ptr().to_vec(),
                        matrix.col_idx().to_vec(),
                        matrix.values().to_vec(),
                    );
                    return self
                        .solve_csr(&csr, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
                        .map(|stats| stats.with_reduction_model(self.reduction_model()));
                }
                RealDispatchRoute::Generic => {}
            }
            self.solve_k(&op, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
                .map(|stats| stats.with_reduction_model(self.reduction_model()))
        }
        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result = self
                .solve_k(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work)
                .map(|stats| stats.with_reduction_model(self.reduction_model()));
            if result.is_ok() {
                for (dst, src) in x.iter_mut().zip(x_s.iter()) {
                    *dst = src.real();
                }
            }
            result
        }
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
        self.solve_k(a, pc, b, x, pc_side, comm, monitors, work)
    }

    #[cfg(feature = "complex")]
    #[allow(clippy::too_many_arguments)]
    pub fn solve_c64<A>(
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
        self.solve_k(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

impl LinearSolver for BiCgStabSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        if w.q_s.len() < 4 {
            w.q_s.resize(4, Vec::new());
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
        self.solve_f64(a, pc.as_deref(), b, x, pc_side, comm, monitors, work)
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    fn approx_vec_eq(a: &[S], b: &[S], tol: R) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() <= tol,
                "mismatch at {i}: lhs={:?}, rhs={:?}",
                a[i],
                b[i]
            );
        }
    }

    fn sample_vec(n: usize, offset: R) -> Vec<S> {
        (0..n)
            .map(|i| S::from_real(((i as R) * 0.37 + offset).sin()))
            .collect()
    }

    #[test]
    fn update_p_matches_inlined_formula() {
        let n = 127;
        let r_or_s = sample_vec(n, 0.1);
        let v = sample_vec(n, -0.4);
        let beta = S::from_real(0.33);
        let omega = S::from_real(-0.21);
        let mut p_expected = sample_vec(n, 0.7);
        let mut p_actual = p_expected.clone();

        for i in 0..n {
            p_expected[i] = r_or_s[i] + beta * (p_expected[i] - omega * v[i]);
        }
        bicgstab_update_p_from_r_or_s(&mut p_actual, &r_or_s, &v, beta, omega);
        approx_vec_eq(&p_actual, &p_expected, 1e-12);
    }

    #[test]
    fn update_s_matches_inlined_formula() {
        let n = 193;
        let r = sample_vec(n, 0.25);
        let v = sample_vec(n, -0.75);
        let alpha = S::from_real(0.42);
        let mut s_expected = sample_vec(n, 0.9);
        let mut s_actual = s_expected.clone();

        for i in 0..n {
            s_expected[i] = r[i] - alpha * v[i];
        }
        bicgstab_update_s_from_r_alpha_v(&mut s_actual, &r, &v, alpha);
        approx_vec_eq(&s_actual, &s_expected, 1e-12);
    }

    #[test]
    fn update_x_matches_inlined_formula() {
        let n = 211;
        let y = sample_vec(n, 0.5);
        let z = sample_vec(n, -0.1);
        let alpha = S::from_real(0.19);
        let omega = S::from_real(-0.63);
        let mut x_expected = sample_vec(n, 0.33);
        let mut x_actual = x_expected.clone();

        for i in 0..n {
            x_expected[i] += alpha * y[i] + omega * z[i];
        }
        bicgstab_update_x_alpha_y_omega_z(&mut x_actual, &y, &z, alpha, omega);
        approx_vec_eq(&x_actual, &x_expected, 1e-12);
    }

    #[test]
    fn update_r_matches_inlined_formula() {
        let n = 257;
        let s = sample_vec(n, 1.3);
        let t = sample_vec(n, -0.3);
        let omega = S::from_real(0.71);
        let mut r_expected = sample_vec(n, -0.8);
        let mut r_actual = r_expected.clone();

        for i in 0..n {
            r_expected[i] = s[i] - omega * t[i];
        }
        bicgstab_update_r_from_s_omega_t(&mut r_actual, &s, &t, omega);
        approx_vec_eq(&r_actual, &r_expected, 1e-12);
    }
}

#[cfg(all(test, feature = "backend-faer"))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use faer::Mat;
    use std::sync::Arc;

    use crate::matrix::csr::CsrMatrix as ScalarCsrMatrix;
    use crate::matrix::op::GenericCsrOp;
    use crate::matrix::sparse::CsrMatrix;
    use crate::matrix::spmv::SpmvTuning;
    use crate::preconditioner::Preconditioner;

    fn nonsym_3x3() -> (Mat<f64>, Vec<f64>) {
        let a = Mat::from_fn(3, 3, |i, j| {
            if i == j {
                4.0
            } else {
                (i + 2 * j) as f64 + 1.0
            }
        });
        let x_true = vec![1.0, 2.0, 3.0];
        let mut b = vec![0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                b[i] += a[(i, j)] * x_true[j];
            }
        }
        (a, b)
    }

    #[test]
    fn bicgstab_solves_well_conditioned_nonsym() {
        let (a, b) = nonsym_3x3();
        let mut x = vec![0.0; 3];
        let mut solver = BiCgStabSolver::new(1e-10, 100);
        let comm = UniverseComm::NoComm(crate::parallel::NoComm);
        let mut ws = Workspace::new(3);
        solver.setup_workspace(&mut ws);
        let stats = solver
            .solve_f64(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .unwrap();
        let x_true = vec![1.0, 2.0, 3.0];
        for i in 0..3 {
            assert_abs_diff_eq!(x[i], x_true[i], epsilon = 1e-8);
        }
        assert!(matches!(
            stats.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ));
    }

    #[test]
    fn bicgstab_right_side_nonzero_initial_guess_sanity() {
        let (a, b) = nonsym_3x3();
        let mut x = vec![0.4, -0.2, 0.1];
        let x0 = x.clone();
        let mut solver = BiCgStabSolver::new(1e-10, 100);
        let comm = UniverseComm::NoComm(crate::parallel::NoComm);
        let mut ws = Workspace::new(3);
        solver.setup_workspace(&mut ws);

        let stats = solver
            .solve_f64(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Right,
                &comm,
                None,
                Some(&mut ws),
            )
            .unwrap();

        assert!(matches!(
            stats.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ));
        assert_ne!(x, x0);
    }

    #[test]
    fn bicgstab_reliable_variant_records_replacements_sanity() {
        let (a, b) = nonsym_3x3();
        let mut x = vec![0.0; 3];
        let mut solver = BiCgStabSolver::new(1e-10, 100);
        solver.set_variant(BiCgStabVariant::Reliable {
            residual_replace_every: 1,
        });
        let comm = UniverseComm::NoComm(crate::parallel::NoComm);
        let mut ws = Workspace::new(3);
        solver.setup_workspace(&mut ws);

        let stats = solver
            .solve_f64(
                &a,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .unwrap();

        assert_ne!(stats.reason, ConvergedReason::Continued);
        assert!(stats.counters.residual_replacements >= stats.iterations);
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn bicgstab_left_ilu_exact_factor_converges_like_right_ilu() {
        use crate::context::ksp_context::{KspContext, SolverType};
        use crate::context::pc_context::PcType;

        let a = Arc::new(CsrMatrix::from_csr(
            4,
            4,
            vec![0, 2, 4, 5, 6],
            vec![0, 1, 0, 1, 2, 3],
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0],
        ));
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let solve = |side| -> (SolveStats<f64>, Vec<f64>) {
            let mut ksp = KspContext::new();
            ksp.set_type(SolverType::BiCgStab).unwrap();
            ksp.set_tolerances(1e-12, 1e-12, 1e3, 20);
            ksp.set_pc_type(PcType::Ilu, None).unwrap();
            ksp.set_pc_side(side);
            ksp.set_operators(a.clone(), None);

            let mut x = vec![0.0; b.len()];
            let stats = ksp.solve(&b, &mut x).unwrap();
            (stats, x)
        };

        let (left_stats, left_x) = solve(PcSide::Left);
        let (right_stats, right_x) = solve(PcSide::Right);

        assert_eq!(
            left_stats.iterations, 1,
            "left ILU should solve this exactly factored system in one BiCGStab iteration"
        );
        assert_eq!(
            right_stats.iterations, 1,
            "right ILU regression sanity check"
        );
        assert!(
            left_stats.final_residual <= 1e-12,
            "left true residual too large: {:.3e}",
            left_stats.final_residual
        );
        assert!(
            right_stats.final_residual <= 1e-12,
            "right true residual too large: {:.3e}",
            right_stats.final_residual
        );
        for i in 0..b.len() {
            assert_abs_diff_eq!(left_x[i], right_x[i], epsilon = 1e-12);
        }
    }

    #[cfg(not(feature = "complex"))]
    struct IdentityHintPc {
        format: OpFormat,
    }

    #[cfg(not(feature = "complex"))]
    impl Preconditioner for IdentityHintPc {
        fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
            Ok(())
        }

        fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
            y.copy_from_slice(x);
            Ok(())
        }

        fn required_format(&self) -> OpFormat {
            self.format
        }
    }

    #[cfg(not(feature = "complex"))]
    fn csr_reference_system() -> (Arc<CsrMatrix<f64>>, Vec<f64>, Vec<f64>) {
        let row_ptr = vec![0, 2, 4];
        let col_idx = vec![0, 1, 0, 1];
        let vals = vec![4.0, 1.0, 2.0, 3.0];
        let csr = Arc::new(CsrMatrix::from_csr(2, 2, row_ptr, col_idx, vals));
        let x_true = vec![1.0, -2.0];
        let mut b = vec![0.0; 2];
        csr.spmv(&x_true, &mut b);
        (csr, b, x_true)
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn bicgstab_dispatch_prefers_csr_materialized_when_pc_requires_csr() {
        let (csr, _, _) = csr_reference_system();
        let scalar = ScalarCsrMatrix::from_real_csr(csr.as_ref());
        let op = GenericCsrOp::new(Arc::new(scalar), &SpmvTuning::default());
        let pc_csr = IdentityHintPc {
            format: OpFormat::Csr,
        };
        let pc_any = IdentityHintPc {
            format: OpFormat::Any,
        };

        assert_eq!(
            BiCgStabSolver::select_real_dispatch(&op, Some(&pc_csr)),
            RealDispatchRoute::CsrMaterialized
        );
        assert_eq!(
            BiCgStabSolver::select_real_dispatch(&op, Some(&pc_any)),
            RealDispatchRoute::Generic
        );
    }

    #[cfg(not(feature = "complex"))]
    #[test]
    fn bicgstab_csr_materialized_and_generic_routes_match_numerics() {
        let (csr, b, x_true) = csr_reference_system();
        let scalar = ScalarCsrMatrix::from_real_csr(csr.as_ref());
        let op = GenericCsrOp::new(Arc::new(scalar), &SpmvTuning::default());
        let comm = UniverseComm::NoComm(crate::parallel::NoComm);

        let mut solver_csr = BiCgStabSolver::new(1e-12, 64);
        let mut ws_csr = Workspace::new(2);
        solver_csr.setup_workspace(&mut ws_csr);
        let mut x_csr = vec![0.0; 2];
        let pc_csr = IdentityHintPc {
            format: OpFormat::Csr,
        };
        let stats_csr = solver_csr
            .solve_f64(
                &op,
                Some(&pc_csr),
                &b,
                &mut x_csr,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws_csr),
            )
            .expect("csr-materialized route should solve");
        assert!(matches!(
            stats_csr.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ));

        let mut solver_generic = BiCgStabSolver::new(1e-12, 64);
        let mut ws_generic = Workspace::new(2);
        solver_generic.setup_workspace(&mut ws_generic);
        let mut x_generic = vec![0.0; 2];
        let pc_any = IdentityHintPc {
            format: OpFormat::Any,
        };
        let stats_generic = solver_generic
            .solve_f64(
                &op,
                Some(&pc_any),
                &b,
                &mut x_generic,
                PcSide::Left,
                &comm,
                None,
                Some(&mut ws_generic),
            )
            .expect("generic route should solve");
        assert!(matches!(
            stats_generic.reason,
            ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
        ));

        for i in 0..2 {
            assert_abs_diff_eq!(x_csr[i], x_true[i], epsilon = 1e-10);
            assert_abs_diff_eq!(x_generic[i], x_true[i], epsilon = 1e-10);
            assert_abs_diff_eq!(x_csr[i], x_generic[i], epsilon = 1e-12);
        }
    }
}
