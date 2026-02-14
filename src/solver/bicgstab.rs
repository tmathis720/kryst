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
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::MonitorCallback;
use crate::solver::common::ReductCtx;
use crate::solver::common::call_monitors;
use crate::utils::convergence::{ConvergedReason, SolveStats};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::trace;

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

/// BiCGStab solver (scalar-generic internal variant)
pub struct BiCgStabSolver {
    pub rtol: R,
    pub atol: R,
    pub dtol: R,
    pub maxits: usize,
}

impl BiCgStabSolver {
    pub fn new(rtol: R, maxits: usize) -> Self {
        Self {
            rtol,
            atol: 1e-12,
            dtol: 1e3,
            maxits,
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
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("BiCGStab");

        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "BiCGStab: square operator and matching b,x required".into(),
            ));
        }
        let mons = monitors.unwrap_or(&[]);

        let pc_apply_side = pc_side;
        let side = match pc_side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        };
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

        if let Some(reason) = ConvergedReason::from_non_finite(res0) {
            return Ok(SolveStats::new(0, res0, reason));
        }
        if call_monitors(mons, 0, res0, 0) {
            return Ok(SolveStats::new(0, res0, ConvergedReason::StoppedByMonitor));
        }
        if res0 <= thr {
            let reason = if res0 <= self.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            return Ok(SolveStats::new(0, res0, reason));
        }

        let mut rho_prev = S::one();
        let mut alpha = S::one();
        let mut omega_prev = S::one();

        let eps_rho = 1e-30;
        let eps_alpha = 1e-30;
        let eps_omega = 1e-30;

        let mut stats = SolveStats::new(0, res0, ConvergedReason::Continued);

        for k in 1..=self.maxits {
            let rho = if need_left {
                red.dot(r_hat, s)
            } else {
                red.dot(r_hat, r)
            };
            if rho.abs() <= eps_rho || !rho.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: rho ~ 0 at iter {k}");
                stats.iterations = k - 1;
                stats.final_residual = if need_left {
                    red.norm2(s)
                } else {
                    red.norm2(r)
                };
                stats.reason = ConvergedReason::DivergedBreakdownBiCG;
                return Ok(stats);
            }

            let beta = if k == 1 {
                S::zero()
            } else {
                (rho / rho_prev) * (alpha / omega_prev)
            };
            if need_left {
                for i in 0..n {
                    p[i] = s[i] + beta * (p[i] - omega_prev * v[i]);
                }
            } else {
                for i in 0..n {
                    p[i] = r[i] + beta * (p[i] - omega_prev * v[i]);
                }
            }

            if need_left {
                let yp_ref: &mut [S] = if let (Some(pc), Some(zp)) = (pc, z_p.as_deref_mut()) {
                    pc.apply_s(pc_apply_side, p, zp, &mut *scratch)?;
                    zp
                } else {
                    p
                };
                let vr = v_raw
                    .as_deref_mut()
                    .expect("workspace: missing v_raw buffer");
                a.matvec_s(&*yp_ref, vr, &mut *scratch);
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
            if alpha_den.abs() <= eps_alpha || !alpha_den.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: alpha_den ~ 0 at iter {k}");
                stats.iterations = k - 1;
                stats.final_residual = if need_left {
                    red.norm2(s)
                } else {
                    red.norm2(r)
                };
                stats.reason = ConvergedReason::DivergedBreakdownBiCG;
                return Ok(stats);
            }
            alpha = rho / alpha_den;

            if need_left {
                for i in 0..n {
                    s[i] -= alpha * v[i];
                }
            } else {
                for i in 0..n {
                    s[i] = r[i] - alpha * v[i];
                }
            }

            let s_norm = red.norm2(s);
            if let Some(reason) = ConvergedReason::from_non_finite(s_norm) {
                stats.iterations = k;
                stats.final_residual = s_norm;
                stats.reason = reason;
                return Ok(stats);
            }
            if call_monitors(mons, k, s_norm, 0) {
                return Ok(SolveStats::new(
                    k,
                    s_norm,
                    ConvergedReason::StoppedByMonitor,
                ));
            }
            if s_norm <= thr {
                if need_left {
                    if let Some(yp) = z_p.as_deref() {
                        for i in 0..n {
                            x[i] += alpha * yp[i];
                        }
                    } else {
                        for i in 0..n {
                            x[i] += alpha * p[i];
                        }
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
                stats.iterations = k;
                stats.final_residual = s_norm;
                stats.reason = if s_norm <= self.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                };
                return Ok(stats);
            }

            if need_left {
                let zs = z_s.as_deref_mut().expect("workspace: missing z_s buffer");
                if let Some(pc) = pc {
                    pc.apply_s(pc_apply_side, s, zs, &mut *scratch)?;
                } else {
                    zs.copy_from_slice(s);
                }
                a.matvec_s(zs, &mut t[..], &mut *scratch);
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

            let mut omega_reds = [S::zero(), S::zero()];
            let dot_pairs = [(&t[..], &t[..]), (&t[..], &s[..])];
            red.dot_many_into(&dot_pairs, &mut omega_reds);

            let omega_den = omega_reds[0];
            if omega_den.abs() <= eps_omega || !omega_den.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: omega_den ~ 0 at iter {k}");
                stats.iterations = k;
                stats.final_residual = red.norm2(s);
                stats.reason = ConvergedReason::DivergedBreakdownBiCG;
                return Ok(stats);
            }
            let omega = omega_reds[1] / omega_den;
            if omega.abs() <= eps_omega || !omega.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: omega ~ 0 at iter {k}");
                stats.iterations = k;
                stats.final_residual = red.norm2(s);
                stats.reason = ConvergedReason::DivergedBreakdownBiCG;
                return Ok(stats);
            }

            if need_left {
                match (z_p.as_deref(), z_s.as_deref()) {
                    (Some(y), Some(tpre)) => {
                        for i in 0..n {
                            x[i] += alpha * y[i] + omega * tpre[i];
                        }
                    }
                    (Some(y), None) => {
                        for i in 0..n {
                            x[i] += alpha * y[i] + omega * s[i];
                        }
                    }
                    (None, Some(tpre)) => {
                        for i in 0..n {
                            x[i] += alpha * p[i] + omega * tpre[i];
                        }
                    }
                    (None, None) => {
                        for i in 0..n {
                            x[i] += alpha * p[i] + omega * s[i];
                        }
                    }
                }
            } else {
                match (side, pc, z_p.as_deref(), z_s.as_deref()) {
                    (PcSide::Right, Some(_), Some(zp), Some(zs)) => {
                        for i in 0..n {
                            x[i] += alpha * zp[i] + omega * zs[i];
                        }
                    }
                    _ => {
                        for i in 0..n {
                            x[i] += alpha * p[i] + omega * s[i];
                        }
                    }
                }
            }

            if need_left {
                let vr = v_raw.as_deref().expect("workspace: missing v_raw buffer");
                for i in 0..n {
                    r[i] -= alpha * vr[i] + omega * t[i];
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
                for i in 0..n {
                    r[i] = s[i] - omega * t[i];
                }
            }

            let r_norm = if need_left {
                red.norm2(s)
            } else {
                red.norm2(r)
            };
            if let Some(reason) = ConvergedReason::from_non_finite(r_norm) {
                stats.iterations = k;
                stats.final_residual = r_norm;
                stats.reason = reason;
                return Ok(stats);
            }
            if call_monitors(mons, k, r_norm, 0) {
                return Ok(SolveStats::new(
                    k,
                    r_norm,
                    ConvergedReason::StoppedByMonitor,
                ));
            }

            if r_norm <= thr {
                stats.iterations = k;
                stats.final_residual = r_norm;
                stats.reason = if r_norm <= self.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                };
                return Ok(stats);
            }
            if r_norm >= self.dtol * bnorm {
                stats.iterations = k;
                stats.final_residual = r_norm;
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }

            rho_prev = rho;
            omega_prev = omega;
        }

        let r_norm = red.norm2(r);
        Ok(SolveStats::new(
            self.maxits,
            r_norm,
            ConvergedReason::DivergedMaxIts,
        ))
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

#[cfg(all(test, feature = "backend-faer"))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use faer::Mat;

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
}
