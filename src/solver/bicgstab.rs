//! BiCGStab over &dyn LinOp<f64>, object-safe, with Workspace-backed buffers.
//!
//! Accepts PcSide::Left or ::Right (PC hooks stubbed for now). Residual monitors report
//! the true ||r||2; final stats also report the true norm via KSP context recomputation.
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

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, SolveStats};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::trace;

/// BiCGStab solver (object-safe f64 variant)
pub struct BiCgStabSolver {
    pub rtol: f64,
    pub atol: f64,
    pub dtol: f64,
    pub maxits: usize,
}

impl BiCgStabSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            rtol,
            atol: 1e-12,
            dtol: 1e3,
            maxits,
        }
    }

    #[inline]
    fn dot(x: &[f64], y: &[f64], comm: &UniverseComm) -> f64 {
        comm.dot(x, y)
    }
    #[inline]
    fn nrm2(x: &[f64], comm: &UniverseComm) -> f64 {
        Self::dot(x, x, comm).sqrt()
    }

    fn take_or_resize(buf: &mut Vec<f64>, n: usize) {
        if buf.len() != n {
            buf.resize(n, 0.0);
        }
    }

    /// Acquire all work vectors from the Workspace (no steady-state allocs).
    /// Layout:
    ///   tmp1 = r, tmp2 = r_hat
    ///   q[0] = v, q[1] = p, q[2] = s, q[3] = t
    fn acquire<'a>(
        n: usize,
        work: &'a mut Workspace,
        need_z: bool,
        need_v_raw: bool,
    ) -> (
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        Option<(&'a mut [f64], &'a mut [f64])>,
        Option<&'a mut [f64]>,
    ) {
        Self::take_or_resize(&mut work.tmp1, n); // r
        Self::take_or_resize(&mut work.tmp2, n); // r_hat
        let need_q = if need_v_raw { 5 } else { 4 };
        while work.q.len() < need_q {
            work.q.push(Vec::new());
        }
        for k in 0..need_q {
            Self::take_or_resize(&mut work.q[k], n);
        }
        let r = &mut work.tmp1[..];
        let r_hat = &mut work.tmp2[..];
        let (q0, rest) = work.q.split_at_mut(1);
        let (q1, rest) = rest.split_at_mut(1);
        let (q2, rest) = rest.split_at_mut(1);
        let (q3, q_more) = rest.split_at_mut(1);
        let v = &mut q0[0][..];
        let p = &mut q1[0][..];
        let s = &mut q2[0][..];
        let t = &mut q3[0][..];
        let z = if need_z {
            while work.z.len() < 2 {
                work.z.push(Vec::new());
            }
            for k in 0..2 {
                Self::take_or_resize(&mut work.z[k], n);
            }
            let (z0, z1) = work.z.split_at_mut(1);
            Some((&mut z0[0][..], &mut z1[0][..]))
        } else {
            None
        };
        let v_raw = if need_v_raw {
            Some(&mut q_more[0][..])
        } else {
            None
        };
        (r, r_hat, v, p, s, t, z, v_raw)
    }
}

impl LinearSolver for BiCgStabSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        if w.q.len() < 4 {
            w.q.resize(4, Vec::new());
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
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("BiCGStab");

        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "BiCGStab: square operator and matching b,x required".into(),
            ));
        }
        let mons = monitors.unwrap_or(&[]);

        // Acquire workspace (required to avoid allocs)
        let side = match pc_side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        };
        let w = work.ok_or_else(|| KError::InvalidInput("BiCGStab requires a Workspace".into()))?;
        let need_right = matches!(side, PcSide::Right) && pc.is_some();
        let need_left = matches!(side, PcSide::Left) && pc.is_some();
        let need_z = need_right || need_left;
        let need_v_raw = need_left;
        let (r, r_hat, v, p, s, t, zopt, mut v_raw_opt) = Self::acquire(n, w, need_z, need_v_raw);
        let mut z_p_opt: Option<&mut [f64]> = None;
        let mut z_s_opt: Option<&mut [f64]> = None;
        if let Some((zp, zs)) = zopt {
            z_p_opt = Some(zp);
            z_s_opt = Some(zs);
        }

        // r = b - A x
        if x.iter().any(|&xi| xi != 0.0) {
            a.matvec(x, v); // reuse v as Ax
            for i in 0..n {
                r[i] = b[i] - v[i];
            }
        } else {
            r.copy_from_slice(b);
        }
        // residual norms / thresholds
        let res0;
        if need_left {
            // Preconditioned residual z = M^{-1} r
            if let Some(zs) = z_s_opt.as_deref_mut() {
                // Use z_s buffer temporarily for z0
                if let Some(pc) = pc.as_deref() {
                    pc.apply(PcSide::Left, r, zs)?;
                } else {
                    zs.copy_from_slice(r);
                }
                r_hat.copy_from_slice(zs); // shadow residual = z0
                s.copy_from_slice(zs); // current z stored in s
                res0 = Self::nrm2(s, comm);
                // Initialize p := z
                p.copy_from_slice(s);
            } else {
                // Fallback shouldn't happen; treat as unpreconditioned
                r_hat.copy_from_slice(r);
                res0 = Self::nrm2(r, comm);
                p.copy_from_slice(r);
            }
        } else {
            // r_hat = r (fixed shadow residual)
            r_hat.copy_from_slice(r);
            res0 = Self::nrm2(r, comm);
            // Initialize p := r
            p.copy_from_slice(r);
        }
        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let thr = self.atol.max(self.rtol * bnorm);

        if !mons.is_empty() {
            for m in mons {
                m(0, res0);
            }
        }
        if res0 <= thr {
            return Ok(SolveStats {
                iterations: 0,
                final_residual: res0,
                reason: if res0 <= self.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                },
            });
        }

        // Parameters
        let mut rho_prev = 1.0;
        let mut alpha = 1.0;
        let mut omega_prev = 1.0;

        // Breakdown epsilons (relative-safe)
        let eps_rho = 1e-30_f64;
        let eps_alpha = 1e-30_f64;
        let eps_omega = 1e-30_f64;

        let mut stats = SolveStats {
            iterations: 0,
            final_residual: res0,
            reason: ConvergedReason::Continued,
        };

        for k in 1..=self.maxits {
            // ρ_k = <r_hat, r> (Right/unpreconditioned) or <r_hat, z> (Left)
            let rho = if need_left {
                Self::dot(r_hat, s, comm)
            } else {
                Self::dot(r_hat, r, comm)
            };
            if rho.abs() <= eps_rho || !rho.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: rho ~ 0 at iter {}", k);
                stats.iterations = k - 1;
                stats.final_residual = if need_left {
                    Self::nrm2(s, comm)
                } else {
                    Self::nrm2(r, comm)
                };
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }

            // β = (ρ/ρ_{k-1}) * (α/ω_{k-1});
            let beta = if k == 1 {
                0.0
            } else {
                (rho / rho_prev) * (alpha / omega_prev)
            };
            if need_left {
                // p = z + β (p − ω ṽ)
                for i in 0..n {
                    p[i] = s[i] + beta * (p[i] - omega_prev * v[i]);
                }
            } else {
                // p = r + β (p − ω v)
                for i in 0..n {
                    p[i] = r[i] + beta * (p[i] - omega_prev * v[i]);
                }
            }

            if need_left {
                // y = M^{-1} p (use z_p buffer)
                let yp = match (pc.as_deref(), z_p_opt.as_deref_mut()) {
                    (Some(pc), Some(zp)) => {
                        pc.apply(PcSide::Left, p, zp)?;
                        zp
                    }
                    _ => p,
                };
                // v_raw = A y
                let vr = v_raw_opt
                    .as_deref_mut()
                    .expect("workspace: missing v_raw buffer");
                a.matvec(yp, vr);
                // ṽ = M^{-1} v_raw -> store in v
                if let Some(pc) = pc.as_deref() {
                    pc.apply(PcSide::Left, vr, v)?;
                } else {
                    v.copy_from_slice(vr);
                }
            } else {
                // v = A p (Right PC: v = A (M^{-1} p))
                match (side, pc.as_deref(), z_p_opt.as_deref_mut()) {
                    (PcSide::Right, Some(pc), Some(zp)) => {
                        pc.apply(PcSide::Right, p, zp)?;
                        a.matvec(zp, v);
                    }
                    _ => {
                        a.matvec(p, v);
                    }
                }
            }

            // α = ρ / <r_hat, v> (Right) or ρ / <r_hat, ṽ> (Left)
            let alpha_den = Self::dot(r_hat, v, comm);
            if alpha_den.abs() <= eps_alpha || !alpha_den.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: alpha_den ~ 0 at iter {}", k);
                stats.iterations = k - 1;
                stats.final_residual = if need_left {
                    Self::nrm2(s, comm)
                } else {
                    Self::nrm2(r, comm)
                };
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }
            alpha = rho / alpha_den;

            if need_left {
                // s = z − α ṽ (reuse s buffer which held z)
                for i in 0..n {
                    s[i] = s[i] - alpha * v[i];
                }
            } else {
                // s = r − α v
                for i in 0..n {
                    s[i] = r[i] - alpha * v[i];
                }
            }

            // Early exit if ||s|| is tiny
            let s_norm = Self::nrm2(s, comm);
            if !mons.is_empty() {
                for m in mons {
                    m(k, s_norm);
                }
            }
            if s_norm <= thr {
                if need_left {
                    if let Some(yp) = z_p_opt.as_deref() {
                        for i in 0..n {
                            x[i] += alpha * yp[i];
                        }
                    } else {
                        for i in 0..n {
                            x[i] += alpha * p[i];
                        }
                    }
                } else {
                    for i in 0..n {
                        x[i] += alpha * p[i];
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

            // t path
            if need_left {
                // tpre = M^{-1} s (use z_s buffer), then At = A tpre -> store in t
                let zs = z_s_opt
                    .as_deref_mut()
                    .expect("workspace: missing z_s buffer");
                if let Some(pc) = pc.as_deref() {
                    pc.apply(PcSide::Left, s, zs)?;
                } else {
                    zs.copy_from_slice(s);
                }
                a.matvec(zs, t);
            } else {
                // t = A s (Right PC: t = A (M^{-1} s))
                match (side, pc.as_deref(), z_s_opt.as_deref_mut()) {
                    (PcSide::Right, Some(pc), Some(zs)) => {
                        pc.apply(PcSide::Right, s, zs)?;
                        a.matvec(zs, t);
                    }
                    _ => {
                        a.matvec(s, t);
                    }
                }
            }

            // ω computation
            let omega_den = Self::dot(t, t, comm);
            if omega_den.abs() <= eps_omega || !omega_den.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: omega_den ~ 0 at iter {}", k);
                stats.iterations = k;
                stats.final_residual = Self::nrm2(s, comm);
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }
            let omega = Self::dot(t, s, comm) / omega_den;
            if omega.abs() <= eps_omega || !omega.is_finite() {
                #[cfg(feature = "logging")]
                trace!("BiCGStab breakdown: omega ~ 0 at iter {}", k);
                stats.iterations = k;
                stats.final_residual = Self::nrm2(s, comm);
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }

            // x update
            if need_left {
                let y = z_p_opt.as_deref();
                let tpre = z_s_opt.as_deref();
                match (y, tpre) {
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
                match (side, pc.as_deref(), z_p_opt.as_deref(), z_s_opt.as_deref()) {
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

            // r update (true residual)
            if need_left {
                // r = r − α v_raw − ω (A tpre) = r − α v_raw − ω t
                let vr = v_raw_opt
                    .as_deref()
                    .expect("workspace: missing v_raw buffer");
                for i in 0..n {
                    r[i] -= alpha * vr[i] + omega * t[i];
                }
                // z = M^{-1} r for next iteration
                let zs = z_s_opt
                    .as_deref_mut()
                    .expect("workspace: missing z_s buffer");
                if let Some(pc) = pc.as_deref() {
                    pc.apply(PcSide::Left, r, zs)?;
                } else {
                    zs.copy_from_slice(r);
                }
                s.copy_from_slice(zs);
            } else {
                // r = s − ω t
                for i in 0..n {
                    r[i] = s[i] - omega * t[i];
                }
            }

            // check convergence on true ||r||
            let r_norm = if need_left {
                Self::nrm2(s, comm)
            } else {
                Self::nrm2(r, comm)
            };
            if !mons.is_empty() {
                for m in mons {
                    m(k, r_norm);
                }
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
            if !r_norm.is_finite() || r_norm >= self.dtol * bnorm {
                stats.iterations = k;
                stats.final_residual = r_norm;
                stats.reason = ConvergedReason::DivergedDtol;
                return Ok(stats);
            }

            rho_prev = rho;
            omega_prev = omega;
        }

        // Max iters
        let r_norm = Self::nrm2(r, comm);
        Ok(SolveStats {
            iterations: self.maxits,
            final_residual: r_norm,
            reason: ConvergedReason::DivergedMaxIts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use faer::Mat;

    // Helper: random well-conditioned non-symmetric 3x3 matrix
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
        let comm = crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm);
        let mut ws = Workspace::new(3);
        solver.setup_workspace(&mut ws);
        let stats = solver
            .solve(
                &a,
                None,
                &b,
                &mut x,
                crate::preconditioner::PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .unwrap();
        eprintln!(
            "BiCGStab stats: {{ reason: {:?}, iters: {}, final_res: {:e} }}",
            stats.reason, stats.iterations, stats.final_residual
        );
        // Compare to true solution
        let x_true = vec![1.0, 2.0, 3.0];
        for i in 0..3 {
            assert_abs_diff_eq!(x[i], x_true[i], epsilon = 1e-8);
        }
        assert!(
            matches!(
                stats.reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ),
            "BiCGStab did not converge: stats = {:?}",
            stats
        );
    }

    // A scripted LinOp that returns preset outputs for successive matvec calls.
    // Ignores the input x; used to force breakdown scenarios deterministically.
    struct ScriptedOp {
        n: usize,
        seq: std::sync::Arc<Vec<Vec<f64>>>,
        idx: std::sync::atomic::AtomicUsize,
    }
    impl crate::matrix::op::LinOp for ScriptedOp {
        type S = f64;
        fn dims(&self) -> (usize, usize) {
            (self.n, self.n)
        }
        fn matvec(&self, _x: &[f64], y: &mut [f64]) {
            use std::sync::atomic::Ordering;
            let i = self.idx.fetch_add(1, Ordering::Relaxed);
            if i < self.seq.len() {
                y.copy_from_slice(&self.seq[i]);
            } else {
                y.fill(0.0);
            }
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn run_with_scripted(seq: Vec<Vec<f64>>, b: Vec<f64>) -> SolveStats<f64> {
        let n = b.len();
        let op = ScriptedOp {
            n,
            seq: std::sync::Arc::new(seq),
            idx: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut x = vec![0.0; n];
        let mut solver = BiCgStabSolver::new(1e-10, 50);
        let comm = crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm);
        let mut ws = Workspace::new(n);
        solver.setup_workspace(&mut ws);
        solver
            .solve(
                &op,
                None,
                &b,
                &mut x,
                crate::preconditioner::PcSide::Left,
                &comm,
                None,
                Some(&mut ws),
            )
            .unwrap()
    }

    #[test]
    fn bicgstab_alpha_den_breakdown() {
        // r0 = b = e1; Ax = 0 initially; v = e2 => <r_hat, v> = 0 triggers alpha_den breakdown
        let b = vec![1.0, 0.0, 0.0];
        let seq = vec![
            vec![0.0, 1.0, 0.0], // v = A p
        ];
        let stats = run_with_scripted(seq, b);
        assert_eq!(stats.reason, ConvergedReason::DivergedDtol);
    }

    #[test]
    fn bicgstab_omega_den_breakdown() {
        // r0 = e1; v = [1,1,0] -> alpha = 1; s = [0,-1,0]; t = 0 => omega_den = 0
        let b = vec![1.0, 0.0, 0.0];
        let seq = vec![
            vec![1.0, 1.0, 0.0], // v = A p
            vec![0.0, 0.0, 0.0], // t = A s
        ];
        let stats = run_with_scripted(seq, b);
        assert_eq!(stats.reason, ConvergedReason::DivergedDtol);
    }

    #[test]
    fn bicgstab_omega_zero_breakdown() {
        // r0 = e1; v = [1,1,0] -> s = [0,-1,0]; t = [1,0,0] orthogonal => omega = 0
        let b = vec![1.0, 0.0, 0.0];
        let seq = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
        let stats = run_with_scripted(seq, b);
        assert_eq!(stats.reason, ConvergedReason::DivergedDtol);
    }

    #[test]
    fn bicgstab_rho_zero_breakdown_second_iter() {
        // 3D: r0 = e1; choose v=[1,-1,0] => alpha=1, s=e2; t=e2+e3 => omega=1/2
        // r1 = s - 0.5 t = 0.5 e2 - 0.5 e3, orthogonal to r0; next rho=0 ⇒ breakdown
        let b = vec![1.0, 0.0, 0.0];
        let seq = vec![
            vec![1.0, -1.0, 0.0], // v = A p
            vec![0.0, 1.0, 1.0],  // t = A s
        ];
        let stats = run_with_scripted(seq, b);
        assert_eq!(stats.reason, ConvergedReason::DivergedDtol);
        // Ensure at least one iteration executed
        assert!(stats.iterations >= 1);
    }
}
