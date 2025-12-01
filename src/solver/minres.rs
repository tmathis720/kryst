//! MINRES solver (Saad §7.4)
//
// … (header doc unchanged) …

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::{Comm, UniverseComm, global_dot_conj_many_into, global_nrm2_many_into};
use crate::preconditioner::{self, PcSide, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::common::{dot_result_to_real, recompute_true_residual_norm_s};
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

struct MinresWorkspace<'a> {
    v_prev: &'a mut [S],
    v_k: &'a mut [S],
    v_next: &'a mut [S],
    w_prev: &'a mut [S],
    w_k: &'a mut [S],
    tmp1: &'a mut [S],
    tmp2: &'a mut [S],
    w_new: &'a mut [S],
    scratch: &'a mut crate::algebra::bridge::BridgeScratch,
}

impl<'a> MinresWorkspace<'a> {
    fn acquire(work: &'a mut Workspace, n: usize) -> Self {
        while work.q_s.len() < 3 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..3] {
            if buf.len() != n {
                buf.resize(n, S::zero());
            }
        }
        while work.z_s.len() < 2 {
            work.z_s.push(Vec::new());
        }
        for buf in &mut work.z_s[..2] {
            if buf.len() != n {
                buf.resize(n, S::zero());
            }
        }
        if work.tmp1.len() != n {
            work.tmp1.resize(n, S::zero());
        }
        if work.tmp2.len() != n {
            work.tmp2.resize(n, S::zero());
        }
        if work.bridge_tmp.len() != n {
            work.bridge_tmp.resize(n, S::zero());
        }
        let (q0, rest) = work.q_s.split_at_mut(1);
        let (q1, rest) = rest.split_at_mut(1);
        let (q2, _) = rest.split_at_mut(1);
        let (z0, rest) = work.z_s.split_at_mut(1);
        let (z1, _) = rest.split_at_mut(1);
        Self {
            v_prev: &mut q0[0][..n],
            v_k: &mut q1[0][..n],
            v_next: &mut q2[0][..n],
            w_prev: &mut z0[0][..n],
            w_k: &mut z1[0][..n],
            tmp1: &mut work.tmp1[..n],
            tmp2: &mut work.tmp2[..n],
            w_new: &mut work.bridge_tmp[..n],
            scratch: &mut work.bridge,
        }
    }
}

pub struct MinresSolver {
    pub conv: Convergence, // { rtol, atol, dtol, max_iters }
}

impl MinresSolver {
    pub fn new(rtol: f64, max_iters: usize) -> Self {
        Self {
            conv: Convergence {
                rtol,
                atol: 1e-12,
                dtol: 1e3,
                max_iters,
            },
        }
    }

    fn solve_internal<A>(
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
        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "MINRES: dimension mismatch or non-square A".into(),
            ));
        }
        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "MINRES requires Left preconditioning (SPD M)".into(),
            ));
        }

        let monitors = monitors.unwrap_or(&[]);

        let mut owned;
        let work = if let Some(work) = work {
            work
        } else {
            owned = Workspace::new(n);
            &mut owned
        };
        let mut buffers = MinresWorkspace::acquire(work, n);
        let MinresWorkspace {
            v_prev,
            v_k,
            v_next,
            w_prev,
            w_k,
            tmp1,
            tmp2,
            w_new,
            scratch,
        } = &mut buffers;

        // r = b - A x
        a.matvec_s(x, tmp1, scratch);
        for i in 0..n {
            tmp1[i] = b[i] - tmp1[i];
        }
        if let Some(pc) = pc {
            pc.apply_s(PcSide::Left, tmp1, tmp2, scratch)?;
        } else {
            tmp2.copy_from_slice(tmp1);
        }

        let mut batched_norms = [R::default(); 2];
        let tmp2_view: &[S] = &tmp2[..n];
        global_nrm2_many_into(comm, &[tmp2_view, b], &mut batched_norms);
        let mut res = batched_norms[0];
        let res0 = res;
        for m in monitors {
            m(0, res);
        }

        let bnorm = batched_norms[1].max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);
        if res <= thr {
            let reason = if res <= self.conv.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            return Ok(SolveStats::new(0, res, reason));
        }

        v_prev.fill(S::zero());
        let res_s = S::from_real(res);
        for i in 0..n {
            v_k[i] = tmp2[i] / res_s;
        }
        w_prev.fill(S::zero());
        w_k.fill(S::zero());

        let mut beta = res;
        let mut rho_bar = beta;
        let mut c_prev = 1.0;
        let mut s_prev = R::default();
        let mut phi = beta;
        let mut final_reason = ConvergedReason::Continued;
        let mut iters = 0usize;

        let mut dot_results = [S::zero(); 3];

        for k in 1..=self.conv.max_iters {
            iters = k;

            a.matvec_s(v_k, tmp1, scratch);
            if let Some(pc) = pc {
                pc.apply_s(PcSide::Left, tmp1, tmp2, scratch)?;
            } else {
                tmp2.copy_from_slice(tmp1);
            }

            let (alpha, prev_projection, tmp2_norm_sq) = {
                let v_k_view: &[S] = &v_k[..n];
                let v_prev_view: &[S] = &v_prev[..n];
                let tmp2_view: &[S] = &tmp2[..n];

                let mut pairs: [(&[S], &[S]); 3] = [(&[], &[]); 3];
                let mut used = 0usize;
                pairs[used] = (v_k_view, tmp2_view);
                used += 1;
                if k > 1 {
                    pairs[used] = (v_prev_view, tmp2_view);
                    used += 1;
                }
                pairs[used] = (tmp2_view, tmp2_view);
                used += 1;

                global_dot_conj_many_into(comm, &pairs[..used], &mut dot_results[..used]);

                let alpha = dot_result_to_real(dot_results[0]);
                let prev_proj = if k > 1 {
                    dot_result_to_real(dot_results[1])
                } else {
                    R::default()
                };
                let tmp2_norm_sq = dot_result_to_real(dot_results[used - 1]);
                (alpha, prev_proj, tmp2_norm_sq)
            };

            let alpha_s = S::from_real(alpha);
            let beta_s = S::from_real(beta);
            for i in 0..n {
                v_next[i] = tmp2[i] - alpha_s * v_k[i] - beta_s * v_prev[i];
            }

            let mut beta_next_sq = tmp2_norm_sq - alpha * alpha;
            if k > 1 {
                beta_next_sq =
                    tmp2_norm_sq + beta * beta - alpha * alpha - 2.0 * beta * prev_projection;
            }
            if beta_next_sq < R::default() {
                beta_next_sq = R::default();
            }
            if comm.size() <= 1 {
                let mut local_sq = R::default();
                for &val in &v_next[..n] {
                    let mag = val.abs();
                    local_sq += mag * mag;
                }
                beta_next_sq = local_sq;
            }
            let mut beta_next = beta_next_sq.sqrt();
            if beta_next <= R::from(1e-30) {
                beta_next = R::default();
            }
            if beta_next == R::default() {
                final_reason = ConvergedReason::ConvergedAtol;
                break;
            }
            if beta_next > R::default() {
                let beta_next_s = S::from_real(beta_next);
                for val in &mut v_next[..n] {
                    *val /= beta_next_s;
                }
            }

            let rho = (rho_bar * rho_bar + alpha * alpha).sqrt();
            let (c, s_val) = if rho <= R::default() {
                (R::from(1.0), R::default())
            } else {
                (rho_bar / rho, alpha / rho)
            };
            let phi_next = c * phi;
            let phi_bar = -s_val * phi;

            let (delta, epsilon) = if k == 1 {
                (R::default(), R::default())
            } else {
                (s_prev * beta, -c_prev * beta)
            };

            if k == 1 {
                let rho_s = S::from_real(rho);
                for i in 0..n {
                    w_new[i] = v_k[i] / rho_s;
                }
            } else {
                let rho_s = S::from_real(rho);
                let delta_s = S::from_real(delta);
                let epsilon_s = S::from_real(epsilon);
                for i in 0..n {
                    let numer = v_k[i] - delta_s * w_k[i] - epsilon_s * w_prev[i];
                    w_new[i] = numer / rho_s;
                }
            }

            let phi_next_s = S::from_real(phi_next);
            for i in 0..n {
                x[i] += phi_next_s * w_new[i];
            }

            res = phi_bar.abs();
            for m in monitors {
                m(k, res);
            }
            let (reason, _) = self.conv.check(res, res0, k);
            if matches!(
                reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ) {
                final_reason = reason;
                break;
            }

            w_prev.copy_from_slice(w_k);
            w_k.copy_from_slice(w_new);
            v_prev.copy_from_slice(v_k);
            v_k.copy_from_slice(v_next);
            beta = beta_next;
            rho_bar = -s_val * beta_next;
            c_prev = c;
            s_prev = s_val;
            phi = phi_next;
        }

        let true_res = recompute_true_residual_norm_s(a, b, x, comm, tmp1, scratch);
        let (reason_post, mut stats) = self.conv.check(true_res, bnorm, iters);
        let reason = match final_reason {
            ConvergedReason::Continued => reason_post,
            other => other,
        };
        if matches!(reason, ConvergedReason::Continued) {
            stats.reason = ConvergedReason::DivergedMaxIts;
        } else {
            stats.reason = reason;
        }
        stats.iterations = iters;
        stats.final_residual = true_res;
        Ok(stats)
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
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        self.solve_internal(a, pc, b, x, pc_side, comm, monitors, work)
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
            self.solve_internal(&op, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
        }

        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result =
                self.solve_internal(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work);
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
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

impl LinearSolver for MinresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        if w.q_s.len() < 3 {
            w.q_s.resize(3, Vec::new());
        }
        if w.z_s.len() < 2 {
            w.z_s.resize(2, Vec::new());
        }
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn preconditioner::Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let pc = pc.map(|m| m as &dyn PreconditionerF64);
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::matrix::MatShell;
    use crate::parallel::UniverseComm;

    // Helper to make a MatShell from a closure
    fn matshell_from<F: Fn(&[f64], &mut [f64]) + Send + Sync + 'static>(
        n: usize,
        f: F,
    ) -> MatShell<f64> {
        MatShell::new(n, n, f)
    }

    #[test]
    #[ignore]
    fn minres_reduces_residual_on_spd() {
        // A small SPD matrix (3×3):
        //   A = [[4,1,0],
        //        [1,3,1],
        //        [0,1,2]]
        let n = 3usize;
        let aop = matshell_from(n, move |x, y| {
            let a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
            for i in 0..3 {
                y[i] = a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2];
            }
        });

        let x_true = vec![1.0, 2.0, 3.0];
        let mut b = vec![R::default(); n];
        crate::matrix::op::LinOp::matvec(&aop, &x_true, &mut b);

        let r0_norm = b.iter().map(|&v| v * v).sum::<f64>().sqrt();

        let mut x = vec![R::default(); n];
        let mut solver = MinresSolver::new(1e-6, 100);
        let stats = solver
            .solve(
                &aop,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            )
            .unwrap();

        let mut r_final = vec![R::default(); n];
        crate::matrix::op::LinOp::matvec(&aop, &x, &mut r_final);
        for i in 0..n {
            r_final[i] = b[i] - r_final[i];
        }
        let r_final_norm = r_final.iter().map(|&v| v * v).sum::<f64>().sqrt();

        assert!(
            r_final_norm < 0.5 * r0_norm,
            "MINRES insufficient reduction: initial = {:.3e}, final = {:.3e}",
            r0_norm,
            r_final_norm
        );
        assert!(stats.iterations <= 10, "Too many iterations");
    }

    #[test]
    #[ignore]
    fn minres_solves_identity() {
        let n = 5usize;
        let aop = matshell_from(n, move |x, y| {
            for i in 0..n {
                y[i] = x[i];
            }
        });

        let b = vec![0.5, -1.2, 3.0, 4.4, -2.2];
        let mut x = vec![R::default(); n];

        let mut solver = MinresSolver::new(1e-14, 100);
        let stats = solver
            .solve(
                &aop,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            )
            .unwrap();

        for i in 0..n {
            assert!(
                (x[i] - b[i]).abs() <= 1e-10,
                "x[{}]={:.6}, b[{}]={:.6}",
                i,
                x[i],
                i,
                b[i]
            );
        }
        assert!(
            stats.iterations <= 2,
            "expected <= 2 MINRES iterations on I"
        );
        assert!(
            matches!(
                stats.reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ),
            "MINRES did not report Converged reason"
        );
    }

    #[test]
    #[ignore]
    fn minres_solves_symmetric_indefinite() {
        // A = [[0,1],[1,0]]
        let aop = matshell_from(2, move |x, y| {
            y[0] = x[1];
            y[1] = x[0];
        });

        let x_true = vec![1.0, 1.0];
        let mut b = vec![R::default(); 2];
        crate::matrix::op::LinOp::matvec(&aop, &x_true, &mut b);

        let mut x = vec![R::default(); 2];
        let mut solver = MinresSolver::new(1e-12, 100);
        let stats = solver
            .solve(
                &aop,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &UniverseComm::NoComm(crate::parallel::NoComm),
                None,
                None,
            )
            .unwrap();

        let mut r = vec![R::default(); 2];
        crate::matrix::op::LinOp::matvec(&aop, &x, &mut r);
        for i in 0..2 {
            r[i] = b[i] - r[i];
        }
        let res_norm = (r[0] * r[0] + r[1] * r[1]).sqrt();

        let tol = 1e-8;
        assert!(
            res_norm <= tol,
            "MINRES failed to drive residual small: ||r|| = {:.3e}, tol = {:.3e}",
            res_norm,
            tol
        );
        assert!(
            matches!(
                stats.reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ),
            "MINRES did not report Converged reason"
        );
    }

    #[test]
    #[ignore]
    fn test_minres_with_monitors() {
        use std::sync::{Arc, Mutex};

        // SPD 2x2: [[2,1],[1,2]]
        let aop = matshell_from(2, move |x, y| {
            y[0] = 2.0 * x[0] + 1.0 * x[1];
            y[1] = 1.0 * x[0] + 2.0 * x[1];
        });

        let b = vec![3.0, 3.0]; // solution x = [1,1]
        let mut x = vec![R::default(); 2];

        let monitor_data = Arc::new(Mutex::new(Vec::<(usize, f64)>::new()));
        let monitor_data_clone = monitor_data.clone();

        let monitor: Box<dyn Fn(usize, f64) + Send + Sync> = Box::new(move |iter, residual| {
            monitor_data_clone.lock().unwrap().push((iter, residual));
        });
        let monitors = vec![monitor];

        let mut solver = MinresSolver::new(1e-8, 10);
        let _stats = solver
            .solve(
                &aop,
                None,
                &b,
                &mut x,
                PcSide::Left,
                &UniverseComm::NoComm(crate::parallel::NoComm),
                Some(&monitors),
                None,
            )
            .unwrap();

        let captured = monitor_data.lock().unwrap();
        assert!(!captured.is_empty(), "Monitors should have been called");
        for (i, &(iter, _)) in captured.iter().enumerate() {
            assert_eq!(iter, i + 1, "Iteration numbers should be sequential");
        }
        for i in 1..captured.len() {
            let prev = captured[i - 1].1;
            let curr = captured[i].1;
            assert!(
                curr <= prev * 2.0,
                "Residual should generally decrease: {} -> {}",
                prev,
                curr
            );
        }
    }
}
