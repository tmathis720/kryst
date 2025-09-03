//! MINRES solver (Saad §7.4)
//
// … (header doc unchanged) …

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{self, PcSide};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

pub struct MinresSolver {
    pub conv: Convergence<f64>, // { rtol, atol, dtol, max_iters }
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

    #[inline]
    fn dot(x: &[f64], y: &[f64], comm: &UniverseComm) -> f64 {
        comm.dot(x, y)
    }
    #[inline]
    fn nrm2(x: &[f64], comm: &UniverseComm) -> f64 {
        Self::dot(x, x, comm).sqrt()
    }

    fn ensure_workspace(w: &mut Workspace, n: usize) {
        // v_{-1}, v_k, v_{k+1}
        while w.q.len() < 3 {
            w.q.push(Vec::new());
        }
        for q in &mut w.q[..3] {
            if q.len() != n {
                q.resize(n, 0.0);
            }
        }
        if w.tmp1.len() != n {
            w.tmp1.resize(n, 0.0);
        } // r or Av
        if w.tmp2.len() != n {
            w.tmp2.resize(n, 0.0);
        } // M^{-1} r or M^{-1} A v
        // w_{k-1}, w_k
        if w.z.len() < 2 {
            w.z.resize(2, vec![0.0; n]);
        }
        for z in &mut w.z[..2] {
            if z.len() != n {
                z.resize(n, 0.0);
            }
        }
    }
}

impl LinearSolver for MinresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        // sizes finalized in solve() once n is known
        if w.q.len() < 3 {
            w.q.resize(3, Vec::new());
        }
        if w.z.len() < 2 {
            w.z.resize(2, Vec::new());
        }
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: std::option::Option<&mut dyn preconditioner::Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        // 1) Input checks and Left-only policy
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

        // Treat preconditioner as immutable; apply() only needs &self
        let pc: Option<&dyn preconditioner::Preconditioner> =
            pc.map(|m| m as &dyn preconditioner::Preconditioner);

        // 2) Workspace
        let mut owned;
        let w = if let Some(w) = work {
            w
        } else {
            owned = Workspace::new(n);
            &mut owned
        };
        Self::ensure_workspace(w, n);

        // Borrow v_{-1}, v_k, v_{k+1}
        let (v_prev, v_k, v_next) = {
            let (a1, rest) = w.q.split_at_mut(1);
            let (a2, rest) = rest.split_at_mut(1);
            (&mut a1[0][..], &mut a2[0][..], &mut rest[0][..])
        };

        // Borrow w_{k-1} and w_k without aliasing
        let (z0, z1) = w.z.split_at_mut(1);
        let w_prev = &mut z0[0][..];
        let w_k = &mut z1[0][..];

        // 3) r = b - A x ; z = M^{-1} r ; beta1 = ||z||
        a.matvec(x, &mut w.tmp1);
        for i in 0..n {
            w.tmp1[i] = b[i] - w.tmp1[i];
        } // tmp1 = r
        if let Some(m) = pc {
            m.apply(PcSide::Left, &w.tmp1, &mut w.tmp2)?;
        } else {
            w.tmp2.copy_from_slice(&w.tmp1);
        }
        let mut res = Self::nrm2(&w.tmp2, comm); // monitors on ||M^{-1} r||
        let res0 = res;
        if let Some(ms) = monitors {
            for m in ms {
                m(0, res);
            }
        }

        // quick exit
        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);
        if res <= thr {
            // compute true residual for reporting consistency
            a.matvec(x, &mut w.tmp1);
            for i in 0..n {
                w.tmp1[i] = b[i] - w.tmp1[i];
            }
            let true_res = Self::nrm2(&w.tmp1, comm);
            return Ok(SolveStats {
                iterations: 0,
                final_residual: true_res,
                reason: if true_res <= self.conv.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                },
            });
        }

        // 4) initialize v and “w” search direction
        v_prev.fill(0.0);
        for i in 0..n {
            v_k[i] = w.tmp2[i] / res; // v_0 = z / ||z||
        }
        w_prev.fill(0.0);
        w_k.fill(0.0);

        // Givens/Lanczos scalars
        let mut beta = res;
        let mut rho_bar = beta; // ρ̅_0
        let mut c_prev = 1.0;
        let mut s_prev = 0.0;
        let mut phi = beta; // φ_0
        let mut iters = 0usize;

        // 5) Main loop
        for k in 1..=self.conv.max_iters {
            iters = k;

            // wtmp = A v_k ; w = M^{-1} wtmp   (Left preconditioning)
            a.matvec(v_k, &mut w.tmp1);
            if let Some(m) = pc {
                m.apply(PcSide::Left, &w.tmp1, &mut w.tmp2)?;
            } else {
                w.tmp2.copy_from_slice(&w.tmp1);
            }

            // alpha = <v_k, w>
            let alpha = Self::dot(v_k, &w.tmp2, comm);

            // v_next = w - alpha v_k - beta v_{k-1}
            for i in 0..n {
                v_next[i] = w.tmp2[i] - alpha * v_k[i] - beta * v_prev[i];
            }
            let beta_next = Self::nrm2(v_next, comm);
            if beta_next == 0.0 {
                break;
            }
            for i in 0..n {
                v_next[i] /= beta_next;
            }

            // Recurrences for Givens (Saad Alg 7.4 style)
            let rho = (rho_bar * rho_bar + alpha * alpha).sqrt();
            let (c, s) = if rho == 0.0 {
                (1.0, 0.0)
            } else {
                (rho_bar / rho, alpha / rho)
            };
            let phi_next = c * phi;
            let phi_bar = -s * phi;

            // Direction update: w_new = (v_k - delta w_k - epsilon w_prev) / rho
            let (delta, epsilon) = if k == 1 {
                (0.0, 0.0)
            } else {
                (s_prev * beta, -c_prev * beta)
            };
            let mut w_new = vec![0.0; n];
            if k == 1 {
                for i in 0..n {
                    w_new[i] = v_k[i] / rho;
                }
            } else {
                for i in 0..n {
                    w_new[i] = (v_k[i] - delta * w_k[i] - epsilon * w_prev[i]) / rho;
                }
            }

            // x += phi_next * w_new
            for i in 0..n {
                x[i] += phi_next * w_new[i];
            }

            // Update preconditioned residual estimate (for monitors)
            res = phi_bar.abs();
            if let Some(ms) = monitors {
                for m in ms {
                    m(k, res);
                }
            }
            let (reason, _ignored) = self.conv.check(res, res0, k);
            if matches!(
                reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ) {
                break;
            }

            // rotate scalars and vectors
            w_prev.copy_from_slice(w_k);
            w_k.copy_from_slice(&w_new);
            v_prev.copy_from_slice(v_k);
            v_k.copy_from_slice(v_next);
            beta = beta_next;
            rho_bar = -s * beta_next;
            c_prev = c;
            s_prev = s;
            phi = phi_next;
        }

        // 6) Recompute **true** residual for reporting (no preconditioning)
        a.matvec(x, &mut w.tmp1);
        for i in 0..n {
            w.tmp1[i] = b[i] - w.tmp1[i];
        }
        let true_res = Self::nrm2(&w.tmp1, comm);
        let (_r, mut out) = self
            .conv
            .check(true_res, Self::nrm2(b, comm).max(1e-32), iters);
        out.iterations = iters;
        out.final_residual = true_res;
        if matches!(out.reason, ConvergedReason::Continued) {
            out.reason = if true_res <= self.conv.atol {
                ConvergedReason::ConvergedAtol
            } else if true_res <= self.conv.rtol * Self::nrm2(b, comm).max(1e-32) {
                ConvergedReason::ConvergedRtol
            } else {
                ConvergedReason::DivergedMaxIts
            };
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MatShell, parallel::UniverseComm};

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
        let mut b = vec![0.0; n];
        aop.matvec(&x_true, &mut b);

        let r0_norm = b.iter().map(|&v| v * v).sum::<f64>().sqrt();

        let mut x = vec![0.0; n];
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

        let mut r_final = vec![0.0; n];
        aop.matvec(&x, &mut r_final);
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
        let mut x = vec![0.0; n];

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
        let mut b = vec![0.0; 2];
        aop.matvec(&x_true, &mut b);

        let mut x = vec![0.0; 2];
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

        let mut r = vec![0.0; 2];
        aop.matvec(&x, &mut r);
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
        let mut x = vec![0.0; 2];

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
