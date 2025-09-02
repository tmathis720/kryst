use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::recompute_true_residual_norm;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
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

pub struct PcgSolver {
    pub(crate) conv: Convergence<f64>,
    norm_type: CgNormType,
    single_reduction: bool,
    reproducible: bool,
    true_residual_monitor: Option<Box<dyn Fn(usize, f64) + Send + Sync>>,
    /// Whether the initial guess in `x` should be treated as nonzero.
    ///
    /// PETSc zeroes the initial guess by default unless told otherwise via
    /// `KSPSetInitialGuessNonzero`. We follow the same policy; when this flag is
    /// `false` and the provided `x` is numerically zero, the solver skips the
    /// initial matvec and assumes a zero guess.
    initial_guess_nonzero: bool,
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
            single_reduction: false,
            reproducible: false,
            true_residual_monitor: None,
            initial_guess_nonzero: false,
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

    pub fn with_single_reduction(mut self, f: bool) -> Self {
        self.single_reduction = f;
        self
    }

    /// Enable a more reproducible (but slightly slower) local dot product using
    /// Kahan summation. When combined with a deterministic MPI reduction this
    /// yields bitwise-identical results across runs.
    pub fn with_reproducible_dot(mut self, f: bool) -> Self {
        self.reproducible = f;
        self
    }

    /// Install a monitor that receives the true residual norm `||b - A x||₂`
    /// at each iteration. This incurs an extra matvec per iteration and is
    /// intended for debugging.
    pub fn with_true_residual_monitor(mut self, m: Box<dyn Fn(usize, f64) + Send + Sync>) -> Self {
        self.true_residual_monitor = Some(m);
        self
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
        self.reproducible = f;
    }

    /// Set or clear the true residual monitor after construction.
    pub fn set_true_residual_monitor(&mut self, m: Option<Box<dyn Fn(usize, f64) + Send + Sync>>) {
        self.true_residual_monitor = m;
    }

    #[inline]
    fn dot<C: Comm>(&self, u: &[f64], v: &[f64], comm: &C) -> f64 {
        if self.reproducible {
            let local = Self::local_dot_kahan(u, v);
            if comm.size() == 1 {
                local
            } else {
                comm.allreduce_sum(local)
            }
        } else {
            comm.dot(u, v)
        }
    }

    #[inline]
    fn local_dot(u: &[f64], v: &[f64]) -> f64 {
        u.iter().zip(v).map(|(a, b)| a * b).sum::<f64>()
    }

    #[inline]
    fn local_dot_kahan(u: &[f64], v: &[f64]) -> f64 {
        let mut sum = 0.0f64;
        let mut c = 0.0f64;
        for (a, b) in u.iter().zip(v.iter()) {
            let y = a * b - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }

    #[inline]
    fn nrm2<C: Comm>(&self, u: &[f64], comm: &C) -> f64 {
        self.dot(u, u, comm).sqrt()
    }

    fn take_or_resize(buf: &mut Vec<f64>, n: usize) {
        if buf.len() != n {
            buf.resize(n, 0.0);
        }
    }

    pub fn set_single_reduction(&mut self, f: bool) {
        self.single_reduction = f;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_comm<C: Comm>(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &C,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        self.solve_impl(a, pc, b, x, pc_side, comm, monitors, work)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_impl<C: Comm>(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &C,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let pc: Option<&dyn Preconditioner> = pc.as_deref();
        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "CG requires left preconditioning with SPD M; use MINRES or GMRES otherwise".into(),
            ));
        }

        let n = b.len();
        if x.len() != n {
            return Err(KError::InvalidInput("dimension mismatch: x,b".into()));
        }

        // Acquire buffers either from workspace or allocate temporaries
        let mut r_store = Vec::new();
        let mut z_store = Vec::new();
        let mut p_store = Vec::new();
        let mut w_store = Vec::new();
        let mut zprev_store = Vec::new();

        let (r, z, p, w, z_prev): (&mut [f64], &mut [f64], &mut [f64], &mut [f64], &mut [f64]);

        if let Some(wk) = work {
            Self::take_or_resize(&mut wk.tmp1, n);
            Self::take_or_resize(&mut wk.tmp2, n);
            while wk.q.len() < 3 {
                wk.q.push(vec![0.0; n]);
            }
            for v in &mut wk.q[0..3] {
                Self::take_or_resize(v, n);
            }
            r = wk.tmp1.as_mut_slice();
            z = wk.tmp2.as_mut_slice();
            let (pbuf, rest) = wk.q.split_at_mut(1);
            let (wbuf, rest2) = rest.split_at_mut(1);
            p = pbuf[0].as_mut_slice();
            w = wbuf[0].as_mut_slice();
            z_prev = rest2[0].as_mut_slice();
        } else {
            r_store.resize(n, 0.0);
            z_store.resize(n, 0.0);
            p_store.resize(n, 0.0);
            w_store.resize(n, 0.0);
            zprev_store.resize(n, 0.0);
            r = r_store.as_mut_slice();
            z = z_store.as_mut_slice();
            p = p_store.as_mut_slice();
            w = w_store.as_mut_slice();
            z_prev = zprev_store.as_mut_slice();
        }

        let mut tmp_true = if self.true_residual_monitor.is_some() {
            vec![0.0; n]
        } else {
            Vec::new()
        };

        // r = b - A x
        let zero_guess = !self.initial_guess_nonzero && x.iter().all(|&xi| xi == 0.0);
        if zero_guess {
            r.copy_from_slice(b);
        } else {
            let mut ax = vec![0.0; n];
            a.matvec(x, &mut ax);
            for i in 0..n {
                r[i] = b[i] - ax[i];
            }
        }

        // z = M^{-1} r
        if let Some(pc) = pc {
            pc.apply(pc_side, r, z)?;
        } else {
            z.copy_from_slice(r);
        }

        let bnorm = self.nrm2(b, comm).max(1e-32);
        let mut rho = self.dot(r, z, comm);
        let mut rho_prev = rho;
        let mut pending_rz_local = 0.0f64;
        let mut has_pending_rz = false;

        let mut res = match self.norm_type {
            CgNormType::Preconditioned => rho.sqrt(),
            CgNormType::Unpreconditioned => self.nrm2(r, comm),
            CgNormType::Natural => self.nrm2(z, comm),
            CgNormType::None => 0.0,
        };

        if let Some(ms) = monitors {
            for m in ms {
                m(0, res);
            }
        }
        if let Some(m) = &self.true_residual_monitor {
            let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp_true);
            m(0, true_res);
        }

        p.copy_from_slice(z);

        // Convergence check at k=0
        let res_check0 = match self.norm_type {
            CgNormType::Preconditioned => rho.sqrt(),
            CgNormType::Unpreconditioned => self.nrm2(r, comm),
            CgNormType::Natural => self.nrm2(z, comm),
            CgNormType::None => self.nrm2(r, comm),
        };
        let (reason0, s0) = self.conv.check(res_check0, bnorm, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let mut tmp = vec![0.0; n];
            let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
            return Ok(SolveStats {
                iterations: 0,
                final_residual: true_res,
                reason: s0.reason,
            });
        }

        let mut iters = 0usize;
        for k in 1..=self.conv.max_iters {
            iters = k;

            if self.single_reduction {
                if k > 1 {
                    let beta = rho / rho_prev;
                    for i in 0..n {
                        p[i] = z_prev[i] + beta * p[i];
                    }
                }
            } else if k > 1 {
                let beta = rho / rho_prev;
                for i in 0..n {
                    p[i] = z[i] + beta * p[i];
                }
            }

            a.matvec(p, w);

            if self.single_reduction {
                let (pw, rho_k) = if has_pending_rz {
                    let pw_local = if self.reproducible {
                        Self::local_dot_kahan(p, w)
                    } else {
                        Self::local_dot(p, w)
                    };
                    let (pw_g, rho_g) = comm.allreduce_sum2(pw_local, pending_rz_local);
                    (pw_g, rho_g)
                } else {
                    let pw_local = if self.reproducible {
                        Self::local_dot_kahan(p, w)
                    } else {
                        Self::local_dot(p, w)
                    };
                    let pw_g = comm.allreduce_sum(pw_local);
                    (pw_g, rho)
                };
                if pw <= 0.0 || !pw.is_finite() {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    return Ok(SolveStats {
                        iterations: k - 1,
                        final_residual: true_res,
                        reason: ConvergedReason::DivergedDtol,
                    });
                }

                let alpha = rho_k / pw;
                for i in 0..n {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * w[i];
                }

                if let Some(pc) = pc {
                    pc.apply(pc_side, r, z)?;
                } else {
                    z.copy_from_slice(r);
                }

                let rz_local_next = if self.reproducible {
                    Self::local_dot_kahan(r, z)
                } else {
                    Self::local_dot(r, z)
                };
                if rz_local_next < 0.0 || !rz_local_next.is_finite() {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    return Ok(SolveStats {
                        iterations: k,
                        final_residual: true_res,
                        reason: ConvergedReason::DivergedDtol,
                    });
                }
                pending_rz_local = rz_local_next;
                has_pending_rz = true;

                res = match self.norm_type {
                    CgNormType::Preconditioned => rho_k.sqrt(),
                    CgNormType::Unpreconditioned => self.nrm2(r, comm),
                    CgNormType::Natural => self.nrm2(z, comm),
                    CgNormType::None => res,
                };

                if let Some(ms) = monitors {
                    for m in ms {
                        m(k, res);
                    }
                }
                if let Some(m) = &self.true_residual_monitor {
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp_true);
                    m(k, true_res);
                }

                let res_check = match self.norm_type {
                    CgNormType::Preconditioned => rho_k.sqrt(),
                    CgNormType::Unpreconditioned => self.nrm2(r, comm),
                    CgNormType::Natural => self.nrm2(z, comm),
                    CgNormType::None => self.nrm2(r, comm),
                };
                let (reason, mut s) = self.conv.check(res_check, bnorm, k);
                if !matches!(reason, ConvergedReason::Continued) {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    s.final_residual = true_res;
                    return Ok(SolveStats {
                        iterations: k,
                        final_residual: s.final_residual,
                        reason: s.reason,
                    });
                }

                z_prev.swap_with_slice(z);
                rho_prev = rho;
                rho = rho_k;
            } else {
                let pw = self.dot(p, w, comm);
                if pw <= 0.0 || !pw.is_finite() {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    return Ok(SolveStats {
                        iterations: k - 1,
                        final_residual: true_res,
                        reason: ConvergedReason::DivergedDtol,
                    });
                }

                let alpha = rho / pw;
                for i in 0..n {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * w[i];
                }

                if let Some(pc) = pc {
                    pc.apply(pc_side, r, z)?;
                } else {
                    z.copy_from_slice(r);
                }

                let rho_new = self.dot(r, z, comm);
                if rho_new < 0.0 || !rho_new.is_finite() {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    return Ok(SolveStats {
                        iterations: k,
                        final_residual: true_res,
                        reason: ConvergedReason::DivergedDtol,
                    });
                }

                res = match self.norm_type {
                    CgNormType::Preconditioned => rho_new.sqrt(),
                    CgNormType::Unpreconditioned => self.nrm2(r, comm),
                    CgNormType::Natural => self.nrm2(z, comm),
                    CgNormType::None => res,
                };

                if let Some(ms) = monitors {
                    for m in ms {
                        m(k, res);
                    }
                }
                if let Some(m) = &self.true_residual_monitor {
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp_true);
                    m(k, true_res);
                }

                let res_check = match self.norm_type {
                    CgNormType::Preconditioned => rho_new.sqrt(),
                    CgNormType::Unpreconditioned => self.nrm2(r, comm),
                    CgNormType::Natural => self.nrm2(z, comm),
                    CgNormType::None => self.nrm2(r, comm),
                };
                let (reason, mut s) = self.conv.check(res_check, bnorm, k);
                if !matches!(reason, ConvergedReason::Continued) {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    s.final_residual = true_res;
                    return Ok(SolveStats {
                        iterations: k,
                        final_residual: s.final_residual,
                        reason: s.reason,
                    });
                }

                let beta = rho_new / rho;
                for i in 0..n {
                    p[i] = z[i] + beta * p[i];
                }
                rho_prev = rho;
                rho = rho_new;
            }
        }

        let mut tmp = vec![0.0; n];
        let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
        Ok(SolveStats {
            iterations: iters,
            final_residual: true_res,
            reason: ConvergedReason::DivergedMaxIts,
        })
    }
}

impl LinearSolver for PcgSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q.len() < 3 {
            work.q.resize(3, Vec::new());
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
        self.solve_impl(a, pc, b, x, pc_side, comm, monitors, work)
    }
}
