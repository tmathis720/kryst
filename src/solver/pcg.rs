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
    /// Monitor the natural norm induced by the operator (also `sqrt(rᵀz)` for
    /// SPD systems). Included for compatibility with PETSc semantics.
    Natural,
    /// Do not compute or report a residual norm
    None,
}

pub struct PcgSolver {
    pub(crate) conv: Convergence<f64>,
    norm_type: CgNormType,
    single_reduction: bool,
}

impl PcgSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self { conv: Convergence { rtol, atol: 1e-50, dtol: 1e5, max_iters: maxits }, norm_type: CgNormType::Preconditioned, single_reduction: false }
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

    #[inline]
    fn dot<C: Comm>(u: &[f64], v: &[f64], comm: &C) -> f64 {
        comm.dot(u, v)
    }

    #[inline]
    fn local_dot(u: &[f64], v: &[f64]) -> f64 {
        u.iter().zip(v).map(|(a, b)| a * b).sum::<f64>()
    }

    #[inline]
    fn nrm2<C: Comm>(u: &[f64], comm: &C) -> f64 {
        Self::dot(u, u, comm).sqrt()
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
                "CG/MINRES require Left preconditioning (SPD M)".into(),
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

        // r = b - A x
        let zero_guess = x.iter().all(|&xi| xi == 0.0);
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

        let bnorm = Self::nrm2(b, comm).max(1e-32);
        let mut rho = Self::dot(r, z, comm);
        let mut rho_prev = rho;
        let mut pending_rz_local = 0.0f64;
        let mut has_pending_rz = false;

        let mut res = match self.norm_type {
            CgNormType::Preconditioned => rho.sqrt(),
            CgNormType::Unpreconditioned => Self::nrm2(r, comm),
            CgNormType::Natural => rho,
            CgNormType::None => 0.0,
        };

        if let Some(ms) = monitors {
            for m in ms {
                m(0, res);
            }
        }

        p.copy_from_slice(z);

        // Convergence check at k=0
        let res_check0 = match self.norm_type {
            CgNormType::Preconditioned | CgNormType::Natural => rho.sqrt(),
            CgNormType::Unpreconditioned | CgNormType::None => Self::nrm2(r, comm),
        };
        let (reason0, s0) = self.conv.check(res_check0, bnorm, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let mut tmp = vec![0.0; n];
            let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
            return Ok(SolveStats { iterations: 0, final_residual: true_res, reason: s0.reason });
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
                    let pw_local = Self::local_dot(p, w);
                    let (pw_g, rho_g) = comm.allreduce_sum2(pw_local, pending_rz_local);
                    (pw_g, rho_g)
                } else {
                    let pw_local = Self::local_dot(p, w);
                    let pw_g = comm.allreduce_sum(pw_local);
                    (pw_g, rho)
                };
                if pw <= 0.0 || !pw.is_finite() {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    return Ok(SolveStats { iterations: k - 1, final_residual: true_res, reason: ConvergedReason::DivergedDtol });
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

                let rz_local_next = Self::local_dot(r, z);
                if rz_local_next < 0.0 || !rz_local_next.is_finite() {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    return Ok(SolveStats { iterations: k, final_residual: true_res, reason: ConvergedReason::DivergedDtol });
                }
                pending_rz_local = rz_local_next;
                has_pending_rz = true;

                res = match self.norm_type {
                    CgNormType::Preconditioned => rho_k.sqrt(),
                    CgNormType::Unpreconditioned => Self::nrm2(r, comm),
                    CgNormType::Natural => rho_k,
                    CgNormType::None => res,
                };

                if let Some(ms) = monitors {
                    for m in ms {
                        m(k, res);
                    }
                }

                let res_check = match self.norm_type {
                    CgNormType::Preconditioned | CgNormType::Natural => rho_k.sqrt(),
                    CgNormType::Unpreconditioned | CgNormType::None => Self::nrm2(r, comm),
                };
                let (reason, mut s) = self.conv.check(res_check, bnorm, k);
                if !matches!(reason, ConvergedReason::Continued) {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    s.final_residual = true_res;
                    return Ok(SolveStats { iterations: k, final_residual: s.final_residual, reason: s.reason });
                }

                z_prev.swap_with_slice(z);
                rho_prev = rho;
                rho = rho_k;
            } else {
                let pw = Self::dot(p, w, comm);
                if pw <= 0.0 || !pw.is_finite() {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    return Ok(SolveStats { iterations: k - 1, final_residual: true_res, reason: ConvergedReason::DivergedDtol });
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

                let rho_new = Self::dot(r, z, comm);
                if rho_new < 0.0 || !rho_new.is_finite() {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    return Ok(SolveStats { iterations: k, final_residual: true_res, reason: ConvergedReason::DivergedDtol });
                }

                res = match self.norm_type {
                    CgNormType::Preconditioned => rho_new.sqrt(),
                    CgNormType::Unpreconditioned => Self::nrm2(r, comm),
                    CgNormType::Natural => rho_new,
                    CgNormType::None => res,
                };

                if let Some(ms) = monitors {
                    for m in ms {
                        m(k, res);
                    }
                }

                let res_check = match self.norm_type {
                    CgNormType::Preconditioned | CgNormType::Natural => rho_new.sqrt(),
                    CgNormType::Unpreconditioned | CgNormType::None => Self::nrm2(r, comm),
                };
                let (reason, mut s) = self.conv.check(res_check, bnorm, k);
                if !matches!(reason, ConvergedReason::Continued) {
                    let mut tmp = vec![0.0; n];
                    let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                    s.final_residual = true_res;
                    return Ok(SolveStats { iterations: k, final_residual: s.final_residual, reason: s.reason });
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
        Ok(SolveStats { iterations: iters, final_residual: true_res, reason: ConvergedReason::DivergedMaxIts })
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
