use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::reduction::{CommDeterministic, DotEngine, ReductionOptions, ReproMode};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
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
    reduction: ReductionOptions,
    true_residual_monitor: Option<Box<dyn Fn(usize, f64) + Send + Sync>>,
    /// Whether the initial guess in `x` should be treated as nonzero.
    ///
    /// PETSc zeroes the initial guess by default unless told otherwise via
    /// `KSPSetInitialGuessNonzero`. We follow the same policy; when this flag is
    /// `false` and the provided `x` is exactly the zero vector, the solver
    /// skips the initial matvec and assumes a zero guess.
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
              reduction: ReductionOptions::default(),
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

    /// Enable a more reproducible (but slightly slower) local dot product using
    /// Kahan summation. When combined with a deterministic MPI reduction this
    /// yields bitwise-identical results across runs.
      pub fn with_reproducible_dot(mut self, f: bool) -> Self {
          self.reduction.mode = if f { ReproMode::Deterministic } else { ReproMode::Fast };
          self
      }

    /// Install a monitor that receives the true residual norm `||b - A x||₂`
    /// at each iteration. This uses the already available residual and is
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
          self.reduction.mode = if f { ReproMode::Deterministic } else { ReproMode::Fast };
      }

    /// Set or clear the true residual monitor after construction.
    pub fn set_true_residual_monitor(&mut self, m: Option<Box<dyn Fn(usize, f64) + Send + Sync>>) {
        self.true_residual_monitor = m;
    }

    #[inline]
      fn dot<C: Comm + CommDeterministic>(&self, u: &[f64], v: &[f64], comm: &C) -> f64 {
          let engine = DotEngine {
              opts: self.reduction,
          };
          engine.dot(u, v, comm)
      }


    #[inline]
      fn nrm2<C: Comm + CommDeterministic>(&self, u: &[f64], comm: &C) -> f64 {
          self.dot(u, u, comm).sqrt()
      }

    fn take_or_resize(buf: &mut Vec<f64>, n: usize) {
        if buf.len() != n {
            buf.resize(n, 0.0);
        }
    }

    #[allow(clippy::too_many_arguments)]
      pub fn solve_with_comm<C: Comm + CommDeterministic>(
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
      fn solve_impl<C: Comm + CommDeterministic>(
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
        let mut ap_store = Vec::new();

        let (r, z, p, ap): (&mut [f64], &mut [f64], &mut [f64], &mut [f64]);

        if let Some(wk) = work {
            Self::take_or_resize(&mut wk.tmp1, n);
            Self::take_or_resize(&mut wk.tmp2, n);
            while wk.q.len() < 2 {
                wk.q.push(vec![0.0; n]);
            }
            for v in &mut wk.q[0..2] {
                Self::take_or_resize(v, n);
            }
            r = wk.tmp1.as_mut_slice();
            z = wk.tmp2.as_mut_slice();
            let (pbuf, rest) = wk.q.split_at_mut(1);
            p = pbuf[0].as_mut_slice();
            ap = rest[0].as_mut_slice();
        } else {
            r_store.resize(n, 0.0);
            z_store.resize(n, 0.0);
            p_store.resize(n, 0.0);
            ap_store.resize(n, 0.0);
            r = r_store.as_mut_slice();
            z = z_store.as_mut_slice();
            p = p_store.as_mut_slice();
            ap = ap_store.as_mut_slice();
        }

        // r = b - A x
        let zero_guess = !self.initial_guess_nonzero && x.iter().all(|&xi| xi == 0.0);
        if zero_guess {
            r.copy_from_slice(b);
        } else {
            a.matvec(x, ap);
            for i in 0..n {
                r[i] = b[i] - ap[i];
            }
        }

        // z = M^{-1} r
        if let Some(pc) = pc {
            pc.apply(pc_side, r, z)?;
        } else {
            z.copy_from_slice(r);
        }

        let mut rho = self.dot(r, z, comm);
        let mut rho_prev = rho;

        let res0 = match self.norm_type {
            CgNormType::Preconditioned => rho.sqrt(),
            CgNormType::Unpreconditioned => self.nrm2(r, comm),
            CgNormType::Natural => self.nrm2(z, comm),
            CgNormType::None => self.nrm2(r, comm),
        };
        let mut res = res0;

        if let Some(ms) = monitors {
            for m in ms {
                m(0, res);
            }
        }
        if let Some(m) = &self.true_residual_monitor {
            let true_res = self.nrm2(r, comm);
            m(0, true_res);
        }

        p.copy_from_slice(z);

        // Convergence check at k=0
        let (reason0, s0) = self.conv.check(res0, res0, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let true_res = self.nrm2(r, comm);
            return Ok(SolveStats {
                iterations: 0,
                final_residual: true_res,
                reason: s0.reason,
            });
        }
        let mut iters = 0usize;
        for k in 1..=self.conv.max_iters {
            iters = k;

            if k > 1 {
                let beta = rho / rho_prev;
                for i in 0..n {
                    p[i] = z[i] + beta * p[i];
                }
            }

            a.matvec(p, ap);
            let p_ap = self.dot(p, ap, comm);
            let eps = 1e-300;
            if !p_ap.is_finite() || p_ap < -eps {
                return Err(KError::IndefiniteMatrix);
            }
            if p_ap <= eps {
                return Ok(SolveStats {
                    iterations: k - 1,
                    final_residual: self.nrm2(r, comm),
                    reason: ConvergedReason::ConvergedHappyBreakdown,
                });
            }

            let alpha = rho / p_ap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            if let Some(pc) = pc {
                pc.apply(pc_side, r, z)?;
            } else {
                z.copy_from_slice(r);
            }

            let mut rho_new = self.dot(r, z, comm);
            if !rho_new.is_finite() || rho_new < -eps {
                return Err(KError::IndefinitePreconditioner);
            }
            if rho_new < eps {
                rho_new = 0.0;
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
                m(k, self.nrm2(r, comm));
            }

            let res_check = match self.norm_type {
                CgNormType::Preconditioned => rho_new.sqrt(),
                CgNormType::Unpreconditioned => self.nrm2(r, comm),
                CgNormType::Natural => self.nrm2(z, comm),
                CgNormType::None => self.nrm2(r, comm),
            };
            let (reason, mut s) = self.conv.check(res_check, res0, k);
            if !matches!(reason, ConvergedReason::Continued) {
                s.final_residual = self.nrm2(r, comm);
                return Ok(SolveStats {
                    iterations: k,
                    final_residual: s.final_residual,
                    reason: s.reason,
                });
            }

            rho_prev = rho;
            rho = rho_new;
        }

        Ok(SolveStats {
            iterations: iters,
            final_residual: self.nrm2(r, comm),
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
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        self.solve_impl(a, pc, b, x, pc_side, comm, monitors, work)
    }
}
