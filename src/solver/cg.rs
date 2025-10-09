//! # CG side semantics
//!
//! CG requires **Left preconditioning** with SPD `M`.
//! If [`PcSide`] is not `Left`, the solver returns `InvalidInput`.
//! Residual norm is the preconditioned norm `||M^{-1} r||`; final stats include true `||r||`.

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
use crate::parallel::{UniverseComm, global_dot_conj, global_dot_conj_many_into, global_nrm2};
use crate::preconditioner::{PcSide, Preconditioner, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::common::dot_result_to_real;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use smallvec::SmallVec;
use std::any::Any;

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::trace;

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
                "CG requires left preconditioning with SPD M; use MINRES or GMRES otherwise".into(),
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
            return Ok(SolveStats::new(0, 0.0, ConvergedReason::ConvergedAtol));
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

        let guess_nonzero =
            self.initial_guess_nonzero || x.iter().any(|&xi| xi.abs() > R::default());
        if guess_nonzero {
            a.matvec_s(x, &mut tmp[..], scratch);
            for i in 0..nrows {
                r[i] = b[i] - tmp[i];
            }
        } else {
            r.copy_from_slice(b);
        }

        if let Some(pc) = pc {
            pc.apply_s(PcSide::Left, r, &mut z[..], scratch)?;
        } else {
            z.copy_from_slice(r);
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

            let mut result_idx = 0;
            let rho = dot_result_to_real(dot_results[result_idx]);
            result_idx += 1;
            let rsq = if want_unpre {
                let value = dot_results[result_idx];
                result_idx += 1;
                Some(dot_result_to_real(value))
            } else {
                None
            };
            let znorm = if want_natural {
                let value = dot_results[result_idx];
                Some(dot_result_to_real(value))
            } else {
                None
            };
            (rho, rsq, znorm)
        };
        let mut rho_prev = rho;
        if rho <= 0.0 || !rho.is_finite() {
            return Err(KError::IndefinitePreconditioner);
        }
        let mut xnorm = if self.trust_region.is_some() {
            global_nrm2(comm, x)
        } else {
            R::zero()
        };

        let res0_reported = match self.norm_type {
            CgNormType::Preconditioned => rho.abs().sqrt(),
            CgNormType::Unpreconditioned => rsq.unwrap().abs().sqrt(),
            CgNormType::Natural => znorm.unwrap().abs().sqrt(),
            CgNormType::None => 0.0,
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

        p.copy_from_slice(z);

        let mut stats = SolveStats::new(0, res0_reported, ConvergedReason::Continued);

        let (reason0, s0) = self.conv.check(res0_reported, res0_reported, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let mut s = s0;
            s.final_residual = global_nrm2(comm, r);
            return Ok(s);
        }

        for k in 1..=self.conv.max_iters {
            if k > 1 {
                let beta = rho / rho_prev;
                let beta_s = S::from_real(beta);
                for i in 0..nrows {
                    p[i] = z[i] + beta_s * p[i];
                }
            }

            a.matvec_s(p, &mut ap[..], scratch);

            let p_ap = dot_result_to_real(global_dot_conj(comm, p, ap));
            if p_ap <= 0.0 || !p_ap.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }

            let alpha = rho / p_ap;
            let alpha_s = S::from_real(alpha);

            if let Some(rmax) = self.trust_region {
                let pnorm = global_nrm2(comm, p);
                if xnorm + alpha.abs() * pnorm > rmax {
                    let step = (rmax - xnorm) / (pnorm + 1e-300);
                    let step_s = S::from_real(step);
                    for i in 0..nrows {
                        x[i] += step_s * p[i];
                        r[i] -= step_s * ap[i];
                    }
                    stats.iterations = k;
                    stats.reason = ConvergedReason::ConvergedTrustRegion;
                    stats.final_residual = global_nrm2(comm, r);
                    return Ok(stats);
                }
            }

            for i in 0..nrows {
                x[i] += alpha_s * p[i];
            }
            for i in 0..nrows {
                r[i] -= alpha_s * ap[i];
            }
            if self.trust_region.is_some() {
                xnorm = global_nrm2(comm, x);
            }

            if let Some(pc) = pc {
                pc.apply_s(PcSide::Left, r, &mut z[..], scratch)?;
            } else {
                z.copy_from_slice(r);
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

                let mut result_idx = 0;
                let rho_new = dot_result_to_real(dot_results[result_idx]);
                result_idx += 1;
                if rho_new <= 0.0 || !rho_new.is_finite() {
                    return Err(KError::IndefinitePreconditioner);
                }

                let rsq_new = if want_unpre {
                    let value = dot_results[result_idx];
                    result_idx += 1;
                    Some(dot_result_to_real(value))
                } else {
                    None
                };
                let znorm_new = if want_natural {
                    let value = dot_results[result_idx];
                    Some(dot_result_to_real(value))
                } else {
                    None
                };
                (rho_new, rsq_new, znorm_new)
            };

            let res_reported = match self.norm_type {
                CgNormType::Preconditioned => rho_new.abs().sqrt(),
                CgNormType::Unpreconditioned => rsq_new.unwrap().abs().sqrt(),
                CgNormType::Natural => znorm_new.unwrap().abs().sqrt(),
                CgNormType::None => 0.0,
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
