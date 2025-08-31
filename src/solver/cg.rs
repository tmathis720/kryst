//! # CG side semantics
//!
//! CG requires **Left preconditioning** with SPD `M`.
//! If [`PcSide`] is not `Left`, the solver returns `InvalidInput`.
//! Residual norm is the preconditioned norm `||M^{-1} r||`; final stats include true `||r||`.

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::recompute_true_residual_norm;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
#[cfg(feature = "logging")]
use log::trace;

#[derive(Debug, Clone, Copy)]
pub enum CgNormType {
    Preconditioned,
    Unpreconditioned,
    Natural,
    None,
}

pub struct CgSolver {
    pub(crate) conv: Convergence<f64>,
    norm_type: CgNormType,
    single_reduction: bool,
    trust_region: Option<f64>,
}

impl CgSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            conv: Convergence { rtol, atol: 1e-12, dtol: 1e3, max_iters: maxits },
            // Default monitors use preconditioned norm per policy
            norm_type: CgNormType::Preconditioned,
            single_reduction: false,
            trust_region: None,
        }
    }

    pub fn with_norm(mut self, n: CgNormType) -> Self {
        self.norm_type = n;
        self
    }
    pub fn with_single_reduction(mut self, f: bool) -> Self {
        self.single_reduction = f;
        self
    }
    pub fn with_trust_region(mut self, r: f64) -> Self {
        self.trust_region = Some(r);
        self
    }

    pub fn set_norm(&mut self, n: CgNormType) {
        self.norm_type = n;
    }
    pub fn set_single_reduction(&mut self, f: bool) {
        self.single_reduction = f;
    }
    pub fn set_trust_region(&mut self, r: f64) {
        self.trust_region = Some(r);
    }

    #[inline]
    fn dot(u: &[f64], v: &[f64], comm: &UniverseComm) -> f64 {
        comm.dot(u, v)
    }
    #[inline]
    fn nrm2(u: &[f64], comm: &UniverseComm) -> f64 {
        Self::dot(u, u, comm).sqrt()
    }

    fn take_or_resize(buf: &mut Vec<f64>, n: usize) {
        if buf.len() != n {
            buf.resize(n, 0.0);
        }
    }

    fn acquire<'a>(
        n: usize,
        work: Option<&'a mut Workspace>,
    ) -> (
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
    ) {
        if let Some(wk) = work {
            while wk.q.len() < 4 {
                wk.q.push(vec![0.0; n]);
            }
            for v in &mut wk.q[0..4] {
                Self::take_or_resize(v, n);
            }
            Self::take_or_resize(&mut wk.tmp1, n);
            let (r_slice, rest) = wk.q.split_at_mut(1);
            let (z_slice, rest) = rest.split_at_mut(1);
            let (p_slice, rest) = rest.split_at_mut(1);
            let (ap_slice, _) = rest.split_at_mut(1);
            let r = &mut r_slice[0][..];
            let z = &mut z_slice[0][..];
            let p = &mut p_slice[0][..];
            let ap = &mut ap_slice[0][..];
            let tmp = &mut wk.tmp1[..];
            (r, z, p, ap, tmp)
        } else {
            let mk = |n| -> &'static mut [f64] { Box::leak(vec![0.0; n].into_boxed_slice()) };
            (mk(n), mk(n), mk(n), mk(n), mk(n))
        }
    }
}

impl LinearSolver for CgSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q.len() < 4 {
            work.q.resize(4, Vec::new());
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
        let pc: Option<&dyn Preconditioner> = pc.as_deref();
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CG");

        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "CG/MINRES require Left preconditioning (SPD M)".into(),
            ));
        }

        let n = b.len();
        if x.len() != n {
            return Err(KError::InvalidInput("dimension mismatch x,b".into()));
        }

        let (r, z, p, ap, tmp) = Self::acquire(n, work);

        if x.iter().any(|&xi| xi != 0.0) {
            a.matvec(x, tmp);
            for i in 0..n {
                r[i] = b[i] - tmp[i];
            }
        } else {
            r.copy_from_slice(b);
        }

        if let Some(m) = pc {
            m.apply(pc_side, r, z)?;
        } else {
            z.copy_from_slice(r);
        }

        let mut rho = Self::dot(r, z, comm);
        if rho <= 0.0 || !rho.is_finite() {
            return Err(KError::IndefinitePreconditioner);
        }
        let rsq = Self::dot(r, r, comm);
        let mut xnorm = Self::nrm2(x, comm);

        // Reported residual for CG (Left) defaults to preconditioned norm sqrt(r^T z)
        let res0_reported = match self.norm_type {
            CgNormType::Preconditioned | CgNormType::Natural => rho.abs().sqrt(),
            CgNormType::Unpreconditioned => rsq.sqrt(),
            CgNormType::None => 0.0,
        };

        if let Some(ms) = monitors {
            for m in ms {
                m(0, res0_reported);
            }
        }
        #[cfg(feature = "logging")]
        trace!("CG initial residual: {:.3e}", res0_reported);

        p.copy_from_slice(z);

        let mut stats = SolveStats { iterations: 0, final_residual: res0_reported, reason: ConvergedReason::Continued };

        // Convergence check at iteration 0 (baseline = res0_reported)
        let (reason0, s0) = self.conv.check(res0_reported, res0_reported, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            // On early exit, recompute true residual for final reporting
            let mut tmp = vec![0.0; n];
            let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
            let mut s = s0;
            s.final_residual = true_res;
            return Ok(s);
        }

        for k in 1..=self.conv.max_iters {
            a.matvec(p, ap);

            let p_ap = Self::dot(p, ap, comm);
            if p_ap <= 0.0 || !p_ap.is_finite() {
                return Err(KError::IndefiniteMatrix);
            }

            let alpha = rho / p_ap;

            if let Some(rmax) = self.trust_region {
                let pnorm = Self::nrm2(p, comm);
                if xnorm + alpha.abs() * pnorm > rmax {
                    let step = (rmax - xnorm) / (pnorm + 1e-300);
                    for i in 0..n {
                        x[i] += step * p[i];
                    }
                    stats.iterations = k;
                    stats.final_residual = Self::nrm2(r, comm);
                    stats.reason = ConvergedReason::DivergedMaxIts;
                    return Ok(stats);
                }
            }

            for i in 0..n {
                x[i] += alpha * p[i];
            }
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }
            xnorm = Self::nrm2(x, comm);

            if let Some(m) = pc {
                m.apply(pc_side, r, z)?;
            } else {
                z.copy_from_slice(r);
            }

            let rho_new = Self::dot(r, z, comm);
            if rho_new <= 0.0 || !rho_new.is_finite() {
                return Err(KError::IndefinitePreconditioner);
            }
            let rsq_new = Self::dot(r, r, comm);

            let res_reported = match self.norm_type {
                CgNormType::Preconditioned | CgNormType::Natural => rho_new.abs().sqrt(),
                CgNormType::Unpreconditioned => rsq_new.sqrt(),
                CgNormType::None => 0.0,
            };

            if let Some(ms) = monitors {
                for m in ms {
                    m(k, res_reported);
                }
            }

            // Use standardized convergence policy against res0_reported baseline
            let (reason, mut s) = self.conv.check(res_reported, res0_reported, k);
            if !matches!(reason, ConvergedReason::Continued) {
                // Exit: recompute true residual and override final_residual
                let mut tmp = vec![0.0; n];
                let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                s.final_residual = true_res;
                return Ok(s);
            }

            let beta = rho_new / rho;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            rho = rho_new;
            stats.iterations = k;
            stats.final_residual = res_reported;
        }

        // Max-its reached: recompute true residual and return divergence
        let mut tmp = vec![0.0; n];
        let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
        Ok(SolveStats { iterations: self.conv.max_iters, final_residual: true_res, reason: ConvergedReason::DivergedMaxIts })
    }
}
