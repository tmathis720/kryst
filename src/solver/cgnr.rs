use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::recompute_true_residual_norm;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;

pub struct CgnrSolver {
    pub(crate) conv: Convergence<f64>,
}

impl CgnrSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self { conv: Convergence { rtol, atol: 1e-12, dtol: 1e3, max_iters: maxits } }
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

    fn acquire<'a>(
        n: usize,
        work: &'a mut Workspace,
    ) -> (
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [f64],
    ) {
        // tmp1=r (len>=m), tmp2=z (len>=n)
        Self::take_or_resize(&mut work.tmp1, n);
        Self::take_or_resize(&mut work.tmp2, n);
        // q: p, Ap, AtAp (optional), zhat
        while work.q.len() < 4 { work.q.push(Vec::new()); }
        for k in 0..4 { Self::take_or_resize(&mut work.q[k], n); }
        let (p_slice, rest) = work.q.split_at_mut(1);
        let (ap_slice, rest) = rest.split_at_mut(1);
        let (atap_slice, rest) = rest.split_at_mut(1);
        let (zhat_slice, _) = rest.split_at_mut(1);
        let r = &mut work.tmp1[..];
        let z = &mut work.tmp2[..];
        let p = &mut p_slice[0][..];
        let ap = &mut ap_slice[0][..];
        let atap = &mut atap_slice[0][..];
        let zhat = &mut zhat_slice[0][..];
        (r, z, p, ap, atap, zhat)
    }
}

impl LinearSolver for CgnrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q.len() < 4 {
            work.q.resize(4, Vec::new());
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
        let pc: Option<&dyn Preconditioner> = pc.as_deref();
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CGNR");

        let (m, ncols) = a.dims();
        if b.len() != m {
            return Err(KError::InvalidInput("CGNR: b has wrong length".into()));
        }
        if x.len() != ncols {
            return Err(KError::InvalidInput("CGNR: x has wrong length".into()));
        }

        if !a.supports_transpose() {
            return Err(KError::InvalidInput(
                "CGNR requires t_matvec; provide an operator that implements A^T·x".into(),
            ));
        }

        // Enforce Left preconditioning semantics for CGNR
        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "CGNR only supports Left preconditioning on the normal equations".into(),
            ));
        }

        // Require a Workspace to avoid heap leaks and repeated allocs.
        let work = work.ok_or_else(|| {
            KError::InvalidInput("CGNR requires a Workspace; use KSP or Workspace::new(n)".into())
        })?;
        // Zero-length fast path
        if b.is_empty() {
            return Ok(SolveStats {
                iterations: 0,
                final_residual: 0.0,
                reason: ConvergedReason::ConvergedAtol,
            });
        }

        let (r_store, z_store, p_store, ap_store, _atap_store, zhat_store) = Self::acquire(ncols.max(m), work);
        let (r, z, p, ap, _atap, zhat) = (
            &mut r_store[..m],
            &mut z_store[..ncols],
            &mut p_store[..ncols],
            &mut ap_store[..m],
            &mut _atap_store[..ncols],
            &mut zhat_store[..ncols],
        );

        if x.iter().any(|&xi| xi != 0.0) {
            a.matvec(x, ap);
            for i in 0..m {
                r[i] = b[i] - ap[i];
            }
        } else {
            r.copy_from_slice(b);
        }

        a.t_matvec(r, z);

        // zhat = M^{-1} z (or copy of z if no PC)
        if let Some(pc) = pc {
            pc.apply(PcSide::Left, z, zhat)?;
        } else {
            zhat.copy_from_slice(z);
        }

        p.copy_from_slice(zhat);

        let mut rz = Self::dot(&z[..], zhat, comm);
        let mut rnow = Self::nrm2(r, comm);
        let bnorm = Self::nrm2(b, comm).max(1e-32);

        if let Some(ms) = monitors {
            for m in ms {
                m(0, rnow);
            }
        }

        let (reason0, mut s0) = self.conv.check(rnow, bnorm, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            // Recompute true residual for consistency
            let mut tmp = vec![0.0; m];
            let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
            s0.final_residual = true_res;
            return Ok(SolveStats { iterations: 0, final_residual: s0.final_residual, reason: s0.reason });
        }

        let mut iters = 0usize;
        for k in 1..=self.conv.max_iters {
            iters = k;

            a.matvec(p, ap);

            let denom = Self::dot(ap, ap, comm);
            if denom <= 0.0 || !denom.is_finite() {
                // Gracefully declare divergence on breakdown
                let mut tmp = vec![0.0; m];
                let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                return Ok(SolveStats { iterations: k - 1, final_residual: true_res, reason: ConvergedReason::DivergedDtol });
            }
            let alpha = rz / denom;

            for i in 0..ncols {
                x[i] += alpha * p[i];
            }
            for i in 0..m {
                r[i] -= alpha * ap[i];
            }

            a.t_matvec(r, z);
            if let Some(pc) = pc {
                pc.apply(PcSide::Left, z, zhat)?;
            } else {
                zhat.copy_from_slice(z);
            }

            let rz_new = Self::dot(&z[..], zhat, comm);
            rnow = Self::nrm2(r, comm);

            if let Some(ms) = monitors {
                for m in ms {
                    m(k, rnow);
                }
            }

            let (reason, mut s) = self.conv.check(rnow, bnorm, k);
            if !matches!(reason, ConvergedReason::Continued) {
                // Report true residual (matches rnow but recompute for consistency)
                let mut tmp = vec![0.0; m];
                let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
                s.final_residual = true_res;
                return Ok(SolveStats { iterations: k, final_residual: s.final_residual, reason: s.reason });
            }

            let beta = rz_new / rz;
            for i in 0..ncols {
                p[i] = zhat[i] + beta * p[i];
            }
            rz = rz_new;
        }

        let mut tmp = vec![0.0; m];
        let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
        Ok(SolveStats { iterations: iters, final_residual: true_res, reason: ConvergedReason::DivergedMaxIts })
    }
}
