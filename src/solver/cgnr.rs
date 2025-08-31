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
    fn dot(x: &[f64], y: &[f64], _comm: &UniverseComm) -> f64 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
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
    ) {
        Self::take_or_resize(&mut work.tmp1, n);
        Self::take_or_resize(&mut work.tmp2, n);
        while work.q.len() < 3 { work.q.push(Vec::new()); }
        for k in 0..3 { Self::take_or_resize(&mut work.q[k], n); }
        let (p_slice, rest) = work.q.split_at_mut(1);
        let (ap_slice, rest) = rest.split_at_mut(1);
        let (atap_slice, _) = rest.split_at_mut(1);
        let r = &mut work.tmp1[..];
        let z = &mut work.tmp2[..];
        let p = &mut p_slice[0][..];
        let ap = &mut ap_slice[0][..];
        let atap = &mut atap_slice[0][..];
        (r, z, p, ap, atap)
    }
}

impl LinearSolver for CgnrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
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

        let (r, z, p, ap, _atap) = Self::acquire(ncols.max(m), work);
        let (r, z, p, ap, _atap) = (
            &mut r[..m],
            &mut z[..ncols],
            &mut p[..ncols],
            &mut ap[..m],
            &mut _atap[..ncols],
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

        let mut zhat_buf: Vec<f64> = if pc.is_some() {
            vec![0.0; ncols]
        } else {
            Vec::new()
        };
        if let Some(pc) = pc {
            pc.apply(pc_side, z, &mut zhat_buf)?;
        }
        let zhat_slice: &[f64] = if pc.is_some() { &zhat_buf[..] } else { &z[..] };

        p.copy_from_slice(zhat_slice);

        let mut rz = Self::dot(&z[..], zhat_slice, comm);
        let mut rnow = Self::nrm2(r, comm);
        let res0_reported = rnow;

        if let Some(ms) = monitors {
            for m in ms {
                m(0, rnow);
            }
        }

        let (reason0, s0) = self.conv.check(rnow, res0_reported, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            return Ok(SolveStats { iterations: 0, final_residual: rnow, reason: s0.reason });
        }

        let mut iters = 0usize;
        for k in 1..=self.conv.max_iters {
            iters = k;

            a.matvec(p, ap);

            let denom = Self::dot(ap, ap, comm);
            if denom <= 0.0 || !denom.is_finite() {
                return Err(KError::IndefiniteMatrix);
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
                pc.apply(pc_side, z, &mut zhat_buf)?;
            }
            let zhat_slice: &[f64] = if pc.is_some() { &zhat_buf[..] } else { &z[..] };

            let rz_new = Self::dot(&z[..], zhat_slice, comm);
            rnow = Self::nrm2(r, comm);

            if let Some(ms) = monitors {
                for m in ms {
                    m(k, rnow);
                }
            }

            let (reason, s) = self.conv.check(rnow, res0_reported, k);
            if !matches!(reason, ConvergedReason::Continued) {
                return Ok(SolveStats { iterations: k, final_residual: rnow, reason: s.reason });
            }

            let beta = rz_new / rz;
            for i in 0..ncols {
                p[i] = zhat_slice[i] + beta * p[i];
            }
            rz = rz_new;
        }

        let mut tmp = vec![0.0; ncols];
        let true_res = recompute_true_residual_norm(a, b, x, comm, &mut tmp);
        Ok(SolveStats { iterations: iters, final_residual: true_res, reason: ConvergedReason::DivergedMaxIts })
    }
}
