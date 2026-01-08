#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::solver::MonitorCallback;
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner, Preconditioner as PreconditionerF64};
use crate::solver::LinearSolver;
use crate::solver::common::{
    dot_result_to_real, recompute_true_residual_norm_s, take_or_resize, ReductCtx,
};
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

#[cfg(feature = "logging")]
use crate::utils::profiling::StageGuard;
pub struct CgnrSolver {
    pub(crate) conv: Convergence,
}

impl CgnrSolver {
    pub fn new(rtol: f64, maxits: usize) -> Self {
        Self {
            conv: Convergence {
                rtol,
                atol: 1e-12,
                dtol: 1e3,
                max_iters: maxits,
            },
        }
    }
}

#[inline]
fn norm_from_dot(result: S) -> R {
    let real = dot_result_to_real(result);
    let zero = R::default();
    let clamped = if real >= zero { real } else { zero };
    clamped.sqrt()
}

struct CgnrWorkspace<'a> {
    r: &'a mut [S],
    z: &'a mut [S],
    p: &'a mut [S],
    ap: &'a mut [S],
    zhat: &'a mut [S],
    tmp_true: &'a mut [S],
    scratch: &'a mut BridgeScratch,
}

impl<'a> CgnrWorkspace<'a> {
    fn acquire(work: &'a mut Workspace, m: usize, n: usize) -> Self {
        take_or_resize(&mut work.tmp1, m);
        take_or_resize(&mut work.tmp2, n);
        if work.bridge_tmp.len() != m {
            work.bridge_tmp.resize(m, S::zero());
        }
        while work.q_s.len() < 2 {
            work.q_s.push(Vec::new());
        }
        for buf in &mut work.q_s[..2] {
            take_or_resize(buf, n);
        }
        if work.z_s.is_empty() {
            work.z_s.push(Vec::new());
        }
        take_or_resize(&mut work.z_s[0], m);
        let (p_slice, rest) = work.q_s.split_at_mut(1);
        let (zhat_slice, _) = rest.split_at_mut(1);
        Self {
            r: &mut work.tmp1[..m],
            z: &mut work.tmp2[..n],
            p: &mut p_slice[0][..n],
            zhat: &mut zhat_slice[0][..n],
            ap: &mut work.z_s[0][..m],
            tmp_true: &mut work.bridge_tmp[..m],
            scratch: &mut work.bridge,
        }
    }
}

impl CgnrSolver {
    #[allow(clippy::too_many_arguments)]
    fn solve_internal<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<R>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        #[cfg(feature = "logging")]
        let _guard = StageGuard::new("CGNR");

        let (m, ncols) = a.dims();
        if b.len() != m {
            return Err(KError::InvalidInput("CGNR: b has wrong length".into()));
        }
        if x.len() != ncols {
            return Err(KError::InvalidInput("CGNR: x has wrong length".into()));
        }
        if !a.supports_t_matvec_s() {
            return Err(KError::InvalidInput(
                "CGNR requires t_matvec; provide an operator that implements A^T·x".into(),
            ));
        }
        if pc_side != PcSide::Left {
            return Err(KError::InvalidInput(
                "CGNR only supports Left preconditioning on the normal equations".into(),
            ));
        }
        let work = work.ok_or_else(|| {
            KError::InvalidInput("CGNR requires a Workspace; use KSP or Workspace::new(n)".into())
        })?;
        let red = ReductCtx::new(comm, Some(&*work));
        if b.is_empty() {
            return Ok(SolveStats::new(
                0,
                R::default(),
                ConvergedReason::ConvergedAtol,
            ));
        }

        let buffers = CgnrWorkspace::acquire(work, m, ncols);
        let r: &mut [S] = &mut *buffers.r;
        let z: &mut [S] = &mut *buffers.z;
        let p: &mut [S] = &mut *buffers.p;
        let ap: &mut [S] = &mut *buffers.ap;
        let zhat: &mut [S] = &mut *buffers.zhat;
        let tmp_true: &mut [S] = &mut *buffers.tmp_true;
        let scratch: &mut BridgeScratch = &mut *buffers.scratch;

        let monitors = monitors.unwrap_or(&[]);
        let mut r_tld = vec![S::zero(); m];

        if x.iter().any(|&xi| xi.abs() > R::default()) {
            a.matvec_s(x, ap, scratch);
            for (ri, (&bi, &api)) in r.iter_mut().zip(b.iter().zip(ap.iter())) {
                *ri = bi - api;
            }
        } else {
            r.copy_from_slice(b);
        }
        r_tld.copy_from_slice(r);

        a.t_matvec_s(r, z, scratch);
        if let Some(pc) = pc {
            pc.apply_s(PcSide::Left, z, zhat, scratch)?;
        } else {
            zhat.copy_from_slice(z);
        }
        p.copy_from_slice(zhat);

        let dot_pairs = [(&z[..], &zhat[..]), (&r[..], &r[..]), (b, b)];
        let mut dot_results = [S::zero(); 3];
        red.dot_many_into(&dot_pairs, &mut dot_results);
        let mut rz = dot_results[0];
        let mut rnow = norm_from_dot(dot_results[1]);
        let bnorm = norm_from_dot(dot_results[2]).max(1e-32);

        for m in monitors {
            let _ = m(0, rnow, 0);
        }

        let (reason0, mut stats0) = self.conv.check(rnow, bnorm, 0);
        if !matches!(reason0, ConvergedReason::Continued) {
            let true_res = recompute_true_residual_norm_s(
                a,
                b,
                x,
                comm,
                red.engine(),
                tmp_true,
                scratch,
            );
            stats0.final_residual = true_res;
            return Ok(stats0);
        }

        let mut iters = 0usize;
        for k in 1..=self.conv.max_iters {
            iters = k;

            a.matvec_s(p, ap, scratch);
            let denom = dot_result_to_real(red.dot(ap, ap));
            if denom <= R::default() || !denom.is_finite() {
                let true_res = recompute_true_residual_norm_s(
                    a,
                    b,
                    x,
                    comm,
                    red.engine(),
                    tmp_true,
                    scratch,
                );
                return Ok(SolveStats::new(
                    k - 1,
                    true_res,
                    ConvergedReason::DivergedDtol,
                ));
            }
            let alpha = rz / S::from_real(denom);
            for i in 0..ncols {
                x[i] += alpha * p[i];
            }
            for i in 0..m {
                r[i] -= alpha * ap[i];
            }

            a.t_matvec_s(r, z, scratch);
            if let Some(pc) = pc {
                pc.apply_s(PcSide::Left, z, zhat, scratch)?;
            } else {
                zhat.copy_from_slice(z);
            }

            let dot_pairs = [(&z[..], &zhat[..]), (&r[..], &r[..])];
            let mut dot_results = [S::zero(); 2];
            red.dot_many_into(&dot_pairs, &mut dot_results);
            let rz_new = dot_results[0];
            rnow = norm_from_dot(dot_results[1]);

            for m in monitors {
                let _ = m(k, rnow, 0);
            }

            let (reason, mut stats) = self.conv.check(rnow, bnorm, k);
            if !matches!(reason, ConvergedReason::Continued) {
                let true_res = recompute_true_residual_norm_s(
                    a,
                    b,
                    x,
                    comm,
                    red.engine(),
                    tmp_true,
                    scratch,
                );
                stats.final_residual = true_res;
                return Ok(stats);
            }

            let beta = rz_new / rz;
            for i in 0..ncols {
                p[i] = zhat[i] + beta * p[i];
            }
            rz = rz_new;
        }

        let true_res = recompute_true_residual_norm_s(
            a,
            b,
            x,
            comm,
            red.engine(),
            tmp_true,
            scratch,
        );
        Ok(SolveStats::new(
            iters,
            true_res,
            ConvergedReason::DivergedMaxIts,
        ))
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
        monitors: Option<&[Box<MonitorCallback<R>>]>,
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
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
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
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError>
    where
        A: LinOpF64 + LinOp<S = f64> + Send + Sync + ?Sized,
    {
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}

impl LinearSolver for CgnrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, work: &mut Workspace) {
        if work.q_s.len() < 2 {
            work.q_s.resize(2, Vec::new());
        }
        if work.z_s.is_empty() {
            work.z_s.resize(1, Vec::new());
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
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let pc = pc.map(|m| m as &dyn PreconditionerF64);
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}
