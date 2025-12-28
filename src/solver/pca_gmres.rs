//! PCA-GMRES (baseline) over &dyn LinOp<f64> with left/right/no preconditioning,
//! using disjoint slabs for V and Z, with semantics enforced by `pc_mode`.

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
#[cfg(feature = "complex")]
use crate::algebra::scalar::copy_scalar_to_real_in;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::givens::{apply_new_givens_and_update_g, apply_prev_givens_to_col};
use crate::solver::common::ReductCtx;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use smallvec::SmallVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcaPcMode {
    None,
    Left,
    Right,
}

pub struct PcaGmresSolver {
    pub restart: usize,
    pub pipeline_depth: usize, // reserved hook; baseline uses 1
    pub block_size: usize,     // reserved hook; baseline uses 1
    pub conv: Convergence,
    pub pc_mode: PcaPcMode,
    /// Choose MGS for better robustness; switch to CGS when s>1 (future)
    pub modified_gs: bool,
    /// Happy breakdown tolerance
    pub haptol: f64,
}

impl PcaGmresSolver {
    pub fn new(
        restart: usize,
        pipeline_depth: usize,
        block_size: usize,
        rtol: f64,
        maxits: usize,
    ) -> Self {
        Self {
            restart: restart.max(1),
            pipeline_depth,
            block_size,
            conv: Convergence {
                rtol,
                atol: 1e-12,
                dtol: 1e3,
                max_iters: maxits,
            },
            pc_mode: PcaPcMode::Left, // default matches historical behavior
            modified_gs: true,
            haptol: 1e-12,
        }
    }

    /// Workspace policy:
    ///   - V basis in `w.q_s[0..=m]`
    ///   - Z basis in `w.z_s[0..m-1]` (only used for Right PC)
    fn ensure_workspace(&self, w: &mut Workspace, n: usize) {
        let m = self.restart;

        // V basis
        if w.q_s.len() < m + 1 {
            w.q_s.resize(m + 1, Vec::new());
        }
        for v in &mut w.q_s[..m + 1] {
            if v.len() != n {
                v.resize(n, S::zero());
            }
        }

        // Z basis (right preconditioning only, but always size for simplicity)
        if w.z_s.len() < m {
            w.z_s.resize(m, Vec::new());
        }
        for z in &mut w.z_s[..m] {
            if z.len() != n {
                z.resize(n, S::zero());
            }
        }

        // Hessenberg
        if w.h_s.len() < m {
            w.h_s.resize(m, Vec::new());
        }
        for col in &mut w.h_s[..m] {
            if col.len() != m + 1 {
                col.resize(m + 1, S::zero());
            }
        }

        // Scalars and temporaries
        if w.tmp1.len() != n {
            w.tmp1.resize(n, S::zero());
        }
        if w.tmp2.len() != n {
            w.tmp2.resize(n, S::zero());
        }
        if w.cs.len() < m {
            w.cs.resize(m, R::default());
        }
        if w.sn.len() < m {
            w.sn.resize(m, S::zero());
        }
        if w.g.len() < m + 1 {
            w.g.resize(m + 1, S::zero());
        }
    }

    /// Map pc_mode to the *expected* PcSide for Arnoldi semantics.
    /// (None doesn't care; we return Left for convenience.)
    #[allow(dead_code)]
    fn expected_side(&self) -> PcSide {
        match self.pc_mode {
            PcaPcMode::None | PcaPcMode::Left => PcSide::Left,
            PcaPcMode::Right => PcSide::Right,
        }
    }

    /// Apply (immutable) preconditioner or act as identity when None.
    #[inline]
    fn apply_pc(
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        if let Some(p) = pc {
            p.apply_s(side, x, y, scratch)
        } else {
            y.copy_from_slice(x);
            Ok(())
        }
    }

    /// Classical (optionally modified) Gram–Schmidt: project `w` on `V[0..k]`,
    /// write the column into `H[0..=k][k]`, return `||w||`.
    /// This is the hook to fuse reductions and, later, hoist k steps for CA/pipelining.
    fn project_and_normalize(
        &self,
        v_basis: &[Vec<S>],
        k: usize,
        w: &mut [S],
        hcols: &mut [Vec<S>],
        red: &ReductCtx,
    ) -> R {
        let hcol = &mut hcols[k];
        // First pass
        let mut first_pass: SmallVec<[S; 32]> = SmallVec::with_capacity(k + 1);
        first_pass.resize(k + 1, S::zero());
        {
            let mut pairs: SmallVec<[(&[S], &[S]); 32]> = SmallVec::with_capacity(k + 1);
            let w_view: &[S] = &w[..];
            for i in 0..=k {
                pairs.push((&v_basis[i], w_view));
            }
            red.dot_many_into(pairs.as_slice(), first_pass.as_mut_slice());
        }
        {
            let w_mut = &mut w[..];
            for (i, hij) in first_pass.iter().copied().enumerate() {
                hcol[i] = hij;
                for (wi, &vi) in w_mut.iter_mut().zip(&v_basis[i]) {
                    *wi -= hij * vi;
                }
            }
        }
        if self.modified_gs {
            // Re-orthogonalize for robustness
            let mut corr: SmallVec<[S; 32]> = SmallVec::with_capacity(k + 1);
            corr.resize(k + 1, S::zero());
            {
                let mut pairs: SmallVec<[(&[S], &[S]); 32]> = SmallVec::with_capacity(k + 1);
                let w_view: &[S] = &w[..];
                for i in 0..=k {
                    pairs.push((&v_basis[i], w_view));
                }
                red.dot_many_into(pairs.as_slice(), corr.as_mut_slice());
            }
            {
                let w_mut = &mut w[..];
                for (i, corr_val) in corr.into_iter().enumerate() {
                    if corr_val.abs() > 1e-12 {
                        hcol[i] += corr_val;
                        for (wi, &vi) in w_mut.iter_mut().zip(&v_basis[i]) {
                            *wi -= corr_val * vi;
                        }
                    }
                }
            }
        }
        let hnorm = red.norm2(w);
        if hcol.len() > k + 1 {
            hcol[k + 1] = S::from_real(hnorm);
            for val in hcol.iter_mut().skip(k + 2) {
                *val = S::zero();
            }
        }
        hnorm
    }
}

impl LinearSolver for PcaGmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        // Outline only; concrete sizes are set in solve()
        let m = self.restart;
        if w.q_s.len() < m + 1 {
            w.q_s.resize(m + 1, Vec::new());
        }
        if w.z_s.len() < m {
            w.z_s.resize(m, Vec::new());
        }
        if w.h_s.len() < m {
            w.h_s.resize(m, Vec::new());
        }
        if w.cs.len() < m {
            w.cs.resize(m, R::default());
        }
        if w.sn.len() < m {
            w.sn.resize(m, S::zero());
        }
        if w.g.len() < m + 1 {
            w.g.resize(m + 1, S::zero());
        }
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side_arg: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        self.solve_f64(a, pc.as_deref(), b, x, pc_side_arg, comm, monitors, work)
    }
}

impl PcaGmresSolver {
    #[allow(clippy::too_many_arguments)]
    pub fn solve_k<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side_arg: PcSide,
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
                "PCA-GMRES: dimension mismatch or non-square operator".into(),
            ));
        }

        let has_pc = pc.is_some();
        let mode = if has_pc {
            self.pc_mode
        } else {
            PcaPcMode::None
        };
        let expected_side = match mode {
            PcaPcMode::None | PcaPcMode::Left => PcSide::Left,
            PcaPcMode::Right => PcSide::Right,
        };

        if has_pc && pc_side_arg != expected_side {
            return Err(KError::InvalidInput(format!(
                "PCA-GMRES: pc_mode={:?} expects pc_side={:?}, got {:?}",
                self.pc_mode, expected_side, pc_side_arg
            )));
        }

        let mut owned;
        let ws = if let Some(w) = work {
            w
        } else {
            owned = Workspace::new(n);
            &mut owned
        };
        self.ensure_workspace(ws, n);
        let red = ReductCtx::new(comm, Some(&*ws));

        a.matvec_s(x, &mut ws.tmp1, &mut ws.bridge);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }

        let (beta0, mut bnorm) = match mode {
            PcaPcMode::None => {
                let mut norms = [R::zero(); 2];
                red.norm2_many_into(&[&ws.tmp1, b], &mut norms);
                let beta = norms[0];
                let v0 = &mut ws.q_s[0][..];
                if beta > R::default() {
                    let denom = S::from_real(beta);
                    for i in 0..n {
                        v0[i] = ws.tmp1[i] / denom;
                    }
                } else {
                    v0.fill(S::zero());
                }
                (beta, norms[1])
            }
            PcaPcMode::Left => {
                Self::apply_pc(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2, &mut ws.bridge)?;
                let mut norms = [R::zero(); 2];
                red.norm2_many_into(&[&ws.tmp2, b], &mut norms);
                let beta = norms[0];
                let v0 = &mut ws.q_s[0][..];
                if beta > R::default() {
                    let denom = S::from_real(beta);
                    for i in 0..n {
                        v0[i] = ws.tmp2[i] / denom;
                    }
                } else {
                    v0.fill(S::zero());
                }
                (beta, norms[1])
            }
            PcaPcMode::Right => {
                let mut norms = [R::zero(); 2];
                red.norm2_many_into(&[&ws.tmp1, b], &mut norms);
                let beta = norms[0];
                let v0 = &mut ws.q_s[0][..];
                if beta > R::default() {
                    let denom = S::from_real(beta);
                    for i in 0..n {
                        v0[i] = ws.tmp1[i] / denom;
                    }
                } else {
                    v0.fill(S::zero());
                }
                (beta, norms[1])
            }
        };

        ws.h_s.iter_mut().for_each(|row| row.fill(S::zero()));
        ws.cs.fill(R::default());
        ws.sn.fill(S::zero());
        ws.g.fill(S::zero());
        ws.g[0] = S::from_real(beta0);

        bnorm = bnorm.max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);

        let mut total_iters = 0usize;
        let mut res = beta0;
        let mut stats = SolveStats::new(0, res, ConvergedReason::Continued);
        let mons = monitors.unwrap_or(&[]);

        for m in mons {
            m(0, res);
        }
        if res <= thr {
            stats.final_residual = res;
            stats.reason = if res <= self.conv.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            return Ok(stats);
        }

        'outer: while total_iters < self.conv.max_iters {
            let max_k = self.restart.min(self.conv.max_iters - total_iters);
            let mut arnoldi_steps = 0usize;

            for k in 0..max_k {
                match mode {
                    PcaPcMode::None => {
                        a.matvec_s(&ws.q_s[k], &mut ws.tmp1, &mut ws.bridge);
                    }
                    PcaPcMode::Left => {
                        a.matvec_s(&ws.q_s[k], &mut ws.tmp1, &mut ws.bridge);
                        Self::apply_pc(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2, &mut ws.bridge)?;
                        ws.tmp1.copy_from_slice(&ws.tmp2);
                    }
                    PcaPcMode::Right => {
                        let zk = &mut ws.z_s[k][..];
                        Self::apply_pc(pc, PcSide::Right, &ws.q_s[k], zk, &mut ws.bridge)?;
                        a.matvec_s(zk, &mut ws.tmp1, &mut ws.bridge);
                    }
                }

                let hnorm =
                    self.project_and_normalize(&ws.q_s, k, &mut ws.tmp1, &mut ws.h_s, &red);

                let vnext = &mut ws.q_s[k + 1][..];
                if hnorm > R::default() {
                    let denom = S::from_real(hnorm);
                    for i in 0..n {
                        vnext[i] = ws.tmp1[i] / denom;
                    }
                } else {
                    vnext.fill(S::zero());
                }

                let hcol = &mut ws.h_s[k];
                apply_prev_givens_to_col(&mut hcol[..=k + 1], k, &ws.cs, &ws.sn);
                apply_new_givens_and_update_g(
                    &mut hcol[..=k + 1],
                    k,
                    &mut ws.cs,
                    &mut ws.sn,
                    &mut ws.g,
                );

                res = ws.g[k + 1].abs();
                total_iters += 1;
                arnoldi_steps = k + 1;

                for m in mons {
                    m(total_iters, res);
                }
                let (reason, sstats) = self.conv.check(res, beta0, total_iters);
                stats = sstats;
                if matches!(
                    reason,
                    ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
                ) {
                    break;
                }
                if total_iters >= self.conv.max_iters {
                    break;
                }
            }

            let k = arnoldi_steps;
            let mut y = vec![S::zero(); k];
            for i in (0..k).rev() {
                let mut sum = ws.g[i];
                for j in (i + 1)..k {
                    sum -= ws.h_s[j][i] * y[j];
                }
                y[i] = sum / ws.h_s[i][i];
            }

            match mode {
                PcaPcMode::Right => {
                    for i in 0..k {
                        let zi = &ws.z_s[i][..];
                        for (xj, &zij) in x.iter_mut().zip(zi) {
                            *xj += y[i] * zij;
                        }
                    }
                }
                PcaPcMode::None | PcaPcMode::Left => {
                    for i in 0..k {
                        let vi = &ws.q_s[i][..];
                        for (xj, &vij) in x.iter_mut().zip(vi) {
                            *xj += y[i] * vij;
                        }
                    }
                }
            }

            a.matvec_s(x, &mut ws.tmp1, &mut ws.bridge);
            for i in 0..n {
                ws.tmp1[i] = b[i] - ws.tmp1[i];
            }
            let beta0_new: R = match mode {
                PcaPcMode::None => {
                    let beta = red.norm2(&ws.tmp1);
                    let v0 = &mut ws.q_s[0][..];
                    if beta > R::default() {
                        let denom = S::from_real(beta);
                        for i in 0..n {
                            v0[i] = ws.tmp1[i] / denom;
                        }
                    } else {
                        v0.fill(S::zero());
                    }
                    beta
                }
                PcaPcMode::Left => {
                    Self::apply_pc(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2, &mut ws.bridge)?;
                    let beta = red.norm2(&ws.tmp2);
                    let v0 = &mut ws.q_s[0][..];
                    if beta > R::default() {
                        let denom = S::from_real(beta);
                        for i in 0..n {
                            v0[i] = ws.tmp2[i] / denom;
                        }
                    } else {
                        v0.fill(S::zero());
                    }
                    beta
                }
                PcaPcMode::Right => {
                    let beta = red.norm2(&ws.tmp1);
                    let v0 = &mut ws.q_s[0][..];
                    if beta > R::default() {
                        let denom = S::from_real(beta);
                        for i in 0..n {
                            v0[i] = ws.tmp1[i] / denom;
                        }
                    } else {
                        v0.fill(S::zero());
                    }
                    beta
                }
            };

            ws.h_s.iter_mut().for_each(|row| row.fill(S::zero()));
            ws.cs.fill(R::default());
            ws.sn.fill(S::zero());
            ws.g.fill(S::zero());
            ws.g[0] = S::from_real(beta0_new);

            if total_iters >= self.conv.max_iters {
                break 'outer;
            }
            if beta0_new <= thr {
                stats.reason = if beta0_new <= self.conv.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                };
                stats.final_residual = beta0_new;
                break 'outer;
            }
        }

        a.matvec_s(x, &mut ws.tmp1, &mut ws.bridge);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }
        let true_res: R = red.norm2(&ws.tmp1);
        let (_r, mut s) = self.conv.check(true_res, bnorm, total_iters);
        s.iterations = total_iters;
        s.final_residual = true_res;
        if matches!(s.reason, ConvergedReason::Continued) {
            s.reason = if true_res <= thr {
                if true_res <= self.conv.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                }
            } else {
                ConvergedReason::DivergedMaxIts
            };
        }
        Ok(s)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_f64<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn Preconditioner>,
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
            self.solve_k(&op, pc_ref, b_s, x_s, pc_side, comm, monitors, work)
        }

        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result = self.solve_k(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors, work);
            if result.is_ok() {
                copy_scalar_to_real_in(&x_s, x);
            }
            result
        }
    }

    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart.max(1);
    }

    pub fn set_pc_mode(&mut self, mode: PcaPcMode) {
        self.pc_mode = mode;
    }

    pub fn set_reorthog(&mut self, flag: bool) {
        self.modified_gs = flag;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::blas::{dot_conj, nrm2};

    use crate::context::ksp_context::Workspace;
    use crate::parallel::NoComm;
    use crate::testkit::ATOL;

    #[test]
    fn arnoldi_project_and_normalize_orthonormalizes_vector() {
        let solver = PcaGmresSolver::new(2, 1, 1, 1e-6, 10);
        let n = 2;
        let mut ws = Workspace::new(n);
        solver.ensure_workspace(&mut ws, n);

        ws.q_s[0][0] = S::one();
        ws.q_s[0][1] = S::zero();
        ws.h_s.iter_mut().for_each(|col| col.fill(S::zero()));

        let mut w = vec![S::zero(); n];
        w[1] = S::one();

        let comm = UniverseComm::NoComm(NoComm);
        let red = ReductCtx::new(&comm, Some(&ws));
        let hnorm = solver.project_and_normalize(&ws.q_s, 0, &mut w, &mut ws.h_s, &red);

        let tol = ATOL;
        assert!((hnorm - S::one().real()).abs() < tol);
        assert!((nrm2(&w) - S::one().real()).abs() < tol);
        assert!(dot_conj(&ws.q_s[0], &w).abs() < tol);
        assert!(ws.h_s[0][0].abs() < tol);
        assert!((ws.h_s[0][1].abs() - S::one().real()).abs() < tol);
    }
}
