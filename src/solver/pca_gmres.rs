//! PCA-GMRES (baseline) over &dyn LinOp<f64> with left/right/no preconditioning,
//! using disjoint slabs for V and Z, with semantics enforced by `pc_mode`.

use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::op_bridge::matvec_s;
use crate::parallel::UniverseComm;
use crate::preconditioner::bridge::apply_pc_s;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};

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
        if w.h.len() < m + 1 {
            w.h.resize(m + 1, Vec::new());
        }
        for row in &mut w.h[..m + 1] {
            if row.len() != m {
                row.resize(m, 0.0);
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
            w.cs.resize(m, 0.0);
        }
        if w.sn.len() < m {
            w.sn.resize(m, 0.0);
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
        pc: Option<&dyn Preconditioner>,
        side: PcSide,
        x: &[S],
        y: &mut [S],
        scratch: &mut BridgeScratch,
    ) -> Result<(), KError> {
        if let Some(p) = pc {
            apply_pc_s(p, side, x, y, scratch)
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
        h: &mut [Vec<f64>],
    ) -> R {
        // First pass
        for i in 0..=k {
            let hij: S = dot_conj(&v_basis[i], w);
            h[i][k] = hij.real();
            for (wi, &vi) in w.iter_mut().zip(&v_basis[i]) {
                *wi -= hij * vi;
            }
        }
        if self.modified_gs {
            // Re-orthogonalize for robustness
            for i in 0..=k {
                let corr: S = dot_conj(&v_basis[i], w);
                if corr.abs() > 1e-12 {
                    h[i][k] += corr.real();
                    for (wi, &vi) in w.iter_mut().zip(&v_basis[i]) {
                        *wi -= corr * vi;
                    }
                }
            }
        }
        nrm2(w)
    }

    #[inline]
    fn apply_givens(hij: &mut f64, hij1: &mut f64, c: f64, s: f64) {
        let t = c * (*hij) + s * (*hij1);
        *hij1 = -s * (*hij) + c * (*hij1);
        *hij = t;
    }
    #[inline]
    fn givens(a: f64, b: f64) -> (f64, f64) {
        if b == 0.0 {
            (1.0, 0.0)
        } else {
            let r = (a * a + b * b).sqrt();
            (a / r, b / r)
        }
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
        if w.h.len() < m + 1 {
            w.h.resize(m + 1, Vec::new());
        }
        if w.cs.len() < m {
            w.cs.resize(m, 0.0);
        }
        if w.sn.len() < m {
            w.sn.resize(m, 0.0);
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
        _comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let pc_in: Option<&dyn Preconditioner> = pc.as_deref();
        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "PCA-GMRES: dimension mismatch or non-square operator".into(),
            ));
        }

        // Determine the *effective* mode (degrade to None if pc is absent).
        let has_pc = pc_in.is_some();
        let mode = if has_pc {
            self.pc_mode
        } else {
            PcaPcMode::None
        };
        let expected_side = match mode {
            PcaPcMode::None | PcaPcMode::Left => PcSide::Left,
            PcaPcMode::Right => PcSide::Right,
        };

        // Enforce semantics: if a preconditioner is active, side must match the mode.
        if has_pc && pc_side_arg != expected_side {
            return Err(KError::InvalidInput(format!(
                "PCA-GMRES: pc_mode={:?} expects pc_side={:?}, got {:?}",
                self.pc_mode, expected_side, pc_side_arg
            )));
        }

        // Workspace
        let mut owned;
        let ws = if let Some(w) = work {
            w
        } else {
            owned = Workspace::new(n);
            &mut owned
        };
        self.ensure_workspace(ws, n);

        // r = b - A x
        matvec_s(a, x, &mut ws.tmp1, &mut ws.bridge);
        for i in 0..n {
            ws.tmp1[i] = S::from_real(b[i]) - ws.tmp1[i];
        }

        // Initialize v0 and g[0] according to mode
        let beta0: R = match mode {
            PcaPcMode::None => {
                let beta = nrm2(&ws.tmp1);
                let v0 = &mut ws.q_s[0][..];
                if beta > 0.0 {
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
                // v0 = M^{-1} r / ||M^{-1}r||
                Self::apply_pc(pc_in, PcSide::Left, &ws.tmp1, &mut ws.tmp2, &mut ws.bridge)?;
                let beta = nrm2(&ws.tmp2);
                let v0 = &mut ws.q_s[0][..];
                if beta > 0.0 {
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
                // v0 = r / ||r||  (Arnoldi on A M^{-1})
                let beta = nrm2(&ws.tmp1);
                let v0 = &mut ws.q_s[0][..];
                if beta > 0.0 {
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

        ws.h.iter_mut().for_each(|row| row.fill(0.0));
        ws.cs.fill(0.0);
        ws.sn.fill(0.0);
        ws.g.fill(S::zero());
        ws.g[0] = S::from_real(beta0);

        let bnorm = b.iter().map(|&bi| bi * bi).sum::<R>().sqrt().max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);

        let mut total_iters = 0usize;
        let mut res = beta0;
        let mut stats = SolveStats::new(0, res, ConvergedReason::Continued);

        if let Some(mons) = monitors {
            for m in mons {
                m(0, res);
            }
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
                // w = Arnoldi matvec depending on mode
                match mode {
                    PcaPcMode::None => {
                        // w = A v_k
                        matvec_s(a, &ws.q_s[k], &mut ws.tmp1, &mut ws.bridge);
                    }
                    PcaPcMode::Left => {
                        // w = M^{-1} A v_k
                        matvec_s(a, &ws.q_s[k], &mut ws.tmp1, &mut ws.bridge);
                        Self::apply_pc(
                            pc_in,
                            PcSide::Left,
                            &ws.tmp1,
                            &mut ws.tmp2,
                            &mut ws.bridge,
                        )?;
                        ws.tmp1.copy_from_slice(&ws.tmp2);
                    }
                    PcaPcMode::Right => {
                        // z_k = M^{-1} v_k; w = A z_k
                        let zk = &mut ws.z_s[k][..];
                        Self::apply_pc(pc_in, PcSide::Right, &ws.q_s[k], zk, &mut ws.bridge)?;
                        matvec_s(a, zk, &mut ws.tmp1, &mut ws.bridge);
                    }
                }

                // Orthonormalize against V
                let hnorm = self.project_and_normalize(&ws.q_s, k, &mut ws.tmp1, &mut ws.h);
                ws.h[k + 1][k] = hnorm;

                // v_{k+1}
                let vnext = &mut ws.q_s[k + 1][..];
                if hnorm > 0.0 {
                    let denom = S::from_real(hnorm);
                    for i in 0..n {
                        vnext[i] = ws.tmp1[i] / denom;
                    }
                } else {
                    vnext.fill(S::zero());
                }

                // Apply stored Givens
                for i in 0..k {
                    let (top, rest) = ws.h.split_at_mut(i + 1);
                    let hij = &mut top[i][k];
                    let hij1 = &mut rest[0][k];
                    Self::apply_givens(hij, hij1, ws.cs[i], ws.sn[i]);
                }
                // New Givens
                let (top, rest) = ws.h.split_at_mut(k + 1);
                let hjk = &mut top[k][k];
                let hj1k = &mut rest[0][k];
                let (c, s) = Self::givens(*hjk, *hj1k);
                ws.cs[k] = c;
                ws.sn[k] = s;
                Self::apply_givens(hjk, hj1k, c, s);

                // Update RHS g
                let gk = ws.g[k];
                let gk1 = ws.g[k + 1];
                let c_s = S::from_real(c);
                let s_s = S::from_real(s);
                ws.g[k] = c_s * gk + s_s * gk1;
                ws.g[k + 1] = -s_s * gk + c_s * gk1;

                res = ws.g[k + 1].abs(); // estimate; equals true ||r|| for Right/None, precond ||M^{-1}r|| for Left
                total_iters += 1;
                arnoldi_steps = k + 1;

                if let Some(mons) = monitors {
                    for m in mons {
                        m(total_iters, res);
                    }
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

            // Back-substitute
            let k = arnoldi_steps;
            let mut y = vec![S::zero(); k];
            for i in (0..k).rev() {
                let mut sum = ws.g[i];
                for l in (i + 1)..k {
                    sum -= S::from_real(ws.h[i][l]) * y[l];
                }
                y[i] = sum / S::from_real(ws.h[i][i]);
            }

            // Update x with V or Z depending on mode
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

            // Restart: r = b - A x, rebuild v0 based on mode
            matvec_s(a, x, &mut ws.tmp1, &mut ws.bridge);
            for i in 0..n {
                ws.tmp1[i] = S::from_real(b[i]) - ws.tmp1[i];
            }
            let beta0_new: R = match mode {
                PcaPcMode::None => {
                    let beta = nrm2(&ws.tmp1);
                    let v0 = &mut ws.q_s[0][..];
                    if beta > 0.0 {
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
                    Self::apply_pc(pc_in, PcSide::Left, &ws.tmp1, &mut ws.tmp2, &mut ws.bridge)?;
                    let beta = nrm2(&ws.tmp2);
                    let v0 = &mut ws.q_s[0][..];
                    if beta > 0.0 {
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
                    let beta = nrm2(&ws.tmp1);
                    let v0 = &mut ws.q_s[0][..];
                    if beta > 0.0 {
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

            // Reset Hessenberg and RHS for next cycle
            ws.h.iter_mut().for_each(|row| row.fill(0.0));
            ws.cs.fill(0.0);
            ws.sn.fill(0.0);
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

        // Report *true* residual at the end for consistency
        matvec_s(a, x, &mut ws.tmp1, &mut ws.bridge);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }
        let true_res: R = nrm2(&ws.tmp1);
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
}

impl PcaGmresSolver {
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
