//! PCA-GMRES (baseline) over &dyn LinOp<f64> with left/right/no preconditioning,
//! using disjoint slabs for V and Z, with semantics enforced by `pc_mode`.

use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::algebra::scalar::{copy_real_to_scalar_in, copy_scalar_to_real_in};
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::op_bridge::matvec_s;
use crate::parallel::UniverseComm;
use crate::preconditioner::bridge::apply_pc_s;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::givens::{apply_new_givens_and_update_g, apply_prev_givens_to_col};
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
            w.cs.resize(m, 0.0);
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
        hcols: &mut [Vec<S>],
    ) -> R {
        let hcol = &mut hcols[k];
        // First pass
        for i in 0..=k {
            let hij: S = dot_conj(&v_basis[i], w);
            hcol[i] = hij;
            for (wi, &vi) in w.iter_mut().zip(&v_basis[i]) {
                *wi -= hij * vi;
            }
        }
        if self.modified_gs {
            // Re-orthogonalize for robustness
            for i in 0..=k {
                let corr: S = dot_conj(&v_basis[i], w);
                if corr.abs() > 1e-12 {
                    hcol[i] += corr;
                    for (wi, &vi) in w.iter_mut().zip(&v_basis[i]) {
                        *wi -= corr * vi;
                    }
                }
            }
        }
        let hnorm = nrm2(w);
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
            w.cs.resize(m, 0.0);
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

        let mut x_s = vec![S::zero(); n];
        copy_real_to_scalar_in(&x[..], &mut x_s);
        let mut write_back = |xs: &[S]| {
            copy_scalar_to_real_in(xs, x);
        };
        let mut b_s = vec![S::zero(); n];
        copy_real_to_scalar_in(b, &mut b_s);

        // r = b - A x
        matvec_s(a, &x_s, &mut ws.tmp1, &mut ws.bridge);
        for i in 0..n {
            ws.tmp1[i] = b_s[i] - ws.tmp1[i];
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

        ws.h_s.iter_mut().for_each(|row| row.fill(S::zero()));
        ws.cs.fill(0.0);
        ws.sn.fill(S::zero());
        ws.g.fill(S::zero());
        ws.g[0] = S::from_real(beta0);

        let bnorm = nrm2(&b_s).max(1e-32);
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
            write_back(&x_s);
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
                let hnorm = self.project_and_normalize(&ws.q_s, k, &mut ws.tmp1, &mut ws.h_s);

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

                let hcol = &mut ws.h_s[k];
                apply_prev_givens_to_col(&mut hcol[..=k + 1], k, &ws.cs, &ws.sn);
                apply_new_givens_and_update_g(
                    &mut hcol[..=k + 1],
                    k,
                    &mut ws.cs,
                    &mut ws.sn,
                    &mut ws.g,
                );

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
                for j in (i + 1)..k {
                    sum -= ws.h_s[j][i] * y[j];
                }
                y[i] = sum / ws.h_s[i][i];
            }

            // Update x with V or Z depending on mode
            match mode {
                PcaPcMode::Right => {
                    for i in 0..k {
                        let zi = &ws.z_s[i][..];
                        for (xj, &zij) in x_s.iter_mut().zip(zi) {
                            *xj += y[i] * zij;
                        }
                    }
                }
                PcaPcMode::None | PcaPcMode::Left => {
                    for i in 0..k {
                        let vi = &ws.q_s[i][..];
                        for (xj, &vij) in x_s.iter_mut().zip(vi) {
                            *xj += y[i] * vij;
                        }
                    }
                }
            }

            // Restart: r = b - A x, rebuild v0 based on mode
            matvec_s(a, &x_s, &mut ws.tmp1, &mut ws.bridge);
            for i in 0..n {
                ws.tmp1[i] = b_s[i] - ws.tmp1[i];
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
            ws.h_s.iter_mut().for_each(|row| row.fill(S::zero()));
            ws.cs.fill(0.0);
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

        // Report *true* residual at the end for consistency
        matvec_s(a, &x_s, &mut ws.tmp1, &mut ws.bridge);
        for i in 0..n {
            ws.tmp1[i] = b_s[i] - ws.tmp1[i];
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
        write_back(&x_s);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::blas::{dot_conj, nrm2};
    use crate::context::ksp_context::Workspace;

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

        let hnorm = solver.project_and_normalize(&ws.q_s, 0, &mut w, &mut ws.h_s);

        assert!((hnorm - 1.0).abs() < 1e-12);
        assert!((nrm2(&w) - 1.0).abs() < 1e-12);
        assert!(dot_conj(&ws.q_s[0], &w).abs() < 1e-12);
        assert!(ws.h_s[0][0].abs() < 1e-12);
        assert!((ws.h_s[0][1].abs() - 1.0).abs() < 1e-12);
    }
}
