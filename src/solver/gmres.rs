//! Standard restarted GMRES implementation over &dyn LinOp<f64> with explicit
//! left/right preconditioning semantics. The preconditioner is always applied
//! as `z <- M^{-1} x` and the solver decides whether to use the preconditioned
//! vectors on the left or right.

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, Convergence, SolveStats};
use std::any::Any;

pub struct GmresSolver {
    pub restart: usize,
    pub conv: Convergence<f64>,
    /// Happy breakdown tolerance
    pub haptol: f64,
}

impl GmresSolver {
    pub fn new(restart: usize, rtol: f64, maxits: usize) -> Self {
        Self {
            restart: restart.max(1),
            conv: Convergence { rtol, atol: 1e-12, dtol: 1e3, max_iters: maxits },
            haptol: 1e-12,
        }
    }

    #[inline]
    fn dot(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
    }
    #[inline]
    fn nrm2(x: &[f64]) -> f64 {
        Self::dot(x, x).sqrt()
    }

    fn ensure_workspace(&self, w: &mut Workspace, n: usize) {
        let m = self.restart;
        if w.tmp1.len() != n { w.tmp1.resize(n, 0.0); }
        if w.tmp2.len() != n { w.tmp2.resize(n, 0.0); }
        if w.q.len() < m + 1 { w.q.resize(m + 1, Vec::new()); }
        for q in &mut w.q[..m + 1] {
            if q.len() != n { q.resize(n, 0.0); }
        }
        if w.z.len() < m { w.z.resize(m, Vec::new()); }
        for z in &mut w.z[..m] {
            if z.len() != n { z.resize(n, 0.0); }
        }
        if w.h.len() < m + 1 { w.h.resize(m + 1, Vec::new()); }
        for row in &mut w.h[..m + 1] {
            if row.len() != m { row.resize(m, 0.0); }
        }
        if w.cs.len() < m { w.cs.resize(m, 0.0); }
        if w.sn.len() < m { w.sn.resize(m, 0.0); }
        if w.g.len() < m + 1 { w.g.resize(m + 1, 0.0); }
    }

    fn apply_precond(
        &self,
        pc: Option<&dyn Preconditioner>,
        side: PcSide,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), KError> {
        if let Some(p) = pc {
            p.apply(side, x, y)
        } else {
            y.copy_from_slice(x);
            Ok(())
        }
    }

    fn orthonormalize(q: &Vec<Vec<f64>>, w: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
        let k = q.len();
        let mut hcol = vec![0.0; k + 1];
        for i in 0..k {
            let hij = Self::dot(w, &q[i]);
            hcol[i] = hij;
            for (wi, &qij) in w.iter_mut().zip(&q[i]) {
                *wi -= hij * qij;
            }
        }
        let hnext = Self::nrm2(w);
        hcol[k] = hnext;
        let mut v_next = vec![0.0; w.len()];
        if hnext > 0.0 {
            for i in 0..w.len() {
                v_next[i] = w[i] / hnext;
            }
        }
        (hcol, v_next)
    }

    fn apply_givens_and_update(
        h: &mut [Vec<f64>],
        cs: &mut [f64],
        sn: &mut [f64],
        g: &mut [f64],
        k: usize,
    ) {
        for i in 0..k {
            let (hij, hij1) = {
                let (top, bottom) = h.split_at_mut(i + 1);
                (&mut top[i][k], &mut bottom[0][k])
            };
            let c = cs[i];
            let s = sn[i];
            let t = c * *hij + s * *hij1;
            *hij1 = -s * *hij + c * *hij1;
            *hij = t;
        }

        let (hkk, hk1k) = {
            let (top, bottom) = h.split_at_mut(k + 1);
            (&mut top[k][k], &mut bottom[0][k])
        };
        let (c, s) = if *hk1k == 0.0 {
            (1.0, 0.0)
        } else {
            let r = (*hkk * *hkk + *hk1k * *hk1k).sqrt();
            (*hkk / r, *hk1k / r)
        };
        cs[k] = c;
        sn[k] = s;
        {
            let (top, bottom) = h.split_at_mut(k + 1);
            let hjk = &mut top[k][k];
            let hj1k = &mut bottom[0][k];
            let t = c * *hjk + s * *hj1k;
            *hj1k = -s * *hjk + c * *hj1k;
            *hjk = t;
        }
        let t = c * g[k] + s * g[k + 1];
        g[k + 1] = -s * g[k] + c * g[k + 1];
        g[k] = t;
    }

    fn backsolve(h: &[Vec<f64>], g: &[f64], k: usize) -> Vec<f64> {
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for l in (i + 1)..k {
                sum -= h[i][l] * y[l];
            }
            y[i] = sum / h[i][i];
        }
        y
    }

    fn axpy_update(x: &mut [f64], basis: &[Vec<f64>], y: &[f64]) {
        for (j, yj) in y.iter().enumerate() {
            let v = &basis[j];
            for (xi, &vj) in x.iter_mut().zip(v) {
                *xi += yj * vj;
            }
        }
    }
}

impl LinearSolver for GmresSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn setup_workspace(&mut self, w: &mut Workspace) {
        if w.q.len() < self.restart + 1 {
            w.q.resize(self.restart + 1, Vec::new());
        }
        if w.z.len() < self.restart {
            w.z.resize(self.restart, Vec::new());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        _comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "GMRES: dimension mismatch or non-square operator".into(),
            ));
        }

        let mut owned_ws;
        let ws = if let Some(w) = work { w } else {
            owned_ws = Workspace::new(n);
            &mut owned_ws
        };
        self.ensure_workspace(ws, n);

        // r0 = b - A x
        a.matvec(x, &mut ws.tmp1);
        for i in 0..n { ws.tmp1[i] = b[i] - ws.tmp1[i]; }

        ws.q.clear();
        ws.h.iter_mut().for_each(|r| r.fill(0.0));
        ws.cs.fill(0.0);
        ws.sn.fill(0.0);
        ws.g.fill(0.0);
        ws.z.clear();

        let beta = match pc_side {
            PcSide::Left => {
                self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                let beta = Self::nrm2(&ws.tmp2);
                if beta > 0.0 {
                    for i in 0..n { ws.tmp2[i] /= beta; }
                }
                ws.q.push(ws.tmp2.clone());
                beta
            }
            PcSide::Right | PcSide::Symmetric => {
                let beta = Self::nrm2(&ws.tmp1);
                let mut v0 = ws.tmp1.clone();
                if beta > 0.0 {
                    for i in 0..n { v0[i] /= beta; }
                }
                ws.q.push(v0);
                if matches!(pc_side, PcSide::Right) {
                    ws.z.resize(1, vec![0.0; n]);
                }
                beta
            }
        };

        ws.g[0] = beta;
        let bnorm = Self::nrm2(b).max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);

        let mut total_iters = 0usize;
        let mut res = beta;
        let mut stats = SolveStats { iterations: 0, final_residual: res, reason: ConvergedReason::Continued };

        if let Some(ms) = monitors {
            for m in ms { m(0, res); }
        }
        if res <= thr {
            stats.reason = if res <= self.conv.atol { ConvergedReason::ConvergedAtol } else { ConvergedReason::ConvergedRtol };
            stats.final_residual = res;
            return Ok(stats);
        }

        'outer: loop {
            let mut k_steps = 0usize;
            for k in 0..self.restart {
                match pc_side {
                    PcSide::Left => {
                        a.matvec(&ws.q[k], &mut ws.tmp1);
                        self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                    }
                    PcSide::Right | PcSide::Symmetric => {
                        self.apply_precond(pc, PcSide::Right, &ws.q[k], &mut ws.tmp2)?;
                        if matches!(pc_side, PcSide::Right) {
                            if ws.z.len() <= k { ws.z.resize(k + 1, vec![0.0; n]); }
                            ws.z[k].copy_from_slice(&ws.tmp2);
                        }
                        a.matvec(&ws.tmp2, &mut ws.tmp1);
                    }
                }

                let (hcol, vnext) = Self::orthonormalize(&ws.q, &mut ws.tmp1);
                for i in 0..=k + 1 { ws.h[i][k] = hcol[i]; }
                ws.q.push(vnext);

                Self::apply_givens_and_update(&mut ws.h, &mut ws.cs, &mut ws.sn, &mut ws.g, k);

                res = ws.g[k + 1].abs();
                total_iters += 1;
                k_steps = k + 1;

                if let Some(ms) = monitors {
                    for m in ms { m(total_iters, res); }
                }

                if res <= thr {
                    break;
                }

                if total_iters >= self.conv.max_iters {
                    break;
                }
            }

            let y = Self::backsolve(&ws.h, &ws.g, k_steps);
            match pc_side {
                PcSide::Left => Self::axpy_update(x, &ws.q, &y),
                PcSide::Right | PcSide::Symmetric => Self::axpy_update(x, &ws.z, &y),
            }

            // Recompute residual
            a.matvec(x, &mut ws.tmp1);
            for i in 0..n { ws.tmp1[i] = b[i] - ws.tmp1[i]; }
            res = match pc_side {
                PcSide::Left => {
                    self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                    Self::nrm2(&ws.tmp2)
                }
                PcSide::Right | PcSide::Symmetric => Self::nrm2(&ws.tmp1),
            };

            if res <= thr || total_iters >= self.conv.max_iters {
                break 'outer;
            }

            // Prepare next cycle
            ws.q.clear();
            ws.z.clear();
            ws.h.iter_mut().for_each(|r| r.fill(0.0));
            ws.cs.fill(0.0);
            ws.sn.fill(0.0);
            ws.g.fill(0.0);

            match pc_side {
                PcSide::Left => {
                    self.apply_precond(pc, PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                    let beta = Self::nrm2(&ws.tmp2);
                    if beta > 0.0 {
                        for i in 0..n { ws.tmp2[i] /= beta; }
                    }
                    ws.q.push(ws.tmp2.clone());
                    ws.g[0] = beta;
                }
                PcSide::Right | PcSide::Symmetric => {
                    let beta = Self::nrm2(&ws.tmp1);
                    let mut v0 = ws.tmp1.clone();
                    if beta > 0.0 {
                        for i in 0..n { v0[i] /= beta; }
                    }
                    ws.q.push(v0);
                    ws.g[0] = beta;
                    if matches!(pc_side, PcSide::Right) {
                        ws.z.resize(1, vec![0.0; n]);
                    }
                }
            }
        }

        // Compute true residual for reporting
        a.matvec(x, &mut ws.tmp1);
        for i in 0..n { ws.tmp1[i] = b[i] - ws.tmp1[i]; }
        let true_res = Self::nrm2(&ws.tmp1);
        let (_reason, mut s) = self.conv.check(true_res, bnorm, total_iters);
        s.final_residual = true_res;
        Ok(s)
    }
}

