//! PCA-GMRES (baseline) over &dyn LinOp<f64> with left/right/no preconditioning.
//! Block size and pipeline depth are accepted but currently executed with s=1,
//! while keeping the data layout and hooks to enable CA/pipelining next.

use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::UniverseComm;
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
    pub conv: Convergence<f64>,
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
            pc_mode: PcaPcMode::Left, // matches former default
            modified_gs: true,
            haptol: 1e-12,
        }
    }

    #[inline]
    fn dot(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
    }
    #[inline]
    fn nrm2(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn ensure_workspace(&self, w: &mut Workspace, n: usize) {
        let m = self.restart;
        // Need: q for basis vectors. For RIGHT PC we also store Z (preconditioned directions).
        // We re-use q as: V: [0..=m], Z: [m+1 .. m+m] (m slots)
        let need_q = if matches!(self.pc_mode, PcaPcMode::Right) {
            2 * m + 1
        } else {
            m + 1
        };
        if w.q.len() < need_q {
            w.q.resize(need_q, Vec::new());
        }
        for q in &mut w.q {
            if q.len() != n {
                q.resize(n, 0.0);
            }
        }

        // H: (m+1) x m
        if w.h.len() < m + 1 {
            w.h.resize(m + 1, Vec::new());
        }
        for r in &mut w.h {
            if r.len() < m {
                r.resize(m, 0.0);
            }
        }

        if w.tmp1.len() != n {
            w.tmp1.resize(n, 0.0);
        }
        if w.tmp2.len() != n {
            w.tmp2.resize(n, 0.0);
        }
        if w.cs.len() < m {
            w.cs.resize(m, 0.0);
        }
        if w.sn.len() < m {
            w.sn.resize(m, 0.0);
        }
        if w.g.len() < m + 1 {
            w.g.resize(m + 1, 0.0);
        }
    }

    fn apply_givens(hij: &mut f64, hij1: &mut f64, c: f64, s: f64) {
        let t = c * (*hij) + s * (*hij1);
        *hij1 = -s * (*hij) + c * (*hij1);
        *hij = t;
    }
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
        // Shapes fixed at solve-time once n is known; reserve outlines now.
        let m = self.restart;
        if w.q.len() < m + 1 {
            w.q.resize(m + 1, Vec::new());
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
            w.g.resize(m + 1, 0.0);
        }
    }

    fn solve(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        _pc_side: PcSide,
        _comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        let (m, n) = a.dims();
        if m != n || b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "PCA-GMRES: dimension mismatch or non-square operator".into(),
            ));
        }

        // Baseline executes with block_size = 1, pipeline_depth = 1,
        // while keeping memory layout and hooks for later CA/pipelining.
        let m_restart = self.restart;

        // Workspace
        let mut owned;
        let ws = if let Some(w) = work {
            w
        } else {
            owned = Workspace {
                tmp1: vec![0.0; n],
                tmp2: vec![0.0; n],
                q: vec![vec![0.0; n]; m_restart + 1],
                h: vec![vec![0.0; m_restart]; m_restart + 1],
                cs: vec![0.0; m_restart],
                sn: vec![0.0; m_restart],
                g: vec![0.0; m_restart + 1],
                z: Vec::new(),
            };
            &mut owned
        };
        self.ensure_workspace(ws, n);

        // Memory layout in ws.q:
        //   V-basis: q[0..=m]
        //   Z-basis (RIGHT PC only): q[m+1 .. m+m] (length m)
        let v_off = 0usize;
        let z_off = m_restart + 1;

        // r = b - A x
        a.matvec(x, &mut ws.tmp1);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }
        let mut beta0 = Self::nrm2(&ws.tmp1);
        let bnorm = Self::nrm2(b).max(1e-32);
        let thr = self.conv.atol.max(self.conv.rtol * bnorm);

        // Initialize v0
        if beta0 > 0.0 {
            let v0 = &mut ws.q[v_off + 0][..];
            for i in 0..n {
                v0[i] = ws.tmp1[i] / beta0;
            }
        } else {
            ws.q[v_off + 0].fill(0.0);
        }
        ws.g.fill(0.0);
        ws.g[0] = beta0;

        let mut total_iters = 0usize;
        let mut res = beta0;
        let mut stats = SolveStats {
            iterations: 0,
            final_residual: res,
            reason: ConvergedReason::Continued,
        };

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
            // Arnoldi up to m_restart steps
            let mut arnoldi_steps = 0usize;
            for j in 0..m_restart.min(self.conv.max_iters - total_iters) {
                // w = A * (direction)
                match self.pc_mode {
                    PcaPcMode::None => {
                        a.matvec(&ws.q[v_off + j], &mut ws.tmp2);
                    }
                    PcaPcMode::Left => {
                        // w = M^{-1} A v_j
                        a.matvec(&ws.q[v_off + j], &mut ws.tmp1);
                        if let Some(p) = pc {
                            p.apply(PcSide::Left, &ws.tmp1, &mut ws.tmp2)?;
                        } else {
                            ws.tmp2.copy_from_slice(&ws.tmp1);
                        }
                    }
                    PcaPcMode::Right => {
                        // z_j = M^{-1} v_j; w = A z_j
                        if let Some(p) = pc {
                            let (v_part, z_part) = ws.q.split_at_mut(z_off);
                            let vj = &v_part[v_off + j];
                            let zj = &mut z_part[j][..];
                            p.apply(PcSide::Right, vj, zj)?;
                            a.matvec(zj, &mut ws.tmp2);
                        } else {
                            // no PC: treat as None path
                            a.matvec(&ws.q[v_off + j], &mut ws.tmp2);
                        }
                    }
                }

                // Classical GS + optional re-orthogonalization
                for i in 0..=j {
                    let hij = Self::dot(&ws.tmp2, &ws.q[v_off + i]);
                    ws.h[i][j] = hij;
                    for (w, &vi) in ws.tmp2.iter_mut().zip(&ws.q[v_off + i]) {
                        *w -= hij * vi;
                    }
                }
                if self.modified_gs {
                    for i in 0..=j {
                        let corr = Self::dot(&ws.tmp2, &ws.q[v_off + i]);
                        if corr.abs() > 1e-12 {
                            ws.h[i][j] += corr;
                            for (w, &vi) in ws.tmp2.iter_mut().zip(&ws.q[v_off + i]) {
                                *w -= corr * vi;
                            }
                        }
                    }
                }
                // h[j+1, j] = ||w||
                let hij1 = Self::nrm2(&ws.tmp2);
                ws.h[j + 1][j] = hij1;

                // v_{j+1}
                let vnext = &mut ws.q[v_off + (j + 1)][..];
                if hij1 > 0.0 {
                    for i in 0..n {
                        vnext[i] = ws.tmp2[i] / hij1;
                    }
                } else {
                    vnext.fill(0.0);
                }

                // Apply previous Givens
                for i in 0..j {
                    let (top, rest) = ws.h.split_at_mut(i + 1);
                    let row_i = &mut top[i];
                    let row_ip1 = &mut rest[0];
                    Self::apply_givens(&mut row_i[j], &mut row_ip1[j], ws.cs[i], ws.sn[i]);
                }
                // New Givens to zero h[j+1, j]
                let (top, rest) = ws.h.split_at_mut(j + 1);
                let row_j = &mut top[j];
                let row_j1 = &mut rest[0];
                let (c, s) = Self::givens(row_j[j], row_j1[j]);
                ws.cs[j] = c;
                ws.sn[j] = s;
                Self::apply_givens(&mut row_j[j], &mut row_j1[j], c, s);

                // Update RHS g
                let t = c * ws.g[j] + s * ws.g[j + 1];
                ws.g[j + 1] = -s * ws.g[j] + c * ws.g[j + 1];
                ws.g[j] = t;

                res = ws.g[j + 1].abs();
                total_iters += 1;
                arnoldi_steps = j + 1;

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
            }

            // Back-substitute y
            let k = arnoldi_steps;
            let mut y = vec![0.0; k];
            for i in (0..k).rev() {
                let mut sum = ws.g[i];
                for l in (i + 1)..k {
                    sum -= ws.h[i][l] * y[l];
                }
                y[i] = sum / ws.h[i][i];
            }

            // Update x
            match self.pc_mode {
                PcaPcMode::Right if pc.is_some() => {
                    // x += Σ y_i * z_i
                    for i in 0..k {
                        let zi = &ws.q[z_off + i][..];
                        for (xj, &zij) in x.iter_mut().zip(zi) {
                            *xj += y[i] * zij;
                        }
                    }
                }
                _ => {
                    // x += Σ y_i * v_i
                    for i in 0..k {
                        let vi = &ws.q[v_off + i][..];
                        for (xj, &vij) in x.iter_mut().zip(vi) {
                            *xj += y[i] * vij;
                        }
                    }
                }
            }

            // Restart: r = b - A x
            a.matvec(x, &mut ws.tmp1);
            for i in 0..n {
                ws.tmp1[i] = b[i] - ws.tmp1[i];
            }
            beta0 = Self::nrm2(&ws.tmp1);
            ws.g.fill(0.0);
            ws.g[0] = beta0;
            let v0 = &mut ws.q[v_off + 0][..];
            if beta0 > 0.0 {
                for i in 0..n {
                    v0[i] = ws.tmp1[i] / beta0;
                }
            } else {
                v0.fill(0.0);
            }

            if total_iters >= self.conv.max_iters {
                break 'outer;
            }
            if beta0 <= thr {
                stats.reason = if beta0 <= self.conv.atol {
                    ConvergedReason::ConvergedAtol
                } else {
                    ConvergedReason::ConvergedRtol
                };
                stats.final_residual = beta0;
                break 'outer;
            }
        }

        if matches!(stats.reason, ConvergedReason::Continued) {
            stats.final_residual = res;
            stats.reason = if res <= thr {
                ConvergedReason::ConvergedRtol
            } else {
                ConvergedReason::DivergedMaxIts
            };
        }
        Ok(stats)
    }
}
