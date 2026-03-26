use crate::algebra::prelude::*;
use crate::context::ksp_context::Workspace;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc_mut};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::common::{ReductCtx, call_monitors, recompute_true_residual_norm_s};
use crate::solver::{LinearSolver, MonitorCallback};
use crate::utils::convergence::{
    ConvergedReason, GcrCounters, ReductionModel, SolveStats, SolverCounters,
};
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GcrOrthog {
    Classical,
    Modified,
}

/// Pipelined flexible GCR with dedicated GCR recurrence/storage.
pub struct PipeGcrSolver {
    pub rtol: f64,
    pub atol: f64,
    pub dtol: f64,
    pub maxits: usize,
    pub restart: usize,
    pub orthog: GcrOrthog,
}

impl PipeGcrSolver {
    pub fn new(restart: usize, rtol: f64, maxits: usize) -> Self {
        Self {
            rtol,
            atol: 1e-12,
            dtol: 1e3,
            maxits,
            restart: restart.max(1),
            orthog: GcrOrthog::Classical,
        }
    }

    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart.max(1);
    }

    pub fn set_orthog(&mut self, orthog: GcrOrthog) {
        self.orthog = orthog;
    }

    fn reduction_model(&self) -> ReductionModel {
        let per_iter = match self.orthog {
            GcrOrthog::Classical => 2.0,
            GcrOrthog::Modified => 2.0,
        };
        ReductionModel {
            variant: "pipegcr",
            startup: 2,
            per_iteration: per_iter,
            tail: 1,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_k<A>(
        &mut self,
        a: &A,
        mut pc: Option<&mut dyn KPreconditioner<Scalar = S>>,
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
        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput(
                "PipeGCR requires a square operator".to_string(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "PipeGCR: vector size mismatch".to_string(),
            ));
        }

        let pc_apply_side = match pc_side {
            PcSide::Right => PcSide::Right,
            PcSide::Left | PcSide::Symmetric => PcSide::Right,
        };

        let mut owned_ws;
        let ws = if let Some(w) = work {
            w
        } else {
            owned_ws = Workspace::new(n);
            &mut owned_ws
        };
        ws.tmp1.resize(n, S::zero());
        ws.tmp2.resize(n, S::zero());
        let red = ReductCtx::new(comm, Some(&*ws));
        let mut v = vec![S::zero(); n];
        let mut true_res_buf = vec![S::zero(); n];

        a.matvec_s(x, &mut ws.tmp1[..n], &mut ws.bridge);
        for i in 0..n {
            ws.tmp1[i] = b[i] - ws.tmp1[i];
        }

        let mut norms = [R::zero(); 2];
        red.norm2_many_into(&[&ws.tmp1[..n], b], &mut norms);
        let bnorm = norms[1].max(1e-32);
        let mut rnorm = norms[0];
        let mons = monitors.unwrap_or(&[]);

        let mut sync_count = 1usize;
        if call_monitors(mons, 0, rnorm, sync_count) {
            let counters = SolverCounters {
                num_global_reductions: sync_count,
                ..SolverCounters::default()
            };
            return Ok(SolveStats::new(0, rnorm, ConvergedReason::StoppedByMonitor)
                .with_counters(counters)
                .with_reduction_model(self.reduction_model())
                .with_gcr_counters(GcrCounters {
                    basis_updates: 0,
                    sync_count,
                    restart_count: 0,
                    restarted: false,
                }));
        }

        let threshold = self.atol.max(self.rtol * bnorm);
        if rnorm <= threshold {
            let reason = if rnorm <= self.atol {
                ConvergedReason::ConvergedAtol
            } else {
                ConvergedReason::ConvergedRtol
            };
            let counters = SolverCounters {
                num_global_reductions: sync_count,
                ..SolverCounters::default()
            };
            return Ok(SolveStats::new(0, rnorm, reason)
                .with_counters(counters)
                .with_reduction_model(self.reduction_model())
                .with_gcr_counters(GcrCounters {
                    basis_updates: 0,
                    sync_count,
                    restart_count: 0,
                    restarted: false,
                }));
        }

        let mut p_basis: Vec<Vec<S>> = Vec::with_capacity(self.restart);
        let mut ap_basis: Vec<Vec<S>> = Vec::with_capacity(self.restart);
        let mut iters = 0usize;
        let mut basis_updates = 0usize;
        let mut restart_count = 0usize;

        while iters < self.maxits {
            if !p_basis.is_empty() {
                p_basis.clear();
                ap_basis.clear();
                restart_count += 1;
            }

            let cycle = self.restart.min(self.maxits - iters);
            for _ in 0..cycle {
                // u = M^{-1} r (or r when no PC)
                if let Some(pc_ref) = pc.as_deref_mut() {
                    pc_ref.apply_mut_s(
                        pc_apply_side,
                        &ws.tmp1[..n],
                        &mut ws.tmp2[..n],
                        &mut ws.bridge,
                    )?;
                } else {
                    ws.tmp2[..n].copy_from_slice(&ws.tmp1[..n]);
                }

                // v = A u
                a.matvec_s(&ws.tmp2[..n], &mut v[..n], &mut ws.bridge);

                // Orthogonalize (u,v) against prior Ap basis
                match self.orthog {
                    GcrOrthog::Classical => {
                        if !ap_basis.is_empty() {
                            let mut pairs: Vec<(&[S], &[S])> = Vec::with_capacity(ap_basis.len());
                            for api in &ap_basis {
                                pairs.push((&api[..], &v[..n]));
                            }
                            let mut numer = vec![S::zero(); ap_basis.len()];
                            red.dot_many_into(&pairs, &mut numer);
                            sync_count += 1;

                            let mut denom_pairs: Vec<(&[S], &[S])> =
                                Vec::with_capacity(ap_basis.len());
                            for api in &ap_basis {
                                denom_pairs.push((&api[..], &api[..]));
                            }
                            let mut denoms = vec![S::zero(); ap_basis.len()];
                            red.dot_many_into(&denom_pairs, &mut denoms);
                            sync_count += 1;

                            for i in 0..ap_basis.len() {
                                let denom = denoms[i];
                                if denom.abs() <= 1e-30 {
                                    continue;
                                }
                                let beta = numer[i] / denom;
                                for k in 0..n {
                                    ws.tmp2[k] -= beta * p_basis[i][k];
                                    v[k] -= beta * ap_basis[i][k];
                                }
                            }
                        }
                    }
                    GcrOrthog::Modified => {
                        for i in 0..ap_basis.len() {
                            let pair = [
                                (&ap_basis[i][..], &v[..n]),
                                (&ap_basis[i][..], &ap_basis[i][..]),
                            ];
                            let mut vals = [S::zero(); 2];
                            red.dot_many_into(&pair, &mut vals);
                            sync_count += 1;
                            if vals[1].abs() <= 1e-30 {
                                continue;
                            }
                            let beta = vals[0] / vals[1];
                            for k in 0..n {
                                ws.tmp2[k] -= beta * p_basis[i][k];
                                v[k] -= beta * ap_basis[i][k];
                            }
                        }
                    }
                }

                let pair = [(&ws.tmp1[..n], &v[..n]), (&v[..n], &v[..n])];
                let mut vals = [S::zero(); 2];
                red.dot_many_into(&pair, &mut vals);
                sync_count += 1;
                if vals[1].abs() <= 1e-30 {
                    let counters = SolverCounters {
                        num_global_reductions: sync_count,
                        ..SolverCounters::default()
                    };
                    return Ok(
                        SolveStats::new(iters, rnorm, ConvergedReason::DivergedBreakdown)
                            .with_counters(counters)
                            .with_reduction_model(self.reduction_model())
                            .with_gcr_counters(GcrCounters {
                                basis_updates,
                                sync_count,
                                restart_count,
                                restarted: restart_count > 0,
                            }),
                    );
                }
                let alpha = vals[0] / vals[1];

                for i in 0..n {
                    x[i] += alpha * ws.tmp2[i];
                    ws.tmp1[i] -= alpha * v[i];
                }

                p_basis.push(ws.tmp2[..n].to_vec());
                ap_basis.push(v[..n].to_vec());
                basis_updates += 1;
                iters += 1;

                rnorm = red.norm2(&ws.tmp1[..n]);
                sync_count += 1;
                if call_monitors(mons, iters, rnorm, sync_count) {
                    let counters = SolverCounters {
                        num_global_reductions: sync_count,
                        ..SolverCounters::default()
                    };
                    return Ok(
                        SolveStats::new(iters, rnorm, ConvergedReason::StoppedByMonitor)
                            .with_counters(counters)
                            .with_reduction_model(self.reduction_model())
                            .with_gcr_counters(GcrCounters {
                                basis_updates,
                                sync_count,
                                restart_count,
                                restarted: restart_count > 0,
                            }),
                    );
                }

                if rnorm <= threshold {
                    let true_res = recompute_true_residual_norm_s(
                        a,
                        b,
                        x,
                        comm,
                        red.engine(),
                        &mut true_res_buf[..n],
                        &mut ws.bridge,
                    );
                    let reason = if true_res <= self.atol {
                        ConvergedReason::ConvergedAtol
                    } else {
                        ConvergedReason::ConvergedRtol
                    };
                    let counters = SolverCounters {
                        num_global_reductions: sync_count,
                        ..SolverCounters::default()
                    };
                    return Ok(SolveStats::new(iters, true_res, reason)
                        .with_counters(counters)
                        .with_reduction_model(self.reduction_model())
                        .with_gcr_counters(GcrCounters {
                            basis_updates,
                            sync_count,
                            restart_count,
                            restarted: restart_count > 0,
                        }));
                }

                if rnorm >= self.dtol * bnorm {
                    let counters = SolverCounters {
                        num_global_reductions: sync_count,
                        ..SolverCounters::default()
                    };
                    return Ok(SolveStats::new(iters, rnorm, ConvergedReason::DivergedDtol)
                        .with_counters(counters)
                        .with_reduction_model(self.reduction_model())
                        .with_gcr_counters(GcrCounters {
                            basis_updates,
                            sync_count,
                            restart_count,
                            restarted: restart_count > 0,
                        }));
                }
            }
        }

        let final_res = recompute_true_residual_norm_s(
            a,
            b,
            x,
            comm,
            red.engine(),
            &mut true_res_buf[..n],
            &mut ws.bridge,
        );
        let counters = SolverCounters {
            num_global_reductions: sync_count,
            ..SolverCounters::default()
        };
        Ok(
            SolveStats::new(iters, final_res, ConvergedReason::DivergedMaxIts)
                .with_counters(counters)
                .with_reduction_model(self.reduction_model())
                .with_gcr_counters(GcrCounters {
                    basis_updates,
                    sync_count,
                    restart_count,
                    restarted: restart_count > 0,
                }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_f64(
        &mut self,
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<MonitorCallback<f64>>]>,
        work: Option<&mut Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput(
                "PipeGCR requires a square operator".to_string(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "PipeGCR: vector size mismatch".to_string(),
            ));
        }

        let a_s = as_s_op(a);
        match pc {
            Some(pc_ref) => {
                let mut pc_s = as_s_pc_mut(pc_ref);
                self.solve_k(&a_s, Some(&mut pc_s), b, x, pc_side, comm, monitors, work)
            }
            None => self.solve_k(&a_s, None, b, x, pc_side, comm, monitors, work),
        }
    }
}

impl LinearSolver for PipeGcrSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn setup_workspace(&mut self, _work: &mut Workspace) {}

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
        self.solve_f64(a, pc, b, x, pc_side, comm, monitors, work)
    }
}
