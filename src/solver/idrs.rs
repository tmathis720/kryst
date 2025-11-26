#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::op::{LinOp, LinOpF64};
use crate::ops::klinop::KLinOp;
use crate::ops::kpc::KPreconditioner;
use crate::ops::wrap::{as_s_op, as_s_pc};
use crate::parallel::{
    UniverseComm, allreduce_sum_scalar_with_mode, global_dot_conj, global_dot_conj_many_into,
    global_nrm2, global_reduction_mode,
};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::solver::common::dot_result_to_real;
use crate::utils::convergence::{ConvergedReason, SolveStats, SolverCounters};
#[cfg(feature = "backend-faer")]
use faer::Mat;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::cmp::min;

fn reduce_real(comm: &UniverseComm, value: R) -> R {
    dot_result_to_real(allreduce_sum_scalar_with_mode(
        comm,
        S::from_real(value),
        global_reduction_mode(),
    ))
}

#[derive(Clone, Debug)]
pub struct IdrsOptions {
    pub s: usize,
    pub tol: f64,
    pub maxit: usize,
    pub omega_strategy: Omega,
    pub p_policy: ShadowP,
    pub breakdown_repair: BreakdownRepair,
    pub monitor_true_residual_every: Option<usize>,
}

impl Default for IdrsOptions {
    fn default() -> Self {
        Self {
            s: 4,
            tol: 1e-8,
            maxit: 10_000,
            omega_strategy: Omega::MinResidual,
            p_policy: ShadowP::RandomOrthonormal { seed: 0xdecafbad },
            breakdown_repair: BreakdownRepair::RegenerateP {
                max_retries: 1,
                seed: 0x1234_5678,
            },
            monitor_true_residual_every: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub enum Omega {
    #[default]
    MinResidual,
    MinResidualClipped {
        cos_min: f64,
        kappa: f64,
    },
}

#[derive(Clone, Debug)]
pub enum ShadowP {
    RandomOrthonormal {
        seed: u64,
    },
    BlockDeflation {
        partition: Vec<usize>,
    },
    #[cfg(feature = "backend-faer")]
    FromVectors {
        p: Mat<f64>,
    },
}

impl Default for ShadowP {
    fn default() -> Self {
        ShadowP::RandomOrthonormal { seed: 0xdecafbad }
    }
}

#[derive(Clone, Debug)]
pub enum BreakdownRepair {
    None,
    RegenerateP { max_retries: usize, seed: u64 },
}

impl Default for BreakdownRepair {
    fn default() -> Self {
        BreakdownRepair::RegenerateP {
            max_retries: 1,
            seed: 0x1234_5678,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IdrsStats {
    pub iters: usize,
    pub matvecs: usize,
    pub dots: usize,
    pub residual_replacements: usize,
}

#[derive(Default)]
pub struct IdrsBuilder {
    opts: IdrsOptions,
}

impl IdrsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn s(mut self, s: usize) -> Self {
        self.opts.s = s.max(1);
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.opts.tol = tol;
        self
    }

    pub fn maxit(mut self, maxit: usize) -> Self {
        self.opts.maxit = maxit;
        self
    }

    pub fn omega_strategy(mut self, omega: Omega) -> Self {
        self.opts.omega_strategy = omega;
        self
    }

    pub fn shadow_policy(mut self, policy: ShadowP) -> Self {
        self.opts.p_policy = policy;
        self
    }

    pub fn breakdown_repair(mut self, repair: BreakdownRepair) -> Self {
        self.opts.breakdown_repair = repair;
        self
    }

    pub fn monitor_true_residual_every(mut self, every: Option<usize>) -> Self {
        self.opts.monitor_true_residual_every = every;
        self
    }

    pub fn build(self) -> IdrsSolver {
        IdrsSolver::with_options(self.opts)
    }
}

#[derive(Default)]
struct IdrsWorkspace {
    n: usize,
    s: usize,
    p: Vec<S>,
    ph_r: Vec<S>,
    ph_drn: Vec<S>,
    c: Vec<S>,
    d_r: Vec<Vec<S>>,
    d_r_raw: Vec<Vec<S>>,
    d_x: Vec<Vec<S>>,
    g_hist: Vec<Vec<S>>,
    g_hist_raw: Vec<Vec<S>>,
    g_hist_x: Vec<Vec<S>>,
    r: Vec<S>,
    r_true: Vec<S>,
    v: Vec<S>,
    t: Vec<S>,
    t_raw: Vec<S>,
    scratch: BridgeScratch,
}

impl IdrsWorkspace {
    fn ensure(&mut self, n: usize, s: usize) {
        if self.n != n || self.s != s {
            self.n = n;
            self.s = s;
            self.p.resize(n.saturating_mul(s), S::zero());
            self.ph_r.resize(s, S::zero());
            self.ph_drn.resize(s.saturating_mul(s), S::zero());
            self.c.resize(s, S::zero());
            self.d_r.resize(s + 1, vec![S::zero(); n]);
            self.d_r_raw.resize(s + 1, vec![S::zero(); n]);
            self.d_x.resize(s + 1, vec![S::zero(); n]);
            self.g_hist.resize(s, vec![S::zero(); n]);
            self.g_hist_raw.resize(s, vec![S::zero(); n]);
            self.g_hist_x.resize(s, vec![S::zero(); n]);
            self.r.resize(n, S::zero());
            self.r_true.resize(n, S::zero());
            self.v.resize(n, S::zero());
            self.t.resize(n, S::zero());
            self.t_raw.resize(n, S::zero());
            self.scratch = BridgeScratch::default();
        } else {
            let need = s + 1;
            if self.d_r.len() != need {
                self.d_r.resize_with(need, || vec![S::zero(); n]);
            }
            if self.d_r_raw.len() != need {
                self.d_r_raw.resize_with(need, || vec![S::zero(); n]);
            }
            if self.d_x.len() != need {
                self.d_x.resize_with(need, || vec![S::zero(); n]);
            }
            if self.g_hist.len() != s {
                self.g_hist.resize_with(s, || vec![S::zero(); n]);
            }
            if self.g_hist_raw.len() != s {
                self.g_hist_raw.resize_with(s, || vec![S::zero(); n]);
            }
            if self.g_hist_x.len() != s {
                self.g_hist_x.resize_with(s, || vec![S::zero(); n]);
            }
            for buf in &mut self.d_r {
                if buf.len() != n {
                    buf.resize(n, S::zero());
                }
            }
            for buf in &mut self.d_r_raw {
                if buf.len() != n {
                    buf.resize(n, S::zero());
                }
            }
            for buf in &mut self.d_x {
                if buf.len() != n {
                    buf.resize(n, S::zero());
                }
            }
            for buf in &mut self.g_hist {
                if buf.len() != n {
                    buf.resize(n, S::zero());
                }
            }
            for buf in &mut self.g_hist_raw {
                if buf.len() != n {
                    buf.resize(n, S::zero());
                }
            }
            for buf in &mut self.g_hist_x {
                if buf.len() != n {
                    buf.resize(n, S::zero());
                }
            }
            self.p.resize(n.saturating_mul(s), S::zero());
            self.ph_r.resize(s, S::zero());
            self.ph_drn.resize(s.saturating_mul(s), S::zero());
            self.c.resize(s, S::zero());
            self.r.resize(n, S::zero());
            self.r_true.resize(n, S::zero());
            self.v.resize(n, S::zero());
            self.t.resize(n, S::zero());
            self.t_raw.resize(n, S::zero());
        }
    }

    #[inline]
    fn p_col(&self, j: usize) -> &[S] {
        let n = self.n;
        &self.p[j * n..(j + 1) * n]
    }

    #[inline]
    fn p_col_mut(&mut self, j: usize) -> &mut [S] {
        let n = self.n;
        &mut self.p[j * n..(j + 1) * n]
    }

    fn push_history_from_buffers(&mut self) {
        if self.s == 0 {
            return;
        }
        self.g_hist.rotate_right(1);
        self.g_hist_raw.rotate_right(1);
        self.g_hist_x.rotate_right(1);
        self.g_hist[0].clone_from_slice(&self.d_r[0]);
        self.g_hist_raw[0].clone_from_slice(&self.d_r_raw[0]);
        self.g_hist_x[0].clone_from_slice(&self.d_x[0]);
    }

    fn normalize_column(
        &mut self,
        col_idx: usize,
        comm: &UniverseComm,
        stats: &mut IdrsStats,
    ) -> Result<(), KError> {
        let col = self.p_col_mut(col_idx);
        let local = col
            .iter()
            .fold(R::default(), |acc, &val| acc + val.abs() * val.abs());
        let norm_sq = reduce_real(comm, local);
        stats.dots += 1;
        let norm = norm_sq.sqrt();
        if norm <= f64::EPSILON {
            return Err(KError::BreakdownOrIndefinite);
        }
        let inv = S::from_real(1.0 / norm);
        for val in col.iter_mut() {
            *val *= inv;
        }
        Ok(())
    }

    fn orthonormalize_column(
        &mut self,
        col_idx: usize,
        comm: &UniverseComm,
        stats: &mut IdrsStats,
    ) -> Result<(), KError> {
        let n = self.n;
        {
            let (prev_cols, tail) = self.p.split_at_mut(col_idx * n);
            let (col, _) = tail.split_at_mut(n);
            for k in 0..col_idx {
                let prev = &prev_cols[k * n..(k + 1) * n];
                let coeff = global_dot_conj(comm, prev, col);
                stats.dots += 1;
                for (entry, &prev_val) in col.iter_mut().zip(prev.iter()) {
                    *entry -= coeff * prev_val;
                }
            }
        }
        self.normalize_column(col_idx, comm, stats)
    }
}

pub struct IdrsSolver {
    opts: IdrsOptions,
    ws: IdrsWorkspace,
    random_bump: u64,
}

impl Default for IdrsSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl IdrsSolver {
    pub fn new() -> Self {
        Self::with_options(IdrsOptions::default())
    }

    pub fn with_options(opts: IdrsOptions) -> Self {
        Self {
            opts,
            ws: IdrsWorkspace::default(),
            random_bump: 0,
        }
    }

    fn build_shadow_space(
        &mut self,
        comm: &UniverseComm,
        stats: &mut IdrsStats,
    ) -> Result<(), KError> {
        match &self.opts.p_policy {
            ShadowP::RandomOrthonormal { seed } => {
                let actual_seed = seed.wrapping_add(self.random_bump);
                let mut rng = StdRng::seed_from_u64(actual_seed);
                for j in 0..self.ws.s {
                    {
                        let col = self.ws.p_col_mut(j);
                        for val in col.iter_mut() {
                            let sample = StandardNormal.sample(&mut rng);
                            *val = S::from_real(sample);
                        }
                    }
                    self.ws.orthonormalize_column(j, comm, stats)?;
                }
            }
            ShadowP::BlockDeflation { partition } => {
                let n = self.ws.n;
                if partition.len() != n {
                    return Err(KError::InvalidInput(
                        "IDR(s): block partition must match problem size".into(),
                    ));
                }
                let mut unique = partition.clone();
                unique.sort_unstable();
                unique.dedup();
                if unique.len() != self.ws.s {
                    return Err(KError::InvalidInput(
                        "IDR(s): block partition cardinality must equal s".into(),
                    ));
                }
                for (col_idx, &blk) in unique.iter().enumerate() {
                    {
                        let col = self.ws.p_col_mut(col_idx);
                        col.fill(S::zero());
                        let mut count = 0usize;
                        for (i, &part) in partition.iter().enumerate() {
                            if part == blk {
                                col[i] = S::one();
                                count += 1;
                            }
                        }
                        if count == 0 {
                            return Err(KError::InvalidInput(
                                "IDR(s): block partition contained empty block".into(),
                            ));
                        }
                    }
                    self.ws.normalize_column(col_idx, comm, stats)?;
                }
            }
            #[cfg(feature = "backend-faer")]
            ShadowP::FromVectors { p } => {
                if p.nrows() != self.ws.n || p.ncols() != self.ws.s {
                    return Err(KError::InvalidInput(
                        "IDR(s): provided shadow space has wrong dimensions".into(),
                    ));
                }
                for j in 0..self.ws.s {
                    let n = self.ws.n;
                    {
                        let dst = self.ws.p_col_mut(j);
                        for i in 0..n {
                            dst[i] = S::from_real(p[(i, j)]);
                        }
                    }
                    self.ws.normalize_column(j, comm, stats)?;
                }
            }
        }
        Ok(())
    }

    fn attempt_breakdown_repair(
        &mut self,
        attempts: &mut usize,
        comm: &UniverseComm,
        stats: &mut IdrsStats,
    ) -> Result<bool, KError> {
        if let BreakdownRepair::RegenerateP { max_retries, seed } = self.opts.breakdown_repair {
            if *attempts >= max_retries {
                return Ok(false);
            }
            *attempts += 1;
            if matches!(self.opts.p_policy, ShadowP::RandomOrthonormal { .. }) {
                self.random_bump = self.random_bump.wrapping_add(1);
                self.build_shadow_space(comm, stats)?;
            } else {
                let saved = self.opts.p_policy.clone();
                self.opts.p_policy = ShadowP::RandomOrthonormal {
                    seed: seed.wrapping_add(*attempts as u64),
                };
                self.random_bump = 0;
                self.build_shadow_space(comm, stats)?;
                self.opts.p_policy = saved;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn compute_ph_r(&mut self, comm: &UniverseComm, stats: &mut IdrsStats) {
        let n = self.ws.n;
        let s = self.ws.s;
        for j in 0..s {
            let col = self.ws.p_col(j);
            let dot = global_dot_conj(comm, col, &self.ws.r[..n]);
            self.ws.ph_r[j] = dot;
        }
        stats.dots += s;
    }

    fn compute_ph_drn(&mut self, comm: &UniverseComm, stats: &mut IdrsStats) {
        let n = self.ws.n;
        let s = self.ws.s;
        for i in 0..s {
            let vec = &self.ws.g_hist[i];
            for j in 0..s {
                let col = self.ws.p_col(j);
                let dot = global_dot_conj(comm, col, &vec[..n]);
                self.ws.ph_drn[i * s + j] = dot;
            }
        }
        stats.dots += s * s;
    }

    fn solve_small_system(&mut self) -> Result<(), ()> {
        let s = self.ws.s;
        if s == 0 {
            return Ok(());
        }
        let mut a = self.ws.ph_drn.clone();
        let mut b = self.ws.ph_r.clone();
        self.ws.c.fill(S::zero());
        for k in 0..s {
            let mut pivot_row = k;
            let mut pivot = a[k * s + k].abs();
            for i in (k + 1)..s {
                let val = a[i * s + k].abs();
                if val > pivot {
                    pivot = val;
                    pivot_row = i;
                }
            }
            if pivot <= 1e-14 {
                return Err(());
            }
            if pivot_row != k {
                for j in k..s {
                    a.swap(k * s + j, pivot_row * s + j);
                }
                b.swap(k, pivot_row);
            }
            let diag = a[k * s + k];
            let row_k = a[k * s..(k + 1) * s].to_vec();
            let pivot_rhs = b[k];
            for i in (k + 1)..s {
                let row_i = &mut a[i * s..(i + 1) * s];
                let factor = if diag.abs() <= R::default() {
                    S::zero()
                } else {
                    row_i[k] / diag
                };
                if factor.abs() > R::default() {
                    for j in k..s {
                        row_i[j] -= factor * row_k[j];
                    }
                    b[i] -= factor * pivot_rhs;
                }
            }
        }
        for i in (0..s).rev() {
            let mut sum = b[i];
            for j in (i + 1)..s {
                sum -= a[i * s + j] * self.ws.c[j];
            }
            let diag = a[i * s + i];
            if diag.abs() <= 1e-14 {
                return Err(());
            }
            self.ws.c[i] = sum / diag;
        }
        Ok(())
    }

    fn combine_delta(dst: &mut [S], coeffs: &[S], src: &[Vec<S>], scale: S) {
        let n = dst.len();
        dst.fill(S::zero());
        for (col, &coeff) in src.iter().zip(coeffs.iter()) {
            if coeff.abs() <= R::default() {
                continue;
            }
            for i in 0..n {
                dst[i] += coeff * col[i];
            }
        }
        if scale != S::one() {
            for val in dst.iter_mut() {
                *val *= scale;
            }
        }
    }

    fn apply_matvec<A: KLinOp<Scalar = S> + ?Sized>(
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        x: &[S],
        raw: &mut [S],
        precond: &mut [S],
        scratch: &mut BridgeScratch,
        stats: &mut IdrsStats,
    ) -> Result<(), KError> {
        a.matvec_s(x, raw, scratch);
        stats.matvecs += 1;
        if let Some(pc_ref) = pc {
            pc_ref.apply_s(PcSide::Left, raw, precond, scratch)?;
        } else {
            precond.copy_from_slice(raw);
        }
        Ok(())
    }

    fn omega_value(&self, comm: &UniverseComm, stats: &mut IdrsStats, t: &[S], v: &[S]) -> S {
        let mut reductions = [S::zero(); 2];
        global_dot_conj_many_into(comm, &[(t, v), (t, t)], &mut reductions);
        stats.dots += reductions.len();
        let tv = reductions[0];
        let tt = reductions[1];
        let tt_real = tt.real();
        let mut omega = if tt_real.abs() <= f64::EPSILON {
            S::zero()
        } else {
            tv / S::from_real(tt_real)
        };
        if let Omega::MinResidualClipped { cos_min, kappa } = self.opts.omega_strategy {
            let local_vv = v
                .iter()
                .fold(R::default(), |acc, &vi| acc + vi.abs() * vi.abs());
            let vv = reduce_real(comm, local_vv);
            stats.dots += 1;
            let denom = (tt_real * vv).sqrt();
            if denom > 0.0 {
                let cos = if denom > 0.0 { tv.real() / denom } else { 0.0 };
                if cos.abs() < cos_min {
                    let sign = if tv.real() >= 0.0 { 1.0 } else { -1.0 };
                    let target = cos_min * denom / tt_real.max(1e-32);
                    omega = S::from_real(kappa) * omega
                        + S::from_real(1.0 - kappa) * S::from_real(sign * target);
                }
            }
        }
        omega
    }

    fn monitor(&self, monitors: &[Box<dyn Fn(usize, f64) + Send + Sync>], iter: usize, res: f64) {
        if monitors.is_empty() {
            return;
        }
        for m in monitors {
            m(iter, res);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_internal<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput(
                "IDR(s) requires square operator".to_string(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "IDR(s): vector length mismatch".to_string(),
            ));
        }
        if !matches!(pc_side, PcSide::Left) {
            return Err(KError::InvalidInput(
                "IDR(s) currently supports only left preconditioning".to_string(),
            ));
        }
        if self.opts.s == 0 {
            return Err(KError::InvalidInput("IDR(s): s must be >= 1".to_string()));
        }

        self.ws.ensure(n, self.opts.s);
        let mut stats = IdrsStats::default();

        let monitors = monitors.unwrap_or(&[]);

        self.random_bump = 0;
        let mut breakdown_attempts = 0usize;

        if x.iter().all(|&xi| xi.abs() <= R::default()) {
            self.ws.r_true[..n].copy_from_slice(b);
        } else {
            a.matvec_s(x, &mut self.ws.t_raw[..n], &mut self.ws.scratch);
            stats.matvecs += 1;
            for i in 0..n {
                self.ws.r_true[i] = b[i] - self.ws.t_raw[i];
            }
        }
        if let Some(pc_ref) = pc {
            pc_ref.apply_s(
                PcSide::Left,
                &self.ws.r_true[..n],
                &mut self.ws.r[..n],
                &mut self.ws.scratch,
            )?;
        } else {
            self.ws.r[..n].copy_from_slice(&self.ws.r_true[..n]);
        }

        let bnorm = global_nrm2(comm, &b[..n]);
        stats.dots += 1;
        let norm_scale = if bnorm > R::default() {
            bnorm
        } else {
            S::one().real()
        };
        let mut res_norm = global_nrm2(comm, &self.ws.r_true[..n]);
        stats.dots += 1;
        self.monitor(monitors, 0, res_norm);
        if res_norm <= self.opts.tol * norm_scale {
            let mut out = SolveStats::new(0, res_norm, ConvergedReason::ConvergedRtol);
            out.counters = SolverCounters {
                num_global_reductions: stats.dots,
                residual_replacements: stats.residual_replacements,
            };
            return Ok(out);
        }

        self.build_shadow_space(comm, &mut stats)?;

        for buf in &mut self.ws.d_r {
            buf.fill(S::zero());
        }
        for buf in &mut self.ws.d_r_raw {
            buf.fill(S::zero());
        }
        for buf in &mut self.ws.d_x {
            buf.fill(S::zero());
        }

        for step in 0..min(self.opts.s, self.opts.maxit) {
            let omega = loop {
                Self::apply_matvec(
                    a,
                    pc,
                    &self.ws.r[..n],
                    &mut self.ws.t_raw[..n],
                    &mut self.ws.v[..n],
                    &mut self.ws.scratch,
                    &mut stats,
                )?;
                let mut reductions = [S::zero(); 2];
                let dot_pairs = [
                    (&self.ws.v[..n], &self.ws.r[..n]),
                    (&self.ws.v[..n], &self.ws.v[..n]),
                ];
                global_dot_conj_many_into(comm, &dot_pairs, &mut reductions);
                stats.dots += reductions.len();
                let vr = reductions[0];
                let vv = reductions[1];
                if vv.abs() <= f64::EPSILON {
                    if self.attempt_breakdown_repair(&mut breakdown_attempts, comm, &mut stats)? {
                        continue;
                    }
                    return Err(KError::BreakdownOrIndefinite);
                }
                let omega = vr / vv;
                if omega.abs() <= f64::EPSILON {
                    if self.attempt_breakdown_repair(&mut breakdown_attempts, comm, &mut stats)? {
                        continue;
                    }
                    return Err(KError::BreakdownOrIndefinite);
                }
                break omega;
            };

            self.ws.d_r.rotate_right(1);
            self.ws.d_r_raw.rotate_right(1);
            self.ws.d_x.rotate_right(1);
            let newest_r = &mut self.ws.d_r[0];
            let newest_r_raw = &mut self.ws.d_r_raw[0];
            let newest_x = &mut self.ws.d_x[0];
            for i in 0..n {
                newest_x[i] = omega * self.ws.r[i];
                newest_r[i] = -omega * self.ws.v[i];
                newest_r_raw[i] = -omega * self.ws.t_raw[i];
            }
            for i in 0..n {
                x[i] += newest_x[i];
                self.ws.r[i] += newest_r[i];
                self.ws.r_true[i] += newest_r_raw[i];
            }

            self.ws.push_history_from_buffers();

            res_norm = global_nrm2(comm, &self.ws.r_true[..n]);
            stats.dots += 1;
            self.monitor(monitors, step + 1, res_norm);
            if res_norm <= self.opts.tol * norm_scale {
                let mut out = SolveStats::new(step + 1, res_norm, ConvergedReason::ConvergedRtol);
                out.counters = SolverCounters {
                    num_global_reductions: stats.dots,
                    residual_replacements: stats.residual_replacements,
                };
                return Ok(out);
            }
        }

        let mut attempts = breakdown_attempts;
        let mut iteration = min(self.opts.s, self.opts.maxit);
        let mut omega_block = S::zero();
        while iteration < self.opts.maxit {
            'block: for inner in 0..=self.opts.s {
                loop {
                    self.compute_ph_r(comm, &mut stats);
                    self.compute_ph_drn(comm, &mut stats);
                    if self.solve_small_system().is_err() {
                        if self.attempt_breakdown_repair(&mut attempts, comm, &mut stats)? {
                            continue 'block;
                        }
                        return Err(KError::BreakdownOrIndefinite);
                    }

                    let src_r = &self.ws.g_hist[..self.opts.s];
                    Self::combine_delta(&mut self.ws.v[..n], &self.ws.c, src_r, -S::one());
                    for i in 0..n {
                        self.ws.v[i] += self.ws.r[i];
                    }

                    if inner == 0 {
                        loop {
                            Self::apply_matvec(
                                a,
                                pc,
                                &self.ws.v[..n],
                                &mut self.ws.t_raw[..n],
                                &mut self.ws.t[..n],
                                &mut self.ws.scratch,
                                &mut stats,
                            )?;
                            omega_block = self.omega_value(
                                comm,
                                &mut stats,
                                &self.ws.t[..n],
                                &self.ws.v[..n],
                            );
                            if omega_block.abs() <= f64::EPSILON {
                                if self.attempt_breakdown_repair(&mut attempts, comm, &mut stats)? {
                                    continue 'block;
                                }
                                return Err(KError::BreakdownOrIndefinite);
                            }

                            self.ws.d_r.rotate_right(1);
                            self.ws.d_r_raw.rotate_right(1);
                            self.ws.d_x.rotate_right(1);
                            let (newest_x, _rest_x) =
                                self.ws.d_x.split_first_mut().expect("nonempty");
                            let (newest_r, _rest_r) =
                                self.ws.d_r.split_first_mut().expect("nonempty");
                            let (newest_r_raw, _rest_rr) =
                                self.ws.d_r_raw.split_first_mut().expect("nonempty");
                            let src_x = &self.ws.g_hist_x[..self.opts.s];
                            Self::combine_delta(newest_x, &self.ws.c, src_x, -S::one());
                            for i in 0..n {
                                newest_x[i] += omega_block * self.ws.v[i];
                            }
                            let src_r = &self.ws.g_hist[..self.opts.s];
                            Self::combine_delta(newest_r, &self.ws.c, src_r, -S::one());
                            for i in 0..n {
                                newest_r[i] -= omega_block * self.ws.t[i];
                            }
                            let src_rr = &self.ws.g_hist_raw[..self.opts.s];
                            Self::combine_delta(newest_r_raw, &self.ws.c, src_rr, -S::one());
                            for i in 0..n {
                                newest_r_raw[i] -= omega_block * self.ws.t_raw[i];
                            }
                            break;
                        }
                    } else {
                        self.ws.d_x.rotate_right(1);
                        self.ws.d_r.rotate_right(1);
                        self.ws.d_r_raw.rotate_right(1);
                        let (newest_x, _rest_x) = self.ws.d_x.split_first_mut().expect("nonempty");
                        let (newest_r, _rest_r) = self.ws.d_r.split_first_mut().expect("nonempty");
                        let (newest_r_raw, _rest_rr) =
                            self.ws.d_r_raw.split_first_mut().expect("nonempty");
                        let src_x = &self.ws.g_hist_x[..self.opts.s];
                        Self::combine_delta(newest_x, &self.ws.c, src_x, -S::one());
                        for i in 0..n {
                            newest_x[i] += omega_block * self.ws.v[i];
                        }
                        Self::apply_matvec(
                            a,
                            pc,
                            &*newest_x,
                            &mut self.ws.t_raw[..n],
                            &mut self.ws.t[..n],
                            &mut self.ws.scratch,
                            &mut stats,
                        )?;
                        for i in 0..n {
                            newest_r[i] = -self.ws.t[i];
                            newest_r_raw[i] = -self.ws.t_raw[i];
                        }
                    }

                    let newest_x = &self.ws.d_x[0];
                    let newest_r = &self.ws.d_r[0];
                    let newest_r_raw = &self.ws.d_r_raw[0];
                    for i in 0..n {
                        x[i] += newest_x[i];
                        self.ws.r[i] += newest_r[i];
                        self.ws.r_true[i] += newest_r_raw[i];
                    }

                    self.ws.push_history_from_buffers();

                    iteration += 1;

                    if let Some(freq) = self.opts.monitor_true_residual_every
                        && iteration % freq == 0
                    {
                        a.matvec_s(x, &mut self.ws.t_raw[..n], &mut self.ws.scratch);
                        stats.matvecs += 1;
                        for i in 0..n {
                            self.ws.r_true[i] = b[i] - self.ws.t_raw[i];
                        }
                        if let Some(pc_ref) = pc {
                            pc_ref.apply_s(
                                PcSide::Left,
                                &self.ws.r_true[..n],
                                &mut self.ws.r[..n],
                                &mut self.ws.scratch,
                            )?;
                        } else {
                            self.ws.r[..n].copy_from_slice(&self.ws.r_true[..n]);
                        }
                        stats.residual_replacements += 1;
                    }

                    res_norm = global_nrm2(comm, &self.ws.r_true[..n]);
                    stats.dots += 1;
                    self.monitor(monitors, iteration, res_norm);
                    if res_norm <= self.opts.tol * norm_scale {
                        let mut out =
                            SolveStats::new(iteration, res_norm, ConvergedReason::ConvergedRtol);
                        out.counters = SolverCounters {
                            num_global_reductions: stats.dots,
                            residual_replacements: stats.residual_replacements,
                        };
                        return Ok(out);
                    }

                    if iteration >= self.opts.maxit {
                        break 'block;
                    }

                    break;
                }
                if iteration >= self.opts.maxit {
                    break;
                }
            }
            if iteration >= self.opts.maxit {
                break;
            }
        }

        let mut out = SolveStats::new(iteration, res_norm, ConvergedReason::DivergedMaxIts);
        out.counters = SolverCounters {
            num_global_reductions: stats.dots,
            residual_replacements: stats.residual_replacements,
        };
        Ok(out)
    }

    pub fn solve_k<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn KPreconditioner<Scalar = S>>,
        b: &[S],
        x: &mut [S],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, R) + Send + Sync>]>,
    ) -> Result<SolveStats<R>, KError>
    where
        A: KLinOp<Scalar = S> + ?Sized,
    {
        self.solve_internal(a, pc, b, x, pc_side, comm, monitors)
    }

    pub fn solve_f64<A>(
        &mut self,
        a: &A,
        pc: Option<&dyn Preconditioner>,
        b: &[f64],
        x: &mut [f64],
        pc_side: PcSide,
        comm: &UniverseComm,
        monitors: Option<&[Box<dyn Fn(usize, f64) + Send + Sync>]>,
    ) -> Result<SolveStats<f64>, KError>
    where
        A: LinOpF64 + LinOp<S = f64> + Send + Sync + ?Sized,
    {
        let op = as_s_op(a);
        let pc_wrapper = pc.map(as_s_pc);
        let pc_ref = pc_wrapper
            .as_ref()
            .map(|p| p as &dyn KPreconditioner<Scalar = S>);

        #[cfg(not(feature = "complex"))]
        {
            let b_s: &[S] = unsafe { &*(b as *const [f64] as *const [S]) };
            let x_s: &mut [S] = unsafe { &mut *(x as *mut [f64] as *mut [S]) };
            self.solve_internal(&op, pc_ref, b_s, x_s, pc_side, comm, monitors)
        }

        #[cfg(feature = "complex")]
        {
            let b_s: Vec<S> = b.iter().copied().map(S::from_real).collect();
            let mut x_s: Vec<S> = x.iter().copied().map(S::from_real).collect();
            let result = self.solve_internal(&op, pc_ref, &b_s, &mut x_s, pc_side, comm, monitors);
            if result.is_ok() {
                for (dst, src) in x.iter_mut().zip(x_s.iter()) {
                    *dst = src.real();
                }
            }
            result
        }
    }
}

impl LinearSolver for IdrsSolver {
    type Error = KError;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn setup_workspace(&mut self, _work: &mut crate::context::ksp_context::Workspace) {}

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
        _work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<f64>, Self::Error> {
        self.solve_f64(a, pc.as_deref(), b, x, pc_side, comm, monitors)
    }
}
