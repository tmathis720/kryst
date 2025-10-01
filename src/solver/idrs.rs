use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::parallel::{Comm, UniverseComm};
use crate::preconditioner::{PcSide, Preconditioner};
use crate::solver::LinearSolver;
use crate::utils::convergence::{ConvergedReason, SolveStats, SolverCounters};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::cmp::min;

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

#[derive(Clone, Debug)]
pub enum Omega {
    MinResidual,
    MinResidualClipped { cos_min: f64, kappa: f64 },
}

impl Default for Omega {
    fn default() -> Self {
        Omega::MinResidual
    }
}

#[derive(Clone, Debug)]
pub enum ShadowP {
    RandomOrthonormal { seed: u64 },
    BlockDeflation { partition: Vec<usize> },
    FromVectors { p: faer::Mat<f64> },
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

pub struct IdrsBuilder {
    opts: IdrsOptions,
}

impl Default for IdrsBuilder {
    fn default() -> Self {
        Self {
            opts: IdrsOptions::default(),
        }
    }
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
    p: Vec<f64>,
    ph_r: Vec<f64>,
    ph_drn: Vec<f64>,
    c: Vec<f64>,
    d_r: Vec<Vec<f64>>,
    d_r_raw: Vec<Vec<f64>>,
    d_x: Vec<Vec<f64>>,
    r: Vec<f64>,
    r_true: Vec<f64>,
    v: Vec<f64>,
    t: Vec<f64>,
    t_raw: Vec<f64>,
}

impl IdrsWorkspace {
    fn ensure(&mut self, n: usize, s: usize) {
        if self.n != n || self.s != s {
            self.n = n;
            self.s = s;
            self.p.resize(n.saturating_mul(s), 0.0);
            self.ph_r.resize(s, 0.0);
            self.ph_drn.resize(s.saturating_mul(s), 0.0);
            self.c.resize(s, 0.0);
            self.d_r.resize(s + 1, vec![0.0; n]);
            self.d_r_raw.resize(s + 1, vec![0.0; n]);
            self.d_x.resize(s + 1, vec![0.0; n]);
            self.r.resize(n, 0.0);
            self.r_true.resize(n, 0.0);
            self.v.resize(n, 0.0);
            self.t.resize(n, 0.0);
            self.t_raw.resize(n, 0.0);
        } else {
            let need = s + 1;
            if self.d_r.len() != need {
                self.d_r.resize_with(need, || vec![0.0; n]);
            }
            if self.d_r_raw.len() != need {
                self.d_r_raw.resize_with(need, || vec![0.0; n]);
            }
            if self.d_x.len() != need {
                self.d_x.resize_with(need, || vec![0.0; n]);
            }
            for buf in &mut self.d_r {
                if buf.len() != n {
                    buf.resize(n, 0.0);
                }
            }
            for buf in &mut self.d_r_raw {
                if buf.len() != n {
                    buf.resize(n, 0.0);
                }
            }
            for buf in &mut self.d_x {
                if buf.len() != n {
                    buf.resize(n, 0.0);
                }
            }
            self.p.resize(n.saturating_mul(s), 0.0);
            self.ph_r.resize(s, 0.0);
            self.ph_drn.resize(s.saturating_mul(s), 0.0);
            self.c.resize(s, 0.0);
            self.r.resize(n, 0.0);
            self.r_true.resize(n, 0.0);
            self.v.resize(n, 0.0);
            self.t.resize(n, 0.0);
            self.t_raw.resize(n, 0.0);
        }
    }

    #[inline]
    fn p_col(&self, j: usize) -> &[f64] {
        let n = self.n;
        &self.p[j * n..(j + 1) * n]
    }

    #[inline]
    fn p_col_mut(&mut self, j: usize) -> &mut [f64] {
        let n = self.n;
        &mut self.p[j * n..(j + 1) * n]
    }

    fn normalize_column(
        &mut self,
        col_idx: usize,
        comm: &UniverseComm,
        stats: &mut IdrsStats,
    ) -> Result<(), KError> {
        let col = self.p_col_mut(col_idx);
        let local = col.iter().map(|x| x * x).sum::<f64>();
        let norm_sq = comm.all_reduce_f64(local);
        stats.dots += 1;
        let norm = norm_sq.sqrt();
        if norm <= f64::EPSILON {
            return Err(KError::BreakdownOrIndefinite);
        }
        let inv = 1.0 / norm;
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
        for k in 0..col_idx {
            let mut local = 0.0;
            for i in 0..n {
                local += self.p[k * n + i] * self.p[col_idx * n + i];
            }
            let dot = comm.all_reduce_f64(local);
            stats.dots += 1;
            for i in 0..n {
                let idx = col_idx * n + i;
                self.p[idx] -= dot * self.p[k * n + i];
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
                            *val = StandardNormal.sample(&mut rng);
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
                        col.fill(0.0);
                        let mut count = 0usize;
                        for (i, &part) in partition.iter().enumerate() {
                            if part == blk {
                                col[i] = 1.0;
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
                            dst[i] = p[(i, j)];
                        }
                    }
                    self.ws.normalize_column(j, comm, stats)?;
                }
            }
        }
        Ok(())
    }

    fn compute_ph_r(&mut self, comm: &UniverseComm, stats: &mut IdrsStats) {
        let n = self.ws.n;
        let s = self.ws.s;
        for j in 0..s {
            let col = self.ws.p_col(j);
            let mut accum = 0.0;
            for i in 0..n {
                accum += col[i] * self.ws.r[i];
            }
            self.ws.ph_r[j] = accum;
        }
        comm.allreduce_sum_slice(&mut self.ws.ph_r);
        stats.dots += s;
    }

    fn compute_ph_drn(&mut self, comm: &UniverseComm, stats: &mut IdrsStats) {
        let n = self.ws.n;
        let s = self.ws.s;
        for i in 0..s {
            let vec = &self.ws.d_r[i + 1];
            for j in 0..s {
                let col = self.ws.p_col(j);
                let mut accum = 0.0;
                for k in 0..n {
                    accum += col[k] * vec[k];
                }
                self.ws.ph_drn[i * s + j] = accum;
            }
        }
        comm.allreduce_sum_slice(&mut self.ws.ph_drn);
        stats.dots += s * s;
    }

    fn solve_small_system(&mut self) -> Result<(), ()> {
        let s = self.ws.s;
        if s == 0 {
            return Ok(());
        }
        let mut a = self.ws.ph_drn.clone();
        let mut b = self.ws.ph_r.clone();
        self.ws.c.fill(0.0);
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
            for i in (k + 1)..s {
                let factor = a[i * s + k] / diag;
                if factor != 0.0 {
                    for j in k..s {
                        a[i * s + j] -= factor * a[k * s + j];
                    }
                    b[i] -= factor * b[k];
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

    fn combine_delta(dst: &mut [f64], coeffs: &[f64], src: &[Vec<f64>], scale: f64) {
        let n = dst.len();
        dst.fill(0.0);
        for (col, &coeff) in src.iter().zip(coeffs.iter()) {
            if coeff == 0.0 {
                continue;
            }
            for i in 0..n {
                dst[i] += coeff * col[i];
            }
        }
        if scale != 1.0 {
            for val in dst.iter_mut() {
                *val *= scale;
            }
        }
    }

    fn apply_matvec(
        a: &dyn LinOp<S = f64>,
        pc: Option<&mut &mut dyn Preconditioner>,
        x: &[f64],
        raw: &mut [f64],
        precond: &mut [f64],
        stats: &mut IdrsStats,
    ) -> Result<(), KError> {
        a.try_matvec(x, raw)?;
        stats.matvecs += 1;
        if let Some(pc_ref) = pc {
            (*pc_ref).apply(PcSide::Left, raw, precond)?;
        } else {
            precond.copy_from_slice(raw);
        }
        Ok(())
    }

    fn omega_value(&self, comm: &UniverseComm, stats: &mut IdrsStats, t: &[f64], v: &[f64]) -> f64 {
        let mut local_tv = 0.0;
        let mut local_tt = 0.0;
        for i in 0..t.len() {
            local_tv += t[i] * v[i];
            local_tt += t[i] * t[i];
        }
        let (tv, tt) = comm.allreduce_sum2(local_tv, local_tt);
        stats.dots += 2;
        let mut omega = if tt.abs() <= f64::EPSILON {
            0.0
        } else {
            tv / tt
        };
        if let Omega::MinResidualClipped { cos_min, kappa } = self.opts.omega_strategy {
            let mut local_vv = 0.0;
            for &vi in v {
                local_vv += vi * vi;
            }
            let vv = comm.all_reduce_f64(local_vv);
            stats.dots += 1;
            let denom = (tt * vv).sqrt();
            if denom > 0.0 {
                let cos = tv / denom;
                if cos.abs() < cos_min {
                    let sign = if tv >= 0.0 { 1.0 } else { -1.0 };
                    let target = cos_min * denom / tt.max(1e-32);
                    omega = kappa * omega + (1.0 - kappa) * sign * target;
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
        let (m, n) = a.dims();
        if m != n {
            return Err(KError::InvalidInput(
                "IDR(s) requires square operator".into(),
            ));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput(
                "IDR(s): vector length mismatch".into(),
            ));
        }
        if !matches!(pc_side, PcSide::Left) {
            return Err(KError::InvalidInput(
                "IDR(s) currently supports only left preconditioning".into(),
            ));
        }
        if self.opts.s == 0 {
            return Err(KError::InvalidInput("IDR(s): s must be >= 1".into()));
        }

        self.ws.ensure(n, self.opts.s);
        let mut stats = IdrsStats::default();

        let monitors = monitors.unwrap_or(&[]);
        let mut pc_opt = pc;

        self.random_bump = 0;

        if x.iter().all(|&xi| xi == 0.0) {
            self.ws.r_true.copy_from_slice(b);
        } else {
            a.try_matvec(x, &mut self.ws.t_raw)?;
            stats.matvecs += 1;
            for i in 0..n {
                self.ws.r_true[i] = b[i] - self.ws.t_raw[i];
            }
        }
        if let Some(pc_ref) = pc_opt.as_mut() {
            (*pc_ref).apply(PcSide::Left, &self.ws.r_true, &mut self.ws.r)?;
        } else {
            self.ws.r.copy_from_slice(&self.ws.r_true);
        }

        let mut local_bnorm = 0.0;
        for &bi in b {
            local_bnorm += bi * bi;
        }
        let bnorm = comm.all_reduce_f64(local_bnorm).sqrt();
        stats.dots += 1;
        let norm_scale = if bnorm > 0.0 { bnorm } else { 1.0 };
        let mut local_res = 0.0;
        for &ri in &self.ws.r_true {
            local_res += ri * ri;
        }
        let mut res_norm = comm.all_reduce_f64(local_res).sqrt();
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
            buf.fill(0.0);
        }
        for buf in &mut self.ws.d_r_raw {
            buf.fill(0.0);
        }
        for buf in &mut self.ws.d_x {
            buf.fill(0.0);
        }

        // Initialization: s minimum-norm steps
        for step in 0..min(self.opts.s, self.opts.maxit) {
            Self::apply_matvec(
                a,
                pc_opt.as_mut(),
                &self.ws.r,
                &mut self.ws.t_raw,
                &mut self.ws.v,
                &mut stats,
            )?;
            let mut local_vr = 0.0;
            let mut local_vv = 0.0;
            for i in 0..n {
                local_vr += self.ws.v[i] * self.ws.r[i];
                local_vv += self.ws.v[i] * self.ws.v[i];
            }
            let (vr, vv) = comm.allreduce_sum2(local_vr, local_vv);
            stats.dots += 2;
            if vv.abs() <= f64::EPSILON {
                return Err(KError::BreakdownOrIndefinite);
            }
            let omega = vr / vv;

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

            let mut local_res = 0.0;
            for &ri in &self.ws.r_true {
                local_res += ri * ri;
            }
            res_norm = comm.all_reduce_f64(local_res).sqrt();
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

        let mut attempts = 0usize;
        let mut iteration = min(self.opts.s, self.opts.maxit);
        let mut omega_block = 0.0;
        while iteration < self.opts.maxit {
            for inner in 0..=self.opts.s {
                self.compute_ph_r(comm, &mut stats);
                self.compute_ph_drn(comm, &mut stats);
                if self.solve_small_system().is_err() {
                    let mut retried = false;
                    if let BreakdownRepair::RegenerateP { max_retries, seed } =
                        self.opts.breakdown_repair
                    {
                        if attempts < max_retries {
                            attempts += 1;
                            match &self.opts.p_policy {
                                ShadowP::RandomOrthonormal { .. } => {
                                    self.random_bump = self.random_bump.wrapping_add(1);
                                    self.build_shadow_space(comm, &mut stats)?;
                                    retried = true;
                                }
                                _ => {
                                    let saved = self.opts.p_policy.clone();
                                    self.opts.p_policy = ShadowP::RandomOrthonormal {
                                        seed: seed.wrapping_add(attempts as u64),
                                    };
                                    self.random_bump = 0;
                                    self.build_shadow_space(comm, &mut stats)?;
                                    self.opts.p_policy = saved;
                                    retried = true;
                                }
                            }
                        }
                    }
                    if retried {
                        continue;
                    }
                    return Err(KError::BreakdownOrIndefinite);
                }

                let src_r = &self.ws.d_r[1..=self.opts.s];
                Self::combine_delta(&mut self.ws.v, &self.ws.c, src_r, -1.0);
                for i in 0..n {
                    self.ws.v[i] += self.ws.r[i];
                }

                if inner == 0 {
                    Self::apply_matvec(
                        a,
                        pc_opt.as_mut(),
                        &self.ws.v,
                        &mut self.ws.t_raw,
                        &mut self.ws.t,
                        &mut stats,
                    )?;
                    omega_block = self.omega_value(comm, &mut stats, &self.ws.t, &self.ws.v);
                    if omega_block.abs() <= f64::EPSILON {
                        return Err(KError::BreakdownOrIndefinite);
                    }

                    self.ws.d_r.rotate_right(1);
                    self.ws.d_r_raw.rotate_right(1);
                    self.ws.d_x.rotate_right(1);
                    let (newest_x, rest_x) = self.ws.d_x.split_first_mut().expect("nonempty");
                    let (newest_r, rest_r) = self.ws.d_r.split_first_mut().expect("nonempty");
                    let (newest_r_raw, rest_rr) =
                        self.ws.d_r_raw.split_first_mut().expect("nonempty");
                    let src_x = &rest_x[..self.opts.s];
                    Self::combine_delta(newest_x, &self.ws.c, src_x, -1.0);
                    for i in 0..n {
                        newest_x[i] += omega_block * self.ws.v[i];
                    }
                    let src_r = &rest_r[..self.opts.s];
                    Self::combine_delta(newest_r, &self.ws.c, src_r, -1.0);
                    for i in 0..n {
                        newest_r[i] -= omega_block * self.ws.t[i];
                    }
                    let src_rr = &rest_rr[..self.opts.s];
                    Self::combine_delta(newest_r_raw, &self.ws.c, src_rr, -1.0);
                    for i in 0..n {
                        newest_r_raw[i] -= omega_block * self.ws.t_raw[i];
                    }
                } else {
                    self.ws.d_x.rotate_right(1);
                    self.ws.d_r.rotate_right(1);
                    self.ws.d_r_raw.rotate_right(1);
                    let (newest_x, rest_x) = self.ws.d_x.split_first_mut().expect("nonempty");
                    let (newest_r, _rest_r) = self.ws.d_r.split_first_mut().expect("nonempty");
                    let (newest_r_raw, _rest_rr) =
                        self.ws.d_r_raw.split_first_mut().expect("nonempty");
                    let src_x = &rest_x[..self.opts.s];
                    Self::combine_delta(newest_x, &self.ws.c, src_x, -1.0);
                    for i in 0..n {
                        newest_x[i] += omega_block * self.ws.v[i];
                    }
                    Self::apply_matvec(
                        a,
                        pc_opt.as_mut(),
                        newest_x,
                        &mut self.ws.t_raw,
                        &mut self.ws.t,
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

                iteration += 1;

                if let Some(freq) = self.opts.monitor_true_residual_every {
                    if iteration % freq == 0 {
                        a.try_matvec(x, &mut self.ws.t_raw)?;
                        stats.matvecs += 1;
                        for i in 0..n {
                            self.ws.r_true[i] = b[i] - self.ws.t_raw[i];
                        }
                        if let Some(pc_ref) = pc_opt.as_mut() {
                            (*pc_ref).apply(PcSide::Left, &self.ws.r_true, &mut self.ws.r)?;
                        }
                        stats.residual_replacements += 1;
                    }
                }

                let mut local_res = 0.0;
                for &ri in &self.ws.r_true {
                    local_res += ri * ri;
                }
                res_norm = comm.all_reduce_f64(local_res).sqrt();
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
}
