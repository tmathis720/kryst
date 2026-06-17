use std::ops::AddAssign;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;

use super::coarse_solver::{CoarseCg, CoarseDenseLu, CoarseSolver};
use super::rap_ops::{CsrPattern, adjoint_csr_with_pos, galerkin_numeric, galerkin_symbolic};
use super::{AMGConfig, AmgStats, AmgTransferOperators, CoarseSolve, LevelStats, RelaxPhase};

struct ScalarLevel<T: KrystScalar<Real = f64>> {
    a: CsrMatrix<T>,
    p: CsrMatrix<T>,
    r: CsrMatrix<T>,
    diag_inv: Vec<T>,
}

pub(crate) struct AmgCore<T: KrystScalar<Real = f64>> {
    levels: Vec<ScalarLevel<T>>,
    coarse_solver: ScalarCoarseSolver<T>,
    workspaces: Vec<ScalarWorkspace<T>>,
    cfg: AMGConfig,
}

enum ScalarCoarseSolver<T: KrystScalar<Real = f64>> {
    Cg(CoarseCg<T>),
    Dense(CoarseDenseLu<T>),
}

impl<T: KrystScalar<Real = f64>> ScalarCoarseSolver<T> {
    fn from_config(cfg: &AMGConfig) -> Result<Self, KError> {
        match cfg.coarse_solve {
            CoarseSolve::CG => Ok(Self::Cg(CoarseCg::new(cfg.tolerance, cfg.max_iterations))),
            CoarseSolve::DirectDense => Ok(Self::Dense(CoarseDenseLu::new())),
            CoarseSolve::ILU => Err(KError::Unsupported(
                "AMG scalar core coarse_solve=ILU is not available for this scalar path".into(),
            )),
            CoarseSolve::Smoother => Err(KError::Unsupported(
                "AMG scalar core coarse_solve=Smoother is not implemented for native scalar hierarchy"
                    .into(),
            )),
        }
    }

    fn setup(&mut self, a: &CsrMatrix<T>) -> Result<(), KError> {
        match self {
            Self::Cg(solver) => solver.setup(a),
            Self::Dense(solver) => solver.setup(a),
        }
    }

    fn solve(&mut self, b: &[T], x: &mut [T]) -> Result<(), KError> {
        match self {
            Self::Cg(solver) => solver.solve(b, x),
            Self::Dense(solver) => solver.solve(b, x),
        }
    }
}

struct ScalarWorkspace<T: KrystScalar<Real = f64>> {
    az: Vec<T>,
    residual: Vec<T>,
    coarse_rhs: Vec<T>,
    coarse_sol: Vec<T>,
    fine_corr: Vec<T>,
}

impl<T: KrystScalar<Real = f64>> ScalarWorkspace<T> {
    fn new(max_n: usize) -> Self {
        Self {
            az: vec![T::zero(); max_n],
            residual: vec![T::zero(); max_n],
            coarse_rhs: vec![T::zero(); max_n],
            coarse_sol: vec![T::zero(); max_n],
            fine_corr: vec![T::zero(); max_n],
        }
    }

    fn ensure(&mut self, n: usize) {
        self.az.resize(n, T::zero());
        self.residual.resize(n, T::zero());
        self.coarse_rhs.resize(n, T::zero());
        self.coarse_sol.resize(n, T::zero());
        self.fine_corr.resize(n, T::zero());
    }
}

impl<T> AmgCore<T>
where
    T: KrystScalar<Real = f64> + AddAssign,
{
    pub(crate) fn setup(fine: &CsrMatrix<T>, cfg: &AMGConfig) -> Result<Self, KError> {
        Self::setup_with_transfer_overrides(fine, cfg, &[])
    }

    pub(crate) fn setup_with_transfer_overrides(
        fine: &CsrMatrix<T>,
        cfg: &AMGConfig,
        transfer_overrides: &[(usize, AmgTransferOperators)],
    ) -> Result<Self, KError> {
        if fine.nrows() != fine.ncols() {
            return Err(KError::InvalidInput(
                "AMG scalar core requires a square matrix".into(),
            ));
        }
        let mut levels = Vec::with_capacity(cfg.max_levels.max(1));
        let mut a_cur = fine.clone();
        let max_levels = cfg.max_levels.max(1);
        for level in 0..max_levels {
            let diag_inv = diag_inv_from_csr(&a_cur, cfg.drop_tol)?;
            let n = a_cur.nrows();
            let override_ops = transfer_override_for_level(transfer_overrides, level);
            let terminal = override_ops.is_none()
                && (level + 1 == max_levels || n <= cfg.max_coarse_size.max(1) || n <= 2);
            if terminal {
                levels.push(ScalarLevel {
                    a: a_cur,
                    p: CsrMatrix::identity(n),
                    r: CsrMatrix::identity(n),
                    diag_inv,
                });
                break;
            }

            let p = if let Some(ops) = override_ops {
                validate_transfer_override(level, n, ops)?;
                clone_csr_cast(&ops.prolongation)
            } else {
                pairwise_piecewise_constant_p::<T>(n)
            };
            let (r, _) = adjoint_csr_with_pos(&p);
            let pat = galerkin_symbolic(&a_cur, &p);
            let mut vals = vec![T::zero(); pat.col_idx.len()];
            galerkin_numeric(&pat, &a_cur, &p, &mut vals);
            let a_next = CsrMatrix::from_csr(
                pat.nrows,
                pat.ncols,
                pat.row_ptr.clone(),
                pat.col_idx.clone(),
                vals,
            );
            levels.push(ScalarLevel {
                a: a_cur,
                p,
                r,
                diag_inv,
            });
            if a_next.nrows() >= n {
                break;
            }
            a_cur = a_next;
        }

        let coarsest = levels
            .last()
            .ok_or_else(|| KError::InvalidInput("AMG scalar core built no levels".into()))?;
        let mut coarse_solver = ScalarCoarseSolver::<T>::from_config(cfg)?;
        coarse_solver.setup(&coarsest.a)?;
        Ok(Self {
            levels,
            coarse_solver,
            workspaces: Vec::new(),
            cfg: cfg.clone(),
        })
    }

    pub(crate) fn update_numeric(&mut self, fine: &CsrMatrix<T>) -> Result<(), KError> {
        let first = self
            .levels
            .first()
            .ok_or_else(|| KError::InvalidInput("AMG scalar core not set up".into()))?;
        if !same_csr_pattern(&first.a, fine) {
            return Err(KError::InvalidInput(
                "AMG scalar core numeric update requires unchanged sparsity".into(),
            ));
        }

        self.levels[0].a.values_mut().copy_from_slice(fine.values());
        self.levels[0].diag_inv = diag_inv_from_csr(&self.levels[0].a, self.cfg.drop_tol)?;

        for level in 0..self.levels.len().saturating_sub(1) {
            let (fine_levels, coarse_levels) = self.levels.split_at_mut(level + 1);
            let fine_level = &fine_levels[level];
            let coarse_level = &mut coarse_levels[0];
            let pat = pattern_from_csr(&coarse_level.a);
            galerkin_numeric(
                &pat,
                &fine_level.a,
                &fine_level.p,
                coarse_level.a.values_mut(),
            );
            coarse_level.diag_inv = diag_inv_from_csr(&coarse_level.a, self.cfg.drop_tol)?;
        }

        let coarsest = self
            .levels
            .last()
            .ok_or_else(|| KError::InvalidInput("AMG scalar core not set up".into()))?;
        self.coarse_solver.setup(&coarsest.a)
    }

    pub(crate) fn apply(&mut self, rhs: &[T], out: &mut [T]) -> Result<(), KError> {
        let n = self
            .levels
            .first()
            .map(|l| l.a.nrows())
            .ok_or_else(|| KError::InvalidInput("AMG scalar core not set up".into()))?;
        if rhs.len() != n || out.len() != n {
            return Err(KError::InvalidInput(
                "AMG scalar core apply length mismatch".into(),
            ));
        }
        out.fill(T::zero());
        self.ensure_apply_workspaces();
        let mut workspaces = std::mem::take(&mut self.workspaces);
        let result = self.v_cycle(0, rhs, out, &mut workspaces);
        self.workspaces = workspaces;
        result
    }

    pub(crate) fn stats(&self) -> AmgStats {
        AmgStats::from_scalar_core(
            self.levels
                .iter()
                .map(|level| (&level.a, level.p.nnz(), level.r.nnz())),
            &self.cfg,
        )
    }

    fn ensure_apply_workspaces(&mut self) {
        let needed = self.levels.len().saturating_sub(1);
        if self.workspaces.len() != needed {
            self.workspaces = self
                .levels
                .iter()
                .take(needed)
                .map(|level| ScalarWorkspace::new(level.a.nrows()))
                .collect();
            return;
        }
        for (ws, level) in self.workspaces.iter_mut().zip(self.levels.iter()) {
            ws.ensure(level.a.nrows());
        }
    }

    fn v_cycle(
        &mut self,
        level: usize,
        rhs: &[T],
        sol: &mut [T],
        workspaces: &mut [ScalarWorkspace<T>],
    ) -> Result<(), KError> {
        let n = self.levels[level].a.nrows();
        if level + 1 == self.levels.len() {
            return self.coarse_solver.solve(rhs, sol);
        }
        let (ws, child_workspaces) = workspaces.split_first_mut().ok_or_else(|| {
            KError::InvalidInput("AMG scalar core missing apply workspace".into())
        })?;

        let pre = self.cfg.num_grid_sweeps[RelaxPhase::Down.ix()];
        let post = self.cfg.num_grid_sweeps[RelaxPhase::Up.ix()];
        jacobi(
            &self.levels[level].a,
            &self.levels[level].diag_inv,
            rhs,
            sol,
            self.cfg.jacobi_omega,
            pre,
            &mut ws.az[..n],
        )?;

        self.levels[level]
            .a
            .spmv_scaled(T::one(), sol, T::zero(), &mut ws.az[..n])?;
        for i in 0..n {
            ws.residual[i] = rhs[i] - ws.az[i];
        }

        let nc = self.levels[level + 1].a.nrows();
        self.levels[level].r.spmv_scaled(
            T::one(),
            &ws.residual[..n],
            T::zero(),
            &mut ws.coarse_rhs[..nc],
        )?;
        for value in &mut ws.coarse_sol[..nc] {
            *value = T::zero();
        }
        self.v_cycle(
            level + 1,
            &ws.coarse_rhs[..nc],
            &mut ws.coarse_sol[..nc],
            child_workspaces,
        )?;
        self.levels[level].p.spmv_scaled(
            T::one(),
            &ws.coarse_sol[..nc],
            T::zero(),
            &mut ws.fine_corr[..n],
        )?;
        for i in 0..n {
            sol[i] = sol[i] + ws.fine_corr[i];
        }

        jacobi(
            &self.levels[level].a,
            &self.levels[level].diag_inv,
            rhs,
            sol,
            self.cfg.jacobi_omega,
            post,
            &mut ws.az[..n],
        )
    }
}

fn transfer_override_for_level(
    overrides: &[(usize, AmgTransferOperators)],
    level: usize,
) -> Option<&AmgTransferOperators> {
    overrides
        .iter()
        .find_map(|(override_level, ops)| (*override_level == level).then_some(ops))
}

fn validate_transfer_override(
    level: usize,
    n_fine: usize,
    ops: &AmgTransferOperators,
) -> Result<(), KError> {
    if ops.prolongation.nrows() != n_fine {
        return Err(KError::InvalidInput(format!(
            "AMG scalar core transfer override level {level} has P rows {}, expected {n_fine}",
            ops.prolongation.nrows()
        )));
    }
    if ops.prolongation.ncols() == 0 {
        return Err(KError::InvalidInput(format!(
            "AMG scalar core transfer override level {level} has zero coarse columns"
        )));
    }
    if ops.restriction.ncols() != n_fine {
        return Err(KError::InvalidInput(format!(
            "AMG scalar core transfer override level {level} has R cols {}, expected {n_fine}",
            ops.restriction.ncols()
        )));
    }
    if ops.prolongation.ncols() != ops.restriction.nrows() {
        return Err(KError::InvalidInput(format!(
            "AMG scalar core transfer override level {level} has inconsistent coarse dims P: {} cols, R: {} rows",
            ops.prolongation.ncols(),
            ops.restriction.nrows()
        )));
    }
    Ok(())
}

fn clone_csr_cast<T: KrystScalar<Real = f64>>(a: &CsrMatrix<crate::S>) -> CsrMatrix<T> {
    CsrMatrix::from_csr(
        a.nrows(),
        a.ncols(),
        a.row_ptr().to_vec(),
        a.col_idx().to_vec(),
        a.values()
            .iter()
            .map(|v| T::from_parts(v.real(), v.imag()))
            .collect(),
    )
}

fn same_csr_pattern<T: KrystScalar<Real = f64>>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> bool {
    a.nrows() == b.nrows()
        && a.ncols() == b.ncols()
        && a.row_ptr() == b.row_ptr()
        && a.col_idx() == b.col_idx()
}

fn pattern_from_csr<T: KrystScalar<Real = f64>>(a: &CsrMatrix<T>) -> CsrPattern {
    CsrPattern {
        nrows: a.nrows(),
        ncols: a.ncols(),
        row_ptr: a.row_ptr().to_vec(),
        col_idx: a.col_idx().to_vec(),
    }
}

fn diag_inv_from_csr<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<T>,
    drop_tol: f64,
) -> Result<Vec<T>, KError> {
    let mut out = Vec::with_capacity(a.nrows());
    for i in 0..a.nrows() {
        let (cols, vals) = a.row(i);
        let diag = cols
            .iter()
            .zip(vals.iter())
            .find_map(|(&j, &v)| (j == i).then_some(v))
            .ok_or_else(|| KError::InvalidInput("AMG scalar core missing diagonal".into()))?;
        if diag.abs() <= drop_tol.max(1e-30) {
            return Err(KError::InvalidInput(
                "AMG scalar core encountered near-zero diagonal".into(),
            ));
        }
        out.push(diag.inv());
    }
    Ok(out)
}

fn pairwise_piecewise_constant_p<T: KrystScalar<Real = f64>>(n: usize) -> CsrMatrix<T> {
    let nc = n.div_ceil(2);
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    row_ptr.push(0);
    for i in 0..n {
        col_idx.push(i / 2);
        vals.push(T::one());
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, nc, row_ptr, col_idx, vals)
}

fn jacobi<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<T>,
    diag_inv: &[T],
    rhs: &[T],
    sol: &mut [T],
    omega: f64,
    sweeps: usize,
    work: &mut [T],
) -> Result<(), KError> {
    let n = a.nrows();
    let omega = T::from_real(omega);
    for _ in 0..sweeps {
        a.spmv_scaled(T::one(), sol, T::zero(), work)?;
        for i in 0..n {
            sol[i] = sol[i] + omega * diag_inv[i] * (rhs[i] - work[i]);
        }
    }
    Ok(())
}

impl AmgStats {
    pub(crate) fn from_scalar_core<'a, T>(
        levels_iter: impl IntoIterator<Item = (&'a CsrMatrix<T>, usize, usize)>,
        cfg: &AMGConfig,
    ) -> Self
    where
        T: KrystScalar<Real = f64> + 'a,
    {
        let raw = levels_iter
            .into_iter()
            .map(|(a, nnz_p, nnz_r)| {
                (
                    a.nrows(),
                    a.nnz(),
                    nnz_p,
                    nnz_r,
                    max_row_sum_abs_scalar(a),
                    eff_nnz_scalar(a, cfg.stats_eps),
                )
            })
            .collect::<Vec<_>>();
        let n0 = raw
            .first()
            .map(|(n, _, _, _, _, _)| *n as f64)
            .unwrap_or(1.0);
        let nnz0 = raw
            .first()
            .map(|(_, nnz, _, _, _, _)| *nnz as f64)
            .unwrap_or(1.0);
        let total_n = raw.iter().map(|(n, _, _, _, _, _)| *n as f64).sum::<f64>();
        let total_nnz = raw.iter().map(|(_, nnz, _, _, _, _)| *nnz).sum::<usize>();
        let levels = raw
            .iter()
            .enumerate()
            .map(
                |(level, &(n, nnz_a, nnz_p, nnz_r, max_row_sum_a, eff_nnz_a))| LevelStats {
                    level,
                    n,
                    nnz_a,
                    nnz_p: if level + 1 == raw.len() { 0 } else { nnz_p },
                    nnz_r: if level + 1 == raw.len() { 0 } else { nnz_r },
                    max_row_sum_a,
                    eff_nnz_a: Some(eff_nnz_a),
                    pre_sweeps: cfg.num_grid_sweeps[RelaxPhase::Down.ix()],
                    post_sweeps: cfg.num_grid_sweeps[RelaxPhase::Up.ix()],
                    pre_work_estimate: cfg.num_grid_sweeps[RelaxPhase::Down.ix()] as f64
                        * nnz_a as f64,
                    post_work_estimate: cfg.num_grid_sweeps[RelaxPhase::Up.ix()] as f64
                        * nnz_a as f64,
                    selected_relax_pre: format!("{:?}", cfg.grid_relax_type[RelaxPhase::Down.ix()]),
                    selected_relax_post: format!("{:?}", cfg.grid_relax_type[RelaxPhase::Up.ix()]),
                    coarse_solver: (level + 1 == raw.len())
                        .then(|| format!("{:?}", cfg.coarse_solve)),
                },
            )
            .collect::<Vec<_>>();
        let total_smoothing_work = levels
            .iter()
            .map(|l| l.pre_work_estimate + l.post_work_estimate)
            .sum();
        Self {
            grid_complexity: total_n / n0,
            operator_complexity: total_nnz as f64 / nnz0,
            total_nnz,
            total_smoothing_work,
            num_levels: raw.len(),
            levels,
            diagnostics: Vec::new(),
            setup: Default::default(),
            last_cycle: None,
            selected_dist_coarse_route: None,
            dist_route_fallback: Vec::new(),
            #[cfg(feature = "complex")]
            complex_setup_mode: super::AmgComplexSetupMode::Unset,
            #[cfg(feature = "complex")]
            complex_setup_fallback_reason: None,
        }
    }
}

fn max_row_sum_abs_scalar<T: KrystScalar<Real = f64>>(a: &CsrMatrix<T>) -> f64 {
    let mut max_sum = 0.0f64;
    for row in 0..a.nrows() {
        let (_, vals) = a.row(row);
        let sum = vals.iter().map(|v| v.abs()).sum::<f64>();
        max_sum = max_sum.max(sum);
    }
    max_sum
}

fn eff_nnz_scalar<T: KrystScalar<Real = f64>>(a: &CsrMatrix<T>, eps: f64) -> usize {
    a.values().iter().filter(|v| v.abs() > eps).count()
}
