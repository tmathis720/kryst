use std::ops::AddAssign;

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;

use super::coarse_solver::{CoarseDenseLu, CoarseSolver};
use super::rap_ops::{adjoint_csr_with_pos, galerkin_numeric, galerkin_symbolic};
use super::{AMGConfig, AmgStats, CoarseSolve, LevelStats, RelaxPhase};

struct ScalarLevel<T: KrystScalar<Real = f64>> {
    a: CsrMatrix<T>,
    p: CsrMatrix<T>,
    r: CsrMatrix<T>,
    diag_inv: Vec<T>,
}

pub(crate) struct AmgCore<T: KrystScalar<Real = f64>> {
    levels: Vec<ScalarLevel<T>>,
    coarse_solver: CoarseDenseLu<T>,
    cfg: AMGConfig,
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
}

impl<T> AmgCore<T>
where
    T: KrystScalar<Real = f64> + AddAssign,
{
    pub(crate) fn setup(fine: &CsrMatrix<T>, cfg: &AMGConfig) -> Result<Self, KError> {
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
            let terminal = level + 1 == max_levels || n <= cfg.max_coarse_size.max(1) || n <= 2;
            if terminal {
                levels.push(ScalarLevel {
                    a: a_cur,
                    p: CsrMatrix::identity(n),
                    r: CsrMatrix::identity(n),
                    diag_inv,
                });
                break;
            }

            let p = pairwise_piecewise_constant_p::<T>(n);
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
        let mut coarse_solver = CoarseDenseLu::<T>::new();
        coarse_solver.setup(&coarsest.a)?;
        Ok(Self {
            levels,
            coarse_solver,
            cfg: cfg.clone(),
        })
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
        let max_n = self.levels.iter().map(|l| l.a.nrows()).max().unwrap_or(n);
        let mut ws = ScalarWorkspace::new(max_n);
        self.v_cycle(0, rhs, out, &mut ws)
    }

    pub(crate) fn stats(&self) -> AmgStats {
        AmgStats::from_scalar_core(
            self.levels
                .iter()
                .map(|level| (level.a.nrows(), level.a.nnz(), level.p.nnz(), level.r.nnz())),
            &self.cfg,
        )
    }

    fn v_cycle(
        &mut self,
        level: usize,
        rhs: &[T],
        sol: &mut [T],
        ws: &mut ScalarWorkspace<T>,
    ) -> Result<(), KError> {
        let n = self.levels[level].a.nrows();
        if level + 1 == self.levels.len() {
            return self.coarse_solver.solve(rhs, sol);
        }

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
        let mut child_ws = ScalarWorkspace::new(ws.az.len());
        self.v_cycle(
            level + 1,
            &ws.coarse_rhs[..nc],
            &mut ws.coarse_sol[..nc],
            &mut child_ws,
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
    pub(crate) fn from_scalar_core(
        levels_iter: impl IntoIterator<Item = (usize, usize, usize, usize)>,
        cfg: &AMGConfig,
    ) -> Self {
        let raw: Vec<_> = levels_iter.into_iter().collect();
        let n0 = raw.first().map(|(n, _, _, _)| *n as f64).unwrap_or(1.0);
        let nnz0 = raw.first().map(|(_, nnz, _, _)| *nnz as f64).unwrap_or(1.0);
        let total_n = raw.iter().map(|(n, _, _, _)| *n as f64).sum::<f64>();
        let total_nnz = raw.iter().map(|(_, nnz, _, _)| *nnz).sum::<usize>();
        let levels = raw
            .iter()
            .enumerate()
            .map(|(level, &(n, nnz_a, nnz_p, nnz_r))| LevelStats {
                level,
                n,
                nnz_a,
                nnz_p: if level + 1 == raw.len() { 0 } else { nnz_p },
                nnz_r: if level + 1 == raw.len() { 0 } else { nnz_r },
                max_row_sum_a: 0.0,
                eff_nnz_a: None,
                pre_sweeps: cfg.num_grid_sweeps[RelaxPhase::Down.ix()],
                post_sweeps: cfg.num_grid_sweeps[RelaxPhase::Up.ix()],
                pre_work_estimate: cfg.num_grid_sweeps[RelaxPhase::Down.ix()] as f64 * nnz_a as f64,
                post_work_estimate: cfg.num_grid_sweeps[RelaxPhase::Up.ix()] as f64 * nnz_a as f64,
                selected_relax_pre: format!("{:?}", cfg.grid_relax_type[RelaxPhase::Down.ix()]),
                selected_relax_post: format!("{:?}", cfg.grid_relax_type[RelaxPhase::Up.ix()]),
                coarse_solver: (level + 1 == raw.len())
                    .then(|| format!("{:?}", CoarseSolve::DirectDense)),
            })
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
