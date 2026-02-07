use crate::algebra::prelude::*;
use crate::algebra::scalar::{S, is_complex_scalar};
use crate::context::pc_context::{PcFactory, PcType};
use crate::error::KError;
use crate::matrix::convert::csr_from_linop;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::utils::rap_opt;
use crate::preconditioner::{PcSide, Preconditioner};
use std::str::FromStr;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MgCycleType {
    V,
    W,
    F,
}

impl MgCycleType {
    fn from_option(cycle_type: Option<&str>) -> Result<Self, KError> {
        match cycle_type.unwrap_or("v").to_lowercase().as_str() {
            "v" | "vcycle" => Ok(MgCycleType::V),
            "w" | "wcycle" => Ok(MgCycleType::W),
            "f" | "fcycle" => Ok(MgCycleType::F),
            other => Err(KError::InvalidInput(format!(
                "unknown pc_mg_cycle_type: {other}"
            ))),
        }
    }
}

#[derive(Debug)]
enum MgCoarseSolve {
    Direct(Box<dyn Preconditioner>),
    Smoother(Box<dyn Preconditioner>, usize),
}

#[derive(Clone)]
struct CsrLinOp {
    csr: Arc<CsrMatrix<f64>>,
}

impl CsrLinOp {
    fn new(csr: Arc<CsrMatrix<f64>>) -> Self {
        Self { csr }
    }
}

impl LinOp for CsrLinOp {
    type S = S;

    fn dims(&self) -> (usize, usize) {
        (self.csr.nrows(), self.csr.ncols())
    }

    fn matvec(&self, x: &[Self::S], y: &mut [Self::S]) {
        if let Err(err) = self.try_matvec(x, y) {
            debug_assert!(false, "CsrLinOp::matvec dimension mismatch: {err}");
        }
    }

    fn try_matvec(&self, x: &[Self::S], y: &mut [Self::S]) -> Result<(), KError> {
        self.csr.try_spmv(x, y)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct MgLevel {
    pub level: usize,
    pub smoother: Option<Box<dyn Preconditioner>>,
    pub operator: Arc<CsrMatrix<f64>>,
    pub prolongation: Option<Arc<CsrMatrix<f64>>>,
    pub restriction: Option<Arc<CsrMatrix<f64>>>,
}

impl MgLevel {
    pub fn new(level: usize, operator: Arc<CsrMatrix<f64>>) -> Self {
        Self {
            level,
            smoother: None,
            operator,
            prolongation: None,
            restriction: None,
        }
    }
}

pub struct MgHierarchy {
    levels: Vec<MgLevel>,
}

impl MgHierarchy {
    pub fn new(levels: Vec<MgLevel>) -> Self {
        Self { levels }
    }

    pub fn set_smoother(&mut self, level: usize, smoother: Box<dyn Preconditioner>) {
        if let Some(entry) = self.levels.get_mut(level) {
            entry.smoother = Some(smoother);
        }
    }

    pub fn levels(&self) -> &[MgLevel] {
        &self.levels
    }

    pub fn levels_mut(&mut self) -> &mut [MgLevel] {
        &mut self.levels
    }
}

pub struct MgPc {
    pub levels: usize,
    pub cycle_type: Option<String>,
    pub smoother: Option<String>,
    pub smoother_steps: Option<usize>,
    hierarchy: Option<MgHierarchy>,
    coarse_solve: Option<MgCoarseSolve>,
    cycle: MgCycleType,
    smoother_sweeps: usize,
}

impl MgPc {
    pub fn new(
        levels: usize,
        cycle_type: Option<String>,
        smoother: Option<String>,
        smoother_steps: Option<usize>,
    ) -> Self {
        let cycle = MgCycleType::from_option(cycle_type.as_deref()).unwrap_or(MgCycleType::V);
        let smoother_sweeps = smoother_steps.unwrap_or(1).max(1);
        Self {
            levels,
            cycle_type,
            smoother,
            smoother_steps,
            hierarchy: None,
            coarse_solve: None,
            cycle,
            smoother_sweeps,
        }
    }

    pub fn hierarchy(&self) -> &MgHierarchy {
        self.hierarchy
            .as_ref()
            .expect("MgPc::hierarchy requires setup")
    }

    fn build_smoother(&self, name: &str) -> Result<Box<dyn Preconditioner>, KError> {
        let pc_type = PcType::from_str(name)?;
        if pc_type == PcType::Mg {
            return Err(KError::InvalidInput(
                "pc_mg_smoother cannot be mg".into(),
            ));
        }
        if pc_type == PcType::None {
            return Err(KError::InvalidInput(
                "pc_mg_smoother cannot be none".into(),
            ));
        }
        PcFactory::create_preconditioner(pc_type, None)
    }

    fn build_transfer(n_fine: usize) -> (CsrMatrix<f64>, CsrMatrix<f64>, usize) {
        let n_coarse = (n_fine + 1) / 2;
        let mut p_row_ptr = Vec::with_capacity(n_fine + 1);
        let mut p_col_idx = Vec::with_capacity(n_fine);
        let mut p_values = Vec::with_capacity(n_fine);
        p_row_ptr.push(0);
        for i in 0..n_fine {
            let coarse = i / 2;
            p_col_idx.push(coarse);
            p_values.push(1.0);
            p_row_ptr.push(p_col_idx.len());
        }
        let mut r_row_ptr = Vec::with_capacity(n_coarse + 1);
        let mut r_col_idx = Vec::with_capacity(n_fine);
        let mut r_values = Vec::with_capacity(n_fine);
        r_row_ptr.push(0);
        for j in 0..n_coarse {
            let fine0 = 2 * j;
            let fine1 = fine0 + 1;
            let mut entries = 0;
            if fine0 < n_fine {
                r_col_idx.push(fine0);
                r_values.push(if fine1 < n_fine { 0.5 } else { 1.0 });
                entries += 1;
            }
            if fine1 < n_fine {
                r_col_idx.push(fine1);
                r_values.push(0.5);
                entries += 1;
            }
            r_row_ptr.push(*r_row_ptr.last().unwrap() + entries);
        }
        let p = CsrMatrix::from_csr(n_fine, n_coarse, p_row_ptr, p_col_idx, p_values);
        let r = CsrMatrix::from_csr(n_coarse, n_fine, r_row_ptr, r_col_idx, r_values);
        (p, r, n_coarse)
    }

    fn smooth_level(
        level: &MgLevel,
        sweeps: usize,
        b: &[S],
        x: &mut [S],
    ) -> Result<(), KError> {
        let smoother = match level.smoother.as_ref() {
            Some(sm) => sm,
            None => return Ok(()),
        };
        let n = level.operator.nrows();
        let mut residual = vec![S::zero(); n];
        let mut correction = vec![S::zero(); n];
        for _ in 0..sweeps {
            level.operator.try_spmv(x, &mut residual)?;
            for i in 0..n {
                residual[i] = b[i] - residual[i];
            }
            smoother.apply(PcSide::Left, &residual, &mut correction)?;
            for i in 0..n {
                x[i] += correction[i];
            }
        }
        Ok(())
    }

    fn mg_cycle(
        &self,
        level_ix: usize,
        b: &[S],
        x: &mut [S],
        cycle: MgCycleType,
    ) -> Result<(), KError> {
        let hierarchy = self
            .hierarchy
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("multigrid hierarchy not set up".into()))?;
        let level = &hierarchy.levels[level_ix];
        let is_coarse = level_ix + 1 == hierarchy.levels.len();
        if is_coarse {
            if let Some(coarse) = &self.coarse_solve {
                match coarse {
                    MgCoarseSolve::Direct(pc) => {
                        let op = CsrLinOp::new(level.operator.clone());
                        if let Err(err) = pc.direct_solve(&op, b, x) {
                            log::warn!(
                                "coarse direct_solve failed ({err}); falling back to apply"
                            );
                            pc.apply(PcSide::Left, b, x)?;
                        }
                    }
                    MgCoarseSolve::Smoother(pc, sweeps) => {
                        let mut residual = vec![S::zero(); b.len()];
                        let mut correction = vec![S::zero(); b.len()];
                        for _ in 0..*sweeps {
                            level.operator.try_spmv(x, &mut residual)?;
                            for i in 0..b.len() {
                                residual[i] = b[i] - residual[i];
                            }
                            pc.apply(PcSide::Left, &residual, &mut correction)?;
                            for i in 0..b.len() {
                                x[i] += correction[i];
                            }
                        }
                    }
                }
            }
            return Ok(());
        }

        Self::smooth_level(level, self.smoother_sweeps, b, x)?;

        let mut residual = vec![S::zero(); b.len()];
        level.operator.try_spmv(x, &mut residual)?;
        for i in 0..b.len() {
            residual[i] = b[i] - residual[i];
        }

        let restriction = level
            .restriction
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("missing restriction operator".into()))?;
        let prolongation = level
            .prolongation
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("missing prolongation operator".into()))?;
        let mut coarse_rhs = vec![S::zero(); restriction.nrows()];
        restriction.try_spmv(&residual, &mut coarse_rhs)?;
        let mut coarse_sol = vec![S::zero(); coarse_rhs.len()];

        match cycle {
            MgCycleType::V => {
                self.mg_cycle(level_ix + 1, &coarse_rhs, &mut coarse_sol, cycle)?;
            }
            MgCycleType::W => {
                self.mg_cycle(level_ix + 1, &coarse_rhs, &mut coarse_sol, cycle)?;
                self.mg_cycle(level_ix + 1, &coarse_rhs, &mut coarse_sol, cycle)?;
            }
            MgCycleType::F => {
                self.mg_cycle(level_ix + 1, &coarse_rhs, &mut coarse_sol, MgCycleType::F)?;
                self.mg_cycle(level_ix + 1, &coarse_rhs, &mut coarse_sol, MgCycleType::V)?;
            }
        }

        let mut fine_correction = vec![S::zero(); prolongation.nrows()];
        prolongation.try_spmv(&coarse_sol, &mut fine_correction)?;
        for i in 0..x.len() {
            x[i] += fine_correction[i];
        }

        Self::smooth_level(level, self.smoother_sweeps, b, x)?;
        Ok(())
    }
}

impl Preconditioner for MgPc {
    fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
        if self.levels < 2 {
            return Err(KError::InvalidInput("pc_mg_levels must be >= 2".into()));
        }
        if is_complex_scalar::<S>() {
            return Err(KError::Unsupported(
                "multigrid is only supported for real scalars".into(),
            ));
        }
        let a = csr_from_linop(_a, 0.0)?;
        let mut levels = Vec::new();
        levels.push(MgLevel::new(0, a.clone()));
        let mut current = a;
        for level in 0..(self.levels - 1) {
            let (p, r, n_coarse) = Self::build_transfer(current.nrows());
            let p = Arc::new(p);
            let r = Arc::new(r);
            let coarse = rap_opt(r.as_ref(), current.as_ref(), p.as_ref())?;
            let coarse = Arc::new(coarse);
            if let Some(entry) = levels.get_mut(level) {
                entry.prolongation = Some(p.clone());
                entry.restriction = Some(r.clone());
            }
            levels.push(MgLevel::new(level + 1, coarse.clone()));
            current = coarse;
            if n_coarse <= 1 {
                break;
            }
        }
        if levels.len() < 2 {
            return Err(KError::InvalidInput(
                "multigrid hierarchy requires at least 2 levels".into(),
            ));
        }
        self.levels = levels.len();
        self.cycle = MgCycleType::from_option(self.cycle_type.as_deref())?;
        self.smoother_sweeps = self.smoother_steps.unwrap_or(1).max(1);

        let smoother_name = self.smoother.as_deref().unwrap_or("jacobi");
        let pc_type = PcType::from_str(smoother_name)?;
        if pc_type == PcType::None {
            return Err(KError::InvalidInput(
                "pc_mg_smoother cannot be none".into(),
            ));
        }
        let mut hierarchy = MgHierarchy::new(levels);
        for lvl in hierarchy.levels_mut().iter_mut().take(self.levels - 1) {
            let mut smoother = self.build_smoother(smoother_name)?;
            let op = CsrLinOp::new(lvl.operator.clone());
            smoother.setup(&op)?;
            lvl.smoother = Some(smoother);
        }

        let mut coarse_solver = self.build_smoother(smoother_name)?;
        let coarse_op = CsrLinOp::new(
            hierarchy
                .levels()
                .last()
                .ok_or_else(|| KError::InvalidInput("missing coarse level".into()))?
                .operator
                .clone(),
        );
        coarse_solver.setup(&coarse_op)?;
        let coarse_solve = match pc_type {
            PcType::Lu | PcType::Qr => MgCoarseSolve::Direct(coarse_solver),
            #[cfg(feature = "superlu_dist")]
            PcType::SuperLuDist => MgCoarseSolve::Direct(coarse_solver),
            _ => MgCoarseSolve::Smoother(coarse_solver, self.smoother_sweeps),
        };
        self.coarse_solve = Some(coarse_solve);
        self.hierarchy = Some(hierarchy);
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(
                "mg input/output length mismatch".into(),
            ));
        }
        y.fill(S::zero());
        self.mg_cycle(0, x, y, self.cycle)?;
        Ok(())
    }
}
