use crate::algebra::prelude::*;
use crate::algebra::scalar::KrystScalar;
use crate::algebra::scalar::S;
use crate::config::options::{KspOptions, PcOptions};
use crate::context::pc_context::{PcFactory, PcType};
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::utils::rap_opt_generic;
use crate::parallel::UniverseComm;
use crate::preconditioner::ksp_pc::KspAsPc;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default)]
pub struct MgLevelPolicy {
    pub level: usize,
    pub smoother_type: Option<String>,
    pub smoother_steps: Option<usize>,
    pub smoother_side: Option<PcSide>,
    pub coarse_pc_type: Option<String>,
    pub coarse_ksp_type: Option<String>,
    pub coarse_ksp_maxits: Option<usize>,
    pub coarse_ksp_rtol: Option<f64>,
    pub coarse_side: Option<PcSide>,
}

#[derive(Clone, Debug)]
pub struct MgLevelDiagnostics {
    pub level: usize,
    pub nnz: usize,
    pub work_estimate: usize,
    pub reduction_count: usize,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MgCoarsenType {
    Injection,
    Linear,
    Aggregation,
}

impl MgCoarsenType {
    fn from_option(v: Option<&str>) -> Result<Self, KError> {
        match v.unwrap_or("linear") {
            "injection" | "inject" => Ok(Self::Injection),
            "linear" | "interp" => Ok(Self::Linear),
            "aggregation" | "agg" => Ok(Self::Aggregation),
            other => Err(KError::InvalidInput(format!(
                "unknown pc_mg_coarsen_type: {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MgInterpType {
    Injection,
    Linear,
}

impl MgInterpType {
    fn from_option(v: Option<&str>) -> Result<Self, KError> {
        match v.unwrap_or("linear") {
            "injection" | "inject" => Ok(Self::Injection),
            "linear" => Ok(Self::Linear),
            other => Err(KError::InvalidInput(format!(
                "unknown pc_mg_interpolation_type: {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MgRestrictType {
    Injection,
    FullWeighting,
}

impl MgRestrictType {
    fn from_option(v: Option<&str>) -> Result<Self, KError> {
        match v.unwrap_or("full_weighting") {
            "injection" | "inject" => Ok(Self::Injection),
            "full_weighting" | "full" | "fw" => Ok(Self::FullWeighting),
            other => Err(KError::InvalidInput(format!(
                "unknown pc_mg_restriction_type: {other}"
            ))),
        }
    }
}

enum MgCoarseSolve {
    Direct(Box<dyn Preconditioner>),
    Smoother(Box<dyn Preconditioner>, usize),
}

#[derive(Clone)]
struct CsrLinOp {
    csr: Arc<CsrMatrix<S>>,
    comm: UniverseComm,
}

impl CsrLinOp {
    fn new(csr: Arc<CsrMatrix<S>>, comm: UniverseComm) -> Self {
        Self { csr, comm }
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
        self.csr.as_ref()
    }

    fn comm(&self) -> UniverseComm {
        self.comm.clone()
    }
}

pub struct MgLevel {
    pub level: usize,
    pub smoother: Option<Box<dyn Preconditioner>>,
    pub operator: Arc<CsrMatrix<S>>,
    pub prolongation: Option<Arc<CsrMatrix<S>>>,
    pub restriction: Option<Arc<CsrMatrix<S>>>,
}

impl MgLevel {
    pub fn new(level: usize, operator: Arc<CsrMatrix<S>>) -> Self {
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
    pub coarsen_type: Option<String>,
    pub interpolation_type: Option<String>,
    pub restriction_type: Option<String>,
    pub coarse_pc_type: Option<String>,
    pub coarse_ksp_type: Option<String>,
    pub coarse_ksp_maxits: Option<usize>,
    pub coarse_ksp_rtol: Option<f64>,
    hierarchy: Option<MgHierarchy>,
    coarse_solve: Option<Mutex<MgCoarseSolve>>,
    cycle: MgCycleType,
    smoother_sweeps: usize,
    coarsen: MgCoarsenType,
    interp: MgInterpType,
    restrict: MgRestrictType,
    user_transfers: Vec<(usize, Arc<CsrMatrix<S>>, Arc<CsrMatrix<S>>)>,
    level_coarse_pc_types: BTreeMap<usize, String>,
    level_policies: Vec<MgLevelPolicy>,
    diagnostics: Vec<MgLevelDiagnostics>,
    comm: UniverseComm,
}

pub struct MgTransferOperators {
    pub prolongation: Arc<CsrMatrix<S>>,
    pub restriction: Arc<CsrMatrix<S>>,
}

fn csr_from_linop_scalar(op: &dyn LinOp<S = S>, drop_tol: R) -> Result<Arc<CsrMatrix<S>>, KError> {
    if let Some(csr) = op.as_any().downcast_ref::<CsrMatrix<S>>() {
        return Ok(Arc::new(csr.clone()));
    }

    let (m, n) = op.dims();
    let mut rows: Vec<Vec<(usize, S)>> = vec![Vec::new(); m];
    let mut e = vec![S::zero(); n];
    let mut y = vec![S::zero(); m];
    for j in 0..n {
        e[j] = S::one();
        op.try_matvec(&e, &mut y)?;
        e[j] = S::zero();
        for i in 0..m {
            let v = y[i];
            if v.abs() > drop_tol {
                rows[i].push((j, v));
            }
        }
    }

    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for entries in rows.iter_mut() {
        entries.sort_unstable_by_key(|(j, _)| *j);
        for (j, v) in entries.iter().copied() {
            col_idx.push(j);
            values.push(v);
        }
        row_ptr.push(col_idx.len());
    }
    Ok(Arc::new(CsrMatrix::from_csr(
        m, n, row_ptr, col_idx, values,
    )))
}

impl MgPc {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        levels: usize,
        cycle_type: Option<String>,
        smoother: Option<String>,
        smoother_steps: Option<usize>,
        coarsen_type: Option<String>,
        interpolation_type: Option<String>,
        restriction_type: Option<String>,
        coarse_pc_type: Option<String>,
        coarse_ksp_type: Option<String>,
        coarse_ksp_maxits: Option<usize>,
        coarse_ksp_rtol: Option<f64>,
    ) -> Self {
        let cycle = MgCycleType::from_option(cycle_type.as_deref()).unwrap_or(MgCycleType::V);
        let smoother_sweeps = smoother_steps.unwrap_or(1).max(1);
        let coarsen =
            MgCoarsenType::from_option(coarsen_type.as_deref()).unwrap_or(MgCoarsenType::Linear);
        let interp = MgInterpType::from_option(interpolation_type.as_deref())
            .unwrap_or(MgInterpType::Linear);
        let restrict = MgRestrictType::from_option(restriction_type.as_deref())
            .unwrap_or(MgRestrictType::FullWeighting);
        Self {
            levels,
            cycle_type,
            smoother,
            smoother_steps,
            coarsen_type,
            interpolation_type,
            restriction_type,
            coarse_pc_type,
            coarse_ksp_type,
            coarse_ksp_maxits,
            coarse_ksp_rtol,
            hierarchy: None,
            coarse_solve: None,
            cycle,
            smoother_sweeps,
            coarsen,
            interp,
            restrict,
            user_transfers: Vec::new(),
            level_coarse_pc_types: BTreeMap::new(),
            level_policies: Vec::new(),
            diagnostics: Vec::new(),
            comm: UniverseComm::NoComm(crate::parallel::NoComm),
        }
    }

    pub fn set_level_policies(&mut self, policies: Vec<MgLevelPolicy>) {
        self.level_policies = policies;
    }

    pub fn diagnostics(&self) -> &[MgLevelDiagnostics] {
        &self.diagnostics
    }

    pub fn set_level_transfer_operators(
        &mut self,
        level: usize,
        operators: MgTransferOperators,
    ) -> Result<(), KError> {
        if level + 1 >= self.levels {
            return Err(KError::InvalidInput(format!(
                "level {level} cannot own transfer operators for {} levels",
                self.levels
            )));
        }
        self.user_transfers
            .retain(|(existing, _, _)| *existing != level);
        self.user_transfers
            .push((level, operators.prolongation, operators.restriction));
        Ok(())
    }

    pub fn set_level_transfer_from_linops(
        &mut self,
        level: usize,
        prolongation: &dyn LinOp<S = S>,
        restriction: &dyn LinOp<S = S>,
    ) -> Result<(), KError> {
        let p = csr_from_linop_scalar(prolongation, 0.0)?;
        let r = csr_from_linop_scalar(restriction, 0.0)?;
        self.set_level_transfer_operators(
            level,
            MgTransferOperators {
                prolongation: p,
                restriction: r,
            },
        )
    }

    pub fn set_level_coarse_solver_type(
        &mut self,
        level: usize,
        pc_type: impl Into<String>,
    ) -> Result<(), KError> {
        if level >= self.levels {
            return Err(KError::InvalidInput(format!(
                "level {level} out of range for {} levels",
                self.levels
            )));
        }
        let value = pc_type.into().to_lowercase();
        let _ = PcType::from_str(&value)?;
        self.level_coarse_pc_types.insert(level, value);
        Ok(())
    }
    pub fn hierarchy(&self) -> &MgHierarchy {
        self.hierarchy
            .as_ref()
            .expect("MgPc::hierarchy requires setup")
    }

    fn pc_type_name(pc: PcType) -> &'static str {
        match pc {
            PcType::Jacobi => "jacobi",
            PcType::Ilu0 => "ilu0",
            PcType::None => "none",
            PcType::Ilu => "ilu",
            PcType::Ilut => "ilut",
            PcType::Ilutp => "ilutp",
            PcType::Ilup => "ilup",
            PcType::BlockJacobi => "block_jacobi",
            PcType::Sor => "sor",
            PcType::Asm => "asm",
            PcType::Chebyshev => "chebyshev",
            PcType::Amg => "amg",
            PcType::ApproxInverse => "approxinv",
            PcType::FieldSplit => "fieldsplit",
            PcType::Shell => "shell",
            PcType::Ksp => "ksp",
            PcType::Mg => "mg",
            PcType::Bddc => "bddc",
            PcType::Gamg => "gamg",
            PcType::Lu => "lu",
            PcType::Qr => "qr",
            #[cfg(feature = "superlu_dist")]
            PcType::SuperLuDist => "superludist",
        }
    }

    fn build_smoother(&self, name: &str) -> Result<Box<dyn Preconditioner>, KError> {
        let pc_type = PcType::from_str(name)?;
        if pc_type == PcType::Mg {
            return Err(KError::InvalidInput("pc_mg_smoother cannot be mg".into()));
        }
        if pc_type == PcType::None {
            return Err(KError::InvalidInput("pc_mg_smoother cannot be none".into()));
        }
        PcFactory::create_preconditioner(pc_type, None)
    }

    fn build_transfer(
        n_fine: usize,
        coarsen: MgCoarsenType,
        interp: MgInterpType,
        restrict: MgRestrictType,
    ) -> (CsrMatrix<S>, CsrMatrix<S>, usize) {
        let coarse_div = match coarsen {
            MgCoarsenType::Aggregation => 3,
            MgCoarsenType::Injection | MgCoarsenType::Linear => 2,
        };
        let n_coarse = (n_fine + coarse_div - 1) / coarse_div;

        let mut p_row_ptr = Vec::with_capacity(n_fine + 1);
        let mut p_col_idx = Vec::with_capacity(n_fine * 2);
        let mut p_values = Vec::with_capacity(n_fine * 2);
        p_row_ptr.push(0);
        for i in 0..n_fine {
            let coarse = i / coarse_div;
            match interp {
                MgInterpType::Injection => {
                    p_col_idx.push(coarse.min(n_coarse.saturating_sub(1)));
                    p_values.push(S::one());
                }
                MgInterpType::Linear => {
                    let j0 = coarse.min(n_coarse.saturating_sub(1));
                    p_col_idx.push(j0);
                    p_values.push(S::one());
                    if i % coarse_div != 0 {
                        let j1 = (j0 + 1).min(n_coarse.saturating_sub(1));
                        if j1 != j0 {
                            p_col_idx.push(j1);
                            p_values.push(S::from_real(0.5));
                        }
                    }
                }
            }
            p_row_ptr.push(p_col_idx.len());
        }

        let mut r_row_ptr = Vec::with_capacity(n_coarse + 1);
        let mut r_col_idx = Vec::with_capacity(n_fine);
        let mut r_values = Vec::with_capacity(n_fine);
        r_row_ptr.push(0);
        for j in 0..n_coarse {
            let start = coarse_div * j;
            let end = (start + coarse_div).min(n_fine);
            for i in start..end {
                r_col_idx.push(i);
                let w = match restrict {
                    MgRestrictType::Injection => {
                        if i == start {
                            S::one()
                        } else {
                            S::zero()
                        }
                    }
                    MgRestrictType::FullWeighting => S::from_real(1.0 / ((end - start) as f64)),
                };
                r_values.push(w);
            }
            r_row_ptr.push(r_col_idx.len());
        }

        let p = CsrMatrix::from_csr(n_fine, n_coarse, p_row_ptr, p_col_idx, p_values);
        let r = CsrMatrix::from_csr(n_coarse, n_fine, r_row_ptr, r_col_idx, r_values);
        (p, r, n_coarse)
    }

    fn smooth_level(level: &MgLevel, sweeps: usize, b: &[S], x: &mut [S]) -> Result<(), KError> {
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

    fn policy_for_level(&self, level: usize) -> Option<&MgLevelPolicy> {
        self.level_policies
            .iter()
            .filter(|p| p.level <= level)
            .max_by_key(|p| p.level)
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
                let coarse_side = self
                    .policy_for_level(level_ix)
                    .and_then(|p| p.coarse_side)
                    .unwrap_or(PcSide::Left);
                let mut guard = coarse
                    .lock()
                    .map_err(|_| KError::SolveError("mg coarse solver mutex poisoned".into()))?;
                match &mut *guard {
                    MgCoarseSolve::Direct(pc) => {
                        let op = CsrLinOp::new(level.operator.clone(), self.comm.clone());
                        if let Err(err) = pc.direct_solve(&op, b, x) {
                            log::warn!("coarse direct_solve failed ({err}); falling back to apply");
                            pc.apply(coarse_side, b, x)?;
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
                            pc.apply(coarse_side, &residual, &mut correction)?;
                            for i in 0..b.len() {
                                x[i] += correction[i];
                            }
                        }
                    }
                }
            }
            return Ok(());
        }

        let pre_sweeps = self
            .policy_for_level(level_ix)
            .and_then(|p| p.smoother_steps)
            .unwrap_or(self.smoother_sweeps);
        Self::smooth_level(level, pre_sweeps, b, x)?;

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

        let post_sweeps = self
            .policy_for_level(level_ix)
            .and_then(|p| p.smoother_steps)
            .unwrap_or(self.smoother_sweeps);
        Self::smooth_level(level, post_sweeps, b, x)?;
        Ok(())
    }
}

impl Preconditioner for MgPc {
    fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
        if self.levels < 2 {
            return Err(KError::InvalidInput("pc_mg_levels must be >= 2".into()));
        }
        self.cycle = MgCycleType::from_option(self.cycle_type.as_deref())?;
        self.coarsen = MgCoarsenType::from_option(self.coarsen_type.as_deref())?;
        self.interp = MgInterpType::from_option(self.interpolation_type.as_deref())?;
        self.restrict = MgRestrictType::from_option(self.restriction_type.as_deref())?;

        let a = csr_from_linop_scalar(_a, 0.0)?;
        let mut levels = Vec::new();
        levels.push(MgLevel::new(0, a.clone()));
        let mut current = a;
        for level in 0..(self.levels - 1) {
            let user_tr = self
                .user_transfers
                .iter()
                .find(|(idx, _, _)| *idx == level)
                .map(|(_, p, r)| (p.clone(), r.clone()));
            let (p, r, n_coarse) = if let Some((p, r)) = user_tr {
                if p.nrows() != current.nrows() || r.ncols() != current.nrows() {
                    return Err(KError::InvalidInput(format!(
                        "user transfer dimensions incompatible at level {level}"
                    )));
                }
                {
                    let n = r.nrows();
                    (p, r, n)
                }
            } else {
                let (p, r, n_coarse) =
                    Self::build_transfer(current.nrows(), self.coarsen, self.interp, self.restrict);
                (Arc::new(p), Arc::new(r), n_coarse)
            };
            let coarse = rap_opt_generic(r.as_ref(), current.as_ref(), p.as_ref())?;
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
        self.smoother_sweeps = self.smoother_steps.unwrap_or(1).max(1);

        let smoother_name = self.smoother.as_deref().unwrap_or("jacobi");
        let smoother_pc_type = PcType::from_str(smoother_name)?;
        if smoother_pc_type == PcType::None {
            return Err(KError::InvalidInput("pc_mg_smoother cannot be none".into()));
        }
        let mut hierarchy = MgHierarchy::new(levels);
        let op_comm = _a.comm();
        self.comm = op_comm.clone();
        for lvl in hierarchy.levels_mut().iter_mut().take(self.levels - 1) {
            let policy = self.policy_for_level(lvl.level);
            let policy_smoother = policy
                .and_then(|p| p.smoother_type.as_deref())
                .unwrap_or(smoother_name);
            let mut smoother = self.build_smoother(policy_smoother)?;
            let op = CsrLinOp::new(lvl.operator.clone(), op_comm.clone());
            smoother.setup(&op)?;
            lvl.smoother = Some(smoother);
        }

        let coarse_level = self.levels.saturating_sub(1);
        let coarse_override = self
            .policy_for_level(coarse_level)
            .and_then(|p| p.coarse_pc_type.as_deref())
            .or_else(|| {
                self.level_coarse_pc_types
                    .iter()
                    .filter(|(lvl, _)| **lvl <= coarse_level)
                    .max_by_key(|(lvl, _)| *lvl)
                    .map(|(_, v)| v.as_str())
            });
        let coarse_policy = self.policy_for_level(coarse_level);
        let coarse_pc_type = coarse_override
            .or(self.coarse_pc_type.as_deref())
            .map(PcType::from_str)
            .transpose()?
            .unwrap_or(smoother_pc_type);
        let mut coarse_solver: Box<dyn Preconditioner> = if let Some(ksp_type) = coarse_policy
            .and_then(|p| p.coarse_ksp_type.as_ref())
            .or(self.coarse_ksp_type.as_ref())
        {
            let mut coarse_pc_opts = PcOptions {
                pc_type: Some(Self::pc_type_name(coarse_pc_type).to_string()),
                ..Default::default()
            };
            if coarse_pc_type == PcType::Ksp {
                coarse_pc_opts.pc_ksp_pc_type = Some("jacobi".to_string());
            }
            Box::new(KspAsPc::new(
                Some(Self::pc_type_name(coarse_pc_type).to_string()),
                Some(ksp_type.clone()),
                coarse_policy
                    .and_then(|p| p.coarse_ksp_maxits)
                    .or(self.coarse_ksp_maxits)
                    .unwrap_or(8),
                coarse_policy
                    .and_then(|p| p.coarse_ksp_rtol)
                    .or(self.coarse_ksp_rtol)
                    .unwrap_or(1e-10),
                Some(KspOptions {
                    ksp_type: Some(ksp_type.clone()),
                    maxits: coarse_policy
                        .and_then(|p| p.coarse_ksp_maxits)
                        .or(self.coarse_ksp_maxits),
                    rtol: coarse_policy
                        .and_then(|p| p.coarse_ksp_rtol)
                        .or(self.coarse_ksp_rtol),
                    ..Default::default()
                }),
                coarse_pc_opts,
            )?)
        } else {
            PcFactory::create_preconditioner(coarse_pc_type, None)?
        };

        let coarse_op = CsrLinOp::new(
            hierarchy
                .levels()
                .last()
                .ok_or_else(|| KError::InvalidInput("missing coarse level".into()))?
                .operator
                .clone(),
            op_comm,
        );
        coarse_solver.setup(&coarse_op)?;
        let coarse_solve = match coarse_pc_type {
            PcType::Lu | PcType::Qr => MgCoarseSolve::Direct(coarse_solver),
            #[cfg(feature = "superlu_dist")]
            PcType::SuperLuDist => MgCoarseSolve::Direct(coarse_solver),
            _ => MgCoarseSolve::Smoother(coarse_solver, self.smoother_sweeps),
        };
        self.coarse_solve = Some(Mutex::new(coarse_solve));
        self.diagnostics = hierarchy
            .levels()
            .iter()
            .map(|lvl| MgLevelDiagnostics {
                level: lvl.level,
                nnz: lvl.operator.nnz(),
                work_estimate: lvl.operator.nnz()
                    * self
                        .policy_for_level(lvl.level)
                        .and_then(|p| p.smoother_steps)
                        .unwrap_or(self.smoother_sweeps),
                reduction_count: 0,
            })
            .collect();
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

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::LocalOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mg_user_transfer_overrides_default() {
        let mut mg = MgPc::new(
            3,
            Some("v".into()),
            Some("jacobi".into()),
            Some(1),
            Some("injection".into()),
            Some("injection".into()),
            Some("injection".into()),
            None,
            None,
            None,
            None,
        );
        let p = Arc::new(CsrMatrix::from_csr(
            4,
            2,
            vec![0, 1, 2, 3, 4],
            vec![0, 0, 1, 1],
            vec![S::one(); 4],
        ));
        let r = Arc::new(CsrMatrix::from_csr(
            2,
            4,
            vec![0, 2, 4],
            vec![0, 1, 2, 3],
            vec![S::from_real(0.5); 4],
        ));
        mg.set_level_transfer_operators(
            0,
            MgTransferOperators {
                prolongation: p,
                restriction: r,
            },
        )
        .expect("set transfer");
        assert_eq!(mg.user_transfers.len(), 1);
    }

    #[test]
    fn mg_transfer_variants_parse() {
        assert!(MgCoarsenType::from_option(Some("aggregation")).is_ok());
        assert!(MgInterpType::from_option(Some("linear")).is_ok());
        assert!(MgRestrictType::from_option(Some("full_weighting")).is_ok());
    }

    #[test]
    fn mg_level_coarse_solver_override_records() {
        let mut mg = MgPc::new(
            3,
            Some("v".into()),
            Some("jacobi".into()),
            Some(1),
            Some("linear".into()),
            Some("linear".into()),
            Some("full_weighting".into()),
            None,
            None,
            None,
            None,
        );
        mg.set_level_coarse_solver_type(1, "ilu0")
            .expect("set level coarse solver");
        assert_eq!(
            mg.level_coarse_pc_types.get(&1).map(String::as_str),
            Some("ilu0")
        );
    }
}
