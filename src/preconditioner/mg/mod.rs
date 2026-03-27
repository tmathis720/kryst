use crate::algebra::prelude::*;
use crate::algebra::scalar::KrystScalar;
use crate::algebra::scalar::S;
use crate::config::options::{KspOptions, PcOptions};
use crate::context::pc_context::{PcFactory, PcType};
use crate::error::KError;
use crate::matrix::DistCsrOp;
use crate::matrix::op::LinOp;
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::utils::rap_opt_generic;
use crate::parallel::{Comm, UniverseComm, allreduce_sum_scalar_slice_in_place};
#[cfg(feature = "backend-faer")]
use crate::preconditioner::dist::DistCoarseSolverRoute;
use crate::preconditioner::ksp_pc::KspAsPc;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Scalar carried through MG hierarchy operators and transfers.
type MgScalar = S;

#[cfg(not(feature = "backend-faer"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DistCoarseSolverRoute {
    Auto,
    Root,
    Local,
    SuperLuDist,
}

#[derive(Clone, Debug, Default)]
pub struct MgLevelPolicy {
    pub level: usize,
    pub level_key: Option<String>,
    pub smoother_type: Option<String>,
    pub smoother_family: Option<String>,
    pub smoother_steps: Option<usize>,
    pub pre_sweeps: Option<usize>,
    pub post_sweeps: Option<usize>,
    pub smoother_side: Option<PcSide>,
    pub coarse_pc_type: Option<String>,
    pub coarse_ksp_type: Option<String>,
    pub coarse_ksp_maxits: Option<usize>,
    pub coarse_ksp_rtol: Option<f64>,
    pub coarse_side: Option<PcSide>,
    pub coarse_routes: Option<Vec<String>>,
    pub level_ksp_type: Option<String>,
    pub level_pc_type: Option<String>,
    pub level_ksp_maxits: Option<usize>,
    pub level_ksp_rtol: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct MgPerfCounters {
    pub setup_per_level: Vec<Duration>,
    pub apply_per_level: Vec<Duration>,
    pub comm_bytes_per_level: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct MgLevelDiagnostics {
    pub level: usize,
    pub nnz: usize,
    pub work_estimate: usize,
    pub reduction_count: usize,
    pub selected_pc: String,
    pub selected_ksp: Option<String>,
    pub pre_sweeps: usize,
    pub post_sweeps: usize,
    pub side: PcSide,
    pub coarse_route: Option<String>,
    pub route_fallback: Option<String>,
    pub native_complex_path: bool,
    pub complex_diagnostic: Option<String>,
    pub grid_complexity: f64,
    pub operator_complexity: f64,
    pub comm_bytes: usize,
}

#[derive(Clone, Debug)]
struct MgResolvedPolicy {
    smoother: String,
    pre_sweeps: usize,
    post_sweeps: usize,
    smoother_side: PcSide,
    coarse_pc_type: Option<String>,
    coarse_ksp_type: Option<String>,
    coarse_ksp_maxits: Option<usize>,
    coarse_ksp_rtol: Option<f64>,
    coarse_side: PcSide,
    coarse_routes: Vec<String>,
    level_ksp_type: Option<String>,
    level_pc_type: Option<String>,
    level_ksp_maxits: Option<usize>,
    level_ksp_rtol: Option<f64>,
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
    csr: Arc<CsrMatrix<MgScalar>>,
    comm: UniverseComm,
}

impl CsrLinOp {
    fn new(csr: Arc<CsrMatrix<MgScalar>>, comm: UniverseComm) -> Self {
        Self { csr, comm }
    }
}

impl LinOp for CsrLinOp {
    type S = MgScalar;

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
    pub operator: Arc<CsrMatrix<MgScalar>>,
    pub prolongation: Option<Arc<CsrMatrix<MgScalar>>>,
    pub restriction: Option<Arc<CsrMatrix<MgScalar>>>,
    pub dist_transfer: Option<MgDistTransferMeta>,
}

impl MgLevel {
    pub fn new(level: usize, operator: Arc<CsrMatrix<MgScalar>>) -> Self {
        Self {
            level,
            smoother: None,
            operator,
            prolongation: None,
            restriction: None,
            dist_transfer: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MgDistTransferMeta {
    pub fine_part: Arc<Vec<usize>>,
    pub coarse_part: Arc<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct MgDistHierarchyMeta {
    valid: bool,
    part_per_level: Vec<Arc<Vec<usize>>>,
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
    user_transfers: Vec<(usize, Arc<CsrMatrix<MgScalar>>, Arc<CsrMatrix<MgScalar>>)>,
    level_coarse_pc_types: BTreeMap<usize, String>,
    level_policies: Vec<MgLevelPolicy>,
    diagnostics: Vec<MgLevelDiagnostics>,
    perf: Arc<Mutex<MgPerfCounters>>,
    hierarchy_pattern_hash: Option<u64>,
    dist_meta: Option<MgDistHierarchyMeta>,
    coarse_route: DistCoarseSolverRoute,
    comm: UniverseComm,
}

pub struct MgTransferOperators {
    pub prolongation: Arc<CsrMatrix<MgScalar>>,
    pub restriction: Arc<CsrMatrix<MgScalar>>,
}

fn csr_from_linop_scalar(
    op: &dyn LinOp<S = S>,
    drop_tol: R,
) -> Result<Arc<CsrMatrix<MgScalar>>, KError> {
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
            perf: Arc::new(Mutex::new(MgPerfCounters::default())),
            hierarchy_pattern_hash: None,
            dist_meta: None,
            coarse_route: DistCoarseSolverRoute::Auto,
            comm: UniverseComm::NoComm(crate::parallel::NoComm),
        }
    }

    pub fn set_level_policies(&mut self, policies: Vec<MgLevelPolicy>) {
        self.level_policies = policies;
    }

    pub fn diagnostics(&self) -> &[MgLevelDiagnostics] {
        &self.diagnostics
    }

    pub fn perf_counters(&self) -> MgPerfCounters {
        self.perf.lock().map(|p| p.clone()).unwrap_or_default()
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

    fn build_level_solver(
        &self,
        policy: &MgResolvedPolicy,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        if let Some(ksp_type) = policy.level_ksp_type.as_ref() {
            let mut inner_pc = PcOptions {
                pc_type: policy
                    .level_pc_type
                    .clone()
                    .or_else(|| Some(policy.smoother.clone())),
                ..Default::default()
            };
            if inner_pc.pc_type.is_none() {
                inner_pc.pc_type = Some("jacobi".to_string());
            }
            return Ok(Box::new(KspAsPc::new(
                KspOptions {
                    ksp_type: Some(ksp_type.clone()),
                    maxits: policy.level_ksp_maxits,
                    rtol: policy.level_ksp_rtol,
                    ..Default::default()
                },
                inner_pc,
            )?));
        }
        self.build_smoother(&policy.smoother)
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
        let n_coarse = n_fine.div_ceil(coarse_div);

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

    fn smooth_level(
        level: &MgLevel,
        sweeps: usize,
        side: PcSide,
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
            smoother.apply(side, &residual, &mut correction)?;
            for i in 0..n {
                x[i] += correction[i];
            }
        }
        Ok(())
    }

    fn resolved_policy_for_level(&self, level: usize) -> MgResolvedPolicy {
        let default_smoother = self.smoother.as_deref().unwrap_or("jacobi").to_string();
        let mut resolved = MgResolvedPolicy {
            smoother: default_smoother,
            pre_sweeps: self.smoother_sweeps,
            post_sweeps: self.smoother_sweeps,
            smoother_side: PcSide::Left,
            coarse_pc_type: self.coarse_pc_type.clone(),
            coarse_ksp_type: self.coarse_ksp_type.clone(),
            coarse_ksp_maxits: self.coarse_ksp_maxits,
            coarse_ksp_rtol: self.coarse_ksp_rtol,
            coarse_side: PcSide::Left,
            coarse_routes: vec!["nested_ksp".to_string(), "pc_apply".to_string()],
            level_ksp_type: None,
            level_pc_type: None,
            level_ksp_maxits: None,
            level_ksp_rtol: None,
        };
        let is_coarse = level + 1 == self.levels;
        let is_fine = level == 0;
        let mut family_matches: Vec<&MgLevelPolicy> = self
            .level_policies
            .iter()
            .filter(|p| {
                p.level_key
                    .as_deref()
                    .map(|k| {
                        matches!(
                            k,
                            "all" | "any" | "fine" | "coarse" | "intermediate" | "mid"
                        )
                    })
                    .unwrap_or(false)
            })
            .collect();
        family_matches.sort_by_key(|p| p.level);
        let mut exact_matches: Vec<&MgLevelPolicy> = self
            .level_policies
            .iter()
            .filter(|p| p.level == level)
            .collect();
        exact_matches.sort_by_key(|p| p.level);
        for p in family_matches.into_iter().chain(exact_matches.into_iter()) {
            if let Some(key) = p.level_key.as_deref() {
                match key {
                    "all" | "any" => {}
                    "fine" if !is_fine => continue,
                    "coarse" if !is_coarse => continue,
                    "intermediate" | "mid" if is_fine || is_coarse => continue,
                    _ => {}
                }
            }
            if let Some(v) = p.smoother_family.as_ref().or(p.smoother_type.as_ref()) {
                resolved.smoother = v.clone();
            }
            if let Some(v) = p.smoother_steps {
                resolved.pre_sweeps = v;
                resolved.post_sweeps = v;
            }
            if let Some(v) = p.pre_sweeps {
                resolved.pre_sweeps = v;
            }
            if let Some(v) = p.post_sweeps {
                resolved.post_sweeps = v;
            }
            if let Some(v) = p.smoother_side {
                resolved.smoother_side = v;
            }
            if let Some(v) = p.coarse_pc_type.as_ref() {
                resolved.coarse_pc_type = Some(v.clone());
            }
            if let Some(v) = p.coarse_ksp_type.as_ref() {
                resolved.coarse_ksp_type = Some(v.clone());
            }
            if let Some(v) = p.coarse_ksp_maxits {
                resolved.coarse_ksp_maxits = Some(v);
            }
            if let Some(v) = p.coarse_ksp_rtol {
                resolved.coarse_ksp_rtol = Some(v);
            }
            if let Some(v) = p.coarse_side {
                resolved.coarse_side = v;
            }
            if let Some(v) = p.coarse_routes.as_ref() {
                resolved.coarse_routes = v.clone();
            }
            if let Some(v) = p.level_ksp_type.as_ref() {
                resolved.level_ksp_type = Some(v.clone());
            }
            if let Some(v) = p.level_pc_type.as_ref() {
                resolved.level_pc_type = Some(v.clone());
            }
            if let Some(v) = p.level_ksp_maxits {
                resolved.level_ksp_maxits = Some(v);
            }
            if let Some(v) = p.level_ksp_rtol {
                resolved.level_ksp_rtol = Some(v);
            }
        }
        if let Some(v) = self.level_coarse_pc_types.get(&level) {
            resolved.coarse_pc_type = Some(v.clone());
        }
        resolved
    }

    fn normalized_coarse_routes(policy: &MgResolvedPolicy) -> Vec<String> {
        let mut routes = Vec::new();
        for route in &policy.coarse_routes {
            let canonical = match route.trim().to_lowercase().as_str() {
                "ksp" | "nested_ksp" => "nested_ksp",
                "pc" | "pc_apply" | "apply" => "pc_apply",
                "root" | "root_gather" | "gather" => "root_gather",
                "local" | "local_prototype" | "prototype" => "local_prototype",
                "direct" | "direct_solve" => "direct",
                _ => continue,
            }
            .to_string();
            if !routes.contains(&canonical) {
                routes.push(canonical);
            }
        }
        if routes.is_empty() {
            routes.push("nested_ksp".to_string());
            routes.push("pc_apply".to_string());
        }
        routes
    }

    fn root_gather_allreduce(&self, v: &mut [S]) {
        if self.comm.size() > 1 {
            allreduce_sum_scalar_slice_in_place(&self.comm, v);
        }
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
        let t0 = Instant::now();
        let dist_valid = self.dist_meta.as_ref().map(|m| m.valid).unwrap_or(false);
        let is_coarse = level_ix + 1 == hierarchy.levels.len();
        if is_coarse {
            if let Some(coarse) = &self.coarse_solve {
                let coarse_side = self.resolved_policy_for_level(level_ix).coarse_side;
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
            if dist_valid && self.coarse_route == DistCoarseSolverRoute::Root {
                self.root_gather_allreduce(x);
            }
            if let Ok(mut perf) = self.perf.lock() {
                if perf.apply_per_level.len() <= level_ix {
                    perf.apply_per_level.resize(level_ix + 1, Duration::ZERO);
                }
                perf.apply_per_level[level_ix] += t0.elapsed();
            }
            return Ok(());
        }

        let level_policy = self.resolved_policy_for_level(level_ix);
        let pre_sweeps = level_policy.pre_sweeps;
        Self::smooth_level(level, pre_sweeps, level_policy.smoother_side, b, x)?;

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
        if dist_valid
            && matches!(
                self.coarse_route,
                DistCoarseSolverRoute::Root | DistCoarseSolverRoute::Auto
            )
        {
            self.root_gather_allreduce(&mut coarse_rhs);
        }
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
        if dist_valid
            && matches!(
                self.coarse_route,
                DistCoarseSolverRoute::Root | DistCoarseSolverRoute::Auto
            )
        {
            self.root_gather_allreduce(&mut coarse_sol);
        }

        let mut fine_correction = vec![S::zero(); prolongation.nrows()];
        prolongation.try_spmv(&coarse_sol, &mut fine_correction)?;
        for i in 0..x.len() {
            x[i] += fine_correction[i];
        }

        let post_sweeps = level_policy.post_sweeps;
        Self::smooth_level(level, post_sweeps, level_policy.smoother_side, b, x)?;
        if let Ok(mut perf) = self.perf.lock() {
            if perf.apply_per_level.len() <= level_ix {
                perf.apply_per_level.resize(level_ix + 1, Duration::ZERO);
            }
            perf.apply_per_level[level_ix] += t0.elapsed();
            if perf.comm_bytes_per_level.len() <= level_ix {
                perf.comm_bytes_per_level.resize(level_ix + 1, 0);
            }
            let bytes = restriction.nnz() * std::mem::size_of::<S>()
                + prolongation.nnz() * std::mem::size_of::<S>();
            perf.comm_bytes_per_level[level_ix] =
                perf.comm_bytes_per_level[level_ix].saturating_add(bytes);
        }
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
        let op_comm = _a.comm();
        let dist_op = _a.as_any().downcast_ref::<DistCsrOp>();
        let pattern_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            a.row_ptr().hash(&mut h);
            a.col_idx().hash(&mut h);
            h.finish()
        };
        if self.hierarchy.is_some() && self.hierarchy_pattern_hash == Some(pattern_hash) {
            return Ok(());
        }
        let mut levels = Vec::new();
        levels.push(MgLevel::new(0, a.clone()));
        let mut current = a;
        let mut dist_partitions: Vec<Arc<Vec<usize>>> = Vec::new();
        if let Some(dist) = dist_op {
            dist_partitions.push(dist.row_partition());
        }
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
                if let Some(fine_part) = dist_partitions.last().cloned() {
                    let coarse_part = Arc::new(
                        fine_part
                            .iter()
                            .map(|&v| v.div_ceil(2))
                            .collect::<Vec<usize>>(),
                    );
                    entry.dist_transfer = Some(MgDistTransferMeta {
                        fine_part,
                        coarse_part: coarse_part.clone(),
                    });
                    dist_partitions.push(coarse_part);
                }
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
        self.dist_meta = if !dist_partitions.is_empty() {
            let global_levels = op_comm.all_reduce_f64(self.levels as f64);
            let expected = (self.levels * op_comm.size()) as f64;
            Some(MgDistHierarchyMeta {
                valid: (global_levels - expected).abs() < 1e-12,
                part_per_level: dist_partitions,
            })
        } else {
            None
        };
        self.smoother_sweeps = self.smoother_steps.unwrap_or(1).max(1);

        let smoother_name = self.smoother.as_deref().unwrap_or("jacobi");
        let smoother_pc_type = PcType::from_str(smoother_name)?;
        if smoother_pc_type == PcType::None {
            return Err(KError::InvalidInput("pc_mg_smoother cannot be none".into()));
        }
        let mut hierarchy = MgHierarchy::new(levels);
        self.comm = op_comm.clone();
        for lvl in hierarchy.levels_mut().iter_mut().take(self.levels - 1) {
            let policy = self.resolved_policy_for_level(lvl.level);
            let mut smoother = self.build_level_solver(&policy)?;
            let op = CsrLinOp::new(lvl.operator.clone(), op_comm.clone());
            let ts = Instant::now();
            smoother.setup(&op)?;
            if let Ok(mut perf) = self.perf.lock() {
                if perf.setup_per_level.len() <= lvl.level {
                    perf.setup_per_level.resize(lvl.level + 1, Duration::ZERO);
                }
                perf.setup_per_level[lvl.level] += ts.elapsed();
            }
            lvl.smoother = Some(smoother);
        }

        let coarse_level = self.levels.saturating_sub(1);
        let coarse_policy = self.resolved_policy_for_level(coarse_level);
        let coarse_routes = Self::normalized_coarse_routes(&coarse_policy);
        self.coarse_route = coarse_routes
            .iter()
            .find_map(|route| match route.as_str() {
                "root_gather" => Some(DistCoarseSolverRoute::Root),
                "local_prototype" => Some(DistCoarseSolverRoute::Local),
                "direct" => Some(DistCoarseSolverRoute::SuperLuDist),
                _ => None,
            })
            .unwrap_or(DistCoarseSolverRoute::Auto);
        let coarse_pc_type = coarse_policy
            .coarse_pc_type
            .as_deref()
            .map(PcType::from_str)
            .transpose()?
            .unwrap_or(smoother_pc_type);
        let mut coarse_solver: Option<Box<dyn Preconditioner>> = None;
        for route in &coarse_routes {
            match route.as_str() {
                "nested_ksp" => {
                    if let Some(ksp_type) = coarse_policy.coarse_ksp_type.as_ref() {
                        let mut coarse_pc_opts = PcOptions {
                            pc_type: Some(Self::pc_type_name(coarse_pc_type).to_string()),
                            ..Default::default()
                        };
                        if coarse_pc_type == PcType::Ksp {
                            coarse_pc_opts.pc_ksp_pc_type = Some("jacobi".to_string());
                        }
                        if coarse_pc_opts.pc_type.is_none() {
                            coarse_pc_opts.pc_type = Some("jacobi".to_string());
                        }
                        coarse_solver = Some(Box::new(KspAsPc::new(
                            KspOptions {
                                ksp_type: Some(ksp_type.clone()),
                                maxits: coarse_policy.coarse_ksp_maxits,
                                rtol: coarse_policy.coarse_ksp_rtol,
                                ..Default::default()
                            },
                            coarse_pc_opts,
                        )?));
                        break;
                    }
                }
                "pc_apply" => {
                    coarse_solver = Some(PcFactory::create_preconditioner(coarse_pc_type, None)?);
                    break;
                }
                _ => {}
            }
        }
        let mut coarse_solver = if let Some(solver) = coarse_solver {
            solver
        } else if let Some(ksp_type) = coarse_policy.coarse_ksp_type.as_ref() {
            let mut coarse_pc_opts = PcOptions {
                pc_type: Some(Self::pc_type_name(coarse_pc_type).to_string()),
                ..Default::default()
            };
            if coarse_pc_type == PcType::Ksp {
                coarse_pc_opts.pc_ksp_pc_type = Some("jacobi".to_string());
            }
            if coarse_pc_opts.pc_type.is_none() {
                coarse_pc_opts.pc_type = Some("jacobi".to_string());
            }
            Box::new(KspAsPc::new(
                KspOptions {
                    ksp_type: Some(ksp_type.clone()),
                    maxits: coarse_policy.coarse_ksp_maxits,
                    rtol: coarse_policy.coarse_ksp_rtol,
                    ..Default::default()
                },
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
            .map(|lvl| {
                let policy = self.resolved_policy_for_level(lvl.level);
                MgLevelDiagnostics {
                    level: lvl.level,
                    nnz: lvl.operator.nnz(),
                    work_estimate: {
                        lvl.operator.nnz() * (policy.pre_sweeps + policy.post_sweeps)
                    },
                    reduction_count: 0,
                    selected_pc: policy
                        .level_pc_type
                        .clone()
                        .unwrap_or_else(|| policy.smoother.clone()),
                    selected_ksp: policy.level_ksp_type.clone(),
                    pre_sweeps: policy.pre_sweeps,
                    post_sweeps: policy.post_sweeps,
                    side: policy.smoother_side,
                    coarse_route: if lvl.level == coarse_level {
                        Some(coarse_routes.join("->"))
                    } else {
                        None
                    },
                    route_fallback: if lvl.level == coarse_level {
                        Some(coarse_routes.join(","))
                    } else {
                        None
                    },
                    native_complex_path: true,
                    complex_diagnostic: None,
                    grid_complexity: 0.0,
                    operator_complexity: 0.0,
                    comm_bytes: 0,
                }
            })
            .collect();
        if let Some(first) = hierarchy.levels().first() {
            let n0 = first.operator.nrows() as f64;
            let nnz0 = first.operator.nnz() as f64;
            let mut nsum = 0.0;
            let mut nnzsum = 0.0;
            for d in &mut self.diagnostics {
                nsum += hierarchy.levels()[d.level].operator.nrows() as f64;
                nnzsum += d.nnz as f64;
                d.grid_complexity = if n0 > 0.0 { nsum / n0 } else { 0.0 };
                d.operator_complexity = if nnz0 > 0.0 { nnzsum / nnz0 } else { 0.0 };
                if let Ok(perf) = self.perf.lock() {
                    d.comm_bytes = perf.comm_bytes_per_level.get(d.level).copied().unwrap_or(0);
                }
            }
        }
        self.hierarchy = Some(hierarchy);
        self.hierarchy_pattern_hash = Some(pattern_hash);
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
        if self
            .dist_meta
            .as_ref()
            .map(|m| m.valid && !m.part_per_level.is_empty())
            .unwrap_or(false)
        {
            PcDistributedSupport::Distributed
        } else {
            PcDistributedSupport::LocalOnly
        }
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

    #[test]
    fn mg_level_policy_is_explicit_per_level_not_cascaded() {
        let mut mg = MgPc::new(
            4,
            Some("v".into()),
            Some("jacobi".into()),
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        mg.set_level_policies(vec![MgLevelPolicy {
            level: 1,
            smoother_type: Some("gs".into()),
            smoother_family: None,
            smoother_steps: Some(3),
            ..Default::default()
        }]);
        let l1 = mg.resolved_policy_for_level(1);
        let l2 = mg.resolved_policy_for_level(2);
        assert_eq!(l1.smoother, "gs");
        assert_eq!(l1.pre_sweeps, 3);
        assert_eq!(l2.smoother, "jacobi");
        assert_eq!(l2.pre_sweeps, 1);
    }

    #[test]
    fn mg_level_policy_supports_mixed_ksp_pc_and_sweep_budgets() {
        let mut mg = MgPc::new(
            4,
            Some("v".into()),
            Some("jacobi".into()),
            Some(1),
            None,
            None,
            None,
            Some("ilu0".into()),
            Some("cg".into()),
            Some(5),
            Some(1e-3),
        );
        mg.set_level_policies(vec![MgLevelPolicy {
            level: 1,
            level_ksp_type: Some("gmres".into()),
            level_pc_type: Some("sor".into()),
            level_ksp_maxits: Some(4),
            pre_sweeps: Some(3),
            post_sweeps: Some(1),
            coarse_routes: Some(vec!["pc_apply".into(), "nested_ksp".into()]),
            ..Default::default()
        }]);

        let l1 = mg.resolved_policy_for_level(1);
        assert_eq!(l1.level_ksp_type.as_deref(), Some("gmres"));
        assert_eq!(l1.level_pc_type.as_deref(), Some("sor"));
        assert_eq!(l1.level_ksp_maxits, Some(4));
        assert_eq!(l1.pre_sweeps, 3);
        assert_eq!(l1.post_sweeps, 1);
        assert_eq!(l1.coarse_routes, vec!["pc_apply", "nested_ksp"]);

        let l2 = mg.resolved_policy_for_level(2);
        assert_eq!(l2.level_ksp_type, None);
        assert_eq!(l2.pre_sweeps, 1);
        assert_eq!(l2.post_sweeps, 1);
    }

    #[test]
    fn mg_coarse_route_normalization_is_deterministic() {
        let policy = MgResolvedPolicy {
            smoother: "jacobi".into(),
            pre_sweeps: 1,
            post_sweeps: 1,
            smoother_side: PcSide::Left,
            coarse_pc_type: None,
            coarse_ksp_type: None,
            coarse_ksp_maxits: None,
            coarse_ksp_rtol: None,
            coarse_side: PcSide::Left,
            coarse_routes: vec!["ksp".into(), "pc".into(), "ksp".into(), "bogus".into()],
            level_ksp_type: None,
            level_pc_type: None,
            level_ksp_maxits: None,
            level_ksp_rtol: None,
        };
        assert_eq!(
            MgPc::normalized_coarse_routes(&policy),
            vec!["nested_ksp".to_string(), "pc_apply".to_string()]
        );
    }

    #[test]
    fn mg_level_policy_precedence_global_family_exact() {
        let mut mg = MgPc::new(
            4,
            Some("v".into()),
            Some("jacobi".into()),
            Some(1),
            None,
            None,
            None,
            Some("ilu0".into()),
            None,
            None,
            None,
        );
        mg.set_level_policies(vec![
            MgLevelPolicy {
                level: 0,
                level_key: Some("all".into()),
                smoother_type: Some("sor".into()),
                coarse_pc_type: Some("jacobi".into()),
                smoother_steps: Some(2),
                ..Default::default()
            },
            MgLevelPolicy {
                level: 0,
                level_key: Some("coarse".into()),
                coarse_ksp_type: Some("cg".into()),
                ..Default::default()
            },
            MgLevelPolicy {
                level: 2,
                smoother_type: Some("gs".into()),
                pre_sweeps: Some(5),
                ..Default::default()
            },
        ]);
        let l2 = mg.resolved_policy_for_level(2);
        assert_eq!(l2.smoother, "gs");
        assert_eq!(l2.pre_sweeps, 5);
        assert_eq!(l2.post_sweeps, 2);
        let lc = mg.resolved_policy_for_level(3);
        assert_eq!(lc.smoother, "sor");
        assert_eq!(lc.coarse_pc_type.as_deref(), Some("jacobi"));
        assert_eq!(lc.coarse_ksp_type.as_deref(), Some("cg"));
    }

    #[test]
    fn mg_level_specific_coarse_map_takes_precedence() {
        let mut mg = MgPc::new(
            3,
            Some("v".into()),
            Some("jacobi".into()),
            Some(1),
            None,
            None,
            None,
            Some("ilu0".into()),
            None,
            None,
            None,
        );
        mg.set_level_policies(vec![MgLevelPolicy {
            level: 2,
            coarse_pc_type: Some("jacobi".into()),
            ..Default::default()
        }]);
        mg.set_level_coarse_solver_type(2, "lu")
            .expect("set override map");
        let coarse = mg.resolved_policy_for_level(2);
        assert_eq!(coarse.coarse_pc_type.as_deref(), Some("lu"));
    }
}
