use crate::algebra::prelude::*;
use crate::config::options::PcOptions;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{Op, OpFormat, PcCaps, PcDistributedSupport, PcSide, Preconditioner};

#[cfg(feature = "backend-faer")]
use crate::config::kinds::{AmgCoarsenKind, AmgInterpKind};
#[cfg(feature = "backend-faer")]
use crate::preconditioner::amg::{
    AMG, AMGConfig, AmgTransferOperators, CoarseSolve, CoarsenType, InterpType, RelaxPhase,
    RelaxType,
};
#[cfg(feature = "backend-faer")]
use crate::preconditioner::dist::{
    DistCoarseRepartition, DistCoarseSolverRoute, DistCoarseStrategy,
};
#[cfg(feature = "backend-faer")]
use std::str::FromStr;

#[cfg(feature = "backend-faer")]
#[derive(Clone, Debug, Default)]
pub struct GamgLevelPolicy {
    pub level: usize,
    pub level_key: Option<String>,
    pub smoother: Option<String>,
    pub smoother_family: Option<String>,
    pub sweeps: Option<usize>,
    pub pre_sweeps: Option<usize>,
    pub post_sweeps: Option<usize>,
    pub coarse_solver: Option<CoarseSolve>,
    pub side: Option<PcSide>,
    pub ksp_type: Option<String>,
    pub pc_type: Option<String>,
    pub ksp_maxits: Option<usize>,
    pub ksp_rtol: Option<f64>,
    pub coarse_routes: Option<Vec<DistCoarseSolverRoute>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamgType {
    Agg,
    Classical,
}

impl GamgType {
    fn from_str(value: &str) -> Result<Self, KError> {
        match value.to_lowercase().as_str() {
            "agg" | "aggregate" => Ok(GamgType::Agg),
            "classical" => Ok(GamgType::Classical),
            other => Err(KError::InvalidInput(format!(
                "unsupported pc_gamg_type: {other}"
            ))),
        }
    }
}

#[cfg(feature = "backend-faer")]
#[derive(Clone, Debug)]
pub struct GamgConfig {
    pub gamg_type: GamgType,
    pub amg_config: AMGConfig,
    pub level_policies: Vec<GamgLevelPolicy>,
    pub dist_coarse_route_fallback: Vec<DistCoarseSolverRoute>,
}

#[cfg(not(feature = "backend-faer"))]
#[derive(Clone, Debug)]
pub struct GamgConfig;

#[cfg(feature = "backend-faer")]
impl GamgConfig {
    pub fn try_from_opts(opts: &PcOptions) -> Result<Self, KError> {
        let mut amg_config = AMGConfig::default();
        let gamg_type = opts
            .pc_gamg_type
            .as_deref()
            .map(GamgType::from_str)
            .transpose()?
            .unwrap_or(GamgType::Agg);
        apply_petsc_gamg_defaults(&mut amg_config, gamg_type);

        if let Some(levels) = opts.pc_gamg_levels {
            if levels == 0 {
                return Err(KError::InvalidInput("pc_gamg_levels must be >= 1".into()));
            }
            amg_config.max_levels = levels;
        }
        if let Some(threshold) = opts.pc_gamg_threshold {
            if !threshold.is_finite() || !(0.0 < threshold && threshold <= 1.0) {
                return Err(KError::InvalidInput(
                    "pc_gamg_threshold must be finite and in (0, 1]".into(),
                ));
            }
            amg_config.strong_threshold = threshold;
        }
        if let Some(coarsen_type) = opts.pc_gamg_coarsen_type.as_deref() {
            amg_config.coarsen_type = map_gamg_coarsen_type(coarsen_type)?;
        }
        if let Some(interp_type) = opts.pc_gamg_interp_type.as_deref() {
            amg_config.interp_type = map_gamg_interp_type(interp_type)?;
        }
        if let Some(levels) = opts.pc_gamg_aggressive_levels {
            if levels == 0 {
                return Err(KError::InvalidInput(
                    "pc_gamg_aggressive_levels must be >= 1".into(),
                ));
            }
            amg_config.agg_num_levels = levels;
        }
        if let Some(mis_k) = opts.pc_gamg_aggressive_mis_k {
            if mis_k < 2 {
                return Err(KError::InvalidInput(
                    "pc_gamg_aggressive_mis_k must be >= 2".into(),
                ));
            }
            amg_config.aggressive_mis_k = mis_k;
        }
        if let Some(mode) = opts.amg_dist_apply_mode.as_deref() {
            amg_config.dist_coarse_strategy = parse_gamg_dist_mode(mode)?;
        }
        if let Some(enabled) = opts.amg_dist_instrumentation {
            amg_config.dist_apply_instrumentation = enabled;
        }
        if let Some(policy) = opts.amg_dist_coarse_repartition.as_deref() {
            amg_config.dist_coarse_repartition = DistCoarseRepartition::from_str(policy)?;
        }
        if let Some(route) = opts.amg_dist_coarse_solver_route.as_deref() {
            amg_config.dist_coarse_solver_route = parse_route_head(route)?;
        }
        if let Some(policy) = opts.amg_dist_coarse_policy.as_deref() {
            amg_config.dist_coarse_strategy = DistCoarseStrategy::from_str(policy)?;
        }

        let mut merged: std::collections::BTreeMap<(usize, Option<String>), GamgLevelPolicy> =
            std::collections::BTreeMap::new();
        for policy in opts
            .pc_gamg_level_policies
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|entry| parse_gamg_level_policy(entry).ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
        {
            let key = (policy.level, policy.level_key.clone());
            let entry = merged
                .entry(key)
                .or_insert_with(|| GamgLevelPolicy {
                    level: policy.level,
                    ..Default::default()
                });
            merge_gamg_policy(entry, &policy);
        }
        for (level, scoped) in &opts.pc_gamg_level_scoped_options {
            let scoped_policy = gamg_policy_from_scoped(opts, *level, scoped)?;
            let entry = merged
                .entry((*level, None))
                .or_insert_with(|| GamgLevelPolicy {
                level: *level,
                ..Default::default()
            });
            merge_gamg_policy(entry, &scoped_policy);
        }
        let level_policies = merged.into_values().collect::<Vec<_>>();

        let dist_coarse_route_fallback = route_fallback_order(
            amg_config.dist_coarse_solver_route,
            amg_config.dist_coarse_strategy,
            opts.amg_dist_coarse_solver_route.as_deref(),
        )?;

        Ok(GamgConfig {
            gamg_type,
            amg_config,
            level_policies,
            dist_coarse_route_fallback,
        })
    }
}

#[cfg(feature = "backend-faer")]
fn gamg_policy_applies(policy: &GamgLevelPolicy, level: usize, max_levels: usize) -> bool {
    let is_fine = level == 0;
    let is_coarse = level + 1 == max_levels;
    match policy.level_key.as_deref() {
        None => policy.level == level,
        Some("all") | Some("any") => true,
        Some("fine") => is_fine,
        Some("coarse") => is_coarse,
        Some("intermediate") | Some("mid") => !is_fine && !is_coarse,
        _ => policy.level == level,
    }
}

#[cfg(feature = "backend-faer")]
fn resolved_gamg_policy_for_level(
    policies: &[GamgLevelPolicy],
    level: usize,
    max_levels: usize,
) -> Option<GamgLevelPolicy> {
    let mut out = GamgLevelPolicy {
        level,
        ..Default::default()
    };
    let mut applied = false;
    for p in policies.iter().filter(|p| p.level_key.is_some()) {
        if gamg_policy_applies(p, level, max_levels) {
            merge_gamg_policy(&mut out, p);
            applied = true;
        }
    }
    for p in policies
        .iter()
        .filter(|p| p.level_key.is_none() && p.level == level)
    {
        merge_gamg_policy(&mut out, p);
        applied = true;
    }
    if applied { Some(out) } else { None }
}
#[cfg(feature = "backend-faer")]
fn parse_coarse_solver(value: &str) -> Result<CoarseSolve, KError> {
    Ok(match value.trim().to_lowercase().as_str() {
        "cg" => CoarseSolve::CG,
        "direct" | "dense" => CoarseSolve::DirectDense,
        "ilu" => CoarseSolve::ILU,
        "smoother" => CoarseSolve::Smoother,
        other => {
            return Err(KError::InvalidInput(format!(
                "invalid gamg coarse solver: {other}"
            )));
        }
    })
}

#[cfg(feature = "backend-faer")]
fn coarse_solve_from_pc_name(pc: &str) -> Option<CoarseSolve> {
    match pc.trim().to_lowercase().as_str() {
        "ilu" | "ilu0" | "ilut" | "iluk" => Some(CoarseSolve::ILU),
        "jacobi" | "sor" | "gs" | "gauss_seidel" => Some(CoarseSolve::Smoother),
        "lu" | "qr" | "direct" => Some(CoarseSolve::DirectDense),
        _ => None,
    }
}

#[cfg(feature = "backend-faer")]
fn gamg_policy_from_scoped(
    global: &PcOptions,
    level: usize,
    scoped: &PcOptions,
) -> Result<GamgLevelPolicy, KError> {
    let coarse_solver = scoped
        .amg_coarse_solver
        .as_deref()
        .or(global.amg_coarse_solver.as_deref())
        .map(parse_coarse_solver)
        .transpose()?
        .or_else(|| {
            scoped
                .pc_type
                .as_deref()
                .or(global.pc_type.as_deref())
                .and_then(coarse_solve_from_pc_name)
        });
    Ok(GamgLevelPolicy {
        level,
        level_key: None,
        smoother: scoped
            .amg_smoother
            .clone()
            .or(scoped.pc_type.clone())
            .or(global.amg_smoother.clone())
            .or(global.pc_type.clone())
            .map(|v| v.to_lowercase()),
        smoother_family: scoped
            .amg_smoother
            .clone()
            .or(scoped.pc_type.clone())
            .or(global.amg_smoother.clone())
            .or(global.pc_type.clone())
            .map(|v| v.to_lowercase()),
        sweeps: scoped
            .amg_smoother_steps
            .or(scoped.pc_mg_smoother_steps)
            .or(global.amg_smoother_steps)
            .or(global.pc_mg_smoother_steps),
        pre_sweeps: scoped.amg_sweeps_down.or(global.amg_sweeps_down),
        post_sweeps: scoped.amg_sweeps_up.or(global.amg_sweeps_up),
        coarse_solver,
        side: None,
        ksp_type: scoped
            .pc_ksp_ksp_type
            .clone()
            .or(global.pc_ksp_ksp_type.clone())
            .map(|v| v.to_lowercase()),
        pc_type: scoped
            .pc_ksp_pc_type
            .clone()
            .or(global.pc_ksp_pc_type.clone())
            .or_else(|| scoped.pc_type.clone())
            .or(global.pc_type.clone())
            .map(|v| v.to_lowercase()),
        ksp_maxits: scoped.pc_ksp_maxits.or(global.pc_ksp_maxits),
        ksp_rtol: scoped.pc_ksp_rtol.or(global.pc_ksp_rtol),
        coarse_routes: scoped
            .amg_dist_coarse_solver_route
            .as_deref()
            .or(global.amg_dist_coarse_solver_route.as_deref())
            .map(parse_route_list)
            .transpose()?
            .filter(|v| !v.is_empty()),
    })
}
#[cfg(feature = "backend-faer")]
fn merge_gamg_policy(dst: &mut GamgLevelPolicy, src: &GamgLevelPolicy) {
    if let Some(v) = src.level_key.as_ref() {
        dst.level_key = Some(v.clone());
    }
    if let Some(v) = src.smoother.as_ref() {
        dst.smoother = Some(v.clone());
    }
    if let Some(v) = src.smoother_family.as_ref() {
        dst.smoother_family = Some(v.clone());
    }
    if let Some(v) = src.sweeps {
        dst.sweeps = Some(v);
    }
    if let Some(v) = src.pre_sweeps {
        dst.pre_sweeps = Some(v);
    }
    if let Some(v) = src.post_sweeps {
        dst.post_sweeps = Some(v);
    }
    if let Some(v) = src.coarse_solver {
        dst.coarse_solver = Some(v);
    }
    if let Some(v) = src.side {
        dst.side = Some(v);
    }
    if let Some(v) = src.ksp_type.as_ref() {
        dst.ksp_type = Some(v.clone());
    }
    if let Some(v) = src.pc_type.as_ref() {
        dst.pc_type = Some(v.clone());
    }
    if let Some(v) = src.ksp_maxits {
        dst.ksp_maxits = Some(v);
    }
    if let Some(v) = src.ksp_rtol {
        dst.ksp_rtol = Some(v);
    }
    if let Some(v) = src.coarse_routes.as_ref() {
        dst.coarse_routes = Some(v.clone());
    }
}
#[cfg(feature = "backend-faer")]
fn parse_gamg_level_policy(value: &str) -> Result<GamgLevelPolicy, KError> {
    let mut policy = GamgLevelPolicy::default();
    for token in value.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if let Some((k, v)) = token.split_once('=') {
            match k.trim() {
                "level" => {
                    policy.level = v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid gamg policy level: {v}"))
                    })?
                }
                "level_key" | "family_key" => policy.level_key = Some(v.trim().to_lowercase()),
                "smoother" => policy.smoother = Some(v.trim().to_lowercase()),
                "smoother_family" | "family" => {
                    policy.smoother_family = Some(v.trim().to_lowercase())
                }
                "sweeps" => {
                    policy.sweeps =
                        Some(v.trim().parse().map_err(|_| {
                            KError::InvalidInput(format!("invalid gamg sweeps: {v}"))
                        })?)
                }
                "pre_sweeps" => {
                    policy.pre_sweeps = Some(v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid gamg pre sweeps: {v}"))
                    })?)
                }
                "post_sweeps" => {
                    policy.post_sweeps = Some(v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid gamg post sweeps: {v}"))
                    })?)
                }
                "coarse" | "coarse_solver" => policy.coarse_solver = Some(parse_coarse_solver(v)?),
                "ksp" | "ksp_type" => policy.ksp_type = Some(v.trim().to_lowercase()),
                "pc" | "pc_type" => policy.pc_type = Some(v.trim().to_lowercase()),
                "ksp_maxits" | "maxits" => {
                    policy.ksp_maxits = Some(v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid gamg ksp maxits: {v}"))
                    })?)
                }
                "ksp_rtol" | "rtol" => {
                    policy.ksp_rtol =
                        Some(v.trim().parse().map_err(|_| {
                            KError::InvalidInput(format!("invalid gamg ksp rtol: {v}"))
                        })?)
                }
                "side" => policy.side = Some(PcSide::from_str(v.trim())?),
                "coarse_route" | "coarse_routes" => {
                    let routes = parse_route_list(v)?;
                    if !routes.is_empty() {
                        policy.coarse_routes = Some(routes);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(policy)
}

#[cfg(not(feature = "backend-faer"))]
impl GamgConfig {
    pub fn try_from_opts(_opts: &PcOptions) -> Result<Self, KError> {
        Err(KError::Unsupported(
            "GAMG requires backend-faer; enable backend-faer to use GAMG options",
        ))
    }
}

#[cfg(feature = "backend-faer")]
fn parse_route_head(value: &str) -> Result<DistCoarseSolverRoute, KError> {
    let head = value.split(',').next().map(str::trim).unwrap_or(value);
    DistCoarseSolverRoute::from_str(head)
}

fn parse_route_list(value: &str) -> Result<Vec<DistCoarseSolverRoute>, KError> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(DistCoarseSolverRoute::from_str)
        .collect()
}

#[cfg(feature = "backend-faer")]
fn route_fallback_order(
    selected: DistCoarseSolverRoute,
    strategy: DistCoarseStrategy,
    raw: Option<&str>,
) -> Result<Vec<DistCoarseSolverRoute>, KError> {
    let mut ordered = Vec::new();
    let mut push_unique = |route: DistCoarseSolverRoute| {
        if !ordered.contains(&route) {
            ordered.push(route);
        }
    };

    if let Some(raw) = raw {
        for item in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            push_unique(DistCoarseSolverRoute::from_str(item)?);
        }
    }

    push_unique(selected);
    match strategy {
        DistCoarseStrategy::RootGather => {
            push_unique(DistCoarseSolverRoute::Root);
            push_unique(DistCoarseSolverRoute::Local);
            push_unique(DistCoarseSolverRoute::SuperLuDist);
        }
        DistCoarseStrategy::LocalPrototype => {
            push_unique(DistCoarseSolverRoute::Local);
            push_unique(DistCoarseSolverRoute::Root);
            push_unique(DistCoarseSolverRoute::SuperLuDist);
        }
        DistCoarseStrategy::SuperLuDist => {
            push_unique(DistCoarseSolverRoute::SuperLuDist);
            push_unique(DistCoarseSolverRoute::Root);
            push_unique(DistCoarseSolverRoute::Local);
        }
        DistCoarseStrategy::None => {
            push_unique(DistCoarseSolverRoute::Local);
            push_unique(DistCoarseSolverRoute::Root);
        }
    }
    Ok(ordered)
}

#[cfg(feature = "backend-faer")]
fn parse_gamg_dist_mode(value: &str) -> Result<DistCoarseStrategy, KError> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok(DistCoarseStrategy::RootGather);
    }
    DistCoarseStrategy::from_str(value)
}

#[cfg(feature = "backend-faer")]
fn map_gamg_coarsen_type(value: &str) -> Result<CoarsenType, KError> {
    let kind = AmgCoarsenKind::from_str(value)?;
    Ok(match kind {
        AmgCoarsenKind::Rs => CoarsenType::RS,
        AmgCoarsenKind::Hmis => CoarsenType::HMIS,
        AmgCoarsenKind::Pmis => CoarsenType::PMIS,
        AmgCoarsenKind::Falgout => CoarsenType::Falgout,
    })
}

#[cfg(feature = "backend-faer")]
fn map_gamg_interp_type(value: &str) -> Result<InterpType, KError> {
    let kind = AmgInterpKind::from_str(value)?;
    Ok(match kind {
        AmgInterpKind::Classical => InterpType::Classical,
        AmgInterpKind::Direct => InterpType::Direct,
        AmgInterpKind::Multipass => InterpType::Multipass,
        AmgInterpKind::Extended => InterpType::Extended,
        AmgInterpKind::Standard => InterpType::Standard,
        AmgInterpKind::He => InterpType::HE,
    })
}

#[cfg(feature = "backend-faer")]
fn apply_petsc_gamg_defaults(cfg: &mut AMGConfig, gamg_type: GamgType) {
    cfg.coarsen_type = match gamg_type {
        GamgType::Agg => CoarsenType::HMIS,
        GamgType::Classical => CoarsenType::RS,
    };
    cfg.interp_type = match gamg_type {
        GamgType::Agg => InterpType::Extended,
        GamgType::Classical => InterpType::Classical,
    };
    cfg.relax_type = RelaxType::Jacobi;
    for phase in RelaxPhase::ALL {
        cfg.grid_relax_type[phase.ix()] = RelaxType::Jacobi;
        cfg.num_grid_sweeps[phase.ix()] = 1;
    }
    cfg.grid_relax_type[RelaxPhase::Coarsest.ix()] = RelaxType::GaussSeidel;
    cfg.num_grid_sweeps[RelaxPhase::Coarsest.ix()] = 0;
    cfg.coarse_solve = CoarseSolve::DirectDense;
    cfg.pre_sweeps = 1;
    cfg.post_sweeps = 1;
}

#[cfg(feature = "backend-faer")]
pub struct Gamg {
    amg: AMG,
    config: GamgConfig,
}

#[cfg(feature = "backend-faer")]
impl Gamg {
    pub fn with_config(config: GamgConfig) -> Self {
        let amg = AMG::with_config(config.amg_config.clone());
        Self { amg, config }
    }

    pub fn config(&self) -> &GamgConfig {
        &self.config
    }

    pub fn set_level_transfer_operators(&mut self, level: usize, operators: AmgTransferOperators) {
        self.amg.set_level_transfer_operators(level, operators);
    }

    pub fn set_level_coarse_solver(&mut self, level: usize, solve: CoarseSolve) {
        self.amg.set_level_coarse_solver(level, solve);
    }
}

#[cfg(feature = "backend-faer")]
impl Preconditioner for Gamg {
    fn dims(&self) -> (usize, usize) {
        self.amg.dims()
    }

    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
        let mut effective_cfg = self.config.amg_config.clone();
        if let Some(coarse_policy) = resolved_gamg_policy_for_level(
            &self.config.level_policies,
            effective_cfg.max_levels.saturating_sub(1),
            effective_cfg.max_levels,
        ) && let Some(route) = coarse_policy
            .coarse_routes
            .as_ref()
            .and_then(|routes| routes.first().copied())
        {
            effective_cfg.dist_coarse_solver_route = route;
        }
        if let Some(fine_policy) =
            resolved_gamg_policy_for_level(&self.config.level_policies, 0, effective_cfg.max_levels)
        {
            if let Some(smoother) = fine_policy
                .smoother_family
                .as_ref()
                .or(fine_policy.smoother.as_ref())
            {
                effective_cfg.relax_type = match smoother.as_str() {
                    "jacobi" => RelaxType::Jacobi,
                    "gs" | "gauss_seidel" => RelaxType::GaussSeidel,
                    "sor" => RelaxType::SymmetricGaussSeidel,
                    "l1jacobi" | "l1_jacobi" => RelaxType::L1Jacobi,
                    "chebyshev" | "cheby" => RelaxType::Chebyshev,
                    _ => effective_cfg.relax_type,
                };
                for phase in RelaxPhase::ALL {
                    effective_cfg.grid_relax_type[phase.ix()] = effective_cfg.relax_type;
                }
            }
            if let Some(sweeps) = fine_policy.sweeps {
                effective_cfg.pre_sweeps = sweeps;
                effective_cfg.post_sweeps = sweeps;
                effective_cfg.num_grid_sweeps[RelaxPhase::Fine.ix()] = sweeps;
                effective_cfg.num_grid_sweeps[RelaxPhase::Down.ix()] = sweeps;
                effective_cfg.num_grid_sweeps[RelaxPhase::Up.ix()] = sweeps;
            }
            if let Some(pre) = fine_policy.pre_sweeps {
                effective_cfg.pre_sweeps = pre;
                effective_cfg.num_grid_sweeps[RelaxPhase::Fine.ix()] = pre;
                effective_cfg.num_grid_sweeps[RelaxPhase::Down.ix()] = pre;
            }
            if let Some(post) = fine_policy.post_sweeps {
                effective_cfg.post_sweeps = post;
                effective_cfg.num_grid_sweeps[RelaxPhase::Up.ix()] = post;
            }
        }
        self.amg = AMG::with_config(effective_cfg.clone());
        for level in 0..=effective_cfg.max_levels {
            if let Some(p) = resolved_gamg_policy_for_level(
                &self.config.level_policies,
                level,
                effective_cfg.max_levels,
            ) {
                if let Some(smoother) = p.smoother_family.as_ref().or(p.smoother.as_ref()) {
                    let relax = match smoother.as_str() {
                        "jacobi" => Some(RelaxType::Jacobi),
                        "gs" | "gauss_seidel" => Some(RelaxType::GaussSeidel),
                        "sor" => Some(RelaxType::SymmetricGaussSeidel),
                        "l1jacobi" | "l1_jacobi" => Some(RelaxType::L1Jacobi),
                        "chebyshev" | "cheby" => Some(RelaxType::Chebyshev),
                        _ => None,
                    };
                    if let Some(relax) = relax {
                        self.amg.set_level_relax_type(level, relax);
                    }
                }
                if p.sweeps.is_some() || p.pre_sweeps.is_some() || p.post_sweeps.is_some() {
                    let sweeps = p.sweeps.unwrap_or(1);
                    let pre = p.pre_sweeps.unwrap_or(sweeps);
                    let post = p.post_sweeps.unwrap_or(sweeps);
                    self.amg.set_level_sweeps(level, pre, post);
                }
                if let Some(solve) = p
                    .coarse_solver
                    .or_else(|| p.pc_type.as_deref().and_then(coarse_solve_from_pc_name))
                    .or_else(|| p.ksp_type.as_ref().map(|_| CoarseSolve::CG))
                {
                    self.amg.set_level_coarse_solver(level, solve);
                }
            }
        }
        self.amg.setup(a)
    }

    fn apply(&self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.amg.apply(side, x, y)
    }

    fn apply_op(&self, op: Op, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.amg.apply_op(op, x, y)
    }

    fn apply_op_inplace(&self, op: Op, y: &mut [S]) -> Result<(), KError> {
        self.amg.apply_op_inplace(op, y)
    }

    fn capabilities(&self) -> PcCaps {
        self.amg.capabilities()
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        self.amg.distributed_support()
    }

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.amg.apply_mut(side, x, y)
    }

    fn on_restart(&mut self, outer_iter: usize, residual_norm: R) -> Result<(), KError> {
        self.amg.on_restart(outer_iter, residual_norm)
    }

    fn supports_numeric_update(&self) -> bool {
        self.amg.supports_numeric_update()
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.amg.update_numeric(op)
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        self.amg.update_symbolic(op)
    }

    fn required_format(&self) -> OpFormat {
        self.amg.required_format()
    }

    fn preferred_drop_tol_for_format(&self) -> Option<R> {
        self.amg.preferred_drop_tol_for_format()
    }
}

#[cfg(all(test, feature = "backend-faer"))]
mod tests {
    use super::*;

    #[test]
    fn gamg_config_parses_advanced_options() {
        let opts = PcOptions {
            pc_gamg_type: Some("agg".into()),
            pc_gamg_levels: Some(5),
            pc_gamg_threshold: Some(0.17),
            pc_gamg_coarsen_type: Some("pmis".into()),
            pc_gamg_interp_type: Some("standard".into()),
            pc_gamg_aggressive_levels: Some(3),
            pc_gamg_aggressive_mis_k: Some(4),
            ..Default::default()
        };

        let cfg = GamgConfig::try_from_opts(&opts).expect("gamg config parse");
        assert_eq!(cfg.gamg_type, GamgType::Agg);
        assert_eq!(cfg.amg_config.max_levels, 5);
        assert_eq!(cfg.amg_config.strong_threshold, 0.17);
        assert_eq!(cfg.amg_config.coarsen_type, CoarsenType::PMIS);
        assert_eq!(cfg.amg_config.interp_type, InterpType::Standard);
        assert_eq!(cfg.amg_config.agg_num_levels, 3);
        assert_eq!(cfg.amg_config.aggressive_mis_k, 4);
        assert!(!cfg.amg_config.dist_apply_instrumentation);
    }

    #[test]
    fn gamg_config_parses_scoped_level_options() {
        let args = vec![
            "-pc_gamg_levels_2_pc_type",
            "ilu",
            "-pc_gamg_levels_2_pc_ksp_ksp_type",
            "cg",
        ];
        let opts = PcOptions::from_args(&args).expect("parse scoped options");
        let cfg = GamgConfig::try_from_opts(&opts).expect("gamg config parse");
        let lvl = cfg
            .level_policies
            .iter()
            .find(|p| p.level == 2)
            .expect("level policy");
        assert_eq!(lvl.pc_type.as_deref(), Some("ilu"));
        assert_eq!(lvl.ksp_type.as_deref(), Some("cg"));
        assert_eq!(lvl.coarse_solver, Some(CoarseSolve::ILU));
    }

    #[test]
    fn gamg_level_policy_parses_smoother_family() {
        let parsed = parse_gamg_level_policy("level=2,smoother_family=chebyshev,sweeps=3")
            .expect("parse policy");
        assert_eq!(parsed.level, 2);
        assert_eq!(parsed.smoother_family.as_deref(), Some("chebyshev"));
        assert_eq!(parsed.sweeps, Some(3));
        assert_eq!(parsed.pre_sweeps, None);
        assert_eq!(parsed.post_sweeps, None);
    }

    #[test]
    fn gamg_scoped_policy_sets_family() {
        let args = vec!["-pc_gamg_levels_1_pc_type", "jacobi"];
        let opts = PcOptions::from_args(&args).expect("parse scoped options");
        let cfg = GamgConfig::try_from_opts(&opts).expect("gamg config parse");
        let lvl = cfg
            .level_policies
            .iter()
            .find(|p| p.level == 1)
            .expect("level policy");
        assert_eq!(lvl.smoother_family.as_deref(), Some("jacobi"));
    }

    #[test]
    fn gamg_level_policy_parses_independent_pre_post_sweeps() {
        let parsed = parse_gamg_level_policy("level=1,smoother=jacobi,pre_sweeps=3,post_sweeps=1")
            .expect("parse policy");
        assert_eq!(parsed.level, 1);
        assert_eq!(parsed.pre_sweeps, Some(3));
        assert_eq!(parsed.post_sweeps, Some(1));
    }

    #[test]
    fn gamg_scoped_policy_inherits_global_ksp_pairing() {
        let args = vec![
            "-pc_ksp_type",
            "gmres",
            "-pc_ksp_pc_type",
            "ilu",
            "-pc_ksp_maxits",
            "8",
            "-pc_gamg_levels_1_pc_type",
            "jacobi",
        ];
        let opts = PcOptions::from_args(&args).expect("parse scoped options");
        let cfg = GamgConfig::try_from_opts(&opts).expect("gamg config parse");
        let lvl = cfg
            .level_policies
            .iter()
            .find(|p| p.level == 1)
            .expect("level policy");
        assert_eq!(lvl.ksp_type.as_deref(), Some("gmres"));
        assert_eq!(lvl.pc_type.as_deref(), Some("ilu"));
        assert_eq!(lvl.ksp_maxits, Some(8));
    }

    #[test]
    fn gamg_dist_route_fallback_order_is_explicit_and_deterministic() {
        let opts = PcOptions {
            amg_dist_apply_mode: Some("root".into()),
            amg_dist_coarse_solver_route: Some("superlu_dist,root,local".into()),
            ..Default::default()
        };
        let cfg = GamgConfig::try_from_opts(&opts).expect("parse distributed options");
        assert_eq!(
            cfg.amg_config.dist_coarse_solver_route,
            DistCoarseSolverRoute::SuperLuDist
        );
        assert_eq!(
            cfg.dist_coarse_route_fallback,
            vec![
                DistCoarseSolverRoute::SuperLuDist,
                DistCoarseSolverRoute::Root,
                DistCoarseSolverRoute::Local,
            ]
        );
    }

    #[test]
    fn gamg_cli_string_round_trip_for_scoped_and_routes() {
        let args = vec![
            "-pc_gamg_levels_2_pc_type",
            "ilu",
            "-pc_gamg_levels_2_pc_ksp_type",
            "cg",
            "-pc_amg_dist_coarse_solver_route",
            "local,root",
        ];
        let opts = PcOptions::from_args(&args).expect("parse options");
        let cfg = GamgConfig::try_from_opts(&opts).expect("build config");
        let lvl = cfg
            .level_policies
            .iter()
            .find(|p| p.level == 2)
            .expect("level policy");
        assert_eq!(lvl.pc_type.as_deref(), Some("ilu"));
        assert_eq!(lvl.ksp_type.as_deref(), Some("cg"));
        assert_eq!(
            cfg.dist_coarse_route_fallback,
            vec![
                DistCoarseSolverRoute::Local,
                DistCoarseSolverRoute::Root,
                DistCoarseSolverRoute::SuperLuDist
            ]
        );
    }
    #[test]
    fn gamg_config_parses_dist_coarse_controls() {
        let opts = PcOptions {
            amg_dist_apply_mode: Some("local_prototype".into()),
            amg_dist_coarse_repartition: Some("uniform".into()),
            amg_dist_coarse_solver_route: Some("local".into()),
            amg_dist_instrumentation: Some(true),
            ..Default::default()
        };
        let cfg = GamgConfig::try_from_opts(&opts).expect("parse distributed options");
        assert_eq!(
            cfg.amg_config.dist_coarse_strategy,
            DistCoarseStrategy::LocalPrototype
        );
        assert!(cfg.amg_config.dist_apply_instrumentation);
        assert_eq!(
            cfg.amg_config.dist_coarse_repartition,
            DistCoarseRepartition::Uniform
        );
        assert_eq!(
            cfg.amg_config.dist_coarse_solver_route,
            DistCoarseSolverRoute::Local
        );
    }

    #[test]
    fn gamg_config_rejects_invalid_dist_route() {
        let opts = PcOptions {
            amg_dist_coarse_solver_route: Some("bogus".into()),
            ..Default::default()
        };
        let err = GamgConfig::try_from_opts(&opts).expect_err("expected invalid route failure");
        assert!(err.to_string().contains("invalid dist coarse solver route"));
    }

    #[test]
    fn gamg_config_accepts_hybrid_dist_aliases() {
        let opts = PcOptions {
            amg_dist_apply_mode: Some("hybrid".into()),
            amg_dist_coarse_repartition: Some("hybrid".into()),
            amg_dist_coarse_solver_route: Some("hybrid".into()),
            ..Default::default()
        };
        let cfg = GamgConfig::try_from_opts(&opts).expect("parse hybrid aliases");
        assert_eq!(
            cfg.amg_config.dist_coarse_strategy,
            DistCoarseStrategy::LocalPrototype
        );
        assert_eq!(
            cfg.amg_config.dist_coarse_repartition,
            DistCoarseRepartition::Uniform
        );
        assert_eq!(
            cfg.amg_config.dist_coarse_solver_route,
            DistCoarseSolverRoute::Local
        );
    }

    #[test]
    fn gamg_level_policy_precedence_scoped_over_family_level() {
        let opts = PcOptions {
            pc_gamg_level_policies: Some(vec!["level=2,smoother=jacobi,ksp=gmres,maxits=3".into()]),
            pc_gamg_level_scoped_options: vec![(
                2,
                Box::new(PcOptions {
                    pc_type: Some("ilu".into()),
                    pc_ksp_ksp_type: Some("cg".into()),
                    pc_ksp_maxits: Some(9),
                    ..Default::default()
                }),
            )],
            ..Default::default()
        };
        let cfg = GamgConfig::try_from_opts(&opts).expect("build config");
        let lvl = cfg
            .level_policies
            .iter()
            .find(|p| p.level == 2)
            .expect("level policy");
        assert_eq!(lvl.pc_type.as_deref(), Some("ilu"));
        assert_eq!(lvl.ksp_type.as_deref(), Some("cg"));
        assert_eq!(lvl.ksp_maxits, Some(9));
    }

    #[test]
    fn gamg_dist_policy_overrides_apply_mode_for_route_fallback() {
        let opts = PcOptions {
            amg_dist_apply_mode: Some("root".into()),
            amg_dist_coarse_policy: Some("local".into()),
            amg_dist_coarse_solver_route: Some("auto".into()),
            ..Default::default()
        };
        let cfg = GamgConfig::try_from_opts(&opts).expect("parse distributed options");
        assert_eq!(
            cfg.amg_config.dist_coarse_strategy,
            DistCoarseStrategy::LocalPrototype
        );
        assert_eq!(
            cfg.dist_coarse_route_fallback,
            vec![
                DistCoarseSolverRoute::Auto,
                DistCoarseSolverRoute::Local,
                DistCoarseSolverRoute::Root,
                DistCoarseSolverRoute::SuperLuDist,
            ]
        );
    }

    #[test]
    fn gamg_level_policy_precedence_global_family_exact() {
        let opts = PcOptions {
            pc_gamg_level_policies: Some(vec![
                "level=0,level_key=all,smoother=jacobi,sweeps=2,pc=ilu".into(),
                "level=0,level_key=coarse,ksp=cg".into(),
                "level=2,smoother=chebyshev,pre_sweeps=5".into(),
            ]),
            ..Default::default()
        };
        let cfg = GamgConfig::try_from_opts(&opts).expect("build config");
        let l2 = resolved_gamg_policy_for_level(&cfg.level_policies, 2, 4).expect("l2");
        assert_eq!(l2.smoother.as_deref(), Some("chebyshev"));
        assert_eq!(l2.pre_sweeps, Some(5));
        assert_eq!(l2.sweeps, Some(2));
        let lc = resolved_gamg_policy_for_level(&cfg.level_policies, 3, 4).expect("coarse");
        assert_eq!(lc.ksp_type.as_deref(), Some("cg"));
        assert_eq!(lc.pc_type.as_deref(), Some("ilu"));
    }

    #[test]
    fn gamg_config_rejects_invalid_aggressive_controls() {
        let opts = PcOptions {
            pc_gamg_aggressive_levels: Some(0),
            ..Default::default()
        };
        let err = GamgConfig::try_from_opts(&opts).expect_err("expected aggressive levels to fail");
        assert!(
            err.to_string()
                .contains("pc_gamg_aggressive_levels must be >= 1")
        );

        let opts = PcOptions {
            pc_gamg_aggressive_mis_k: Some(1),
            ..Default::default()
        };
        let err = GamgConfig::try_from_opts(&opts).expect_err("expected mis k to fail");
        assert!(
            err.to_string()
                .contains("pc_gamg_aggressive_mis_k must be >= 2")
        );
    }
}
