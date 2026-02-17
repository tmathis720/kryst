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
    pub smoother: Option<String>,
    pub sweeps: Option<usize>,
    pub coarse_solver: Option<CoarseSolve>,
    pub side: Option<PcSide>,
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
            amg_config.dist_coarse_solver_route = DistCoarseSolverRoute::from_str(route)?;
        }
        if let Some(policy) = opts.amg_dist_coarse_policy.as_deref() {
            amg_config.dist_coarse_strategy = DistCoarseStrategy::from_str(policy)?;
        }

        let level_policies = opts
            .pc_gamg_level_policies
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|entry| parse_gamg_level_policy(entry).ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(GamgConfig {
            gamg_type,
            amg_config,
            level_policies,
        })
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
                "smoother" => policy.smoother = Some(v.trim().to_lowercase()),
                "sweeps" => {
                    policy.sweeps =
                        Some(v.trim().parse().map_err(|_| {
                            KError::InvalidInput(format!("invalid gamg sweeps: {v}"))
                        })?)
                }
                "coarse" | "coarse_solver" => {
                    policy.coarse_solver = Some(match v.trim().to_lowercase().as_str() {
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
                "side" => policy.side = Some(PcSide::from_str(v.trim())?),
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
fn parse_gamg_dist_mode(value: &str) -> Result<DistCoarseStrategy, KError> {
    match value {
        "none" => Ok(DistCoarseStrategy::None),
        "root" | "root_gather" | "gather" => Ok(DistCoarseStrategy::RootGather),
        "local" | "local_prototype" => Ok(DistCoarseStrategy::LocalPrototype),
        "superlu_dist" => Ok(DistCoarseStrategy::SuperLuDist),
        "auto" => Ok(DistCoarseStrategy::RootGather),
        other => Err(KError::InvalidInput(format!(
            "unsupported GAMG coarse apply mode: {other}"
        ))),
    }
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
        for p in &self.config.level_policies {
            if let Some(solve) = p.coarse_solver {
                self.amg.set_level_coarse_solver(p.level, solve);
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
