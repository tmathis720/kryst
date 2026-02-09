use crate::algebra::prelude::*;
use crate::config::options::PcOptions;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::{
    Op, OpFormat, PcCaps, PcDistributedSupport, PcSide, Preconditioner,
};

#[cfg(feature = "backend-faer")]
use crate::preconditioner::amg::{
    AMGConfig, CoarseSolve, CoarsenType, InterpType, RelaxPhase, RelaxType, AMG,
};

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
                return Err(KError::InvalidInput(
                    "pc_gamg_levels must be >= 1".into(),
                ));
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

        Ok(GamgConfig {
            gamg_type,
            amg_config,
        })
    }
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
}

#[cfg(feature = "backend-faer")]
impl Preconditioner for Gamg {
    fn dims(&self) -> (usize, usize) {
        self.amg.dims()
    }

    fn setup(&mut self, a: &dyn LinOp<S = S>) -> Result<(), KError> {
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
