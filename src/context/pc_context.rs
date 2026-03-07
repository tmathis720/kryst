use crate::algebra::prelude::*;
use crate::config::kinds::SorMatSideKind;
use crate::config::options::{KspOptions, PcOptions};
use crate::error::KError;
use crate::matrix::op::LinOp;
#[cfg(feature = "backend-faer")]
use crate::preconditioner::asm::AsmInnerPc;
use crate::preconditioner::bddc::{BddcConstraintSelection, BddcScaling};
use crate::preconditioner::mg::MgLevelPolicy;
use crate::preconditioner::{PcSide, Preconditioner};
use crate::utils::conditioning::ConditioningOptions;
use std::str::FromStr;

#[cfg(feature = "backend-faer")]
use crate::preconditioner::amg::AMGConfig;
#[cfg(feature = "backend-faer")]
use crate::preconditioner::gamg::GamgConfig;

#[cfg(not(feature = "backend-faer"))]
#[derive(Clone, Debug)]
pub struct AMGConfig;

#[cfg(not(feature = "backend-faer"))]
impl AMGConfig {
    pub fn try_from_opts(_opts: &PcOptions) -> Result<Self, KError> {
        Err(KError::Unsupported(
            "AMG requires backend-faer; enable backend-faer to use AMG options",
        ))
    }
}

#[cfg(not(feature = "backend-faer"))]
#[derive(Clone, Debug)]
pub struct GamgConfig;

#[cfg(not(feature = "backend-faer"))]
impl GamgConfig {
    pub fn try_from_opts(_opts: &PcOptions) -> Result<Self, KError> {
        Err(KError::Unsupported(
            "GAMG requires backend-faer; enable backend-faer to use GAMG options",
        ))
    }
}

#[cfg(feature = "backend-faer")]
type MatSorSide = crate::preconditioner::sor::MatSorType;

#[cfg(not(feature = "backend-faer"))]
bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MatSorSide: u32 {
        const APPLY_LOWER = 0b0001;
        const APPLY_UPPER = 0b0010;
        const SYMMETRIC_SWEEP = 0b0100;
        const EISENSTAT = 0b1000;
    }
}

#[cfg(not(feature = "backend-faer"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AsmInnerPc {
    Jacobi,
    Ilu0,
    Ilut {
        drop_tol: R,
        max_fill: usize,
    },
    Ilutp {
        drop_tol: R,
        max_fill: usize,
        perm_tol: R,
    },
}

#[cfg(feature = "backend-faer")]
type ApproxInvKindAlias = crate::preconditioner::approxinv_csr::ApproxInvKind;
#[cfg(not(feature = "backend-faer"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApproxInvKindAlias {
    FSAI,
    SPAI,
}

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static CHAIN_STRICT_OVERRIDE: Cell<Option<bool>> = Cell::new(None);
}

#[cfg(test)]
pub(crate) struct ChainStrictGuard(Option<bool>);

#[cfg(test)]
impl Drop for ChainStrictGuard {
    fn drop(&mut self) {
        CHAIN_STRICT_OVERRIDE.with(|cell| cell.set(self.0));
    }
}

/// Supported preconditioner types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcType {
    Jacobi,
    Ilu0,
    None,
    Ilu,
    Ilut,
    Ilutp,
    Ilup,
    BlockJacobi,
    Sor,
    Asm,
    Chebyshev,
    Amg,
    ApproxInverse,
    FieldSplit,
    Shell,
    Ksp,
    Mg,
    Bddc,
    Gamg,
    Lu,
    Qr,
    #[cfg_attr(docsrs, doc(cfg(feature = "superlu_dist")))]
    #[cfg(feature = "superlu_dist")]
    SuperLuDist,
}

impl FromStr for PcType {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "jacobi" => Ok(PcType::Jacobi),
            "ilu0" => Ok(PcType::Ilu0),
            "none" => Ok(PcType::None),
            "ilu" => Ok(PcType::Ilu),
            "ilut" => Ok(PcType::Ilut),
            "ilutp" => Ok(PcType::Ilutp),
            "ilup" => Ok(PcType::Ilup),
            "block_jacobi" => Ok(PcType::BlockJacobi),
            "sor" => Ok(PcType::Sor),
            "asm" | "ras" => Ok(PcType::Asm),
            "chebyshev" => Ok(PcType::Chebyshev),
            "amg" => Ok(PcType::Amg),
            "approxinv" | "approxinverse" => Ok(PcType::ApproxInverse),
            "fieldsplit" => Ok(PcType::FieldSplit),
            "shell" => Ok(PcType::Shell),
            "ksp" => Ok(PcType::Ksp),
            "mg" => Ok(PcType::Mg),
            "bddc" => Ok(PcType::Bddc),
            "gamg" => Ok(PcType::Gamg),
            "lu" => Ok(PcType::Lu),
            "qr" => Ok(PcType::Qr),
            "superludist" => {
                #[cfg(feature = "superlu_dist")]
                {
                    Ok(PcType::SuperLuDist)
                }
                #[cfg(not(feature = "superlu_dist"))]
                {
                    Err(KError::Unsupported(
                        "build without feature=\"superlu_dist\"".into(),
                    ))
                }
            }
            other => Err(KError::UnrecognizedPcType(other.to_string())),
        }
    }
}

/// Placeholder for deferred preconditioner construction info.
#[derive(Debug, Clone)]
pub struct DeferredPcInfo {
    pub pc_type: PcType,
    pub options: Option<PcOptions>,
}

/// Lightweight PC context for diagnostics and metadata reporting.
#[derive(Debug, Clone)]
pub struct PcContext {
    pub pc_type: PcType,
    pub options: Option<PcOptions>,
}

impl PcContext {
    pub fn new(pc_type: PcType, options: Option<PcOptions>) -> Self {
        Self { pc_type, options }
    }

    pub fn view(&self) -> crate::utils::diagnostics::PcDiagnostics {
        crate::utils::diagnostics::PcDiagnostics::from_options(
            Some(self.pc_type),
            self.options.as_ref(),
        )
    }
}

impl From<DeferredPcInfo> for PcContext {
    fn from(spec: DeferredPcInfo) -> Self {
        Self::new(spec.pc_type, spec.options)
    }
}

/// Simple no-op preconditioner.
pub struct NoOpPreconditioner;

impl Preconditioner for NoOpPreconditioner {
    fn setup(&mut self, _a: &dyn LinOp<S = S>) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, _side: PcSide, r: &[S], z: &mut [S]) -> Result<(), KError> {
        z.copy_from_slice(r);
        Ok(())
    }

    fn apply_mut(&mut self, side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        self.apply(side, x, y)
    }
}

/// Typed configuration parsed from options.
#[derive(Debug, Clone)]
pub enum PcConfig {
    None,
    Jacobi,
    BlockJacobi {
        block: usize,
    },
    Ilu0 {
        conditioning: ConditioningOptions,
    },
    Iluk {
        level: usize,
        conditioning: ConditioningOptions,
    },
    Ilut {
        drop_tol: R,
        max_fill: usize,
        reordering: Option<String>,
        conditioning: ConditioningOptions,
    },
    Ilutp {
        drop_tol: R,
        max_fill: usize,
        perm_tol: R,
        reordering: Option<String>,
        conditioning: ConditioningOptions,
    },
    Milu0 {
        conditioning: ConditioningOptions,
    },
    Sor {
        omega: R,
        sweeps: usize,
        mat_side: MatSorSide,
    },
    Chebyshev {
        degree: usize,
        eig_lo: R,
        eig_hi: R,
    },
    Asm {
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: Option<String>,
        mode: Option<String>,
        weighting: Option<String>,
        inner_pc: AsmInnerPc,
    },
    Amg {
        config: AMGConfig,
        conditioning: ConditioningOptions,
    },
    ApproxInv {
        kind: ApproxInvKindAlias,
        levels: usize,
        max_per_col: usize,
        drop_tol: R,
        reg: R,
        max_cond: R,
        parallel: bool,
    },
    FieldSplit {
        block_sizes: Vec<usize>,
        child_pc_type: Option<String>,
        options: PcOptions,
    },
    Shell {
        name: Option<String>,
        apply_transpose: Option<String>,
        apply_conjugate_transpose: Option<String>,
        apply_symmetric: Option<String>,
        apply_symmetric_left: Option<String>,
        apply_symmetric_right: Option<String>,
        setup: Option<String>,
        destroy: Option<String>,
        context: Option<String>,
    },
    Ksp {
        ksp_options: KspOptions,
        pc_options: PcOptions,
    },
    Mg {
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
        coarse_ksp_rtol: Option<R>,
        level_policies: Vec<MgLevelPolicy>,
    },
    Bddc {
        coarse_ksp_type: Option<String>,
        coarse_pc_type: Option<String>,
        use_vertices: bool,
        constraint_selection: BddcConstraintSelection,
        scaling: BddcScaling,
    },
    Gamg {
        config: GamgConfig,
        conditioning: ConditioningOptions,
    },
    Lu,
    Qr,
    #[cfg_attr(docsrs, doc(cfg(feature = "superlu_dist")))]
    #[cfg(feature = "superlu_dist")]
    SuperLuDist,
}

fn parse_mg_level_policy(value: &str) -> Result<MgLevelPolicy, KError> {
    let mut policy = MgLevelPolicy::default();
    for token in value.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if let Some((k, v)) = token.split_once('=') {
            match k.trim() {
                "level" => {
                    policy.level = v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid mg policy level: {v}"))
                    })?
                }
                "level_key" | "family_key" => policy.level_key = Some(v.trim().to_lowercase()),
                "smoother" => policy.smoother_type = Some(v.trim().to_lowercase()),
                "smoother_family" | "family" => {
                    policy.smoother_family = Some(v.trim().to_lowercase())
                }
                "steps" | "sweeps" => {
                    policy.smoother_steps = Some(v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid mg policy steps: {v}"))
                    })?)
                }
                "pre_sweeps" => {
                    policy.pre_sweeps =
                        Some(v.trim().parse().map_err(|_| {
                            KError::InvalidInput(format!("invalid mg pre sweeps: {v}"))
                        })?)
                }
                "post_sweeps" => {
                    policy.post_sweeps = Some(v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid mg post sweeps: {v}"))
                    })?)
                }
                "side" | "smoother_side" => {
                    policy.smoother_side = Some(PcSide::from_str(v.trim())?)
                }
                "coarse_pc" => policy.coarse_pc_type = Some(v.trim().to_lowercase()),
                "coarse_ksp" => policy.coarse_ksp_type = Some(v.trim().to_lowercase()),
                "coarse_maxits" => {
                    policy.coarse_ksp_maxits = Some(v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid mg coarse maxits: {v}"))
                    })?)
                }
                "coarse_rtol" => {
                    policy.coarse_ksp_rtol = Some(v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid mg coarse rtol: {v}"))
                    })?)
                }
                "coarse_side" => policy.coarse_side = Some(PcSide::from_str(v.trim())?),
                "coarse_route" | "coarse_routes" => {
                    let routes = v
                        .split('|')
                        .flat_map(|chunk| chunk.split(','))
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_lowercase())
                        .collect::<Vec<_>>();
                    if !routes.is_empty() {
                        policy.coarse_routes = Some(routes);
                    }
                }
                "ksp" | "ksp_type" => policy.level_ksp_type = Some(v.trim().to_lowercase()),
                "pc" | "pc_type" => policy.level_pc_type = Some(v.trim().to_lowercase()),
                "ksp_maxits" | "maxits" => {
                    policy.level_ksp_maxits = Some(v.trim().parse().map_err(|_| {
                        KError::InvalidInput(format!("invalid mg level maxits: {v}"))
                    })?)
                }
                "ksp_rtol" | "rtol" => {
                    policy.level_ksp_rtol =
                        Some(v.trim().parse().map_err(|_| {
                            KError::InvalidInput(format!("invalid mg level rtol: {v}"))
                        })?)
                }
                _ => {}
            }
        }
    }
    Ok(policy)
}

fn mg_policy_from_scoped_level(
    global: &PcOptions,
    level: usize,
    scoped: &PcOptions,
) -> MgLevelPolicy {
    let coarse_routes_from_policy = scoped
        .amg_dist_coarse_policy
        .as_deref()
        .or(global.amg_dist_coarse_policy.as_deref())
        .map(|policy| match policy {
            "local" | "local_prototype" | "hybrid" => {
                vec!["pc_apply".to_string(), "nested_ksp".to_string()]
            }
            "root" | "root_gather" | "auto" | "superlu_dist" => {
                vec!["nested_ksp".to_string(), "pc_apply".to_string()]
            }
            _ => vec!["nested_ksp".to_string(), "pc_apply".to_string()],
        });
    let inherited_pc_type = scoped
        .pc_type
        .clone()
        .or_else(|| global.pc_mg_smoother.clone());
    let inherited_ksp_type = scoped
        .pc_ksp_ksp_type
        .clone()
        .or_else(|| global.pc_ksp_ksp_type.clone());
    let inherited_ksp_pc = scoped
        .pc_ksp_pc_type
        .clone()
        .or_else(|| scoped.pc_type.clone())
        .or_else(|| global.pc_ksp_pc_type.clone())
        .or_else(|| global.pc_mg_smoother.clone());
    MgLevelPolicy {
        level,
        level_key: None,
        smoother_type: inherited_pc_type.clone().map(|v| v.to_lowercase()),
        smoother_family: inherited_pc_type.map(|v| v.to_lowercase()),
        smoother_steps: scoped.pc_mg_smoother_steps.or(global.pc_mg_smoother_steps),
        pre_sweeps: scoped.amg_sweeps_down.or(global.amg_sweeps_down),
        post_sweeps: scoped.amg_sweeps_up.or(global.amg_sweeps_up),
        smoother_side: None,
        coarse_pc_type: scoped
            .pc_mg_coarse_pc_type
            .clone()
            .or_else(|| scoped.pc_type.clone())
            .or_else(|| global.pc_mg_coarse_pc_type.clone())
            .map(|v| v.to_lowercase()),
        coarse_ksp_type: scoped
            .pc_mg_coarse_ksp_type
            .clone()
            .or_else(|| global.pc_mg_coarse_ksp_type.clone())
            .map(|v| v.to_lowercase()),
        coarse_ksp_maxits: scoped
            .pc_mg_coarse_ksp_maxits
            .or(scoped.pc_ksp_maxits)
            .or(global.pc_mg_coarse_ksp_maxits)
            .or(global.pc_ksp_maxits),
        coarse_ksp_rtol: scoped
            .pc_mg_coarse_ksp_rtol
            .or(scoped.pc_ksp_rtol)
            .or(global.pc_mg_coarse_ksp_rtol)
            .or(global.pc_ksp_rtol),
        coarse_side: None,
        coarse_routes: scoped
            .amg_dist_coarse_solver_route
            .as_ref()
            .map(|v| {
                v.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_lowercase())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .or(coarse_routes_from_policy)
            .or_else(|| {
                global.amg_dist_coarse_solver_route.as_ref().map(|v| {
                    v.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_lowercase())
                        .collect::<Vec<_>>()
                })
            })
            .filter(|v| !v.is_empty()),
        level_ksp_type: inherited_ksp_type.map(|v| v.to_lowercase()),
        level_pc_type: inherited_ksp_pc.map(|v| v.to_lowercase()),
        level_ksp_maxits: scoped.pc_ksp_maxits.or(global.pc_ksp_maxits),
        level_ksp_rtol: scoped.pc_ksp_rtol.or(global.pc_ksp_rtol),
    }
}

fn merge_mg_policy(dst: &mut MgLevelPolicy, src: &MgLevelPolicy) {
    if let Some(v) = src.level_key.as_ref() {
        dst.level_key = Some(v.clone());
    }
    if let Some(v) = src.smoother_type.as_ref() {
        dst.smoother_type = Some(v.clone());
    }
    if let Some(v) = src.smoother_family.as_ref() {
        dst.smoother_family = Some(v.clone());
    }
    if let Some(v) = src.smoother_steps {
        dst.smoother_steps = Some(v);
    }
    if let Some(v) = src.pre_sweeps {
        dst.pre_sweeps = Some(v);
    }
    if let Some(v) = src.post_sweeps {
        dst.post_sweeps = Some(v);
    }
    if let Some(v) = src.smoother_side {
        dst.smoother_side = Some(v);
    }
    if let Some(v) = src.coarse_pc_type.as_ref() {
        dst.coarse_pc_type = Some(v.clone());
    }
    if let Some(v) = src.coarse_ksp_type.as_ref() {
        dst.coarse_ksp_type = Some(v.clone());
    }
    if let Some(v) = src.coarse_ksp_maxits {
        dst.coarse_ksp_maxits = Some(v);
    }
    if let Some(v) = src.coarse_ksp_rtol {
        dst.coarse_ksp_rtol = Some(v);
    }
    if let Some(v) = src.coarse_side {
        dst.coarse_side = Some(v);
    }
    if let Some(v) = src.coarse_routes.as_ref() {
        dst.coarse_routes = Some(v.clone());
    }
    if let Some(v) = src.level_ksp_type.as_ref() {
        dst.level_ksp_type = Some(v.clone());
    }
    if let Some(v) = src.level_pc_type.as_ref() {
        dst.level_pc_type = Some(v.clone());
    }
    if let Some(v) = src.level_ksp_maxits {
        dst.level_ksp_maxits = Some(v);
    }
    if let Some(v) = src.level_ksp_rtol {
        dst.level_ksp_rtol = Some(v);
    }
}

impl PcConfig {
    pub fn from_type_and_options(
        pc_type: PcType,
        opts: Option<&PcOptions>,
    ) -> Result<Self, KError> {
        use PcType::*;
        let default_opts = PcOptions::default();
        let o = opts.unwrap_or(&default_opts);
        let conditioning = o.conditioning_options()?;
        Ok(match pc_type {
            None => PcConfig::None,

            Jacobi => match o.jacobi_block_size {
                Some(b) if b > 1 => PcConfig::BlockJacobi { block: b },
                _ => PcConfig::Jacobi,
            },

            Ilu0 => PcConfig::Ilu0 {
                conditioning: conditioning.clone(),
            },

            Ilu => match o.ilu_variant.as_deref() {
                Some("ilu0") | Option::None
                    if o.ilu_level.is_none() && o.ilut_drop_tol.is_none() =>
                {
                    PcConfig::Ilu0 {
                        conditioning: conditioning.clone(),
                    }
                }
                Some("iluk") | Option::None if o.ilu_level.is_some() => {
                    let level = o.ilu_level.ok_or_else(|| {
                        KError::InvalidInput("iluk requires PcOptions.ilu_level".into())
                    })?;
                    PcConfig::Iluk {
                        level,
                        conditioning: conditioning.clone(),
                    }
                }
                Some("ilut") | Option::None if o.ilut_drop_tol.is_some() => PcConfig::Ilut {
                    drop_tol: o.ilut_drop_tol.unwrap_or(1e-4),
                    max_fill: o.ilut_max_fill.unwrap_or(20),
                    reordering: o.ilu_reordering.clone(),
                    conditioning: conditioning.clone(),
                },
                Some("milu0") => PcConfig::Milu0 {
                    conditioning: conditioning.clone(),
                },
                Some(other) => {
                    return Err(KError::InvalidInput(format!(
                        "unknown ilu_variant: {other}"
                    )));
                }
                Option::None => PcConfig::Ilu0 {
                    conditioning: conditioning.clone(),
                },
            },
            Ilut => PcConfig::Ilut {
                drop_tol: o.ilut_drop_tol.unwrap_or(1e-4),
                max_fill: o.ilut_max_fill.unwrap_or(20),
                reordering: o.ilu_reordering.clone(),
                conditioning: conditioning.clone(),
            },
            Ilutp => PcConfig::Ilutp {
                drop_tol: o.ilutp_drop_tol.unwrap_or(1e-4),
                max_fill: o.ilutp_max_fill.unwrap_or(10),
                perm_tol: o.ilutp_perm_tol.unwrap_or(0.1),
                reordering: o.ilu_reordering.clone(),
                conditioning: conditioning.clone(),
            },
            Ilup => PcConfig::Iluk {
                level: o.ilu_level.unwrap_or(0),
                conditioning: conditioning.clone(),
            },

            Sor => {
                let mut mat_side = if let Some(ref side) = o.sor_mat_side {
                    match SorMatSideKind::from_str(side)? {
                        SorMatSideKind::Lower => MatSorSide::APPLY_LOWER,
                        SorMatSideKind::Upper => MatSorSide::APPLY_UPPER,
                        SorMatSideKind::Symmetric => MatSorSide::SYMMETRIC_SWEEP,
                        SorMatSideKind::Eisenstat => {
                            MatSorSide::SYMMETRIC_SWEEP | MatSorSide::EISENSTAT
                        }
                    }
                } else {
                    MatSorSide::APPLY_LOWER
                };
                if o.sor_symmetric.unwrap_or(false) {
                    mat_side |= MatSorSide::SYMMETRIC_SWEEP;
                }
                let omega = o.sor_omega.unwrap_or(1.0);
                if !(0.0..2.0).contains(&omega) {
                    return Err(KError::InvalidInput("sor_omega must be in (0,2)".into()));
                }
                PcConfig::Sor {
                    omega,
                    sweeps: o.sor_sweeps.unwrap_or(1),
                    mat_side,
                }
            }

            Chebyshev => {
                let degree = o.cheb_degree.unwrap_or(2);
                let eig_lo = o.cheb_eig_lo.unwrap_or(0.0);
                let eig_hi = o.cheb_eig_hi.unwrap_or(1.0);
                if degree < 1 || eig_hi <= eig_lo || eig_lo < 0.0 {
                    return Err(KError::InvalidInput("invalid Chebyshev bounds".into()));
                }
                PcConfig::Chebyshev {
                    degree,
                    eig_lo,
                    eig_hi,
                }
            }

            Asm => PcConfig::Asm {
                overlap: o.asm_overlap.unwrap_or(0),
                subdomain_hint: o.asm_subdomain_size,
                block_solver: o.asm_block_solver.clone(),
                mode: o.asm_mode.clone(),
                weighting: o.asm_weighting.clone(),
                inner_pc: match o.asm_inner_pc.as_deref() {
                    Some("jacobi") => AsmInnerPc::Jacobi,
                    Some("ilut") => AsmInnerPc::Ilut {
                        drop_tol: o.ilut_drop_tol.unwrap_or(1e-4),
                        max_fill: o.ilut_max_fill.unwrap_or(20),
                    },
                    Some("ilutp") => AsmInnerPc::Ilutp {
                        drop_tol: o.ilutp_drop_tol.unwrap_or(1e-4),
                        max_fill: o.ilutp_max_fill.unwrap_or(10),
                        perm_tol: o.ilutp_perm_tol.unwrap_or(0.1),
                    },
                    Some("ilu") | Some("ilu0") | std::option::Option::None => AsmInnerPc::Ilu0,
                    Some(other) => {
                        return Err(KError::InvalidInput(format!(
                            "unknown pc_asm_inner_pc: {other}"
                        )));
                    }
                },
            },
            Amg => {
                let cfg = AMGConfig::try_from_opts(o)?;
                PcConfig::Amg {
                    config: cfg,
                    conditioning: conditioning.clone(),
                }
            }

            ApproxInverse => {
                // Interpret options for CSR-based SPAI/FSAI
                let kind = match o
                    .approxinv_kind
                    .as_deref()
                    .unwrap_or("fsai")
                    .to_lowercase()
                    .as_str()
                {
                    "fsai" => ApproxInvKindAlias::FSAI,
                    "spai" => ApproxInvKindAlias::SPAI,
                    other => {
                        return Err(KError::InvalidInput(format!(
                            "unknown pc_approxinv_kind: {other}"
                        )));
                    }
                };
                let levels = o.approxinv_levels.unwrap_or(1);
                let max_per_col = o.approxinv_max_per_col.unwrap_or(20);
                let drop_tol = o.approxinv_drop_tol.or(o.drop_tol).unwrap_or(1e-3);
                let reg = o.approxinv_reg.unwrap_or(1e-12);
                let max_cond = o.approxinv_max_cond.unwrap_or(1e12);
                let parallel = o.approxinv_parallel.unwrap_or(cfg!(feature = "rayon"));
                PcConfig::ApproxInv {
                    kind,
                    levels,
                    max_per_col,
                    drop_tol,
                    reg,
                    max_cond,
                    parallel,
                }
            }
            FieldSplit => {
                let block_sizes = o
                    .pc_fieldsplit_block_sizes
                    .clone()
                    .unwrap_or_else(|| vec![1]);
                PcConfig::FieldSplit {
                    block_sizes,
                    child_pc_type: o.pc_fieldsplit_child_pc_type.clone(),
                    options: o.clone(),
                }
            }
            Shell => PcConfig::Shell {
                name: o.pc_shell_apply.clone().or_else(|| o.pc_shell_name.clone()),
                apply_transpose: o.pc_shell_apply_transpose.clone(),
                apply_conjugate_transpose: o.pc_shell_apply_conjugate_transpose.clone(),
                apply_symmetric: o.pc_shell_apply_symmetric.clone(),
                apply_symmetric_left: o.pc_shell_apply_symmetric_left.clone(),
                apply_symmetric_right: o.pc_shell_apply_symmetric_right.clone(),
                setup: o.pc_shell_setup.clone(),
                destroy: o.pc_shell_destroy.clone(),
                context: o.pc_shell_context.clone(),
            },
            Ksp => PcConfig::Ksp {
                ksp_options: {
                    let mut ksp = o.pc_ksp_ksp_options.clone().unwrap_or_default();
                    if ksp.ksp_type.is_none() && o.pc_ksp_ksp_type.is_some() {
                        ksp.ksp_type = o.pc_ksp_ksp_type.clone();
                    }
                    if ksp.maxits.is_none() && o.pc_ksp_maxits.is_some() {
                        ksp.maxits = o.pc_ksp_maxits;
                    }
                    if ksp.rtol.is_none() && o.pc_ksp_rtol.is_some() {
                        ksp.rtol = o.pc_ksp_rtol;
                    }
                    if ksp.atol.is_none() && o.pc_ksp_atol.is_some() {
                        ksp.atol = o.pc_ksp_atol;
                    }
                    if ksp.dtol.is_none() && o.pc_ksp_dtol.is_some() {
                        ksp.dtol = o.pc_ksp_dtol;
                    }
                    if ksp.restart.is_none() && o.pc_ksp_restart.is_some() {
                        ksp.restart = o.pc_ksp_restart;
                    }
                    if ksp.gmres_orthog.is_none() && o.pc_ksp_gmres_orthog.is_some() {
                        ksp.gmres_orthog = o.pc_ksp_gmres_orthog.clone();
                    }
                    if ksp.fgmres_orthog.is_none() && o.pc_ksp_fgmres_orthog.is_some() {
                        ksp.fgmres_orthog = o.pc_ksp_fgmres_orthog.clone();
                    }
                    if ksp.ksp_monitor_rank0.is_none() && o.pc_ksp_monitor_rank0.is_some() {
                        ksp.ksp_monitor_rank0 = o.pc_ksp_monitor_rank0;
                    }
                    if ksp.reduction.is_none() && o.pc_ksp_reduction.is_some() {
                        ksp.reduction = o.pc_ksp_reduction.clone();
                    }
                    if ksp.reproducible.is_none() && o.pc_ksp_reproducible.is_some() {
                        ksp.reproducible = o.pc_ksp_reproducible;
                    }
                    if ksp.threads.is_none() && o.pc_ksp_threads.is_some() {
                        ksp.threads = o.pc_ksp_threads;
                    }
                    if ksp.threads_mode.is_none() && o.pc_ksp_threads_mode.is_some() {
                        ksp.threads_mode = o.pc_ksp_threads_mode.clone();
                    }
                    if ksp.pc_side.is_none() && o.pc_ksp_pc_side.is_some() {
                        ksp.pc_side = o.pc_ksp_pc_side.clone();
                    }
                    ksp
                },
                pc_options: {
                    let mut pc = o
                        .pc_ksp_pc_options
                        .as_ref()
                        .map(|opts| opts.as_ref().clone())
                        .unwrap_or_default();
                    if pc.pc_type.is_none() && o.pc_ksp_pc_type.is_some() {
                        pc.pc_type = o.pc_ksp_pc_type.clone();
                    }
                    pc
                },
            },
            Mg => PcConfig::Mg {
                levels: o.pc_mg_levels.unwrap_or(2),
                cycle_type: o.pc_mg_cycle_type.clone(),
                smoother: o.pc_mg_smoother.clone(),
                smoother_steps: o.pc_mg_smoother_steps,
                coarsen_type: o.pc_mg_coarsen_type.clone(),
                interpolation_type: o.pc_mg_interpolation_type.clone(),
                restriction_type: o.pc_mg_restriction_type.clone(),
                coarse_pc_type: o.pc_mg_coarse_pc_type.clone(),
                coarse_ksp_type: o.pc_mg_coarse_ksp_type.clone(),
                coarse_ksp_maxits: o.pc_mg_coarse_ksp_maxits,
                coarse_ksp_rtol: o.pc_mg_coarse_ksp_rtol,
                level_policies: {
                    let mut merged: std::collections::BTreeMap<usize, MgLevelPolicy> =
                        std::collections::BTreeMap::new();
                    for policy in o
                        .pc_mg_level_policies
                        .as_ref()
                        .map(|entries| {
                            entries
                                .iter()
                                .filter_map(|entry| parse_mg_level_policy(entry).ok())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                    {
                        let entry = merged.entry(policy.level).or_insert_with(|| MgLevelPolicy {
                            level: policy.level,
                            ..Default::default()
                        });
                        merge_mg_policy(entry, &policy);
                    }
                    for (level, scoped) in &o.pc_mg_level_scoped_options {
                        let scoped_policy = mg_policy_from_scoped_level(o, *level, scoped);
                        let entry = merged.entry(*level).or_insert_with(|| MgLevelPolicy {
                            level: *level,
                            ..Default::default()
                        });
                        merge_mg_policy(entry, &scoped_policy);
                    }
                    merged.into_values().collect()
                },
            },
            Bddc => PcConfig::Bddc {
                coarse_ksp_type: o.pc_bddc_coarse_ksp_type.clone(),
                coarse_pc_type: o.pc_bddc_coarse_pc_type.clone(),
                use_vertices: o.pc_bddc_use_vertices.unwrap_or(false),
                constraint_selection: match o.pc_bddc_constraint_selection.as_deref() {
                    Some("vertices") => BddcConstraintSelection::Vertices,
                    Some("vertices_and_interface") | Some("all") => {
                        BddcConstraintSelection::VerticesAndInterface
                    }
                    _ => BddcConstraintSelection::Interface,
                },
                scaling: match o.pc_bddc_scaling.as_deref() {
                    Some("deluxe") | Some("deluxe_like") => BddcScaling::DeluxeLike,
                    _ => BddcScaling::Uniform,
                },
            },
            Gamg => {
                let cfg = GamgConfig::try_from_opts(o)?;
                PcConfig::Gamg {
                    config: cfg,
                    conditioning: conditioning.clone(),
                }
            }

            Lu => PcConfig::Lu,
            Qr => PcConfig::Qr,
            #[cfg(feature = "superlu_dist")]
            SuperLuDist => PcConfig::SuperLuDist,
            BlockJacobi => PcConfig::BlockJacobi {
                block: o.jacobi_block_size.unwrap_or(1),
            },
        })
    }
}

/// # PcFactory
///
/// Runtime selection of preconditioners with option parsing.
///
/// - `PcOptions` → typed `PcConfig` → concrete builder
/// - Feature gates:
///   - `superlu_dist`: enables the SuperLU_DIST preconditioner
///   - `legacy-pc-bridge`: enables adapters for legacy implementations (no per-apply allocs)
///
/// ## Chains
/// - String form: `"jacobi->ilut"` via [`PcFactory::create_pc_chain_from_str`]
/// - Structured form: `PcOptions.chain: Vec<PcOptions>`
/// - Construction is deferred until a matrix is available (see KSP docs).
///
/// Suites of ILU/ILUT options stay available when the crate is built with `feature = "complex"`
/// because the factory relies on the real `Ilu` and `Ilutp` implementations plus the `KPreconditioner`
/// bridge (`BridgeScratch`). When the specialized `backend-faer` path is unavailable, we fall back
/// to the generic `Ilup` or `Ilut` implementations that already operate over the Kryst scalar `S`.
pub struct PcFactory;

impl PcFactory {
    fn composite_mode_from_opts(
        opts: Option<&PcOptions>,
    ) -> Result<crate::preconditioner::chain::PcCompositeMode, KError> {
        match opts.and_then(|o| o.pc_composite_type.as_deref()) {
            None | Some("multiplicative") | Some("mul") => {
                Ok(crate::preconditioner::chain::PcCompositeMode::Multiplicative)
            }
            Some("additive") | Some("add") => {
                Ok(crate::preconditioner::chain::PcCompositeMode::Additive)
            }
            Some("schur") => Ok(crate::preconditioner::chain::PcCompositeMode::Schur),
            Some(other) => Err(KError::InvalidInput(format!(
                "unknown pc_composite_type: {other}"
            ))),
        }
    }

    fn split_chain_tokens(chain: &str) -> Vec<String> {
        chain
            .replace("->", ",")
            .replace('+', ",")
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|token| token.to_string())
            .collect()
    }

    #[inline]
    fn is_direct(pc: PcType) -> bool {
        match pc {
            PcType::Lu | PcType::Qr => true,
            #[cfg(feature = "superlu_dist")]
            PcType::SuperLuDist => true,
            _ => false,
        }
    }

    #[inline]
    fn chain_strict() -> bool {
        #[cfg(test)]
        if let Some(val) = CHAIN_STRICT_OVERRIDE.with(|cell| cell.get()) {
            return val;
        }
        // Opt-in strict mode via env var.
        // KRYST_PC_CHAIN_STRICT=1|true enforces selected warnings as errors.
        std::env::var("KRYST_PC_CHAIN_STRICT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    #[cfg(test)]
    pub(crate) fn override_chain_strict(value: Option<bool>) -> ChainStrictGuard {
        CHAIN_STRICT_OVERRIDE.with(|cell| {
            let prev = cell.replace(value);
            ChainStrictGuard(prev)
        })
    }

    /// Validate high-level invariants for a PC chain.
    /// - Emits log::warn! for suspect patterns.
    /// - If KRYST_PC_CHAIN_STRICT is set, some warnings become errors.
    fn validate_chain_specs(specs: &[DeferredPcInfo]) -> Result<(), KError> {
        if specs.is_empty() {
            return Err(KError::InvalidInput("empty PC chain".into()));
        }

        let strict = Self::chain_strict();

        // Rule 1: multiple direct PCs
        let direct_positions: Vec<usize> = specs
            .iter()
            .enumerate()
            .filter_map(|(i, s)| Self::is_direct(s.pc_type).then_some(i))
            .collect();
        if direct_positions.len() > 1 {
            let msg = format!(
                "PC chain contains multiple direct PCs at positions {direct_positions:?}. \
                 Stacking direct factorizations is usually unintended."
            );
            if strict {
                return Err(KError::InvalidInput(msg));
            } else {
                log::warn!("{msg}");
            }
        }

        // Rule 2: direct PC should be last
        if let Some((i, s)) = specs
            .iter()
            .enumerate()
            .find(|(i, s)| Self::is_direct(s.pc_type) && *i + 1 != specs.len())
        {
            let msg = format!(
                "Direct PC {:?} is not the last stage (index {}, chain len {}). \
                 Subsequent stages will likely be redundant or ignored.",
                s.pc_type,
                i,
                specs.len()
            );
            if strict {
                return Err(KError::InvalidInput(msg));
            } else {
                log::warn!("{msg}");
            }
        }

        // Rule 3: consecutive duplicates (same PcType twice)
        // Intentionally warn-only (even in strict mode) to avoid flakiness when tests
        // mutate environment variables concurrently. Redundant stages are allowed.
        for w in specs.windows(2) {
            if w[0].pc_type == w[1].pc_type {
                let msg = format!(
                    "Consecutive duplicate PCs: {:?} -> {:?}. \
                     This is typically redundant unless options differ.",
                    w[0].pc_type, w[1].pc_type
                );
                log::warn!("{msg}");
            }
        }

        // Rule 4: BlockJacobi block_size <= 1 behaves like Jacobi
        for (i, spec) in specs.iter().enumerate() {
            if matches!(spec.pc_type, PcType::BlockJacobi)
                && let Some(ref o) = spec.options
                && o.jacobi_block_size.unwrap_or(1) <= 1
            {
                log::warn!(
                    "PC chain stage {i}: BlockJacobi with block_size <= 1 behaves like Jacobi; \
                             consider using 'jacobi' instead."
                );
            }
        }

        Ok(())
    }
    pub fn create_preconditioner(
        pc_type: PcType,
        options: Option<&PcOptions>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        let cfg = PcConfig::from_type_and_options(pc_type, options)?;
        if let Some(pc) = crate::preconditioner::builders_none::try_build(&cfg)? {
            return Ok(pc);
        }

        #[cfg(feature = "backend-faer")]
        if let Some(pc) = crate::preconditioner::builders_faer::try_build(&cfg)? {
            return Ok(pc);
        }

        #[cfg(feature = "backend-nalgebra")]
        if let Some(pc) = crate::preconditioner::builders_nalgebra::try_build(&cfg)? {
            return Ok(pc);
        }

        Err(KError::InvalidInput(format!(
            "Preconditioner {:?} requires a backend that is not enabled/supported for this build",
            pc_type
        )))
    }

    /// Convenience: build directly from options (when `pc_type` lives inside options)
    pub fn create_from_options(opts: &PcOptions) -> Result<Box<dyn Preconditioner>, KError> {
        let pct = if let Some(ref s) = opts.pc_type {
            PcType::from_str(s)?
        } else {
            PcType::None
        };
        Self::create_preconditioner(pct, Some(opts))
    }

    pub fn create_deferred_pc(
        pc_type: PcType,
        options: Option<PcOptions>,
    ) -> Result<DeferredPcInfo, KError> {
        Ok(DeferredPcInfo { pc_type, options })
    }

    pub fn construct_deferred_preconditioner(
        info: DeferredPcInfo,
        _op: &dyn LinOp<S = S>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        // The concrete operator format is deferred to the preconditioner itself.
        Self::create_preconditioner(info.pc_type, info.options.as_ref())
    }

    /// Parse a string chain and clone the same [`PcOptions`] for every stage.
    ///
    /// To tune stages individually, populate [`PcOptions::chain`].
    pub fn create_pc_chain_from_str(
        chain: &str,
        opts: Option<&PcOptions>,
    ) -> Result<Vec<DeferredPcInfo>, KError> {
        let mut specs = Vec::new();
        let prefixes = opts
            .and_then(|o| o.pc_composite_prefixes.clone())
            .unwrap_or_default();
        for (i, token) in Self::split_chain_tokens(chain).into_iter().enumerate() {
            let pct = PcType::from_str(&token)?;
            let mut stage_opts = opts.cloned();
            if let Some(prefix) = prefixes.get(i)
                && let Some(scoped) = opts.and_then(|o| o.scoped_child(prefix)).cloned()
            {
                let mut merged = stage_opts.unwrap_or_default();
                merged.overlay_from(scoped);
                stage_opts = Some(merged);
            }
            if token.eq_ignore_ascii_case("ras") {
                stage_opts.get_or_insert_with(PcOptions::default).asm_mode =
                    Some("ras".to_string());
            }
            specs.push(DeferredPcInfo {
                pc_type: pct,
                options: stage_opts,
            });
        }
        if specs.is_empty() {
            return Err(KError::InvalidInput("empty PC chain".into()));
        }
        // validate
        Self::validate_chain_specs(&specs)?;
        Ok(specs)
    }

    /// Parse a string containing fallback chains separated by `||`.
    ///
    /// Example: `"amg||ras+ilutp"` tries AMG first, then RAS+ILUTP on setup failure.
    pub fn create_pc_chain_candidates_from_str(
        chain: &str,
        opts: Option<&PcOptions>,
    ) -> Result<Vec<Vec<DeferredPcInfo>>, KError> {
        let mut candidates = Vec::new();
        for candidate in chain
            .split("||")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            candidates.push(Self::create_pc_chain_from_str(candidate, opts)?);
        }
        if candidates.is_empty() {
            return Err(KError::InvalidInput("empty PC chain".into()));
        }
        Ok(candidates)
    }

    pub fn construct_deferred_pc_chain(
        specs: Vec<DeferredPcInfo>,
        op: &dyn LinOp<S = S>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        // validate again in case specs were assembled elsewhere
        Self::validate_chain_specs(&specs)?;
        use crate::preconditioner::chain::PcChain;

        let mode = Self::composite_mode_from_opts(specs.first().and_then(|s| s.options.as_ref()))?;
        let mut stages: Vec<Box<dyn Preconditioner>> = Vec::with_capacity(specs.len());
        for (i, spec) in specs.into_iter().enumerate() {
            let pc_type = spec.pc_type;
            let stage = Self::construct_deferred_preconditioner(spec, op).map_err(|e| {
                KError::InvalidInput(format!("PC chain stage {i} ({pc_type:?}) failed: {e}",))
            })?;
            stages.push(stage);
        }
        Ok(Box::new(PcChain::with_mode(stages, mode)))
    }

    pub fn create_pc_chain(
        chain: &str,
        op: &dyn LinOp<S = S>,
        opts: Option<PcOptions>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        let specs = Self::create_pc_chain_from_str(chain, opts.as_ref())?;
        Self::construct_deferred_pc_chain(specs, op)
    }

    pub fn create_deferred_pc_chain_from_options(
        chain_opts: &[PcOptions],
    ) -> Result<Vec<DeferredPcInfo>, KError> {
        let mut specs = Vec::with_capacity(chain_opts.len());
        for co in chain_opts {
            let pct = if let Some(ref s) = co.pc_type {
                PcType::from_str(s)?
            } else {
                return Err(KError::InvalidInput(
                    "PcOptions in chain missing pc_type".into(),
                ));
            };
            specs.push(DeferredPcInfo {
                pc_type: pct,
                options: Some(co.clone()),
            });
        }
        if specs.is_empty() {
            return Err(KError::InvalidInput("empty PcOptions.chain".into()));
        }
        // validate
        Self::validate_chain_specs(&specs)?;
        Ok(specs)
    }
}

/// Sparsity pattern for approximate inverse preconditioner.
#[derive(Clone, Debug)]
pub enum SparsityPattern {
    Manual(Vec<Vec<usize>>),
    Auto,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preconditioner::Preconditioner;

    #[cfg(feature = "dense-direct")]
    #[test]
    fn factory_builds_lu_qr() {
        let lu = PcFactory::create_preconditioner(PcType::from_str("lu").unwrap(), None).unwrap();
        let qr = PcFactory::create_preconditioner(PcType::from_str("qr").unwrap(), None).unwrap();

        fn _is_pc(_p: &Box<dyn Preconditioner>) {}
        _is_pc(&lu);
        _is_pc(&qr);
    }

    #[cfg(feature = "legacy-pc-bridge")]
    #[test]
    fn factory_uses_options_for_ilut() {
        let opts = PcOptions {
            pc_type: Some("ilut".into()),
            ilut_drop_tol: Some(1e-6),
            ilut_max_fill: Some(50),
            ..Default::default()
        };
        let pc = PcFactory::create_from_options(&opts).unwrap();
        fn _is_pc(_: &Box<dyn Preconditioner>) {}
        _is_pc(&pc);
    }

    #[cfg(feature = "legacy-pc-bridge")]
    #[test]
    fn factory_builds_sor_from_options() {
        let opts = PcOptions {
            pc_type: Some("sor".into()),
            sor_omega: Some(1.5),
            sor_sweeps: Some(2),
            sor_mat_side: Some("lower".into()),
            ..Default::default()
        };
        let pc = PcFactory::create_from_options(&opts).unwrap();
        fn _is_pc(_: &Box<dyn Preconditioner>) {}
        _is_pc(&pc);
    }

    #[cfg(feature = "backend-faer")]
    #[test]
    fn chebyshev_validates_bounds() {
        let bad = PcOptions {
            pc_type: Some("chebyshev".into()),
            cheb_degree: Some(0),
            cheb_eig_lo: Some(2.0),
            cheb_eig_hi: Some(1.0),
            ..Default::default()
        };
        let err = PcFactory::create_from_options(&bad).err().unwrap();
        assert!(matches!(err, KError::InvalidInput(_)));
    }

    #[cfg(feature = "backend-faer")]
    #[test]
    fn factory_builds_asm_from_options() {
        let opts = crate::config::options::PcOptions {
            pc_type: Some("asm".into()),
            asm_block_solver: Some("ludense".into()),
            ..Default::default()
        };
        let pc = PcFactory::create_from_options(&opts).unwrap_or_else(|_| {
            // When dense-direct is disabled, builder still constructs ASM (LuDense maps to CSR fallback)
            PcFactory::create_from_options(&crate::config::options::PcOptions {
                pc_type: Some("asm".into()),
                asm_block_solver: Some("csr".into()),
                ..Default::default()
            })
            .unwrap()
        });
        fn _is_pc(_: &Box<dyn Preconditioner>) {}
        _is_pc(&pc);
    }

    #[test]
    fn chain_direct_not_last_is_error_in_strict_mode() {
        // flip strict mode via override for this test
        let _guard = PcFactory::override_chain_strict(Some(true));
        let opts = crate::config::options::PcOptions::default();

        // "lu->jacobi" => direct not last
        let specs = PcFactory::create_pc_chain_from_str("lu->jacobi", Some(&opts));
        assert!(specs.is_err(), "expected validation error in strict mode");
    }

    #[test]
    fn chain_duplicate_consecutive_warns_but_allows_by_default() {
        // Default (non-strict): should allow "ilu->ilu"
        let opts = crate::config::options::PcOptions::default();
        let specs = PcFactory::create_pc_chain_from_str("ilu->ilu", Some(&opts))
            .expect("duplicates allowed with warning by default");
        assert!(!specs.is_empty());
    }

    #[test]
    fn chain_fallback_parses_candidates_and_aliases() {
        let opts = crate::config::options::PcOptions::default();
        let candidates =
            PcFactory::create_pc_chain_candidates_from_str("amg||ras+ilutp", Some(&opts))
                .expect("fallback parse");
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0][0].pc_type, PcType::Amg);
        assert_eq!(candidates[1][0].pc_type, PcType::Asm);
        assert_eq!(candidates[1][1].pc_type, PcType::Ilutp);
        assert_eq!(
            candidates[1][0]
                .options
                .as_ref()
                .and_then(|o| o.asm_mode.as_deref()),
            Some("ras")
        );
    }
    #[test]
    fn chain_prefix_scoped_stage_options_are_merged() {
        let args = vec![
            "-pc_chain",
            "jacobi->ilu",
            "-pc_composite_prefixes",
            "s0_,s1_",
            "-s1_pc_ilu_levels",
            "4",
        ];
        let opts = crate::config::options::PcOptions::from_args(&args).unwrap();
        let specs =
            PcFactory::create_pc_chain_from_str(opts.pc_chain.as_deref().unwrap(), Some(&opts))
                .unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[1].options.as_ref().and_then(|o| o.ilu_level), Some(4));
    }
}
