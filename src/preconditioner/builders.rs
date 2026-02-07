use crate::error::KError;
use crate::preconditioner::amg::AMGConfig;
use crate::preconditioner::gamg::GamgConfig;
#[cfg(feature = "dense-direct")]
use crate::preconditioner::direct::{LuPc, QrPc};
use crate::preconditioner::{
    Preconditioner,
    bddc::BddcConfig,
    asm::{AsmCombine, AsmConfig, AsmLocalSolver},
    asm_amg::{AsmAmg, TwoLevelConfig, TwoLevelMode},
    block_jacobi::BlockJacobi,
    jacobi::Jacobi,
    sor::MatSorType,
};

use crate::preconditioner::asm::AsmInnerPc;
use crate::preconditioner::chebyshev::ChebyshevPc;
use crate::preconditioner::ilu_csr::ReorderingOptions;
use crate::preconditioner::sor::SorPc;
use crate::utils::conditioning::ConditioningOptions;

#[cfg(feature = "superlu_dist")]
use crate::preconditioner::direct::SuperLuDistPc;

// no faer::Mat needed here for object-safe builders

/// Build a Jacobi preconditioner.
pub fn build_jacobi() -> Result<Box<dyn Preconditioner>, KError> {
    Ok(Box::new(Jacobi::new()))
}

/// Build a Block Jacobi preconditioner.
pub fn build_block_jacobi(block: usize) -> Result<Box<dyn Preconditioner>, KError> {
    if block <= 1 {
        return build_jacobi();
    }
    #[cfg(feature = "complex")]
    {
        return Err(KError::Unsupported(
            "BlockJacobi is not supported for complex scalars".into(),
        ));
    }
    #[cfg(not(feature = "complex"))]
    {
        let pc = BlockJacobi {
            blocks: Vec::new(),
            block_size: block,
            #[cfg(feature = "dense-direct")]
            block_factors: Vec::new(),
            #[cfg(all(not(feature = "dense-direct"), not(feature = "complex")))]
            block_factors_ilu: Vec::new(),
        };
        Ok(Box::new(pc))
    }
}

/// Build an SOR preconditioner.
pub fn build_sor(
    omega: f64,
    sweeps: usize,
    mat_side: MatSorType,
) -> Result<Box<dyn Preconditioner>, KError> {
    let pc = SorPc::new(omega, sweeps, mat_side, 0.0);
    Ok(Box::new(pc))
}

/// Build a Chebyshev preconditioner.
pub fn build_chebyshev(
    degree: usize,
    eig_lo: f64,
    eig_hi: f64,
) -> Result<Box<dyn Preconditioner>, KError> {
    if degree < 1 {
        return Err(KError::InvalidInput("chebyshev degree must be >= 1".into()));
    }
    if !eig_lo.is_finite() || !eig_hi.is_finite() || eig_hi <= eig_lo || eig_lo < 0.0 {
        return Err(KError::InvalidInput(
            "invalid Chebyshev bounds (require 0 <= lambda_min < lambda_max)".into(),
        ));
    }
    let pc = ChebyshevPc::new(degree, eig_lo, eig_hi);
    Ok(Box::new(pc))
}

pub fn build_lu() -> Result<Box<dyn Preconditioner>, KError> {
    #[cfg(feature = "dense-direct")]
    {
        return Ok(Box::new(LuPc::new()));
    }
    #[cfg(not(feature = "dense-direct"))]
    {
        Err(KError::Unsupported("dense-direct feature not enabled"))
    }
}

pub fn build_qr() -> Result<Box<dyn Preconditioner>, KError> {
    #[cfg(feature = "dense-direct")]
    {
        return Ok(Box::new(QrPc::new()));
    }
    #[cfg(not(feature = "dense-direct"))]
    {
        Err(KError::Unsupported("dense-direct feature not enabled"))
    }
}

pub fn build_bddc(config: BddcConfig) -> Result<Box<dyn Preconditioner>, KError> {
    Ok(Box::new(crate::preconditioner::bddc::BddcPc::new(
        config,
    )))
}

pub fn build_superlu_dist() -> Result<Box<dyn Preconditioner>, KError> {
    #[cfg(feature = "superlu_dist")]
    {
        Ok(Box::new(SuperLuDistPc::new()))
    }
    #[cfg(not(feature = "superlu_dist"))]
    {
        Err(KError::Unsupported("superlu_dist feature not enabled"))
    }
}

// ---- ILU family builders -------------------------------------------------

pub fn build_ilu0() -> Result<Box<dyn Preconditioner>, KError> {
    build_ilu0_with_conditioning(ConditioningOptions::default())
}

pub fn build_ilu0_with_conditioning(
    conditioning: ConditioningOptions,
) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::ilu_csr::{
        IluCsr, IluCsrConfig, IluKind, PivotStrategy, ReorderingOptions,
    };
    let cfg = IluCsrConfig {
        kind: IluKind::Ilu0,
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: cfg!(feature = "rayon"),
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions::default(),
        conditioning,
    };
    let pc = IluCsr::new_with_config(cfg);
    Ok(Box::new(pc))
}

pub fn build_iluk(level: usize) -> Result<Box<dyn Preconditioner>, KError> {
    build_iluk_with_conditioning(level, ConditioningOptions::default())
}

pub fn build_iluk_with_conditioning(
    level: usize,
    conditioning: ConditioningOptions,
) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::ilu_csr::{
        IluCsr, IluCsrConfig, IluKind, PivotStrategy, ReorderingOptions,
    };
    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: level },
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: cfg!(feature = "rayon"),
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions::default(),
        conditioning,
    };
    Ok(Box::new(IluCsr::new_with_config(cfg)))
}

pub fn build_ilut(
    drop_tol: f64,
    max_fill: usize,
    _reordering: Option<String>,
) -> Result<Box<dyn Preconditioner>, KError> {
    build_ilut_with_conditioning(
        drop_tol,
        max_fill,
        _reordering,
        ConditioningOptions::default(),
    )
}

pub fn build_ilut_with_conditioning(
    drop_tol: f64,
    max_fill: usize,
    reordering: Option<String>,
    conditioning: ConditioningOptions,
) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::ilu_csr::{
        IluCsr, IluCsrConfig, IluKind, IlutParams, PivotPolicy, PivotStrategy, Pivoting,
        ReorderingOptions,
    };
    let params = IlutParams {
        droptol_abs: drop_tol,
        droptol_rel: 0.0,
        p_l: max_fill,
        p_u: max_fill,
        early_drop: true,
        pivot: PivotPolicy::DiagonalPerturbation,
        pivot_tau: 1e-12,
        reproducible_order: true,
        pivoting: Pivoting::None,
    };
    let cfg = IluCsrConfig {
        kind: IluKind::Ilut { params },
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: cfg!(feature = "rayon"),
        // For ILUT, fast numeric update requires fixed pattern. Let callers override later if needed.
        numeric_update_fixed: true,
        logging: 0,
        reordering: parse_reordering_options(reordering)?,
        conditioning,
    };
    Ok(Box::new(IluCsr::new_with_config(cfg)))
}

pub fn build_ilutp(
    max_fill: usize,
    drop_tol: f64,
    perm_tol: f64,
    reordering: Option<String>,
) -> Result<Box<dyn Preconditioner>, KError> {
    build_ilutp_with_conditioning(
        max_fill,
        drop_tol,
        perm_tol,
        reordering,
        ConditioningOptions::default(),
    )
}

pub fn build_ilutp_with_conditioning(
    max_fill: usize,
    drop_tol: f64,
    perm_tol: f64,
    reordering: Option<String>,
    conditioning: ConditioningOptions,
) -> Result<Box<dyn Preconditioner>, KError> {
    #[cfg(all(feature = "legacy-pc-bridge", feature = "backend-faer"))]
    {
        use crate::preconditioner::ilutp::Ilutp;
        use crate::preconditioner::{LegacyOpPreconditioner, legacy::Preconditioner as LegacyPc};

        let mut pc = Ilutp::with_params(max_fill, drop_tol, perm_tol);
        if let Some(reordering) = reordering {
            let reordering = parse_reordering_options(Some(reordering))?;
            pc.set_reordering(reordering);
        }
        pc.set_conditioning(conditioning);
        let legacy: Box<dyn LegacyPc<faer::Mat<f64>, Vec<f64>> + Send + Sync> = Box::new(pc);
        return Ok(Box::new(LegacyOpPreconditioner::new(legacy)));
    }
    #[cfg(not(all(feature = "legacy-pc-bridge", feature = "backend-faer")))]
    {
        let _ = (max_fill, drop_tol, perm_tol, reordering, conditioning);
        Err(KError::Unsupported(
            "ILUTP requires features \"backend-faer\" and \"legacy-pc-bridge\"".into(),
        ))
    }
}

pub fn build_milu0() -> Result<Box<dyn Preconditioner>, KError> {
    build_milu0_with_conditioning(ConditioningOptions::default())
}

pub fn build_milu0_with_conditioning(
    conditioning: ConditioningOptions,
) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::ilu_csr::{
        IluCsr, IluCsrConfig, IluKind, PivotStrategy, ReorderingOptions,
    };
    let cfg = IluCsrConfig {
        kind: IluKind::Milu0,
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: cfg!(feature = "rayon"),
        numeric_update_fixed: true,
        logging: 0,
        reordering: ReorderingOptions::default(),
        conditioning,
    };
    Ok(Box::new(IluCsr::new_with_config(cfg)))
}

fn parse_reordering_options(reordering: Option<String>) -> Result<ReorderingOptions, KError> {
    use crate::preconditioner::ilu_csr::{ReorderingKind, ReorderingOptions};
    let mut opts = ReorderingOptions::default();
    let Some(kind) = reordering else {
        return Ok(opts);
    };
    opts.kind = match kind.to_lowercase().as_str() {
        "none" | "natural" => ReorderingKind::None,
        "rcm" => ReorderingKind::Rcm,
        "amd" => ReorderingKind::Amd,
        other => {
            return Err(KError::InvalidInput(format!(
                "unknown ILU reordering: {other}"
            )));
        }
    };
    Ok(opts)
}

// ---- ASM / AMG stubs -----------------------------------------------------

#[cfg(feature = "backend-faer")]
pub fn build_asm(
    overlap: usize,
    subdomain_hint: Option<usize>,
    block_solver: Option<String>,
    mode: Option<String>,
    weighting: Option<String>,
    inner_pc: AsmInnerPc,
) -> Result<Box<dyn Preconditioner>, KError> {
    // Map the optional block solver string to the enum used by AdditiveSchwarz.
    use crate::preconditioner::asm::{AsmBlockSolver, AsmInnerPc, AsmMode, AsmPc, Weighting};

    if block_solver
        .as_deref()
        .is_some_and(|s| s.eq_ignore_ascii_case("amg"))
    {
        return build_asm_amg(overlap);
    }

    let block_solver = match block_solver.as_deref() {
        Some("csr") => AsmBlockSolver::Csr,
        _ => AsmBlockSolver::LuDense,
    };

    let mode = match mode.as_deref() {
        Some("asm") => AsmMode::ASM,
        Some("ras") => AsmMode::RAS,
        Some(other) => {
            return Err(KError::InvalidInput(format!(
                "unknown pc_asm_mode: {other}"
            )));
        }
        None => AsmMode::ASM,
    };

    let weighting = match weighting.as_deref() {
        Some("none") => Weighting::None,
        Some("uniform") => Weighting::Uniform,
        Some("linear") => Weighting::SmoothLinear,
        Some(s) if s.starts_with("poly:") => {
            let pstr = &s[5..];
            let p: u32 = pstr.parse().map_err(|_| {
                KError::InvalidInput(format!("invalid poly exponent in pc_asm_weighting: {s}"))
            })?;
            if p < 2 {
                return Err(KError::InvalidInput("poly exponent must be >= 2".into()));
            }
            Weighting::SmoothPoly(p)
        }
        Some(other) => {
            return Err(KError::InvalidInput(format!(
                "unknown pc_asm_weighting: {other}"
            )));
        }
        None => {
            if overlap == 0 {
                Weighting::None
            } else {
                Weighting::Uniform
            }
        }
    };

    let asm = AsmPc::new(
        overlap,
        subdomain_hint,
        block_solver,
        inner_pc,
        mode,
        weighting,
    );
    Ok(Box::new(asm))
}

#[cfg(not(feature = "backend-faer"))]
pub fn build_asm(
    _overlap: usize,
    _hint: Option<usize>,
    _block_solver: Option<String>,
    _mode: Option<String>,
    _weighting: Option<String>,
    _inner_pc: crate::preconditioner::asm::AsmInnerPc,
) -> Result<Box<dyn Preconditioner>, KError> {
    Err(KError::Unsupported(
        "ASM builder requires the backend-faer feature".to_string(),
    ))
}

pub fn build_amg(cfg: AMGConfig) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::amg::AMG;
    let amg = AMG::with_config(cfg);
    Ok(Box::new(amg))
}

pub fn build_gamg(cfg: GamgConfig) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::gamg::Gamg;
    let gamg = Gamg::with_config(cfg);
    Ok(Box::new(gamg))
}

pub fn build_asm_amg(overlap: usize) -> Result<Box<dyn Preconditioner>, KError> {
    let mut asm_cfg = AsmConfig::default();
    asm_cfg.overlap = overlap;
    asm_cfg.combine = AsmCombine::Restricted;
    asm_cfg.local_solver = AsmLocalSolver::ILU;
    asm_cfg.deterministic = true;
    let mut two_cfg = TwoLevelConfig::default();
    two_cfg.mode = TwoLevelMode::AdditiveCoarse;
    two_cfg.coarse_every = 1;
    let pc = AsmAmg::with_configs(asm_cfg, two_cfg);
    Ok(Box::new(pc))
}
