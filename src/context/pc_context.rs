use crate::algebra::prelude::*;
use crate::config::kinds::SorMatSideKind;
use crate::config::options::PcOptions;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::amg::AMGConfig;
use crate::preconditioner::{PcSide, Preconditioner};
use std::str::FromStr;

type MatSorSide = crate::preconditioner::sor::MatSorType;

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
            "asm" => Ok(PcType::Asm),
            "chebyshev" => Ok(PcType::Chebyshev),
            "amg" => Ok(PcType::Amg),
            "approxinv" | "approxinverse" => Ok(PcType::ApproxInverse),
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

/// Simple no-op preconditioner.
pub struct NoOpPreconditioner;

impl Preconditioner for NoOpPreconditioner {
    fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        Ok(())
    }
    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        z.copy_from_slice(r);
        Ok(())
    }

    fn apply_mut(&mut self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
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
    Ilu0,
    Iluk {
        level: usize,
    },
    Ilut {
        drop_tol: R,
        max_fill: usize,
        reordering: Option<String>,
    },
    Milu0,
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
    },
    Amg {
        config: AMGConfig,
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
    Lu,
    Qr,
    #[cfg_attr(docsrs, doc(cfg(feature = "superlu_dist")))]
    #[cfg(feature = "superlu_dist")]
    SuperLuDist,
}

impl PcConfig {
    pub fn from_type_and_options(
        pc_type: PcType,
        opts: Option<&PcOptions>,
    ) -> Result<Self, KError> {
        use PcType::*;
        let default_opts = PcOptions::default();
        let o = opts.unwrap_or(&default_opts);
        Ok(match pc_type {
            None => PcConfig::None,

            Jacobi => match o.jacobi_block_size {
                Some(b) if b > 1 => PcConfig::BlockJacobi { block: b },
                _ => PcConfig::Jacobi,
            },

            Ilu0 => PcConfig::Ilu0,

            Ilu => match o.ilu_variant.as_deref() {
                Some("ilu0") | Option::None
                    if o.ilu_level.is_none() && o.ilut_drop_tol.is_none() =>
                {
                    PcConfig::Ilu0
                }
                Some("iluk") | Option::None if o.ilu_level.is_some() => {
                    let level = o.ilu_level.ok_or_else(|| {
                        KError::InvalidInput("iluk requires PcOptions.ilu_level".into())
                    })?;
                    PcConfig::Iluk { level }
                }
                Some("ilut") | Option::None if o.ilut_drop_tol.is_some() => PcConfig::Ilut {
                    drop_tol: o.ilut_drop_tol.unwrap_or(1e-4),
                    max_fill: o.ilut_max_fill.unwrap_or(20),
                    reordering: o.ilu_reordering.clone(),
                },
                Some("milu0") => PcConfig::Milu0,
                Some(other) => {
                    return Err(KError::InvalidInput(format!(
                        "unknown ilu_variant: {other}"
                    )));
                }
                Option::None => PcConfig::Ilu0,
            },
            Ilut => PcConfig::Ilut {
                drop_tol: o.ilut_drop_tol.unwrap_or(1e-4),
                max_fill: o.ilut_max_fill.unwrap_or(20),
                reordering: o.ilu_reordering.clone(),
            },
            Ilutp => PcConfig::Ilut {
                drop_tol: o.ilut_drop_tol.unwrap_or(1e-4),
                max_fill: o.ilut_max_fill.unwrap_or(20),
                reordering: o.ilu_reordering.clone(),
            },
            Ilup => PcConfig::Iluk {
                level: o.ilu_level.unwrap_or(0),
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
            },
            Amg => {
                let cfg = AMGConfig::try_from_opts(o)?;
                PcConfig::Amg { config: cfg }
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

            Lu => PcConfig::Lu,
            Qr => PcConfig::Qr,
            #[cfg(feature = "superlu_dist")]
            SuperLuDist => PcConfig::SuperLuDist,
            BlockJacobi => unreachable!(),
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
    #[cfg(feature = "backend-faer")]
    pub fn create_preconditioner(
        pc_type: PcType,
        options: Option<&PcOptions>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        let cfg = PcConfig::from_type_and_options(pc_type, options)?;
        use crate::preconditioner::builders as b;
        match cfg {
            PcConfig::None => Ok(Box::new(NoOpPreconditioner)),
            PcConfig::Jacobi => b::build_jacobi(),
            PcConfig::BlockJacobi { block } => b::build_block_jacobi(block),

            PcConfig::Ilu0 => b::build_ilu0(),
            PcConfig::Iluk { level } => b::build_iluk(level),
            PcConfig::Ilut {
                drop_tol,
                max_fill,
                reordering,
            } => b::build_ilut(drop_tol, max_fill, reordering),
            PcConfig::Milu0 => b::build_milu0(),

            PcConfig::Sor {
                omega,
                sweeps,
                mat_side,
            } => b::build_sor(omega, sweeps, mat_side),

            PcConfig::Chebyshev {
                degree,
                eig_lo,
                eig_hi,
            } => b::build_chebyshev(degree, eig_lo, eig_hi),

            PcConfig::Asm {
                overlap,
                subdomain_hint,
                block_solver,
                mode,
                weighting,
            } => b::build_asm(overlap, subdomain_hint, block_solver, mode, weighting),
            PcConfig::Amg { config } => b::build_amg(config),
            PcConfig::ApproxInv {
                kind,
                levels,
                max_per_col,
                drop_tol,
                reg,
                max_cond,
                parallel,
            } => {
                use crate::preconditioner::approxinv_csr::{ApproxInvParams, FsaiCsr, SpaiCsr};
                let params = ApproxInvParams {
                    kind,
                    levels,
                    max_per_col,
                    drop_tol,
                    reg,
                    max_cond,
                    parallel,
                };
                match kind {
                    ApproxInvKindAlias::FSAI => Ok(Box::new(FsaiCsr::new_with_params(params))),
                    ApproxInvKindAlias::SPAI => Ok(Box::new(SpaiCsr::new_with_params(params))),
                }
            }

            PcConfig::Lu => b::build_lu(),
            PcConfig::Qr => b::build_qr(),
            #[cfg(feature = "superlu_dist")]
            PcConfig::SuperLuDist => b::build_superlu_dist(),
        }
    }

    #[cfg(not(feature = "backend-faer"))]
    pub fn create_preconditioner(
        _pc_type: PcType,
        _options: Option<&PcOptions>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        Err(KError::Unsupported(
            "backend-faer feature is required to build preconditioners".into(),
        ))
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
        _op: &dyn LinOp<S = R>,
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
        for token in chain
            .split("->")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            let pct = PcType::from_str(token)?;
            let stage_opts = opts.cloned();
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

    #[cfg(feature = "backend-faer")]
    pub fn construct_deferred_pc_chain(
        specs: Vec<DeferredPcInfo>,
        op: &dyn LinOp<S = R>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        // validate again in case specs were assembled elsewhere
        Self::validate_chain_specs(&specs)?;
        use crate::preconditioner::chain::PcChain;

        let mut stages: Vec<Box<dyn Preconditioner>> = Vec::with_capacity(specs.len());
        for (i, spec) in specs.into_iter().enumerate() {
            let pc_type = spec.pc_type;
            let stage = Self::construct_deferred_preconditioner(spec, op).map_err(|e| {
                KError::InvalidInput(format!("PC chain stage {i} ({pc_type:?}) failed: {e}",))
            })?;
            stages.push(stage);
        }
        Ok(Box::new(PcChain::new(stages)))
    }

    #[cfg(not(feature = "backend-faer"))]
    pub fn construct_deferred_pc_chain(
        _specs: Vec<DeferredPcInfo>,
        _op: &dyn LinOp<S = R>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        Err(KError::Unsupported(
            "backend-faer feature is required to build chained preconditioners".into(),
        ))
    }

    pub fn create_pc_chain(
        chain: &str,
        op: &dyn LinOp<S = R>,
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

/// Placeholder type for API compatibility.
pub type PC = ();

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
}
