Here is the overall directly structure:

```bash
│   lib.rs
│   reduction.rs
│   error.rs
│
├───solver
│   │   dense_qr.rs
│   │   mod.rs
│   │   cgs.rs
│   │   bicgstab.rs
│   │   direct_lu.rs
│   │   fgmres.rs
│   │   idrs.rs
│   │   gmres.rs
│   │   superlu_dist.rs
│   │   pca_gmres.rs
│   │   api.rs
│   │   cgnr.rs
│   │   adapters.rs
│   │   tfqmr.rs
│   │   qmr.rs
│   │   dense_lu.rs
│   │   pcg.rs
│   │   minres.rs
│   │   cg.rs
│   │
│   ├───tests
│   │       block_solvers.rs
│   │       cg_side.rs
│   │       gmres_workspace_reuse.rs
│   │       mod.rs
│   │       sync_counts.rs
│   │       gmres_left_right.rs
│   │       block_arnoldi.rs
│   │       idrs.rs
│   │       gmres_right_z_basis.rs
│   │       stability.rs
│   │       gmres_variants.rs
│   │       cg_pipelined.rs
│   │       aug_recycling.rs
│   │
│   ├───common
│   │       buffer.rs
│   │       mod.rs
│   │       givens.rs
│   │
│   └───block
│           mod.rs
│           block_vec.rs
│           bicgstab.rs
│           gmres.rs
│           kernels.rs
│           arnoldi.rs
│
├───utils
│       tuning.rs
│       buffer_pool.rs
│       mod.rs
│       partition.rs
│       convergence.rs
│       profiling.rs
│       coloring.rs
│       permutation.rs
│       reduction.rs
│       metrics.rs
│       matrix_market.rs
│       reordering.rs
│       merge.rs
│       monitor.rs
│
├───parallel
│       mod.rs
│       repro.rs
│       threads.rs
│       reduce_async.rs
│       reduce.rs
│       rayon_comm.rs
│       mpi_comm.rs
│
├───algebra
│       parallel.rs
│       mod.rs
│       blas.rs
│       bridge.rs
│       scalar.rs
│       parallel_cfg.rs
│       prelude.rs
│
├───matrix
│   │   mod.rs
│   │   sparse.rs
│   │   csc.rs
│   │   dense.rs
│   │   utils.rs
│   │   format_impls.rs
│   │   csr.rs
│   │   op_bridge.rs
│   │   convert.rs
│   │   op_shell.rs
│   │   format.rs
│   │   dist_csr.rs
│   │   op.rs
│   │
│   ├───spmv
│   │       mod.rs
│   │       tests.rs
│   │       scalar.rs
│   │       plan.rs
│   │       simd_csr.rs
│   │       sellc.rs
│   │
│   ├───parcsr
│   │       mod.rs
│   │       mat.rs
│   │       halo.rs
│   │       builder.rs
│   │
│   └───dist
│           mod.rs
│           spmv_dist.rs
│           halo.rs
│
├───config
│       mod.rs
│       kinds.rs
│       options_core.rs
│       registry.rs
│       options.rs
│
├───core
│   │   mod.rs
│   │   traits.rs
│   │   wrappers.rs
│   │   block.rs
│   │
│   └───mat
│           mod.rs
│           shell.rs
│
├───ops
│       mod.rs
│       wrap.rs
│       kpc.rs
│       klinop.rs
│
├───preconditioner
│   │   ilu.rs
│   │   block_jacobi.rs
│   │   mod.rs
│   │   approxinv.rs
│   │   asm_amg.rs
│   │   approxinv_csr.rs
│   │   deflation.rs
│   │   chebyshev.rs
│   │   builders.rs
│   │   amg.rs
│   │   bridge.rs
│   │   asm.rs
│   │   jacobi.rs
│   │   ilutp.rs
│   │   sor.rs
│   │   stats.rs
│   │   ilu_options.rs
│   │   ilup.rs
│   │   ilut.rs
│   │   pc_bridge.rs
│   │   pivot.rs
│   │   chain.rs
│   │
│   ├───chain
│   │       tests.rs
│   │
│   ├───tests
│   │       legacy_bridge.rs
│   │       mod.rs
│   │       asm_amg.rs
│   │       near_nullspace.rs
│   │       deflation.rs
│   │       classical.rs
│   │       ilu_history.rs
│   │       nodal.rs
│   │       spd.rs
│   │       coarsen.rs
│   │       direct_apply.rs
│   │       ilu_csr.rs
│   │       post_interp.rs
│   │
│   ├───sor
│   │       tests_symmetric.rs
│   │
│   ├───ilu_csr
│   │       row_work.rs
│   │       mod.rs
│   │       tri_solve.rs
│   │       symbolic.rs
│   │       csr_builder.rs
│   │       pos_map.rs
│   │       pivot.rs
│   │       ilut_params.rs
│   │
│   ├───amg
│   │   │   rap_ops.rs
│   │   │   strength_nodal.rs
│   │   │   row_filter.rs
│   │   │   non_galerkin.rs
│   │   │   coarse_solver.rs
│   │   │   strength.rs
│   │   │   coarsen.rs
│   │   │   util.rs
│   │   │   prolong.rs
│   │   │
│   │   └───tests
│   │           rank_galerkin.rs
│   │           fsai_smoother.rs
│   │           chebyshev.rs
│   │           nodal_strength.rs
│   │           mixed_precision.rs
│   │           cycle_policy.rs
│   │           nodal_nns.rs
│   │
│   └───direct
│           mod.rs
│           qr_pc.rs
│           lu_pc.rs
│           superlu_dist_pc.rs
│
├───testkit
│       mod.rs
│
└───context
    │   mod.rs
    │   pc_context.rs
    │
    └───ksp_context
            mod.rs
            workspace.rs
```

We will start the review in parts, looking at one module at a time in isolation, working from the most foundational elements to the highest-level ones.

Here we will start with the context module:

`src/context/mod.rs`

```rust
//! Context module for KrylovKit linear algebra library.
//!
//! This module provides context/factory types for configuring and managing solver and preconditioner objects.
//! Contexts encapsulate algorithm selection, parameter management, and construction of solver/preconditioner pipelines.
//!
//! Modules:
//! - [`ksp_context`]: Contains the `KspContext` struct for Krylov subspace solver configuration and management.
//! - [`pc_context`]: Contains the preconditioner context types and factories.
//!
//! Usage:
//! Import the desired context type and use it to configure and instantiate solvers or preconditioners.
//!
//! # Example
//! ```rust,ignore
//! use crate::context::KspContext;
//! let ksp = KspContext::new();
//! // Configure and use the context...
//! ```
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.
//! - PETSc documentation: https://petsc.org/release/docs/manualpages/KSP/

pub mod ksp_context;
pub use ksp_context::KspContext;
pub mod pc_context;
pub use pc_context::{DeferredPcInfo, NoOpPreconditioner, PC, PcFactory, PcType, SparsityPattern};
```

`src/context/pc_context.rs`

```rust
use crate::config::options::PcOptions;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::approxinv_csr::ApproxInvKind;
use crate::preconditioner::{PcSide, Preconditioner};
use std::str::FromStr;

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
            "superludist" => Ok(PcType::SuperLuDist),
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
        drop_tol: f64,
        max_fill: usize,
        reordering: Option<String>,
    },
    Milu0,
    Sor {
        omega: f64,
        sweeps: usize,
        mat_side: crate::preconditioner::sor::MatSorType,
        symmetric: bool,
    },
    Chebyshev {
        degree: usize,
        eig_lo: f64,
        eig_hi: f64,
    },
    Asm {
        overlap: usize,
        subdomain_hint: Option<usize>,
        block_solver: Option<String>,
        mode: Option<String>,
        weighting: Option<String>,
    },
    Amg {
        levels: Option<usize>,
        smoother: Option<String>,
    },
    ApproxInv {
        kind: ApproxInvKind,
        levels: usize,
        max_per_col: usize,
        drop_tol: f64,
        reg: f64,
        max_cond: f64,
        parallel: bool,
    },
    Lu,
    Qr,
    #[cfg_attr(docsrs, doc(cfg(feature = "superlu_dist")))]
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
                use crate::preconditioner::sor::MatSorType;
                let mat_side = match o.sor_mat_side.as_deref() {
                    Some("lower") | Option::None => MatSorType::APPLY_LOWER,
                    Some("upper") => MatSorType::APPLY_UPPER,
                    Some("symmetric") => MatSorType::SYMMETRIC_SWEEP,
                    Some(s) => {
                        return Err(KError::InvalidInput(format!("unknown sor_mat_side: {s}")));
                    }
                };
                let omega = o.sor_omega.unwrap_or(1.0);
                if !(0.0..2.0).contains(&omega) {
                    return Err(KError::InvalidInput("sor_omega must be in (0,2)".into()));
                }
                PcConfig::Sor {
                    omega,
                    sweeps: o.sor_sweeps.unwrap_or(1),
                    mat_side,
                    symmetric: o.sor_symmetric.unwrap_or(false),
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
            Amg => PcConfig::Amg {
                levels: o.amg_levels,
                smoother: o.amg_smoother.clone(),
            },

            ApproxInverse => {
                // Interpret options for CSR-based SPAI/FSAI
                let kind = match o
                    .approxinv_kind
                    .as_deref()
                    .unwrap_or("fsai")
                    .to_lowercase()
                    .as_str()
                {
                    "fsai" => ApproxInvKind::FSAI,
                    "spai" => ApproxInvKind::SPAI,
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
///   - `superlu_dist`: enables [`PcType::SuperLuDist`]
///   - `legacy-pc-bridge`: enables adapters for legacy implementations (no per-apply allocs)
///
/// ## Chains
/// - String form: `"jacobi->ilut"` via [`PcFactory::create_pc_chain_from_str`]
/// - Structured form: `PcOptions.chain: Vec<PcOptions>`
/// - Construction is deferred until a matrix is available (see KSP docs).
pub struct PcFactory;

impl PcFactory {
    #[inline]
    fn is_direct(pc: PcType) -> bool {
        matches!(pc, PcType::Lu | PcType::Qr | PcType::SuperLuDist)
    }

    #[inline]
    fn chain_strict() -> bool {
        // Opt-in strict mode via env var.
        // KRYST_PC_CHAIN_STRICT=1|true enforces selected warnings as errors.
        std::env::var("KRYST_PC_CHAIN_STRICT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
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
        use crate::preconditioner::builders as b;
        match cfg {
            PcConfig::None => Ok(Box::new(NoOpPreconditioner)),
            PcConfig::Jacobi => {
                if cfg!(feature = "complex") {
                    return Err(KError::Unsupported(
                        "Jacobi preconditioner is not yet supported for complex scalars".into(),
                    ));
                }
                b::build_jacobi()
            }
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
                symmetric,
            } => b::build_sor(omega, sweeps, mat_side, symmetric),

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
            PcConfig::Amg { levels, smoother } => b::build_amg(levels, smoother),
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
                    ApproxInvKind::FSAI => Ok(Box::new(FsaiCsr::new_with_params(params))),
                    ApproxInvKind::SPAI => Ok(Box::new(SpaiCsr::new_with_params(params))),
                }
            }

            PcConfig::Lu => b::build_lu(),
            PcConfig::Qr => b::build_qr(),
            PcConfig::SuperLuDist => b::build_superlu_dist(),
        }
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
        _op: &dyn LinOp<S = f64>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        // The concrete operator format is deferred to the preconditioner itself.
        match info.pc_type {
            PcType::Amg => Err(KError::NotImplemented(
                "AMG not yet implemented".to_string(),
            )),
            PcType::Asm => Err(KError::NotImplemented(
                "ASM not yet implemented".to_string(),
            )),
            _ => Self::create_preconditioner(info.pc_type, info.options.as_ref()),
        }
    }

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

    pub fn construct_deferred_pc_chain(
        specs: Vec<DeferredPcInfo>,
        op: &dyn LinOp<S = f64>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        // validate again in case specs were assembled elsewhere
        Self::validate_chain_specs(&specs)?;
        use crate::preconditioner::chain::PcChain;

        let mut stages: Vec<Box<dyn Preconditioner>> = Vec::with_capacity(specs.len());
        for spec in specs {
            let stage = Self::construct_deferred_preconditioner(spec, op)?;
            stages.push(stage);
        }
        Ok(Box::new(PcChain::new(stages)))
    }

    pub fn create_pc_chain(
        chain: &str,
        op: &dyn LinOp<S = f64>,
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
        // flip strict mode via env var for this test
        unsafe { std::env::set_var("KRYST_PC_CHAIN_STRICT", "1") };
        let opts = crate::config::options::PcOptions::default();

        // "lu->jacobi" => direct not last
        let specs = PcFactory::create_pc_chain_from_str("lu->jacobi", Some(&opts));
        assert!(specs.is_err(), "expected validation error in strict mode");
        unsafe { std::env::remove_var("KRYST_PC_CHAIN_STRICT") };
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
```

`src/context/ksp_context/mod.rs`

```rust
//! # KSP context
//!
//! ## Operator/PC lifecycle
//! 1. [`set_operators`] stores `A` and `P` (or `A` if `P` is `None`).
//! 2. Enforces communicator equality via [`LinOp::comm()`]. Prefer
//!    [`try_set_operators`] in library code: it returns an error on mismatch, while
//!    [`set_operators`] panics for backward compatibility.
//! 3. [`setup`] resolves any deferred PC specs (including chains), then calls
//!    [`Preconditioner::setup`] followed by reuse logic:
//!    - If structure id changed → [`update_symbolic`]
//!    - Else if values id changed and numeric reuse allowed → [`update_numeric`]
//!    - Else unchanged.
//!
//! For efficient reuse across nonlinear iterations or time steps, wrap matrices in
//! [`DenseOp`](crate::matrix::op::DenseOp) or [`CsrOp`](crate::matrix::op::CsrOp) and call
//! [`mark_values_changed`](crate::matrix::op::DenseOp::mark_values_changed) or
//! [`mark_structure_changed`](crate::matrix::op::DenseOp::mark_structure_changed) after
//! in-place modifications. This ensures cache keys and reuse decisions reflect updates.
//!
//! ## Side policy
//! [`pc_side`](struct.KspContext.html#structfield.pc_side) is passed to solvers; PCs **do not** decide left vs right placement.
//!
//! ### Solver vs preconditioning side
//!
//! | Solver            | Allowed sides   | Notes                                                   |
//! |-------------------|-----------------|---------------------------------------------------------|
//! | `CG`, `PCG`       | Left only       | Requires HPD `A` and HPD left preconditioner `M`.       |
//! | `FGMRES`          | Right only      | Flexible right-only pipeline.                           |
//! | `PCA-GMRES`       | Left or Right   | Mapped to [`PcaPcMode`] during setup.                   |
//! | All other solvers | Left or Right   | `Symmetric` is normalized to `Left` before dispatch.    |
//!
//! Incompatible combinations return [`KError::InvalidInput`] during configuration.
//!
//! ## Deferred PCs / Chaining
//! [`PcFactory::create_deferred_pc`] stores type+options without a matrix.
//! [`PcFactory::construct_deferred_preconditioner`] materializes it once `P` is known.
//! [`PcChain`] composes multiple PCs: `y = P_k(...P_1(x))`.
//!
//! ## Monitors
//! Iteration monitors receive `(iter, residual)` where the residual is solver-specific
//! (preconditioned norm for Left CG/GMRES, true norm for Right GMRES). Final stats
//! always include the true residual.
//!
//! ## PREONLY behavior
//! `Preonly` is a non-iterative mode: it invokes `Preconditioner::direct_solve` on the
//! selected preconditioner using the preconditioner operator (`P`, or `A` when `P` is `None`).
//! Use it with direct PCs such as `LU`, `QR`, or `SuperLU_DIST`.

#[cfg(feature = "rayon")]
use crate::algebra::parallel::set_rayon_threads;
use crate::algebra::parallel_cfg::{parallel_tune, set_parallel_tune, set_rayon_threads_for_repro};
use crate::config::options::{CgVariant, KspOptions, KspType, PcOptions};
use crate::context::pc_context::{DeferredPcInfo, PcFactory, PcType};
use crate::error::KError;
use crate::matrix::convert::materialize_linop_with_hint;
use crate::matrix::op::{LinOp, StructureId, ValuesId, wrap_with_comm};
use crate::parallel::{Comm, set_global_reduction_mode, set_global_reduction_mode_scoped};
use crate::preconditioner::{PcReusePolicy, PcSide, Preconditioner};
use crate::reduction::ReproMode;
use crate::solver::{
    BiCgStabSolver, CgSolver, CgnrSolver, CgsSolver, FgmresSolver, GmresSolver, LinearSolver,
    MinresSolver, PCG_PIPELINED_DEFAULT_REPLACE_EVERY, PcaGmresSolver, PcaPcMode, PcgSolver,
    PcgVariant,
};
use crate::utils::convergence::{ConvergedReason, SolveStats};
use crate::utils::reduction::{ReductMode, ReductOptions};
use std::str::FromStr;
use std::sync::Arc;
mod workspace;
pub use crate::core::block::BlockVec;
pub use workspace::{GmresSStepWorkspace, GmresSpec, ReorthPolicy, Workspace};

/// Supported solver types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    Cg,
    Cgnr,
    Gmres,
    Fgmres,
    BiCgStab,
    Cgs,
    Pcg,
    Minres,
    PcaGmres,
    Qmr,
    Tfqmr,
    Preonly,
}

impl SolverType {
    /// Return the preconditioning side required by this solver, if any.
    #[inline]
    pub fn required_pc_side(self) -> Option<PcSide> {
        match self {
            SolverType::Cg | SolverType::Pcg => Some(PcSide::Left),
            _ => None,
        }
    }
}

impl FromStr for SolverType {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cg" => Ok(SolverType::Cg),
            "cgnr" => Ok(SolverType::Cgnr),
            "gmres" => Ok(SolverType::Gmres),
            "fgmres" => Ok(SolverType::Fgmres),
            "bicgstab" => Ok(SolverType::BiCgStab),
            "cgs" => Ok(SolverType::Cgs),
            "pcg" => Ok(SolverType::Pcg),
            "minres" => Ok(SolverType::Minres),
            "pca_gmres" | "pcagmres" => Ok(SolverType::PcaGmres),
            "qmr" => Ok(SolverType::Qmr),
            "tfqmr" => Ok(SolverType::Tfqmr),
            "preonly" => Ok(SolverType::Preonly),
            other => Err(KError::UnrecognizedSolverType(other.to_string())),
        }
    }
}

/// Minimal KSP context holding solver, preconditioner, and operators.
pub struct KspContext {
    solver: Option<Box<dyn LinearSolver<Error = KError> + 'static>>,
    pc: Option<Box<dyn Preconditioner>>,
    pub(crate) pending_pc: Option<DeferredPcInfo>,
    pub(crate) pending_chain: Option<Vec<DeferredPcInfo>>,
    amat: Option<Arc<dyn LinOp<S = f64>>>,
    pmat: Option<Arc<dyn LinOp<S = f64>>>,
    work: Option<Workspace>,
    setup_called: bool,
    monitors: Vec<Box<dyn Fn(usize, f64) + Send + Sync>>,
    solver_type: Option<SolverType>,
    pub rtol: f64,
    pub atol: f64,
    pub dtol: f64,
    pub maxits: usize,
    pub restart: usize,
    pub pc_side: PcSide,
    pc_side_explicit: bool,
    pc_reuse: PcReusePolicy,
    last_pc_sid: Option<StructureId>,
    last_pc_vid: Option<ValuesId>,
    reduction_opts: ReductOptions,
    reproducible: bool,
    // Pending/staged solver-specific options to apply when solver type is set
    pending_gmres: PendingGmres,
    pending_fgmres: PendingFgmres,
    pending_pcg: PendingPcg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingGmresVariant {
    Classical,
    Pipelined,
    SStep,
}

#[derive(Clone, Debug, Default)]
struct PendingGmres {
    restart: Option<usize>,
    orthog: Option<crate::solver::gmres::GmresOrthog>,
    reorth: Option<ReorthPolicy>,
    reorth_tol: Option<f64>,
    happy_breakdown: Option<bool>,
    variant: Option<PendingGmresVariant>,
    sstep: Option<usize>,
    sstep_max_cond: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct PendingFgmres {
    restart: Option<usize>,
    orthog: Option<crate::solver::fgmres::Orthog>,
    reorth: Option<ReorthPolicy>,
    reorth_tol: Option<f64>,
    happy_breakdown: Option<bool>,
    variant: Option<crate::solver::fgmres::FgmresVariant>,
}

#[derive(Clone, Debug, Default)]
struct PendingPcg {
    pipelined: Option<bool>,
    replace_every: Option<usize>,
}

impl Default for KspContext {
    fn default() -> Self {
        Self::new()
    }
}

impl KspContext {
    #[inline]
    fn normalize_side(side: PcSide) -> PcSide {
        match side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        }
    }

    fn parse_reorth_policy(label: &str) -> Result<ReorthPolicy, KError> {
        match label.to_lowercase().as_str() {
            "never" => Ok(ReorthPolicy::Never),
            "ifneeded" | "if-needed" => Ok(ReorthPolicy::IfNeeded),
            "always" => Ok(ReorthPolicy::Always),
            other => Err(KError::SolveError(format!(
                "Unrecognized reorth policy: {other} (expected 'never'|'ifneeded'|'always')"
            ))),
        }
    }

    fn parse_reduction_mode(label: &str) -> Result<ReductMode, KError> {
        match label.to_lowercase().as_str() {
            "fast" => Ok(ReductMode::Fast),
            "deterministic" | "det" => Ok(ReductMode::Deterministic),
            "deterministic-accurate" | "deterministic_accurate" | "accurate" => {
                Ok(ReductMode::DeterministicAccurate)
            }
            other => Err(KError::SolveError(format!(
                "Unrecognized ksp_reduction mode: {other} (expected 'fast'|'deterministic'|'deterministic-accurate')"
            ))),
        }
    }

    fn repro_from_mode(mode: ReductMode) -> ReproMode {
        match mode {
            ReductMode::Fast => ReproMode::Fast,
            ReductMode::Deterministic => ReproMode::Deterministic,
            ReductMode::DeterministicAccurate => ReproMode::DeterministicAccurate,
        }
    }

    fn apply_global_reduction_mode(&self) {
        set_global_reduction_mode(Self::repro_from_mode(self.reduction_opts.mode));
    }

    /// Validate that `side` is compatible with `solver_type` (if set).
    /// Mirrors `configure_pc_side()` logic but used at set-time to fail fast.
    fn check_pc_side_now(&self, side: PcSide) -> Result<(), KError> {
        let normalized = Self::normalize_side(side);
        if let Some(st) = self.solver_type {
            if let Some(required) = st.required_pc_side() {
                if normalized != required {
                    return Err(KError::InvalidInput(format!(
                        "{st:?} requires left preconditioning; got {side:?}"
                    )));
                }
            } else if matches!(st, SolverType::Fgmres) {
                if normalized != PcSide::Right {
                    return Err(KError::InvalidInput(
                        "FGMRES only supports right preconditioning".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    pub fn new() -> Self {
        Self {
            solver: None,
            pc: None,
            pending_pc: None,
            pending_chain: None,
            amat: None,
            pmat: None,
            work: None,
            setup_called: false,
            monitors: Vec::new(),
            solver_type: None,
            rtol: 1e-5,
            atol: 1e-50,
            dtol: 1e5,
            maxits: 10_000,
            restart: 30,
            pc_side: PcSide::Left,
            pc_side_explicit: false,
            pc_reuse: PcReusePolicy::Auto,
            last_pc_sid: None,
            last_pc_vid: None,
            reduction_opts: ReductOptions::default(),
            reproducible: false,
            pending_gmres: PendingGmres::default(),
            pending_fgmres: PendingFgmres::default(),
            pending_pcg: PendingPcg::default(),
        }
    }

    pub fn set_type(&mut self, solver_type: SolverType) -> Result<&mut Self, KError> {
        if let Some(required) = solver_type.required_pc_side() {
            let normalized = Self::normalize_side(self.pc_side);
            if self.pc_side_explicit {
                if normalized != required {
                    return Err(KError::InvalidInput(format!(
                        "{solver_type:?} requires left preconditioning; got {:?}",
                        self.pc_side
                    )));
                }
            } else {
                self.pc_side = required;
            }
        }

        self.solver_type = Some(solver_type);
        let solver: Option<Box<dyn LinearSolver<Error = KError> + 'static>> = match solver_type {
            SolverType::Cg => Some(Box::new(
                CgSolver::new(self.rtol, self.maxits)
                    .with_norm(crate::solver::cg::CgNormType::Preconditioned),
            )),
            SolverType::Cgnr => Some(Box::new(CgnrSolver::new(self.rtol, self.maxits))),
            SolverType::Gmres => {
                let mut s = GmresSolver::new(self.restart, self.rtol, self.maxits);
                // Apply any staged GMRES parameters
                self.apply_gmres_pending_to(&mut s);
                Some(Box::new(s))
            }
            SolverType::Fgmres => {
                let mut s = FgmresSolver::new(self.rtol, self.maxits, self.restart);
                self.apply_fgmres_pending_to(&mut s);
                Some(Box::new(s))
            }
            SolverType::BiCgStab => Some(Box::new(BiCgStabSolver::new(self.rtol, self.maxits))),
            SolverType::Cgs => Some(Box::new(CgsSolver::new(self.rtol, self.maxits))),
            SolverType::Pcg => Some(Box::new({
                let mut s = PcgSolver::new(self.rtol, self.maxits)
                    .with_norm(crate::solver::pcg::CgNormType::Preconditioned);
                self.apply_pcg_pending_to(&mut s);
                s
            })),
            SolverType::Minres => Some(Box::new(MinresSolver::new(self.rtol, self.maxits))),
            SolverType::PcaGmres => {
                let mut s = PcaGmresSolver::new(self.restart, 1, 1, self.rtol, self.maxits);
                s.pc_mode = crate::solver::PcaPcMode::Left;
                Some(Box::new(s))
            }
            SolverType::Qmr => Some(Box::new(crate::solver::QmrSolver::new(
                self.rtol,
                self.maxits,
            ))),
            SolverType::Tfqmr => Some(Box::new(crate::solver::TfqmrSolver::new(
                self.rtol,
                self.maxits,
            ))),
            SolverType::Preonly => {
                // PREONLY is intentionally "no iterative solver".
                // We’ll dispatch to `pc.direct_solve()` in `solve()`.
                None
            }
        };
        self.solver = solver;
        self.apply_global_reduction_mode();
        // Fail fast if an explicit side was set and is incompatible with the selected solver
        if self.pc_side_explicit {
            self.check_pc_side_now(self.pc_side)?
        }
        self.invalidate_setup();
        Ok(self)
    }

    pub fn set_type_from_str(&mut self, solver_type: &str) -> Result<&mut Self, KError> {
        let st = SolverType::from_str(solver_type)?;
        self.set_type(st)
    }

    pub fn set_pc_type(
        &mut self,
        pc_type: PcType,
        opts: Option<&PcOptions>,
    ) -> Result<&mut Self, KError> {
        match PcFactory::create_preconditioner(pc_type, opts) {
            Ok(pc) => {
                self.pc = Some(pc);
                self.pending_pc = None;
                self.pending_chain = None;
            }
            Err(_) => {
                let spec = PcFactory::create_deferred_pc(pc_type, opts.cloned())?;
                self.pc = None;
                self.pending_pc = Some(spec);
                self.pending_chain = None;
            }
        }
        self.invalidate_setup();
        Ok(self)
    }

    pub fn set_pc_type_from_str(&mut self, pc_type: &str) -> Result<&mut Self, KError> {
        let pct = PcType::from_str(pc_type)?;
        self.set_pc_type(pct, None)
    }

    /// Convenience for PREONLY: set solver type and a direct PC in one call.
    pub fn set_preonly_with_pc(
        &mut self,
        pc_type: PcType,
        opts: Option<&PcOptions>,
    ) -> Result<&mut Self, KError> {
        self.set_type(SolverType::Preonly)?;
        self.set_pc_type(pc_type, opts)?;
        Ok(self)
    }

    /// Set the preconditioning side directly (panics if incompatible with the active solver).
    ///
    /// Prefer `try_set_pc_side` in library code to handle errors.
    pub fn set_pc_side(&mut self, side: PcSide) -> &mut Self {
        self.try_set_pc_side(side).unwrap()
    }

    /// Set the preconditioning side, failing early if incompatible with the current solver.
    pub fn try_set_pc_side(&mut self, side: PcSide) -> Result<&mut Self, KError> {
        self.check_pc_side_now(side)?;
        self.pc_side = side;
        self.pc_side_explicit = true;
        self.invalidate_setup();
        Ok(self)
    }

    /// Set the preconditioning side from a string ("left", "right", or "symmetric").
    /// Fails fast if incompatible with the active solver.
    pub fn set_pc_side_from_str(&mut self, side: &str) -> Result<&mut Self, KError> {
        let ps = PcSide::from_str(side)?;
        self.try_set_pc_side(ps)
    }

    /// Configure the KSP context using parsed KSP options.
    pub fn set_from_options(&mut self, opts: &KspOptions) -> Result<&mut Self, KError> {
        #[cfg(feature = "rayon")]
        if let Some(n) = opts.threads {
            set_rayon_threads(n);
        }

        #[cfg(all(not(feature = "rayon"), feature = "logging"))]
        if opts.threads.is_some() {
            log::warn!("Ignoring ksp_threads: build without feature=\"rayon\"");
        }

        if opts.min_len_vec.is_some()
            || opts.min_rows_spmv.is_some()
            || opts.chunk_rows_spmv.is_some()
        {
            let mut tune = parallel_tune();
            if let Some(v) = opts.min_len_vec {
                tune.min_len_vec = v;
            }
            if let Some(v) = opts.min_rows_spmv {
                tune.min_rows_spmv = v;
            }
            if let Some(v) = opts.chunk_rows_spmv {
                tune.chunk_rows_spmv = v;
            }
            set_parallel_tune(tune);
        }

        if let Some(ref t) = opts.ksp_type {
            let st = SolverType::from_str(t)?;
            self.set_type(st)?;
        }
        if let Some(rtol) = opts.rtol {
            self.rtol = rtol;
        }
        if let Some(atol) = opts.atol {
            self.atol = atol;
        }
        if let Some(dtol) = opts.dtol {
            self.dtol = dtol;
        }
        if let Some(maxits) = opts.maxits {
            self.maxits = maxits;
        }
        if let Some(restart) = opts.restart {
            self.restart = restart;
        }
        if let Some(ref side) = opts.pc_side {
            self.set_pc_side_from_str(side)?;
        }
        if let Some(ref mode) = opts.reduction {
            let parsed = Self::parse_reduction_mode(mode)?;
            self.reduction_opts.mode = parsed;
            if let Some(ref mut w) = self.work {
                w.set_reduction_mode(parsed);
            }
            self.apply_global_reduction_mode();
        }

        if let Some(flag) = opts.reproducible {
            self.reproducible = flag;
            if flag && opts.threads.is_none() {
                set_rayon_threads_for_repro(true);
            }
        }

        let requested_cg_variant = opts.cg_variant.or_else(|| {
            opts.cg_pipelined.map(|flag| {
                if flag {
                    CgVariant::Pipelined
                } else {
                    CgVariant::Classic
                }
            })
        });

        // --- GMRES options ---
        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<GmresSolver>())
        {
            if let Some(r) = opts.effective_restart_for(KspType::GMRES) {
                s.set_restart(r);
                self.restart = r;
                self.pending_gmres.restart = Some(r);
            }
            if let Some(ref orth) = opts.gmres_orthog {
                let o = match orth.as_str() {
                    "mgs" => crate::solver::gmres::GmresOrthog::Mgs,
                    "cgs" => crate::solver::gmres::GmresOrthog::Cgs,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_gmres_orthog: {other} (expected 'mgs'|'cgs')"
                        )));
                    }
                };
                s.set_orthog(o);
                self.pending_gmres.orthog = Some(o);
            }
            if let Some(ref mode) = opts.gmres_reorth {
                let policy = Self::parse_reorth_policy(mode)?;
                s.set_reorth_policy(policy);
                self.pending_gmres.reorth = Some(policy);
            } else if let Some(flag) = opts.gmres_reorthog {
                s.set_reorthog(flag);
                self.pending_gmres.reorth = Some(if flag {
                    ReorthPolicy::Always
                } else {
                    ReorthPolicy::Never
                });
            }
            if let Some(tol) = opts.gmres_reorth_tol {
                s.set_reorth_tol(tol);
                self.pending_gmres.reorth_tol = Some(tol);
            }
            if let Some(flag) = opts.gmres_happy_breakdown {
                s.set_happy_breakdown(flag);
                self.pending_gmres.happy_breakdown = Some(flag);
            }
            if let Some(ref variant) = opts.gmres_variant {
                let pv = match variant.as_str() {
                    "classical" => PendingGmresVariant::Classical,
                    "pipelined" => PendingGmresVariant::Pipelined,
                    "sstep" => PendingGmresVariant::SStep,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_gmres_variant: {other} (expected 'classical'|'pipelined'|'sstep')"
                        )));
                    }
                };
                self.pending_gmres.variant = Some(pv);
            }
            if let Some(sstep) = opts.gmres_sstep {
                self.pending_gmres.sstep = Some(sstep);
            }
            if let Some(cond) = opts.gmres_sstep_max_cond {
                self.pending_gmres.sstep_max_cond = Some(cond);
            }
        } else {
            if let Some(r) = opts.effective_restart_for(KspType::GMRES) {
                self.pending_gmres.restart = Some(r);
                self.restart = r;
            }
            if let Some(ref orth) = opts.gmres_orthog {
                self.pending_gmres.orthog = Some(match orth.as_str() {
                    "mgs" => crate::solver::gmres::GmresOrthog::Mgs,
                    "cgs" => crate::solver::gmres::GmresOrthog::Cgs,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_gmres_orthog: {other} (expected 'mgs'|'cgs')"
                        )));
                    }
                });
            }
            if let Some(ref mode) = opts.gmres_reorth {
                self.pending_gmres.reorth = Some(Self::parse_reorth_policy(mode)?);
            } else if let Some(flag) = opts.gmres_reorthog {
                self.pending_gmres.reorth = Some(if flag {
                    ReorthPolicy::Always
                } else {
                    ReorthPolicy::Never
                });
            }
            if let Some(tol) = opts.gmres_reorth_tol {
                self.pending_gmres.reorth_tol = Some(tol);
            }
            if let Some(flag) = opts.gmres_happy_breakdown {
                self.pending_gmres.happy_breakdown = Some(flag);
            }
            if let Some(ref variant) = opts.gmres_variant {
                self.pending_gmres.variant = Some(match variant.as_str() {
                    "classical" => PendingGmresVariant::Classical,
                    "pipelined" => PendingGmresVariant::Pipelined,
                    "sstep" => PendingGmresVariant::SStep,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_gmres_variant: {other} (expected 'classical'|'pipelined'|'sstep')"
                        )));
                    }
                });
            }
            if let Some(sstep) = opts.gmres_sstep {
                self.pending_gmres.sstep = Some(sstep);
            }
            if let Some(cond) = opts.gmres_sstep_max_cond {
                self.pending_gmres.sstep_max_cond = Some(cond);
            }
        }

        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<GmresSolver>())
        {
            let snapshot = self.pending_gmres.clone();
            Self::apply_gmres_pending(&snapshot, s);
        }

        // --- FGMRES options ---
        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<FgmresSolver>())
        {
            if let Some(r) = opts.effective_restart_for(KspType::FGMRES) {
                s.set_restart(r);
                self.restart = r;
                self.pending_fgmres.restart = Some(r);
            }
            // Map "mgs"/"cgs" to Modified/Classical
            if let Some(ref orth) = opts.fgmres_orthog {
                let o = match orth.as_str() {
                    "mgs" => crate::solver::fgmres::Orthog::Modified,
                    "cgs" => crate::solver::fgmres::Orthog::Classical,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_orthog: {other} (expected 'mgs'|'cgs')"
                        )));
                    }
                };
                s.set_orthog(o);
                self.pending_fgmres.orthog = Some(o);
            }
            if let Some(ref mode) = opts.fgmres_reorth {
                let policy = Self::parse_reorth_policy(mode)?;
                s.set_reorth_policy(policy);
                self.pending_fgmres.reorth = Some(policy);
            } else if let Some(flag) = opts.fgmres_reorthog {
                s.set_reorthog(flag);
                self.pending_fgmres.reorth = Some(if flag {
                    ReorthPolicy::Always
                } else {
                    ReorthPolicy::Never
                });
            }
            if let Some(tol) = opts.fgmres_reorth_tol {
                s.set_reorth_tol(tol);
                self.pending_fgmres.reorth_tol = Some(tol);
            }
            if let Some(flag) = opts.fgmres_happy_breakdown {
                s.set_happy_breakdown(flag);
                self.pending_fgmres.happy_breakdown = Some(flag);
            }
            if let Some(ref variant) = opts.fgmres_variant {
                let v = match variant.as_str() {
                    "classical" => crate::solver::fgmres::FgmresVariant::Classical,
                    "pipelined" => crate::solver::fgmres::FgmresVariant::Pipelined,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_variant: {other} (expected 'classical'|'pipelined')"
                        )));
                    }
                };
                s.set_variant(v);
                self.pending_fgmres.variant = Some(v);
            }
        } else {
            if let Some(r) = opts.effective_restart_for(KspType::FGMRES) {
                self.pending_fgmres.restart = Some(r);
                self.restart = r;
            }
            if let Some(ref orth) = opts.fgmres_orthog {
                self.pending_fgmres.orthog = Some(match orth.as_str() {
                    "mgs" => crate::solver::fgmres::Orthog::Modified,
                    "cgs" => crate::solver::fgmres::Orthog::Classical,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_orthog: {other} (expected 'mgs'|'cgs')"
                        )));
                    }
                });
            }
            if let Some(ref mode) = opts.fgmres_reorth {
                self.pending_fgmres.reorth = Some(Self::parse_reorth_policy(mode)?);
            } else if let Some(flag) = opts.fgmres_reorthog {
                self.pending_fgmres.reorth = Some(if flag {
                    ReorthPolicy::Always
                } else {
                    ReorthPolicy::Never
                });
            }
            if let Some(tol) = opts.fgmres_reorth_tol {
                self.pending_fgmres.reorth_tol = Some(tol);
            }
            if let Some(flag) = opts.fgmres_happy_breakdown {
                self.pending_fgmres.happy_breakdown = Some(flag);
            }
            if let Some(ref variant) = opts.fgmres_variant {
                self.pending_fgmres.variant = Some(match variant.as_str() {
                    "classical" => crate::solver::fgmres::FgmresVariant::Classical,
                    "pipelined" => crate::solver::fgmres::FgmresVariant::Pipelined,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_fgmres_variant: {other} (expected 'classical'|'pipelined')"
                        )));
                    }
                });
            }
        }

        if let Some(s) = self
            .solver
            .as_mut()
            .and_then(|b| b.as_any_mut().downcast_mut::<CgSolver>())
        {
            if let Some(variant) = requested_cg_variant {
                s.set_variant(variant);
            }
            if let Some(ref norm) = opts.cg_norm {
                let n = match norm.as_str() {
                    "precond" => crate::solver::cg::CgNormType::Preconditioned,
                    "unprecond" => crate::solver::cg::CgNormType::Unpreconditioned,
                    "natural" => crate::solver::cg::CgNormType::Natural,
                    "none" => crate::solver::cg::CgNormType::None,
                    other => {
                        return Err(KError::SolveError(format!(
                            "Unrecognized ksp_cg_norm: {other}"
                        )));
                    }
                };
                s.set_norm(n);
            }
            if let Some(r) = opts.trust_region {
                s.set_trust_region(r);
            }
            if let Some(flag) = opts.cg_use_async {
                s.set_async_enabled(flag);
            }
            if let Some(min_n) = opts.cg_async_min_n {
                s.set_async_min_n(min_n);
            }
        }
        let mut pcg_pending_updated = false;
        if let Some(variant) = requested_cg_variant {
            if matches!(variant, CgVariant::Pipelined)
                && Self::normalize_side(self.pc_side) != PcSide::Left
            {
                return Err(KError::InvalidInput(
                    "Pipelined PCG requires left preconditioning".into(),
                ));
            }
            self.pending_pcg.pipelined = Some(matches!(variant, CgVariant::Pipelined));
            pcg_pending_updated = true;
        }
        if let Some(repl) = opts.cg_replace_every {
            self.pending_pcg.replace_every = Some(repl);
            pcg_pending_updated = true;
        }
        if pcg_pending_updated {
            let snapshot = self.pending_pcg.clone();
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|b| b.as_any_mut().downcast_mut::<PcgSolver>())
            {
                Self::apply_pcg_pending(&snapshot, s);
            }
        }
        self.invalidate_setup();
        Ok(self)
    }

    fn apply_gmres_pending(pending: &PendingGmres, s: &mut GmresSolver) {
        if let Some(r) = pending.restart {
            s.set_restart(r);
        }
        if let Some(o) = pending.orthog {
            s.set_orthog(o);
        }
        if let Some(p) = pending.reorth {
            s.set_reorth_policy(p);
        }
        if let Some(tol) = pending.reorth_tol {
            s.set_reorth_tol(tol);
        }
        if let Some(f) = pending.happy_breakdown {
            s.set_happy_breakdown(f);
        }
        let mut variant_kind = pending.variant;
        if variant_kind.is_none()
            && matches!(s.variant, crate::solver::gmres::GmresVariant::SStep { .. })
            && (pending.sstep.is_some() || pending.sstep_max_cond.is_some())
        {
            variant_kind = Some(PendingGmresVariant::SStep);
        }

        if let Some(kind) = variant_kind {
            match kind {
                PendingGmresVariant::Classical => {
                    s.set_variant(crate::solver::gmres::GmresVariant::Classical);
                }
                PendingGmresVariant::Pipelined => {
                    s.set_variant(crate::solver::gmres::GmresVariant::Pipelined);
                }
                PendingGmresVariant::SStep => {
                    let current = match s.variant {
                        crate::solver::gmres::GmresVariant::SStep {
                            s,
                            reorth,
                            max_cond,
                        } => Some((s, reorth, max_cond)),
                        _ => None,
                    };
                    let block_s = pending
                        .sstep
                        .or_else(|| current.map(|(s, _, _)| s))
                        .unwrap_or(2);
                    let max_cond = pending
                        .sstep_max_cond
                        .or_else(|| current.map(|(_, _, cond)| cond))
                        .unwrap_or(1e8);
                    let reorth = pending
                        .reorth
                        .or_else(|| current.map(|(_, r, _)| r))
                        .unwrap_or_else(|| s.reorth_policy());
                    s.set_variant(crate::solver::gmres::GmresVariant::SStep {
                        s: block_s,
                        reorth,
                        max_cond,
                    });
                }
            }
        }
    }

    fn apply_gmres_pending_to(&self, s: &mut GmresSolver) {
        Self::apply_gmres_pending(&self.pending_gmres, s);
    }

    fn apply_fgmres_pending_to(&self, s: &mut FgmresSolver) {
        if let Some(r) = self.pending_fgmres.restart {
            s.set_restart(r);
        }
        if let Some(o) = self.pending_fgmres.orthog {
            s.set_orthog(o);
        }
        if let Some(p) = self.pending_fgmres.reorth {
            s.set_reorth_policy(p);
        }
        if let Some(tol) = self.pending_fgmres.reorth_tol {
            s.set_reorth_tol(tol);
        }
        if let Some(f) = self.pending_fgmres.happy_breakdown {
            s.set_happy_breakdown(f);
        }
        if let Some(v) = self.pending_fgmres.variant {
            s.set_variant(v);
        }
    }

    fn apply_pcg_pending(pending: &PendingPcg, s: &mut PcgSolver) {
        if let Some(flag) = pending.pipelined {
            if flag {
                let replace_every = pending
                    .replace_every
                    .unwrap_or(PCG_PIPELINED_DEFAULT_REPLACE_EVERY);
                s.set_variant(PcgVariant::Pipelined { replace_every });
            } else {
                s.set_variant(PcgVariant::Classic);
            }
        } else if matches!(s.variant(), PcgVariant::Pipelined { .. })
            && let Some(replace_every) = pending.replace_every
        {
            s.set_variant(PcgVariant::Pipelined { replace_every });
        }
    }

    fn apply_pcg_pending_to(&self, s: &mut PcgSolver) {
        Self::apply_pcg_pending(&self.pending_pcg, s);
    }

    /// Configure both KSP and PC from their respective option sets.
    pub fn set_from_all_options(
        &mut self,
        ksp_opts: &KspOptions,
        pc_opts: &PcOptions,
    ) -> Result<&mut Self, KError> {
        self.set_from_options(ksp_opts)?;
        if let Some(ref pct) = pc_opts.pc_type {
            let pct = PcType::from_str(pct)?;
            self.set_pc_type(pct, Some(pc_opts))?;
        }
        if let Some(ref pol) = pc_opts.reuse_policy {
            let pol = match pol.as_str() {
                "never" => PcReusePolicy::Never,
                "reuse_numeric" => PcReusePolicy::ReuseNumeric,
                _ => PcReusePolicy::Auto,
            };
            self.set_pc_reuse_policy(pol);
        }
        if let Some(ref side) = ksp_opts.pc_side {
            self.set_pc_side_from_str(side)?;
        }
        if let Some(ref chain_opts) = pc_opts.chain {
            let specs = PcFactory::create_deferred_pc_chain_from_options(chain_opts)?;
            self.pc = None;
            self.pending_pc = None;
            self.pending_chain = Some(specs);
            self.invalidate_setup();
        }
        Ok(self)
    }

    /// Assign the system and preconditioner operators.
    ///
    /// Returns an error if the communicators of `A` and `P` differ.
    /// `LinOp::comm()` is the single source of truth for parallel context;
    /// mismatches indicate a caller bug.
    ///
    /// On success, invalidates any prior setup (PC reuse and workspace).
    pub fn try_set_operators(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
    ) -> Result<&mut Self, KError> {
        let pmat = pmat.unwrap_or_else(|| amat.clone());
        let ac = amat.comm();
        let pc = pmat.comm();
        if ac != pc {
            self.invalidate_setup();
            return Err(KError::InvalidInput(format!(
                "Amat/Pmat communicator mismatch: A={}, P={}",
                ac.id(),
                pc.id()
            )));
        }
        self.amat = Some(amat);
        self.pmat = Some(pmat);
        self.invalidate_setup();
        Ok(self)
    }

    /// Like `try_set_operators`, but first wraps operators with an explicit communicator.
    pub fn try_set_operators_with_comm(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
        comm: crate::parallel::UniverseComm,
    ) -> Result<&mut Self, KError> {
        let a_wrapped = wrap_with_comm(amat, comm.clone());
        let p_wrapped = pmat.map(|p| wrap_with_comm(p, comm.clone()));
        self.try_set_operators(a_wrapped, p_wrapped)
    }

    /// Assign the system and preconditioner operators.
    ///
    /// Panics if the communicators of `A` and `P` differ. Prefer
    /// [`KspContext::try_set_operators`] in libraries to handle errors.
    pub fn set_operators(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
    ) -> &mut Self {
        self.try_set_operators(amat, pmat).unwrap()
    }

    /// Like `set_operators`, but first wraps operators with an explicit communicator.
    /// Panics on communicator mismatch. Prefer
    /// [`KspContext::try_set_operators_with_comm`].
    pub fn set_operators_with_comm(
        &mut self,
        amat: Arc<dyn LinOp<S = f64>>,
        pmat: Option<Arc<dyn LinOp<S = f64>>>,
        comm: crate::parallel::UniverseComm,
    ) -> &mut Self {
        self.try_set_operators_with_comm(amat, pmat, comm).unwrap()
    }

    pub fn set_pc_reuse_policy(&mut self, policy: PcReusePolicy) -> &mut Self {
        self.pc_reuse = policy;
        self
    }

    fn reset_pc_ids(&mut self) {
        self.last_pc_sid = None;
        self.last_pc_vid = None;
    }

    pub fn last_pc_sid(&self) -> Option<StructureId> {
        self.last_pc_sid
    }
    pub fn last_pc_vid(&self) -> Option<ValuesId> {
        self.last_pc_vid
    }

    /// Prepare preconditioner and workspace.
    pub fn setup(&mut self) -> Result<(), KError> {
        let pmat = self
            .pmat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Pmat not set".into()))?;
        let amat = self
            .amat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;

        if self.pc.is_none() {
            if let Some(specs) = self.pending_chain.take() {
                let chain = PcFactory::construct_deferred_pc_chain(specs, pmat.as_ref())?;
                self.pc = Some(chain);
            } else if let Some(spec) = self.pending_pc.take() {
                let pc = PcFactory::construct_deferred_preconditioner(spec, pmat.as_ref())?;
                self.pc = Some(pc);
            }
        }

        let sid = {
            let id = pmat.structure_id();
            if id.0 != 0 {
                id
            } else {
                StructureId(Arc::as_ptr(pmat) as *const () as usize as u64)
            }
        };
        let vid = pmat.values_id();

        if self.pc.is_none() {
            // no factory hook here; assume pc set elsewhere
            self.last_pc_sid = None;
            self.last_pc_vid = None;
        }

        if let Some(pc) = self.pc.as_mut() {
            // Pre-convert once to the PC's requested format, preserving communicator.
            let hint = pc.required_format();
            let tol = pc.preferred_drop_tol_for_format().unwrap_or(0.0);
            let pmat_view = materialize_linop_with_hint(pmat.as_ref(), hint, tol)?;

            match self.last_pc_sid {
                None => {
                    pc.setup(pmat_view.as_ref())?;
                    self.last_pc_sid = Some(sid);
                    self.last_pc_vid = Some(vid);
                }
                Some(old_sid) if old_sid != sid => {
                    pc.update_symbolic(pmat_view.as_ref())?;
                    self.last_pc_sid = Some(sid);
                    self.last_pc_vid = Some(vid);
                }
                Some(_old_sid) => {
                    let vid_known = vid.0 != 0;
                    let values_changed = self.last_pc_vid != Some(vid);
                    match self.pc_reuse {
                        PcReusePolicy::Never => {
                            if !vid_known || values_changed {
                                pc.update_symbolic(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            }
                        }
                        PcReusePolicy::ReuseNumeric => {
                            if pc.supports_numeric_update() {
                                if !vid_known {
                                    log::debug!(
                                        "ValuesId unknown; conservatively refreshing numeric data. Wrap your matrix in DenseOp/CsrOp and call mark_values_changed() to enable exact reuse."
                                    );
                                }
                                pc.update_numeric(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            } else if !vid_known || values_changed {
                                pc.update_symbolic(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            }
                        }
                        PcReusePolicy::Auto => {
                            if (!vid_known || values_changed)
                                && pc.supports_numeric_update()
                                && self.pc_reuse.allow_numeric()
                            {
                                if !vid_known {
                                    log::debug!(
                                        "ValuesId unknown; conservatively refreshing numeric data. Wrap your matrix in DenseOp/CsrOp and call mark_values_changed() to enable exact reuse."
                                    );
                                }
                                pc.update_numeric(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            } else if values_changed {
                                pc.update_symbolic(pmat_view.as_ref())?;
                                self.last_pc_vid = Some(vid);
                            }
                        }
                    }
                }
            }
        }

        let (m, _) = amat.dims();
        if self
            .work
            .as_ref()
            .map(|w| w.tmp1.len() != m)
            .unwrap_or(true)
        {
            self.work = Some(Workspace::new(m));
            if let Some(ref mut w) = self.work {
                w.set_reduction_options(self.reduction_opts.clone());
            }
            if let Some(ref mut solver) = self.solver
                && let Some(ref mut w) = self.work
            {
                solver.setup_workspace(w);
            }
        }
        self.setup_called = true;
        Ok(())
    }

    /// Solve the linear system using stored operators.
    pub fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<SolveStats<f64>, KError> {
        if !self.setup_called {
            self.setup()?;
        }
        if matches!(self.solver_type, Some(SolverType::Preonly)) {
            let pmat = self
                .pmat
                .as_ref()
                .ok_or_else(|| KError::InvalidInput("Pmat not set".into()))?;
            let pc = self.pc.as_mut().ok_or_else(|| {
                KError::SolveError("PREONLY requires a direct PC (LU/QR/SuperLU_DIST)".into())
            })?;
            if !pc.supports_numeric_update() {
                // Not a reliable indicator of directness, but provides a hint if a user
                // accidentally selects a non-direct PC like Jacobi/ILU.
                log::debug!(
                    "PREONLY: selected PC may not be a direct solver; expecting LU/QR/SuperLU_DIST."
                );
            }
            pc.direct_solve(pmat.as_ref(), b, x)?;
            return Ok(SolveStats::new(1, 0.0, ConvergedReason::ConvergedAtol));
        }

        // Ensure the configured reduction mode is active while solving and configure
        // solver preconditioning side, validating compatibility along the way.
        let _reduction_mode_guard =
            set_global_reduction_mode_scoped(Self::repro_from_mode(self.reduction_opts.mode));
        self.configure_pc_side()?;

        let amat = self
            .amat
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("Amat not set".into()))?;

        let monitors = if self.monitors.is_empty() {
            None
        } else {
            Some(self.monitors.as_slice())
        };
        let comm = amat.comm();
        comm.set_reproducible(self.reproducible);
        let pc = self
            .pc
            .as_mut()
            .map(|b| b.as_mut() as &mut dyn Preconditioner);
        let solver = self
            .solver
            .as_mut()
            .ok_or_else(|| KError::SolveError("No solver".into()))?;
        let mut stats = solver.solve(
            amat.as_ref(),
            pc,
            b,
            x,
            self.pc_side,
            &comm,
            monitors,
            self.work.as_mut(),
        )?;

        // Compute true residual r = b - A x and use its norm for reporting
        let mut residual = vec![0.0f64; b.len()];
        if let Err(e) = amat.try_matvec(x, &mut residual) {
            return Err(KError::SolveError(format!("residual matvec failed: {e}")));
        }
        for (ri, &bi) in residual.iter_mut().zip(b.iter()) {
            *ri = bi - *ri;
        }
        let res_sq = comm.dot(&residual, &residual);
        stats.final_residual = res_sq.sqrt();
        Ok(stats)
    }

    fn invalidate_setup(&mut self) {
        self.setup_called = false;
        self.reset_pc_ids();
    }

    /// Add an iteration monitor callback.
    pub fn add_monitor<F>(&mut self, f: F)
    where
        F: Fn(usize, f64) + Send + Sync + 'static,
    {
        self.monitors.push(Box::new(f));
    }

    /// Return the number of registered monitors.
    pub fn num_monitors(&self) -> usize {
        self.monitors.len()
    }

    /// Clear all registered monitors.
    pub fn clear_monitors(&mut self) {
        self.monitors.clear();
    }

    #[cfg(test)]
    pub fn set_preconditioner(&mut self, pc: Box<dyn Preconditioner>) {
        self.pc = Some(pc);
    }

    /// Invoke all monitors with the provided iteration and residual.
    pub fn invoke_monitors(&self, iter: usize, residual: f64) {
        for m in &self.monitors {
            m(iter, residual);
        }
    }

    /// Set solver tolerances and maximum iterations.
    pub fn set_tolerances(&mut self, rtol: f64, atol: f64, dtol: f64, maxits: usize) -> &mut Self {
        self.rtol = rtol;
        self.atol = atol;
        self.dtol = dtol;
        self.maxits = maxits;
        self.invalidate_setup();
        self
    }

    /// Configure the underlying solver based on the requested preconditioning side.
    fn configure_pc_side(&mut self) -> Result<(), KError> {
        // Treat symmetric as left; only specialized PCs interpret it differently.
        let side = match self.pc_side {
            PcSide::Symmetric => PcSide::Left,
            s => s,
        };

        if let Some(SolverType::PcaGmres) = self.solver_type {
            if let Some(s) = self
                .solver
                .as_mut()
                .and_then(|s| s.as_any_mut().downcast_mut::<PcaGmresSolver>())
            {
                s.pc_mode = match side {
                    PcSide::Left => PcaPcMode::Left,
                    PcSide::Right => PcaPcMode::Right,
                    PcSide::Symmetric => unreachable!(),
                };
            }
        }

        if let Some(st) = self.solver_type {
            if let Some(required) = st.required_pc_side() {
                if side != required {
                    return Err(KError::InvalidInput(format!(
                        "{st:?} requires left preconditioning; got {side:?}"
                    )));
                }
            } else if matches!(st, SolverType::Fgmres) {
                if side != PcSide::Right {
                    return Err(KError::InvalidInput(
                        "FGMRES only supports right preconditioning".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Query whether setup has been performed.
    pub fn is_setup(&self) -> bool {
        self.setup_called
    }

    /// Set the GMRES restart parameter.
    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart;
        self.invalidate_setup();
    }
}

impl KspContext {
    /// Test-only: view current workspace (e.g., to inspect GMRES V/Z basis sizes).
    pub fn debug_workspace(&self) -> Option<&Workspace> {
        self.work.as_ref()
    }

    /// Test-only: inject a preconditioner for controlled testing.
    pub fn set_pc_box_for_tests(&mut self, pc: Box<dyn Preconditioner>) {
        self.pc = Some(pc);
    }
}
```

`src/context/ksp_context/workspace.rs`

```rust
use crate::algebra::bridge::BridgeScratch;
#[allow(unused_imports)]
use crate::algebra::prelude::*;
use crate::core::block::BlockVec;
use crate::solver::common::givens::{apply_new_givens_and_update_g, apply_prev_givens_to_col};
use crate::solver::gmres::AugmentationPolicy;

#[derive(Debug, Clone, Default)]
pub struct Workspace {
    pub tmp1: Vec<S>,
    pub tmp2: Vec<S>,
    // Legacy buffers for solvers not yet migrated
    pub q_s: Vec<Vec<S>>,
    pub z_s: Vec<Vec<S>>,
    pub h_s: Vec<Vec<S>>,
    pub q: Vec<Vec<S>>,
    pub z: Vec<Vec<S>>,
    pub h: Vec<Vec<S>>,
    pub v_mem: Vec<S>,
    pub z_mem: Vec<S>,
    // Column-major Hessenberg storage for GMRES/FGMRES
    pub h_mem: Vec<S>,
    pub cs: Vec<R>,
    pub sn: Vec<S>,
    pub g: Vec<S>,
    pub blk_scratch: Vec<S>,
    pub blk_payload: Vec<R>,
    pub bridge: BridgeScratch,
    pub bridge_tmp: Vec<S>,
    pub block_buf: Option<BlockVec>,
    pub tsqr: Option<TsqrWorkspace>,
    pub pipelined_w: Vec<S>,
    pub pipelined_wtmp: Vec<S>,
    pub pipelined_payload: Vec<S>,
    pub gmres_sstep: Option<GmresSStepWorkspace>,
    pub gmres_recycle: RecyclingSpace,
    pub reduction: crate::utils::reduction::ReductOptions,
    // Shared communication arenas
    pub send_arena: crate::utils::buffer_pool::BufferPool<u8>,
    pub recv_arena: crate::utils::buffer_pool::BufferPool<u8>,
    pub packet_arena: crate::utils::buffer_pool::BufferPool<u8>,
    n: usize,
    m: usize,
    need_z: bool,
}

#[derive(Debug, Clone)]
pub struct RecyclingSpace {
    u: Vec<S>,
    au: Vec<S>,
    n: usize,
    rmax: usize,
    cols: usize,
    policy: AugmentationPolicy,
}

impl Default for RecyclingSpace {
    fn default() -> Self {
        Self {
            u: Vec::new(),
            au: Vec::new(),
            n: 0,
            rmax: 0,
            cols: 0,
            policy: AugmentationPolicy::None,
        }
    }
}

impl RecyclingSpace {
    pub fn configure(&mut self, n: usize, rmax: usize, policy: AugmentationPolicy) {
        if self.n != n || self.rmax != rmax {
            self.u.resize(n.saturating_mul(rmax), S::zero());
            self.au.resize(n.saturating_mul(rmax), S::zero());
            self.n = n;
            self.rmax = rmax;
            self.cols = 0;
        }
        self.policy = policy;
    }

    #[inline]
    pub fn policy(&self) -> AugmentationPolicy {
        self.policy.clone()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.rmax
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn clear(&mut self) {
        self.cols = 0;
    }

    pub fn col(&self, j: usize) -> &[S] {
        let n = self.n;
        &self.u[j * n..(j + 1) * n]
    }

    pub fn col_mut(&mut self, j: usize) -> &mut [S] {
        let n = self.n;
        &mut self.u[j * n..(j + 1) * n]
    }

    pub fn a_col(&self, j: usize) -> &[S] {
        let n = self.n;
        &self.au[j * n..(j + 1) * n]
    }

    pub fn a_col_mut(&mut self, j: usize) -> &mut [S] {
        let n = self.n;
        &mut self.au[j * n..(j + 1) * n]
    }

    pub fn push_from(&mut self, u: &[S], au: &[S]) {
        if self.cols >= self.rmax {
            return;
        }
        let n = self.n;
        let dst_u = &mut self.u[self.cols * n..(self.cols + 1) * n];
        let dst_au = &mut self.au[self.cols * n..(self.cols + 1) * n];
        dst_u.copy_from_slice(u);
        dst_au.copy_from_slice(au);
        self.cols += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReorthPolicy {
    Never,
    #[default]
    IfNeeded,
    Always,
}

/// Specification for sizing GMRES/FGMRES workspaces.
#[derive(Debug, Clone, Copy)]
pub struct GmresSpec {
    pub n: usize,
    pub m: usize,
    pub need_z: bool,
    pub block_s: usize,
}

#[derive(Debug, Clone)]
pub struct GmresSStepWorkspace {
    pub w: BlockVec,
    pub q: BlockVec,
    pub aq: BlockVec,
    pub gram: Vec<S>,
    pub c_prev: Vec<R>,
    pub payload: Vec<S>,
    pub r: Vec<R>,
}

impl GmresSStepWorkspace {
    pub fn new(n: usize, s: usize, m: usize) -> Self {
        let mut ws = Self {
            w: BlockVec::new(n, s),
            q: BlockVec::new(n, s),
            aq: BlockVec::new(n, s),
            gram: vec![S::zero(); s.saturating_mul(s)],
            c_prev: vec![R::default(); m.saturating_mul(s)],
            payload: vec![S::zero(); s.saturating_mul(s + 1) / 2 + m.saturating_mul(s)],
            r: vec![R::default(); s.saturating_mul(s)],
        };
        ws.ensure(n, s, m);
        ws
    }

    pub fn ensure(&mut self, n: usize, s: usize, m: usize) {
        self.w.resize(n, s);
        self.q.resize(n, s);
        self.aq.resize(n, s);
        ensure_len(&mut self.gram, s.saturating_mul(s));
        ensure_len(&mut self.c_prev, m.saturating_mul(s));
        let payload_len = s.saturating_mul(s + 1) / 2 + m.saturating_mul(s);
        ensure_len(&mut self.payload, payload_len);
        ensure_len(&mut self.r, s.saturating_mul(s));
    }
}

/// Scratch buffers for TSQR factorizations.
#[derive(Debug, Clone)]
pub struct TsqrWorkspace {
    pub taus: Vec<S>,
    pub rmat: Vec<S>,
    pub w_max: usize,
}

impl TsqrWorkspace {
    pub fn with_width(w_max: usize) -> Self {
        Self {
            taus: vec![S::zero(); w_max],
            rmat: vec![S::zero(); w_max.saturating_mul(w_max)],
            w_max,
        }
    }
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        let mut ws = Self::default();
        ws.tmp1.resize(n, S::zero());
        ws.tmp2.resize(n, S::zero());
        ws.n = n;
        ws
    }

    /// Ensure communication buffers have enough bytes for upcoming operations.
    pub fn ensure_comm_bytes(&mut self, max_send: usize, max_recv: usize) {
        self.send_arena.ensure_len(max_send);
        self.recv_arena.ensure_len(max_recv);
    }

    /// Ensure the reusable block vector has capacity `n x p`.
    pub fn ensure_block(&mut self, n: usize, p: usize) {
        if p == 0 {
            self.block_buf = None;
            return;
        }
        let replace = match self.block_buf {
            Some(ref buf) if buf.nrows() == n && buf.ncols() >= p => false,
            _ => true,
        };
        if replace {
            self.block_buf = Some(BlockVec::new(n, p));
        }
    }

    /// Ensure the TSQR workspace supports panels up to width `w_max`.
    pub fn ensure_tsqr(&mut self, w_max: usize) {
        if w_max == 0 {
            self.tsqr = None;
            return;
        }
        let replace = match self.tsqr {
            Some(ref tsqr) if tsqr.w_max >= w_max => false,
            _ => true,
        };
        if replace {
            self.tsqr = Some(TsqrWorkspace::with_width(w_max));
        }
    }

    pub fn ensure_sstep(&mut self, n: usize, s: usize, m: usize) {
        if s == 0 {
            self.gmres_sstep = None;
            return;
        }
        let need_new = match self.gmres_sstep {
            Some(ref buf) => {
                buf.w.nrows() != n || buf.w.ncols() < s || buf.c_prev.len() < m.saturating_mul(s)
            }
            None => true,
        };
        if need_new {
            self.gmres_sstep = Some(GmresSStepWorkspace::new(n, s, m));
        } else if let Some(ref mut buf) = self.gmres_sstep {
            buf.ensure(n, s, m);
        }
    }

    #[inline]
    pub fn sstep_mut(&mut self) -> Option<&mut GmresSStepWorkspace> {
        self.gmres_sstep.as_mut()
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }
    #[inline]
    pub fn has_z(&self) -> bool {
        self.need_z
    }

    #[inline]
    pub fn ld_h(&self) -> usize {
        self.m + 1
    }

    /// Ensure capacity for a (F)GMRES run. Idempotent and allocation-friendly.
    pub fn acquire_gmres(&mut self, spec: GmresSpec) {
        // Remember shape for indexers
        self.n = spec.n;
        self.m = spec.m;
        self.need_z = spec.need_z;

        let n = spec.n;
        let m = spec.m;

        let v_len = (m + 1).checked_mul(n).expect("v_len overflow");
        let z_len = if spec.need_z {
            m.checked_mul(n).expect("z_len overflow")
        } else {
            0
        };
        let h_len = (m + 1).checked_mul(m).expect("h_len overflow");
        let g_len = m + 1;

        ensure_len(&mut self.tmp1, n);
        ensure_len(&mut self.tmp2, n);
        ensure_len(&mut self.v_mem, v_len);
        if spec.need_z {
            ensure_len(&mut self.z_mem, z_len);
        } else {
            self.z_mem.clear();
            self.z_mem.shrink_to_fit();
        }
        ensure_len(&mut self.h_mem, h_len);
        ensure_len(&mut self.cs, m);
        ensure_len(&mut self.sn, m);
        ensure_len(&mut self.g, g_len);
        ensure_len(&mut self.pipelined_w, n);
        ensure_len(&mut self.pipelined_wtmp, n);
        ensure_len(&mut self.pipelined_payload, m + 2);

        if spec.block_s > 0 {
            ensure_len(&mut self.blk_scratch, n * spec.block_s);
            let payload_cap = block_payload_capacity(spec.m.saturating_add(1), spec.block_s);
            ensure_capacity(&mut self.blk_payload, payload_cap);
        } else {
            self.blk_scratch.clear();
            self.blk_payload.clear();
        }

        self.ensure_sstep(n, spec.block_s, m);
    }

    pub fn set_reduction_options(&mut self, opt: crate::utils::reduction::ReductOptions) {
        self.reduction = opt;
    }

    pub fn set_reduction_mode(&mut self, mode: crate::utils::reduction::ReductMode) {
        self.reduction.mode = mode;
    }

    pub fn reduction_options(&self) -> &crate::utils::reduction::ReductOptions {
        &self.reduction
    }

    #[inline]
    pub fn v_col(&mut self, j: usize) -> &mut [S] {
        debug_assert!(j <= self.m);
        let n = self.n;
        let off = j.checked_mul(n).expect("v offset overflow");
        &mut self.v_mem[off..off + n]
    }

    #[inline]
    pub fn z_col(&mut self, j: usize) -> &mut [S] {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let off = j.checked_mul(n).expect("z offset overflow");
        &mut self.z_mem[off..off + n]
    }

    #[inline]
    pub fn h_at(&self, i: usize, j: usize) -> S {
        debug_assert!(i <= self.m && j < self.m);
        self.h_mem[j * (self.m + 1) + i]
    }
    #[inline]
    pub fn h_at_mut(&mut self, i: usize, j: usize) -> &mut S {
        debug_assert!(i <= self.m && j < self.m);
        let idx = j * (self.m + 1) + i;
        &mut self.h_mem[idx]
    }

    pub fn v_cols2(&mut self, a: usize, b: usize) -> (&mut [S], &mut [S]) {
        debug_assert!(a <= self.m && b <= self.m && a != b);
        let n = self.n;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let lo_off = lo * n;
        let hi_off = hi * n;
        let (lo_part, rest) = self.v_mem.split_at_mut(hi_off);
        let (_, lo_slice) = lo_part.split_at_mut(lo_off);
        let (hi_slice, _) = rest.split_at_mut(n);
        if a < b {
            (&mut lo_slice[..n], hi_slice)
        } else {
            (hi_slice, &mut lo_slice[..n])
        }
    }

    pub fn z_cols2(&mut self, a: usize, b: usize) -> (&mut [S], &mut [S]) {
        debug_assert!(self.need_z && a < self.m && b < self.m && a != b);
        let n = self.n;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let lo_off = lo * n;
        let hi_off = hi * n;
        let (lo_part, rest) = self.z_mem.split_at_mut(hi_off);
        let (_, lo_slice) = lo_part.split_at_mut(lo_off);
        let (hi_slice, _) = rest.split_at_mut(n);
        if a < b {
            (&mut lo_slice[..n], hi_slice)
        } else {
            (hi_slice, &mut lo_slice[..n])
        }
    }

    // --- Composite view helpers -------------------------------------------------
    #[inline]
    pub fn v_and_z_mut(&mut self, j: usize) -> (&[S], &mut [S]) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let off = j * n;
        let vj: &[S] = &self.v_mem[off..off + n];
        let zj: &mut [S] = &mut self.z_mem[off..off + n];
        (vj, zj)
    }

    #[inline]
    pub fn tmp1_and_z_mut(&mut self, j: usize) -> (&[S], &mut [S]) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let tmp: &[S] = &self.tmp1[..n];
        let z: &mut [S] = &mut self.z_mem[j * n..(j + 1) * n];
        (tmp, z)
    }

    #[inline]
    pub fn tmp2_and_z_mut(&mut self, j: usize) -> (&[S], &mut [S]) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let tmp: &[S] = &self.tmp2[..n];
        let z: &mut [S] = &mut self.z_mem[j * n..(j + 1) * n];
        (tmp, z)
    }

    #[inline]
    pub fn z_and_tmp2_mut(&mut self, j: usize) -> (&[S], &mut [S]) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let z: &[S] = &self.z_mem[j * n..(j + 1) * n];
        let tmp: &mut [S] = &mut self.tmp2[..n];
        (z, tmp)
    }

    // --- Copy helpers -----------------------------------------------------------
    #[inline]
    pub fn copy_tmp2_into_vcol(&mut self, j: usize) {
        let n = self.n;
        let dst = &mut self.v_mem[j * n..(j + 1) * n];
        let src = &self.tmp2[..n];
        dst.copy_from_slice(src);
    }

    #[inline]
    pub fn copy_tmp1_into_vcol(&mut self, j: usize) {
        let n = self.n;
        let dst = &mut self.v_mem[j * n..(j + 1) * n];
        let src = &self.tmp1[..n];
        dst.copy_from_slice(src);
    }

    #[inline]
    pub fn copy_vcol_into_zcol(&mut self, j: usize) {
        debug_assert!(self.need_z && j < self.m);
        let n = self.n;
        let src = &self.v_mem[j * n..(j + 1) * n];
        let dst = &mut self.z_mem[j * n..(j + 1) * n];
        dst.copy_from_slice(src);
    }

    #[inline]
    pub fn copy_vcol_into_tmp1(&mut self, j: usize) {
        let n = self.n;
        let src = &self.v_mem[j * n..(j + 1) * n];
        self.tmp1[..n].copy_from_slice(src);
    }

    // --- Hessenberg helpers -----------------------------------------------------
    #[inline]
    pub fn apply_prev_givens_to_col(&mut self, j: usize, upto: usize) {
        use smallvec::SmallVec;

        if upto == 0 {
            return;
        }

        let ld = self.ld_h();
        let base = j * ld;
        let mut hcol: SmallVec<[S; 64]> = SmallVec::with_capacity(upto + 1);
        for row in 0..=upto {
            hcol.push(self.h_mem[base + row]);
        }

        apply_prev_givens_to_col(&mut hcol, upto, &self.cs, &self.sn);

        for (row, val) in hcol.into_iter().enumerate() {
            self.h_mem[base + row] = val;
        }
    }

    #[inline]
    pub fn apply_final_givens_and_update_g(&mut self, j: usize) {
        use smallvec::SmallVec;

        let ld = self.ld_h();
        let base = j * ld;
        let mut hcol: SmallVec<[S; 64]> = SmallVec::with_capacity(j + 2);
        for row in 0..=j + 1 {
            hcol.push(self.h_mem[base + row]);
        }

        apply_new_givens_and_update_g(
            &mut hcol,
            j,
            &mut self.cs[..],
            &mut self.sn[..],
            &mut self.g[..],
        );

        for (row, val) in hcol.into_iter().enumerate() {
            self.h_mem[base + row] = val;
        }
    }

    #[cfg(not(feature = "complex"))]
    pub fn pipelined_arnoldi_step(
        &mut self,
        k: usize,
        n: usize,
        comm: &crate::parallel::UniverseComm,
        policy: ReorthPolicy,
        tol: f64,
    ) -> Result<usize, crate::error::KError> {
        debug_assert!(k < self.m);

        let w = &self.pipelined_w[..n];
        let payload_len = k + 2;
        let send = {
            let payload = &mut self.pipelined_payload[..payload_len];
            for i in 0..=k {
                let vi = &self.v_mem[i * n..(i + 1) * n];
                payload[i] = vi.iter().zip(w).map(|(a, b)| a * b).sum();
            }
            payload[k + 1] = w.iter().map(|val| val * val).sum();
            payload.to_vec()
        };
        let opt = self.reduction.clone();
        let (handle, _) = <crate::parallel::UniverseComm as crate::utils::reduction::AllreduceOps>::
            allreduce_n_async(comm, send, &opt)?;
        let glob =
            <crate::parallel::UniverseComm as crate::utils::reduction::AllreduceOps>::wait_vec(
                handle,
            );

        let mut reductions = 1usize;

        self.pipelined_wtmp[..n].copy_from_slice(w);

        let mut sum_h2 = R::zero();
        for i in 0..=k {
            let hij = glob[i];
            sum_h2 += hij * hij;
            let vi = &self.v_mem[i * n..(i + 1) * n];
            for idx in 0..n {
                self.pipelined_wtmp[idx] -= hij * vi[idx];
            }
            *self.h_at_mut(i, k) = hij;
        }

        let total_norm_sq = glob[k + 1];
        let mut hnext_sq = (total_norm_sq - sum_h2).max(R::zero());
        if !hnext_sq.is_finite() {
            hnext_sq = R::zero();
        }

        let tol = tol.max(0.0);
        let tol_sq = tol * tol;
        let trigger_reorth = match policy {
            ReorthPolicy::Never => false,
            ReorthPolicy::Always => true,
            ReorthPolicy::IfNeeded => total_norm_sq > 0.0 && hnext_sq < tol_sq * total_norm_sq,
        };

        if trigger_reorth {
            reductions += 1;

            let send = {
                let payload = &mut self.pipelined_payload[..payload_len];
                for i in 0..=k {
                    let vi = &self.v_mem[i * n..(i + 1) * n];
                    payload[i] = vi
                        .iter()
                        .zip(&self.pipelined_wtmp[..n])
                        .map(|(a, b)| a * b)
                        .sum();
                }
                payload[k + 1] = self.pipelined_wtmp[..n].iter().map(|val| val * val).sum();
                payload.to_vec()
            };
            let (handle, _) =
                <crate::parallel::UniverseComm as crate::utils::reduction::AllreduceOps>::
                    allreduce_n_async(comm, send, &opt)?;
            let corr =
                <crate::parallel::UniverseComm as crate::utils::reduction::AllreduceOps>::wait_vec(
                    handle,
                );

            let mut delta_norm_sq = R::zero();
            for i in 0..=k {
                let delta = corr[i];
                delta_norm_sq += delta * delta;
                let vi = &self.v_mem[i * n..(i + 1) * n];
                for idx in 0..n {
                    self.pipelined_wtmp[idx] -= delta * vi[idx];
                }
                let hij = *self.h_at_mut(i, k) + delta;
                *self.h_at_mut(i, k) = hij;
            }

            sum_h2 = R::zero();
            for i in 0..=k {
                let hij = *self.h_at_mut(i, k);
                sum_h2 += hij * hij;
            }

            let wtmp_norm_sq = corr[k + 1];
            hnext_sq = (wtmp_norm_sq - delta_norm_sq).max(R::zero());
            if !hnext_sq.is_finite() {
                hnext_sq = R::zero();
            }
        }

        let hnext = hnext_sq.sqrt();
        *self.h_at_mut(k + 1, k) = hnext;

        let base = (k + 1) * n;
        if hnext > R::zero() {
            let inv = S::from_real(hnext.recip());
            for idx in 0..n {
                self.v_mem[base + idx] = self.pipelined_wtmp[idx] * inv;
            }
        } else {
            for idx in 0..n {
                self.v_mem[base + idx] = S::zero();
            }
        }

        Ok(reductions)
    }

    #[cfg(feature = "complex")]
    pub fn pipelined_arnoldi_step(
        &mut self,
        k: usize,
        n: usize,
        _comm: &crate::parallel::UniverseComm,
        _policy: ReorthPolicy,
        _tol: f64,
    ) -> Result<usize, crate::error::KError> {
        let _ = (k, n);
        Err(crate::error::KError::NotImplemented(
            "pipelined GMRES is not yet implemented for complex scalars".into(),
        ))
    }
}

/// Grow vector to `need` length without zeroing. Never shrinks silently.
#[inline]
fn ensure_len<T: Copy>(v: &mut Vec<T>, need: usize) {
    if v.len() != need {
        if v.capacity() < need {
            v.reserve_exact(need - v.capacity());
        }
        unsafe {
            v.set_len(need);
        }
    }
}

#[inline]
fn ensure_capacity<T>(v: &mut Vec<T>, need: usize) {
    if v.capacity() < need {
        v.reserve_exact(need - v.capacity());
    }
}

#[inline]
fn block_payload_capacity(max_blocks: usize, block_size: usize) -> usize {
    let scalars = max_blocks
        .checked_mul(block_size)
        .and_then(|v| v.checked_mul(block_size))
        .unwrap_or(usize::MAX);
    #[cfg(feature = "complex")]
    {
        scalars.checked_mul(2).unwrap_or(usize::MAX)
    }
    #[cfg(not(feature = "complex"))]
    {
        scalars
    }
}
```

We specifically want to focus on (1) overall code correctness, scalability, and robustness; and (2) completeness of the complex, mpi, and rayon implementations.

Now we want to focus on the algebra module. Here is the source code. In addition to reviewing the code for completeness, robustness, MPI/rayon capabilities, and complex support, we also want to scope out removal of `faer` as a dependency for the whole crate, so that means feature gating the `faer` code specifically, and likely introducing new feature gates for other linear algebra crates (i.e., `nalgebra`, `rulinalg`, `alga`, `sprs`, `cgmath`, etc.) numerical implementations.

`src/algebra/mod.rs`

```rust
//! Basic numeric traits and operations used throughout the crate.

pub mod blas;
pub mod bridge;
pub mod parallel;
pub mod parallel_cfg;
pub mod prelude;
pub mod scalar;

pub use scalar::{KrystScalar, R, S};
```

`src/algebra/blas.rs`

```rust
#[allow(unused_imports)]
use crate::algebra::prelude::*;

#[inline]
pub fn dot_conj(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = S::zero();
    for i in 0..x.len() {
        acc = x[i].conj().mul_add(y[i], acc);
    }
    acc
}

#[inline]
pub fn nrm2(x: &[S]) -> R {
    dot_conj(x, x).abs().sqrt()
}
```

`src/algebra/bridge.rs`

```rust
use crate::algebra::prelude::*;

/// Temporary buffers reused by solver bridges when converting between `S` and `f64`.
#[derive(Default, Clone, Debug)]
pub struct BridgeScratch {
    buf: Vec<f64>,
}

impl BridgeScratch {
    /// Create an empty scratch buffer.
    #[inline]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    #[inline]
    fn ensure(&mut self, want: usize) {
        if self.buf.len() < want {
            self.buf.resize(want, 0.0);
        }
    }

    /// Loan two disjoint real buffers of length `n` at once.
    #[inline]
    pub fn with_pair<F, Rv>(&mut self, n: usize, f: F) -> Rv
    where
        F: FnOnce(&mut [f64], &mut [f64]) -> Rv,
    {
        self.ensure(2 * n);
        let (xr, rest) = self.buf.split_at_mut(n);
        let (yr, _) = rest.split_at_mut(n);
        f(xr, yr)
    }

    /// Loan a single temporary buffer of length `n`.
    #[inline]
    pub fn with_one<F, Rv>(&mut self, n: usize, f: F) -> Rv
    where
        F: FnOnce(&mut [f64]) -> Rv,
    {
        self.ensure(n);
        f(&mut self.buf[..n])
    }
}

#[inline]
pub fn copy_scalar_to_real_in<T: KrystScalar<Real = f64>>(x: &[T], xr: &mut [f64]) {
    debug_assert_eq!(x.len(), xr.len());
    for (dst, &src) in xr.iter_mut().zip(x.iter()) {
        *dst = src.real();
    }
}

#[inline]
pub fn copy_real_into_scalar<T: KrystScalar<Real = f64>>(yr: &[f64], y: &mut [T]) {
    debug_assert_eq!(yr.len(), y.len());
    for (dst, &src) in y.iter_mut().zip(yr.iter()) {
        *dst = T::from_real(src);
    }
}
```

`src/algebra/parallel_cfg.rs`

```rust
use once_cell::sync::OnceCell;
use std::sync::RwLock;

#[derive(Clone, Copy, Debug)]
pub struct ParallelTune {
    /// Minimum vector length to enable Rayon in elementwise kernels.
    pub min_len_vec: usize,
    /// Minimum rows to enable Rayon in CSR SpMV.
    pub min_rows_spmv: usize,
    /// Target chunk size in rows for CSR SpMV (approx).
    pub chunk_rows_spmv: usize,
}

impl Default for ParallelTune {
    fn default() -> Self {
        Self {
            min_len_vec: 8192,
            min_rows_spmv: 2048,
            chunk_rows_spmv: 512,
        }
    }
}

static PAR_TUNE: OnceCell<RwLock<ParallelTune>> = OnceCell::new();

fn cell() -> &'static RwLock<ParallelTune> {
    PAR_TUNE.get_or_init(|| RwLock::new(ParallelTune::default()))
}

pub fn set_parallel_tune(t: ParallelTune) {
    if let Ok(mut guard) = cell().write() {
        *guard = t;
    }
}

pub fn parallel_tune() -> ParallelTune {
    cell()
        .read()
        .map(|g| *g)
        .unwrap_or_else(|_| ParallelTune::default())
}

/// Configure Rayon for reproducible runs by constraining the global pool.
pub fn set_rayon_threads_for_repro(enable: bool) {
    #[cfg(feature = "rayon")]
    {
        if enable {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build_global();
        }
    }
    let _ = enable;
}
```

`src/algebra/parallel.rs`

```rust
//! Thread-parallel vector kernels with scalar fallback.
//!
//! - Works with or without `feature="rayon"`.
//! - Uses stable chunking (configurable) to keep reductions numerically steady.
//! - Provides scalar fallbacks for small problems or builds without Rayon.
//!
//! The kernels assume crate-level aliases/traits brought in via
//! [`crate::algebra::prelude`].

#![allow(clippy::needless_borrow)]

use crate::algebra::prelude::*;

#[cfg(feature = "rayon")]
use rayon::ThreadPoolBuilder;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

const VEC_CHUNK: usize = 1 << 14;
const REPRO_CHUNK: usize = 1 << 14;

/// Configure the global Rayon thread pool used by Kryst's parallel kernels.
///
/// This is a thin wrapper around [`rayon::ThreadPoolBuilder::build_global`]; it is
/// safe to call multiple times, but only the first successful invocation takes
/// effect. Subsequent calls are ignored once the global pool has been initialised.
#[cfg(feature = "rayon")]
pub fn set_rayon_threads(n: usize) {
    let _ = ThreadPoolBuilder::new().num_threads(n).build_global();
}

// -------------------- scalar fallbacks --------------------

#[inline]
fn s_copy(src: &[S], dst: &mut [S]) {
    debug_assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

#[inline]
fn s_fill_zero(dst: &mut [S]) {
    for value in dst {
        *value = S::zero();
    }
}

#[inline]
fn s_scale(alpha: S, y: &mut [S]) {
    if alpha == S::from_real(1.0) {
        return;
    }
    if alpha == S::zero() {
        s_fill_zero(y);
        return;
    }
    for yi in y {
        *yi = alpha * *yi;
    }
}

#[inline]
fn s_axpy(x: &[S], alpha: S, y: &mut [S]) {
    debug_assert_eq!(x.len(), y.len());
    if alpha == S::zero() {
        return;
    }
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = *yi + alpha * xi;
    }
}

#[inline]
fn s_axpby(x: &[S], alpha: S, y: &mut [S], beta: S) {
    debug_assert_eq!(x.len(), y.len());
    if beta == S::zero() {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = alpha * xi;
        }
    } else if beta == S::from_real(1.0) {
        s_axpy(x, alpha, y);
    } else {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = alpha * xi + beta * *yi;
        }
    }
}

#[inline]
fn s_dot_conj_local(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = S::zero();
    const BLK: usize = 1 << 14;
    let mut i = 0;
    while i < x.len() {
        let end = (i + BLK).min(x.len());
        let mut blk = S::zero();
        for j in i..end {
            blk = blk + x[j].conj() * y[j];
        }
        acc = acc + blk;
        i = end;
    }
    acc
}

#[inline]
fn s_sum_abs2_local(x: &[S]) -> R {
    let mut acc = R::default();
    const BLK: usize = 1 << 14;
    let mut i = 0;
    while i < x.len() {
        let end = (i + BLK).min(x.len());
        let mut blk = R::default();
        for j in i..end {
            let a = x[j].abs();
            blk = blk + a * a;
        }
        acc = acc + blk;
        i = end;
    }
    acc
}

// -------------------- public API (dual-path) --------------------

#[inline]
pub fn par_copy(src: &[S], dst: &mut [S]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(feature = "rayon")]
    {
        let n = src.len();
        let min_len = crate::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            src.par_chunks(chunk)
                .zip(dst.par_chunks_mut(chunk))
                .for_each(|(s, d)| d.copy_from_slice(s));
            return;
        }
    }
    s_copy(src, dst);
}

#[inline]
pub fn par_fill_zero(dst: &mut [S]) {
    #[cfg(feature = "rayon")]
    {
        let n = dst.len();
        let min_len = crate::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            dst.par_chunks_mut(chunk)
                .for_each(|chunk| s_fill_zero(chunk));
            return;
        }
    }
    s_fill_zero(dst);
}

#[inline]
pub fn par_scale(alpha: S, y: &mut [S]) {
    #[cfg(feature = "rayon")]
    {
        let n = y.len();
        let min_len = crate::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            if alpha == S::from_real(1.0) {
                return;
            }
            if alpha == S::zero() {
                par_fill_zero(y);
                return;
            }
            y.par_chunks_mut(chunk).for_each(|yc| {
                for yi in yc {
                    *yi = alpha * *yi;
                }
            });
            return;
        }
    }
    s_scale(alpha, y);
}

#[inline]
pub fn par_axpy(x: &[S], alpha: S, y: &mut [S]) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = crate::parallel_cfg::parallel_tune().min_len_vec;
        if n >= min_len {
            if alpha == S::zero() {
                return;
            }
            y.par_iter_mut()
                .zip(x.par_iter().copied())
                .for_each(|(yi, xi)| {
                    *yi = *yi + alpha * xi;
                });
            return;
        }
    }
    s_axpy(x, alpha, y);
}

#[inline]
pub fn par_axpby(x: &[S], alpha: S, y: &mut [S], beta: S) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = crate::parallel_cfg::parallel_tune().min_len_vec;
        if n >= min_len {
            if beta == S::zero() {
                y.par_iter_mut()
                    .zip(x.par_iter().copied())
                    .for_each(|(yi, xi)| {
                        *yi = alpha * xi;
                    });
            } else if beta == S::from_real(1.0) {
                par_axpy(x, alpha, y);
            } else {
                y.par_iter_mut()
                    .zip(x.par_iter().copied())
                    .for_each(|(yi, xi)| {
                        *yi = alpha * xi + beta * *yi;
                    });
            }
            return;
        }
    }
    s_axpby(x, alpha, y, beta);
}

/// Compute `y = x + alpha * y`.
#[inline]
pub fn par_xpay(x: &[S], alpha: S, y: &mut [S]) {
    par_axpby(x, S::one(), y, alpha);
}

#[inline]
pub fn par_dot_conj_local(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = crate::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            return x
                .par_chunks(chunk)
                .zip(y.par_chunks(chunk))
                .map(|(xc, yc)| {
                    let mut acc = S::zero();
                    for (&xi, &yi) in xc.iter().zip(yc) {
                        acc = acc + xi.conj() * yi;
                    }
                    acc
                })
                .reduce(S::zero, |a, b| a + b);
        }
    }
    s_dot_conj_local(x, y)
}

#[inline]
pub fn par_sum_abs2_local(x: &[S]) -> R {
    #[cfg(feature = "rayon")]
    {
        let n = x.len();
        let min_len = crate::parallel_cfg::parallel_tune().min_len_vec;
        let chunk = VEC_CHUNK;
        if n >= min_len {
            return x
                .par_chunks(chunk)
                .map(|xc| {
                    let mut ssq = R::default();
                    for &value in xc {
                        let a = value.abs();
                        ssq = ssq + a * a;
                    }
                    ssq
                })
                .reduce(R::default, |a, b| a + b);
        }
    }
    s_sum_abs2_local(x)
}

/// Deterministic conjugated dot product using fixed chunking.
pub fn dot_conj_local_repro(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return S::zero();
    }

    let nchunks = (x.len() + REPRO_CHUNK - 1) / REPRO_CHUNK;
    let mut parts = vec![S::zero(); nchunks];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        parts.par_iter_mut().enumerate().for_each(|(cid, slot)| {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = S::zero();
            for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                acc = acc + xi.conj() * yi;
            }
            *slot = acc;
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for cid in 0..nchunks {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = S::zero();
            for (&xi, &yi) in x[start..end].iter().zip(&y[start..end]) {
                acc = acc + xi.conj() * yi;
            }
            parts[cid] = acc;
        }
    }

    let mut total = S::zero();
    for part in parts {
        total = total + part;
    }
    total
}

/// Deterministic sum of squared magnitudes using fixed chunking.
pub fn sum_abs2_local_repro(x: &[S]) -> R {
    if x.is_empty() {
        return R::zero();
    }

    let nchunks = (x.len() + REPRO_CHUNK - 1) / REPRO_CHUNK;
    let mut parts = vec![R::zero(); nchunks];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        parts.par_iter_mut().enumerate().for_each(|(cid, slot)| {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = R::zero();
            for &value in &x[start..end] {
                let a = value.abs();
                acc = acc + a * a;
            }
            *slot = acc;
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for cid in 0..nchunks {
            let start = cid * REPRO_CHUNK;
            let end = ((cid + 1) * REPRO_CHUNK).min(x.len());
            let mut acc = R::zero();
            for &value in &x[start..end] {
                let a = value.abs();
                acc = acc + a * a;
            }
            parts[cid] = acc;
        }
    }

    let mut total = R::zero();
    for part in parts {
        total = total + part;
    }
    total
}
```

`src/algebra/prelude.rs`

```rust
//! Bring scalar aliases and `KrystScalar` into scope in one shot.
//! Usage in modules: `use crate::algebra::prelude::*;`

pub use super::scalar::{KrystScalar, R, S};
```

`src/algebra/scalar.rs`

```rust
#![allow(clippy::excessive_precision)]

use core::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "complex")]
use num_complex::Complex64;

/// Scalar abstraction used internally by kryst.  The goal is to
/// keep the public API monomorphic (f64), while the internals use `S`.
pub trait KrystScalar:
    Copy
    + Clone
    + Send
    + Sync
    + 'static
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The corresponding real type (always `f64` for now).
    type Real: Copy
        + Clone
        + Send
        + Sync
        + 'static
        + Default
        + PartialEq
        + PartialOrd
        + Add<Output = Self::Real>
        + Sub<Output = Self::Real>
        + Mul<Output = Self::Real>
        + Div<Output = Self::Real>;

    // Constructors / constants
    fn zero() -> Self;
    fn one() -> Self;

    /// Convert from a real (`f64`) to this scalar type.
    fn from_real(x: Self::Real) -> Self;

    /// Extract the real part (identity for real, `.re` for complex).
    fn real(self) -> Self::Real;

    /// Extract the imaginary part (zero for real scalars).
    fn imag(self) -> Self::Real;

    /// Construct a scalar from its real and imaginary components.
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;

    // Basic ops we need everywhere
    fn abs(self) -> Self::Real; // |z| for complex, |x| for real
    fn conj(self) -> Self; // identity for real
    fn inv(self) -> Self; // 1/self (caller ensures nonzero)
    fn is_finite(self) -> bool;

    /// Fused multiply-add.  For `f64` we use HW FMA; for complex we fall back.
    fn mul_add(self, a: Self, b: Self) -> Self;
}

// ==================== Implementations ====================

impl KrystScalar for f64 {
    type Real = f64;

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn from_real(x: Self::Real) -> Self {
        x
    }

    #[inline]
    fn real(self) -> Self::Real {
        self
    }

    #[inline]
    fn imag(self) -> Self::Real {
        0.0
    }

    #[inline]
    fn from_parts(re: Self::Real, _im: Self::Real) -> Self {
        re
    }

    #[inline]
    fn abs(self) -> Self::Real {
        f64::abs(self)
    }

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn inv(self) -> Self {
        1.0 / self
    }

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
}

#[cfg(feature = "complex")]
impl KrystScalar for Complex64 {
    type Real = f64;

    #[inline]
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }

    #[inline]
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    #[inline]
    fn from_real(x: Self::Real) -> Self {
        Complex64::new(x, 0.0)
    }

    #[inline]
    fn real(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn imag(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn from_parts(re: Self::Real, im: Self::Real) -> Self {
        Complex64::new(re, im)
    }

    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }

    #[inline]
    fn conj(self) -> Self {
        Complex64::new(self.re, -self.im)
    }

    #[inline]
    fn inv(self) -> Self {
        let n2 = self.re * self.re + self.im * self.im;
        Complex64::new(self.re / n2, -self.im / n2)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
}

// ==================== Feature-gated scalar choice ====================

#[cfg(feature = "complex")]
pub type S = Complex64;
#[cfg(not(feature = "complex"))]
pub type S = f64;

/// Real partner of `S` (currently always `f64`)
pub type R = <S as KrystScalar>::Real;

#[cfg(feature = "complex")]
#[inline]
pub fn copy_scalar_to_real_in(z: &[S], out: &mut [f64]) {
    debug_assert_eq!(z.len(), out.len());
    for (dst, &src) in out.iter_mut().zip(z.iter()) {
        *dst = src.real();
    }
}

#[cfg(feature = "complex")]
#[inline]
pub fn copy_real_to_scalar_in(x: &[f64], out: &mut [S]) {
    debug_assert_eq!(x.len(), out.len());
    for (dst, &src) in out.iter_mut().zip(x.iter()) {
        *dst = S::from_real(src);
    }
}

#[cfg(not(feature = "complex"))]
#[inline]
pub fn copy_scalar_to_real_in(z: &[S], out: &mut [f64]) {
    debug_assert_eq!(z.len(), out.len());
    if core::ptr::eq(z.as_ptr() as *const f64, out.as_ptr()) {
        return;
    }
    // SAFETY: when the complex feature is disabled we have S == f64.
    let z_as_f64: &[f64] = unsafe { &*(z as *const [S] as *const [f64]) };
    out.copy_from_slice(z_as_f64);
}

#[cfg(not(feature = "complex"))]
#[inline]
pub fn copy_real_to_scalar_in(x: &[f64], out: &mut [S]) {
    debug_assert_eq!(x.len(), out.len());
    if core::ptr::eq(x.as_ptr(), out.as_ptr() as *const f64) {
        return;
    }
    // SAFETY: when the complex feature is disabled we have S == f64.
    let out_as_f64: &mut [f64] = unsafe { &mut *(out as *mut [S] as *mut [f64]) };
    out_as_f64.copy_from_slice(x);
}
```

Next, provide a review of the core module:

`src/core/mod.rs`

```rust
pub mod block;
pub mod mat;
pub mod traits;
pub mod wrappers;

```

`src/core/block.rs`

```rust
use crate::algebra::prelude::*;
use crate::error::KError;

/// Column-major dense block vector storage used by block Krylov variants.
#[derive(Debug, Clone, Default)]
pub struct BlockVec {
    data: Vec<S>,
    n: usize,
    p: usize,
}

impl BlockVec {
    /// Create a new block vector with `n` rows and `p` columns.
    pub fn new(n: usize, p: usize) -> Self {
        Self {
            data: vec![S::zero(); n.saturating_mul(p)],
            n,
            p,
        }
    }

    /// Resize the block vector to `n` rows and `p` columns, zero-filling new entries.
    pub fn resize(&mut self, n: usize, p: usize) {
        if self.n != n || self.p != p {
            self.data.resize(n.saturating_mul(p), S::zero());
            self.n = n;
            self.p = p;
        } else {
            let needed = n.saturating_mul(p);
            if self.data.len() != needed {
                self.data.resize(needed, S::zero());
            }
        }
    }

    /// Number of rows in the block vector.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.n
    }

    /// Number of columns in the block vector.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.p
    }

    /// Immutable view into the `j`-th column.
    #[inline]
    pub fn col(&self, j: usize) -> &[S] {
        let offset = j * self.n;
        &self.data[offset..offset + self.n]
    }

    /// Mutable view into the `j`-th column.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [S] {
        let offset = j * self.n;
        &mut self.data[offset..offset + self.n]
    }

    /// Immutable view into the raw column-major storage.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }

    /// Mutable view into the raw column-major storage.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.data
    }
}

impl BlockVec {
    /// Fill the block vector with zeros.
    pub fn fill_zero(&mut self) {
        for v in &mut self.data {
            *v = S::zero();
        }
    }
}

/// Convenience helper for verifying block dimensions.
#[allow(dead_code)]
pub(crate) fn assert_block_dims(expected_rows: usize, vec: &BlockVec) -> Result<(), KError> {
    if vec.nrows() != expected_rows {
        return Err(KError::InvalidInput(format!(
            "BlockVec has {} rows but expected {}",
            vec.nrows(),
            expected_rows
        )));
    }
    Ok(())
}
```

`src/core/traits.rs`

```rust
//! Core linear-algebra traits for kryst.

/// Matrix–vector product: y ← A x.
pub trait MatVec<V> {
    /// Compute y = A · x.
    fn matvec(&self, x: &V, y: &mut V);
}

/// Matrix–transpose–vector product: y ← Aᵗ x.
pub trait MatTransVec<V> {
    /// Compute y = Aᵗ · x.
    fn mattransvec(&self, x: &V, y: &mut V);
}

// Blanket implementations of MatVec/MatTransVec for LinOp types using Vec storage.
use crate::algebra::scalar::{copy_real_to_scalar_in, copy_scalar_to_real_in};
use crate::core::block::BlockVec;
use crate::error::KError;
use crate::matrix::op::LinOp;
use faer::traits::ComplexField;

impl<T, L> MatVec<Vec<T>> for L
where
    L: LinOp<S = T> + ?Sized,
    T: ComplexField,
{
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        LinOp::matvec(self, &x[..], &mut y[..]);
    }
}

impl<T, L> MatTransVec<Vec<T>> for L
where
    L: LinOp<S = T> + ?Sized,
    T: ComplexField,
{
    fn mattransvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        if !LinOp::supports_transpose(self) {
            panic!("t_matvec not supported");
        }
        LinOp::t_matvec(self, &x[..], &mut y[..]);
    }
}

/// Optional extension trait for block matvec operations while remaining matrix-free.
pub trait BlockOp {
    /// Apply the operator to multiple columns at once. Default implementation calls
    /// [`apply`](Self::apply) per column to remain format agnostic.
    fn apply_many(&self, x: &BlockVec, y: &mut BlockVec) -> Result<(), KError> {
        if x.ncols() != y.ncols() {
            return Err(KError::InvalidInput(format!(
                "apply_many column mismatch: {} vs {}",
                x.ncols(),
                y.ncols()
            )));
        }
        let mut x_real = vec![0.0; x.nrows()];
        let mut y_real = vec![0.0; y.nrows()];
        for c in 0..x.ncols() {
            copy_scalar_to_real_in(x.col(c), &mut x_real);
            copy_scalar_to_real_in(y.col(c), &mut y_real);
            self.apply(&x_real, &mut y_real)?;
            copy_real_to_scalar_in(&y_real, y.col_mut(c));
        }
        Ok(())
    }

    /// Apply the operator to a single column.
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError>;

    /// Apply the transpose of the operator if available.
    fn apply_t(&self, _x: &[f64], _y: &mut [f64]) -> Result<(), KError> {
        Err(KError::Unsupported("transpose not available"))
    }
}

impl<T> BlockOp for T
where
    T: LinOp<S = f64> + ?Sized,
{
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        LinOp::try_matvec(self, x, y)
    }

    fn apply_t(&self, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if !LinOp::supports_transpose(self) {
            return Err(KError::Unsupported(
                "LinOp::t_matvec called but transpose not supported",
            ));
        }
        LinOp::t_matvec(self, x, y);
        Ok(())
    }
}

/// Inner products & norms.
pub trait InnerProduct<V> {
    /// Associated scalar type.
    type Scalar: Copy + PartialOrd + From<f64> + Into<f64>;
    /// Compute dot(x, y) with communicator support for parallel reductions.
    fn dot(&self, x: &V, y: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar;
    /// Compute ‖x‖₂ with communicator support for parallel reductions.
    fn norm(&self, x: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar {
        let local_sq = self.dot(x, x, comm);
        let global_sq = comm.all_reduce_f64(local_sq.into());
        (global_sq.sqrt()).into()
    }
}

/// Uniform indexing into vectors (dense or sparse).
pub trait Indexing {
    /// Number of rows (or length for a vector).
    fn nrows(&self) -> usize;
}

/// Matrix shape trait: provides nrows/ncols for matrices and vectors.
pub trait MatShape {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Trait for extracting the sparsity pattern of a matrix row.
pub trait RowPattern {
    /// Returns the column indices of nonzeros in row i.
    fn row_indices(&self, i: usize) -> &[usize];
}

/// Trait for extracting elements from a matrix.
pub trait MatrixGet<T> {
    /// Get the element at position (i, j).
    fn get(&self, i: usize, j: usize) -> T;
}

/// Trait for extracting a submatrix by index set (for block/ASM preconditioners).
pub trait SubmatrixExtract {
    /// Returns the submatrix with rows and columns given by `indices`.
    fn submatrix(&self, indices: &[usize]) -> Self;
}

/// Sparse-aware matrix-vector operations for AMG and iterative solvers
pub trait MatVecOp<T> {
    /// Compute y = alpha * A * x + beta * y
    fn mat_vec(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) -> Result<(), crate::error::KError>;

    /// Compute y = alpha * A^T * x + beta * y (transpose operation)
    fn mat_vec_trans(
        &self,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
    ) -> Result<(), crate::error::KError>;

    /// Get the number of rows
    fn nrows(&self) -> usize;

    /// Get the number of columns  
    fn ncols(&self) -> usize;
}

/// Sparse-aware dot product operations
pub trait DotOp<T> {
    /// Compute the dot product x^T * y
    fn dot(&self, x: &[T], y: &[T]) -> T;

    /// Compute the 2-norm of a vector
    fn norm2(&self, x: &[T]) -> T;
}

/// Implementation for dense matrices (faer Mat)
impl MatVecOp<f64> for faer::Mat<f64> {
    fn mat_vec(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".to_string(),
            ));
        }

        for i in 0..self.nrows() {
            let mut sum = 0.0;
            for j in 0..self.ncols() {
                sum += self[(i, j)] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
        Ok(())
    }

    fn mat_vec_trans(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), crate::error::KError> {
        if x.len() != self.nrows() || y.len() != self.ncols() {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".to_string(),
            ));
        }

        for j in 0..self.ncols() {
            let mut sum = 0.0;
            for i in 0..self.nrows() {
                sum += self[(i, j)] * x[i];
            }
            y[j] = alpha * sum + beta * y[j];
        }
        Ok(())
    }

    fn nrows(&self) -> usize {
        faer::Mat::nrows(self)
    }
    fn ncols(&self) -> usize {
        faer::Mat::ncols(self)
    }
}

/// Implementation for sparse matrices (CsrMatrix)
impl MatVecOp<f64> for crate::matrix::sparse::CsrMatrix<f64> {
    fn mat_vec(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), crate::error::KError> {
        use crate::matrix::sparse::SparseMatrix;
        // Dimension checks
        if x.len() != SparseMatrix::ncols(self) || y.len() != SparseMatrix::nrows(self) {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".to_string(),
            ));
        }

        // Quick exits for alpha/beta
        if alpha.abs() <= f64::EPSILON {
            if beta.abs() <= f64::EPSILON {
                for v in y.iter_mut() {
                    *v = 0.0;
                }
            } else if (beta - 1.0).abs() > f64::EPSILON {
                for v in y.iter_mut() {
                    *v *= beta;
                }
            }
            return Ok(());
        }

        // Canonical CSR access (no allocations)
        let rp = self.row_ptr();
        let cj = self.col_idx();
        let vv = self.values();

        #[cfg(debug_assertions)]
        {
            // Basic CSR integrity checks
            assert_eq!(rp.len(), self.nrows() + 1, "row_ptr length must be nrows+1");
            assert!(
                rp.windows(2).all(|w| w[0] <= w[1]),
                "row_ptr must be non-decreasing"
            );
            let nnz = *rp.last().unwrap();
            assert_eq!(cj.len(), nnz, "col_idx length must equal nnz");
            assert_eq!(vv.len(), nnz, "values length must equal nnz");
        }

        let m = self.nrows();
        if beta == 0.0 {
            // y[i] = alpha * sum_j a[i,j] x[j]
            for i in 0..m {
                let rs = rp[i];
                let re = rp[i + 1];
                let mut acc = 0.0;
                for p in rs..re {
                    let j = cj[p];
                    acc = f64::mul_add(vv[p], x[j], acc);
                }
                y[i] = alpha * acc;
            }
        } else if beta == 1.0 {
            // y[i] += alpha * A x
            for i in 0..m {
                let rs = rp[i];
                let re = rp[i + 1];
                let mut acc = 0.0;
                for p in rs..re {
                    let j = cj[p];
                    acc = f64::mul_add(vv[p], x[j], acc);
                }
                y[i] += alpha * acc;
            }
        } else {
            // y[i] = alpha * (A x)_i + beta * y[i]
            for i in 0..m {
                let rs = rp[i];
                let re = rp[i + 1];
                let mut acc = 0.0;
                for p in rs..re {
                    let j = cj[p];
                    acc = f64::mul_add(vv[p], x[j], acc);
                }
                y[i] = alpha * acc + beta * y[i];
            }
        }
        Ok(())
    }

    fn mat_vec_trans(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), crate::error::KError> {
        use crate::matrix::sparse::SparseMatrix;

        // Dimension checks: x is in R^{m}, y in R^{n} for A^T (A is m×n)
        if x.len() != SparseMatrix::nrows(self) || y.len() != SparseMatrix::ncols(self) {
            return Err(crate::error::KError::InvalidInput(
                "Matrix-vector dimension mismatch".to_string(),
            ));
        }

        // Quick exits
        if alpha == 0.0 {
            // y = beta * y
            if beta == 0.0 {
                for v in y.iter_mut() {
                    *v = 0.0;
                }
            } else {
                for v in y.iter_mut() {
                    *v *= beta;
                }
            }
            return Ok(());
        }

        // Scale y by beta (or zero) up front
        if beta == 0.0 {
            for v in y.iter_mut() {
                *v = 0.0;
            }
        } else if beta != 1.0 {
            for v in y.iter_mut() {
                *v *= beta;
            }
        }
        // If beta == 1.0, leave y as-is and accumulate into it.

        // Access CSR structure. These accessor names assume your CSR exposes them.
        // If your type uses different getters, adjust accordingly.
        let row_ptr = self.row_ptr(); // &[usize] of length m+1
        let col_idx = self.col_idx(); // &[usize] of length nnz
        let values = self.values(); // &[f64]   of length nnz

        // y_j += alpha * a_ij * x_i  for all nonzeros a_ij
        let m = SparseMatrix::nrows(self);
        for i in 0..m {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            // SAFETY: bounds guaranteed by CSR invariants
            for k in start..end {
                let j = col_idx[k];
                y[j] += alpha * values[k] * xi;
            }
        }

        Ok(())
    }

    fn nrows(&self) -> usize {
        use crate::matrix::sparse::SparseMatrix;
        SparseMatrix::nrows(self)
    }
    fn ncols(&self) -> usize {
        use crate::matrix::sparse::SparseMatrix;
        SparseMatrix::ncols(self)
    }
}

/// Standard dot product implementation
pub struct StandardDotOp;

impl DotOp<f64> for StandardDotOp {
    fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn norm2(&self, x: &[f64]) -> f64 {
        self.dot(x, x).sqrt()
    }
}

/// Unified kernel trait for local vs distributed operations
/// Provides a consistent interface for AMG operations that can work
/// both in single-process (local) and multi-process (MPI) scenarios
pub trait KernelOp<T> {
    /// The communicator type for this kernel (e.g., UniverseComm for MPI, () for local)
    type Comm: crate::parallel::Comm;

    /// Matrix-vector product with communicator support: y = alpha * A * x + beta * y
    fn kernel_mat_vec(
        &self,
        matrix: &dyn MatVecOp<T>,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>;

    /// Transpose matrix-vector product: y = alpha * A^T * x + beta * y
    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<T>,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut [T],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>;

    /// Global dot product with reduction across processes
    fn kernel_dot(&self, x: &[T], y: &[T], comm: &Self::Comm) -> T;

    /// Global norm computation with reduction
    fn kernel_norm2(&self, x: &[T], comm: &Self::Comm) -> T;

    /// Vector operations: y = alpha * x + beta * y
    fn kernel_axpby(&self, alpha: T, x: &[T], beta: T, y: &mut [T]);

    /// Copy operation: y = x
    fn kernel_copy(&self, x: &[T], y: &mut [T]);

    /// Scale operation: x = alpha * x
    fn kernel_scale(&self, alpha: T, x: &mut [T]);
}

/// Local (single-process) kernel implementation
pub struct LocalKernel;

impl KernelOp<f64> for LocalKernel {
    type Comm = crate::parallel::NoComm;

    fn kernel_mat_vec(
        &self,
        matrix: &dyn MatVecOp<f64>,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        _comm: &Self::Comm,
    ) -> Result<(), crate::error::KError> {
        // For local operations, no communication needed
        matrix.mat_vec(alpha, x, beta, y)
    }

    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<f64>,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        _comm: &Self::Comm,
    ) -> Result<(), crate::error::KError> {
        matrix.mat_vec_trans(alpha, x, beta, y)
    }

    fn kernel_dot(&self, x: &[f64], y: &[f64], _comm: &Self::Comm) -> f64 {
        let dot_op = StandardDotOp;
        dot_op.dot(x, y)
    }

    fn kernel_norm2(&self, x: &[f64], _comm: &Self::Comm) -> f64 {
        let dot_op = StandardDotOp;
        dot_op.norm2(x)
    }

    fn kernel_axpby(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
            *y_val = alpha * x_val + beta * (*y_val);
        }
    }

    fn kernel_copy(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }

    fn kernel_scale(&self, alpha: f64, x: &mut [f64]) {
        for val in x.iter_mut() {
            *val *= alpha;
        }
    }
}

/// Distributed (MPI) kernel implementation for future use
/// Currently a placeholder that delegates to local operations
pub struct DistributedKernel;

impl KernelOp<f64> for DistributedKernel {
    type Comm = crate::parallel::UniverseComm;

    fn kernel_mat_vec(
        &self,
        matrix: &dyn MatVecOp<f64>,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError> {
        let mut local = vec![0.0f64; y.len()];
        matrix.mat_vec(alpha, x, 0.0, &mut local)?;
        use crate::parallel::Comm as _;
        comm.allreduce_sum_slice(&mut local);
        if beta == 0.0 {
            y.copy_from_slice(&local);
        } else {
            let original: Vec<f64> = y.to_vec();
            for (out, (accum, orig)) in y
                .iter_mut()
                .zip(local.into_iter().zip(original.into_iter()))
            {
                *out = accum + beta * orig;
            }
        }
        Ok(())
    }

    fn kernel_mat_vec_trans(
        &self,
        matrix: &dyn MatVecOp<f64>,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError> {
        let mut local = vec![0.0f64; y.len()];
        matrix.mat_vec_trans(alpha, x, 0.0, &mut local)?;
        use crate::parallel::Comm as _;
        comm.allreduce_sum_slice(&mut local);
        if beta == 0.0 {
            y.copy_from_slice(&local);
        } else {
            let original: Vec<f64> = y.to_vec();
            for (out, (accum, orig)) in y
                .iter_mut()
                .zip(local.into_iter().zip(original.into_iter()))
            {
                *out = accum + beta * orig;
            }
        }
        Ok(())
    }

    fn kernel_dot(&self, x: &[f64], y: &[f64], comm: &Self::Comm) -> f64 {
        use crate::parallel::Comm;
        // Compute local dot product
        let local_dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        // Reduce across all processes
        comm.all_reduce_f64(local_dot)
    }

    fn kernel_norm2(&self, x: &[f64], comm: &Self::Comm) -> f64 {
        self.kernel_dot(x, x, comm).sqrt()
    }

    fn kernel_axpby(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        // Vector operations are local in distributed setting
        for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
            *y_val = alpha * x_val + beta * (*y_val);
        }
    }

    fn kernel_copy(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }

    fn kernel_scale(&self, alpha: f64, x: &mut [f64]) {
        for val in x.iter_mut() {
            *val *= alpha;
        }
    }
}

/// Unified AMG kernel trait to eliminate code duplication between local and MPI variants
pub trait AmgKernel {
    /// Associated communicator type  
    type Comm: crate::parallel::Comm;

    /// Matrix-vector multiplication with alpha/beta scaling
    fn matvec<M>(
        &self,
        alpha: f64,
        a: &M,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>
    where
        M: MatVecOp<f64>;

    /// Global dot product with communicator reduction
    fn dot(&self, x: &[f64], y: &[f64], comm: &Self::Comm) -> f64;

    /// Global norm with communicator reduction  
    fn norm(&self, x: &[f64], comm: &Self::Comm) -> f64 {
        self.dot(x, x, comm).sqrt()
    }

    /// Vector scaling: x = alpha * x
    fn scale(&self, alpha: f64, x: &mut [f64]);

    /// Vector copy: y = x
    fn copy(&self, x: &[f64], y: &mut [f64]);

    /// AXPY operation: y = alpha * x + y
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]);
}

/// Local (single-process) AMG kernel implementation
pub struct LocalAmgKernel;

impl LocalAmgKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LocalAmgKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl AmgKernel for LocalAmgKernel {
    type Comm = crate::parallel::NoComm;

    fn matvec<M>(
        &self,
        alpha: f64,
        a: &M,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        _comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>
    where
        M: MatVecOp<f64>,
    {
        a.mat_vec(alpha, x, beta, y)
    }

    fn dot(&self, x: &[f64], y: &[f64], _comm: &Self::Comm) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn scale(&self, alpha: f64, x: &mut [f64]) {
        for val in x.iter_mut() {
            *val *= alpha;
        }
    }

    fn copy(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }

    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
            *y_val += alpha * x_val;
        }
    }
}

/// Distributed (MPI) AMG kernel implementation
pub struct DistributedAmgKernel;

impl DistributedAmgKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DistributedAmgKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl AmgKernel for DistributedAmgKernel {
    type Comm = crate::parallel::UniverseComm;

    fn matvec<M>(
        &self,
        alpha: f64,
        a: &M,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
        comm: &Self::Comm,
    ) -> Result<(), crate::error::KError>
    where
        M: MatVecOp<f64>,
    {
        let mut local = vec![0.0f64; y.len()];
        a.mat_vec(alpha, x, 0.0, &mut local)?;
        use crate::parallel::Comm as _;
        comm.allreduce_sum_slice(&mut local);
        if beta == 0.0 {
            y.copy_from_slice(&local);
        } else {
            let original: Vec<f64> = y.to_vec();
            for (out, (accum, orig)) in y
                .iter_mut()
                .zip(local.into_iter().zip(original.into_iter()))
            {
                *out = accum + beta * orig;
            }
        }
        Ok(())
    }

    fn dot(&self, x: &[f64], y: &[f64], comm: &Self::Comm) -> f64 {
        use crate::parallel::Comm;
        // Compute local dot product, then reduce across processes
        let local_dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        comm.all_reduce_f64(local_dot)
    }

    fn scale(&self, alpha: f64, x: &mut [f64]) {
        // Vector operations are local even in distributed setting
        for val in x.iter_mut() {
            *val *= alpha;
        }
    }

    fn copy(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }

    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for (y_val, x_val) in y.iter_mut().zip(x.iter()) {
            *y_val += alpha * x_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::CsrMatrix;

    // Simple test to verify traits can be imported and used
    #[test]
    fn test_traits_exist() {
        // This test just verifies that all traits compile and can be referenced
        // More comprehensive tests would require mock implementations

        // Test that trait bounds can be specified
        fn _test_matvec_bound<T, V>(_: &T)
        where
            T: MatVec<V>,
        {
        }
        fn _test_mattransvec_bound<T, V>(_: &T)
        where
            T: MatTransVec<V>,
        {
        }
        fn _test_inner_product_bound<T, V>(_: &T)
        where
            T: InnerProduct<V>,
        {
        }
        fn _test_indexing_bound<T>(_: &T)
        where
            T: Indexing,
        {
        }
        fn _test_mat_shape_bound<T>(_: &T)
        where
            T: MatShape,
        {
        }
        fn _test_row_pattern_bound<T>(_: &T)
        where
            T: RowPattern,
        {
        }
        fn _test_matrix_get_bound<T, U>(_: &T)
        where
            T: MatrixGet<U>,
        {
        }
        fn _test_submatrix_extract_bound<T>(_: &T)
        where
            T: SubmatrixExtract,
        {
        }

        // All traits should compile
        assert!(true);
    }

    #[test]
    fn test_inner_product_scalar_trait_bounds() {
        // Test that the associated Scalar type has the required bounds
        fn _check_scalar_bounds<T: Copy + PartialOrd + From<f64> + Into<f64>>() {}

        // f64 should satisfy the bounds
        _check_scalar_bounds::<f64>();

        assert!(true);
    }

    #[test]
    fn test_trait_names_and_methods() {
        // Verify method names exist by checking trait signatures
        trait TestMatVec<V> {
            fn matvec(&self, x: &V, y: &mut V);
        }

        trait TestMatTransVec<V> {
            fn mattransvec(&self, x: &V, y: &mut V);
        }

        trait TestInnerProduct<V> {
            type Scalar: Copy + PartialOrd + From<f64> + Into<f64>;
            fn dot(&self, x: &V, y: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar;
            fn norm(&self, x: &V, comm: &impl crate::parallel::Comm) -> Self::Scalar {
                let local_sq = self.dot(x, x, comm);
                let global_sq = comm.all_reduce_f64(local_sq.into());
                (global_sq.sqrt()).into()
            }
        }
        struct Dummy;

        impl TestMatVec<Vec<f64>> for Dummy {
            fn matvec(&self, _x: &Vec<f64>, _y: &mut Vec<f64>) {}
        }

        impl TestMatTransVec<Vec<f64>> for Dummy {
            fn mattransvec(&self, _x: &Vec<f64>, _y: &mut Vec<f64>) {}
        }

        impl TestInnerProduct<Vec<f64>> for Dummy {
            type Scalar = f64;
            fn dot(
                &self,
                _x: &Vec<f64>,
                _y: &Vec<f64>,
                _comm: &impl crate::parallel::Comm,
            ) -> Self::Scalar {
                0.0
            }
        }

        fn _use_traits<
            T: TestMatVec<Vec<f64>> + TestMatTransVec<Vec<f64>> + TestInnerProduct<Vec<f64>>,
        >() {
        }
        _use_traits::<Dummy>();

        let dummy = Dummy;
        let comm = crate::parallel::NoComm;
        let v = vec![0.0; 1];
        let mut y = vec![0.0; 1];
        dummy.matvec(&v, &mut y);
        dummy.mattransvec(&v, &mut y);
        let _ = dummy.dot(&v, &v, &comm);
        let _ = dummy.norm(&v, &comm);

        // All method signatures should compile without panicking.
    }

    #[test]
    fn csr_matvec_happy_path() {
        // 2x3 CSR: row_ptr=[0,2,3], col_idx=[0,2,1], val=[1,4,5]
        // A = [1 0 4; 0 5 0]
        let a = CsrMatrix::from_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 4.0, 5.0]);
        let x = [10.0, 20.0, 30.0];
        let mut y = [0.0; 2];
        MatVecOp::mat_vec(&a, 1.0, &x, 0.0, &mut y).unwrap();
        let expected = [130.0, 100.0];
        for (got, target) in y.iter().zip(expected.iter()) {
            assert!((got - target).abs() < 1e-12);
        }
        // with scaling
        let mut y2 = [1.0, 2.0];
        MatVecOp::mat_vec(&a, 2.0, &x, 3.0, &mut y2).unwrap();
        // 2*A*x + 3*y0
        assert!((y2[0] - (2.0 * 130.0 + 3.0 * 1.0)).abs() < 1e-12);
        assert!((y2[1] - (2.0 * 100.0 + 3.0 * 2.0)).abs() < 1e-12);
    }
}
```

`src/core/wrappers.rs`

```rust
//! Wrappers for faer dense matrix types and vector operations.
//!
//! This module provides implementations of core linear algebra traits for `faer::Mat`, `faer::MatRef`, and `Vec<T>`,
//! enabling their use in generic iterative solvers and preconditioners. It also provides parallel and distributed
//! inner product operations, supporting both single-threaded, multi-threaded (Rayon), and MPI-based distributed environments.
//!
//! # Features
//! - Matrix-vector and matrix-transpose-vector multiplication for `faer` dense matrices.
//! - Inner product and norm operations for vectors, with optional Rayon parallelism.
//! - Distributed inner product and norm for MPI-enabled builds.
//! - Indexing trait implementations for vectors and matrices.
//!
//! # Usage
//! These wrappers allow the use of `faer` matrices and Rust vectors as generic types in the KrylovKit solver framework.
//!
//! # References
//! - [faer crate documentation](https://docs.rs/faer)
//! - [num-traits crate documentation](https://docs.rs/num-traits)

use crate::core::traits::{Indexing, InnerProduct, MatTransVec, MatVec};
use faer::{Mat, MatRef};
use num_traits::Float;

/// Implements matrix-vector multiplication for a matrix reference (`faer::MatRef`).
impl<'a, T: Float> MatVec<Vec<T>> for MatRef<'a, T> {
    fn matvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(
            self.nrows(),
            y.len(),
            "Output vector y has incorrect length"
        );
        assert_eq!(self.ncols(), x.len(), "Input vector x has incorrect length");
        for i in 0..self.nrows() {
            y[i] = T::zero();
            for j in 0..self.ncols() {
                y[i] = y[i] + self[(i, j)] * x[j];
            }
        }
    }
}

/// Implements matrix-transpose-vector multiplication for a matrix reference (`faer::MatRef`).
impl<'a, T: Float> MatTransVec<Vec<T>> for MatRef<'a, T> {
    fn mattransvec(&self, x: &Vec<T>, y: &mut Vec<T>) {
        assert_eq!(
            self.ncols(),
            y.len(),
            "Output vector y has incorrect length"
        );
        assert_eq!(self.nrows(), x.len(), "Input vector x has incorrect length");
        for j in 0..self.ncols() {
            y[j] = T::zero();
            for i in 0..self.nrows() {
                y[j] = y[j] + self[(i, j)] * x[i];
            }
        }
    }
}

/// Implements inner product and norm for vectors, with optional Rayon parallelism.
///
/// If the `rayon` feature is enabled, uses parallel iterators for performance.
impl<T: Float + From<f64> + Into<f64> + Send + Sync> InnerProduct<Vec<T>> for () {
    type Scalar = T;
    /// Computes the dot product of two vectors: `x^T y` with parallel reduction.
    fn dot(&self, x: &Vec<T>, y: &Vec<T>, comm: &impl crate::parallel::Comm) -> T {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");
        let local_dot = {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                x.as_slice()
                    .par_iter()
                    .zip(y.as_slice().par_iter())
                    .map(|(xi, yi)| *xi * *yi)
                    .reduce(|| T::zero(), |acc, v| acc + v)
            }
            #[cfg(not(feature = "rayon"))]
            {
                x.iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| *xi * *yi)
                    .fold(T::zero(), |acc, v| acc + v)
            }
        };
        let global_dot = comm.all_reduce_f64(local_dot.into());
        global_dot.into()
    }
}

/// Distributed inner product and norm for MPI-enabled builds.
///
/// This struct is only available if the `mpi` feature is enabled. It wraps a communicator and provides
/// collective dot product and norm operations across distributed memory processes.
#[cfg(feature = "mpi")]
pub struct DistributedInnerProduct<'a, C: crate::parallel::Comm> {
    /// Reference to the communicator implementing the `Comm` trait.
    pub comm: &'a C,
}

#[cfg(feature = "mpi")]
impl<'a, C: crate::parallel::Comm> DistributedInnerProduct<'a, C> {
    /// Computes the distributed dot product of two slices, reducing across all processes.
    pub fn dot<
        T: Copy
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + num_traits::FromPrimitive
            + num_traits::ToPrimitive
            + num_traits::Zero,
    >(
        &self,
        x: &[T],
        y: &[T],
    ) -> T {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");
        // Convert local dot product to f64 for reduction
        let local: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a * b).to_f64().unwrap_or(0.0))
            .sum();
        let global = self.comm.all_reduce(local);
        T::from_f64(global).unwrap_or(T::zero())
    }
    /// Computes the distributed Euclidean norm of a slice, reducing across all processes.
    pub fn norm<
        T: Copy
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + num_traits::FromPrimitive
            + num_traits::ToPrimitive
            + num_traits::Zero
            + num_traits::Float,
    >(
        &self,
        x: &[T],
    ) -> T {
        let local: f64 = x.iter().map(|&a| (a * a).to_f64().unwrap_or(0.0)).sum();
        let global = self.comm.all_reduce(local);
        T::from_f64(global.sqrt()).unwrap_or(T::zero())
    }
}

/// Implements the `Indexing` trait for `Vec<T>`, treating a vector as a column vector.
impl<T> Indexing for Vec<T> {
    /// Returns the number of rows (length) of the vector.
    fn nrows(&self) -> usize {
        self.len()
    }
}

/// Implements the `Indexing` trait for `faer::Mat`, returning the number of rows.
impl<T> Indexing for Mat<T> {
    fn nrows(&self) -> usize {
        self.nrows()
    }
}
```

`src/core/mat/mod.rs`

```rust
//! Matrix types and operations for kryst.

pub mod shell;

pub use shell::ShellMat;
```

`src/core/mat/shell.rs`

```rust
//! Matrix-free ("shell") operators for kryst.
//!
//! This module provides `ShellMat`, which allows users to define matrix operations
//! via callbacks rather than storing matrix entries explicitly. This is useful for:
//! - Large matrices that are expensive to store
//! - Matrices defined by algorithms (e.g., finite difference operators)
//! - Hierarchical or adaptive methods
//! - GPU-based or distributed matrix operations
//!
//! # Usage
//!
//! ```rust,ignore
//! use kryst::core::mat::shell::ShellMat;
//!
//! // Create a 3x3 diagonal matrix with entries [2.0, 3.0, 4.0]
//! let shell = ShellMat::new(
//!     3,
//!     |x, y| {
//!         let x_ref = x.as_ref();
//!         let y_mut = y.as_mut();
//!         y_mut[0] = 2.0 * x_ref[0];
//!         y_mut[1] = 3.0 * x_ref[1];
//!         y_mut[2] = 4.0 * x_ref[2];
//!     },
//!     |x, y| {
//!         // For a diagonal matrix, transpose is the same
//!         let x_ref = x.as_ref();
//!         let y_mut = y.as_mut();
//!         y_mut[0] = 2.0 * x_ref[0];
//!         y_mut[1] = 3.0 * x_ref[1];
//!         y_mut[2] = 4.0 * x_ref[2];
//!     },
//! );
//! ```

use crate::core::traits::{MatShape, MatTransVec, MatVec};
use std::marker::PhantomData;

type ShellFn<V> = dyn Fn(&V, &mut V) + Send + Sync;
/// A "shell" matrix: user-supplied callbacks for A·x and Aᵀ·x
///
/// `ShellMat` provides a matrix-free interface where matrix operations are defined
/// by user-provided closures. This allows for efficient representation of matrices
/// that don't need to be stored explicitly.
pub struct ShellMat<V> {
    pub dim: usize,
    mult: Box<ShellFn<V>>,
    mult_trans: Box<ShellFn<V>>,
    // Makes the dependency on V explicit without requiring V: Send/Sync.
    // Using `fn(&V)` (not `V`) avoids imposing Send/Sync bounds on V.
    _marker: PhantomData<fn(&V)>,
}

impl<V> ShellMat<V> {
    /// Construct a new shell matrix of size `dim` with user-provided operations.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the square matrix
    /// * `mult` - Closure computing y = A·x
    /// * `mult_trans` - Closure computing y = Aᵀ·x
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kryst::core::mat::shell::ShellMat;
    ///
    /// // Identity matrix
    /// let identity = ShellMat::new(
    ///     3,
    ///     |x, y| {
    ///         let x_ref = x.as_ref();
    ///         let y_mut = y.as_mut();
    ///         for i in 0..x_ref.len() {
    ///             y_mut[i] = x_ref[i];
    ///         }
    ///     },
    ///     |x, y| {
    ///         let x_ref = x.as_ref();
    ///         let y_mut = y.as_mut();
    ///         for i in 0..x_ref.len() {
    ///             y_mut[i] = x_ref[i];
    ///         }
    ///     },
    /// );
    /// ```
    pub fn new<F, G>(dim: usize, mult: F, mult_trans: G) -> Self
    where
        F: Fn(&V, &mut V) + Send + Sync + 'static,
        G: Fn(&V, &mut V) + Send + Sync + 'static,
    {
        ShellMat {
            dim,
            mult: Box::new(mult),
            mult_trans: Box::new(mult_trans),
            _marker: PhantomData,
        }
    }

    /// Create a shell matrix where the transpose operation is the same as the forward operation.
    /// This is useful for symmetric matrices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the square matrix
    /// * `mult` - Closure computing y = A·x (used for both A·x and Aᵀ·x)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kryst::core::mat::shell::ShellMat;
    ///
    /// // Symmetric diagonal matrix
    /// let symmetric = ShellMat::new_symmetric(
    ///     3,
    ///     |x, y| {
    ///         let x_ref = x.as_ref();
    ///         let y_mut = y.as_mut();
    ///         y_mut[0] = 5.0 * x_ref[0];
    ///         y_mut[1] = 3.0 * x_ref[1];
    ///         y_mut[2] = 7.0 * x_ref[2];
    ///     },
    /// );
    /// ```
    pub fn new_symmetric<F>(dim: usize, mult: F) -> Self
    where
        F: Fn(&V, &mut V) + Send + Sync + 'static + Clone,
    {
        ShellMat {
            dim,
            mult: Box::new(mult.clone()),
            mult_trans: Box::new(mult),
            _marker: PhantomData,
        }
    }

    /// Get the dimension of this shell matrix.
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

impl<V> MatVec<V> for ShellMat<V>
where
    V: AsRef<[f64]> + AsMut<[f64]>,
{
    /// Apply the matrix-vector product: y = A·x
    fn matvec(&self, x: &V, y: &mut V) {
        (self.mult)(x, y);
    }
}

impl<V> MatTransVec<V> for ShellMat<V>
where
    V: AsRef<[f64]> + AsMut<[f64]>,
{
    /// Apply the matrix-transpose-vector product: y = Aᵀ·x
    fn mattransvec(&self, x: &V, y: &mut V) {
        (self.mult_trans)(x, y);
    }
}

impl<V> MatShape for ShellMat<V> {
    /// Number of rows in the matrix
    fn nrows(&self) -> usize {
        self.dim
    }

    /// Number of columns in the matrix
    fn ncols(&self) -> usize {
        self.dim
    }
}
```

