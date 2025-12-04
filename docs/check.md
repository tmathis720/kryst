I'm conducting a complete, end-to-end audit of `kryst` to flesh out features such as MPI, Rayon, and flexible backend linear algebra libraries (a la `faer`, `nalgebra`, `blas`, `cgmath`, etc.) via a thin interface layer. I want every numerical implementation to rely on `kryst` built-in capabilities for working with CSR CSC, and dense matrices as necessary. I obviously want to avoid having to write duplicative code. Additionally, we are working on making sure that `kryst` is capable of handling real and complex scalar types. We've started this process, but it is still partially complete. We will start from the highest level of the program and work our way down into more detail as we go. Ultimately, we'll be going through each preconditioner and solver one-by-one, but for now we want to focus on the overall architecture and making sure it's totally sound.

Here is the layout of the library overall as it stands:

```bash
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

We will start from the `context` module, which describes the generic interfaces for the Solver and Preconditioners. Remember, if everything looks to be in order in the module, feel free to say "this looks good" or to give some kind of vote of confidence that what has been done looks robust and production-ready.

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
use crate::preconditioner::{PcSide, Preconditioner};
use std::str::FromStr;

#[cfg(feature = "backend-faer")]
type MatSorSide = crate::preconditioner::sor::MatSorType;
#[cfg(not(feature = "backend-faer"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatSorSide {
    APPLY_LOWER,
    APPLY_UPPER,
    SYMMETRIC_SWEEP,
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
        drop_tol: f64,
        max_fill: usize,
        reordering: Option<String>,
    },
    Milu0,
    Sor {
        omega: f64,
        sweeps: usize,
        mat_side: MatSorSide,
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
        kind: ApproxInvKindAlias,
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
                let mat_side = match o.sor_mat_side.as_deref() {
                    Some("lower") | Option::None => MatSorSide::APPLY_LOWER,
                    Some("upper") => MatSorSide::APPLY_UPPER,
                    Some("symmetric") => MatSorSide::SYMMETRIC_SWEEP,
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
        _op: &dyn LinOp<S = S>,
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

    #[cfg(not(feature = "backend-faer"))]
    pub fn construct_deferred_pc_chain(
        _specs: Vec<DeferredPcInfo>,
        _op: &dyn LinOp<S = f64>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        Err(KError::Unsupported(
            "backend-faer feature is required to build chained preconditioners".into(),
        ))
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
#[cfg(feature = "backend-faer")]
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
pub struct KspContext<S: Scalar> {
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

        #[cfg(feature = "backend-faer")]
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

        #[cfg(not(feature = "backend-faer"))]
        if let Some(_pc) = self.pc.as_mut() {
            return Err(KError::Unsupported(
                "preconditioner materialization requires the backend-faer feature".into(),
            ));
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
            let mat_for_residual = self
                .amat
                .as_ref()
                .map(|a| a.as_ref())
                .unwrap_or_else(|| pmat.as_ref());
            let mut residual = vec![0.0f64; b.len()];
            if let Err(e) = mat_for_residual.try_matvec(x, &mut residual) {
                return Err(KError::SolveError(format!("residual matvec failed: {e}")));
            }
            for (ri, &bi) in residual.iter_mut().zip(b.iter()) {
                *ri = bi - *ri;
            }
            let comm = mat_for_residual.comm();
            let res_sq = comm.dot(&residual, &residual);
            return Ok(SolveStats::new(
                0,
                res_sq.sqrt(),
                ConvergedReason::ConvergedAtol,
            ));
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
fn ensure_len<T: Copy + Default>(v: &mut Vec<T>, need: usize) {
    if v.len() < need {
        v.resize(need, T::default());
    } else if v.len() > need {
        v.truncate(need);
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
