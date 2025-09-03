//! ILU configuration options and builders.
//!
//! This module provides a stable, serde-friendly configuration surface for the
//! ILU preconditioner family.  It mirrors the design sketched in the
//! user-facing Step 5 API and focuses on the option schema, layering via an
//! `Overlay` trait, validation/derived defaults through `resolve`, and simple
//! serialization helpers.  Runtime wiring into the various ILU engines will be
//! completed in subsequent steps.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::error::KError;

/// Reordering strategies for ILU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReorderingType {
    None,
    RCM,
    AMD,
}

bitflags! {
    /// Miscellaneous feature toggles for iterative ParILU setup.
    #[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
    pub struct IterativeSetupBits: u32 {
        const DAMPING_AUTO  = 1 << 0;
        const REPRODUCIBLE  = 1 << 1;
        const TRACE_HISTORY = 1 << 2;
        const PIVOT_STAB    = 1 << 3;
        const OVERLAP_COMM  = 1 << 4;
    }
}

/// Iterative setup strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IterativeSetupType {
    Disabled,
    ParILUFixedPoint,
}

/// Triangular solve strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriSolveType {
    Exact,
    Iterative {
        lower_jacobi_iters: u32,
        upper_jacobi_iters: u32,
    },
}

/// ILU kind.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum IluKind {
    ILU0,
    ILUK { k: u32 },
    ILUT { droptol: f64, max_fill_per_row: u32 },
}

/// Pivoting policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PivotPolicy {
    Strict,
    Threshold { tau: f64 },
    DiagPerturb { eta: f64 },
}

/// Reduction/reproducibility mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReproMode {
    Fast,
    Deterministic,
    DeterministicAccurate,
}

/// Configuration for Schur complement handling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SchurConfig {
    pub enable: bool,
    pub droptol_b: f64,
    pub droptol_ef: f64,
    pub droptol_s: f64,
    pub max_row_nnz: u32,
}

impl Default for SchurConfig {
    fn default() -> Self {
        Self {
            enable: false,
            droptol_b: 1e-4,
            droptol_ef: 1e-4,
            droptol_s: 1e-4,
            max_row_nnz: 64,
        }
    }
}

/// Configuration for ParILU fixed-point iterations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterativeSetupConfig {
    pub ty: IterativeSetupType,
    pub tol: f64,
    pub max_iter: u32,
    pub option_bits: IterativeSetupBits,
    pub keep_history: bool,
}

impl Default for IterativeSetupConfig {
    fn default() -> Self {
        Self {
            ty: IterativeSetupType::Disabled,
            tol: 1e-2,
            max_iter: 10,
            option_bits: IterativeSetupBits::PIVOT_STAB,
            keep_history: false,
        }
    }
}

/// Configuration for triangular solve phase.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriSolveConfig {
    pub kind: TriSolveType,
}

impl Default for TriSolveConfig {
    fn default() -> Self {
        Self {
            kind: TriSolveType::Exact,
        }
    }
}

/// Execution toggles governing parallel/distributed paths.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionToggles {
    pub enable_parallel_factorization: bool,
    pub enable_parallel_triangular_solve: bool,
    pub enable_distributed: bool,
}

impl Default for ExecutionToggles {
    fn default() -> Self {
        Self {
            enable_parallel_factorization: false,
            enable_parallel_triangular_solve: true,
            enable_distributed: false,
        }
    }
}

/// Options for reproducible reductions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReductionOptions {
    pub mode: ReproMode,
    pub single_thread_local: bool,
    pub chunk_len: usize,
    pub packet_width: usize,
}

impl Default for ReductionOptions {
    fn default() -> Self {
        Self {
            mode: ReproMode::Fast,
            single_thread_local: true,
            chunk_len: 32_768,
            packet_width: 2,
        }
    }
}

/// Top-level ILU options exposed publicly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IluOptions {
    pub kind: IluKind,
    pub reordering: ReorderingType,
    pub iterative_setup: IterativeSetupConfig,
    pub tri_solve: TriSolveConfig,
    pub schur: SchurConfig,
    pub exec: ExecutionToggles,
    pub pivot: PivotPolicy,
    pub reductions: ReductionOptions,
    pub logging_level: u8,
}

impl Default for IluOptions {
    fn default() -> Self {
        Self {
            kind: IluKind::ILU0,
            reordering: ReorderingType::RCM,
            iterative_setup: IterativeSetupConfig::default(),
            tri_solve: TriSolveConfig::default(),
            schur: SchurConfig::default(),
            exec: ExecutionToggles::default(),
            pivot: PivotPolicy::DiagPerturb { eta: 1e-10 },
            reductions: ReductionOptions::default(),
            logging_level: 0,
        }
    }
}

/// Overlay trait for hierarchical option resolution.
pub trait Overlay {
    fn overlay(self, top: &Self) -> Self;
}

impl Overlay for IluOptions {
    fn overlay(mut self, top: &Self) -> Self {
        // Shallow overlay—`top` replaces `self`.
        self.kind = top.kind;
        self.reordering = top.reordering;
        self.iterative_setup = top.iterative_setup.clone();
        self.tri_solve = top.tri_solve.clone();
        self.schur = top.schur.clone();
        self.exec = top.exec.clone();
        self.pivot = top.pivot;
        self.reductions = top.reductions.clone();
        self.logging_level = top.logging_level;
        self
    }
}

/// Resolved options after validation and derived defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedIluOptions {
    pub o: IluOptions,
}

impl IluOptions {
    /// Validate and derive smart defaults, returning a resolved set of options.
    pub fn resolve(self, matrix_is_distributed: bool) -> Result<ResolvedIluOptions, KError> {
        // Reordering feature gate.
        if matches!(self.reordering, ReorderingType::AMD) && !cfg!(feature = "amd") {
            return Err(KError::InvalidInput(
                "AMD reordering requested but 'amd' feature not enabled".into(),
            ));
        }

        // Schur complement options are not yet implemented.
        if self.schur.enable || self.schur.droptol_s != SchurConfig::default().droptol_s {
            #[cfg(feature = "logging")]
            if self.logging_level > 0 {
                log::warn!("schur_drop_tolerance is reserved for future use and will be ignored");
            }
        }

        // Derived behavior #1: ILUK/ILUT imply ParILU unless explicitly set.
        let mut it = self.iterative_setup.clone();
        if matches!(self.kind, IluKind::ILUK { .. } | IluKind::ILUT { .. })
            && matches!(it.ty, IterativeSetupType::Disabled)
        {
            it.ty = IterativeSetupType::ParILUFixedPoint;
            it.option_bits.insert(IterativeSetupBits::PIVOT_STAB);
        }

        // Derived behavior #2: enforce distributed toggles.
        let mut exec = self.exec.clone();
        if matrix_is_distributed {
            exec.enable_distributed = true;
            exec.enable_parallel_factorization = true;
            exec.enable_parallel_triangular_solve = true;
        }

        let tol = it.tol.clamp(1e-16, 1.0);
        let max_iter = it.max_iter.max(1);

        let pivot = match self.pivot {
            PivotPolicy::Strict => PivotPolicy::Strict,
            PivotPolicy::Threshold { tau } => PivotPolicy::Threshold { tau: tau.max(0.0) },
            PivotPolicy::DiagPerturb { eta } => PivotPolicy::DiagPerturb { eta: eta.max(0.0) },
        };

        let reductions = prefer_repro_in_ci(self.reductions.clone());

        let keep_history =
            it.keep_history || it.option_bits.contains(IterativeSetupBits::TRACE_HISTORY);

        let o2 = IluOptions {
            iterative_setup: IterativeSetupConfig {
                ty: it.ty,
                tol,
                max_iter,
                option_bits: it.option_bits,
                keep_history,
            },
            exec,
            pivot,
            reductions,
            ..self
        };

        Ok(ResolvedIluOptions { o: o2 })
    }
}

fn prefer_repro_in_ci(mut r: ReductionOptions) -> ReductionOptions {
    if std::env::var("KRYST_REPRO").is_ok() || std::env::var("CI").is_ok() {
        r.mode = ReproMode::Deterministic;
    }
    r
}

impl IluOptions {
    /// Parse options from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, KError> {
        toml::from_str(s).map_err(|e| KError::InvalidInput(e.to_string()))
    }

    /// Serialize options to a TOML string.
    pub fn to_toml(&self) -> Result<String, KError> {
        toml::to_string_pretty(self).map_err(|e| KError::InvalidInput(e.to_string()))
    }
}

/// Simple builder façade over [`IluOptions`].
#[derive(Clone, Debug)]
pub struct IluBuilder {
    opts: IluOptions,
}

impl IluBuilder {
    pub fn new() -> Self {
        Self {
            opts: IluOptions::default(),
        }
    }

    pub fn ilu_kind(mut self, kind: IluKind) -> Self {
        self.opts.kind = kind;
        self
    }

    pub fn reordering(mut self, r: ReorderingType) -> Self {
        self.opts.reordering = r;
        self
    }

    pub fn iterative_setup(mut self, f: impl FnOnce(&mut IterativeSetupConfig)) -> Self {
        f(&mut self.opts.iterative_setup);
        self
    }

    pub fn tri_solve(mut self, kind: TriSolveType) -> Self {
        self.opts.tri_solve.kind = kind;
        self
    }

    pub fn schur(mut self, f: impl FnOnce(&mut SchurConfig)) -> Self {
        f(&mut self.opts.schur);
        self
    }

    pub fn exec(mut self, f: impl FnOnce(&mut ExecutionToggles)) -> Self {
        f(&mut self.opts.exec);
        self
    }

    pub fn reductions(mut self, f: impl FnOnce(&mut ReductionOptions)) -> Self {
        f(&mut self.opts.reductions);
        self
    }

    pub fn pivot(mut self, p: PivotPolicy) -> Self {
        self.opts.pivot = p;
        self
    }

    pub fn logging_level(mut self, level: u8) -> Self {
        self.opts.logging_level = level;
        self
    }

    pub fn build(self) -> IluOptions {
        self.opts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_parilu_for_iluk() {
        let opts = IluOptions {
            kind: IluKind::ILUK { k: 2 },
            ..Default::default()
        };
        let resolved = opts.resolve(false).unwrap();
        assert!(matches!(
            resolved.o.iterative_setup.ty,
            IterativeSetupType::ParILUFixedPoint
        ));
        assert!(
            resolved
                .o
                .iterative_setup
                .option_bits
                .contains(IterativeSetupBits::PIVOT_STAB)
        );
    }

    #[test]
    #[cfg(not(feature = "amd"))]
    fn amd_requires_feature() {
        let opts = IluOptions {
            reordering: ReorderingType::AMD,
            ..Default::default()
        };
        let err = opts.resolve(false).unwrap_err();
        match err {
            KError::InvalidInput(msg) => {
                assert!(msg.contains("amd"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn serde_roundtrip() {
        let opts = IluOptions {
            logging_level: 1,
            ..Default::default()
        };
        let toml = opts.to_toml().unwrap();
        let parsed = IluOptions::from_toml_str(&toml).unwrap();
        assert_eq!(opts, parsed);
    }
}
