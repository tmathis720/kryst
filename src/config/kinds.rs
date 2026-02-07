//! Centralized enums and parsers for string-valued PC options.
use std::str::FromStr;

use crate::error::KError;

#[inline]
pub fn invalid_choice(field: &str, value: &str, allowed: &[&'static str]) -> KError {
    KError::SolveError(format!(
        "Invalid {field}: '{value}'. Allowed: {}",
        allowed.join(", ")
    ))
}

macro_rules! simple_kind {
    (
        $(#[$m:meta])*
        $vis:vis enum $Name:ident { $($v:ident => $s:literal),+ $(,)? }
    ) => {
        $(#[$m])*
        $vis enum $Name { $( $v ),+ }

        impl $Name {
            pub const fn allowed() -> &'static [&'static str] {
                &[ $( $s ),+ ]
            }
        }

        impl FromStr for $Name {
            type Err = KError;
            fn from_str(raw: &str) -> Result<Self, Self::Err> {
                let s = raw.to_ascii_lowercase();
                match s.as_str() {
                    $( $s => Ok(Self::$v), )+
                    other => {
                        let name_lower = stringify!($Name).to_ascii_lowercase();
                        Err(invalid_choice(name_lower.as_str(), other, Self::allowed()))
                    },
                }
            }
        }
    };
}

// Keep field naming consistent with options keys in error messages

// pc_reorder
simple_kind! {
    pub enum ReorderKind { None => "none", Colamd => "colamd", Amd => "amd", Rcm => "rcm", CuthillMckee => "cuthill_mckee" }
}

// pc_scaling
simple_kind! {
    pub enum ScalingKind { None => "none", Diagonal => "diagonal", Symmetric => "symmetric" }
}

// pc_ilu_type
simple_kind! {
    pub enum IluTypeKind {
        Ilu0 => "ilu0",
        Iluk => "iluk",
        Ilut => "ilut",
        Milu0 => "milu0",
        BlockJacobi => "block_jacobi",
        GmresIluk => "gmres_iluk",
        GmresIlut => "gmres_ilut",
    }
}

// pc_ilu_reordering_type
simple_kind! {
    pub enum IluReorderKind { None => "none", Rcm => "rcm", Amd => "amd", Natural => "natural" }
}

// pc_ilu_triangular_solve
simple_kind! {
    pub enum IluTriSolveKind { Exact => "exact", Iterative => "iterative" }
}

simple_kind! {
    pub enum IluParFactorKind { None => "none", Block => "block", ParILU => "parilu" }
}

// pc_amg_coarsen_type
simple_kind! {
    pub enum AmgCoarsenKind { Rs => "rs", Hmis => "hmis", Pmis => "pmis", Falgout => "falgout" }
}

// pc_amg_interp_type
simple_kind! {
    pub enum AmgInterpKind {
        Classical => "classical",
        Direct => "direct",
        Multipass => "multipass",
        Extended => "extended",
        Standard => "standard",
        He => "he",
    }
}

// pc_amg_relax_type
simple_kind! {
    pub enum AmgRelaxKind {
        Jacobi => "jacobi",
        Gs => "gs",
        Gsr => "gsr",
        Sgs => "sgs",
        Hgs => "hgs",
        L1Jacobi => "l1jacobi",
        Chebyshev => "chebyshev",
        ChebyshevSafe => "chebyshev_safe",
        SafeguardedGs => "safegs",
        Ilu0 => "ilu0",
        Ras => "ras",
    }
}

// pc_amg_cycle_type
simple_kind! {
    pub enum AmgCycleKind { V => "v", W => "w" }
}

// pc_amg_strength_type
pub enum AmgStrengthKind {
    Classical,
    Symmetric,
    Normalized,
}

impl AmgStrengthKind {
    pub const fn allowed() -> &'static [&'static str] {
        &["classical", "symmetric", "normalized", "sym", "norm"]
    }
}

impl FromStr for AmgStrengthKind {
    type Err = KError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let s = raw.to_ascii_lowercase();
        match s.as_str() {
            "classical" => Ok(Self::Classical),
            "symmetric" | "sym" => Ok(Self::Symmetric),
            "normalized" | "norm" => Ok(Self::Normalized),
            other => Err(invalid_choice("amg_strength_type", other, Self::allowed())),
        }
    }
}

// pc_amg_coarse_solver
pub enum AmgCoarseSolveKind {
    Cg,
    Direct,
    Ilu,
    Smoother,
}

impl AmgCoarseSolveKind {
    pub const fn allowed() -> &'static [&'static str] {
        &["cg", "direct", "direct_dense", "lu", "ilu", "smoother"]
    }
}

impl FromStr for AmgCoarseSolveKind {
    type Err = KError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let s = raw.to_ascii_lowercase();
        match s.as_str() {
            "cg" => Ok(Self::Cg),
            "direct" | "direct_dense" | "lu" => Ok(Self::Direct),
            "ilu" => Ok(Self::Ilu),
            "smoother" => Ok(Self::Smoother),
            other => Err(invalid_choice("amg_coarse_solver", other, Self::allowed())),
        }
    }
}

// pc_asm_block_solver
simple_kind! {
    pub enum AsmBlockSolverKind { Ludense => "ludense", Csr => "csr" }
}

// pc_asm_mode
simple_kind! {
    pub enum AsmModeKind { Asm => "asm", Ras => "ras" }
}

// pc_sor_mat_side
simple_kind! {
    pub enum SorMatSideKind {
        Lower => "lower",
        Upper => "upper",
        Symmetric => "symmetric",
        Eisenstat => "eisenstat",
    }
}
