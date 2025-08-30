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

// pc_amg_coarsen_type
simple_kind! {
    pub enum AmgCoarsenKind { Rs => "rs", Hmis => "hmis", Pmis => "pmis", Falgout => "falgout" }
}

// pc_amg_interp_type
simple_kind! {
    pub enum AmgInterpKind { Classical => "classical", Direct => "direct", Multipass => "multipass", Extended => "extended", Standard => "standard" }
}

// pc_amg_relax_type
simple_kind! {
    pub enum AmgRelaxKind { Jacobi => "jacobi", Gs => "gs", Gsr => "gsr", Sgs => "sgs", Hgs => "hgs", L1Jacobi => "l1jacobi", Chebyshev => "chebyshev" }
}

// pc_asm_block_solver
simple_kind! {
    pub enum AsmBlockSolverKind { Ludense => "ludense", Csr => "csr" }
}
