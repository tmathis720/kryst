#![cfg(feature = "backend-faer")]

use crate::context::pc_context::PcConfig;
use crate::error::KError;
use crate::preconditioner::Preconditioner;

pub fn try_build(cfg: &PcConfig) -> Result<Option<Box<dyn Preconditioner>>, KError> {
    use crate::preconditioner::builders as b;

    match cfg {
        PcConfig::None => Ok(None),
        PcConfig::Jacobi => b::build_jacobi().map(Some),
        PcConfig::BlockJacobi { block } => b::build_block_jacobi(*block).map(Some),

        PcConfig::Ilu0 => b::build_ilu0().map(Some),
        PcConfig::Iluk { level } => b::build_iluk(*level).map(Some),
        PcConfig::Ilut {
            drop_tol,
            max_fill,
            reordering,
        } => b::build_ilut(*drop_tol, *max_fill, reordering.clone()).map(Some),
        PcConfig::Milu0 => b::build_milu0().map(Some),

        PcConfig::Sor {
            omega,
            sweeps,
            mat_side,
        } => b::build_sor(*omega, *sweeps, *mat_side).map(Some),

        PcConfig::Chebyshev {
            degree,
            eig_lo,
            eig_hi,
        } => b::build_chebyshev(*degree, *eig_lo, *eig_hi).map(Some),

        PcConfig::Asm {
            overlap,
            subdomain_hint,
            block_solver,
            mode,
            weighting,
        } => b::build_asm(
            *overlap,
            *subdomain_hint,
            block_solver.clone(),
            mode.clone(),
            weighting.clone(),
        )
        .map(Some),
        PcConfig::Amg { config } => b::build_amg(config.clone()).map(Some),
        PcConfig::ApproxInv {
            kind,
            levels,
            max_per_col,
            drop_tol,
            reg,
            max_cond,
            parallel,
        } => {
            use crate::preconditioner::approxinv_csr::{ApproxInvKind, ApproxInvParams, FsaiCsr, SpaiCsr};
            let params = ApproxInvParams {
                kind: *kind,
                levels: *levels,
                max_per_col: *max_per_col,
                drop_tol: *drop_tol,
                reg: *reg,
                max_cond: *max_cond,
                parallel: *parallel,
            };
            let pc: Box<dyn Preconditioner> = match *kind {
                ApproxInvKind::FSAI => Box::new(FsaiCsr::new_with_params(params)),
                ApproxInvKind::SPAI => Box::new(SpaiCsr::new_with_params(params)),
            };
            Ok(Some(pc))
        }

        PcConfig::Lu => {
            #[cfg(feature = "dense-direct")]
            {
                b::build_lu().map(Some)
            }
            #[cfg(not(feature = "dense-direct"))]
            {
                Ok(None)
            }
        }
        PcConfig::Qr => {
            #[cfg(feature = "dense-direct")]
            {
                b::build_qr().map(Some)
            }
            #[cfg(not(feature = "dense-direct"))]
            {
                Ok(None)
            }
        }
        #[cfg(feature = "superlu_dist")]
        PcConfig::SuperLuDist => b::build_superlu_dist().map(Some),
    }
}
