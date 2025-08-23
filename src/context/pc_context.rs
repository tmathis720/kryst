use crate::config::options::PcOptions;
use crate::error::KError;
use crate::matrix::op::LinOp;
use crate::preconditioner::direct::{LuPc, QrPc, SuperLuDistPc};
use crate::preconditioner::{jacobi::Jacobi, LegacyOpPreconditioner, PcSide, Preconditioner};
#[cfg(feature = "legacy-pc-bridge")]
use crate::preconditioner::{ChebyshevPre, Ilup, Ilut, Ilutp, MatSorType, Sor};
use faer::Mat;
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

    fn apply_mut(
        &mut self,
        side: PcSide,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), KError> {
        self.apply(side, x, y)
    }
}


/// Factory for creating preconditioners.
pub struct PcFactory;

impl PcFactory {
    pub fn create_preconditioner(
        pc_type: PcType,
        _options: Option<&PcOptions>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        match pc_type {
            PcType::Jacobi => Ok(Box::new(Jacobi::new())),
            PcType::Ilut => {
                #[cfg(feature = "legacy-pc-bridge")]
                {
                    let ilut = Ilut::new(0, 0.0);
                    return Ok(Box::new(LegacyOpPreconditioner::new(Box::new(ilut))));
                }
                #[cfg(not(feature = "legacy-pc-bridge"))]
                {
                    return Err(KError::Unsupported(
                        "Ilut requires --features legacy-pc-bridge (or port to modern Preconditioner)",
                    ));
                }
            },
            PcType::Ilutp => {
                #[cfg(feature = "legacy-pc-bridge")]
                {
                    let ilutp = Ilutp::new();
                    return Ok(Box::new(LegacyOpPreconditioner::new(Box::new(ilutp))));
                }
                #[cfg(not(feature = "legacy-pc-bridge"))]
                {
                    return Err(KError::Unsupported(
                        "Ilutp requires --features legacy-pc-bridge (or port to modern Preconditioner)",
                    ));
                }
            },
            PcType::Ilup => {
                #[cfg(feature = "legacy-pc-bridge")]
                {
                    let ilup = Ilup::new(0);
                    return Ok(Box::new(LegacyOpPreconditioner::new(Box::new(ilup))));
                }
                #[cfg(not(feature = "legacy-pc-bridge"))]
                {
                    return Err(KError::Unsupported(
                        "Ilup requires --features legacy-pc-bridge (or port to modern Preconditioner)",
                    ));
                }
            },
            PcType::Sor => {
                #[cfg(feature = "legacy-pc-bridge")]
                {
                    let sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(
                        1.0,
                        1,
                        0,
                        MatSorType::APPLY_LOWER,
                        0.0,
                    );
                    return Ok(Box::new(LegacyOpPreconditioner::new(Box::new(sor))));
                }
                #[cfg(not(feature = "legacy-pc-bridge"))]
                {
                    return Err(KError::Unsupported(
                        "SOR requires --features legacy-pc-bridge (or port to modern Preconditioner)",
                    ));
                }
            },
            PcType::Chebyshev => {
                #[cfg(feature = "legacy-pc-bridge")]
                {
                    let pre = ChebyshevPre::new(Mat::zeros(0, 0), 0, 1.0, 1.0);
                    return Ok(Box::new(LegacyOpPreconditioner::new(Box::new(pre))));
                }
                #[cfg(not(feature = "legacy-pc-bridge"))]
                {
                    return Err(KError::Unsupported(
                        "Chebyshev requires --features legacy-pc-bridge (or port to modern Preconditioner)",
                    ));
                }
            },
            PcType::None => Ok(Box::new(NoOpPreconditioner)),
            PcType::Lu => Ok(Box::new(LuPc::new())),
            PcType::Qr => Ok(Box::new(QrPc::new())),
            #[cfg(feature = "superlu_dist")]
            PcType::SuperLuDist => Ok(Box::new(SuperLuDistPc::new())),
            #[cfg(not(feature = "superlu_dist"))]
            PcType::SuperLuDist => Err(KError::SolveError(
                "superlu_dist feature not enabled".into(),
            )),
            other => Err(KError::UnrecognizedPcType(format!(
                "{:?} not implemented",
                other
            ))),
        }
    }

    pub fn create_deferred_pc(
        pc_type: PcType,
        options: Option<PcOptions>,
    ) -> Result<DeferredPcInfo, KError> {
        Err(KError::UnrecognizedPcType(format!(
            "{:?} deferred construction not supported",
            pc_type
        )))
    }

    pub fn construct_deferred_preconditioner(
        _info: DeferredPcInfo,
        _matrix: &Mat<f64>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        Err(KError::SolveError(
            "deferred preconditioners not supported".into(),
        ))
    }

    pub fn create_pc_chain(
        _chain: &str,
        _matrix: &Mat<f64>,
        _opts: Option<PcOptions>,
    ) -> Result<Box<dyn Preconditioner>, KError> {
        Err(KError::SolveError("PC chaining not supported".into()))
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
    use std::str::FromStr;

    #[test]
    fn factory_builds_lu_qr() {
        let lu = PcFactory::create_preconditioner(PcType::from_str("lu").unwrap(), None).unwrap();
        let qr = PcFactory::create_preconditioner(PcType::from_str("qr").unwrap(), None).unwrap();

        fn _is_pc(_p: &Box<dyn Preconditioner>) {}
        _is_pc(&lu);
        _is_pc(&qr);
    }

    #[test]
    fn factory_builds_superludist_or_errors_by_feature() {
        let r = PcFactory::create_preconditioner(PcType::from_str("superludist").unwrap(), None);

        #[cfg(feature = "superlu_dist")]
        assert!(r.is_ok());

        #[cfg(not(feature = "superlu_dist"))]
        assert!(r.is_err());
    }
}
