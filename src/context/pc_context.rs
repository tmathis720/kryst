use crate::config::options::PcOptions;
use crate::error::KError;
use crate::preconditioner::{
    Preconditioner,
    PreconditionerMat,
    PcSide,
    jacobi::Jacobi,
    LegacyOpPreconditioner,
    Ilut,
    Ilutp,
    Ilup,
    Sor,
    MatSorType,
    ChebyshevPre,
};
use crate::matrix::op::LinOp;
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
    fn setup(&mut self, _a: &dyn LinOp<S = f64>) -> Result<(), KError> { Ok(()) }
    fn apply(&self, _side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        z.copy_from_slice(r);
        Ok(())
    }
}

/// Adapter for matrix-based preconditioners implementing [`PreconditionerMat`].
struct MatOpPreconditioner {
    inner: Box<dyn PreconditionerMat>,
}

impl MatOpPreconditioner {
    fn new(inner: Box<dyn PreconditionerMat>) -> Self { Self { inner } }
}

impl Preconditioner for MatOpPreconditioner {
    fn setup(&mut self, a: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let m = a
            .as_any()
            .downcast_ref::<Mat<f64>>()
            .ok_or_else(|| KError::InvalidInput("expected faer::Mat<f64>".into()))?;
        self.inner.setup_mat(m)
    }

    fn apply(&self, side: PcSide, r: &[f64], z: &mut [f64]) -> Result<(), KError> {
        self.inner.apply_vec(side, r, z)
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
                let ilut = Ilut::new(0, 0.0);
                Ok(Box::new(LegacyOpPreconditioner::new(Box::new(ilut))))
            }
            PcType::Ilutp => {
                let ilutp = Ilutp::new();
                Ok(Box::new(LegacyOpPreconditioner::new(Box::new(ilutp))))
            }
            PcType::Ilup => {
                let ilup = Ilup::new(0);
                Ok(Box::new(LegacyOpPreconditioner::new(Box::new(ilup))))
            }
            PcType::Sor => {
                let sor = Sor::<Mat<f64>, Vec<f64>, f64>::new(1.0, 1, 0, MatSorType::APPLY_LOWER, 0.0);
                Ok(Box::new(LegacyOpPreconditioner::new(Box::new(sor))))
            }
            PcType::Chebyshev => {
                let pre = ChebyshevPre::new(Mat::zeros(0, 0), 0, 1.0, 1.0);
                Ok(Box::new(LegacyOpPreconditioner::new(Box::new(pre))))
            }
            PcType::None => Ok(Box::new(NoOpPreconditioner)),
            _ => Err(KError::UnrecognizedPcType(format!("{:?} not implemented", pc_type))),
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
        Err(KError::SolveError("deferred preconditioners not supported".into()))
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
