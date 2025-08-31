use crate::error::KError;
use crate::preconditioner::{
    Preconditioner,
    direct::{LuPc, QrPc},
    jacobi::Jacobi,
    sor::MatSorType,
};

#[cfg(feature = "legacy-pc-bridge")]
use crate::preconditioner::{ilup::Ilup, ilut::Ilut, LegacyOpPreconditioner};

use crate::preconditioner::sor::SorPc;
use crate::preconditioner::chebyshev::ChebyshevPc;

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
    Err(KError::NotImplemented(
        "BlockJacobi not yet implemented".into(),
    ))
}

/// Build an SOR preconditioner.
pub fn build_sor(
    omega: f64,
    sweeps: usize,
    mat_side: MatSorType,
    _symmetric: bool,
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
    let pc = ChebyshevPc::new(degree, eig_lo, eig_hi);
    Ok(Box::new(pc))
}

pub fn build_lu() -> Result<Box<dyn Preconditioner>, KError> {
    Ok(Box::new(LuPc::new()))
}

pub fn build_qr() -> Result<Box<dyn Preconditioner>, KError> {
    Ok(Box::new(QrPc::new()))
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
    use crate::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind, PivotStrategy};
    let cfg = IluCsrConfig {
            kind: IluKind::Ilu0,
            pivot: PivotStrategy::DiagonalPerturbation,
            pivot_threshold: 1e-12,
            diag_perturb_factor: 1e-10,
            level_sched: cfg!(feature = "rayon"),
            numeric_update_fixed: true,
            logging: 0,
        };
    let pc = IluCsr::new_with_config(cfg);
    Ok(Box::new(pc))
}

pub fn build_iluk(level: usize) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind, PivotStrategy};
    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: level },
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: cfg!(feature = "rayon"),
        numeric_update_fixed: true,
        logging: 0,
    };
    Ok(Box::new(IluCsr::new_with_config(cfg)))
}

pub fn build_ilut(
    drop_tol: f64,
    max_fill: usize,
    _reordering: Option<String>,
) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind, PivotStrategy};
    let cfg = IluCsrConfig {
        kind: IluKind::Ilut { drop_tol, max_per_row: max_fill },
        pivot: PivotStrategy::DiagonalPerturbation,
        pivot_threshold: 1e-12,
        diag_perturb_factor: 1e-10,
        level_sched: cfg!(feature = "rayon"),
        // For ILUT, fast numeric update requires fixed pattern. Let callers override later if needed.
        numeric_update_fixed: true,
        logging: 0,
    };
    Ok(Box::new(IluCsr::new_with_config(cfg)))
}

pub fn build_milu0() -> Result<Box<dyn Preconditioner>, KError> {
    build_ilu0()
}

// ---- ASM / AMG stubs -----------------------------------------------------

pub fn build_asm(
    overlap: usize,
    _hint: Option<usize>,
    block_solver: Option<String>,
) -> Result<Box<dyn Preconditioner>, KError> {
    // Map the optional block solver string to the enum used by AdditiveSchwarz.
    use crate::preconditioner::asm::BlockSolverFactory;
    use crate::preconditioner::asm::AdditiveSchwarz;

    let factory = match block_solver.as_deref() {
        Some("csr") => BlockSolverFactory::CsrSolver,
        _ => BlockSolverFactory::LuDense,
    };

    // Construct ASM with empty subdomains (will be partitioned on setup).
    let asm = AdditiveSchwarz::<faer::Mat<f64>, Vec<f64>, f64>::new(overlap, Vec::new(), factory);
    Ok(Box::new(asm))
}

pub fn build_amg(
    _levels: Option<usize>,
    _smoother: Option<String>,
) -> Result<Box<dyn Preconditioner>, KError> {
    use crate::preconditioner::amg::AMG;
    let amg = AMG::with_config(Default::default());
    Ok(Box::new(amg))
}
