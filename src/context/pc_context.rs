//! Preconditioner Context for kryst.
//!
//! This module provides the context and configuration structures for preconditioners
//! in the kryst linear algebra library. It includes runtime configuration for different
//! preconditioner types and their parameters, moved from ksp_context for better
//! separation of concerns.
//!
//! # Supported Preconditioners
//!
//! - Jacobi: Diagonal scaling preconditioner.
//! - Ssor: Symmetric Successive Over-Relaxation.
//! - Ilu0: Incomplete LU factorization with zero fill-in.
//! - Ilup: Incomplete LU with fixed fill-in level.
//! - Ilut: Incomplete LU with threshold-based dropping.
//! - Chebyshev: Polynomial preconditioner using Chebyshev polynomials.
//! - ApproxInv: Approximate inverse preconditioner with configurable sparsity.
//! - BlockJacobi: Block-diagonal Jacobi preconditioner.
//! - Multicolor: Multicoloring-based preconditioner.
//! - AMG: Algebraic Multigrid preconditioner.
//! - AdditiveSchwarz: Additive Schwarz domain decomposition preconditioner.

use std::str::FromStr;
use faer::Mat;
use crate::preconditioner::{Preconditioner, PcSide};
use crate::solver::{LuSolver, direct_lu::QrSolver, LinearSolver}; 
use crate::config::options::PcOptions;
use crate::error::KError;

/// All supported preconditioner types for runtime selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcType {
    /// Jacobi (diagonal scaling) preconditioner
    Jacobi,
    /// Incomplete LU factorization with zero fill-in
    Ilu0,
    /// No preconditioning
    None,
    /// Incomplete LU factorization (generic)
    Ilu,
    /// Incomplete LU factorization with threshold
    Ilut,
    /// Incomplete LU factorization with threshold and pivoting
    Ilutp,
    /// Incomplete LU factorization with partial pivoting
    Ilup,
    /// Block Jacobi preconditioner
    BlockJacobi,
    /// Successive Over-Relaxation (SOR) preconditioner
    Sor,
    /// Additive Schwarz Method (ASM) preconditioner
    Asm,
    /// Chebyshev preconditioner
    Chebyshev,
    /// Algebraic Multigrid (AMG) preconditioner
    Amg,
    /// Approximate inverse preconditioner
    ApproxInverse,
    /// Direct LU factorization solver (for PREONLY)
    Lu,
    /// Direct QR factorization solver (for PREONLY)
    Qr,
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
            "blockjacobi" | "block_jacobi" | "bjacobi" => Ok(PcType::BlockJacobi),
            "sor" => Ok(PcType::Sor),
            "asm" => Ok(PcType::Asm),
            "chebyshev" => Ok(PcType::Chebyshev),
            "amg" => Ok(PcType::Amg),
            "approxinverse" | "approx_inverse" => Ok(PcType::ApproxInverse),
            "lu" => Ok(PcType::Lu),
            "qr" => Ok(PcType::Qr),
            _ => Err(KError::UnrecognizedPcType(s.to_string())),
        }
    }
}

/// Information for deferred preconditioner construction.
///
/// For matrix-dependent preconditioners (ASM, AMG, Chebyshev), we store the type
/// and configuration options, then construct the actual preconditioner during setup()
/// when we have access to the matrix.
#[derive(Debug, Clone)]
pub struct DeferredPcInfo {
    /// Preconditioner type to construct
    pub pc_type: PcType,
    /// Configuration options (if any)
    pub options: Option<PcOptions>,
}

/// A no-op preconditioner that performs the identity operation.
pub struct NoOpPreconditioner;

impl Preconditioner<Mat<f64>, Vec<f64>> for NoOpPreconditioner {
    fn setup(&mut self, _a: &Mat<f64>) -> Result<(), KError> {
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        z.clone_from(r);
        Ok(())
    }
}

/// Wrapper for LuSolver to act as a preconditioner for PREONLY method.
/// 
/// This allows using direct LU factorization as a "preconditioner" when
/// KSP type is set to PREONLY.
pub struct LuPreconditioner {
    solver: LuSolver<f64>,
}

impl LuPreconditioner {
    pub fn new() -> Self {
        Self {
            solver: LuSolver::new(),
        }
    }
    
    /// Perform direct solve (for PREONLY usage).
    pub fn solve_direct(&mut self, a: &Mat<f64>, b: &Vec<f64>, x: &mut Vec<f64>) -> Result<(), KError> {
        self.solver.solve(a, None, b, x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm))?;
        Ok(())
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for LuPreconditioner {
    fn setup(&mut self, _a: &Mat<f64>) -> Result<(), KError> {
        // Setup is handled in the solve call for direct methods
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        // For direct methods used as preconditioners, this would typically
        // not be called in PREONLY mode. But we provide a reasonable implementation.
        z.clone_from(r);
        Ok(())
    }
}

/// Wrapper for QrSolver to act as a preconditioner for PREONLY method.
/// 
/// This allows using direct QR factorization as a "preconditioner" when  
/// KSP type is set to PREONLY.
pub struct QrPreconditioner {
    solver: QrSolver,
}

impl QrPreconditioner {
    pub fn new() -> Self {
        Self {
            solver: QrSolver::new(),
        }
    }
    
    /// Perform direct solve (for PREONLY usage).
    pub fn solve_direct(&mut self, a: &Mat<f64>, b: &Vec<f64>, x: &mut Vec<f64>) -> Result<(), KError> {
        self.solver.solve(a, None, b, x, &crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm))?;
        Ok(())
    }
}

impl Preconditioner<Mat<f64>, Vec<f64>> for QrPreconditioner {
    fn setup(&mut self, _a: &Mat<f64>) -> Result<(), KError> {
        // Setup is handled in the solve call for direct methods
        Ok(())
    }

    fn apply(&self, _side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        // For direct methods used as preconditioners, this would typically
        // not be called in PREONLY mode. But we provide a reasonable implementation.
        z.clone_from(r);
        Ok(())
    }
}

/// Factory for constructing preconditioners based on PcType.
pub struct PcFactory;

impl PcFactory {
    /// Create a preconditioner based on the specified type and options.
    ///
    /// # Arguments
    /// * `pc_type` - The type of preconditioner to create
    /// * `options` - Optional configuration for the preconditioner
    ///
    /// # Returns
    /// * `Ok(Box<dyn Preconditioner<...>>)` - The constructed preconditioner  
    /// * `Err(KError)` - If the preconditioner type is not implemented or construction fails
    pub fn create_preconditioner(
        pc_type: PcType, 
        options: Option<&PcOptions>
    ) -> Result<Box<dyn Preconditioner<Mat<f64>, Vec<f64>>>, KError> {
        use crate::preconditioner::{Jacobi, Ilu0, Ilutp};
        
        match pc_type {
            // Matrix-independent preconditioners - construct immediately
            PcType::Jacobi => {
                Ok(Box::new(Jacobi::new()))
            },
            PcType::Ilu0 => {
                Ok(Box::new(Ilu0::new()))
            },
            PcType::None => {
                Ok(Box::new(NoOpPreconditioner))
            },
            PcType::Lu => {
                Ok(Box::new(LuPreconditioner::new()))
            },
            PcType::Qr => {
                Ok(Box::new(QrPreconditioner::new()))
            },
            PcType::Ilutp => {
                // Get parameters from options if available
                let ilutp = if let Some(opts) = options {
                    let max_fill = opts.ilut_max_fill.unwrap_or(10);
                    let drop_tol = opts.drop_tol.unwrap_or(1e-4);
                    let perm_tol = opts.ilut_perm_tol.unwrap_or(0.1);
                    Ilutp::with_params(max_fill, drop_tol, perm_tol)
                } else {
                    Ilutp::new()
                };
                Ok(Box::new(ilutp))
            },
            
            // Matrix-dependent preconditioners cannot be constructed without a matrix
            PcType::Asm | PcType::Chebyshev | PcType::Amg => {
                Err(KError::SolveError(format!("{:?} requires matrix for construction - use create_deferred_pc", pc_type)))
            },
            
            // Not yet implemented preconditioners
            PcType::Ilu | PcType::Ilut | PcType::Ilup | PcType::BlockJacobi | 
            PcType::Sor | PcType::ApproxInverse => {
                Err(KError::UnrecognizedPcType(format!("{:?} preconditioner not yet implemented", pc_type)))
            }
        }
    }

    /// Create a DeferredPcInfo for matrix-dependent preconditioners.
    ///
    /// # Arguments
    /// * `pc_type` - The type of preconditioner to defer
    /// * `options` - Optional configuration for the preconditioner
    ///
    /// # Returns
    /// * `Ok(DeferredPcInfo)` - Information for later construction
    /// * `Err(KError)` - If the preconditioner type doesn't require deferring
    pub fn create_deferred_pc(pc_type: PcType, options: Option<PcOptions>) -> Result<DeferredPcInfo, KError> {
        match pc_type {
            PcType::Asm | PcType::Chebyshev | PcType::Amg => {
                Ok(DeferredPcInfo {
                    pc_type,
                    options,
                })
            },
            _ => {
                Err(KError::SolveError(format!("{:?} does not require deferred construction", pc_type)))
            }
        }
    }

    /// Construct a deferred preconditioner with access to the matrix.
    ///
    /// # Arguments
    /// * `deferred_info` - Information about the preconditioner to construct
    /// * `matrix` - The system matrix
    ///
    /// # Returns
    /// * `Ok(Box<dyn Preconditioner<...>>)` - The constructed preconditioner
    /// * `Err(KError)` - If construction fails
    pub fn construct_deferred_preconditioner(
        deferred_info: DeferredPcInfo,
        matrix: &Mat<f64>
    ) -> Result<Box<dyn Preconditioner<Mat<f64>, Vec<f64>>>, KError> {
        #[cfg(feature = "logging")]
        log::trace!("Constructing deferred preconditioner: {:?}", deferred_info.pc_type);
        
        match deferred_info.pc_type {
            PcType::Asm => {
                // TODO: Implement ASM preconditioner construction with matrix
                Err(KError::UnrecognizedPcType("ASM preconditioner construction not yet implemented".to_string()))
            },
            PcType::Chebyshev => {
                // Create ChebyshevPre preconditioner with matrix
                let options = deferred_info.options.as_ref();
                let degree = options.and_then(|opt| opt.chebyshev_degree).unwrap_or(4);
                
                let (lambda_min, lambda_max) = if let (Some(min), Some(max)) = 
                    (options.and_then(|opt| opt.chebyshev_lambda_min), options.and_then(|opt| opt.chebyshev_lambda_max)) {
                    (min, max)
                } else {
                    // Estimate eigenvalue bounds
                    use crate::preconditioner::chebyshev::ChebyshevPre;
                    let (estimated_min, estimated_max) = ChebyshevPre::estimate_eigenvalue_bounds(matrix, 50, 1e-6);
                    #[cfg(feature = "logging")]
                    log::trace!("Estimated Chebyshev eigenvalue bounds: λ_min={:.6e}, λ_max={:.6e}", estimated_min, estimated_max);
                    (estimated_min, estimated_max)
                };
                
                #[cfg(feature = "logging")]
                log::trace!("Creating Chebyshev preconditioner: degree={}, λ_min={:.6e}, λ_max={:.6e}", degree, lambda_min, lambda_max);
                
                Ok(Box::new(crate::preconditioner::chebyshev::ChebyshevPre::new(
                    matrix.clone(), degree, lambda_min, lambda_max
                )))
            },
            PcType::Amg => {
                // Create AMG preconditioner with matrix
                let options = deferred_info.options.as_ref();
                let max_levels = options.and_then(|opt| opt.amg_levels).unwrap_or(10);
                let threshold = options.and_then(|opt| opt.amg_strength_threshold).unwrap_or(0.25);
                let nu_pre = options.and_then(|opt| opt.amg_nu_pre).unwrap_or(1);
                let nu_post = options.and_then(|opt| opt.amg_nu_post).unwrap_or(1);
                
                #[cfg(feature = "logging")]
                log::trace!("Creating AMG preconditioner: max_levels={}, threshold={:.6e}, nu_pre={}, nu_post={}", 
                       max_levels, threshold, nu_pre, nu_post);
                
                Ok(Box::new(crate::preconditioner::amg::AMG::with_smoothing(matrix, max_levels, threshold, nu_pre, nu_post)))
            },
            _ => {
                Err(KError::UnrecognizedPcType(format!("Unexpected deferred PC type: {:?}", deferred_info.pc_type)))
            }
        }
    }

    /// Create PC chain from configuration string.
    ///
    /// # Arguments
    /// * `chain_str` - Comma-separated list of preconditioner names (e.g., "jacobi,ilu0,chebyshev")
    /// * `matrix` - The system matrix
    /// * `options` - Optional configuration for individual preconditioners
    ///
    /// # Returns
    /// * `Ok(Box<dyn Preconditioner<...>>)` - The constructed PC chain
    /// * `Err(KError)` - If construction fails
    pub fn create_pc_chain(
        chain_str: &str,
        matrix: &Mat<f64>,
        options: Option<&PcOptions>
    ) -> Result<Box<dyn Preconditioner<Mat<f64>, Vec<f64>>>, KError> {
        #[cfg(feature = "logging")]
        log::trace!("Constructing PC chain: {}", chain_str);
        
        // Parse and construct PC chain
        use crate::preconditioner::chain::PcChain;
        let mut pc_chain = PcChain::new();
        
        // Parse the chain string (e.g., "jacobi,ilu0,chebyshev")
        let pc_names = PcChain::parse_chain_string(chain_str);
        
        // Construct each preconditioner in the chain
        for pc_name in pc_names {
            match pc_name.as_str() {
                "jacobi" => {
                    let mut jacobi = crate::preconditioner::jacobi::Jacobi::new();
                    jacobi.setup(matrix)?;
                    pc_chain.add_preconditioner(Box::new(jacobi));
                },
                "ilu0" => {
                    let mut ilu0 = crate::preconditioner::ilu::Ilu0::new();
                    ilu0.setup(matrix)?;
                    pc_chain.add_preconditioner(Box::new(ilu0));
                },
                "chebyshev" => {
                    // Construct Chebyshev with default or configured parameters
                    let degree = options.and_then(|opts| opts.chebyshev_degree).unwrap_or(4);
                    let (lambda_min, lambda_max) = if let (Some(min), Some(max)) = 
                        (options.and_then(|opts| opts.chebyshev_lambda_min), options.and_then(|opts| opts.chebyshev_lambda_max)) {
                        (min, max)
                    } else {
                        use crate::preconditioner::chebyshev::ChebyshevPre;
                        ChebyshevPre::estimate_eigenvalue_bounds(matrix, 50, 1e-6)
                    };
                    
                    let mut cheb = crate::preconditioner::chebyshev::ChebyshevPre::new(
                        matrix.clone(), degree, lambda_min, lambda_max
                    );
                    cheb.setup(matrix)?;
                    pc_chain.add_preconditioner(Box::new(cheb));
                },
                "amg" => {
                    // Construct AMG with default or configured parameters
                    let max_levels = options.and_then(|opts| opts.amg_levels).unwrap_or(10);
                    let threshold = options.and_then(|opts| opts.amg_strength_threshold).unwrap_or(0.25);
                    let nu_pre = options.and_then(|opts| opts.amg_nu_pre).unwrap_or(1);
                    let nu_post = options.and_then(|opts| opts.amg_nu_post).unwrap_or(1);
                    
                    let mut amg = crate::preconditioner::amg::AMG::with_smoothing(
                        matrix, max_levels, threshold, nu_pre, nu_post
                    );
                    amg.setup(matrix)?;
                    pc_chain.add_preconditioner(Box::new(amg));
                },
                "none" => {
                    let mut noop = NoOpPreconditioner;
                    noop.setup(matrix)?;
                    pc_chain.add_preconditioner(Box::new(noop));
                },
                other => {
                    return Err(KError::SolveError(format!("Unknown preconditioner in chain: {}", other)));
                }
            }
        }
        
        #[cfg(feature = "logging")]
        log::trace!("Successfully created PC chain with {} preconditioners", pc_chain.len());
        Ok(Box::new(pc_chain))
    }
}

/// Represents different types of preconditioners available in kryst.
///
/// Each variant includes specific configuration parameters for that preconditioner type.
/// This allows runtime selection and configuration of preconditioners, making it easy
/// to experiment with different preconditioning strategies for various linear systems.
///
/// # Example
/// ```rust,ignore
/// use kryst::context::pc_context::PC;
///
/// // Simple diagonal (Jacobi) preconditioner
/// let pc = PC::Jacobi;
///
/// // ILU(0) with zero fill-in
/// let pc = PC::Ilu0;
///
/// // ILUT with specific fill level and drop tolerance
/// let pc = PC::Ilut { fill: 10, droptol: 1e-4 };
///
/// // Chebyshev with custom polynomial degree and spectrum bounds
/// let pc = PC::Chebyshev { 
///     degree: 3, 
///     emin: Some(0.1), 
///     emax: Some(10.0) 
/// };
/// ```
#[derive(Debug, Clone)]
pub enum PC<T> {
    /// Jacobi (diagonal scaling) preconditioner.
    Jacobi,
    /// Symmetric Successive Over-Relaxation preconditioner.
    Ssor,
    /// Incomplete LU factorization with zero fill-in.
    Ilu0,
    /// Incomplete LU factorization with partial pivoting.
    ///
    /// - `fill`: Maximum number of additional non-zeros per row.
    Ilup { fill: usize },
    /// Incomplete LU factorization with threshold and pivoting.
    ///
    /// - `fill`: Maximum number of additional non-zeros per row.
    /// - `droptol`: Drop tolerance for discarding small elements.
    Ilut { fill: usize, droptol: T },
    /// Chebyshev polynomial preconditioner.
    ///
    /// - `degree`: Degree of the Chebyshev polynomial.
    /// - `emin`: Optional lower bound on the spectrum.
    /// - `emax`: Optional upper bound on the spectrum.
    Chebyshev { degree: usize, emin: Option<T>, emax: Option<T> },
    /// Approximate inverse preconditioner.
    ///
    /// - `pattern`: Sparsity pattern for the approximate inverse.
    /// - `tol`: Convergence tolerance for the iterative construction.
    /// - `max_iter`: Maximum number of iterations for the construction algorithm.
    ApproxInv { pattern: SparsityPattern, tol: T, max_iter: usize },
    /// Block Jacobi preconditioner.
    ///
    /// - `blocks`: List of index blocks, each block is a list of row/column indices.
    BlockJacobi { blocks: Vec<Vec<usize>> },
    /// Multicolor preconditioner.
    ///
    /// - `colors`: Color assignment for each row/column (for parallelization).
    Multicolor { colors: Vec<usize> },
    /// Algebraic Multigrid (AMG) preconditioner.
    AMG,
    /// Additive Schwarz domain decomposition preconditioner.
    AdditiveSchwarz,
}

/// Sparsity pattern for approximate inverse preconditioners.
///
/// Used to control the nonzero structure of the approximate inverse. The `Auto` variant
/// lets the library choose a pattern automatically, while `Manual` allows the user to
/// specify the sparsity structure explicitly.
#[derive(Debug, Clone)]
pub enum SparsityPattern {
    /// Let the library choose the sparsity pattern automatically.
    Auto,
    /// User-specified sparsity pattern.
    ///
    /// Each inner vector contains the column indices for the corresponding row.
    Manual(Vec<Vec<usize>>), // for each row, the list of column indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pc_type_from_str() {
        assert_eq!(PcType::from_str("jacobi").unwrap(), PcType::Jacobi);
        assert_eq!(PcType::from_str("ilu0").unwrap(), PcType::Ilu0);
        assert_eq!(PcType::from_str("none").unwrap(), PcType::None);
        assert_eq!(PcType::from_str("chebyshev").unwrap(), PcType::Chebyshev);
        assert_eq!(PcType::from_str("amg").unwrap(), PcType::Amg);
        assert_eq!(PcType::from_str("lu").unwrap(), PcType::Lu);
        assert_eq!(PcType::from_str("qr").unwrap(), PcType::Qr);
        
        // Test case insensitive
        assert_eq!(PcType::from_str("JACOBI").unwrap(), PcType::Jacobi);
        assert_eq!(PcType::from_str("ILU0").unwrap(), PcType::Ilu0);
        
        // Test aliases
        assert_eq!(PcType::from_str("block_jacobi").unwrap(), PcType::BlockJacobi);
        assert_eq!(PcType::from_str("bjacobi").unwrap(), PcType::BlockJacobi);
        assert_eq!(PcType::from_str("approx_inverse").unwrap(), PcType::ApproxInverse);
        
        // Test invalid
        assert!(PcType::from_str("invalid").is_err());
    }

    #[test]
    fn test_pc_factory_create_preconditioner() {
        // Test matrix-independent preconditioners
        assert!(PcFactory::create_preconditioner(PcType::Jacobi, None).is_ok());
        assert!(PcFactory::create_preconditioner(PcType::Ilu0, None).is_ok());
        assert!(PcFactory::create_preconditioner(PcType::None, None).is_ok());
        assert!(PcFactory::create_preconditioner(PcType::Lu, None).is_ok());
        assert!(PcFactory::create_preconditioner(PcType::Qr, None).is_ok());
        
        // Test matrix-dependent preconditioners fail without matrix
        assert!(PcFactory::create_preconditioner(PcType::Chebyshev, None).is_err());
        assert!(PcFactory::create_preconditioner(PcType::Amg, None).is_err());
        assert!(PcFactory::create_preconditioner(PcType::Asm, None).is_err());
        
        // Test unimplemented preconditioners
        assert!(PcFactory::create_preconditioner(PcType::Ilu, None).is_err());
        assert!(PcFactory::create_preconditioner(PcType::Ilut, None).is_err());
    }

    #[test]
    fn test_pc_factory_create_deferred_pc() {
        // Test matrix-dependent preconditioners
        assert!(PcFactory::create_deferred_pc(PcType::Chebyshev, None).is_ok());
        assert!(PcFactory::create_deferred_pc(PcType::Amg, None).is_ok());
        assert!(PcFactory::create_deferred_pc(PcType::Asm, None).is_ok());
        
        // Test matrix-independent preconditioners fail
        assert!(PcFactory::create_deferred_pc(PcType::Jacobi, None).is_err());
        assert!(PcFactory::create_deferred_pc(PcType::Ilu0, None).is_err());
        assert!(PcFactory::create_deferred_pc(PcType::None, None).is_err());
    }

    #[test]
    fn test_noop_preconditioner() {
        let mut pc = NoOpPreconditioner;
        let dummy_matrix = Mat::<f64>::zeros(3, 3);
        assert!(pc.setup(&dummy_matrix).is_ok());
        
        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0, 0.0, 0.0];
        assert!(pc.apply(PcSide::Left, &r, &mut z).is_ok());
        assert_eq!(z, r);
    }

    #[test]
    fn test_pc_jacobi() {
        let pc: PC<f64> = PC::Jacobi;
        match pc {
            PC::Jacobi => assert!(true),
            _ => panic!("Expected Jacobi variant"),
        }
    }

    #[test]
    fn test_pc_ssor() {
        let pc: PC<f64> = PC::Ssor;
        match pc {
            PC::Ssor => assert!(true),
            _ => panic!("Expected Ssor variant"),
        }
    }

    #[test]
    fn test_pc_ilu0() {
        let pc: PC<f64> = PC::Ilu0;
        match pc {
            PC::Ilu0 => assert!(true),
            _ => panic!("Expected Ilu0 variant"),
        }
    }

    #[test]
    fn test_pc_ilup() {
        let pc: PC<f64> = PC::Ilup { fill: 5 };
        match pc {
            PC::Ilup { fill } => assert_eq!(fill, 5),
            _ => panic!("Expected Ilup variant"),
        }
    }

    #[test]
    fn test_pc_ilut() {
        let pc: PC<f64> = PC::Ilut { fill: 10, droptol: 1e-3 };
        match pc {
            PC::Ilut { fill, droptol } => {
                assert_eq!(fill, 10);
                assert_eq!(droptol, 1e-3);
            },
            _ => panic!("Expected Ilut variant"),
        }
    }

    #[test]
    fn test_pc_chebyshev() {
        let pc: PC<f64> = PC::Chebyshev { 
            degree: 3, 
            emin: Some(0.1), 
            emax: Some(10.0) 
        };
        match pc {
            PC::Chebyshev { degree, emin, emax } => {
                assert_eq!(degree, 3);
                assert_eq!(emin, Some(0.1));
                assert_eq!(emax, Some(10.0));
            },
            _ => panic!("Expected Chebyshev variant"),
        }
    }

    #[test]
    fn test_pc_chebyshev_no_bounds() {
        let pc: PC<f64> = PC::Chebyshev { 
            degree: 5, 
            emin: None, 
            emax: None 
        };
        match pc {
            PC::Chebyshev { degree, emin, emax } => {
                assert_eq!(degree, 5);
                assert_eq!(emin, None);
                assert_eq!(emax, None);
            },
            _ => panic!("Expected Chebyshev variant"),
        }
    }

    #[test]
    fn test_pc_approx_inv() {
        let pattern = SparsityPattern::Auto;
        let pc: PC<f64> = PC::ApproxInv { 
            pattern, 
            tol: 1e-6, 
            max_iter: 100 
        };
        match pc {
            PC::ApproxInv { tol, max_iter, .. } => {
                assert_eq!(tol, 1e-6);
                assert_eq!(max_iter, 100);
            },
            _ => panic!("Expected ApproxInv variant"),
        }
    }

    #[test]
    fn test_pc_block_jacobi() {
        let blocks = vec![vec![0, 1], vec![2, 3], vec![4]];
        let pc: PC<f64> = PC::BlockJacobi { blocks: blocks.clone() };
        match pc {
            PC::BlockJacobi { blocks: b } => assert_eq!(b, blocks),
            _ => panic!("Expected BlockJacobi variant"),
        }
    }

    #[test]
    fn test_pc_multicolor() {
        let colors = vec![0, 1, 0, 2, 1];
        let pc: PC<f64> = PC::Multicolor { colors: colors.clone() };
        match pc {
            PC::Multicolor { colors: c } => assert_eq!(c, colors),
            _ => panic!("Expected Multicolor variant"),
        }
    }

    #[test]
    fn test_pc_amg() {
        let pc: PC<f64> = PC::AMG;
        match pc {
            PC::AMG => assert!(true),
            _ => panic!("Expected AMG variant"),
        }
    }

    #[test]
    fn test_pc_additive_schwarz() {
        let pc: PC<f64> = PC::AdditiveSchwarz;
        match pc {
            PC::AdditiveSchwarz => assert!(true),
            _ => panic!("Expected AdditiveSchwarz variant"),
        }
    }

    #[test]
    fn test_sparsity_pattern_auto() {
        let pattern = SparsityPattern::Auto;
        match pattern {
            SparsityPattern::Auto => assert!(true),
            _ => panic!("Expected Auto variant"),
        }
    }

    #[test]
    fn test_sparsity_pattern_manual() {
        let structure = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let pattern = SparsityPattern::Manual(structure.clone());
        match pattern {
            SparsityPattern::Manual(s) => assert_eq!(s, structure),
            _ => panic!("Expected Manual variant"),
        }
    }

    #[test]
    fn test_pc_clone() {
        let pc1: PC<f64> = PC::Ilut { fill: 5, droptol: 1e-4 };
        let pc2 = pc1.clone();
        
        match (pc1, pc2) {
            (PC::Ilut { fill: f1, droptol: d1 }, PC::Ilut { fill: f2, droptol: d2 }) => {
                assert_eq!(f1, f2);
                assert_eq!(d1, d2);
            },
            _ => panic!("Clone should preserve variant and values"),
        }
    }

    #[test]
    fn test_sparsity_pattern_clone() {
        let pattern1 = SparsityPattern::Manual(vec![vec![0, 1], vec![1]]);
        let pattern2 = pattern1.clone();
        
        match (pattern1, pattern2) {
            (SparsityPattern::Manual(s1), SparsityPattern::Manual(s2)) => {
                assert_eq!(s1, s2);
            },
            _ => panic!("Clone should preserve sparsity pattern"),
        }
    }

    #[test]
    fn test_pc_debug() {
        let pc: PC<f64> = PC::Jacobi;
        let debug_str = format!("{:?}", pc);
        assert!(debug_str.contains("Jacobi"));

        let pc2: PC<f64> = PC::Ilut { fill: 3, droptol: 1e-5 };
        let debug_str2 = format!("{:?}", pc2);
        assert!(debug_str2.contains("Ilut"));
        assert!(debug_str2.contains("3"));
    }

    #[test]
    fn test_sparsity_pattern_debug() {
        let pattern = SparsityPattern::Auto;
        let debug_str = format!("{:?}", pattern);
        assert!(debug_str.contains("Auto"));

        let pattern2 = SparsityPattern::Manual(vec![vec![0]]);
        let debug_str2 = format!("{:?}", pattern2);
        assert!(debug_str2.contains("Manual"));
    }

    #[test]
    fn test_pc_with_different_types() {
        // Test with f32
        let pc_f32: PC<f32> = PC::Ilut { fill: 2, droptol: 1e-3f32 };
        match pc_f32 {
            PC::Ilut { fill, droptol } => {
                assert_eq!(fill, 2);
                assert_eq!(droptol, 1e-3f32);
            },
            _ => panic!("Expected Ilut variant for f32"),
        }

        // Test with f64
        let pc_f64: PC<f64> = PC::Chebyshev { 
            degree: 4, 
            emin: Some(0.5), 
            emax: Some(5.0) 
        };
        match pc_f64 {
            PC::Chebyshev { degree, emin, emax } => {
                assert_eq!(degree, 4);
                assert_eq!(emin, Some(0.5));
                assert_eq!(emax, Some(5.0));
            },
            _ => panic!("Expected Chebyshev variant for f64"),
        }
    }

    #[test]
    fn test_complex_pc_configurations() {
        // Test ApproxInv with manual sparsity pattern
        let manual_pattern = SparsityPattern::Manual(vec![
            vec![0, 1, 2],
            vec![0, 1],
            vec![1, 2],
        ]);
        let pc: PC<f64> = PC::ApproxInv { 
            pattern: manual_pattern, 
            tol: 1e-8, 
            max_iter: 50 
        };
        
        match pc {
            PC::ApproxInv { pattern, tol, max_iter } => {
                assert_eq!(tol, 1e-8);
                assert_eq!(max_iter, 50);
                match pattern {
                    SparsityPattern::Manual(s) => {
                        assert_eq!(s.len(), 3);
                        assert_eq!(s[0], vec![0, 1, 2]);
                    },
                    _ => panic!("Expected manual pattern"),
                }
            },
            _ => panic!("Expected ApproxInv variant"),
        }
    }

    #[test]
    fn test_empty_configurations() {
        // Test empty blocks for BlockJacobi
        let pc: PC<f64> = PC::BlockJacobi { blocks: vec![] };
        match pc {
            PC::BlockJacobi { blocks } => assert!(blocks.is_empty()),
            _ => panic!("Expected BlockJacobi variant"),
        }

        // Test empty colors for Multicolor
        let pc2: PC<f64> = PC::Multicolor { colors: vec![] };
        match pc2 {
            PC::Multicolor { colors } => assert!(colors.is_empty()),
            _ => panic!("Expected Multicolor variant"),
        }

        // Test empty sparsity pattern
        let pattern = SparsityPattern::Manual(vec![]);
        match pattern {
            SparsityPattern::Manual(s) => assert!(s.is_empty()),
            _ => panic!("Expected Manual pattern"),
        }
    }
}
