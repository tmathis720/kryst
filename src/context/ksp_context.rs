//! Unified KSP Context for runtime solver selection and configuration.
//!
//! This module provides a PETSc-style unified interface for Krylov subspace methods.
//! The `KspContext` allows runtime selection of solver types, preconditioners, and
//! tolerances using enum-based configuration.
//!
//! # Usage
//!
//! ```rust,ignore
//! use kryst::context::ksp_context::{KspContext, SolverType};
//! use kryst::context::pc_context::PcType;
//!
//! let mut ksp = KspContext::new();
//! ksp.set_type(SolverType::Cg)?
//!    .set_pc_type(PcType::Jacobi)?
//!    .set_tolerances(1e-6, 1e-12, 1e3, 1000);
//! let stats = ksp.solve(&A, &b, &mut x)?;
//! ```
//!
//! # References
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM.
//! - PETSc Documentation: https://petsc.org/release/docs/manualpages/KSP/

use std::str::FromStr;
use faer::Mat;
use crate::solver::{LinearSolver, CgSolver, GmresSolver, FgmresSolver, BiCgStabSolver, MinresSolver, 
                   TfqmrSolver, CgnrSolver, CgsSolver, QmrSolver, LuSolver, PcgSolver};
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::{SolveStats, ConvergedReason};
use crate::utils::profiling::StageGuard;
use crate::utils::reordering::{preprocess_matrix, ReorderingMethod, ScalingMethod, MatrixPreprocessing};
use crate::error::KError;
use crate::context::pc_context::{PcType, DeferredPcInfo, PcFactory};

#[cfg(feature = "logging")]
use log::trace;

/// Workspace for Krylov solver operations to enable buffer reuse.
///
/// This struct contains all the working arrays needed by Krylov solvers,
/// allocated once during setup and reused across multiple solves.
#[derive(Debug)]
pub struct Workspace {
    /// Krylov basis vectors Q: size (restart+1) × n for GMRES
    pub q: Vec<Vec<f64>>,
    /// Preconditioned basis vectors Z: size restart × n for FGMRES  
    pub z: Vec<Vec<f64>>,
    /// Hessenberg matrix H: (restart+1) × restart for GMRES
    pub h: Vec<Vec<f64>>,
    /// Givens rotation cosines for GMRES
    pub cs: Vec<f64>,
    /// Givens rotation sines for GMRES
    pub sn: Vec<f64>,
    /// Residual vector g in least-squares subproblem
    pub g: Vec<f64>,
    /// Temporary vector 1 for general use
    pub tmp1: Vec<f64>,
    /// Temporary vector 2 for general use  
    pub tmp2: Vec<f64>,
    /// Temporary vector 3 for general use
    pub tmp3: Vec<f64>,
    /// Temporary vector 4 for general use
    pub tmp4: Vec<f64>,
    /// Vector dimension (for validation)
    pub n: usize,
    /// Current restart parameter (for validation)
    pub restart: usize,
}

/// All supported Krylov solver types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    /// Conjugate Gradient (CG) method (for SPD matrices)
    Cg,
    /// Preconditioned Conjugate Gradient (PCG)
    Pcg,
    /// Generalized Minimal Residual (GMRES)
    Gmres,
    /// Flexible Generalized Minimal Residual (FGMRES)
    Fgmres,
    /// BiConjugate Gradient Stabilized (BiCGStab)
    BiCgStab,
    /// Conjugate Gradient Squared (CGS)
    Cgs,
    /// Quasi-Minimal Residual (QMR)
    Qmr,
    /// Transpose-Free QMR (TFQMR)
    Tfqmr,
    /// Minimal Residual (MINRES)
    Minres,
    /// Conjugate Gradient on the Normal Equations (CGNR)
    Cgnr,
    /// Direct solver (LU factorization)
    Preonly,
}

impl FromStr for SolverType {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cg" => Ok(SolverType::Cg),
            "pcg" => Ok(SolverType::Pcg),
            "gmres" => Ok(SolverType::Gmres),
            "fgmres" => Ok(SolverType::Fgmres),
            "bicgstab" => Ok(SolverType::BiCgStab),
            "cgs" => Ok(SolverType::Cgs),
            "qmr" => Ok(SolverType::Qmr),
            "tfqmr" => Ok(SolverType::Tfqmr),
            "minres" => Ok(SolverType::Minres),
            "cgnr" => Ok(SolverType::Cgnr),
            "preonly" => Ok(SolverType::Preonly),
            _ => Err(KError::UnrecognizedSolverType(s.to_string())),
        }
    }
}

/// Unified context for Krylov subspace solver configuration and execution.
///
/// This struct provides a PETSc-like interface for runtime selection of solvers
/// and preconditioners, along with tolerance settings. It works with dense matrices
/// (`faer::Mat<f64>`) and vectors (`Vec<f64>`).
///
/// The context follows a two-phase API:
/// 1. `setup()` - Prepare preconditioner and allocate workspaces
/// 2. `solve()` - Solve linear systems efficiently using cached data
pub struct KspContext {
    solver: Option<Box<dyn LinearSolver<Mat<f64>, Vec<f64>, Scalar = f64, Error = KError>>>,
    pc: Option<Box<dyn Preconditioner<Mat<f64>, Vec<f64>>>>,
    /// Deferred preconditioner construction info (for matrix-dependent PCs)
    deferred_pc: Option<DeferredPcInfo>,
    /// Current solver type (for PREONLY handling)
    solver_type: Option<SolverType>,
    /// Current preconditioner type (for PREONLY handling)
    pc_type: Option<PcType>,
    /// Preconditioner side (Left/Right/Symmetric)
    pub pc_side: crate::preconditioner::PcSide,
    /// Preconditioner options for advanced configuration
    pc_options: Option<crate::config::options::PcOptions>,
    /// Matrix preprocessing info (reordering + scaling)
    preprocessing: Option<MatrixPreprocessing>,
    /// Relative tolerance for convergence
    pub rtol: f64,
    /// Absolute tolerance for convergence  
    pub atol: f64,
    /// Divergence tolerance
    pub dtol: f64,
    /// Maximum number of iterations
    pub maxits: usize,
    /// Restart parameter for GMRES-type solvers
    pub restart: usize,
    /// Cached workspace for efficient repeated solves (None until setup)
    work: Option<Workspace>,
    /// Flag indicating if setup has been called
    setup_called: bool,
    /// Optional custom convergence test function
    custom_conv: Option<Box<dyn Fn(usize, f64, f64) -> ConvergedReason>>,
    /// Communicator for parallel operations  
    pub comm: Option<crate::parallel::UniverseComm>,
    /// Registered iteration monitors (callbacks for each iteration)
    monitors: Vec<Box<dyn Fn(usize, f64) + Send + Sync>>,
}

impl Workspace {
    /// Create a new workspace for the given problem dimensions.
    ///
    /// # Arguments
    /// * `n` - Vector dimension
    /// * `restart` - Restart parameter for GMRES-type solvers
    fn new(n: usize, restart: usize) -> Self {
        let maxv = restart + 1; // Maximum number of Arnoldi vectors
        
        // Allocate Krylov basis vectors Q
        let mut q = Vec::with_capacity(maxv);
        for _ in 0..maxv {
            q.push(vec![0.0; n]);
        }
        
        // Allocate preconditioned basis vectors Z  
        let mut z = Vec::with_capacity(restart);
        for _ in 0..restart {
            z.push(vec![0.0; n]);
        }
        
        // Allocate Hessenberg matrix H
        let mut h = Vec::with_capacity(maxv);
        for _ in 0..maxv {
            h.push(vec![0.0; restart]);
        }
        
        Self {
            q,
            z,
            h,
            cs: vec![0.0; restart],
            sn: vec![0.0; restart],
            g: vec![0.0; maxv],
            tmp1: vec![0.0; n],
            tmp2: vec![0.0; n],
            tmp3: vec![0.0; n],
            tmp4: vec![0.0; n],
            n,
            restart,
        }
    }
    
    /// Check if workspace is compatible with given dimensions.
    fn is_compatible(&self, n: usize, restart: usize) -> bool {
        self.n == n && self.restart == restart
    }
}

impl KspContext {
    /// Create a new KspContext with default settings.
    pub fn new() -> Self {
        Self {
            solver: None,
            pc: None,
            deferred_pc: None,
            solver_type: None,
            pc_type: None,
            pc_side: crate::preconditioner::PcSide::Left,
            pc_options: None,
            preprocessing: None,
            rtol: 1e-6,
            atol: 1e-12,
            dtol: 1e3,
            maxits: 1000,
            restart: 50,
            work: None,
            setup_called: false,
            custom_conv: None,
            comm: None,
            monitors: Vec::new(),
        }
    }

    /// Set the solver type and create the appropriate solver instance.
    ///
    /// # Arguments
    /// * `solver_type` - The type of solver to use
    ///
    /// # Returns
    /// * `Ok(&mut Self)` for method chaining on success
    /// * `Err(KError)` if the solver type is not supported
    pub fn set_type(&mut self, solver_type: SolverType) -> Result<&mut Self, KError> {
        // Store the solver type for PREONLY handling
        self.solver_type = Some(solver_type);
        
        match solver_type {
            SolverType::Preonly => {
                // For PREONLY, we don't need a Krylov solver - the PC will do direct solve
                self.solver = None;
            },
            _ => {
                // Create the appropriate Krylov solver
                let solver: Box<dyn LinearSolver<Mat<f64>, Vec<f64>, Scalar = f64, Error = KError>> = match solver_type {
                    SolverType::Cg => {
                        let cg = CgSolver::new(self.rtol, self.maxits);
                        Box::new(cg)
                    },
                    SolverType::Pcg => {
                        let pcg = PcgSolver::new(self.rtol, self.maxits);
                        Box::new(pcg)
                    },
                    SolverType::Gmres => {
                        let gmres = GmresSolver::new(self.restart, self.rtol, self.maxits);
                        Box::new(gmres)
                    },
                    SolverType::Fgmres => {
                        let fgmres = FgmresSolver::new(self.rtol, self.maxits, self.restart);
                        Box::new(fgmres)
                    },
                    SolverType::BiCgStab => {
                        let bicgstab = BiCgStabSolver::new(self.rtol, self.maxits);
                        Box::new(bicgstab)
                    },
                    SolverType::Cgs => {
                        let cgs = CgsSolver::new(self.rtol, self.maxits);
                        Box::new(cgs)
                    },
                    SolverType::Qmr => {
                        let qmr = QmrSolver::new(self.rtol, self.maxits);
                        Box::new(qmr)
                    },
                    SolverType::Tfqmr => {
                        let tfqmr = TfqmrSolver::new(self.rtol, self.maxits);
                        Box::new(tfqmr)
                    },
                    SolverType::Minres => {
                        let minres = MinresSolver::new(self.rtol, self.maxits);
                        Box::new(minres)
                    },
                    SolverType::Cgnr => {
                        let cgnr = CgnrSolver::new(self.rtol, self.maxits);
                        Box::new(cgnr)
                    },
                    SolverType::Preonly => unreachable!(), // Handled above
                };
                self.solver = Some(solver);
            }
        }
        Ok(self)
    }

    /// Set the solver type from a string.
    ///
    /// # Arguments
    /// * `solver_name` - String name of the solver (e.g., "cg", "gmres")
    ///
    /// # Returns
    /// * `Ok(&mut Self)` for method chaining on success
    /// * `Err(KError)` if the solver name is not recognized
    pub fn set_type_from_str(&mut self, solver_name: &str) -> Result<&mut Self, KError> {
        let solver_type = SolverType::from_str(solver_name)?;
        self.set_type(solver_type)
    }

    /// Set the solver type from a string.
    ///
    /// # Arguments
    /// * `solver_name` - String name of the solver (e.g., "cg", "gmres")
    ///
    /// Set the preconditioner type and create the appropriate preconditioner instance.
    ///
    /// # Arguments
    /// * `pc_type` - The type of preconditioner to use
    ///
    /// # Returns
    /// * `Ok(&mut Self)` for method chaining on success
    /// * `Err(KError)` if the preconditioner type is not supported
    pub fn set_pc_type(&mut self, pc_type: PcType) -> Result<&mut Self, KError> {
        // Clear any existing PC state
        self.pc = None;
        self.deferred_pc = None;
        self.pc_type = Some(pc_type);

        match pc_type {
            // Matrix-independent preconditioners - construct immediately
            PcType::Jacobi | PcType::Ilu0 | PcType::None | PcType::Lu | PcType::Qr | PcType::Ilutp | PcType::SuperLuDist => {
                self.pc = Some(PcFactory::create_preconditioner(pc_type, self.pc_options.as_ref())?);
            },
            
            // Matrix-dependent preconditioners - defer until setup()
            PcType::Asm | PcType::Chebyshev | PcType::Amg => {
                self.deferred_pc = Some(PcFactory::create_deferred_pc(pc_type, self.pc_options.clone())?);
            },
            
            // Not yet implemented preconditioners
            PcType::Ilu | PcType::Ilut | PcType::Ilup | PcType::BlockJacobi | 
            PcType::Sor | PcType::ApproxInverse => {
                return Err(KError::UnrecognizedPcType(format!("{:?} preconditioner not yet implemented", pc_type)));
            }
        }
        
        Ok(self)
    }

    /// Set the preconditioner type from a string.
    ///
    /// # Arguments
    /// * `pc_name` - String name of the preconditioner (e.g., "jacobi", "ilu0")
    ///
    /// # Returns
    /// * `Ok(&mut Self)` for method chaining on success
    /// * `Err(KError)` if the preconditioner name is not recognized
    pub fn set_pc_type_from_str(&mut self, pc_name: &str) -> Result<&mut Self, KError> {
        let pc_type = PcType::from_str(pc_name)?;
        self.set_pc_type(pc_type)
    }

    /// Set preconditioner options for advanced configuration.
    ///
    /// # Arguments
    /// * `options` - Preconditioner options for ILUTP, reordering, etc.
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn set_pc_options(&mut self, options: crate::config::options::PcOptions) -> &mut Self {
        self.pc_options = Some(options);
        self
    }

    /// Set the preconditioner side (Left/Right/Symmetric).
    ///
    /// # Arguments
    /// * `side` - The preconditioner side to use
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn set_pc_side(&mut self, side: crate::preconditioner::PcSide) -> &mut Self {
        self.pc_side = side;
        self
    }

    /// Set the preconditioner side from a string.
    ///
    /// # Arguments
    /// * `side_name` - String name of the side ("left", "right", "symmetric")
    ///
    /// # Returns
    /// * `Ok(&mut Self)` for method chaining on success
    /// * `Err(KError)` if the side name is not recognized
    pub fn set_pc_side_from_str(&mut self, side_name: &str) -> Result<&mut Self, KError> {
        let side = crate::preconditioner::PcSide::from_str(side_name)?;
        self.set_pc_side(side);
        Ok(self)
    }

    /// Set convergence tolerances and iteration limits.
    ///
    /// # Arguments
    /// * `rtol` - Relative tolerance
    /// * `atol` - Absolute tolerance  
    /// * `dtol` - Divergence tolerance
    /// * `maxits` - Maximum number of iterations
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn set_tolerances(&mut self, rtol: f64, atol: f64, dtol: f64, maxits: usize) -> &mut Self {
        self.rtol = rtol;
        self.atol = atol;
        self.dtol = dtol;
        self.maxits = maxits;
        self
    }

    /// Set the restart parameter for GMRES-type solvers.
    ///
    /// # Arguments
    /// * `restart` - Restart parameter (typically 20-100)
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn set_restart(&mut self, restart: usize) -> &mut Self {
        self.restart = restart;
        // Invalidate workspace if restart parameter changed
        self.work = None;
        self.setup_called = false;
        self
    }

    /// Set a custom convergence test function.
    ///
    /// This allows users to define their own convergence criteria beyond the standard
    /// relative/absolute/divergence tolerance tests. The custom function will be called
    /// with (iteration_count, residual_norm, rhs_norm) and should return a ConvergedReason.
    ///
    /// # Arguments
    /// * `f` - Custom convergence test function with signature `Fn(usize, f64, f64) -> ConvergedReason`
    ///   - First argument: current iteration count
    ///   - Second argument: current residual norm ‖r‖  
    ///   - Third argument: right-hand side norm ‖b‖
    ///   - Return: ConvergedReason indicating whether to continue, converge, or diverge
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    ///
    /// # Example
    /// ```rust,ignore
    /// use kryst::context::ksp_context::{KspContext, ConvergedReason};
    /// 
    /// let mut ksp = KspContext::new();
    /// ksp.set_convergence_test(|iters, rnorm, bnorm| {
    ///     if rnorm / bnorm < 1e-3 {
    ///         ConvergedReason::ConvergedRtol
    ///     } else if iters > 10 {
    ///         ConvergedReason::DivergedMaxIts  
    ///     } else {
    ///         ConvergedReason::Continued
    ///     }
    /// });
    /// ```
    pub fn set_convergence_test<F>(&mut self, f: F) -> &mut Self
    where 
        F: Fn(usize, f64, f64) -> ConvergedReason + 'static
    {
        self.custom_conv = Some(Box::new(f));
        self
    }

    /// Clear the custom convergence test and revert to default convergence criteria.
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn clear_convergence_test(&mut self) -> &mut Self {
        self.custom_conv = None;
        self
    }

    /// Check if a custom convergence test has been set.
    ///
    /// # Returns
    /// * `true` if a custom convergence test is active, `false` otherwise
    pub fn has_custom_convergence_test(&self) -> bool {
        self.custom_conv.is_some()
    }

    /// Register a callback to be invoked at each iteration.
    ///
    /// The callback will be called with the current iteration number (0-based)
    /// and the residual norm at that iteration.
    ///
    /// # Arguments
    /// * `f` - Callback function with signature `Fn(usize, f64)`
    ///   - First argument: current iteration number (starting from 0)
    ///   - Second argument: current residual norm ‖r‖
    ///
    /// # Example
    /// ```rust
    /// use kryst::context::ksp_context::KspContext;
    /// 
    /// let mut ksp = KspContext::new();
    /// ksp.add_monitor(|iter, resid| {
    ///     println!("Iteration {}: residual = {:.3e}", iter, resid);
    /// });
    /// ```
    pub fn add_monitor<F>(&mut self, f: F)
    where
        F: Fn(usize, f64) + Send + Sync + 'static,
    {
        self.monitors.push(Box::new(f));
    }

    /// Drop all registered monitors.
    ///
    /// This clears all iteration callbacks that were previously registered
    /// with `add_monitor()`.
    pub fn clear_monitors(&mut self) {
        self.monitors.clear();
    }

    /// Get the number of registered monitors.
    ///
    /// # Returns
    /// * The number of active iteration monitors
    pub fn num_monitors(&self) -> usize {
        self.monitors.len()
    }

    /// Invoke all registered monitors with the current iteration data.
    ///
    /// This method is called internally by solvers at each iteration.
    /// 
    /// # Arguments
    /// * `iteration` - Current iteration number (0-based)
    /// * `residual_norm` - Current residual norm
    pub fn invoke_monitors(&self, iteration: usize, residual_norm: f64) {
        for monitor in &self.monitors {
            monitor(iteration, residual_norm);
        }
    }

    /// Explicit setup phase: prepare preconditioner and allocate workspaces.
    ///
    /// This method should be called once before solving, or after changing 
    /// solver parameters. It:
    /// 1. Sets up the preconditioner with the given matrix
    /// 2. Allocates all workspace arrays for efficient repeated solves
    ///
    /// # Arguments
    /// * `a` - System matrix used for preconditioner setup
    /// * `n` - Vector dimension (length of solution vectors)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(KError)` if setup fails
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut ksp = KspContext::new();
    /// ksp.set_type(SolverType::Gmres)?;
    /// ksp.setup(&A, n)?;  // Setup once
    /// ksp.solve(&A, &b1, &mut x1)?;  // Fast solve
    /// ksp.solve(&A, &b2, &mut x2)?;  // Reuses workspace
    /// ```
    pub fn setup(&mut self, a: &Mat<f64>, n: usize) -> Result<(), KError> {
        // Use default communicator (no parallelism) if none specified
        #[cfg(not(any(feature="mpi", feature="rayon")))]
        let default_comm = crate::parallel::UniverseComm::Serial;
        #[cfg(all(feature="rayon", not(feature="mpi")))]
        let default_comm = crate::parallel::UniverseComm::Rayon(crate::parallel::RayonComm::new());
        #[cfg(feature="mpi")]
        let default_comm = match crate::parallel::MpiComm::try_new() {
            Some(mpi_comm) => crate::parallel::UniverseComm::Mpi(mpi_comm),
            None => {
                // Fallback to Rayon if available, otherwise Serial
                #[cfg(feature="rayon")]
                {
                    crate::parallel::UniverseComm::Rayon(crate::parallel::RayonComm::new())
                }
                #[cfg(not(feature="rayon"))]
                {
                    crate::parallel::UniverseComm::Serial
                }
            }
        };
        
        self.setup_with_comm(a, n, default_comm)
    }

    /// Setup the KspContext with a matrix, problem size, and communicator for parallel operations.
    ///
    /// This prepares the preconditioner and allocates workspace for efficient repeated solves.
    /// The communicator will be used for parallel reductions in dot products and norms.
    ///
    /// # Arguments
    /// * `a` - The coefficient matrix
    /// * `n` - The problem size (number of unknowns)
    /// * `comm` - The communicator for parallel operations
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(KError)` if setup fails
    pub fn setup_with_comm(&mut self, a: &Mat<f64>, n: usize, comm: crate::parallel::UniverseComm) -> Result<(), KError> {
        let _setup_stage = StageGuard::new("KSPSetupWithComm");
        
        #[cfg(feature = "logging")]
        trace!("Setting up KSP context with {} unknowns", n);
        
        // Store the communicator
        self.comm = Some(comm);
        
        // Apply matrix preprocessing (reordering + scaling) if specified
        let processed_matrix = if let Some(ref pc_opts) = self.pc_options {
            let reorder_method = match pc_opts.reorder.as_deref() {
                Some("rcm") => ReorderingMethod::Rcm,
                Some("cuthill_mckee") => ReorderingMethod::CuthillMckee,
                Some("colamd") => ReorderingMethod::Colamd,
                Some("amd") => ReorderingMethod::Amd,
                Some("none") | None => ReorderingMethod::None,
                Some(other) => {
                    return Err(KError::SolveError(format!("Unknown reordering method: {}", other)));
                }
            };
            
            let scaling_method = match pc_opts.scaling.as_deref() {
                Some("diagonal") => ScalingMethod::Diagonal,
                Some("symmetric") => ScalingMethod::Symmetric,
                Some("none") | None => ScalingMethod::None,
                Some(other) => {
                    return Err(KError::SolveError(format!("Unknown scaling method: {}", other)));
                }
            };
            
            if reorder_method != ReorderingMethod::None || scaling_method != ScalingMethod::None {
                let _preprocessing_stage = StageGuard::new("MatrixPreprocessing");
                #[cfg(feature = "logging")]
                trace!("Applying matrix preprocessing: reorder={:?}, scaling={:?}", reorder_method, scaling_method);
                
                let (processed, preprocessing_info) = preprocess_matrix(a, reorder_method, scaling_method)?;
                self.preprocessing = Some(preprocessing_info);
                processed
            } else {
                self.preprocessing = Some(MatrixPreprocessing::identity(n));
                a.clone()
            }
        } else {
            self.preprocessing = Some(MatrixPreprocessing::identity(n));
            a.clone()
        };
        
        // Check for PC-chaining before deferred PC construction
        if let Some(ref pc_opts) = self.pc_options {
            if let Some(ref chain_str) = pc_opts.pc_chain {
                let _pc_chain_stage = StageGuard::new("PCChainConstruction");
                #[cfg(feature = "logging")]
                trace!("Constructing PC chain: {}", chain_str);
                
                // Use factory to create PC chain
                let pc_chain = PcFactory::create_pc_chain(chain_str, &processed_matrix, Some(pc_opts))?;
                self.pc = Some(pc_chain);
                
                // Skip regular preconditioner construction since we have a chain
                self.setup_called = true;
                return Ok(());
            }
        }
        
        // Handle deferred preconditioner construction (matrix-dependent PCs)
        if let Some(deferred_info) = self.deferred_pc.take() {
            let _pc_construction_stage = StageGuard::new("DeferredPCConstruction");
            #[cfg(feature = "logging")]
            trace!("Constructing deferred preconditioner: {:?}", deferred_info.pc_type);
            
            let pc = PcFactory::construct_deferred_preconditioner(deferred_info, &processed_matrix)?;
            self.pc = Some(pc);
        }
        
        // Setup preconditioner if present
        if let Some(ref mut pc) = self.pc {
            let _pc_setup_stage = StageGuard::new("PCSetup");
            #[cfg(feature = "logging")]
            trace!("Setting up preconditioner: {:?}", self.pc_type);
            pc.setup(&processed_matrix)?;
        }

        // Allocate or resize workspace if needed
        if let Some(ref work) = self.work {
            if work.is_compatible(n, self.restart) {
                // Workspace is already compatible, no need to reallocate
                #[cfg(feature = "logging")]
                trace!("Reusing compatible workspace");
                self.setup_called = true;
                return Ok(());
            }
        }

        // Allocate new workspace
        {
            let _workspace_stage = StageGuard::new("WorkspaceAllocation");
            #[cfg(feature = "logging")]
            trace!("Allocating new workspace for {} unknowns, restart={}", n, self.restart);
            self.work = Some(Workspace::new(n, self.restart));
            
            // Setup solver-specific workspace if needed
            if let Some(ref mut solver) = self.solver {
                if let Some(ref mut work) = self.work {
                    solver.setup_workspace(work);
                    #[cfg(feature = "logging")]
                    trace!("Setup solver workspace");
                }
            }
        }
        
        self.setup_called = true;
        #[cfg(feature = "logging")]
        trace!("KSP setup completed successfully");
        Ok(())
    }

    /// Check if setup has been called and workspace is ready.
    pub fn is_setup(&self) -> bool {
        self.setup_called && self.work.is_some()
    }

    /// Invalidate the setup, forcing re-setup on next solve.
    /// 
    /// Call this when the matrix structure or size changes.
    pub fn invalidate_setup(&mut self) {
        self.work = None;
        self.setup_called = false;
    }

    /// Get the matrix preprocessing information applied during setup.
    ///
    /// Returns `None` if no preprocessing was applied or setup hasn't been called.
    pub fn get_preprocessing(&self) -> Option<&MatrixPreprocessing> {
        self.preprocessing.as_ref()
    }

    /// Solve the linear system Ax = b using the configured solver and preconditioner.
    ///
    /// This unified method always uses workspace for efficiency and supports optional
    /// runtime-controlled monitoring and profiling. If setup has not been called, 
    /// it will be called automatically.
    ///
    /// # Arguments
    /// * `a` - System matrix
    /// * `b` - Right-hand side vector
    /// * `x` - Solution vector (will be overwritten with the result)
    ///
    /// # Returns
    /// * `Ok(SolveStats)` with convergence information on success
    /// * `Err(KError)` on failure or if solver/preconditioner not configured
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut ksp = KspContext::new();
    /// ksp.set_type(SolverType::Gmres)?.set_pc_type(PcType::Jacobi)?;
    /// 
    /// // Automatic setup on first solve
    /// let stats1 = ksp.solve(&A, &b1, &mut x1)?;
    /// 
    /// // Subsequent solves reuse workspace  
    /// let stats2 = ksp.solve(&A, &b2, &mut x2)?;
    /// ```
    pub fn solve(&mut self, a: &Mat<f64>, b: &Vec<f64>, x: &mut Vec<f64>) -> Result<SolveStats<f64>, KError> {
        let _solve_stage = StageGuard::new("KSPSolve");
        
        #[cfg(feature = "logging")]
        trace!("Starting KSP solve with {} unknowns", b.len());

        // Check if we're in PREONLY mode
        if let Some(SolverType::Preonly) = self.solver_type {
            let _preonly_stage = StageGuard::new("KSPSolvePreonly");
            
            // For PREONLY, bypass Krylov iteration and use the PC as a direct solver
            if self.pc.is_none() {
                return Err(KError::SolveError("No preconditioner configured for PREONLY. Call set_pc_type() first.".to_string()));
            }

            // Auto-setup preconditioner if needed
            if !self.setup_called {
                let _setup_stage = StageGuard::new("KSPSetup");
                self.setup(a, b.len())?;
            }

            // Apply direct solve based on preconditioner type
            self.apply_direct_solve(a, b, x)
        } else {
            // Standard Krylov solver path
            if self.solver.is_none() {
                return Err(KError::SolveError("No solver configured. Call set_type() first.".to_string()));
            }

            // Auto-setup if needed - this ensures workspace is always available
            let needs_setup = !self.setup_called || self.work.is_none();
            if needs_setup {
                let _setup_stage = StageGuard::new("KSPSetup");
                #[cfg(feature = "logging")]
                trace!("Setting up KSP solver and preconditioner");
                self.setup(a, b.len())?;
            }

            // Verify workspace compatibility
            if let Some(ref work) = self.work {
                if !work.is_compatible(b.len(), self.restart) {
                    return Err(KError::SolveError(
                        "Workspace incompatible with problem size. Call setup() or invalidate_setup().".to_string()
                    ));
                }
            } else {
                return Err(KError::SolveError(
                    "Workspace not available after setup. This is a bug.".to_string()
                ));
            }

            let _krylov_stage = StageGuard::new("KSPSolveKrylov");
            #[cfg(feature = "logging")]
            trace!("Starting Krylov iteration with solver type: {:?}", self.solver_type);

            // Prepare monitors for runtime-controlled monitoring
            let monitors_slice = if crate::utils::profiling::is_monitoring_enabled() && !self.monitors.is_empty() {
                Some(self.monitors.as_slice())
            } else {
                None
            };

            // Use the unified solve method with workspace and optional monitors
            let solver = self.solver.as_mut().unwrap(); // Safe because we checked above
            let comm = self.comm.as_ref().unwrap_or(&crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm));
            let stats = solver.solve(
                a, 
                self.pc.as_deref(), 
                b, 
                x, 
                comm,
                monitors_slice,
                Some(self.work.as_mut().unwrap()) // Always use workspace
            )?;
            
            #[cfg(feature = "logging")]
            trace!("Krylov solve completed: {} iterations, final residual: {:.3e}", 
                   stats.iterations, stats.final_residual);
            
            // Apply custom convergence test if provided
            if let Some(ref custom_test) = self.custom_conv {
                let reason = custom_test(stats.iterations, stats.final_residual, b.iter().map(|x| x*x).sum::<f64>().sqrt());
                // Create new stats with custom reason
                let custom_stats = SolveStats {
                    iterations: stats.iterations,
                    final_residual: stats.final_residual,
                    reason,
                };
                Ok(custom_stats)
            } else {
                Ok(stats)
            }
        }
    }

    /// Apply direct solve for PREONLY mode.
    /// 
    /// This method handles the direct solve by using the stored pc_type to
    /// determine which direct solver to use and calling it appropriately.
    fn apply_direct_solve(&mut self, a: &Mat<f64>, b: &Vec<f64>, x: &mut Vec<f64>) -> Result<SolveStats<f64>, KError> {
        match self.pc_type {
            Some(PcType::Lu) => {
                // For LU, use the LuSolver directly with the new API
                let mut lu_solver = LuSolver::new();
                let comm = self.comm.as_ref().unwrap_or(&crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm));
                lu_solver.solve(a, None, b, x, comm, None, self.work.as_mut())?;
                Ok(SolveStats {
                    iterations: 1,
                    final_residual: 0.0,
                    reason: ConvergedReason::ConvergedAtol,
                })
            },
            Some(PcType::Qr) => {
                // For QR, use the QrSolver directly with the new API
                let mut qr_solver = crate::solver::direct_lu::QrSolver::new();
                let comm = self.comm.as_ref().unwrap_or(&crate::parallel::UniverseComm::NoComm(crate::parallel::NoComm));
                qr_solver.solve(a, None, b, x, comm, None, self.work.as_mut())?;
                Ok(SolveStats {
                    iterations: 1,
                    final_residual: 0.0,
                    reason: ConvergedReason::ConvergedAtol,
                })
            },
            Some(other_pc) => {
                Err(KError::SolveError(
                    format!("PREONLY requires a direct solver preconditioner (Lu or Qr). Current preconditioner type {:?} is not supported for direct solve.", other_pc)
                ))
            },
            None => {
                Err(KError::SolveError(
                    "PREONLY requires a preconditioner to be set. Call set_pc_type() with Lu or Qr first.".to_string()
                ))
            }
        }
    }

    /// Configure the KSP context from parsed command-line options.
    ///
    /// This method applies options with the following precedence:
    /// 1. Existing context values (highest)
    /// 2. Command-line options
    /// 3. Default values (lowest)
    ///
    /// # Arguments
    /// * `opts` - Parsed KSP options from command-line arguments
    ///
    /// # Returns
    /// * `Ok(&mut Self)` for method chaining on success
    /// * `Err(KError)` if any option values are invalid
    ///
    /// # Example
    /// ```rust,ignore
    /// use kryst::context::ksp_context::KspContext;
    /// use kryst::config::options::KspOptions;
    ///
    /// let args = vec!["-ksp_type", "gmres", "-ksp_rtol", "1e-8"];
    /// let opts = KspOptions::from_args(&args)?;
    /// let mut ksp = KspContext::new();
    /// ksp.set_from_options(&opts)?;
    /// ```
    pub fn set_from_options(&mut self, opts: &crate::config::options::KspOptions) -> Result<&mut Self, KError> {
        // Apply solver type
        if let Some(ref solver_type_str) = opts.ksp_type {
            self.set_type_from_str(solver_type_str)?;
        }
        
        // Apply tolerances and iteration settings
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
        
        // Apply preconditioning side (if specified)
        if let Some(ref pc_side_str) = opts.pc_side {
            let _pc_side = crate::config::options::PcSide::from_str(pc_side_str)?;
            // Note: PcSide handling would be implemented in solver-specific logic
            // For now, we just validate the string
        }
        
        Ok(self)
    }

    /// Configure both solver and preconditioner from options with integrated setup.
    ///
    /// This is a convenience method that configures both KSP and PC options in one call.
    ///
    /// # Arguments
    /// * `ksp_opts` - KSP (solver) options
    /// * `pc_opts` - PC (preconditioner) options
    ///
    /// # Returns
    /// * `Ok(&mut Self)` for method chaining on success
    /// * `Err(KError)` if any option values are invalid
    pub fn set_from_all_options(&mut self, ksp_opts: &crate::config::options::KspOptions, pc_opts: &crate::config::options::PcOptions) -> Result<&mut Self, KError> {
        // Configure preconditioner first
        if let Some(ref pc_type_str) = pc_opts.pc_type {
            self.set_pc_type_from_str(pc_type_str)?;
        }
        
        // Configure solver
        self.set_from_options(ksp_opts)?;
        
        Ok(self)
    }
    
    /// Enable runtime profiling globally.
    ///
    /// When enabled, profiling guards will automatically time solver phases
    /// like setup, matrix-vector products, preconditioner application, etc.
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn enable_profiling(&mut self) -> &mut Self {
        crate::utils::profiling::enable_profiling();
        self
    }
    
    /// Disable runtime profiling globally.
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn disable_profiling(&mut self) -> &mut Self {
        crate::utils::profiling::disable_profiling();
        self
    }
    
    /// Enable runtime monitoring globally.
    ///
    /// When enabled, registered monitors will be called during iterations
    /// to track convergence progress, residual norms, etc.
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn enable_monitoring(&mut self) -> &mut Self {
        crate::utils::profiling::enable_monitoring();
        self
    }
    
    /// Disable runtime monitoring globally.
    ///
    /// # Returns
    /// * `&mut Self` for method chaining
    pub fn disable_monitoring(&mut self) -> &mut Self {
        crate::utils::profiling::disable_monitoring();
        self
    }
    
    /// Check if profiling is currently enabled.
    ///
    /// # Returns
    /// * `true` if profiling is active, `false` otherwise
    pub fn is_profiling_enabled(&self) -> bool {
        crate::utils::profiling::is_profiling_enabled()
    }
    
    /// Check if monitoring is currently enabled.
    ///
    /// # Returns
    /// * `true` if monitoring is active, `false` otherwise
    pub fn is_monitoring_enabled(&self) -> bool {
        crate::utils::profiling::is_monitoring_enabled()
    }
}

impl Default for KspContext {
    fn default() -> Self {
        Self::new()
    }
}
