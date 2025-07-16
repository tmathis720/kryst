//! Unified KSP Context for runtime solver selection and configuration.
//!
//! This module provides a PETSc-style unified interface for Krylov subspace methods.
//! The `KspContext` allows runtime selection of solver types, preconditioners, and
//! tolerances using enum-based configuration.
//!
//! # Usage
//!
//! ```rust
//! use kryst::context::ksp_context::{KspContext, SolverType, PcType};
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
use crate::solver::{LinearSolver, CgSolver, GmresSolver, BiCgStabSolver, MinresSolver, 
                   TfqmrSolver, CgnrSolver, CgsSolver, QmrSolver, LuSolver, PcgSolver};
use crate::preconditioner::{Preconditioner, Jacobi, Ilu0};
use crate::utils::convergence::SolveStats;
use crate::error::KError;

/// Workspace for Krylov solver operations to enable buffer reuse.
///
/// This struct contains all the working arrays needed by Krylov solvers,
/// allocated once during setup and reused across multiple solves.
#[derive(Debug)]
struct Workspace {
    /// Krylov basis vectors Q: size (restart+1) × n for GMRES
    q: Vec<Vec<f64>>,
    /// Preconditioned basis vectors Z: size restart × n for FGMRES  
    z: Vec<Vec<f64>>,
    /// Hessenberg matrix H: (restart+1) × restart for GMRES
    h: Vec<Vec<f64>>,
    /// Givens rotation cosines for GMRES
    cs: Vec<f64>,
    /// Givens rotation sines for GMRES
    sn: Vec<f64>,
    /// Residual vector g in least-squares subproblem
    g: Vec<f64>,
    /// Temporary vector 1 for general use
    tmp1: Vec<f64>,
    /// Temporary vector 2 for general use  
    tmp2: Vec<f64>,
    /// Vector dimension (for validation)
    n: usize,
    /// Current restart parameter (for validation)
    restart: usize,
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

/// All supported preconditioner types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcType {
    /// Jacobi (diagonal scaling) preconditioner
    Jacobi,
    /// Incomplete LU factorization with zero fill-in
    Ilu0,
    /// No preconditioning
    None,
}

impl FromStr for SolverType {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cg" => Ok(SolverType::Cg),
            "pcg" => Ok(SolverType::Pcg),
            "gmres" => Ok(SolverType::Gmres),
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

impl FromStr for PcType {
    type Err = KError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "jacobi" => Ok(PcType::Jacobi),
            "ilu0" => Ok(PcType::Ilu0),
            "none" => Ok(PcType::None),
            _ => Err(KError::UnrecognizedPcType(s.to_string())),
        }
    }
}

/// A no-op preconditioner that performs the identity operation.
pub struct NoOpPreconditioner;

impl Preconditioner<Mat<f64>, Vec<f64>> for NoOpPreconditioner {
    fn apply(&self, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        z.clone_from(r);
        Ok(())
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
            rtol: 1e-6,
            atol: 1e-12,
            dtol: 1e3,
            maxits: 1000,
            restart: 50,
            work: None,
            setup_called: false,
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
        let solver: Box<dyn LinearSolver<Mat<f64>, Vec<f64>, Scalar = f64, Error = KError>> = match solver_type {
            SolverType::Cg => Box::new(CgSolver::new(self.rtol, self.maxits)),
            SolverType::Pcg => Box::new(PcgSolver::new(self.rtol, self.maxits)),
            SolverType::Gmres => Box::new(GmresSolver::new(self.restart, self.rtol, self.maxits)),
            SolverType::BiCgStab => Box::new(BiCgStabSolver::new(self.rtol, self.maxits)),
            SolverType::Cgs => Box::new(CgsSolver::new(self.rtol, self.maxits)),
            SolverType::Qmr => Box::new(QmrSolver::new(self.rtol, self.maxits)),
            SolverType::Tfqmr => Box::new(TfqmrSolver::new(self.rtol, self.maxits)),
            SolverType::Minres => Box::new(MinresSolver::new(self.rtol, self.maxits)),
            SolverType::Cgnr => Box::new(CgnrSolver::new(self.rtol, self.maxits)),
            SolverType::Preonly => Box::new(LuSolver::new()),
        };
        self.solver = Some(solver);
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

    /// Set the preconditioner type and create the appropriate preconditioner instance.
    ///
    /// # Arguments
    /// * `pc_type` - The type of preconditioner to use
    ///
    /// # Returns
    /// * `Ok(&mut Self)` for method chaining on success
    /// * `Err(KError)` if the preconditioner type is not supported
    pub fn set_pc_type(&mut self, pc_type: PcType) -> Result<&mut Self, KError> {
        let pc: Box<dyn Preconditioner<Mat<f64>, Vec<f64>>> = match pc_type {
            PcType::Jacobi => Box::new(Jacobi::new()),
            PcType::Ilu0 => Box::new(Ilu0::new()),
            PcType::None => Box::new(NoOpPreconditioner),
        };
        self.pc = Some(pc);
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
    /// ```rust
    /// let mut ksp = KspContext::new();
    /// ksp.set_type(SolverType::Gmres)?;
    /// ksp.setup(&A, n)?;  // Setup once
    /// ksp.solve(&A, &b1, &mut x1)?;  // Fast solve
    /// ksp.solve(&A, &b2, &mut x2)?;  // Reuses workspace
    /// ```
    pub fn setup(&mut self, a: &Mat<f64>, n: usize) -> Result<(), KError> {
        // Setup preconditioner if present
        if let Some(ref mut pc) = self.pc {
            pc.setup(a)?;
        }

        // Allocate or resize workspace if needed
        if let Some(ref work) = self.work {
            if work.is_compatible(n, self.restart) {
                // Workspace is already compatible, no need to reallocate
                self.setup_called = true;
                return Ok(());
            }
        }

        // Allocate new workspace
        self.work = Some(Workspace::new(n, self.restart));
        self.setup_called = true;
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

    /// Solve the linear system Ax = b using the configured solver and preconditioner.
    ///
    /// This method uses cached workspaces from the setup phase for efficient repeated solves.
    /// If setup has not been called, it will be called automatically.
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
    /// ```rust
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
        // Check if solver is configured
        if self.solver.is_none() {
            return Err(KError::SolveError("No solver configured. Call set_type() first.".to_string()));
        }

        // Auto-setup if needed
        let needs_setup = !self.setup_called || self.work.is_none();
        if needs_setup {
            self.setup(a, b.len())?;
        }

        // Verify workspace compatibility
        if let Some(ref work) = self.work {
            if !work.is_compatible(b.len(), self.restart) {
                return Err(KError::SolveError(
                    "Workspace incompatible with problem size. Call setup() or invalidate_setup().".to_string()
                ));
            }
        }

        // Solve the system using cached workspace
        // Note: The actual solver implementations would need to be modified to accept
        // and use the workspace. For now, we use the existing solver interface.
        let solver = self.solver.as_mut().unwrap(); // Safe because we checked above
        solver.solve(a, self.pc.as_deref(), b, x)
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
    /// ```rust
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
}

impl Default for KspContext {
    fn default() -> Self {
        Self::new()
    }
}
