use super::{
    CudaAmg, CudaBlockJacobi, CudaChebyshev, CudaCsrOp, CudaDistCsrOp, CudaIlu0, CudaJacobi,
    CudaLinOp, CudaNone, CudaOperation, CudaPreconditioner, CudaRuntime, CudaVector,
};
use crate::algebra::prelude::*;
use crate::context::ksp_context::SolverType;
use crate::context::pc_context::PcType;
use crate::error::{CudaErrorKind, KError};
use crate::matrix::op::{StructureId, ValuesId};
use crate::parallel::UniverseComm;
use crate::preconditioner::PcSide;
use crate::solver::common::givens::{apply_new_givens_and_update_g, apply_prev_givens_to_col};
use crate::solver::{MonitorAction, MonitorCallback};
use crate::utils::convergence::{
    ConvergedReason, Convergence, GcrCounters, ReductionModel, SolveStats, SolverCounters,
};
use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::runtime::CudaTimingGuard;
use super::vector::{DeviceScalar, device_to_host};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CudaCgVariant {
    #[default]
    Classical,
    /// Chronopoulos/Gear-style PCG recurrence with one two-scalar collective
    /// per ordinary iteration.
    Pipelined,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CudaGmresVariant {
    #[default]
    Classical,
    /// Classical Gram-Schmidt Arnoldi with all basis projections and the
    /// candidate-vector norm combined into one scalar collective.
    Pipelined,
}

pub struct CudaKspContext {
    runtime: Arc<CudaRuntime>,
    operator: Option<Arc<dyn CudaLinOp>>,
    preconditioning_operator: Option<Arc<dyn CudaLinOp>>,
    preconditioner: Option<Arc<dyn CudaPreconditioner>>,
    solver_type: SolverType,
    pc_type: PcType,
    pc_side: PcSide,
    convergence: Convergence,
    restart: usize,
    richardson_omega: R,
    chebyshev_omega: R,
    chebyshev_pc_degree: usize,
    chebyshev_pc_eig_lo: R,
    chebyshev_pc_eig_hi: R,
    block_jacobi_size: usize,
    cg_variant: CudaCgVariant,
    gmres_variant: CudaGmresVariant,
    monitors: Vec<Box<MonitorCallback<R>>>,
    workspace: Option<CudaWorkspace>,
    custom_preconditioner: bool,
    setup_called: bool,
    last_pc_structure: Option<StructureId>,
    last_pc_values: Option<ValuesId>,
    communicator: Option<UniverseComm>,
}

struct CgWorkspace {
    n: usize,
    r: CudaVector,
    z: CudaVector,
    p: CudaVector,
    ap: CudaVector,
    ax: CudaVector,
    tmp: CudaVector,
    reduction_payload: CudaVector,
}

impl CgWorkspace {
    fn new(runtime: Arc<CudaRuntime>, n: usize) -> Result<Self, KError> {
        Ok(Self {
            n,
            r: CudaVector::zeros(runtime.clone(), n)?,
            z: CudaVector::zeros(runtime.clone(), n)?,
            p: CudaVector::zeros(runtime.clone(), n)?,
            ap: CudaVector::zeros(runtime.clone(), n)?,
            ax: CudaVector::zeros(runtime.clone(), n)?,
            tmp: CudaVector::zeros(runtime.clone(), n)?,
            reduction_payload: CudaVector::zeros(runtime, 2)?,
        })
    }
}

struct RichardsonWorkspace {
    n: usize,
    residual: CudaVector,
    correction: CudaVector,
    ax: CudaVector,
}

struct BiCgStabWorkspace {
    n: usize,
    r: CudaVector,
    r_hat: CudaVector,
    p: CudaVector,
    v: CudaVector,
    s: CudaVector,
    t: CudaVector,
    z_p: CudaVector,
    z_s: CudaVector,
    ax: CudaVector,
    reduction_payload: CudaVector,
}

struct CgsWorkspace {
    n: usize,
    r: CudaVector,
    r_hat: CudaVector,
    u: CudaVector,
    p: CudaVector,
    q: CudaVector,
    u_plus_q: CudaVector,
    v: CudaVector,
    w: CudaVector,
    z_p: CudaVector,
    z_u_plus_q: CudaVector,
    ax: CudaVector,
    reduction_payload: CudaVector,
}

struct CgnrWorkspace {
    rows: usize,
    cols: usize,
    r: CudaVector,
    z: CudaVector,
    p: CudaVector,
    ap: CudaVector,
    zhat: CudaVector,
    extra: CudaVector,
    ax: CudaVector,
    reduction_payload: CudaVector,
}

struct QmrWorkspace {
    n: usize,
    r: CudaVector,
    t: CudaVector,
    r_tld: CudaVector,
    p: CudaVector,
    p_tld: CudaVector,
    v: CudaVector,
    v_tld: CudaVector,
    s: CudaVector,
    ax: CudaVector,
    tmp_pc: CudaVector,
    reduction_payload: CudaVector,
}

impl QmrWorkspace {
    fn new(runtime: Arc<CudaRuntime>, n: usize) -> Result<Self, KError> {
        Ok(Self {
            n,
            r: CudaVector::zeros(runtime.clone(), n)?,
            t: CudaVector::zeros(runtime.clone(), n)?,
            r_tld: CudaVector::zeros(runtime.clone(), n)?,
            p: CudaVector::zeros(runtime.clone(), n)?,
            p_tld: CudaVector::zeros(runtime.clone(), n)?,
            v: CudaVector::zeros(runtime.clone(), n)?,
            v_tld: CudaVector::zeros(runtime.clone(), n)?,
            s: CudaVector::zeros(runtime.clone(), n)?,
            ax: CudaVector::zeros(runtime.clone(), n)?,
            tmp_pc: CudaVector::zeros(runtime.clone(), n)?,
            reduction_payload: CudaVector::zeros(runtime, 2)?,
        })
    }
}

impl CgnrWorkspace {
    fn new(runtime: Arc<CudaRuntime>, rows: usize, cols: usize) -> Result<Self, KError> {
        Ok(Self {
            rows,
            cols,
            r: CudaVector::zeros(runtime.clone(), rows)?,
            z: CudaVector::zeros(runtime.clone(), cols)?,
            p: CudaVector::zeros(runtime.clone(), cols)?,
            ap: CudaVector::zeros(runtime.clone(), rows)?,
            zhat: CudaVector::zeros(runtime.clone(), cols)?,
            extra: CudaVector::zeros(runtime.clone(), cols)?,
            ax: CudaVector::zeros(runtime.clone(), rows)?,
            reduction_payload: CudaVector::zeros(runtime, 2)?,
        })
    }
}

impl CgsWorkspace {
    fn new(runtime: Arc<CudaRuntime>, n: usize) -> Result<Self, KError> {
        Ok(Self {
            n,
            r: CudaVector::zeros(runtime.clone(), n)?,
            r_hat: CudaVector::zeros(runtime.clone(), n)?,
            u: CudaVector::zeros(runtime.clone(), n)?,
            p: CudaVector::zeros(runtime.clone(), n)?,
            q: CudaVector::zeros(runtime.clone(), n)?,
            u_plus_q: CudaVector::zeros(runtime.clone(), n)?,
            v: CudaVector::zeros(runtime.clone(), n)?,
            w: CudaVector::zeros(runtime.clone(), n)?,
            z_p: CudaVector::zeros(runtime.clone(), n)?,
            z_u_plus_q: CudaVector::zeros(runtime.clone(), n)?,
            ax: CudaVector::zeros(runtime.clone(), n)?,
            reduction_payload: CudaVector::zeros(runtime, 2)?,
        })
    }
}

impl BiCgStabWorkspace {
    fn new(runtime: Arc<CudaRuntime>, n: usize) -> Result<Self, KError> {
        Ok(Self {
            n,
            r: CudaVector::zeros(runtime.clone(), n)?,
            r_hat: CudaVector::zeros(runtime.clone(), n)?,
            p: CudaVector::zeros(runtime.clone(), n)?,
            v: CudaVector::zeros(runtime.clone(), n)?,
            s: CudaVector::zeros(runtime.clone(), n)?,
            t: CudaVector::zeros(runtime.clone(), n)?,
            z_p: CudaVector::zeros(runtime.clone(), n)?,
            z_s: CudaVector::zeros(runtime.clone(), n)?,
            ax: CudaVector::zeros(runtime.clone(), n)?,
            reduction_payload: CudaVector::zeros(runtime, 2)?,
        })
    }
}

impl RichardsonWorkspace {
    fn new(runtime: Arc<CudaRuntime>, n: usize) -> Result<Self, KError> {
        Ok(Self {
            n,
            residual: CudaVector::zeros(runtime.clone(), n)?,
            correction: CudaVector::zeros(runtime.clone(), n)?,
            ax: CudaVector::zeros(runtime, n)?,
        })
    }
}

struct GmresWorkspace {
    n: usize,
    restart: usize,
    residual: CudaVector,
    work: CudaVector,
    temp: CudaVector,
    basis: Vec<CudaVector>,
    correction_basis: Vec<CudaVector>,
    h: Vec<S>,
    cs: Vec<R>,
    sn: Vec<S>,
    g: Vec<S>,
    y: Vec<S>,
    reduction_payload: CudaVector,
    host_reduction_payload: Vec<DeviceScalar>,
    basis_ptrs: CudaSlice<u64>,
}

impl GmresWorkspace {
    fn new(runtime: Arc<CudaRuntime>, n: usize, restart: usize) -> Result<Self, KError> {
        let mut basis = Vec::with_capacity(restart + 1);
        let mut correction_basis = Vec::with_capacity(restart);
        for _ in 0..=restart {
            basis.push(CudaVector::zeros(runtime.clone(), n)?);
        }
        for _ in 0..restart {
            correction_basis.push(CudaVector::zeros(runtime.clone(), n)?);
        }
        let basis_ptrs = runtime.upload_vector_pointer_table(&basis)?;
        Ok(Self {
            n,
            restart,
            residual: CudaVector::zeros(runtime.clone(), n)?,
            work: CudaVector::zeros(runtime.clone(), n)?,
            temp: CudaVector::zeros(runtime.clone(), n)?,
            basis,
            correction_basis,
            h: vec![S::zero(); (restart + 1) * restart],
            cs: vec![R::zero(); restart],
            sn: vec![S::zero(); restart],
            g: vec![S::zero(); restart + 1],
            y: vec![S::zero(); restart],
            reduction_payload: CudaVector::zeros(runtime.clone(), restart + 1)?,
            host_reduction_payload: vec![DeviceScalar::default(); restart + 1],
            basis_ptrs,
        })
    }
}

enum CudaWorkspace {
    Cg(CgWorkspace),
    BiCgStab(BiCgStabWorkspace),
    Cgs(CgsWorkspace),
    Cgnr(CgnrWorkspace),
    Qmr(QmrWorkspace),
    Gmres(GmresWorkspace),
    Richardson(RichardsonWorkspace),
}

impl std::fmt::Debug for CudaKspContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKspContext")
            .field("runtime", &self.runtime)
            .field("solver_type", &self.solver_type)
            .field("pc_type", &self.pc_type)
            .field("pc_side", &self.pc_side)
            .field("restart", &self.restart)
            .field("setup_called", &self.setup_called)
            .finish_non_exhaustive()
    }
}

impl CudaKspContext {
    pub fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            operator: None,
            preconditioning_operator: None,
            preconditioner: None,
            solver_type: SolverType::Gmres,
            pc_type: PcType::None,
            pc_side: PcSide::Left,
            convergence: Convergence::new(1e-5, 1e-50, 1e5, 10_000),
            restart: 30,
            richardson_omega: 1.0,
            chebyshev_omega: 0.8,
            chebyshev_pc_degree: 2,
            chebyshev_pc_eig_lo: 0.0,
            chebyshev_pc_eig_hi: 1.0,
            block_jacobi_size: 4,
            cg_variant: CudaCgVariant::Classical,
            gmres_variant: CudaGmresVariant::Classical,
            monitors: Vec::new(),
            workspace: None,
            custom_preconditioner: false,
            setup_called: false,
            last_pc_structure: None,
            last_pc_values: None,
            communicator: None,
        }
    }

    pub fn runtime(&self) -> &Arc<CudaRuntime> {
        &self.runtime
    }

    pub fn set_gmres_variant(&mut self, variant: CudaGmresVariant) -> Result<&mut Self, KError> {
        self.gmres_variant = variant;
        Ok(self)
    }

    pub fn supports_solver_type(solver_type: SolverType) -> bool {
        matches!(
            solver_type,
            SolverType::Cg
                | SolverType::Pcg
                | SolverType::Gmres
                | SolverType::Fgmres
                | SolverType::BiCgStab
                | SolverType::Cgs
                | SolverType::Cgnr
                | SolverType::Cr
                | SolverType::Lsqr
                | SolverType::Lsmr
                | SolverType::Gcr
                | SolverType::PipeGcr
                | SolverType::Qmr
                | SolverType::Tfqmr
                | SolverType::Tcqmr
                | SolverType::Richardson
                | SolverType::Chebyshev
        )
    }

    pub fn supports_pc_type(pc_type: PcType) -> bool {
        matches!(
            pc_type,
            PcType::None
                | PcType::Jacobi
                | PcType::BlockJacobi
                | PcType::Chebyshev
                | PcType::Ilu0
                | PcType::Amg
        )
    }

    fn workspace_matches(&self, rows: usize, cols: usize) -> bool {
        match (&self.workspace, self.solver_type) {
            (Some(CudaWorkspace::Cg(workspace)), SolverType::Cg | SolverType::Pcg) => {
                workspace.n == rows && rows == cols
            }
            (Some(CudaWorkspace::BiCgStab(workspace)), SolverType::BiCgStab) => {
                workspace.n == rows && rows == cols
            }
            (Some(CudaWorkspace::Cgs(workspace)), SolverType::Cgs) => {
                workspace.n == rows && rows == cols
            }
            (
                Some(CudaWorkspace::Cgnr(workspace)),
                SolverType::Cgnr | SolverType::Cr | SolverType::Lsqr | SolverType::Lsmr,
            ) => workspace.rows == rows && workspace.cols == cols,
            (
                Some(CudaWorkspace::Qmr(workspace)),
                SolverType::Qmr | SolverType::Tfqmr | SolverType::Tcqmr,
            ) => workspace.n == rows && rows == cols,
            (
                Some(CudaWorkspace::Gmres(workspace)),
                SolverType::Gmres | SolverType::Fgmres | SolverType::Gcr | SolverType::PipeGcr,
            ) => {
                workspace.n == rows
                    && rows == cols
                    && workspace.restart == self.restart.min(self.convergence.max_iters).max(1)
            }
            (
                Some(CudaWorkspace::Richardson(workspace)),
                SolverType::Richardson | SolverType::Chebyshev,
            ) => workspace.n == rows && rows == cols,
            _ => false,
        }
    }

    pub fn set_type(&mut self, solver_type: SolverType) -> Result<&mut Self, KError> {
        if !Self::supports_solver_type(solver_type) {
            return Err(KError::Unsupported(
                "CUDA currently supports CG, PCG, CGNR, CR, LSQR, LSMR, QMR, TFQMR, TCQMR, GMRES, FGMRES, GCR, PipeGCR, BiCGStab, CGS, Richardson, and Chebyshev",
            ));
        }
        match solver_type {
            SolverType::Cg
            | SolverType::Pcg
            | SolverType::Cgnr
            | SolverType::Cr
            | SolverType::Lsqr
            | SolverType::Lsmr
            | SolverType::Qmr
            | SolverType::Tfqmr
            | SolverType::Tcqmr
            | SolverType::Richardson
            | SolverType::Chebyshev => self.pc_side = PcSide::Left,
            SolverType::Fgmres | SolverType::Gcr | SolverType::PipeGcr => {
                self.pc_side = PcSide::Right
            }
            SolverType::BiCgStab | SolverType::Cgs => self.pc_side = PcSide::Right,
            _ => {}
        }
        self.solver_type = solver_type;
        self.setup_called = false;
        Ok(self)
    }

    pub fn set_pc_type(&mut self, pc_type: PcType) -> Result<&mut Self, KError> {
        if !Self::supports_pc_type(pc_type) {
            return Err(KError::Unsupported(
                "CUDA currently supports None, Jacobi, block Jacobi, Chebyshev, ILU(0), and AMG preconditioners",
            ));
        }
        self.pc_type = pc_type;
        self.preconditioner = None;
        self.custom_preconditioner = false;
        self.setup_called = false;
        Ok(self)
    }

    pub fn set_pc_side(&mut self, side: PcSide) -> Result<&mut Self, KError> {
        if matches!(self.solver_type, SolverType::Cg | SolverType::Pcg) && side != PcSide::Left {
            return Err(KError::InvalidInput(
                "CUDA CG/PCG requires left preconditioning".into(),
            ));
        }
        if matches!(
            self.solver_type,
            SolverType::Cgnr | SolverType::Cr | SolverType::Lsqr | SolverType::Lsmr
        ) && side != PcSide::Left
        {
            return Err(KError::InvalidInput(
                "CUDA CGNR/CR/LSQR/LSMR requires the left preconditioning side".into(),
            ));
        }
        if matches!(
            self.solver_type,
            SolverType::Fgmres | SolverType::Gcr | SolverType::PipeGcr
        ) && side != PcSide::Right
        {
            return Err(KError::InvalidInput(
                "CUDA FGMRES/GCR/PipeGCR requires right preconditioning".into(),
            ));
        }
        if matches!(self.solver_type, SolverType::BiCgStab | SolverType::Cgs)
            && side != PcSide::Right
        {
            return Err(KError::InvalidInput(
                "CUDA BiCGStab/CGS currently requires right preconditioning".into(),
            ));
        }
        if side == PcSide::Symmetric {
            return Err(KError::Unsupported(
                "CUDA symmetric split preconditioning is not implemented",
            ));
        }
        self.pc_side = side;
        Ok(self)
    }

    pub fn set_tolerances(
        &mut self,
        rtol: R,
        atol: R,
        dtol: R,
        max_iters: usize,
    ) -> Result<&mut Self, KError> {
        if !rtol.is_finite()
            || !atol.is_finite()
            || !dtol.is_finite()
            || rtol < 0.0
            || atol < 0.0
            || dtol <= 0.0
            || max_iters == 0
        {
            return Err(KError::InvalidInput(
                "CUDA solver tolerances must be finite/non-negative, dtol positive, and max_iters nonzero"
                    .into(),
            ));
        }
        self.convergence = Convergence::new(rtol, atol, dtol, max_iters);
        Ok(self)
    }

    pub fn set_restart(&mut self, restart: usize) -> Result<&mut Self, KError> {
        if restart == 0 {
            return Err(KError::InvalidInput(
                "CUDA GMRES restart must be nonzero".into(),
            ));
        }
        self.restart = restart;
        Ok(self)
    }

    /// Set the stationary Richardson update factor used in
    /// `x <- x + omega M^-1 (b - A x)`.
    pub fn set_richardson_omega(&mut self, omega: R) -> Result<&mut Self, KError> {
        if !omega.is_finite() || omega == 0.0 {
            return Err(KError::InvalidInput(
                "CUDA Richardson omega must be finite and nonzero".into(),
            ));
        }
        self.richardson_omega = omega;
        Ok(self)
    }

    /// Set the damping factor for CUDA Chebyshev KSP mode. This matches the
    /// current host implementation, where Chebyshev KSP is a damped
    /// stationary iteration distinct from the polynomial Chebyshev PC.
    pub fn set_chebyshev_omega(&mut self, omega: R) -> Result<&mut Self, KError> {
        if !omega.is_finite() || omega <= 0.0 {
            return Err(KError::InvalidInput(
                "CUDA Chebyshev omega must be finite and positive".into(),
            ));
        }
        self.chebyshev_omega = omega;
        Ok(self)
    }

    /// Configure the polynomial Chebyshev preconditioner. These bounds apply
    /// to the attached preconditioning operator, not to Chebyshev KSP's
    /// stationary damping parameter.
    pub fn set_chebyshev_pc(
        &mut self,
        degree: usize,
        eig_lo: R,
        eig_hi: R,
    ) -> Result<&mut Self, KError> {
        if degree == 0
            || !eig_lo.is_finite()
            || !eig_hi.is_finite()
            || eig_lo < 0.0
            || eig_hi <= eig_lo
        {
            return Err(KError::InvalidInput(
                "CUDA Chebyshev PC requires degree >= 1 and finite bounds 0 <= eig_lo < eig_hi"
                    .into(),
            ));
        }
        if self.chebyshev_pc_degree != degree
            || self.chebyshev_pc_eig_lo != eig_lo
            || self.chebyshev_pc_eig_hi != eig_hi
        {
            self.chebyshev_pc_degree = degree;
            self.chebyshev_pc_eig_lo = eig_lo;
            self.chebyshev_pc_eig_hi = eig_hi;
            if self.pc_type == PcType::Chebyshev {
                self.preconditioner = None;
                self.custom_preconditioner = false;
                self.setup_called = false;
            }
        }
        Ok(self)
    }

    pub fn set_block_jacobi_size(&mut self, block_size: usize) -> Result<&mut Self, KError> {
        if block_size == 0 {
            return Err(KError::InvalidInput(
                "CUDA block Jacobi block size must be nonzero".into(),
            ));
        }
        if self.block_jacobi_size != block_size {
            self.block_jacobi_size = block_size;
            if self.pc_type == PcType::BlockJacobi {
                self.preconditioner = None;
                self.custom_preconditioner = false;
                self.setup_called = false;
            }
        }
        Ok(self)
    }

    pub fn set_cg_variant(&mut self, variant: CudaCgVariant) -> &mut Self {
        self.cg_variant = variant;
        self
    }

    pub fn set_operators(
        &mut self,
        operator: Arc<dyn CudaLinOp>,
        preconditioning_operator: Option<Arc<dyn CudaLinOp>>,
    ) -> Result<&mut Self, KError> {
        if operator.device_ordinal() != self.runtime.device_ordinal() {
            return Err(super::runtime::cuda_error(
                CudaErrorKind::DeviceMismatch,
                "set CUDA operators",
                "operator and CUDA context use different devices",
            ));
        }
        let pmat = preconditioning_operator
            .as_ref()
            .map(|op| op.as_ref())
            .unwrap_or(operator.as_ref());
        if pmat.device_ordinal() != self.runtime.device_ordinal() {
            return Err(super::runtime::cuda_error(
                CudaErrorKind::DeviceMismatch,
                "set CUDA preconditioning operator",
                "preconditioning operator and CUDA context use different devices",
            ));
        }
        if let (Some(operator_comm), Some(pmat_comm)) =
            (operator.communicator(), pmat.communicator())
            && !operator_comm.congruent(pmat_comm)
        {
            return Err(KError::InvalidInput(
                "CUDA A/P communicators are not congruent".into(),
            ));
        }
        self.communicator = operator.communicator().cloned();
        self.operator = Some(operator);
        self.preconditioning_operator = preconditioning_operator;
        self.preconditioner = None;
        self.custom_preconditioner = false;
        self.workspace = None;
        self.setup_called = false;
        Ok(self)
    }

    pub fn set_preconditioner(
        &mut self,
        preconditioner: Arc<dyn CudaPreconditioner>,
    ) -> Result<&mut Self, KError> {
        if preconditioner.device_ordinal() != self.runtime.device_ordinal() {
            return Err(super::runtime::cuda_error(
                CudaErrorKind::DeviceMismatch,
                "set CUDA preconditioner",
                "preconditioner and CUDA context use different devices",
            ));
        }
        self.preconditioner = Some(preconditioner);
        self.custom_preconditioner = true;
        self.setup_called = false;
        Ok(self)
    }

    pub fn add_monitor<F>(&mut self, monitor: F)
    where
        F: Fn(usize, R, usize) -> MonitorAction + Send + Sync + 'static,
    {
        self.monitors.push(Box::new(monitor));
    }

    pub fn clear_monitors(&mut self) {
        self.monitors.clear();
    }

    pub fn setup(&mut self) -> Result<&mut Self, KError> {
        let _timing = CudaTimingGuard::setup(self.runtime.clone());
        let operator = self
            .operator
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("CUDA operator is not set".into()))?;
        let (rows, cols) = operator.dims();
        let rectangular_solver = matches!(
            self.solver_type,
            SolverType::Cgnr | SolverType::Cr | SolverType::Lsqr | SolverType::Lsmr
        );
        let transpose_solver = rectangular_solver || self.solver_type == SolverType::Qmr;
        if !rectangular_solver && rows != cols {
            return Err(KError::InvalidInput(format!(
                "CUDA {:?} requires a square operator, got {rows}x{cols}",
                self.solver_type
            )));
        }
        if transpose_solver && !operator.supports_transpose() {
            return Err(KError::InvalidInput(format!(
                "CUDA {:?} requires an operator with transpose/adjoint support",
                self.solver_type
            )));
        }
        if matches!(
            self.solver_type,
            SolverType::Cg
                | SolverType::Pcg
                | SolverType::Cgnr
                | SolverType::Cr
                | SolverType::Lsqr
                | SolverType::Lsmr
                | SolverType::Tfqmr
                | SolverType::Tcqmr
                | SolverType::Richardson
                | SolverType::Chebyshev
        ) && self.pc_side != PcSide::Left
        {
            return Err(KError::InvalidInput(
                "CUDA CG/PCG/CGNR/CR/LSQR/TFQMR/TCQMR/Richardson/Chebyshev requires left preconditioning".into(),
            ));
        }
        if matches!(
            self.solver_type,
            SolverType::Fgmres | SolverType::Gcr | SolverType::PipeGcr
        ) && self.pc_side != PcSide::Right
        {
            return Err(KError::InvalidInput(
                "CUDA FGMRES/GCR/PipeGCR requires right preconditioning".into(),
            ));
        }
        if matches!(self.solver_type, SolverType::BiCgStab | SolverType::Cgs)
            && self.pc_side != PcSide::Right
        {
            return Err(KError::InvalidInput(
                "CUDA BiCGStab/CGS currently requires right preconditioning".into(),
            ));
        }
        let pmat = self.preconditioning_operator.as_ref().unwrap_or(operator);
        if matches!(self.solver_type, SolverType::Lsqr | SolverType::Lsmr)
            && (self.pc_type != PcType::None || self.custom_preconditioner)
        {
            return Err(KError::Unsupported(
                "CUDA LSQR/LSMR preconditioning is not yet supported",
            ));
        }
        if matches!(
            self.solver_type,
            SolverType::Qmr | SolverType::Tfqmr | SolverType::Tcqmr
        ) && (self.pc_type != PcType::None || self.custom_preconditioner)
        {
            return Err(KError::Unsupported(
                "CUDA QMR-family preconditioning is not yet supported",
            ));
        }
        let expected_pmat_dims = if rectangular_solver {
            (cols, cols)
        } else {
            (rows, cols)
        };
        if self.preconditioning_operator.is_some() && pmat.dims() != expected_pmat_dims {
            return Err(KError::InvalidInput(format!(
                "CUDA {:?} preconditioning operator must have dimensions {:?}, got {:?}",
                self.solver_type,
                expected_pmat_dims,
                pmat.dims()
            )));
        }
        if rows == 0 {
            if !self.custom_preconditioner {
                self.preconditioner = Some(Arc::new(CudaNone::new(self.runtime.clone(), cols)));
            }
            let pc = self.preconditioner.as_ref().ok_or_else(|| {
                KError::InvalidInput("CUDA preconditioner is not configured".into())
            })?;
            if pc.dims() != (cols, cols) {
                return Err(KError::InvalidInput(format!(
                    "CUDA preconditioner dimension mismatch: {:?} vs ({cols}, {cols})",
                    pc.dims(),
                )));
            }
            self.workspace = None;
            self.last_pc_structure = Some(pmat.structure_id());
            self.last_pc_values = Some(pmat.values_id());
            self.setup_called = true;
            return Ok(self);
        }
        operator.prepare()?;
        let structure_changed = self.last_pc_structure != Some(pmat.structure_id());
        let values_changed = self.last_pc_values != Some(pmat.values_id());
        if !self.custom_preconditioner {
            match self.pc_type {
                PcType::None => {
                    if self.preconditioner.is_none() || structure_changed {
                        self.preconditioner =
                            Some(Arc::new(CudaNone::new(self.runtime.clone(), cols)));
                    }
                }
                PcType::Jacobi => {
                    if pmat.dims() != (cols, cols) {
                        return Err(KError::InvalidInput(format!(
                            "CUDA {:?} Jacobi requires a {cols}x{cols} normal-equation preconditioning operator; got {:?}",
                            self.solver_type,
                            pmat.dims()
                        )));
                    }
                    let csr = pc_csr_operator(pmat.as_ref())?;
                    let reusable = if structure_changed {
                        false
                    } else if let Some(jacobi) = self
                        .preconditioner
                        .as_ref()
                        .and_then(|pc| pc.as_any().downcast_ref::<CudaJacobi>())
                    {
                        if values_changed {
                            jacobi.update_from_csr(csr)?;
                        }
                        true
                    } else {
                        false
                    };
                    if !reusable {
                        self.preconditioner = Some(Arc::new(CudaJacobi::from_csr(csr)?));
                    }
                }
                PcType::BlockJacobi => {
                    if pmat.dims() != (cols, cols) {
                        return Err(KError::InvalidInput(format!(
                            "CUDA {:?} block Jacobi requires a {cols}x{cols} normal-equation preconditioning operator; got {:?}",
                            self.solver_type,
                            pmat.dims()
                        )));
                    }
                    let csr = pc_csr_operator(pmat.as_ref())?;
                    let reusable = if structure_changed {
                        false
                    } else if let Some(block_jacobi) = self
                        .preconditioner
                        .as_ref()
                        .and_then(|pc| pc.as_any().downcast_ref::<CudaBlockJacobi>())
                        .filter(|pc| pc.block_size() == self.block_jacobi_size)
                    {
                        if values_changed {
                            block_jacobi.update_from_csr(csr)?;
                        }
                        true
                    } else {
                        false
                    };
                    if !reusable {
                        self.preconditioner = Some(Arc::new(CudaBlockJacobi::from_csr(
                            csr,
                            self.block_jacobi_size,
                        )?));
                    }
                }
                PcType::Chebyshev => {
                    if pmat.dims() != (cols, cols) {
                        return Err(KError::InvalidInput(format!(
                            "CUDA {:?} Chebyshev PC requires a {cols}x{cols} preconditioning operator; got {:?}",
                            self.solver_type,
                            pmat.dims()
                        )));
                    }
                    let reusable = !structure_changed
                        && self
                            .preconditioner
                            .as_ref()
                            .and_then(|pc| pc.as_any().downcast_ref::<CudaChebyshev>())
                            .is_some_and(|pc| {
                                pc.degree() == self.chebyshev_pc_degree
                                    && pc.spectral_bounds()
                                        == (self.chebyshev_pc_eig_lo, self.chebyshev_pc_eig_hi)
                            });
                    if !reusable {
                        self.preconditioner = Some(Arc::new(CudaChebyshev::new(
                            self.runtime.clone(),
                            pmat.clone(),
                            self.chebyshev_pc_degree,
                            self.chebyshev_pc_eig_lo,
                            self.chebyshev_pc_eig_hi,
                        )?));
                    }
                }
                PcType::Ilu0 => {
                    if pmat.dims() != (cols, cols) {
                        return Err(KError::InvalidInput(format!(
                            "CUDA {:?} ILU(0) requires a {cols}x{cols} preconditioning operator; got {:?}",
                            self.solver_type,
                            pmat.dims()
                        )));
                    }
                    let csr = pc_csr_operator(pmat.as_ref())?;
                    let reusable = if structure_changed {
                        false
                    } else if let Some(ilu) = self
                        .preconditioner
                        .as_ref()
                        .and_then(|pc| pc.as_any().downcast_ref::<CudaIlu0>())
                    {
                        if values_changed {
                            ilu.update_from_csr(csr)?;
                        }
                        true
                    } else {
                        false
                    };
                    if !reusable {
                        self.preconditioner = Some(Arc::new(CudaIlu0::from_csr(csr)?));
                    }
                }
                PcType::Amg => {
                    if pmat.dims() != (cols, cols) {
                        return Err(KError::InvalidInput(format!(
                            "CUDA {:?} AMG requires a {cols}x{cols} preconditioning operator; got {:?}",
                            self.solver_type,
                            pmat.dims()
                        )));
                    }
                    let csr = pc_csr_operator(pmat.as_ref())?;
                    let reusable = !structure_changed
                        && !values_changed
                        && self
                            .preconditioner
                            .as_ref()
                            .and_then(|pc| pc.as_any().downcast_ref::<CudaAmg>())
                            .is_some();
                    if !reusable {
                        self.preconditioner = Some(Arc::new(CudaAmg::from_csr(csr)?));
                    }
                }
                _ => {
                    return Err(KError::Unsupported(
                        "selected preconditioner has no CUDA implementation",
                    ));
                }
            }
        }
        let pc = self.preconditioner.as_ref().unwrap();
        if pc.dims() != (cols, cols) {
            return Err(KError::InvalidInput(format!(
                "CUDA preconditioner dimension mismatch: {:?} vs ({cols}, {cols})",
                pc.dims()
            )));
        }
        pc.prepare()?;
        if rows > 0 && !self.workspace_matches(rows, cols) {
            self.workspace = Some(match self.solver_type {
                SolverType::Cg | SolverType::Pcg => {
                    CudaWorkspace::Cg(CgWorkspace::new(self.runtime.clone(), rows)?)
                }
                SolverType::BiCgStab => {
                    CudaWorkspace::BiCgStab(BiCgStabWorkspace::new(self.runtime.clone(), rows)?)
                }
                SolverType::Cgs => {
                    CudaWorkspace::Cgs(CgsWorkspace::new(self.runtime.clone(), rows)?)
                }
                SolverType::Cgnr | SolverType::Cr | SolverType::Lsqr | SolverType::Lsmr => {
                    CudaWorkspace::Cgnr(CgnrWorkspace::new(self.runtime.clone(), rows, cols)?)
                }
                SolverType::Qmr | SolverType::Tfqmr | SolverType::Tcqmr => {
                    CudaWorkspace::Qmr(QmrWorkspace::new(self.runtime.clone(), rows)?)
                }
                SolverType::Gmres | SolverType::Fgmres | SolverType::Gcr | SolverType::PipeGcr => {
                    CudaWorkspace::Gmres(GmresWorkspace::new(
                        self.runtime.clone(),
                        rows,
                        self.restart.min(self.convergence.max_iters).max(1),
                    )?)
                }
                SolverType::Richardson | SolverType::Chebyshev => {
                    CudaWorkspace::Richardson(RichardsonWorkspace::new(self.runtime.clone(), rows)?)
                }
                _ => {
                    return Err(KError::Unsupported(
                        "selected solver has no CUDA implementation",
                    ));
                }
            });
        }
        self.last_pc_structure = Some(pmat.structure_id());
        self.last_pc_values = Some(pmat.values_id());
        self.setup_called = true;
        Ok(self)
    }

    pub fn solve(&mut self, b: &CudaVector, x: &mut CudaVector) -> Result<SolveStats<R>, KError> {
        let _timing = CudaTimingGuard::solve(self.runtime.clone());
        // Setup is deliberately cheap when IDs are unchanged, and checking on
        // every solve makes shared numeric operator updates visible.
        self.setup()?;
        let operator = self.operator.as_ref().unwrap().clone();
        let pc = self.preconditioner.as_ref().unwrap().clone();
        let (rows, cols) = operator.dims();
        if b.len() != rows || x.len() != cols {
            return Err(KError::InvalidInput(format!(
                "CUDA solve dimensions require b={rows}, x={cols}; got b={}, x={}",
                b.len(),
                x.len()
            )));
        }
        if b.device_ordinal() != self.runtime.device_ordinal()
            || x.device_ordinal() != self.runtime.device_ordinal()
        {
            return Err(super::runtime::cuda_error(
                CudaErrorKind::DeviceMismatch,
                "solve CUDA system",
                "vectors and CUDA context use different devices",
            ));
        }
        if rows == 0 {
            let mut stats = SolveStats::new(0, R::zero(), ConvergedReason::ConvergedAtol);
            stats.final_recurrence_residual = Some(R::zero());
            stats.final_true_residual = Some(R::zero());
            return Ok(stats.finalize_reason_counters());
        }
        let mut workspace = self.workspace.take().ok_or_else(|| {
            KError::SolveError("CUDA solver workspace was not initialized".into())
        })?;
        let result = match (&mut workspace, self.solver_type) {
            (CudaWorkspace::Cg(workspace), SolverType::Cg | SolverType::Pcg) => {
                match self.cg_variant {
                    CudaCgVariant::Classical => {
                        self.solve_cg(operator.as_ref(), pc.as_ref(), b, x, workspace)
                    }
                    CudaCgVariant::Pipelined => {
                        self.solve_pipelined_cg(operator.as_ref(), pc.as_ref(), b, x, workspace)
                    }
                }
            }
            (CudaWorkspace::BiCgStab(workspace), SolverType::BiCgStab) => {
                self.solve_bicgstab(operator.as_ref(), pc.as_ref(), b, x, workspace)
            }
            (CudaWorkspace::Cgs(workspace), SolverType::Cgs) => {
                self.solve_cgs(operator.as_ref(), pc.as_ref(), b, x, workspace)
            }
            (CudaWorkspace::Cgnr(workspace), SolverType::Cgnr | SolverType::Cr) => {
                self.solve_cgnr(operator.as_ref(), pc.as_ref(), b, x, workspace)
            }
            (CudaWorkspace::Cgnr(workspace), SolverType::Lsqr) => {
                self.solve_lsqr(operator.as_ref(), b, x, workspace)
            }
            (CudaWorkspace::Cgnr(workspace), SolverType::Lsmr) => {
                self.solve_lsmr(operator.as_ref(), b, x, workspace)
            }
            (CudaWorkspace::Qmr(workspace), SolverType::Qmr) => {
                self.solve_qmr(operator.as_ref(), b, x, workspace)
            }
            (CudaWorkspace::Qmr(workspace), SolverType::Tfqmr | SolverType::Tcqmr) => {
                self.solve_tfqmr_standard(operator.as_ref(), b, x, workspace)
            }
            (
                CudaWorkspace::Gmres(workspace),
                SolverType::Gmres | SolverType::Fgmres | SolverType::Gcr,
            ) => self.solve_gmres(operator.as_ref(), pc.as_ref(), b, x, workspace),
            (CudaWorkspace::Gmres(workspace), SolverType::PipeGcr) => {
                self.solve_pipegcr(operator.as_ref(), pc.as_ref(), b, x, workspace)
            }
            (
                CudaWorkspace::Richardson(workspace),
                SolverType::Richardson | SolverType::Chebyshev,
            ) => self.solve_richardson(operator.as_ref(), pc.as_ref(), b, x, workspace),
            _ => Err(KError::SolveError(
                "CUDA solver workspace does not match the selected solver".into(),
            )),
        };
        self.workspace = Some(workspace);
        result
    }

    pub fn solve_host(&mut self, b: &[S], x: &mut [S]) -> Result<SolveStats<R>, KError> {
        let b_device = CudaVector::from_host(self.runtime.clone(), b)?;
        let mut x_device = CudaVector::from_host(self.runtime.clone(), x)?;
        let stats = self.solve(&b_device, &mut x_device)?;
        x_device.copy_to_host(x)?;
        Ok(stats)
    }

    fn invoke_monitors(&self, iteration: usize, residual: R, reductions: usize) -> bool {
        self.monitors
            .iter()
            .any(|monitor| monitor(iteration, residual, reductions) == MonitorAction::Stop)
    }

    fn dot(&self, x: &CudaVector, y: &CudaVector) -> Result<S, KError> {
        let local = self.runtime.dot(x.buffer(), y.buffer())?;
        Ok(match self.communicator.as_ref() {
            Some(comm) => comm.allreduce_sum_scalar(local),
            None => local,
        })
    }

    fn dot2(
        &self,
        x0: &CudaVector,
        y0: &CudaVector,
        x1: &CudaVector,
        y1: &CudaVector,
        payload: &mut CudaVector,
    ) -> Result<[S; 2], KError> {
        let mut values = self.runtime.dot2(
            x0.buffer(),
            y0.buffer(),
            x1.buffer(),
            y1.buffer(),
            payload.buffer_mut(),
        )?;
        if let Some(comm) = self.communicator.as_ref() {
            comm.allreduce_sum_scalars(&mut values);
        }
        Ok(values)
    }

    fn norm2(&self, x: &CudaVector) -> Result<R, KError> {
        if let Some(comm) = self.communicator.as_ref() {
            let local = self.runtime.dot(x.buffer(), x.buffer())?.real();
            let global = comm.allreduce_sum_real(local);
            if !global.is_finite() {
                return Err(KError::NonFiniteReduction {
                    kind: if global.is_nan() {
                        crate::error::NonFiniteKind::Nan
                    } else {
                        crate::error::NonFiniteKind::Inf
                    },
                    context: "distributed CUDA norm",
                });
            }
            Ok(global.max(0.0).sqrt())
        } else {
            self.runtime.norm2(x.buffer())
        }
    }

    /// Compute a set of local device dot products and combine their scalar
    /// payload with one MPI collective. This keeps basis vectors resident on
    /// the device while giving PipeGCR a collective count independent of the
    /// current restart depth.
    fn basis_dots(
        &self,
        basis: &[CudaVector],
        rhs: &CudaVector,
        output: &mut [S],
    ) -> Result<(), KError> {
        if basis.len() != output.len() {
            return Err(KError::InvalidInput(format!(
                "CUDA basis-dot output length mismatch: {} vs {}",
                output.len(),
                basis.len()
            )));
        }
        for (value, vector) in output.iter_mut().zip(basis) {
            *value = self.runtime.dot(vector.buffer(), rhs.buffer())?;
        }
        if let Some(comm) = self.communicator.as_ref() {
            comm.allreduce_sum_scalars(output);
        }
        Ok(())
    }

    fn pipelined_arnoldi_step(
        &self,
        basis: &[CudaVector],
        work: &mut CudaVector,
        payload: &mut [S],
        device_payload: &mut CudaVector,
        host_payload: &mut [DeviceScalar],
        basis_ptrs: &CudaSlice<u64>,
    ) -> Result<R, KError> {
        let needed = basis.len() + 1;
        if payload.len() < needed {
            return Err(KError::InvalidInput(format!(
                "CUDA pipelined Arnoldi payload requires {needed} scalars, got {}",
                payload.len()
            )));
        }
        self.runtime.arnoldi_multi_dot(
            basis_ptrs,
            basis.len(),
            work.buffer(),
            device_payload.buffer_mut(),
            &mut host_payload[..needed],
        )?;
        device_to_host(&host_payload[..needed], &mut payload[..needed]);
        if let Some(comm) = self.communicator.as_ref() {
            comm.allreduce_sum_scalars(&mut payload[..needed]);
        }

        let total_norm_sq = nonnegative_hermitian_real(
            payload[basis.len()],
            "CUDA pipelined Arnoldi candidate norm squared",
            false,
        )?;
        let mut projected_norm_sq = R::zero();
        for (coefficient, vector) in payload[..basis.len()].iter().copied().zip(basis) {
            ensure_finite(coefficient, "CUDA pipelined Arnoldi projection")?;
            projected_norm_sq += coefficient.abs() * coefficient.abs();
            self.runtime
                .axpby(-coefficient, vector.buffer(), S::one(), work.buffer_mut())?;
        }
        let next_norm_sq = (total_norm_sq - projected_norm_sq).max(R::zero());
        if !next_norm_sq.is_finite() {
            return Err(KError::NonFiniteReduction {
                kind: crate::error::NonFiniteKind::Nan,
                context: "CUDA pipelined Arnoldi orthogonal norm squared",
            });
        }
        Ok(next_norm_sq.sqrt())
    }

    fn solve_cg(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut CgWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let CgWorkspace {
            r, z, p, ap, ax, ..
        } = workspace;

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime.axpy(-S::one(), ax.buffer(), r.buffer_mut())?;
        let bnorm = self.norm2(b)?;
        let mut rnorm = self.norm2(r)?;
        let (reason, mut stats) = self.convergence.check(rnorm, bnorm, 0);
        if reason != ConvergedReason::Continued {
            stats.final_true_residual = Some(rnorm);
            stats.final_recurrence_residual = Some(rnorm);
            return Ok(stats.finalize_reason_counters());
        }

        pc.apply(r, z)?;
        self.runtime.copy(z.buffer(), p.buffer_mut())?;
        let mut rho = self.dot(r, z)?;
        let mut reductions = 3usize;
        ensure_positive_real(rho, "CUDA CG r^H M^-1 r", true)?;

        let mut final_reason = ConvergedReason::DivergedMaxIts;
        let mut iterations = 0usize;
        for iteration in 1..=self.convergence.max_iters {
            operator.apply(CudaOperation::NonTranspose, p, ap)?;
            let p_ap = self.dot(p, ap)?;
            reductions += 1;
            ensure_positive_real(p_ap, "CUDA CG p^H A p", false)?;
            let alpha = rho / p_ap;
            self.runtime.cg_update(
                alpha,
                p.buffer(),
                ap.buffer(),
                x.buffer_mut(),
                r.buffer_mut(),
            )?;
            rnorm = self.norm2(r)?;
            reductions += 1;
            iterations = iteration;
            if self.invoke_monitors(iteration, rnorm, reductions) {
                final_reason = ConvergedReason::StoppedByMonitor;
                break;
            }
            let (reason, _) = self.convergence.check(rnorm, bnorm, iteration);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
                break;
            }
            pc.apply(r, z)?;
            let rho_new = self.dot(r, z)?;
            reductions += 1;
            ensure_positive_real(rho_new, "CUDA CG r^H M^-1 r", true)?;
            let beta = rho_new / rho;
            self.runtime
                .axpby(S::one(), z.buffer(), beta, p.buffer_mut())?;
            rho = rho_new;
        }

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime.axpy(-S::one(), ax.buffer(), r.buffer_mut())?;
        let true_residual = self.norm2(r)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (true_reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if true_reason != ConvergedReason::Continued {
                final_reason = true_reason;
            }
        }
        let mut stats = SolveStats::new(iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(rnorm);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: "cuda-fused-classical-cg",
            startup: 3,
            per_iteration: 3.0,
            tail: 1,
        });
        stats.effective_variant = Some("cuda-fused-classical".into());
        Ok(stats.finalize_reason_counters())
    }

    fn solve_pipelined_cg(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut CgWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let CgWorkspace {
            r,
            z: u,
            p,
            ap: s,
            ax: w,
            tmp,
            reduction_payload,
            ..
        } = workspace;

        operator.apply(CudaOperation::NonTranspose, x, tmp)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), tmp.buffer(), S::one(), r.buffer_mut())?;
        let bnorm = self.norm2(b)?;
        let mut true_residual = self.norm2(r)?;
        let mut reductions = 2usize;
        let (reason, mut initial_stats) = self.convergence.check(true_residual, bnorm, 0);
        if reason != ConvergedReason::Continued {
            initial_stats.final_true_residual = Some(true_residual);
            initial_stats.final_recurrence_residual = Some(true_residual);
            return Ok(initial_stats.finalize_reason_counters());
        }

        pc.apply(r, u)?;
        operator.apply(CudaOperation::NonTranspose, u, w)?;
        let initial = self.dot2(r, u, u, w, reduction_payload)?;
        reductions += 1;
        ensure_positive_real(initial[0], "CUDA pipelined CG r^H M^-1 r", true)?;
        ensure_positive_real(initial[1], "CUDA pipelined CG u^H A u", false)?;
        let mut rho = initial[0].real();
        let mut alpha = rho / initial[1].real();
        self.runtime.copy(u.buffer(), p.buffer_mut())?;
        self.runtime.copy(w.buffer(), s.buffer_mut())?;

        let mut recurrence_residual = rho.sqrt();
        let mut final_reason = ConvergedReason::DivergedMaxIts;
        let mut iterations = 0usize;
        for iteration in 1..=self.convergence.max_iters {
            let alpha_scalar = S::from_real(alpha);
            self.runtime.cg_update(
                alpha_scalar,
                p.buffer(),
                s.buffer(),
                x.buffer_mut(),
                r.buffer_mut(),
            )?;
            pc.apply(r, u)?;
            operator.apply(CudaOperation::NonTranspose, u, w)?;
            let next = self.dot2(r, u, u, w, reduction_payload)?;
            reductions += 1;
            let rho_new =
                nonnegative_hermitian_real(next[0], "CUDA pipelined CG updated r^H M^-1 r", true)?;
            recurrence_residual = rho_new.sqrt();
            iterations = iteration;

            if self.invoke_monitors(iteration, recurrence_residual, reductions) {
                final_reason = ConvergedReason::StoppedByMonitor;
                break;
            }
            let (recurrence_reason, _) =
                self.convergence
                    .check(recurrence_residual, bnorm, iteration);
            if recurrence_reason != ConvergedReason::Continued {
                // Verify optimistic recursive convergence before returning.
                operator.apply(CudaOperation::NonTranspose, x, tmp)?;
                self.runtime.copy(b.buffer(), r.buffer_mut())?;
                self.runtime
                    .axpby(-S::one(), tmp.buffer(), S::one(), r.buffer_mut())?;
                true_residual = self.norm2(r)?;
                reductions += 1;
                let (true_reason, _) = self.convergence.check(true_residual, bnorm, iteration);
                if true_reason != ConvergedReason::Continued {
                    final_reason = true_reason;
                    break;
                }

                // Residual replacement restarts the short recurrence.
                pc.apply(r, u)?;
                operator.apply(CudaOperation::NonTranspose, u, w)?;
                let refreshed = self.dot2(r, u, u, w, reduction_payload)?;
                reductions += 1;
                ensure_positive_real(refreshed[0], "CUDA pipelined CG refreshed r^H M^-1 r", true)?;
                ensure_positive_real(refreshed[1], "CUDA pipelined CG refreshed u^H A u", false)?;
                rho = refreshed[0].real();
                alpha = rho / refreshed[1].real();
                self.runtime.copy(u.buffer(), p.buffer_mut())?;
                self.runtime.copy(w.buffer(), s.buffer_mut())?;
                continue;
            }

            ensure_positive_real(next[1], "CUDA pipelined CG updated u^H A u", false)?;
            let beta = rho_new / rho;
            let denominator = next[1].real() - (beta / alpha) * rho_new;
            if !denominator.is_finite() {
                return Err(KError::NonFiniteReduction {
                    kind: if denominator.is_nan() {
                        crate::error::NonFiniteKind::Nan
                    } else {
                        crate::error::NonFiniteKind::Inf
                    },
                    context: "CUDA pipelined CG denominator",
                });
            }
            if denominator <= 0.0 {
                return Err(KError::IndefiniteMatrix);
            }
            let beta_scalar = S::from_real(beta);
            self.runtime
                .axpby(S::one(), u.buffer(), beta_scalar, p.buffer_mut())?;
            self.runtime
                .axpby(S::one(), w.buffer(), beta_scalar, s.buffer_mut())?;
            rho = rho_new;
            alpha = rho / denominator;
        }

        operator.apply(CudaOperation::NonTranspose, x, tmp)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), tmp.buffer(), S::one(), r.buffer_mut())?;
        true_residual = self.norm2(r)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        let mut stats = SolveStats::new(iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(recurrence_residual);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: "cuda-pipelined-cg",
            startup: 3,
            per_iteration: 1.0,
            tail: 1,
        });
        stats.effective_variant = Some("cuda-pipelined".into());
        Ok(stats.finalize_reason_counters())
    }

    fn solve_bicgstab(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut BiCgStabWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let BiCgStabWorkspace {
            r,
            r_hat,
            p,
            v,
            s,
            t,
            z_p,
            z_s,
            ax,
            reduction_payload,
            ..
        } = workspace;

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), r.buffer_mut())?;
        self.runtime.copy(r.buffer(), r_hat.buffer_mut())?;
        self.runtime.copy(r.buffer(), p.buffer_mut())?;

        let bnorm = self.norm2(b)?;
        let mut rnorm = self.norm2(r)?;
        let mut reductions = 2usize;
        let (initial_reason, mut initial_stats) = self.convergence.check(rnorm, bnorm, 0);
        if initial_reason != ConvergedReason::Continued {
            initial_stats.final_true_residual = Some(rnorm);
            initial_stats.final_recurrence_residual = Some(rnorm);
            initial_stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(initial_stats.finalize_reason_counters());
        }

        let mut rho_previous = S::one();
        let mut alpha = S::one();
        let mut omega = S::one();
        let mut iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;

        for iteration in 1..=self.convergence.max_iters {
            let rho = self.dot(r_hat, r)?;
            reductions += 1;
            ensure_nonzero_finite(rho, "CUDA BiCGStab rho")?;
            if iteration > 1 {
                let beta = (rho / rho_previous) * (alpha / omega);
                ensure_finite(beta, "CUDA BiCGStab beta")?;
                self.runtime
                    .axpby(-omega, v.buffer(), S::one(), p.buffer_mut())?;
                self.runtime
                    .axpby(S::one(), r.buffer(), beta, p.buffer_mut())?;
            }

            pc.apply(p, z_p)?;
            operator.apply(CudaOperation::NonTranspose, z_p, v)?;
            let alpha_denominator = self.dot(r_hat, v)?;
            reductions += 1;
            ensure_nonzero_finite(alpha_denominator, "CUDA BiCGStab alpha denominator")?;
            alpha = rho / alpha_denominator;
            ensure_nonzero_finite(alpha, "CUDA BiCGStab alpha")?;

            self.runtime.copy(r.buffer(), s.buffer_mut())?;
            self.runtime
                .axpby(-alpha, v.buffer(), S::one(), s.buffer_mut())?;
            let s_norm = self.norm2(s)?;
            reductions += 1;
            let (s_reason, _) = self.convergence.check(s_norm, bnorm, iteration);
            if s_reason != ConvergedReason::Continued {
                self.runtime
                    .axpby(alpha, z_p.buffer(), S::one(), x.buffer_mut())?;
                iterations = iteration;
                rnorm = s_norm;
                final_reason = if self.invoke_monitors(iteration, rnorm, reductions) {
                    ConvergedReason::StoppedByMonitor
                } else {
                    s_reason
                };
                break;
            }

            pc.apply(s, z_s)?;
            operator.apply(CudaOperation::NonTranspose, z_s, t)?;
            let omega_dots = self.dot2(t, t, t, s, reduction_payload)?;
            reductions += 1;
            ensure_positive_real(omega_dots[0], "CUDA BiCGStab omega denominator", false)?;
            omega = omega_dots[1] / omega_dots[0];
            ensure_nonzero_finite(omega, "CUDA BiCGStab omega")?;

            self.runtime
                .axpby(alpha, z_p.buffer(), S::one(), x.buffer_mut())?;
            self.runtime
                .axpby(omega, z_s.buffer(), S::one(), x.buffer_mut())?;
            self.runtime.copy(s.buffer(), r.buffer_mut())?;
            self.runtime
                .axpby(-omega, t.buffer(), S::one(), r.buffer_mut())?;
            rnorm = self.norm2(r)?;
            reductions += 1;
            iterations = iteration;
            if self.invoke_monitors(iteration, rnorm, reductions) {
                final_reason = ConvergedReason::StoppedByMonitor;
                break;
            }
            let (reason, _) = self.convergence.check(rnorm, bnorm, iteration);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
                break;
            }
            rho_previous = rho;
        }

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), r.buffer_mut())?;
        let true_residual = self.norm2(r)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        let mut stats = SolveStats::new(iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(rnorm);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: "cuda-bicgstab",
            startup: 2,
            per_iteration: 5.0,
            tail: 1,
        });
        stats.effective_variant = Some("cuda-classical-bicgstab".into());
        Ok(stats.finalize_reason_counters())
    }

    fn solve_cgs(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut CgsWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let CgsWorkspace {
            r,
            r_hat,
            u,
            p,
            q,
            u_plus_q,
            v,
            w,
            z_p,
            z_u_plus_q,
            ax,
            reduction_payload,
            ..
        } = workspace;

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), r.buffer_mut())?;
        self.runtime.copy(r.buffer(), r_hat.buffer_mut())?;
        self.runtime.copy(r.buffer(), u.buffer_mut())?;
        self.runtime.copy(r.buffer(), p.buffer_mut())?;

        let bnorm = self.norm2(b)?;
        let initial = self.dot2(r, r, r_hat, r, reduction_payload)?;
        let mut reductions = 2usize;
        let mut rnorm = nonnegative_hermitian_real(
            initial[0],
            "CUDA CGS initial residual norm squared",
            false,
        )?
        .sqrt();
        let mut rho = initial[1];
        let (initial_reason, mut initial_stats) = self.convergence.check(rnorm, bnorm, 0);
        if self.invoke_monitors(0, rnorm, reductions) {
            initial_stats.reason = ConvergedReason::StoppedByMonitor;
            initial_stats.final_true_residual = Some(rnorm);
            initial_stats.final_recurrence_residual = Some(rnorm);
            initial_stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(initial_stats.finalize_reason_counters());
        }
        if initial_reason != ConvergedReason::Continued {
            initial_stats.final_true_residual = Some(rnorm);
            initial_stats.final_recurrence_residual = Some(rnorm);
            initial_stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(initial_stats.finalize_reason_counters());
        }
        ensure_nonzero_finite(rho, "CUDA CGS rho")?;

        let mut iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;
        for iteration in 1..=self.convergence.max_iters {
            pc.apply(p, z_p)?;
            operator.apply(CudaOperation::NonTranspose, z_p, v)?;
            let sigma = self.dot(r_hat, v)?;
            reductions += 1;
            ensure_nonzero_finite(sigma, "CUDA CGS sigma")?;
            let alpha = rho / sigma;
            ensure_nonzero_finite(alpha, "CUDA CGS alpha")?;

            self.runtime.copy(u.buffer(), q.buffer_mut())?;
            self.runtime
                .axpby(-alpha, v.buffer(), S::one(), q.buffer_mut())?;
            self.runtime.copy(u.buffer(), u_plus_q.buffer_mut())?;
            self.runtime
                .axpby(S::one(), q.buffer(), S::one(), u_plus_q.buffer_mut())?;
            pc.apply(u_plus_q, z_u_plus_q)?;
            self.runtime
                .axpby(alpha, z_u_plus_q.buffer(), S::one(), x.buffer_mut())?;
            operator.apply(CudaOperation::NonTranspose, z_u_plus_q, w)?;
            self.runtime
                .axpby(-alpha, w.buffer(), S::one(), r.buffer_mut())?;

            let residual_dots = self.dot2(r, r, r_hat, r, reduction_payload)?;
            reductions += 1;
            rnorm = nonnegative_hermitian_real(
                residual_dots[0],
                "CUDA CGS residual norm squared",
                false,
            )?
            .sqrt();
            iterations = iteration;
            if self.invoke_monitors(iteration, rnorm, reductions) {
                final_reason = ConvergedReason::StoppedByMonitor;
                break;
            }
            let (reason, _) = self.convergence.check(rnorm, bnorm, iteration);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
                break;
            }

            let rho_new = residual_dots[1];
            ensure_nonzero_finite(rho_new, "CUDA CGS updated rho")?;
            let beta = rho_new / rho;
            ensure_finite(beta, "CUDA CGS beta")?;
            self.runtime.copy(q.buffer(), u.buffer_mut())?;
            self.runtime
                .axpby(S::one(), r.buffer(), beta, u.buffer_mut())?;
            self.runtime
                .axpby(S::one(), q.buffer(), beta, p.buffer_mut())?;
            self.runtime
                .axpby(S::one(), u.buffer(), beta, p.buffer_mut())?;
            rho = rho_new;
        }

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), r.buffer_mut())?;
        let true_residual = self.norm2(r)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        let mut stats = SolveStats::new(iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(rnorm);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: "cuda-cgs",
            startup: 2,
            per_iteration: 2.0,
            tail: 1,
        });
        stats.effective_variant = Some("cuda-right-preconditioned-cgs".into());
        Ok(stats.finalize_reason_counters())
    }

    /// CG on the normal equations, matching the host CGNR kernel. The host CR
    /// surface delegates to that same kernel, so both solver types share this
    /// implementation while retaining distinct diagnostics.
    fn solve_cgnr(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut CgnrWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let CgnrWorkspace {
            r,
            z,
            p,
            ap,
            zhat,
            ax,
            reduction_payload,
            ..
        } = workspace;
        let adjoint = if cfg!(feature = "complex") {
            CudaOperation::ConjugateTranspose
        } else {
            CudaOperation::Transpose
        };

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), r.buffer_mut())?;
        operator.apply(adjoint, r, z)?;
        pc.apply(z, zhat)?;
        self.runtime.copy(zhat.buffer(), p.buffer_mut())?;

        let bnorm = self.norm2(b)?.max(1e-32);
        let initial = self.dot2(z, zhat, r, r, reduction_payload)?;
        let mut reductions = 2usize;
        let mut rz = initial[0];
        let mut rnorm = nonnegative_hermitian_real(
            initial[1],
            "CUDA CGNR initial residual norm squared",
            false,
        )?
        .sqrt();
        let (initial_reason, mut initial_stats) = self.convergence.check(rnorm, bnorm, 0);
        if self.invoke_monitors(0, rnorm, reductions) {
            initial_stats.reason = ConvergedReason::StoppedByMonitor;
            initial_stats.final_true_residual = Some(rnorm);
            initial_stats.final_recurrence_residual = Some(rnorm);
            initial_stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(initial_stats.finalize_reason_counters());
        }
        if initial_reason != ConvergedReason::Continued {
            initial_stats.final_true_residual = Some(rnorm);
            initial_stats.final_recurrence_residual = Some(rnorm);
            initial_stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(initial_stats.finalize_reason_counters());
        }
        ensure_positive_real(rz, "CUDA CGNR z^H M^-1 z", true)?;

        let mut iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;
        for iteration in 1..=self.convergence.max_iters {
            operator.apply(CudaOperation::NonTranspose, p, ap)?;
            let ap_ap = self.dot(ap, ap)?;
            reductions += 1;
            ensure_positive_real(ap_ap, "CUDA CGNR (A p)^H A p", false)?;
            let alpha = rz / S::from_real(ap_ap.real());
            ensure_finite(alpha, "CUDA CGNR alpha")?;
            self.runtime
                .axpby(alpha, p.buffer(), S::one(), x.buffer_mut())?;
            self.runtime
                .axpby(-alpha, ap.buffer(), S::one(), r.buffer_mut())?;

            operator.apply(adjoint, r, z)?;
            pc.apply(z, zhat)?;
            let next = self.dot2(z, zhat, r, r, reduction_payload)?;
            reductions += 1;
            let rz_new = next[0];
            rnorm = nonnegative_hermitian_real(next[1], "CUDA CGNR residual norm squared", false)?
                .sqrt();
            iterations = iteration;
            if self.invoke_monitors(iteration, rnorm, reductions) {
                final_reason = ConvergedReason::StoppedByMonitor;
                break;
            }
            let (reason, _) = self.convergence.check(rnorm, bnorm, iteration);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
                break;
            }

            ensure_positive_real(rz_new, "CUDA CGNR updated z^H M^-1 z", true)?;
            let beta = rz_new / rz;
            ensure_finite(beta, "CUDA CGNR beta")?;
            self.runtime
                .axpby(S::one(), zhat.buffer(), beta, p.buffer_mut())?;
            rz = rz_new;
        }

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), r.buffer_mut())?;
        let true_residual = self.norm2(r)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        let is_cr = self.solver_type == SolverType::Cr;
        let mut stats = SolveStats::new(iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(rnorm);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: if is_cr {
                "cuda-cr-via-cgnr"
            } else {
                "cuda-cgnr"
            },
            startup: 2,
            per_iteration: 2.0,
            tail: 1,
        });
        stats.effective_variant = Some(if is_cr {
            "cuda-cr-via-cgnr".into()
        } else {
            "cuda-cgnr".into()
        });
        Ok(stats.finalize_reason_counters())
    }

    fn solve_lsqr(
        &self,
        operator: &dyn CudaLinOp,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut CgnrWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let CgnrWorkspace {
            r: u,
            z: v,
            p: w,
            ap: av,
            zhat: at_u,
            ax: true_residual_buffer,
            reduction_payload,
            ..
        } = workspace;
        let adjoint = if cfg!(feature = "complex") {
            CudaOperation::ConjugateTranspose
        } else {
            CudaOperation::Transpose
        };

        operator.apply(CudaOperation::NonTranspose, x, av)?;
        self.runtime.copy(b.buffer(), u.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), av.buffer(), S::one(), u.buffer_mut())?;
        let initial_norms = self.dot2(u, u, b, b, reduction_payload)?;
        let mut reductions = 1usize;
        let mut beta = nonnegative_hermitian_real(
            initial_norms[0],
            "CUDA LSQR initial residual norm squared",
            false,
        )?
        .sqrt();
        let bnorm = nonnegative_hermitian_real(
            initial_norms[1],
            "CUDA LSQR right-hand-side norm squared",
            false,
        )?
        .sqrt()
        .max(1e-32);
        if beta == 0.0 {
            let mut stats = SolveStats::new(0, 0.0, ConvergedReason::ConvergedAtol);
            stats.final_recurrence_residual = Some(0.0);
            stats.final_true_residual = Some(0.0);
            stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(stats.finalize_reason_counters());
        }
        self.runtime
            .scale(S::from_real(1.0 / beta), u.buffer_mut())?;

        operator.apply(adjoint, u, v)?;
        let mut alpha = self.norm2(v)?;
        reductions += 1;
        if alpha == 0.0 {
            let mut stats = SolveStats::new(0, beta, ConvergedReason::ConvergedAtol);
            stats.final_recurrence_residual = Some(beta);
            stats.final_true_residual = Some(beta);
            stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(stats.finalize_reason_counters());
        }
        self.runtime
            .scale(S::from_real(1.0 / alpha), v.buffer_mut())?;
        self.runtime.copy(v.buffer(), w.buffer_mut())?;

        let mut rho_bar = alpha;
        let mut phi_bar = beta;
        let recurrence_baseline = beta;
        let mut recurrence_residual = beta;
        if self.invoke_monitors(0, recurrence_residual, reductions) {
            let mut stats = SolveStats::new(0, beta, ConvergedReason::StoppedByMonitor);
            stats.final_recurrence_residual = Some(recurrence_residual);
            stats.final_true_residual = Some(beta);
            stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(stats.finalize_reason_counters());
        }

        let mut iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;
        for iteration in 1..=self.convergence.max_iters {
            operator.apply(CudaOperation::NonTranspose, v, av)?;
            self.runtime
                .axpby(S::one(), av.buffer(), S::from_real(-alpha), u.buffer_mut())?;
            beta = self.norm2(u)?;
            reductions += 1;
            if beta == 0.0 {
                iterations = iteration;
                final_reason = ConvergedReason::ConvergedAtol;
                break;
            }
            self.runtime
                .scale(S::from_real(1.0 / beta), u.buffer_mut())?;

            operator.apply(adjoint, u, at_u)?;
            self.runtime
                .axpby(S::one(), at_u.buffer(), S::from_real(-beta), v.buffer_mut())?;
            alpha = self.norm2(v)?;
            reductions += 1;
            if alpha == 0.0 {
                iterations = iteration;
                final_reason = ConvergedReason::ConvergedAtol;
                break;
            }
            self.runtime
                .scale(S::from_real(1.0 / alpha), v.buffer_mut())?;

            let rho = (rho_bar * rho_bar + beta * beta).sqrt();
            if !rho.is_finite() {
                return Err(KError::NonFiniteReduction {
                    kind: if rho.is_nan() {
                        crate::error::NonFiniteKind::Nan
                    } else {
                        crate::error::NonFiniteKind::Inf
                    },
                    context: "CUDA LSQR Givens denominator",
                });
            }
            if rho <= 0.0 {
                return Err(KError::BreakdownOrIndefinite);
            }
            let cosine = rho_bar / rho;
            let sine = beta / rho;
            let theta = sine * alpha;
            rho_bar = -cosine * alpha;
            let phi = cosine * phi_bar;
            phi_bar *= sine;

            self.runtime.axpby(
                S::from_real(phi / rho),
                w.buffer(),
                S::one(),
                x.buffer_mut(),
            )?;
            self.runtime.axpby(
                S::one(),
                v.buffer(),
                S::from_real(-theta / rho),
                w.buffer_mut(),
            )?;
            recurrence_residual = phi_bar.abs();
            iterations = iteration;

            operator.apply(CudaOperation::NonTranspose, x, true_residual_buffer)?;
            self.runtime
                .scale(-S::one(), true_residual_buffer.buffer_mut())?;
            self.runtime
                .axpy(S::one(), b.buffer(), true_residual_buffer.buffer_mut())?;
            let true_residual = self.norm2(true_residual_buffer)?;
            reductions += 1;
            if self.invoke_monitors(iteration, recurrence_residual, reductions) {
                final_reason = ConvergedReason::StoppedByMonitor;
                break;
            }
            let (recurrence_reason, _) =
                self.convergence
                    .check(recurrence_residual, recurrence_baseline, iteration);
            if matches!(
                recurrence_reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ) {
                let (true_reason, _) = self.convergence.check(true_residual, bnorm, iteration);
                if true_reason != ConvergedReason::Continued {
                    final_reason = true_reason;
                    break;
                }
            } else if recurrence_reason != ConvergedReason::Continued {
                final_reason = recurrence_reason;
                break;
            }
        }

        operator.apply(CudaOperation::NonTranspose, x, true_residual_buffer)?;
        self.runtime
            .scale(-S::one(), true_residual_buffer.buffer_mut())?;
        self.runtime
            .axpy(S::one(), b.buffer(), true_residual_buffer.buffer_mut())?;
        let true_residual = self.norm2(true_residual_buffer)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            final_reason = if true_residual <= self.convergence.atol {
                ConvergedReason::ConvergedAtol
            } else if true_residual / bnorm <= self.convergence.rtol * 10.0 {
                ConvergedReason::ConvergedRtol
            } else {
                ConvergedReason::DivergedMaxIts
            };
        }
        let mut stats = SolveStats::new(iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(recurrence_residual);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: "cuda-lsqr",
            startup: 2,
            per_iteration: 3.0,
            tail: 1,
        });
        stats.effective_variant = Some("cuda-golub-kahan-lsqr".into());
        Ok(stats.finalize_reason_counters())
    }

    fn solve_lsmr(
        &self,
        operator: &dyn CudaLinOp,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut CgnrWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let CgnrWorkspace {
            r: u,
            z: v,
            p: h,
            ap: av,
            zhat: at_u,
            extra: hbar,
            ax: true_residual_buffer,
            reduction_payload,
            ..
        } = workspace;
        let adjoint = if cfg!(feature = "complex") {
            CudaOperation::ConjugateTranspose
        } else {
            CudaOperation::Transpose
        };

        operator.apply(CudaOperation::NonTranspose, x, av)?;
        self.runtime.copy(b.buffer(), u.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), av.buffer(), S::one(), u.buffer_mut())?;
        let initial_norms = self.dot2(u, u, b, b, reduction_payload)?;
        let mut reductions = 1usize;
        let mut beta = nonnegative_hermitian_real(
            initial_norms[0],
            "CUDA LSMR initial residual norm squared",
            false,
        )?
        .sqrt();
        let bnorm = nonnegative_hermitian_real(
            initial_norms[1],
            "CUDA LSMR right-hand-side norm squared",
            false,
        )?
        .sqrt()
        .max(1e-32);
        if beta == 0.0 {
            return Ok(cuda_initial_convergence_stats(0.0, reductions));
        }
        self.runtime
            .scale(S::from_real(1.0 / beta), u.buffer_mut())?;
        operator.apply(adjoint, u, v)?;
        let mut alpha = self.norm2(v)?;
        reductions += 1;
        if alpha == 0.0 {
            return Ok(cuda_initial_convergence_stats(beta, reductions));
        }
        self.runtime
            .scale(S::from_real(1.0 / alpha), v.buffer_mut())?;

        let mut zetabar = alpha * beta;
        let mut alphabar = alpha;
        let mut rho = 1.0;
        let mut rhobar = 1.0;
        let mut cbar = 1.0;
        let mut sbar = 0.0;
        self.runtime.copy(v.buffer(), h.buffer_mut())?;
        hbar.fill_zero()?;

        let mut betadd = beta;
        let mut betad = 0.0;
        let mut rhodold = 1.0;
        let mut tautildeold = 0.0;
        let mut thetatilde = 0.0;
        let mut zeta = 0.0;
        let mut d = 0.0;
        let recurrence_baseline = beta;
        let mut recurrence_residual = beta;
        if self.invoke_monitors(0, recurrence_residual, reductions) {
            let mut stats = SolveStats::new(0, beta, ConvergedReason::StoppedByMonitor);
            stats.final_recurrence_residual = Some(beta);
            stats.final_true_residual = Some(beta);
            stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(stats.finalize_reason_counters());
        }

        let mut iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;
        for iteration in 1..=self.convergence.max_iters {
            operator.apply(CudaOperation::NonTranspose, v, av)?;
            self.runtime
                .axpby(S::one(), av.buffer(), S::from_real(-alpha), u.buffer_mut())?;
            beta = self.norm2(u)?;
            reductions += 1;
            if beta == 0.0 {
                iterations = iteration;
                final_reason = ConvergedReason::ConvergedAtol;
                break;
            }
            self.runtime
                .scale(S::from_real(1.0 / beta), u.buffer_mut())?;
            operator.apply(adjoint, u, at_u)?;
            self.runtime
                .axpby(S::one(), at_u.buffer(), S::from_real(-beta), v.buffer_mut())?;
            alpha = self.norm2(v)?;
            reductions += 1;
            if alpha == 0.0 {
                iterations = iteration;
                final_reason = ConvergedReason::ConvergedAtol;
                break;
            }
            self.runtime
                .scale(S::from_real(1.0 / alpha), v.buffer_mut())?;

            let (chat, shat, alphahat) = sym_ortho_cuda(alphabar, 0.0);
            let rhoold = rho;
            let (cosine, sine, new_rho) = sym_ortho_cuda(alphahat, beta);
            rho = new_rho;
            let thetanew = sine * alpha;
            alphabar = cosine * alpha;

            let rhobarold = rhobar;
            let zetaold = zeta;
            let thetabar = sbar * rho;
            let (new_cbar, new_sbar, new_rhobar) = sym_ortho_cuda(cbar * rho, thetanew);
            cbar = new_cbar;
            sbar = new_sbar;
            rhobar = new_rhobar;
            zeta = cbar * zetabar;
            zetabar = -sbar * zetabar;

            let hbar_denominator = rhoold * rhobarold;
            let x_denominator = rho * rhobar;
            if hbar_denominator.abs() <= 1e-30 || x_denominator.abs() <= 1e-30 {
                final_reason = ConvergedReason::DivergedBreakdown;
                iterations = iteration - 1;
                break;
            }
            let h_scale = -(thetabar * rho) / hbar_denominator;
            self.runtime.axpby(
                S::one(),
                h.buffer(),
                S::from_real(h_scale),
                hbar.buffer_mut(),
            )?;
            self.runtime.axpby(
                S::from_real(zeta / x_denominator),
                hbar.buffer(),
                S::one(),
                x.buffer_mut(),
            )?;
            self.runtime.axpby(
                S::one(),
                v.buffer(),
                S::from_real(-(thetanew / rho)),
                h.buffer_mut(),
            )?;

            let betaacute = chat * betadd;
            let betacheck = -shat * betadd;
            let betahat = cosine * betaacute;
            betadd = -sine * betaacute;
            let thetatildeold = thetatilde;
            let (ctildeold, stildeold, rhotildeold) = sym_ortho_cuda(rhodold, thetabar);
            thetatilde = stildeold * rhobar;
            rhodold = ctildeold * rhobar;
            betad = -stildeold * betad + ctildeold * betahat;
            if rhotildeold.abs() <= 1e-30 || rhodold.abs() <= 1e-30 {
                final_reason = ConvergedReason::DivergedBreakdown;
                iterations = iteration;
                break;
            }
            tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
            let taud = (zeta - thetatilde * tautildeold) / rhodold;
            d += betacheck * betacheck;
            recurrence_residual = (d + (betad - taud) * (betad - taud) + betadd * betadd).sqrt();
            if !recurrence_residual.is_finite() {
                return Err(KError::NonFiniteReduction {
                    kind: if recurrence_residual.is_nan() {
                        crate::error::NonFiniteKind::Nan
                    } else {
                        crate::error::NonFiniteKind::Inf
                    },
                    context: "CUDA LSMR recurrence residual",
                });
            }
            iterations = iteration;

            operator.apply(CudaOperation::NonTranspose, x, true_residual_buffer)?;
            self.runtime
                .scale(-S::one(), true_residual_buffer.buffer_mut())?;
            self.runtime
                .axpy(S::one(), b.buffer(), true_residual_buffer.buffer_mut())?;
            let true_residual = self.norm2(true_residual_buffer)?;
            reductions += 1;
            if self.invoke_monitors(iteration, recurrence_residual, reductions) {
                final_reason = ConvergedReason::StoppedByMonitor;
                break;
            }
            let (recurrence_reason, _) =
                self.convergence
                    .check(recurrence_residual, recurrence_baseline, iteration);
            if matches!(
                recurrence_reason,
                ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
            ) {
                let (true_reason, _) = self.convergence.check(true_residual, bnorm, iteration);
                if true_reason != ConvergedReason::Continued {
                    final_reason = true_reason;
                    break;
                }
            } else if recurrence_reason != ConvergedReason::Continued {
                final_reason = recurrence_reason;
                break;
            }
        }

        operator.apply(CudaOperation::NonTranspose, x, true_residual_buffer)?;
        self.runtime
            .scale(-S::one(), true_residual_buffer.buffer_mut())?;
        self.runtime
            .axpy(S::one(), b.buffer(), true_residual_buffer.buffer_mut())?;
        let true_residual = self.norm2(true_residual_buffer)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            final_reason = if true_residual <= self.convergence.atol {
                ConvergedReason::ConvergedAtol
            } else if true_residual / bnorm <= self.convergence.rtol * 10.0 {
                ConvergedReason::ConvergedRtol
            } else {
                ConvergedReason::DivergedMaxIts
            };
        }
        let mut stats = SolveStats::new(iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(recurrence_residual);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: "cuda-lsmr",
            startup: 2,
            per_iteration: 3.0,
            tail: 1,
        });
        stats.effective_variant = Some("cuda-golub-kahan-lsmr".into());
        Ok(stats.finalize_reason_counters())
    }

    fn solve_tfqmr_standard(
        &self,
        operator: &dyn CudaLinOp,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut QmrWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let QmrWorkspace {
            r: w,
            t: ax,
            r_tld: shadow,
            p: y1,
            p_tld: v,
            v: u1,
            v_tld: y2,
            s: direction,
            ax: u2,
            tmp_pc: true_residual_vector,
            reduction_payload,
            ..
        } = workspace;

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), w.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), w.buffer_mut())?;
        self.runtime.copy(w.buffer(), shadow.buffer_mut())?;
        self.runtime.copy(w.buffer(), y1.buffer_mut())?;
        operator.apply(CudaOperation::NonTranspose, y1, u1)?;
        self.runtime.copy(u1.buffer(), v.buffer_mut())?;
        direction.fill_zero()?;

        let initial = self.dot2(shadow, w, b, b, reduction_payload)?;
        let mut reductions = 1usize;
        let mut rho = initial[0];
        ensure_finite(rho, "CUDA TFQMR initial rho")?;
        let initial_residual = nonnegative_hermitian_real(
            initial[0],
            "CUDA TFQMR initial residual norm squared",
            false,
        )?
        .sqrt();
        let bnorm = nonnegative_hermitian_real(
            initial[1],
            "CUDA TFQMR right-hand-side norm squared",
            false,
        )?
        .sqrt()
        .max(1e-32);
        if self.invoke_monitors(0, initial_residual, reductions) {
            return Ok(cuda_tfqmr_stats(
                self.solver_type,
                0,
                initial_residual,
                initial_residual,
                ConvergedReason::StoppedByMonitor,
                reductions,
            ));
        }
        let (initial_reason, _) = self.convergence.check(initial_residual, bnorm, 0);
        if initial_reason != ConvergedReason::Continued {
            return Ok(cuda_tfqmr_stats(
                self.solver_type,
                0,
                initial_residual,
                initial_residual,
                initial_reason,
                reductions,
            ));
        }
        if rho.abs() <= 1e-30 {
            return Ok(cuda_tfqmr_stats(
                self.solver_type,
                0,
                initial_residual,
                initial_residual,
                ConvergedReason::DivergedBreakdown,
                reductions,
            ));
        }

        let mut tau = initial_residual;
        let mut theta_previous = R::zero();
        let mut eta_previous = S::zero();
        let mut iterations = 0usize;
        let mut recurrence_residual = initial_residual;
        let mut final_reason = ConvergedReason::DivergedMaxIts;

        'solve: while iterations < self.convergence.max_iters {
            let sigma = self.dot(shadow, v)?;
            reductions += 1;
            if sigma.abs() <= 1e-30 {
                final_reason = ConvergedReason::DivergedBreakdown;
                break;
            }
            let alpha = rho / sigma;
            ensure_finite(alpha, "CUDA TFQMR alpha")?;

            for half_step in 0..2 {
                if iterations >= self.convergence.max_iters {
                    break;
                }
                if half_step == 0 {
                    self.runtime
                        .axpby(-alpha, u1.buffer(), S::one(), w.buffer_mut())?;
                } else {
                    self.runtime.copy(y1.buffer(), y2.buffer_mut())?;
                    self.runtime
                        .axpby(-alpha, v.buffer(), S::one(), y2.buffer_mut())?;
                    operator.apply(CudaOperation::NonTranspose, y2, u2)?;
                    self.runtime
                        .axpby(-alpha, u2.buffer(), S::one(), w.buffer_mut())?;
                }

                let source = if half_step == 0 { &*y1 } else { &*y2 };
                let coefficient = if iterations == 0 {
                    S::zero()
                } else {
                    S::from_real(theta_previous * theta_previous) * (eta_previous / alpha)
                };
                ensure_finite(coefficient, "CUDA TFQMR direction coefficient")?;
                self.runtime.axpby(
                    S::one(),
                    source.buffer(),
                    coefficient,
                    direction.buffer_mut(),
                )?;

                let wnorm = self.norm2(w)?;
                reductions += 1;
                let theta = wnorm / tau.max(1e-300);
                let cosine = 1.0 / (1.0 + theta * theta).sqrt();
                tau *= theta * cosine;
                let eta = S::from_real(cosine * cosine) * alpha;
                self.runtime.axpy(eta, direction.buffer(), x.buffer_mut())?;
                theta_previous = theta;
                eta_previous = eta;
                iterations += 1;
                recurrence_residual = ((iterations + 1) as R).sqrt() * tau;

                if self.invoke_monitors(iterations, recurrence_residual, reductions) {
                    final_reason = ConvergedReason::StoppedByMonitor;
                    break 'solve;
                }
                let (reason, _) = self
                    .convergence
                    .check(recurrence_residual, bnorm, iterations);
                if reason == ConvergedReason::Continued {
                    continue;
                }
                if !reason.is_converged() {
                    final_reason = reason;
                    break 'solve;
                }

                operator.apply(CudaOperation::NonTranspose, x, ax)?;
                self.runtime
                    .copy(b.buffer(), true_residual_vector.buffer_mut())?;
                self.runtime.axpby(
                    -S::one(),
                    ax.buffer(),
                    S::one(),
                    true_residual_vector.buffer_mut(),
                )?;
                let true_residual = self.norm2(true_residual_vector)?;
                reductions += 1;
                let (true_reason, _) = self.convergence.check(true_residual, bnorm, iterations);
                if true_reason.is_converged() {
                    recurrence_residual = true_residual;
                    final_reason = true_reason;
                    break 'solve;
                }
                if true_reason != ConvergedReason::Continued {
                    recurrence_residual = true_residual;
                    final_reason = true_reason;
                    break 'solve;
                }
            }

            if iterations >= self.convergence.max_iters {
                break;
            }
            let rho_new = self.dot(shadow, w)?;
            reductions += 1;
            ensure_finite(rho_new, "CUDA TFQMR updated rho")?;
            if rho_new.abs() <= 1e-30 {
                final_reason = ConvergedReason::DivergedBreakdown;
                break;
            }
            let beta = rho_new / rho;
            ensure_finite(beta, "CUDA TFQMR beta")?;
            rho = rho_new;

            self.runtime.scale(beta, y2.buffer_mut())?;
            self.runtime.axpy(S::one(), w.buffer(), y2.buffer_mut())?;
            self.runtime.copy(y2.buffer(), y1.buffer_mut())?;
            operator.apply(CudaOperation::NonTranspose, y1, u1)?;
            self.runtime.scale(beta * beta, v.buffer_mut())?;
            self.runtime.axpy(beta, u2.buffer(), v.buffer_mut())?;
            self.runtime.axpy(S::one(), u1.buffer(), v.buffer_mut())?;
        }

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime
            .copy(b.buffer(), true_residual_vector.buffer_mut())?;
        self.runtime.axpby(
            -S::one(),
            ax.buffer(),
            S::one(),
            true_residual_vector.buffer_mut(),
        )?;
        let true_residual = self.norm2(true_residual_vector)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        Ok(cuda_tfqmr_stats(
            self.solver_type,
            iterations,
            true_residual,
            recurrence_residual,
            final_reason,
            reductions,
        ))
    }

    #[allow(dead_code)]
    fn solve_tfqmr(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut QmrWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let QmrWorkspace {
            r,
            t: au,
            r_tld,
            p: u,
            p_tld: v,
            v: wv,
            v_tld: yv,
            s: d,
            ax: qv,
            tmp_pc,
            reduction_payload,
            ..
        } = workspace;

        operator.apply(CudaOperation::NonTranspose, x, au)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), au.buffer(), S::one(), r.buffer_mut())?;
        pc.apply(r, tmp_pc)?;
        self.runtime.copy(tmp_pc.buffer(), r.buffer_mut())?;
        self.runtime.copy(r.buffer(), r_tld.buffer_mut())?;

        let initial = self.dot2(r_tld, r, b, b, reduction_payload)?;
        let mut reductions = 1usize;
        let mut rho = initial[0];
        ensure_finite(rho, "CUDA TFQMR initial rho")?;
        let mut dpold = nonnegative_hermitian_real(
            initial[0],
            "CUDA TFQMR initial residual norm squared",
            false,
        )?
        .sqrt();
        let bnorm = nonnegative_hermitian_real(
            initial[1],
            "CUDA TFQMR right-hand-side norm squared",
            false,
        )?
        .sqrt()
        .max(1e-32);
        if self.invoke_monitors(0, dpold, reductions) {
            return Ok(cuda_tfqmr_stats(
                self.solver_type,
                0,
                dpold,
                dpold,
                ConvergedReason::StoppedByMonitor,
                reductions,
            ));
        }
        let (initial_reason, _) = self.convergence.check(dpold, bnorm, 0);
        if initial_reason != ConvergedReason::Continued {
            return Ok(cuda_tfqmr_stats(
                self.solver_type,
                0,
                dpold,
                dpold,
                initial_reason,
                reductions,
            ));
        }
        if rho.abs() <= 1e-30 {
            return Ok(cuda_tfqmr_stats(
                self.solver_type,
                0,
                dpold,
                dpold,
                ConvergedReason::DivergedBreakdown,
                reductions,
            ));
        }

        self.runtime.copy(r.buffer(), yv.buffer_mut())?;
        self.runtime.copy(r.buffer(), wv.buffer_mut())?;
        d.fill_zero()?;
        let mut theta_prev = R::zero();
        let mut eta_prev = S::zero();
        let mut recurrence_residual = dpold;
        let mut iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;

        'solve: while iterations < self.convergence.max_iters {
            let outer = iterations / 2 + 1;
            operator.apply(CudaOperation::NonTranspose, yv, v)?;
            pc.apply(v, tmp_pc)?;
            self.runtime.copy(tmp_pc.buffer(), v.buffer_mut())?;

            let sigma = self.dot(r_tld, v)?;
            reductions += 1;
            if sigma.abs() <= 1e-30 {
                final_reason = ConvergedReason::DivergedBreakdown;
                break;
            }
            let alpha = rho / sigma;
            ensure_finite(alpha, "CUDA TFQMR alpha")?;
            if alpha.abs() <= R::zero() {
                final_reason = ConvergedReason::DivergedBreakdown;
                break;
            }

            self.runtime.copy(r.buffer(), u.buffer_mut())?;
            self.runtime
                .axpby(-alpha, v.buffer(), S::one(), u.buffer_mut())?;
            let u_norm = self.norm2(u)?;
            reductions += 1;
            let mut tau = (u_norm * dpold).sqrt();

            for half_step in 0..2 {
                if iterations >= self.convergence.max_iters {
                    break;
                }
                if half_step == 0 {
                    self.runtime.copy(u.buffer(), qv.buffer_mut())?;
                    self.runtime
                        .axpby(-alpha, v.buffer(), S::one(), qv.buffer_mut())?;
                }

                self.runtime.copy(u.buffer(), tmp_pc.buffer_mut())?;
                self.runtime
                    .axpby(S::one(), qv.buffer(), S::one(), tmp_pc.buffer_mut())?;
                operator.apply(CudaOperation::NonTranspose, tmp_pc, au)?;
                pc.apply(au, tmp_pc)?;
                self.runtime.copy(tmp_pc.buffer(), au.buffer_mut())?;
                self.runtime
                    .axpby(-alpha, au.buffer(), S::one(), r.buffer_mut())?;

                let src = if half_step == 0 { &*u } else { &*qv };
                let src_norm = if half_step == 0 {
                    u_norm
                } else {
                    reductions += 1;
                    self.norm2(src)?
                };
                let psi = src_norm / tau.max(1e-300);
                let cosine = 1.0 / (1.0 + psi * psi).sqrt();
                let eta = S::from_real(cosine * cosine) * alpha;
                let coefficient = if outer == 1 && half_step == 0 {
                    S::zero()
                } else {
                    S::from_real(theta_prev * theta_prev) * (eta_prev / alpha)
                };
                ensure_finite(coefficient, "CUDA TFQMR direction coefficient")?;
                self.runtime
                    .axpby(S::one(), src.buffer(), coefficient, d.buffer_mut())?;
                self.runtime.axpy(eta, d.buffer(), x.buffer_mut())?;

                iterations += 1;
                recurrence_residual = ((iterations + 2) as R).sqrt() * tau;
                theta_prev = psi;
                eta_prev = eta;
                tau *= psi * cosine;

                if self.invoke_monitors(iterations, recurrence_residual, reductions) {
                    final_reason = ConvergedReason::StoppedByMonitor;
                    break 'solve;
                }
                let (reason, _) = self
                    .convergence
                    .check(recurrence_residual, bnorm, iterations);
                if reason != ConvergedReason::Continued {
                    if reason.is_converged() {
                        operator.apply(CudaOperation::NonTranspose, x, au)?;
                        self.runtime.copy(b.buffer(), tmp_pc.buffer_mut())?;
                        self.runtime.axpby(
                            -S::one(),
                            au.buffer(),
                            S::one(),
                            tmp_pc.buffer_mut(),
                        )?;
                        let true_residual = self.norm2(tmp_pc)?;
                        reductions += 1;
                        let (true_reason, _) =
                            self.convergence.check(true_residual, bnorm, iterations);
                        if true_reason.is_converged() {
                            recurrence_residual = true_residual;
                            final_reason = true_reason;
                            break 'solve;
                        }
                    } else {
                        final_reason = reason;
                        break 'solve;
                    }
                }

                if half_step == 0 {
                    self.runtime
                        .axpby(-alpha, v.buffer(), S::one(), qv.buffer_mut())?;
                    self.runtime
                        .axpby(-alpha, v.buffer(), S::one(), u.buffer_mut())?;
                }
            }

            if iterations >= self.convergence.max_iters {
                break;
            }
            let update = self.dot2(r_tld, r, r, r, reduction_payload)?;
            reductions += 1;
            let rho_new = update[0];
            ensure_finite(rho_new, "CUDA TFQMR updated rho")?;
            if rho_new.abs() <= 1e-30 {
                final_reason = ConvergedReason::DivergedBreakdown;
                break;
            }
            let beta = rho_new / rho;
            ensure_finite(beta, "CUDA TFQMR beta")?;
            rho = rho_new;

            self.runtime.scale(beta * beta, wv.buffer_mut())?;
            self.runtime.axpy(beta, qv.buffer(), wv.buffer_mut())?;
            self.runtime.axpy(S::one(), r.buffer(), wv.buffer_mut())?;
            self.runtime.scale(beta * beta, yv.buffer_mut())?;
            self.runtime.axpy(beta, qv.buffer(), yv.buffer_mut())?;
            self.runtime.axpy(S::one(), r.buffer(), yv.buffer_mut())?;
            dpold =
                nonnegative_hermitian_real(update[1], "CUDA TFQMR residual norm squared", false)?
                    .sqrt();
        }

        operator.apply(CudaOperation::NonTranspose, x, au)?;
        self.runtime.copy(b.buffer(), tmp_pc.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), au.buffer(), S::one(), tmp_pc.buffer_mut())?;
        let true_residual = self.norm2(tmp_pc)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        Ok(cuda_tfqmr_stats(
            self.solver_type,
            iterations,
            true_residual,
            recurrence_residual,
            final_reason,
            reductions,
        ))
    }

    fn solve_qmr(
        &self,
        operator: &dyn CudaLinOp,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut QmrWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let QmrWorkspace {
            r,
            t,
            r_tld,
            p,
            p_tld,
            v,
            v_tld,
            s,
            ax,
            reduction_payload,
            ..
        } = workspace;
        let adjoint = if cfg!(feature = "complex") {
            CudaOperation::ConjugateTranspose
        } else {
            CudaOperation::Transpose
        };

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), r.buffer_mut())?;
        self.runtime.copy(r.buffer(), r_tld.buffer_mut())?;
        let initial = self.dot2(r, r, b, b, reduction_payload)?;
        let mut reductions = 1usize;
        let mut rnorm = nonnegative_hermitian_real(
            initial[0],
            "CUDA QMR initial residual norm squared",
            false,
        )?
        .sqrt();
        let bnorm =
            nonnegative_hermitian_real(initial[1], "CUDA QMR right-hand-side norm squared", false)?
                .sqrt()
                .max(1e-32);
        let (initial_reason, _) = self.convergence.check(rnorm, bnorm, 0);
        if self.invoke_monitors(0, rnorm, reductions) {
            return Ok(cuda_qmr_stats(
                0,
                rnorm,
                rnorm,
                ConvergedReason::StoppedByMonitor,
                reductions,
            ));
        }
        if initial_reason != ConvergedReason::Continued {
            return Ok(cuda_qmr_stats(0, rnorm, rnorm, initial_reason, reductions));
        }

        let mut rho = self.dot(r_tld, r)?;
        reductions += 1;
        let mut iterations = 0usize;
        let mut final_reason = if rho.abs() <= 1e-30 {
            ConvergedReason::DivergedBreakdown
        } else {
            ConvergedReason::DivergedMaxIts
        };

        if final_reason != ConvergedReason::DivergedBreakdown {
            for k in 0..self.convergence.max_iters {
                if k == 0 {
                    self.runtime.copy(r.buffer(), p.buffer_mut())?;
                    self.runtime.copy(r_tld.buffer(), p_tld.buffer_mut())?;
                } else {
                    let rho_new = self.dot(r_tld, r)?;
                    reductions += 1;
                    if rho_new.abs() <= 1e-30 {
                        final_reason = ConvergedReason::DivergedBreakdown;
                        break;
                    }
                    let beta = rho_new / rho;
                    ensure_finite(beta, "CUDA QMR beta")?;
                    self.runtime
                        .axpby(S::one(), r.buffer(), beta, p.buffer_mut())?;
                    self.runtime
                        .axpby(S::one(), r_tld.buffer(), beta, p_tld.buffer_mut())?;
                    rho = rho_new;
                }

                operator.apply(CudaOperation::NonTranspose, p, v)?;
                operator.apply(adjoint, p_tld, v_tld)?;
                let sigma = self.dot(p_tld, v)?;
                reductions += 1;
                if sigma.abs() <= 1e-30 {
                    final_reason = ConvergedReason::DivergedBreakdown;
                    break;
                }
                let alpha = rho / sigma;
                ensure_finite(alpha, "CUDA QMR alpha")?;

                self.runtime.copy(r.buffer(), s.buffer_mut())?;
                self.runtime
                    .axpby(-alpha, v.buffer(), S::one(), s.buffer_mut())?;
                operator.apply(CudaOperation::NonTranspose, s, t)?;
                let pair = self.dot2(t, t, t, s, reduction_payload)?;
                reductions += 1;
                let tt = nonnegative_hermitian_real(pair[0], "CUDA QMR t norm squared", false)?;
                if tt <= 1e-30 {
                    final_reason = ConvergedReason::DivergedBreakdown;
                    break;
                }
                ensure_finite(pair[1], "CUDA QMR t-s projection")?;
                let omega = pair[1] / S::from_real(tt);
                ensure_finite(omega, "CUDA QMR omega")?;

                self.runtime.axpy(alpha, p.buffer(), x.buffer_mut())?;
                self.runtime.axpy(omega, s.buffer(), x.buffer_mut())?;
                self.runtime.copy(s.buffer(), r.buffer_mut())?;
                self.runtime
                    .axpby(-omega, t.buffer(), S::one(), r.buffer_mut())?;
                self.runtime.copy(s.buffer(), r_tld.buffer_mut())?;
                self.runtime
                    .axpby(-omega.conj(), t.buffer(), S::one(), r_tld.buffer_mut())?;

                rnorm = self.norm2(r)?;
                reductions += 1;
                iterations = k + 1;
                if self.invoke_monitors(iterations, rnorm, reductions) {
                    final_reason = ConvergedReason::StoppedByMonitor;
                    break;
                }
                let (reason, _) = self.convergence.check(rnorm, bnorm, iterations);
                if reason != ConvergedReason::Continued {
                    final_reason = reason;
                    break;
                }
            }
        }

        operator.apply(CudaOperation::NonTranspose, x, ax)?;
        self.runtime.copy(b.buffer(), r.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), ax.buffer(), S::one(), r.buffer_mut())?;
        let true_residual = self.norm2(r)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        Ok(cuda_qmr_stats(
            iterations,
            true_residual,
            rnorm,
            final_reason,
            reductions,
        ))
    }

    fn solve_richardson(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut RichardsonWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let (omega, reduction_variant) = if self.solver_type == SolverType::Chebyshev {
            (self.chebyshev_omega, "cuda-fused-chebyshev")
        } else {
            (self.richardson_omega, "cuda-fused-richardson")
        };
        let RichardsonWorkspace {
            residual,
            correction,
            ax,
            ..
        } = workspace;
        let bnorm = self.norm2(b)?;
        let mut reductions = 1usize;
        let mut rnorm = R::zero();
        let mut iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;

        for iteration in 0..=self.convergence.max_iters {
            operator.apply(CudaOperation::NonTranspose, x, ax)?;
            self.runtime.copy(b.buffer(), residual.buffer_mut())?;
            self.runtime
                .axpby(-S::one(), ax.buffer(), S::one(), residual.buffer_mut())?;
            rnorm = self.norm2(residual)?;
            reductions += 1;
            iterations = iteration;
            if self.invoke_monitors(iteration, rnorm, reductions) {
                final_reason = ConvergedReason::StoppedByMonitor;
                break;
            }
            let (reason, _) = self.convergence.check(rnorm, bnorm, iteration);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
                break;
            }
            if iteration == self.convergence.max_iters {
                break;
            }
            pc.apply(residual, correction)?;
            self.runtime.axpby(
                S::from_real(omega),
                correction.buffer(),
                S::one(),
                x.buffer_mut(),
            )?;
        }

        let mut stats = SolveStats::new(iterations, rnorm, final_reason);
        stats.final_recurrence_residual = Some(rnorm);
        stats.final_true_residual = Some(rnorm);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: reduction_variant,
            startup: 1,
            per_iteration: 1.0,
            tail: 0,
        });
        stats.effective_variant = Some("cuda-fused-stationary".into());
        Ok(stats.finalize_reason_counters())
    }

    fn solve_pipegcr(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut GmresWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let m = workspace.restart;
        let GmresWorkspace {
            residual,
            work,
            temp,
            basis: ap_basis,
            correction_basis: p_basis,
            h: projections,
            y: ap_norms,
            reduction_payload,
            ..
        } = workspace;

        operator.apply(CudaOperation::NonTranspose, x, temp)?;
        self.runtime.copy(b.buffer(), residual.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), temp.buffer(), S::one(), residual.buffer_mut())?;
        let initial = self.dot2(residual, residual, b, b, reduction_payload)?;
        let mut reductions = 1usize;
        let mut residual_sq = nonnegative_hermitian_real(
            initial[0],
            "CUDA PipeGCR initial residual norm squared",
            false,
        )?;
        let bnorm = nonnegative_hermitian_real(
            initial[1],
            "CUDA PipeGCR right-hand-side norm squared",
            false,
        )?
        .sqrt()
        .max(1e-32);
        let mut recurrence_residual = residual_sq.sqrt();
        let (initial_reason, _) = self.convergence.check(recurrence_residual, bnorm, 0);
        if self.invoke_monitors(0, recurrence_residual, reductions) {
            return Ok(cuda_pipegcr_stats(
                0,
                recurrence_residual,
                recurrence_residual,
                ConvergedReason::StoppedByMonitor,
                reductions,
                0,
                0,
                m,
            ));
        }
        if initial_reason != ConvergedReason::Continued {
            return Ok(cuda_pipegcr_stats(
                0,
                recurrence_residual,
                recurrence_residual,
                initial_reason,
                reductions,
                0,
                0,
                m,
            ));
        }

        let mut iterations = 0usize;
        let mut basis_updates = 0usize;
        let mut restart_count = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;

        'solve: while iterations < self.convergence.max_iters {
            let cycle = m.min(self.convergence.max_iters - iterations);
            let mut restart_from_verified_residual = false;

            for j in 0..cycle {
                pc.apply(residual, &mut p_basis[j])?;
                operator.apply(CudaOperation::NonTranspose, &p_basis[j], &mut ap_basis[j])?;

                if j > 0 {
                    self.basis_dots(&ap_basis[..j], &ap_basis[j], &mut projections[..j])?;
                    reductions += 1;
                    let (prior_p, current_p) = p_basis.split_at_mut(j);
                    let p_j = &mut current_p[0];
                    let (prior_ap, current_ap) = ap_basis.split_at_mut(j);
                    let ap_j = &mut current_ap[0];
                    for i in 0..j {
                        ensure_positive_real(
                            ap_norms[i],
                            "CUDA PipeGCR stored A p norm squared",
                            false,
                        )?;
                        let beta = projections[i] / ap_norms[i];
                        ensure_finite(beta, "CUDA PipeGCR orthogonalization coefficient")?;
                        self.runtime.axpby(
                            -beta,
                            prior_p[i].buffer(),
                            S::one(),
                            p_j.buffer_mut(),
                        )?;
                        self.runtime.axpby(
                            -beta,
                            prior_ap[i].buffer(),
                            S::one(),
                            ap_j.buffer_mut(),
                        )?;
                    }
                }

                let pair = self.dot2(
                    &ap_basis[j],
                    residual,
                    &ap_basis[j],
                    &ap_basis[j],
                    reduction_payload,
                )?;
                reductions += 1;
                ensure_finite(pair[0], "CUDA PipeGCR residual projection")?;
                let ap_norm_sq =
                    nonnegative_hermitian_real(pair[1], "CUDA PipeGCR A p norm squared", false)?;
                if ap_norm_sq <= 1e-30 {
                    final_reason = ConvergedReason::DivergedBreakdown;
                    break 'solve;
                }
                ap_norms[j] = S::from_real(ap_norm_sq);
                let alpha = pair[0] / ap_norms[j];
                ensure_finite(alpha, "CUDA PipeGCR step coefficient")?;
                self.runtime
                    .axpby(alpha, p_basis[j].buffer(), S::one(), x.buffer_mut())?;
                self.runtime.axpby(
                    -alpha,
                    ap_basis[j].buffer(),
                    S::one(),
                    residual.buffer_mut(),
                )?;

                let decrease = pair[0].abs() * pair[0].abs() / ap_norm_sq;
                residual_sq = (residual_sq - decrease).max(0.0);
                recurrence_residual = residual_sq.sqrt();
                iterations += 1;
                basis_updates += 1;

                if self.invoke_monitors(iterations, recurrence_residual, reductions) {
                    final_reason = ConvergedReason::StoppedByMonitor;
                    break 'solve;
                }
                let (reason, _) = self
                    .convergence
                    .check(recurrence_residual, bnorm, iterations);
                if reason == ConvergedReason::Continued {
                    continue;
                }
                if !reason.is_converged() {
                    final_reason = reason;
                    break 'solve;
                }

                // The recurrence norm avoids a third collective in ordinary
                // iterations. Verify it before accepting convergence.
                operator.apply(CudaOperation::NonTranspose, x, temp)?;
                self.runtime.copy(b.buffer(), residual.buffer_mut())?;
                self.runtime
                    .axpby(-S::one(), temp.buffer(), S::one(), residual.buffer_mut())?;
                let true_residual = self.norm2(residual)?;
                reductions += 1;
                residual_sq = true_residual * true_residual;
                let (true_reason, _) = self.convergence.check(true_residual, bnorm, iterations);
                if true_reason.is_converged() {
                    recurrence_residual = true_residual;
                    final_reason = true_reason;
                    break 'solve;
                }
                if true_reason != ConvergedReason::Continued {
                    recurrence_residual = true_residual;
                    final_reason = true_reason;
                    break 'solve;
                }
                recurrence_residual = true_residual;
                restart_from_verified_residual = true;
                break;
            }

            if iterations >= self.convergence.max_iters {
                break;
            }

            if !restart_from_verified_residual {
                operator.apply(CudaOperation::NonTranspose, x, temp)?;
                self.runtime.copy(b.buffer(), residual.buffer_mut())?;
                self.runtime
                    .axpby(-S::one(), temp.buffer(), S::one(), residual.buffer_mut())?;
                recurrence_residual = self.norm2(residual)?;
                reductions += 1;
                residual_sq = recurrence_residual * recurrence_residual;
                let (reason, _) = self
                    .convergence
                    .check(recurrence_residual, bnorm, iterations);
                if reason != ConvergedReason::Continued {
                    final_reason = reason;
                    break;
                }
            }
            restart_count += 1;
        }

        operator.apply(CudaOperation::NonTranspose, x, temp)?;
        self.runtime.copy(b.buffer(), work.buffer_mut())?;
        self.runtime
            .axpby(-S::one(), temp.buffer(), S::one(), work.buffer_mut())?;
        let true_residual = self.norm2(work)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        Ok(cuda_pipegcr_stats(
            iterations,
            true_residual,
            recurrence_residual,
            final_reason,
            reductions,
            basis_updates,
            restart_count,
            m,
        ))
    }

    fn solve_gmres(
        &self,
        operator: &dyn CudaLinOp,
        pc: &dyn CudaPreconditioner,
        b: &CudaVector,
        x: &mut CudaVector,
        workspace: &mut GmresWorkspace,
    ) -> Result<SolveStats<R>, KError> {
        let GmresWorkspace {
            residual,
            work,
            temp,
            basis,
            correction_basis,
            h,
            cs,
            sn,
            g,
            y,
            reduction_payload,
            host_reduction_payload,
            basis_ptrs,
            restart: m,
            ..
        } = workspace;
        let m = *m;
        let bnorm = self.norm2(b)?;
        let mut reductions = 1usize;
        let mut total_iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;
        let mut recurrence_residual = R::default();

        while total_iterations < self.convergence.max_iters {
            operator.apply(CudaOperation::NonTranspose, x, temp)?;
            self.runtime.copy(b.buffer(), residual.buffer_mut())?;
            self.runtime
                .axpy(-S::one(), temp.buffer(), residual.buffer_mut())?;

            let true_cycle_residual = self.norm2(residual)?;
            reductions += 1;

            if self.pc_side == PcSide::Left {
                pc.apply(residual, work)?;
                self.runtime.copy(work.buffer(), basis[0].buffer_mut())?;
            } else {
                self.runtime
                    .copy(residual.buffer(), basis[0].buffer_mut())?;
            }
            let beta = if self.pc_side == PcSide::Left {
                reductions += 1;
                self.norm2(&basis[0])?
            } else {
                true_cycle_residual
            };
            recurrence_residual = beta;
            let (reason, _) = self
                .convergence
                .check(true_cycle_residual, bnorm, total_iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
                break;
            }
            self.runtime
                .scale(S::from_real(1.0 / beta), basis[0].buffer_mut())?;
            h.fill(S::zero());
            cs.fill(R::default());
            sn.fill(S::zero());
            g.fill(S::zero());
            g[0] = S::from_real(beta);
            let cycle = m.min(self.convergence.max_iters - total_iterations);
            let mut used = 0usize;

            for j in 0..cycle {
                if self.pc_side == PcSide::Right {
                    pc.apply(&basis[j], &mut correction_basis[j])?;
                    operator.apply(CudaOperation::NonTranspose, &correction_basis[j], work)?;
                } else {
                    self.runtime
                        .copy(basis[j].buffer(), correction_basis[j].buffer_mut())?;
                    operator.apply(CudaOperation::NonTranspose, &basis[j], temp)?;
                    pc.apply(temp, work)?;
                }

                let hcol = &mut h[j * (m + 1)..(j + 1) * (m + 1)];
                let next_norm = if self.gmres_variant == CudaGmresVariant::Pipelined
                    && self.solver_type != SolverType::Gcr
                {
                    let next_norm = self.pipelined_arnoldi_step(
                        &basis[..=j],
                        work,
                        &mut hcol[..j + 2],
                        reduction_payload,
                        host_reduction_payload,
                        basis_ptrs,
                    )?;
                    reductions += 1;
                    next_norm
                } else {
                    for i in 0..=j {
                        hcol[i] = self.dot(&basis[i], work)?;
                        reductions += 1;
                        self.runtime
                            .axpy(-hcol[i], basis[i].buffer(), work.buffer_mut())?;
                    }
                    let next_norm = self.norm2(work)?;
                    reductions += 1;
                    next_norm
                };
                hcol[j + 1] = S::from_real(next_norm);
                if next_norm > R::default() {
                    self.runtime
                        .copy(work.buffer(), basis[j + 1].buffer_mut())?;
                    self.runtime
                        .scale(S::from_real(1.0 / next_norm), basis[j + 1].buffer_mut())?;
                }
                apply_prev_givens_to_col(hcol, j, &cs, &sn);
                apply_new_givens_and_update_g(hcol, j, cs, sn, g);
                recurrence_residual = g[j + 1].abs();
                total_iterations += 1;
                used = j + 1;
                if self.invoke_monitors(total_iterations, recurrence_residual, reductions) {
                    final_reason = ConvergedReason::StoppedByMonitor;
                    break;
                }
                let (reason, _) =
                    self.convergence
                        .check(recurrence_residual, bnorm, total_iterations);
                if reason != ConvergedReason::Continued {
                    final_reason = reason;
                    break;
                }
                if next_norm <= R::default() {
                    final_reason = ConvergedReason::ConvergedHappyBreakdown;
                    break;
                }
            }

            backsolve_hessenberg(h, m + 1, g, used, y)?;
            for j in 0..used {
                self.runtime
                    .axpy(y[j], correction_basis[j].buffer(), x.buffer_mut())?;
            }

            operator.apply(CudaOperation::NonTranspose, x, temp)?;
            self.runtime.copy(b.buffer(), residual.buffer_mut())?;
            self.runtime
                .axpy(-S::one(), temp.buffer(), residual.buffer_mut())?;
            let true_residual = self.norm2(residual)?;
            reductions += 1;
            let (true_reason, _) = self
                .convergence
                .check(true_residual, bnorm, total_iterations);
            if true_reason != ConvergedReason::Continued {
                final_reason = true_reason;
                break;
            }
            match final_reason {
                ConvergedReason::StoppedByMonitor => break,
                ConvergedReason::ConvergedHappyBreakdown => {
                    final_reason = ConvergedReason::DivergedArnoldiRankLoss;
                    break;
                }
                _ => {
                    // A recurrence estimate may be optimistic. Restart from
                    // the verified residual instead of reporting convergence.
                    final_reason = ConvergedReason::DivergedMaxIts;
                }
            }
        }

        operator.apply(CudaOperation::NonTranspose, x, temp)?;
        self.runtime.copy(b.buffer(), residual.buffer_mut())?;
        self.runtime
            .axpy(-S::one(), temp.buffer(), residual.buffer_mut())?;
        let true_residual = self.norm2(residual)?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self
                .convergence
                .check(true_residual, bnorm, total_iterations);
            final_reason = if reason == ConvergedReason::Continued {
                ConvergedReason::DivergedMaxIts
            } else {
                reason
            };
        }
        let mut stats = SolveStats::new(total_iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(recurrence_residual);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        let pipelined = self.gmres_variant == CudaGmresVariant::Pipelined
            && self.solver_type != SolverType::Gcr;
        stats.reduction_model = Some(ReductionModel {
            variant: if self.solver_type == SolverType::Gcr {
                "cuda-gcr-via-fgmres"
            } else if pipelined {
                "cuda-pipelined-cgs-gmres"
            } else {
                "cuda-restarted-mgs-gmres"
            },
            startup: 1,
            per_iteration: if pipelined {
                1.0
            } else {
                (m as f64 + 3.0) / 2.0
            },
            tail: 1,
        });
        stats.effective_variant = Some(if self.solver_type == SolverType::Gcr {
            "cuda-gcr-via-fgmres".into()
        } else if pipelined {
            "cuda-pipelined-cgs".into()
        } else {
            "cuda-classical-mgs".into()
        });
        stats.effective_restart = Some(m);
        Ok(stats.finalize_reason_counters())
    }
}

fn pc_csr_operator(pmat: &dyn CudaLinOp) -> Result<&CudaCsrOp, KError> {
    pmat.as_any()
        .downcast_ref::<CudaCsrOp>()
        .or_else(|| {
            pmat.as_any()
                .downcast_ref::<CudaDistCsrOp>()
                .map(CudaDistCsrOp::diagonal_block)
        })
        .ok_or(KError::Unsupported(
            "CUDA Jacobi-family setup requires CudaCsrOp or CudaDistCsrOp",
        ))
}

fn cuda_initial_convergence_stats(residual: R, reductions: usize) -> SolveStats<R> {
    let mut stats = SolveStats::new(0, residual, ConvergedReason::ConvergedAtol);
    stats.final_recurrence_residual = Some(residual);
    stats.final_true_residual = Some(residual);
    stats.counters = SolverCounters {
        num_global_reductions: reductions,
        ..SolverCounters::default()
    };
    stats.finalize_reason_counters()
}

#[allow(clippy::too_many_arguments)]
fn cuda_pipegcr_stats(
    iterations: usize,
    true_residual: R,
    recurrence_residual: R,
    reason: ConvergedReason,
    reductions: usize,
    basis_updates: usize,
    restart_count: usize,
    restart: usize,
) -> SolveStats<R> {
    let mut stats = SolveStats::new(iterations, true_residual, reason);
    stats.final_true_residual = Some(true_residual);
    stats.final_recurrence_residual = Some(recurrence_residual);
    stats.counters = SolverCounters {
        num_global_reductions: reductions,
        ..SolverCounters::default()
    };
    stats.reduction_model = Some(ReductionModel {
        variant: "cuda-pipegcr",
        startup: 1,
        per_iteration: 2.0,
        tail: 1,
    });
    stats.gcr_counters = Some(GcrCounters {
        basis_updates,
        sync_count: reductions,
        restart_count,
        restarted: restart_count > 0,
    });
    stats.effective_variant = Some("cuda-pipegcr-classical".into());
    stats.effective_restart = Some(restart);
    stats.finalize_reason_counters()
}

fn cuda_qmr_stats(
    iterations: usize,
    true_residual: R,
    recurrence_residual: R,
    reason: ConvergedReason,
    reductions: usize,
) -> SolveStats<R> {
    let mut stats = SolveStats::new(iterations, true_residual, reason);
    stats.final_true_residual = Some(true_residual);
    stats.final_recurrence_residual = Some(recurrence_residual);
    stats.counters = SolverCounters {
        num_global_reductions: reductions,
        ..SolverCounters::default()
    };
    stats.reduction_model = Some(ReductionModel {
        variant: "cuda-qmr-compat",
        startup: 2,
        per_iteration: 4.0,
        tail: 1,
    });
    stats.effective_variant = Some("cuda-qmr-compat".into());
    stats.finalize_reason_counters()
}

fn cuda_tfqmr_stats(
    solver_type: SolverType,
    iterations: usize,
    true_residual: R,
    recurrence_residual: R,
    reason: ConvergedReason,
    reductions: usize,
) -> SolveStats<R> {
    let variant = if solver_type == SolverType::Tcqmr {
        "cuda-tcqmr-via-tfqmr"
    } else {
        "cuda-tfqmr"
    };
    let mut stats = SolveStats::new(iterations, true_residual, reason);
    stats.final_true_residual = Some(true_residual);
    stats.final_recurrence_residual = Some(recurrence_residual);
    stats.counters = SolverCounters {
        num_global_reductions: reductions,
        ..SolverCounters::default()
    };
    stats.reduction_model = Some(ReductionModel {
        variant,
        startup: 1,
        per_iteration: 2.0,
        tail: 1,
    });
    stats.effective_variant = Some(variant.into());
    stats.finalize_reason_counters()
}

#[inline]
fn sym_ortho_cuda(a: R, b: R) -> (R, R, R) {
    if b == 0.0 {
        (1.0, 0.0, a)
    } else if a == 0.0 {
        (0.0, 1.0, b)
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let sine = 1.0 / (1.0 + tau * tau).sqrt();
        let cosine = sine * tau;
        (cosine, sine, b / sine)
    } else {
        let tau = b / a;
        let cosine = 1.0 / (1.0 + tau * tau).sqrt();
        let sine = cosine * tau;
        (cosine, sine, a / cosine)
    }
}

fn ensure_positive_real(
    value: S,
    context: &'static str,
    preconditioner: bool,
) -> Result<(), KError> {
    if !value.is_finite() {
        return Err(KError::NonFiniteReduction {
            kind: if value.real().is_nan() || value.imag().is_nan() {
                crate::error::NonFiniteKind::Nan
            } else {
                crate::error::NonFiniteKind::Inf
            },
            context,
        });
    }
    let tolerance = 64.0 * f64::EPSILON * value.abs().max(1.0);
    if value.imag().abs() > tolerance || value.real() <= 0.0 {
        return Err(if preconditioner {
            KError::IndefinitePreconditioner
        } else {
            KError::IndefiniteMatrix
        });
    }
    Ok(())
}

fn nonnegative_hermitian_real(
    value: S,
    context: &'static str,
    preconditioner: bool,
) -> Result<R, KError> {
    if !value.is_finite() {
        return Err(KError::NonFiniteReduction {
            kind: if value.real().is_nan() || value.imag().is_nan() {
                crate::error::NonFiniteKind::Nan
            } else {
                crate::error::NonFiniteKind::Inf
            },
            context,
        });
    }
    let tolerance = 64.0 * f64::EPSILON * value.abs().max(1.0);
    if value.imag().abs() > tolerance || value.real() < 0.0 {
        return Err(if preconditioner {
            KError::IndefinitePreconditioner
        } else {
            KError::IndefiniteMatrix
        });
    }
    Ok(value.real())
}

fn ensure_nonzero_finite(value: S, context: &'static str) -> Result<(), KError> {
    ensure_finite(value, context)?;
    if value.abs() <= 1e-30 {
        return Err(KError::SolveError(format!(
            "BiCGStab breakdown: {context} is numerically zero"
        )));
    }
    Ok(())
}

fn ensure_finite(value: S, context: &'static str) -> Result<(), KError> {
    if !value.is_finite() {
        return Err(KError::NonFiniteReduction {
            kind: if value.real().is_nan() || value.imag().is_nan() {
                crate::error::NonFiniteKind::Nan
            } else {
                crate::error::NonFiniteKind::Inf
            },
            context,
        });
    }
    Ok(())
}

fn backsolve_hessenberg(
    h: &[S],
    leading_dimension: usize,
    g: &[S],
    k: usize,
    y: &mut [S],
) -> Result<(), KError> {
    y.fill(S::zero());
    for row in (0..k).rev() {
        let diagonal = h[row * leading_dimension + row];
        if diagonal.abs() <= 64.0 * f64::EPSILON {
            return Err(KError::BreakdownOrIndefinite);
        }
        let mut rhs = g[row];
        for column in (row + 1)..k {
            rhs = rhs - h[column * leading_dimension + row] * y[column];
        }
        y[row] = rhs / diagonal;
    }
    Ok(())
}
