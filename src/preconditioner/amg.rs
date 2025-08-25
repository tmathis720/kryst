use std::sync::Arc;

use crate::error::KError;
use crate::matrix::op::{StructureId, ValuesId};
use crate::matrix::{convert::csr_from_linop, op::LinOp, sparse::CsrMatrix};
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;

/// Coarsening strategies (stub).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarsenType {
    RS,
    HMIS,
    PMIS,
    Falgout,
}

/// Interpolation strategies (stub).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpType {
    Classical,
    Direct,
    Multipass,
    Extended,
    Standard,
}

/// Relaxation/smoothing choices (stub).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelaxType {
    Jacobi,
    GaussSeidel,
    GaussSeidelBackward,
    SymmetricGaussSeidel,
    HybridGaussSeidel,
    L1Jacobi,
    Chebyshev,
}

/// Builder for `AMG` preserving old construction-style API.
pub struct AMGBuilder {
    drop_tol: f64,
}

impl AMGBuilder {
    pub fn new() -> Self {
        Self { drop_tol: 0.0 }
    }
    pub fn max_levels(self, _levels: usize) -> Self {
        self
    }
    pub fn strong_threshold(self, _thr: f64) -> Self {
        self
    }
    pub fn coarse_threshold(self, _thr: usize) -> Self {
        self
    }
    pub fn max_coarse_size(self, _size: usize) -> Self {
        self
    }
    pub fn min_coarse_size(self, _size: usize) -> Self {
        self
    }
    pub fn truncation_factor(self, _f: f64) -> Self {
        self
    }
    pub fn interpolation_truncation(self, _f: f64) -> Self {
        self
    }
    pub fn smoothing_sweeps(self, _pre: usize, _post: usize) -> Self {
        self
    }
    pub fn coarsening_type(self, _ct: CoarsenType) -> Self {
        self
    }
    pub fn interpolation_type(self, _it: InterpType) -> Self {
        self
    }
    pub fn relaxation_type(self, _rt: RelaxType) -> Self {
        self
    }
    pub fn enable_logging(self) -> Self {
        self
    }
    pub fn enable_printing(self) -> Self {
        self
    }
    pub fn build(self, _matrix: &Mat<f64>) -> Result<AMG, KError> {
        Ok(AMG::default())
    }
}

/// Minimal AMG hierarchy storing data required for solves.
///
/// This placeholder currently stores only the inverse diagonal for a Jacobi smoother.
#[derive(Default, Clone)]
struct AmgHierarchy {
    diag_inv: Vec<f64>,
}

impl AmgHierarchy {
    /// Build a new hierarchy from a CSR matrix (symbolic + numeric phase).
    fn symbolic(csr: &CsrMatrix<f64>) -> Result<Self, KError> {
        let mut diag_inv = Vec::with_capacity(csr.nrows());
        for i in 0..csr.nrows() {
            let rs = csr.row_ptr()[i];
            let re = csr.row_ptr()[i + 1];
            let mut aii = 0.0;
            for p in rs..re {
                if csr.col_idx()[p] == i {
                    aii = csr.values()[p];
                    break;
                }
            }
            if aii.abs() < 1e-14 {
                return Err(KError::SolveError(format!(
                    "zero or near-zero diagonal at row {i}",
                )));
            }
            diag_inv.push(1.0 / aii);
        }
        Ok(Self { diag_inv })
    }

    /// Refresh numeric values in-place assuming the sparsity pattern is unchanged.
    fn numeric(&mut self, csr: &CsrMatrix<f64>) -> Result<(), KError> {
        if self.diag_inv.len() != csr.nrows() {
            self.diag_inv.resize(csr.nrows(), 0.0);
        }
        for i in 0..csr.nrows() {
            let rs = csr.row_ptr()[i];
            let re = csr.row_ptr()[i + 1];
            let mut aii = 0.0;
            for p in rs..re {
                if csr.col_idx()[p] == i {
                    aii = csr.values()[p];
                    break;
                }
            }
            if aii.abs() < 1e-14 {
                return Err(KError::SolveError(format!(
                    "zero or near-zero diagonal at row {i}",
                )));
            }
            self.diag_inv[i] = 1.0 / aii;
        }
        Ok(())
    }

    /// Apply the (Jacobi) smoother stored in this hierarchy.
    fn apply(&self, _side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if x.len() != y.len() || x.len() != self.diag_inv.len() {
            return Err(KError::InvalidInput(format!(
                "apply: dimension mismatch x={} y={} n={}",
                x.len(),
                y.len(),
                self.diag_inv.len()
            )));
        }
        for i in 0..self.diag_inv.len() {
            y[i] = self.diag_inv[i] * x[i];
        }
        Ok(())
    }
}

/// Algebraic Multigrid (AMG) preconditioner skeleton following the common CSR template.
pub struct AMG {
    csr: Option<Arc<CsrMatrix<f64>>>,
    state: Option<AmgHierarchy>,
    last_sid: Option<StructureId>,
    last_vid: Option<ValuesId>,
    drop_tol: f64,
}

impl Default for AMG {
    fn default() -> Self {
        Self {
            csr: None,
            state: None,
            last_sid: None,
            last_vid: None,
            drop_tol: 0.0,
        }
    }
}

impl AMG {
    /// Legacy constructor retained for examples; arguments are ignored.
    pub fn new(_matrix: &Mat<f64>, _max_levels: usize, _coarsening_threshold: f64) -> Self {
        AMG::default()
    }

    /// Builder entry point mirroring historical API.
    pub fn builder() -> AMGBuilder {
        AMGBuilder::new()
    }

    /// Build or rebuild the full hierarchy from the CSR matrix.
    fn build_or_rebuild_symbolic(&mut self, csr: &CsrMatrix<f64>) -> Result<(), KError> {
        self.state = Some(AmgHierarchy::symbolic(csr)?);
        Ok(())
    }

    /// Refresh only numeric values in the existing hierarchy.
    fn refresh_numeric(&mut self, csr: &CsrMatrix<f64>) -> Result<(), KError> {
        if let Some(st) = self.state.as_mut() {
            st.numeric(csr)
        } else {
            self.build_or_rebuild_symbolic(csr)
        }
    }

    /// Inherent apply to avoid trait ambiguity in examples.
    pub fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        Preconditioner::apply(self, side, x, y)
    }
}

impl Preconditioner for AMG {
    fn setup(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let csr = csr_from_linop(op, self.drop_tol)?;
        let sid = op.structure_id();
        let vid = op.values_id();

        match (self.last_sid, self.last_vid) {
            (None, _) => {
                self.build_or_rebuild_symbolic(&csr)?;
            }
            (Some(old_sid), _) if old_sid != sid => {
                self.build_or_rebuild_symbolic(&csr)?;
            }
            (Some(_), Some(old_vid)) if old_vid != vid => {
                if self.supports_numeric_update() {
                    self.refresh_numeric(&csr)?;
                } else {
                    self.build_or_rebuild_symbolic(&csr)?;
                }
            }
            _ => {}
        }

        self.csr = Some(csr);
        self.last_sid = Some(sid);
        self.last_vid = Some(vid);
        Ok(())
    }

    fn apply(&self, side: PcSide, x: &[f64], y: &mut [f64]) -> Result<(), KError> {
        if x.len() != y.len() {
            return Err(KError::InvalidInput(format!(
                "apply: x/y length mismatch: {} vs {}",
                x.len(),
                y.len()
            )));
        }
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("PC not set up".into()))?;
        state.apply(side, x, y)
    }

    fn supports_numeric_update(&self) -> bool {
        true
    }

    fn update_numeric(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let csr = csr_from_linop(op, self.drop_tol)?;
        self.refresh_numeric(&csr)?;
        self.csr = Some(csr);
        self.last_vid = Some(op.values_id());
        Ok(())
    }

    fn update_symbolic(&mut self, op: &dyn LinOp<S = f64>) -> Result<(), KError> {
        let csr = csr_from_linop(op, self.drop_tol)?;
        self.build_or_rebuild_symbolic(&csr)?;
        self.csr = Some(csr);
        self.last_sid = Some(op.structure_id());
        self.last_vid = Some(op.values_id());
        Ok(())
    }
}

impl crate::preconditioner::legacy::Preconditioner<Mat<f64>, Vec<f64>> for AMG {
    fn setup(&mut self, a: &Mat<f64>) -> Result<(), KError> {
        Preconditioner::setup(self, a)
    }

    fn apply(&self, side: PcSide, r: &Vec<f64>, z: &mut Vec<f64>) -> Result<(), KError> {
        Preconditioner::apply(self, side, r.as_slice(), z.as_mut_slice())
    }
}
