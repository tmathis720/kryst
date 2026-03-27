//! Balancing Domain Decomposition by Constraints (BDDC) preconditioner.

use crate::algebra::scalar::{KrystScalar, R, S};
use crate::config::options::PcOptions;
use crate::context::ksp_context::{KspContext, SolverType};
use crate::context::pc_context::PcType;
use crate::error::KError;
use crate::matrix::MatShell;
#[cfg(all(not(feature = "complex"), feature = "backend-faer"))]
use crate::matrix::convert::csr_from_linop;
use crate::matrix::csr::CsrMatrix;
use crate::matrix::op::{DistLayout, LinOp, StructureId, ValuesId};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};
use crate::utils::convergence::ConvergedReason;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BddcConstraintSelection {
    Vertices,
    Interface,
    VerticesAndInterface,
}

impl BddcConstraintSelection {
    fn from_use_vertices(use_vertices: bool) -> Self {
        if use_vertices {
            Self::VerticesAndInterface
        } else {
            Self::Interface
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BddcScaling {
    Uniform,
    DeluxeLike,
}

#[derive(Debug, Clone)]
pub struct BddcConfig {
    pub coarse_ksp_type: Option<String>,
    pub coarse_pc_type: Option<String>,
    pub local_ksp_type: Option<String>,
    pub local_pc_type: Option<String>,
    pub use_vertices: bool,
    pub constraint_selection: Option<BddcConstraintSelection>,
    pub scaling: Option<BddcScaling>,
}

#[derive(Debug, Clone, Default)]
pub struct BddcDiagnostics {
    pub local_nnz: usize,
    pub coarse_nnz: usize,
    pub local_solve_route: String,
    pub local_solve_estimated_flops: usize,
    pub coarse_solve_route: String,
    pub coarse_iterations: usize,
    pub coarse_final_residual: R,
    pub coarse_fallback_route: Option<String>,
    pub coarse_failure_reason: Option<String>,
    pub comm_interface_dofs: usize,
    pub comm_coarse_dofs: usize,
}

#[derive(Debug, Clone)]
struct BddcSymbolic {
    dims: (usize, usize),
    local_n: usize,
    layout: Option<DistLayout>,
    subdomains: Vec<(usize, usize)>,
    interface_dofs: Vec<usize>,
    coarse_dofs: Vec<usize>,
    interface_multiplicity: Vec<R>,
    structure_id: StructureId,
}

#[derive(Debug, Clone)]
struct BddcNumeric {
    operator: CsrMatrix<S>,
    coarse_operator: CsrMatrix<S>,
    values_id: ValuesId,
}

/// BDDC preconditioner with reusable symbolic and numeric phases.
pub struct BddcPc {
    config: BddcConfig,
    dims: (usize, usize),
    comm: UniverseComm,
    symbolic: Option<BddcSymbolic>,
    numeric: Option<BddcNumeric>,
    diagnostics: Mutex<Option<BddcDiagnostics>>,
}

impl BddcPc {
    pub fn new(config: BddcConfig) -> Self {
        Self {
            config,
            dims: (0, 0),
            comm: UniverseComm::NoComm(crate::parallel::NoComm),
            symbolic: None,
            numeric: None,
            diagnostics: Mutex::new(None),
        }
    }

    pub fn diagnostics(&self) -> Option<BddcDiagnostics> {
        self.diagnostics.lock().ok().and_then(|d| d.clone())
    }

    fn build_subdomains(n: usize) -> Vec<(usize, usize)> {
        if n <= 1 {
            return vec![(0, n)];
        }
        let mid = n / 2;
        vec![(0, (mid + 1).min(n)), (mid, n)]
    }

    fn build_interface(subdomains: &[(usize, usize)]) -> Vec<usize> {
        if subdomains.len() <= 1 {
            return Vec::new();
        }
        let mut interface = Vec::new();
        for window in subdomains.windows(2) {
            if let Some((_, end)) = window.first() {
                if *end > 0 {
                    interface.push(end - 1);
                }
            }
            if let Some((start, _)) = window.get(1) {
                interface.push(*start);
            }
        }
        interface.sort_unstable();
        interface.dedup();
        interface
    }

    fn build_constraints(
        subdomains: &[(usize, usize)],
        interface_dofs: &[usize],
        selection: BddcConstraintSelection,
    ) -> Vec<usize> {
        let mut coarse_dofs = Vec::new();
        if matches!(
            selection,
            BddcConstraintSelection::Vertices | BddcConstraintSelection::VerticesAndInterface
        ) {
            for (start, end) in subdomains {
                if *start < *end {
                    coarse_dofs.push(*start);
                    coarse_dofs.push(end.saturating_sub(1));
                }
            }
        }
        if matches!(
            selection,
            BddcConstraintSelection::Interface | BddcConstraintSelection::VerticesAndInterface
        ) {
            coarse_dofs.extend_from_slice(interface_dofs);
        }
        coarse_dofs.sort_unstable();
        coarse_dofs.dedup();
        coarse_dofs
    }

    fn assemble_interface_multiplicity(
        local_n: usize,
        interface_dofs: &[usize],
        subdomains: &[(usize, usize)],
        layout: Option<&DistLayout>,
        comm: &UniverseComm,
    ) -> Vec<R> {
        let mut local_counts = vec![S::zero(); local_n];
        for &(start, end) in subdomains {
            for i in start..end {
                local_counts[i] = local_counts[i] + S::one();
            }
        }

        let global_counts = if let Some(l) = layout {
            let mut owned = vec![S::zero(); l.global_rows];
            for (local_i, val) in local_counts.iter().copied().enumerate() {
                let gi = l.row_start + local_i;
                if gi < owned.len() {
                    owned[gi] = val;
                }
            }
            comm.allreduce_sum_scalars(&mut owned);
            owned
        } else {
            comm.allreduce_sum_scalars(&mut local_counts);
            local_counts
        };

        interface_dofs
            .iter()
            .map(|&i| {
                let idx = layout.map(|l| l.row_start + i).unwrap_or(i);
                if idx < global_counts.len() {
                    global_counts[idx].real().max(1.0)
                } else {
                    1.0
                }
            })
            .collect()
    }

    fn extract_dense_operator(op: &dyn LinOp<S = S>, n: usize) -> Result<Vec<Vec<S>>, KError> {
        let mut a = vec![vec![S::zero(); n]; n];
        let mut ej = vec![S::zero(); n];
        let mut col = vec![S::zero(); n];
        for j in 0..n {
            ej.fill(S::zero());
            ej[j] = S::one();
            op.try_matvec(&ej, &mut col)?;
            for i in 0..n {
                a[i][j] = col[i];
            }
        }
        Ok(a)
    }

    fn csr_submatrix(a: &CsrMatrix<S>, dofs: &[usize]) -> CsrMatrix<S> {
        let n = dofs.len();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        let mut inv = vec![usize::MAX; a.ncols];
        for (i, &dof) in dofs.iter().enumerate() {
            if dof < inv.len() {
                inv[dof] = i;
            }
        }
        for &global_i in dofs {
            for p in a.rowptr[global_i]..a.rowptr[global_i + 1] {
                let j = a.colind[p];
                if j < inv.len() {
                    let local_j = inv[j];
                    if local_j != usize::MAX {
                        col_idx.push(local_j);
                        values.push(a.values[p]);
                    }
                }
            }
            row_ptr.push(col_idx.len());
        }
        CsrMatrix::new(n, n, row_ptr, col_idx, values)
    }

    fn csr_matvec(a: &CsrMatrix<S>, x: &[S], y: &mut [S]) {
        for (i, yi) in y.iter_mut().enumerate().take(a.nrows) {
            let mut acc = S::zero();
            for p in a.rowptr[i]..a.rowptr[i + 1] {
                acc = acc + a.values[p] * x[a.colind[p]];
            }
            *yi = acc;
        }
    }

    fn solve_csr_jacobi(a: &CsrMatrix<S>, rhs: &[S]) -> Result<Vec<S>, KError> {
        let mut x = vec![S::zero(); rhs.len()];
        for i in 0..rhs.len() {
            let mut diag = None;
            for p in a.rowptr[i]..a.rowptr[i + 1] {
                if a.colind[p] == i {
                    diag = Some(a.values[p]);
                    break;
                }
            }
            let d = diag.ok_or_else(|| {
                KError::FactorError("BDDC Jacobi solve encountered missing diagonal".into())
            })?;
            if d.abs() <= 1e-14 {
                return Err(KError::FactorError(
                    "BDDC Jacobi solve encountered near-zero diagonal".into(),
                ));
            }
            x[i] = rhs[i] / d;
        }
        Ok(x)
    }

    fn solve_csr_cg(a: &CsrMatrix<S>, rhs: &[S], maxits: usize, tol: R) -> Result<Vec<S>, KError> {
        let n = rhs.len();
        let mut x = vec![S::zero(); n];
        let mut r = rhs.to_vec();
        let mut p = r.clone();
        let mut ap = vec![S::zero(); n];
        let mut rr = r
            .iter()
            .map(|v| v.conj() * *v)
            .fold(S::zero(), |acc, v| acc + v);
        let rr0 = rr.abs().max(1e-30);
        for _ in 0..maxits {
            Self::csr_matvec(a, &p, &mut ap);
            let denom = p
                .iter()
                .zip(ap.iter())
                .map(|(pi, api)| pi.conj() * *api)
                .fold(S::zero(), |acc, v| acc + v);
            if denom.abs() <= 1e-20 {
                break;
            }
            let alpha = rr / denom;
            for i in 0..n {
                x[i] = x[i] + alpha * p[i];
                r[i] = r[i] - alpha * ap[i];
            }
            let rr_new = r
                .iter()
                .map(|v| v.conj() * *v)
                .fold(S::zero(), |acc, v| acc + v);
            if (rr_new.abs() / rr0).sqrt() < tol {
                return Ok(x);
            }
            let beta = rr_new / rr;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rr = rr_new;
        }
        Ok(x)
    }

    fn solve_csr_local(&self, a: &CsrMatrix<S>, rhs: Vec<S>) -> Result<Vec<S>, KError> {
        let ksp = self
            .config
            .local_ksp_type
            .as_deref()
            .unwrap_or("preonly")
            .to_lowercase();
        let pc = self
            .config
            .local_pc_type
            .as_deref()
            .unwrap_or("jacobi")
            .to_lowercase();
        match (ksp.as_str(), pc.as_str()) {
            ("preonly", "jacobi") => Self::solve_csr_jacobi(a, &rhs),
            ("cg", "jacobi") => Self::solve_csr_cg(a, &rhs, 60, 1e-8),
            ("gmres", "jacobi") => Self::solve_csr_cg(a, &rhs, 90, 1e-8),
            _ => Err(KError::InvalidInput(format!(
                "unsupported BDDC local backend combination: ksp={ksp}, pc={pc}"
            ))),
        }
    }

    fn coarse_solve_with_ksp(
        &self,
        coarse_op: &CsrMatrix<S>,
        rhs: &[S],
        ksp_name: &str,
        pc_name: &str,
    ) -> Result<(Vec<S>, usize, R, ConvergedReason), KError> {
        let solver_type = SolverType::from_str(ksp_name)?;
        let pc_type = PcType::from_str(pc_name)?;
        let mut ksp = KspContext::new();
        ksp.set_type(solver_type)?;
        let pc_opts = PcOptions {
            pc_type: Some(pc_name.to_string()),
            ..Default::default()
        };
        ksp.set_pc_type(pc_type, Some(&pc_opts))?;
        let coarse_op_owned = Arc::new(coarse_op.clone());
        let shell = MatShell::new(rhs.len(), rhs.len(), move |x: &[S], y: &mut [S]| {
            coarse_op_owned.spmv(x, y);
        });
        ksp.set_operators(Arc::new(shell), None);
        ksp.setup()?;
        let mut x = vec![S::zero(); rhs.len()];
        let stats = ksp.solve(rhs, &mut x)?;
        Ok((x, stats.iterations, stats.final_residual, stats.reason))
    }

    fn coarse_solve(
        &self,
        coarse_op: &CsrMatrix<S>,
        rhs: Vec<S>,
    ) -> Result<(Vec<S>, usize, R, String, Option<String>, Option<String>), KError> {
        let ksp = self
            .config
            .coarse_ksp_type
            .as_deref()
            .unwrap_or("preonly")
            .to_lowercase();
        let pc = self
            .config
            .coarse_pc_type
            .as_deref()
            .unwrap_or("lu")
            .to_lowercase();
        let primary = (ksp.clone(), pc.clone());
        let mut routes = vec![primary.clone()];
        if ksp == "preonly" {
            routes.push(("gmres".to_string(), "ilu".to_string()));
            routes.push(("cg".to_string(), "jacobi".to_string()));
        } else {
            routes.push(("preonly".to_string(), "lu".to_string()));
            routes.push(("gmres".to_string(), "ilu".to_string()));
            routes.push(("cg".to_string(), "jacobi".to_string()));
        }
        routes.dedup();
        let mut first_failure: Option<String> = None;
        for (idx, (rksp, rpc)) in routes.iter().enumerate() {
            let route = format!("{rksp}+{rpc}");
            match self.coarse_solve_with_ksp(coarse_op, &rhs, rksp, rpc) {
                Ok((x, its, res, reason)) => {
                    if matches!(
                        reason,
                        ConvergedReason::ConvergedAtol
                            | ConvergedReason::ConvergedRtol
                            | ConvergedReason::ConvergedHappyBreakdown
                            | ConvergedReason::ConvergedTrustRegion
                    ) {
                        let fallback = (idx > 0).then(|| format!("{}+{}", primary.0, primary.1));
                        return Ok((x, its, res, route, fallback, first_failure));
                    }
                    let msg = format!("{route} stopped with {:?}", reason);
                    if first_failure.is_none() {
                        first_failure = Some(msg);
                    }
                }
                Err(err) => {
                    if first_failure.is_none() {
                        first_failure = Some(format!("{route} failed: {err}"));
                    }
                }
            }
        }
        Err(KError::PcFailed(first_failure.unwrap_or_else(|| {
            "BDDC coarse solve failed for all configured/fallback routes".to_string()
        })))
    }

    fn apply_scaling(&self, value: S, diag: S, multiplicity: R) -> S {
        match self.config.scaling.unwrap_or(BddcScaling::Uniform) {
            BddcScaling::Uniform => value * S::from_real(1.0 / multiplicity.max(1.0)),
            BddcScaling::DeluxeLike => {
                let denom = diag.abs().max(1e-12) * multiplicity.max(1.0);
                value * S::from_real(1.0 / denom)
            }
        }
    }
}

impl Preconditioner for BddcPc {
    fn dims(&self) -> (usize, usize) {
        self.dims
    }

    fn setup(&mut self, op: &dyn LinOp<S = S>) -> Result<(), KError> {
        let dims = op.dims();
        if dims.0 != dims.1 {
            return Err(KError::InvalidInput(
                "BDDC requires a square operator".into(),
            ));
        }
        let layout = op.dist_layout().cloned();
        if let Some(l) = &layout {
            if l.global_rows != l.global_cols {
                return Err(KError::InvalidInput(
                    "BDDC requires a square global distributed operator".into(),
                ));
            }
            let local_rows = l.row_end.saturating_sub(l.row_start);
            let local_cols = l.col_end.saturating_sub(l.col_start);
            if local_rows != dims.0 || local_cols != dims.1 {
                return Err(KError::InvalidInput(
                    "BDDC local dimensions must match distributed layout ownership range".into(),
                ));
            }
        }

        let structure_id = op.structure_id();
        let values_id = op.values_id();
        let mut symbolic_rebuild = true;
        if let Some(sym) = &self.symbolic {
            symbolic_rebuild = sym.structure_id != structure_id || sym.dims != dims;
        }

        if symbolic_rebuild {
            let local_n = dims.0;
            let subdomains = Self::build_subdomains(local_n);
            let interface_dofs = Self::build_interface(&subdomains);
            let selection = self.config.constraint_selection.unwrap_or_else(|| {
                BddcConstraintSelection::from_use_vertices(self.config.use_vertices)
            });
            let coarse_dofs = Self::build_constraints(&subdomains, &interface_dofs, selection);
            let comm = op.comm();
            let interface_multiplicity = Self::assemble_interface_multiplicity(
                local_n,
                &interface_dofs,
                &subdomains,
                layout.as_ref(),
                &comm,
            );

            self.symbolic = Some(BddcSymbolic {
                dims,
                local_n,
                layout: layout.clone(),
                subdomains,
                interface_dofs,
                coarse_dofs,
                interface_multiplicity,
                structure_id,
            });
        }

        let mut numeric_rebuild = true;
        if let Some(num) = &self.numeric {
            numeric_rebuild = num.values_id != values_id || symbolic_rebuild;
        }
        if numeric_rebuild {
            let sym = self
                .symbolic
                .as_ref()
                .ok_or_else(|| KError::PcFailed("BDDC symbolic phase missing".into()))?;
            #[cfg(all(not(feature = "complex"), feature = "backend-faer"))]
            let operator = {
                let csr = csr_from_linop(op, 0.0)?;
                CsrMatrix::from_real_csr(csr.as_ref())
            };
            #[cfg(any(feature = "complex", not(feature = "backend-faer")))]
            let operator = {
                let dense = Self::extract_dense_operator(op, sym.local_n)?;
                let mut row_ptr = Vec::with_capacity(sym.local_n + 1);
                let mut col_idx = Vec::new();
                let mut vals = Vec::new();
                row_ptr.push(0);
                for (i, row) in dense.iter().enumerate() {
                    for (j, &v) in row.iter().enumerate() {
                        if v.abs() > 0.0 {
                            col_idx.push(j);
                            vals.push(v);
                        }
                    }
                    row_ptr.push(col_idx.len());
                    let _ = i;
                }
                CsrMatrix::new(sym.local_n, sym.local_n, row_ptr, col_idx, vals)
            };
            let coarse_operator = Self::csr_submatrix(&operator, &sym.coarse_dofs);
            let local_route = format!(
                "{}+{}",
                self.config.local_ksp_type.as_deref().unwrap_or("preonly"),
                self.config.local_pc_type.as_deref().unwrap_or("jacobi")
            );
            let coarse_route = format!(
                "{}+{}",
                self.config.coarse_ksp_type.as_deref().unwrap_or("preonly"),
                self.config.coarse_pc_type.as_deref().unwrap_or("lu")
            );
            let diagnostics = BddcDiagnostics {
                local_nnz: operator.nnz(),
                coarse_nnz: coarse_operator.nnz(),
                local_solve_route: local_route,
                local_solve_estimated_flops: operator.nnz().saturating_mul(2),
                coarse_solve_route: coarse_route,
                coarse_iterations: 0,
                coarse_final_residual: 0.0,
                coarse_fallback_route: None,
                coarse_failure_reason: None,
                comm_interface_dofs: sym.interface_dofs.len(),
                comm_coarse_dofs: sym.coarse_dofs.len(),
            };
            if let Ok(mut d) = self.diagnostics.lock() {
                *d = Some(diagnostics);
            }
            self.numeric = Some(BddcNumeric {
                operator,
                coarse_operator,
                values_id,
            });
        }

        self.comm = op.comm();
        self.dims = dims;
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        let sym = self
            .symbolic
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("BDDC preconditioner not setup".into()))?;
        let num = self
            .numeric
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("BDDC numeric phase not setup".into()))?;

        if x.len() != sym.local_n || y.len() != sym.local_n {
            return Err(KError::InvalidInput(
                "BDDC apply expects vectors matching local ownership size".into(),
            ));
        }

        // Phase 1: subdomain interior solve.
        y.fill(S::zero());
        let mut multiplicity = vec![0usize; sym.local_n];
        for &(start, end) in &sym.subdomains {
            let dofs: Vec<usize> = (start..end).collect();
            let rhs: Vec<S> = dofs.iter().map(|&i| x[i]).collect();
            let a_sub = Self::csr_submatrix(&num.operator, &dofs);
            let local_sol = self.solve_csr_local(&a_sub, rhs)?;
            for (&dof, &val) in dofs.iter().zip(local_sol.iter()) {
                y[dof] = y[dof] + val;
                multiplicity[dof] += 1;
            }
        }
        for (i, yi) in y.iter_mut().enumerate() {
            if multiplicity[i] > 0 {
                *yi = *yi * S::from_real(1.0 / multiplicity[i] as R);
            }
        }

        // Phase 2: primal coarse correction.
        let mut az = vec![S::zero(); sym.local_n];
        Self::csr_matvec(&num.operator, y, &mut az);
        let residual: Vec<S> = x
            .iter()
            .zip(az.iter())
            .map(|(&xi, &azi)| xi - azi)
            .collect();

        if !sym.coarse_dofs.is_empty() {
            let rc: Vec<S> = sym.coarse_dofs.iter().map(|&dof| residual[dof]).collect();
            let (ec, coarse_its, coarse_res, route, fallback, failure) =
                self.coarse_solve(&num.coarse_operator, rc)?;
            if let Ok(mut diag_guard) = self.diagnostics.lock() {
                if let Some(diag) = diag_guard.as_mut() {
                    diag.coarse_iterations = coarse_its;
                    diag.coarse_final_residual = coarse_res;
                    diag.coarse_solve_route = route;
                    diag.coarse_fallback_route = fallback;
                    diag.coarse_failure_reason = failure;
                }
            }
            for (k, &dof) in sym.coarse_dofs.iter().enumerate() {
                y[dof] = y[dof] + ec[k];
            }
        }

        // Phase 3: interface constraint enforcement with selectable scaling.
        for (k, &dof) in sym.interface_dofs.iter().enumerate() {
            if dof >= sym.local_n {
                continue;
            }
            let mut diag = S::one();
            for p in num.operator.rowptr[dof]..num.operator.rowptr[dof + 1] {
                if num.operator.colind[p] == dof {
                    diag = num.operator.values[p];
                    break;
                }
            }
            let multiplicity = sym.interface_multiplicity.get(k).copied().unwrap_or(1.0);
            y[dof] = self.apply_scaling(y[dof], diag, multiplicity);
        }
        Ok(())
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
    }
}
