//! Balancing Domain Decomposition by Constraints (BDDC) preconditioner.

use crate::algebra::scalar::{KrystScalar, R, S};
use crate::error::KError;
use crate::matrix::op::{DistLayout, LinOp, StructureId, ValuesId};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};

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
    pub use_vertices: bool,
    pub constraint_selection: Option<BddcConstraintSelection>,
    pub scaling: Option<BddcScaling>,
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
    operator: Vec<Vec<S>>,
    coarse_operator: Vec<Vec<S>>,
    values_id: ValuesId,
}

/// BDDC preconditioner with reusable symbolic and numeric phases.
pub struct BddcPc {
    config: BddcConfig,
    dims: (usize, usize),
    comm: UniverseComm,
    symbolic: Option<BddcSymbolic>,
    numeric: Option<BddcNumeric>,
}

impl BddcPc {
    pub fn new(config: BddcConfig) -> Self {
        Self {
            config,
            dims: (0, 0),
            comm: UniverseComm::NoComm(crate::parallel::NoComm),
            symbolic: None,
            numeric: None,
        }
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

        let mut global_counts = if let Some(l) = layout {
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

    fn solve_dense(mut a: Vec<Vec<S>>, mut b: Vec<S>) -> Result<Vec<S>, KError> {
        let n = b.len();
        if a.len() != n || a.iter().any(|row| row.len() != n) {
            return Err(KError::InvalidInput(
                "BDDC dense solve received non-square matrix".into(),
            ));
        }
        for k in 0..n {
            let mut piv = k;
            let mut best = a[k][k].abs();
            for (i, row) in a.iter().enumerate().skip(k + 1) {
                let cand = row[k].abs();
                if cand > best {
                    best = cand;
                    piv = i;
                }
            }
            if best <= 1e-14 {
                return Err(KError::FactorError(
                    "BDDC local/coarse solve encountered singular pivot".into(),
                ));
            }
            if piv != k {
                a.swap(piv, k);
                b.swap(piv, k);
            }
            let pivot = a[k][k];
            let pivot_row = a[k].clone();
            let b_k = b[k];
            for (i, row) in a.iter_mut().enumerate().skip(k + 1) {
                let factor = row[k] / pivot;
                row[k] = S::zero();
                for j in (k + 1)..n {
                    row[j] = row[j] - factor * pivot_row[j];
                }
                b[i] = b[i] - factor * b_k;
            }
        }
        let mut x = vec![S::zero(); n];
        for i in (0..n).rev() {
            let mut rhs = b[i];
            for (j, xj) in x.iter().enumerate().skip(i + 1) {
                rhs = rhs - a[i][j] * *xj;
            }
            x[i] = rhs / a[i][i];
        }
        Ok(x)
    }

    fn coarse_solve(&self, coarse_op: &[Vec<S>], rhs: Vec<S>) -> Result<Vec<S>, KError> {
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

        match ksp.as_str() {
            "preonly" | "cg" | "gmres" => {}
            other => {
                return Err(KError::InvalidInput(format!(
                    "unsupported BDDC coarse KSP backend: {other}"
                )));
            }
        }

        match pc.as_str() {
            "lu" | "cholesky" | "none" => Self::solve_dense(coarse_op.to_vec(), rhs),
            "jacobi" => {
                let mut x = vec![S::zero(); rhs.len()];
                for (i, row) in coarse_op.iter().enumerate() {
                    let d = row[i];
                    if d.abs() <= 1e-14 {
                        return Err(KError::FactorError(
                            "BDDC coarse Jacobi backend encountered zero diagonal".into(),
                        ));
                    }
                    x[i] = rhs[i] / d;
                }
                Ok(x)
            }
            other => Err(KError::InvalidInput(format!(
                "unsupported BDDC coarse PC backend: {other}"
            ))),
        }
    }

    fn submatrix(a: &[Vec<S>], dofs: &[usize]) -> Vec<Vec<S>> {
        let mut out = vec![vec![S::zero(); dofs.len()]; dofs.len()];
        for (ii, &i) in dofs.iter().enumerate() {
            for (jj, &j) in dofs.iter().enumerate() {
                out[ii][jj] = a[i][j];
            }
        }
        out
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
            let operator = Self::extract_dense_operator(op, sym.local_n)?;
            let coarse_operator = Self::submatrix(&operator, &sym.coarse_dofs);
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
            let a_sub = Self::submatrix(&num.operator, &dofs);
            let local_sol = Self::solve_dense(a_sub, rhs)?;
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
        for (i, row) in num.operator.iter().enumerate() {
            let mut sum = S::zero();
            for (aij, zj) in row.iter().zip(y.iter()) {
                sum = sum + (*aij * *zj);
            }
            az[i] = sum;
        }
        let residual: Vec<S> = x
            .iter()
            .zip(az.iter())
            .map(|(&xi, &azi)| xi - azi)
            .collect();

        if !sym.coarse_dofs.is_empty() {
            let rc: Vec<S> = sym.coarse_dofs.iter().map(|&dof| residual[dof]).collect();
            let ec = self.coarse_solve(&num.coarse_operator, rc)?;
            for (k, &dof) in sym.coarse_dofs.iter().enumerate() {
                y[dof] = y[dof] + ec[k];
            }
        }

        // Phase 3: interface constraint enforcement with selectable scaling.
        for (k, &dof) in sym.interface_dofs.iter().enumerate() {
            if dof >= sym.local_n {
                continue;
            }
            let diag = num.operator[dof][dof];
            let multiplicity = sym.interface_multiplicity.get(k).copied().unwrap_or(1.0);
            y[dof] = self.apply_scaling(y[dof], diag, multiplicity);
        }
        Ok(())
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
    }
}
