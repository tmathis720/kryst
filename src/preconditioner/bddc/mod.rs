//! Balancing Domain Decomposition by Constraints (BDDC) preconditioner.

use crate::algebra::scalar::{KrystScalar, R, S};
use crate::error::KError;
use crate::matrix::op::{DistLayout, LinOp};
use crate::parallel::UniverseComm;
use crate::preconditioner::{PcDistributedSupport, PcSide, Preconditioner};

#[derive(Debug, Clone)]
pub struct BddcConfig {
    pub coarse_ksp_type: Option<String>,
    pub coarse_pc_type: Option<String>,
    pub use_vertices: bool,
}

#[derive(Debug, Clone)]
pub struct BddcCoarseSpace {
    pub dofs: Vec<usize>,
    pub weights: Vec<R>,
}

#[derive(Debug, Clone)]
pub struct BddcConstraints {
    pub vertex_dofs: Vec<usize>,
    pub interface_dofs: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct BddcInterfaceCoupling {
    pub interface_dofs: Vec<usize>,
    pub subdomains: Vec<(usize, usize)>,
}

/// Prototype BDDC preconditioner.
///
/// Currently builds coarse space metadata and constraint/interface sets, while
/// applying a no-op preconditioning operator.
pub struct BddcPc {
    config: BddcConfig,
    dims: (usize, usize),
    local_n: usize,
    comm: UniverseComm,
    layout: Option<DistLayout>,
    coarse_space: Option<BddcCoarseSpace>,
    constraints: Option<BddcConstraints>,
    interface: Option<BddcInterfaceCoupling>,
    operator: Option<Vec<Vec<S>>>,
    coarse_operator: Option<Vec<Vec<S>>>,
}

impl BddcPc {
    pub fn new(config: BddcConfig) -> Self {
        Self {
            config,
            dims: (0, 0),
            local_n: 0,
            comm: UniverseComm::NoComm(crate::parallel::NoComm),
            layout: None,
            coarse_space: None,
            constraints: None,
            interface: None,
            operator: None,
            coarse_operator: None,
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
        use_vertices: bool,
    ) -> BddcConstraints {
        let mut vertex_dofs = Vec::new();
        if use_vertices {
            for (start, end) in subdomains {
                if *start < *end {
                    vertex_dofs.push(*start);
                    vertex_dofs.push(end.saturating_sub(1));
                }
            }
            vertex_dofs.sort_unstable();
            vertex_dofs.dedup();
        }
        BddcConstraints {
            vertex_dofs,
            interface_dofs: interface_dofs.to_vec(),
        }
    }

    fn build_coarse_space(constraints: &BddcConstraints) -> BddcCoarseSpace {
        let mut dofs = constraints.vertex_dofs.clone();
        dofs.extend_from_slice(&constraints.interface_dofs);
        dofs.sort_unstable();
        dofs.dedup();
        let weights = dofs.iter().map(|_| 1.0).collect();
        BddcCoarseSpace { dofs, weights }
    }

    fn prune_constraints(constraints: &mut BddcConstraints, n: usize) {
        constraints.vertex_dofs.retain(|&d| d < n);
        constraints.interface_dofs.retain(|&d| d < n);
        constraints.vertex_dofs.sort_unstable();
        constraints.vertex_dofs.dedup();
        constraints.interface_dofs.sort_unstable();
        constraints.interface_dofs.dedup();
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
        let local_n = dims.0;
        let subdomains = Self::build_subdomains(local_n);
        let interface_dofs = Self::build_interface(&subdomains);
        let mut constraints =
            Self::build_constraints(&subdomains, &interface_dofs, self.config.use_vertices);
        Self::prune_constraints(&mut constraints, local_n);
        let coarse_space = Self::build_coarse_space(&constraints);
        let operator = Self::extract_dense_operator(op, local_n)?;
        let coarse_operator = Self::submatrix(&operator, &coarse_space.dofs);

        self.interface = Some(BddcInterfaceCoupling {
            interface_dofs,
            subdomains,
        });
        self.constraints = Some(constraints);
        self.coarse_space = Some(coarse_space);
        self.coarse_operator = Some(coarse_operator);
        self.operator = Some(operator);
        self.local_n = local_n;
        self.comm = op.comm();
        self.layout = layout;
        self.dims = dims;
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if self.coarse_space.is_none()
            || self.constraints.is_none()
            || self.interface.is_none()
            || self.operator.is_none()
            || self.coarse_operator.is_none()
        {
            return Err(KError::InvalidInput("BDDC preconditioner not setup".into()));
        }
        if x.len() != self.local_n || y.len() != self.local_n {
            return Err(KError::InvalidInput(
                "BDDC apply expects vectors matching local ownership size".into(),
            ));
        }

        let a = self.operator.as_ref().expect("checked above");
        let interface = self.interface.as_ref().expect("checked above");
        let coarse_space = self.coarse_space.as_ref().expect("checked above");
        let coarse_op = self.coarse_operator.as_ref().expect("checked above");

        y.fill(S::zero());
        let mut multiplicity = vec![0usize; self.local_n];
        for &(start, end) in &interface.subdomains {
            let dofs: Vec<usize> = (start..end).collect();
            let rhs: Vec<S> = dofs.iter().map(|&i| x[i]).collect();
            let a_sub = Self::submatrix(a, &dofs);
            let local_sol = Self::solve_dense(a_sub, rhs)?;
            for (&dof, &val) in dofs.iter().zip(local_sol.iter()) {
                y[dof] = y[dof] + val;
                multiplicity[dof] += 1;
            }
        }
        for (i, yi) in y.iter_mut().enumerate() {
            if multiplicity[i] > 0 {
                let w = S::from_real(1.0 / multiplicity[i] as R);
                *yi = *yi * w;
            }
        }

        let mut az = vec![S::zero(); self.local_n];
        for (i, row) in a.iter().enumerate() {
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

        if !coarse_space.dofs.is_empty() {
            let mut rc = vec![S::zero(); coarse_space.dofs.len()];
            for (k, &dof) in coarse_space.dofs.iter().enumerate() {
                rc[k] = residual[dof] * S::from_real(coarse_space.weights[k]);
            }
            let ec = self.coarse_solve(coarse_op, rc)?;
            for (k, &dof) in coarse_space.dofs.iter().enumerate() {
                let wk = S::from_real(coarse_space.weights[k]);
                y[dof] = y[dof] + wk * ec[k];
            }
        }

        for &dof in &interface.interface_dofs {
            if dof < self.local_n && multiplicity[dof] > 1 {
                let w = S::from_real(1.0 / multiplicity[dof] as R);
                y[dof] = y[dof] * w;
            }
        }
        Ok(())
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
    }
}
