//! Balancing Domain Decomposition by Constraints (BDDC) preconditioner.

use crate::algebra::scalar::{R, S};
use crate::error::KError;
use crate::matrix::op::LinOp;
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
    coarse_space: Option<BddcCoarseSpace>,
    constraints: Option<BddcConstraints>,
    interface: Option<BddcInterfaceCoupling>,
}

impl BddcPc {
    pub fn new(config: BddcConfig) -> Self {
        Self {
            config,
            dims: (0, 0),
            coarse_space: None,
            constraints: None,
            interface: None,
        }
    }

    fn build_subdomains(n: usize) -> Vec<(usize, usize)> {
        if n <= 1 {
            return vec![(0, n)];
        }
        let mid = n / 2;
        vec![(0, mid), (mid, n)]
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
        let subdomains = Self::build_subdomains(dims.0);
        let interface_dofs = Self::build_interface(&subdomains);
        let constraints =
            Self::build_constraints(&subdomains, &interface_dofs, self.config.use_vertices);
        let coarse_space = Self::build_coarse_space(&constraints);
        self.interface = Some(BddcInterfaceCoupling {
            interface_dofs,
            subdomains,
        });
        self.constraints = Some(constraints);
        self.coarse_space = Some(coarse_space);
        self.dims = dims;
        Ok(())
    }

    fn apply(&self, _side: PcSide, x: &[S], y: &mut [S]) -> Result<(), KError> {
        if self.coarse_space.is_none() || self.constraints.is_none() || self.interface.is_none() {
            return Err(KError::InvalidInput("BDDC preconditioner not setup".into()));
        }
        y.copy_from_slice(x);
        Ok(())
    }

    fn distributed_support(&self) -> PcDistributedSupport {
        PcDistributedSupport::Distributed
    }
}
