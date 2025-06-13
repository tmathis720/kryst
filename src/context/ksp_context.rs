//! Factory for Krylov methods (KSP).

use crate::error::KError;
use crate::solver::{CgSolver, GmresSolver, LinearSolver};

pub enum KspType {
    Cg { tol: f64, max_iters: usize },
    Gmres { restart: usize, tol: f64, max_iters: usize },
}

pub struct KspContext {
    pub kind: KspType,
}

impl KspContext {
    pub fn new(kind: KspType) -> Self {
        Self { kind }
    }

    /// Build an instance of LinearSolver<M,V>.
    pub fn build<M, V, T>(&self) -> Box<dyn LinearSolver<M, V, Error = KError, Scalar = T>>
    where
        M: 'static + crate::core::traits::MatVec<V>,
        (): crate::core::traits::InnerProduct<V, Scalar = T>,
        V: 'static + AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone,
        T: 'static + num_traits::Float + Clone + From<f64>,
    {
        match self.kind {
            KspType::Cg { tol, max_iters } => Box::new(CgSolver::new(tol.into(), max_iters)),
            KspType::Gmres { restart, tol, max_iters } => {
                Box::new(GmresSolver::new(restart, tol.into(), max_iters))
            }
        }
    }
}
