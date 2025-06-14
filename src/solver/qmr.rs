//! QMR solver (Saad §7.3)

use crate::solver::LinearSolver;
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::SolveStats;
use crate::error::KError;
use crate::utils::convergence::Convergence;

pub struct QmrSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float> QmrSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for QmrSolver<T>
where
    T: num_traits::Float + From<f64>,
{
    type Error = KError;
    type Scalar = T;
    fn solve(&mut self, _a: &M, _pc: Option<&dyn Preconditioner<M, V>>, _b: &V, _x: &mut V) -> Result<SolveStats<T>, KError> {
        unimplemented!("QMR solver not yet implemented");
    }
}
