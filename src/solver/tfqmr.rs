//! TFQMR solver (Saad §7.4)

use crate::solver::LinearSolver;
use crate::preconditioner::Preconditioner;
use crate::utils::convergence::SolveStats;
use crate::error::KError;

use crate::utils::convergence::Convergence;

pub struct TfqmrSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float> TfqmrSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}

impl<M, V, T> LinearSolver<M, V> for TfqmrSolver<T>
where
    T: num_traits::Float + From<f64>,
{
    type Error = KError;
    type Scalar = T;
    fn solve(&mut self, _a: &M, _pc: Option<&dyn Preconditioner<M, V>>, _b: &V, _x: &mut V) -> Result<SolveStats<T>, KError> {
        unimplemented!("TFQMR solver not yet implemented");
    }
}
