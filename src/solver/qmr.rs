//! QMR solver (Saad §7.3)

use crate::utils::convergence::Convergence;

pub struct QmrSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float> QmrSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}
