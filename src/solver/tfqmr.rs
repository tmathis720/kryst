//! TFQMR solver (Saad §7.4)

use crate::utils::convergence::Convergence;

pub struct TfqmrSolver<T> {
    pub conv: Convergence<T>,
}

impl<T: num_traits::Float> TfqmrSolver<T> {
    pub fn new(tol: T, max_iters: usize) -> Self {
        Self { conv: Convergence { tol, max_iters } }
    }
}
