//! Convergence tracking & tolerance checks for iterative solvers.

/// Stopping criteria & stats.
pub struct Convergence<T> {
    pub tol: T,
    pub max_iters: usize,
}

#[derive(Clone, Debug)]
pub struct SolveStats<T> {
    pub iterations: usize,
    pub final_residual: T,
    pub converged: bool,
}

impl<T: Copy + num_traits::Float> Convergence<T> {
    /// Returns (should_stop, stats) given current `res_norm` and iteration `i`.
    pub fn check(
        &self,
        res_norm: T,
        res0_norm: T,
        i: usize,
    ) -> (bool, SolveStats<T>) {
        let rel = res_norm / res0_norm;
        let converged = rel <= self.tol || i >= self.max_iters;
        (
            converged,
            SolveStats {
                iterations: i,
                final_residual: res_norm,
                converged,
            },
        )
    }
}
