use crate::parallel::UniverseComm;
use crate::utils::convergence::SolveStats;

pub trait Solver<A, S = f64> {
    type Error;

    fn setup(&mut self, a: &A, comm: &UniverseComm) -> Result<(), Self::Error>;

    fn factor(&mut self, a: &A) -> Result<(), Self::Error>;

    fn solve(&mut self, b: &[S], x: &mut [S], comm: &UniverseComm)
        -> Result<SolveStats<S>, Self::Error>;

    fn reuse_factorization(&self) -> bool {
        false
    }
}
