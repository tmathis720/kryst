use crate::matrix::sparse::CsrMatrix;
use crate::parallel::UniverseComm;
use crate::solver::MonitorCallback;
use crate::solver::api::Solver as CanonicalSolver;
use crate::solver::legacy;
use crate::{error::KError, utils::convergence::SolveStats};

pub struct LegacyDirectAdapter<'a, S: CanonicalSolver<CsrMatrix<f64>, f64, Error = KError>> {
    inner: &'a mut S,
    comm: &'a UniverseComm,
}

impl<'a, S: CanonicalSolver<CsrMatrix<f64>, f64, Error = KError>> LegacyDirectAdapter<'a, S> {
    pub fn new(inner: &'a mut S, comm: &'a UniverseComm) -> Self {
        Self { inner, comm }
    }
}

impl<'a, S: CanonicalSolver<CsrMatrix<f64>, f64, Error = KError>>
    legacy::LinearSolver<CsrMatrix<f64>, Vec<f64>> for LegacyDirectAdapter<'a, S>
{
    type Error = KError;
    type Scalar = f64;

    fn solve(
        &mut self,
        a: &CsrMatrix<f64>,
        _pc: Option<
            &(dyn crate::preconditioner::legacy::Preconditioner<CsrMatrix<f64>, Vec<f64>> + '_),
        >,
        b: &Vec<f64>,
        x: &mut Vec<f64>,
        _pc_side: crate::preconditioner::PcSide,
        _comm: &UniverseComm,
        _monitors: Option<&[Box<MonitorCallback<Self::Scalar>>]>,
        _work: Option<&mut crate::context::ksp_context::Workspace>,
    ) -> Result<SolveStats<f64>, KError> {
        self.inner.setup(a, self.comm)?;
        self.inner.factor(a)?;
        self.inner.solve(b.as_slice(), x.as_mut_slice(), self.comm)
    }
}
