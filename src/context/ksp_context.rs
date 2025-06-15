//! Factory for Krylov methods (KSP).

use crate::solver::{CgSolver, GmresSolver, LinearSolver, PcgSolver, gmres::Preconditioning};
use crate::solver::fgmres::FgmresSolver;
use crate::preconditioner::Preconditioner;

pub enum SolverKind {
    Cg,
    Pcg,
    GmresLeft,
    GmresRight,
    Fgmres,
    Bicgstab,
    Cgs,
    Qmr,
    Tfqmr,
    Minres,
    Cgnr,
}

pub struct KspContext<M, V, T> {
    pub kind: SolverKind,
    pub a: M,
    pub pc: Option<Box<dyn Preconditioner<M, V>>>,
    pub flex_pc: Option<Box<dyn crate::preconditioner::FlexiblePreconditioner<M, V>>>,
    pub tol: T,
    pub max_it: usize,
    pub restart: usize, // for GMRES
}

impl<M, V, T> KspContext<M, V, T>
where
    M: 'static + crate::core::traits::MatVec<V> + crate::core::traits::MatTransVec<V> + std::fmt::Debug,
    (): crate::core::traits::InnerProduct<V, Scalar = T>,
    V: 'static + AsMut<[T]> + AsRef<[T]> + From<Vec<T>> + Clone + std::fmt::Debug,
    T: 'static + num_traits::Float + Clone + From<f64> + std::fmt::Debug + std::ops::AddAssign + std::iter::Sum + num_traits::FromPrimitive,
{
    pub fn solve_context(&mut self, b: &V, x: &mut V) -> Result<crate::utils::convergence::SolveStats<T>, crate::error::KError> {
        match self.kind {
            SolverKind::Cg => {
                let mut solver = CgSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Pcg => {
                let mut solver = PcgSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::GmresLeft => {
                let mut solver = GmresSolver::new(self.restart, self.tol, self.max_it).with_preconditioning(Preconditioning::Left);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::GmresRight => {
                let mut solver = GmresSolver::new(self.restart, self.tol, self.max_it).with_preconditioning(Preconditioning::Right);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Fgmres => {
                let mut solver = FgmresSolver::new(self.tol, self.max_it, self.restart);
                match self.flex_pc {
                    Some(ref mut flex_pc) => solver.solve_flex(&self.a, Some(flex_pc.as_mut()), b, x),
                    None => solver.solve_flex(&self.a, None, b, x),
                }
            }
            SolverKind::Bicgstab => {
                let mut solver = crate::solver::bicgstab::BiCgStabSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Cgs => {
                let mut solver = crate::solver::cgs::CgsSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Qmr => {
                return self.solve_qmr(b, x);
            }
            SolverKind::Tfqmr => {
                let mut solver = crate::solver::tfqmr::TfqmrSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Minres => {
                let mut solver = crate::solver::minres::MinresSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
            SolverKind::Cgnr => {
                let mut solver = crate::solver::cgnr::CgnrSolver::new(self.tol, self.max_it);
                solver.solve(&self.a, self.pc.as_deref(), b, x)
            }
        }
    }
    #[allow(unused)]
    fn solve_qmr(&mut self, b: &V, x: &mut V) -> Result<crate::utils::convergence::SolveStats<T>, crate::error::KError>
    where
        M: crate::core::traits::MatTransVec<V>,
    {
        let mut solver = crate::solver::qmr::QmrSolver::new(self.tol, self.max_it);
        solver.solve(&self.a, self.pc.as_deref(), b, x)
    }
}
