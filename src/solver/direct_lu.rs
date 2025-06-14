//! Direct dense solves via Faer.

use crate::error::KError;
use crate::solver::LinearSolver;
use faer::linalg::solvers::{FullPivLu, Qr, SolveCore};
use faer::{Mat, MatMut, Conj};
use faer::traits::{ComplexField, RealField};

// LU solver
pub struct LuSolver<T> {
    factor: Option<FullPivLu<T>>,
}

impl<T: ComplexField + RealField> LuSolver<T> {
    pub fn new() -> Self {
        LuSolver { factor: None }
    }
}

impl<T: ComplexField + RealField + Copy + PartialOrd + From<f64>> LinearSolver<Mat<T>, Vec<T>> for LuSolver<T> {
    type Error = KError;
    type Scalar = T;

    fn solve(&mut self, a: &Mat<T>, pc: Option<&dyn crate::preconditioner::Preconditioner<Mat<T>, Vec<T>>>, b: &Vec<T>, x: &mut Vec<T>) -> Result<crate::utils::convergence::SolveStats<T>, KError> {
        let _ = pc; // Direct solvers do not use preconditioner
        // Factorize if needed
        let factor = FullPivLu::new(a.as_ref());
        self.factor = Some(factor);
        // Copy b into x
        x.clone_from(b);
        // Avoid double borrow: get len first
        let n = x.len();
        let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
        self.factor
            .as_ref()
            .unwrap()
            .solve_in_place_with_conj(Conj::No, x_mat);
        // For direct solvers, always converged in 1 iteration
        Ok(crate::utils::convergence::SolveStats {
            iterations: 1,
            final_residual: T::zero(),
            converged: true,
        })
    }
}

// QR solver
pub struct QrSolver;

impl QrSolver {
    pub fn new() -> Self {
        QrSolver
    }
}

impl<T: ComplexField + RealField + Copy + PartialOrd + From<f64>> LinearSolver<Mat<T>, Vec<T>> for QrSolver {
    type Error = KError;
    type Scalar = T;

    fn solve(&mut self, a: &Mat<T>, pc: Option<&dyn crate::preconditioner::Preconditioner<Mat<T>, Vec<T>>>, b: &Vec<T>, x: &mut Vec<T>) -> Result<crate::utils::convergence::SolveStats<T>, KError> {
        let _ = pc; // Direct solvers do not use preconditioner
        let factor = Qr::new(a.as_ref());
        x.clone_from(b);
        let n = x.len();
        let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
        factor.solve_in_place_with_conj(Conj::No, x_mat);
        Ok(crate::utils::convergence::SolveStats {
            iterations: 1,
            final_residual: T::zero(),
            converged: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Mat};
    use crate::solver::LinearSolver;

    #[test]
    fn lu_solver_solves_dense_system() {
        // 3x3 system: [[2,1,1],[1,3,2],[1,0,0]] x = [4,5,6]
        // True solution: [6,15,-23]
        let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0,0) => 2.0, (0,1) => 1.0, (0,2) => 1.0,
            (1,0) => 1.0, (1,1) => 3.0, (1,2) => 2.0,
            (2,0) => 1.0, (2,1) => 0.0, (2,2) => 0.0,
            _ => 0.0
        });
        let b = vec![4.0, 5.0, 6.0];
        let mut x = vec![0.0; 3];
        let mut solver = LuSolver::<f64>::new();
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let expected = vec![6.0, 15.0, -23.0];
        let tol = 1e-10;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((*xi as f64 - *ei as f64).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged);
    }

    #[test]
    fn qr_solver_solves_dense_system() {
        // 3x3 system: [[2,1,1],[1,3,2],[1,0,0]] x = [4,5,6]
        // True solution: [6,15,-23]
        let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0,0) => 2.0, (0,1) => 1.0, (0,2) => 1.0,
            (1,0) => 1.0, (1,1) => 3.0, (1,2) => 2.0,
            (2,0) => 1.0, (2,1) => 0.0, (2,2) => 0.0,
            _ => 0.0
        });
        let b = vec![4.0, 5.0, 6.0];
        let mut x = vec![0.0; 3];
        let mut solver = QrSolver::new();
        let stats = solver.solve(&a, None, &b, &mut x).unwrap();
        let expected = vec![6.0, 15.0, -23.0];
        let tol = 1e-10;
        for (xi, ei) in x.iter().zip(expected.iter()) {
            assert!((*xi as f64 - *ei as f64).abs() < tol, "xi = {}, expected = {}", xi, ei);
        }
        assert!(stats.converged);
    }
}
