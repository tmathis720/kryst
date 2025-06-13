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

    fn solve(&mut self, a: &Mat<T>, b: &Vec<T>, x: &mut Vec<T>) -> Result<crate::utils::convergence::SolveStats<T>, KError> {
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

    fn solve(&mut self, a: &Mat<T>, b: &Vec<T>, x: &mut Vec<T>) -> Result<crate::utils::convergence::SolveStats<T>, KError> {
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
