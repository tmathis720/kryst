#![allow(dead_code)]

use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseSolve { CG, DirectDense, ILU }

pub trait CoarseSolver {
    fn setup(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError>;
    fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), KError>;
}

pub struct CoarseCg {
    tol: f64,
    maxit: usize,
    a: Option<CsrMatrix<f64>>,
}

impl CoarseCg {
    pub fn new(tol: f64, maxit: usize) -> Self { Self { tol, maxit, a: None } }
}

impl CoarseSolver for CoarseCg {
    fn setup(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> { self.a = Some(a.clone()); Ok(()) }
    fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), KError> {
        let a = self.a.as_ref().ok_or_else(|| KError::InvalidInput("CoarseCg not set up".into()))?;
        super::cg_sparse(a, b, x, self.tol, self.maxit)
    }
}

pub struct CoarseDenseLu {
    a: Option<faer::Mat<f64>>, // store dense matrix
}

impl CoarseDenseLu {
    pub fn new() -> Self { Self { a: None } }
}

impl CoarseSolver for CoarseDenseLu {
    fn setup(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        let dense = a.to_dense();
        self.a = Some(dense);
        Ok(())
    }
    fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), KError> {
        use faer::{MatRef, Conj, MatMut};
        use faer::linalg::solvers::{FullPivLu, SolveCore};
        let a = self.a.as_ref().ok_or_else(|| KError::InvalidInput("CoarseDenseLu not set up".into()))?;
        let n = a.nrows();
        if b.len() != n || x.len() != n { return Err(KError::InvalidInput("coarse LU: dim mismatch".into())); }
        // Compute factor and solve (small n)
        let lu = FullPivLu::new(MatRef::from(a.as_ref()));
        x.clone_from_slice(b);
        let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
        lu.solve_in_place_with_conj(Conj::No, x_mat);
        Ok(())
    }
}
