#![allow(dead_code)]

use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind, IlutParams};
use crate::preconditioner::{PcSide, Preconditioner};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseSolve {
    CG,
    DirectDense,
    ILU,
}

pub trait CoarseSolver {
    fn setup(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError>;
    fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), KError>;
    fn nsetups(&self) -> usize {
        0
    }
}

pub struct CoarseCg {
    tol: f64,
    maxit: usize,
    a: Option<CsrMatrix<f64>>,
}

impl CoarseCg {
    pub fn new(tol: f64, maxit: usize) -> Self {
        Self {
            tol,
            maxit,
            a: None,
        }
    }
}

impl CoarseSolver for CoarseCg {
    fn setup(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        self.a = Some(a.clone());
        Ok(())
    }
    fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), KError> {
        let a = self
            .a
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("CoarseCg not set up".into()))?;
        super::cg_sparse(a, b, x, self.tol, self.maxit)
    }
    fn nsetups(&self) -> usize {
        0
    }
}

pub struct CoarseDenseLu {
    a: Option<faer::Mat<f64>>, // store dense matrix
}

impl CoarseDenseLu {
    pub fn new() -> Self {
        Self { a: None }
    }
}

impl CoarseSolver for CoarseDenseLu {
    fn setup(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        let dense = a.to_dense();
        self.a = Some(dense);
        Ok(())
    }
    fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), KError> {
        use faer::linalg::solvers::{FullPivLu, SolveCore};
        use faer::{Conj, MatMut, MatRef};
        let a = self
            .a
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("CoarseDenseLu not set up".into()))?;
        let n = a.nrows();
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput("coarse LU: dim mismatch".into()));
        }
        // Compute factor and solve (small n)
        let lu = FullPivLu::new(MatRef::from(a.as_ref()));
        x.clone_from_slice(b);
        let x_mat = MatMut::from_column_major_slice_mut(x, n, 1);
        lu.solve_in_place_with_conj(Conj::No, x_mat);
        Ok(())
    }

    fn nsetups(&self) -> usize {
        0
    }
}

pub struct CoarseIlu {
    a: Option<CsrMatrix<f64>>,
    ilu: Option<IluCsr>,
    tol: f64,
    maxit: usize,
    drop_tol: f64,
    fill_per_row: usize,
    nsetup: usize,
}

impl CoarseIlu {
    pub fn new(tol: f64, maxit: usize, drop_tol: f64, fill_per_row: usize) -> Self {
        Self {
            a: None,
            ilu: None,
            tol,
            maxit,
            drop_tol,
            fill_per_row,
            nsetup: 0,
        }
    }
}

impl CoarseSolver for CoarseIlu {
    fn setup(&mut self, a: &CsrMatrix<f64>) -> Result<(), KError> {
        self.a = Some(a.clone());
        let mut cfg = IluCsrConfig::default();
        cfg.kind = IluKind::Ilut {
            params: IlutParams {
                droptol_abs: self.drop_tol,
                droptol_rel: self.drop_tol,
                p_l: self.fill_per_row,
                p_u: self.fill_per_row,
                ..Default::default()
            },
        };
        cfg.level_sched = false;
        let mut ilu = IluCsr::new_with_config(cfg);
        ilu.setup(a)?;
        self.ilu = Some(ilu);
        self.nsetup += 1;
        Ok(())
    }

    fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), KError> {
        let a = self
            .a
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("CoarseIlu not set up".into()))?;
        let ilu = self
            .ilu
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("CoarseIlu missing ILU".into()))?;
        super::pcg_left_precond(a, b, x, self.tol, self.maxit, |r, z| {
            ilu.apply(PcSide::Left, r, z)
        })
    }

    fn nsetups(&self) -> usize {
        self.nsetup
    }
}
