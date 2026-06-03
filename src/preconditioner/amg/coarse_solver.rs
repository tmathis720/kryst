#![allow(dead_code)]

#[allow(unused_imports)]
use crate::algebra::blas::{dot_conj, nrm2};
#[allow(unused_imports)]
use crate::algebra::prelude::*;

use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
#[cfg(not(feature = "complex"))]
use crate::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind, IlutParams};
#[cfg(not(feature = "complex"))]
use crate::preconditioner::{PcSide, Preconditioner};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseSolve {
    CG,
    DirectDense,
    ILU,
    Smoother,
}

pub trait CoarseSolver<T: KrystScalar<Real = f64> = f64> {
    fn setup(&mut self, a: &CsrMatrix<T>) -> Result<(), KError>;
    fn solve(&mut self, b: &[T], x: &mut [T]) -> Result<(), KError>;
    fn nsetups(&self) -> usize {
        0
    }
}

pub struct CoarseCg<T: KrystScalar<Real = f64> = f64> {
    tol: f64,
    maxit: usize,
    a: Option<CsrMatrix<T>>,
}

impl<T: KrystScalar<Real = f64>> CoarseCg<T> {
    pub fn new(tol: f64, maxit: usize) -> Self {
        Self {
            tol,
            maxit,
            a: None,
        }
    }
}

impl<T: KrystScalar<Real = f64>> CoarseSolver<T> for CoarseCg<T> {
    fn setup(&mut self, a: &CsrMatrix<T>) -> Result<(), KError> {
        self.a = Some(a.clone());
        Ok(())
    }
    fn solve(&mut self, b: &[T], x: &mut [T]) -> Result<(), KError> {
        let a = self
            .a
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("CoarseCg not set up".into()))?;
        cg_sparse_generic(a, b, x, self.tol, self.maxit)
    }
    fn nsetups(&self) -> usize {
        0
    }
}

pub struct CoarseDenseLu<T: KrystScalar<Real = f64> = f64> {
    is_setup: bool,
    n: usize,
    a: Vec<T>,
}

impl<T: KrystScalar<Real = f64>> CoarseDenseLu<T> {
    pub fn new() -> Self {
        Self {
            is_setup: false,
            n: 0,
            a: Vec::new(),
        }
    }
}

impl<T: KrystScalar<Real = f64>> CoarseSolver<T> for CoarseDenseLu<T> {
    fn setup(&mut self, a: &CsrMatrix<T>) -> Result<(), KError> {
        if a.nrows() != a.ncols() {
            return Err(KError::InvalidInput(
                "coarse dense LU requires a square matrix".into(),
            ));
        }
        self.n = a.nrows();
        self.a = vec![T::zero(); self.n * self.n];
        for i in 0..self.n {
            let (cols, vals) = a.row(i);
            for (&j, &v) in cols.iter().zip(vals.iter()) {
                self.a[i * self.n + j] = v;
            }
        }
        self.is_setup = true;
        Ok(())
    }
    fn solve(&mut self, b: &[T], x: &mut [T]) -> Result<(), KError> {
        let n = self.n;
        if !self.is_setup {
            return Err(KError::InvalidInput("CoarseDenseLu not set up".into()));
        }
        if b.len() != n || x.len() != n {
            return Err(KError::InvalidInput("coarse LU: dim mismatch".into()));
        }
        dense_lu_solve(&self.a, b, x, n)?;
        Ok(())
    }

    fn nsetups(&self) -> usize {
        0
    }
}

#[cfg(not(feature = "complex"))]
pub struct CoarseIlu {
    a: Option<CsrMatrix<f64>>,
    ilu: Option<IluCsr>,
    tol: f64,
    maxit: usize,
    drop_tol: f64,
    fill_per_row: usize,
    nsetup: usize,
}

#[cfg(feature = "complex")]
pub struct CoarseIlu {
    _private: (),
}

#[cfg(not(feature = "complex"))]
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

#[cfg(feature = "complex")]
impl CoarseIlu {
    pub fn new(_tol: f64, _maxit: usize, _drop_tol: f64, _fill_per_row: usize) -> Self {
        Self { _private: () }
    }
}

#[cfg(not(feature = "complex"))]
impl CoarseSolver<f64> for CoarseIlu {
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

#[cfg(feature = "complex")]
impl CoarseSolver<f64> for CoarseIlu {
    fn setup(&mut self, _a: &CsrMatrix<f64>) -> Result<(), KError> {
        Err(KError::Unsupported(
            "AMG coarse ILU does not support complex scalars yet".into(),
        ))
    }

    fn solve(&mut self, _b: &[f64], _x: &mut [f64]) -> Result<(), KError> {
        Err(KError::Unsupported(
            "AMG coarse ILU does not support complex scalars yet".into(),
        ))
    }
}

fn dot_conj_generic<T: KrystScalar<Real = f64>>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y.iter())
        .fold(T::zero(), |acc, (&xi, &yi)| acc + xi.conj() * yi)
}

fn norm2_generic<T: KrystScalar<Real = f64>>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs2()).sum::<f64>().sqrt()
}

fn cg_sparse_generic<T: KrystScalar<Real = f64>>(
    a: &CsrMatrix<T>,
    b: &[T],
    x: &mut [T],
    tol: f64,
    maxit: usize,
) -> Result<(), KError> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n || x.len() != n {
        return Err(KError::InvalidInput("coarse CG: dim mismatch".into()));
    }
    if n == 0 {
        return Ok(());
    }
    x.fill(T::zero());

    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut ap = vec![T::zero(); n];

    let mut rsold = dot_conj_generic(&r, &r).real().max(0.0);
    let atol = tol.max(1e-12) * rsold.sqrt().max(1e-30);

    for _ in 0..maxit {
        a.spmv_scaled(T::one(), &p, T::zero(), &mut ap)?;
        let denom = dot_conj_generic(&p, &ap).real();
        if denom.abs() < 1e-30 {
            break;
        }
        let alpha = T::from_real(rsold / denom);
        for i in 0..n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * ap[i];
        }

        let rsnew = dot_conj_generic(&r, &r).real().max(0.0);
        if rsnew.sqrt() < atol {
            break;
        }
        let beta = T::from_real(rsnew / rsold);
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }
    Ok(())
}

fn dense_lu_solve<T: KrystScalar<Real = f64>>(
    a: &[T],
    b: &[T],
    x: &mut [T],
    n: usize,
) -> Result<(), KError> {
    if a.len() != n * n || b.len() != n || x.len() != n {
        return Err(KError::InvalidInput("coarse dense LU: dim mismatch".into()));
    }
    let mut lu = a.to_vec();
    let mut rhs = b.to_vec();
    for k in 0..n {
        let mut pivot = k;
        let mut pivot_abs = lu[k * n + k].abs();
        for i in (k + 1)..n {
            let mag = lu[i * n + k].abs();
            if mag > pivot_abs {
                pivot = i;
                pivot_abs = mag;
            }
        }
        if pivot_abs <= 1e-30 {
            return Err(KError::InvalidInput(
                "coarse dense LU: singular pivot".into(),
            ));
        }
        if pivot != k {
            for j in 0..n {
                lu.swap(k * n + j, pivot * n + j);
            }
            rhs.swap(k, pivot);
        }
        let pivot_val = lu[k * n + k];
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot_val;
            lu[i * n + k] = T::zero();
            for j in (k + 1)..n {
                lu[i * n + j] = lu[i * n + j] - factor * lu[k * n + j];
            }
            rhs[i] = rhs[i] - factor * rhs[k];
        }
    }

    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum = sum - lu[i * n + j] * x[j];
        }
        x[i] = sum / lu[i * n + i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr_from_rows<T: KrystScalar<Real = f64>>(
        n: usize,
        rows: Vec<Vec<(usize, T)>>,
    ) -> CsrMatrix<T> {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        row_ptr.push(0);
        for row in rows {
            for (j, v) in row {
                col_idx.push(j);
                vals.push(v);
            }
            row_ptr.push(col_idx.len());
        }
        CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
    }

    #[test]
    fn dense_lu_solves_real_coarse_system() {
        let a = csr_from_rows(2, vec![vec![(0, 4.0), (1, 1.0)], vec![(0, 2.0), (1, 3.0)]]);
        let mut solver = CoarseDenseLu::<f64>::new();
        solver.setup(&a).unwrap();
        let mut x = vec![0.0; 2];
        solver.solve(&[1.0, 1.0], &mut x).unwrap();
        assert!((x[0] - 0.2).abs() < 1e-12);
        assert!((x[1] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn cg_solves_real_spd_coarse_system() {
        let a = csr_from_rows(2, vec![vec![(0, 4.0), (1, 1.0)], vec![(0, 1.0), (1, 3.0)]]);
        let mut solver = CoarseCg::<f64>::new(1e-14, 20);
        solver.setup(&a).unwrap();
        let mut x = vec![0.0; 2];
        solver.solve(&[1.0, 2.0], &mut x).unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-10);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-10);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn dense_lu_preserves_complex_values() {
        let a = csr_from_rows(
            2,
            vec![
                vec![
                    (0, crate::S::from_parts(2.0, 1.0)),
                    (1, crate::S::from_parts(0.0, -1.0)),
                ],
                vec![
                    (0, crate::S::from_parts(1.0, 0.5)),
                    (1, crate::S::from_parts(3.0, 0.0)),
                ],
            ],
        );
        let x_expected = vec![
            crate::S::from_parts(1.0, -0.5),
            crate::S::from_parts(0.25, 0.75),
        ];
        let mut b = vec![crate::S::zero(); 2];
        a.spmv_scaled(crate::S::one(), &x_expected, crate::S::zero(), &mut b)
            .unwrap();
        let mut solver = CoarseDenseLu::<crate::S>::new();
        solver.setup(&a).unwrap();
        let mut x = vec![crate::S::zero(); 2];
        solver.solve(&b, &mut x).unwrap();
        for (got, want) in x.iter().zip(x_expected.iter()) {
            assert!((*got - *want).abs() < 1e-12);
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn cg_uses_conjugate_products_for_hpd_complex_system() {
        let a = csr_from_rows(
            2,
            vec![
                vec![
                    (0, crate::S::from_parts(3.0, 0.0)),
                    (1, crate::S::from_parts(1.0, 1.0)),
                ],
                vec![
                    (0, crate::S::from_parts(1.0, -1.0)),
                    (1, crate::S::from_parts(4.0, 0.0)),
                ],
            ],
        );
        let x_expected = vec![
            crate::S::from_parts(0.5, -0.25),
            crate::S::from_parts(-0.1, 0.3),
        ];
        let mut b = vec![crate::S::zero(); 2];
        a.spmv_scaled(crate::S::one(), &x_expected, crate::S::zero(), &mut b)
            .unwrap();
        let mut solver = CoarseCg::<crate::S>::new(1e-14, 20);
        solver.setup(&a).unwrap();
        let mut x = vec![crate::S::zero(); 2];
        solver.solve(&b, &mut x).unwrap();
        assert!(norm2_generic(&[x[0] - x_expected[0], x[1] - x_expected[1]]) < 1e-10);
    }
}
