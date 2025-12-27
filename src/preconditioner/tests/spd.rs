#![cfg(not(feature = "complex"))]

use crate::algebra::prelude::*;
use crate::error::KError;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::amg::{AMGBuilder, CoarseSolve, RelaxType};
use crate::preconditioner::{PcSide, Preconditioner};
use faer::Mat;

fn poisson_1d(n: usize) -> CsrMatrix<R> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(R::from(-1.0));
        }
        col_idx.push(i);
        values.push(R::from(2.0));
        if i + 1 < n {
            col_idx.push(i + 1);
            values.push(R::from(-1.0));
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, values)
}

#[test]
#[cfg(not(feature = "complex"))]
fn spd_preconditioned_operator_positive() {
    let a = poisson_1d(16);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .require_spd(true)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();

    let caps = amg.capabilities();
    assert!(caps.is_spd);
    assert_eq!(caps.side_restriction, Some(PcSide::Left));

    let n = a.nrows();
    let mut x = vec![R::default(); n];
    let mut ax = vec![R::default(); n];
    let mut y = vec![R::default(); n];
    for t in 0..5 {
        for i in 0..n {
            x[i] = R::from(((i + 17 * t) % 11) as f64) - R::from(5.0);
        }
        let alpha = S::from_real(1.0).real();
        let beta = R::default();
        a.spmv_scaled(alpha, &x, beta, &mut ax).unwrap();
        y.fill(R::default());
        amg.apply(PcSide::Left, &ax, &mut y).unwrap();
        let qf = x.iter().zip(&y).map(|(xi, yi)| *xi * *yi).sum::<R>();
        assert!(
            qf > R::default(),
            "quadratic form should be positive, got {qf}"
        );
    }
}

#[test]
#[cfg(not(feature = "complex"))]
fn spd_enforces_r_equals_pt() {
    let a = poisson_1d(12);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .require_spd(true)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();
    assert!(amg.debug_levels_r_equals_pt());
}

#[test]
fn spd_rejects_ilu_coarse_solver() {
    let a = poisson_1d(20);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .coarse_solve(CoarseSolve::ILU)
        .require_spd(true)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    let err = amg.setup(&a).unwrap_err();
    match err {
        KError::InvalidInput(msg) => {
            assert!(msg.contains("SPD mode requires"))
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn spd_requires_symmetric_sweeps() {
    let a = poisson_1d(10);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .smoothing_sweeps(2, 1)
        .require_spd(true)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    let err = amg.setup(&a).unwrap_err();
    match err {
        KError::InvalidInput(msg) => {
            assert!(msg.to_lowercase().contains("symmetric pre/post"))
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn spd_rejects_non_galerkin_when_forbidden() {
    let a = poisson_1d(18);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .non_galerkin(true, 1, R::from(1e-3), R::from(1e-2), 4)
        .require_spd(true)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    let err = amg.setup(&a).unwrap_err();
    match err {
        KError::InvalidInput(msg) => {
            assert!(msg.to_lowercase().contains("non-galerkin filtering"))
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn spd_right_side_errors() {
    let a = poisson_1d(14);
    let mut amg = AMGBuilder::new()
        .relaxation_type(RelaxType::Jacobi)
        .grid_relax_type_all(RelaxType::Jacobi)
        .require_spd(true)
        .build(&Mat::<R>::zeros(0, 0))
        .unwrap();
    amg.setup(&a).unwrap();
    let rhs = vec![R::from(1.0); a.nrows()];
    let mut z = vec![R::default(); a.nrows()];
    let err = amg.apply(PcSide::Right, &rhs, &mut z).unwrap_err();
    match err {
        KError::InvalidInput(msg) => {
            assert!(msg.to_lowercase().contains("left preconditioning"))
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}
