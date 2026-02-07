#![cfg(all(feature = "complex", feature = "backend-faer"))]

use approx::assert_abs_diff_eq;
use kryst::KspContext;
use kryst::algebra::prelude::*;
use kryst::context::ksp_context::SolverType;
use kryst::context::pc_context::PcType;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::utils::convergence::ConvergedReason;
use std::sync::Arc;

fn apply_csr(a: &CsrMatrix<S>, x: &[S]) -> Vec<S> {
    let mut y = vec![S::zero(); a.nrows()];
    a.spmv(x, &mut y);
    y
}

#[test]
fn ksp_gmres_identity_complex() {
    let n = 4;
    let csr = CsrMatrix::identity(n);
    let op = Arc::new(CsrOp::new(Arc::new(csr)));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.set_operators(op, None);

    let b = vec![
        S::from_parts(1.0, -0.5),
        S::from_parts(-2.0, 0.75),
        S::from_parts(0.5, 1.25),
        S::from_parts(-1.5, -0.25),
    ];
    let mut x = vec![S::zero(); n];

    let stats = ksp.solve(&b, &mut x).expect("GMRES solve");
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    for (xi, bi) in x.iter().zip(b.iter()) {
        assert_abs_diff_eq!(xi.real(), bi.real(), epsilon = 1e-10);
        assert_abs_diff_eq!(xi.imag(), bi.imag(), epsilon = 1e-10);
    }
}

#[test]
fn ksp_cg_hermitian_pd_complex() {
    // Hermitian positive definite: [[2, 1+i], [1-i, 2]]
    let row_ptr = vec![0, 2, 4];
    let col_idx = vec![0, 1, 0, 1];
    let values = vec![
        S::from_real(2.0),
        S::from_parts(1.0, 1.0),
        S::from_parts(1.0, -1.0),
        S::from_real(2.0),
    ];
    let csr = CsrMatrix::from_csr(2, 2, row_ptr, col_idx, values);
    let op = Arc::new(CsrOp::new(Arc::new(csr.clone())));

    let x_true = vec![S::from_parts(1.0, 0.5), S::from_parts(-0.25, 1.25)];
    let b = apply_csr(&csr, &x_true);

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Cg).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 20);
    ksp.set_operators(op, None);

    let mut x = vec![S::zero(); 2];
    let stats = ksp.solve(&b, &mut x).expect("CG solve");
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    for (xi, xt) in x.iter().zip(x_true.iter()) {
        assert_abs_diff_eq!(xi.real(), xt.real(), epsilon = 1e-9);
        assert_abs_diff_eq!(xi.imag(), xt.imag(), epsilon = 1e-9);
    }
}
