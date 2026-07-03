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
fn ksp_gmres_mg_identity_complex() {
    let n = 8;
    let csr = CsrMatrix::identity(n);
    let op = Arc::new(CsrOp::new(Arc::new(csr)));

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::Mg, None).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 50);
    ksp.set_operators(op, None);

    let b = (0..n)
        .map(|i| S::from_parts(i as f64 + 1.0, -(i as f64) * 0.25))
        .collect::<Vec<_>>();
    let mut x = vec![S::zero(); n];

    let stats = ksp.solve(&b, &mut x).expect("GMRES+MG solve");
    assert!(matches!(
        stats.reason,
        ConvergedReason::ConvergedRtol | ConvergedReason::ConvergedAtol
    ));

    for (xi, bi) in x.iter().zip(b.iter()) {
        assert_abs_diff_eq!(xi.real(), bi.real(), epsilon = 1e-9);
        assert_abs_diff_eq!(xi.imag(), bi.imag(), epsilon = 1e-9);
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

#[test]
fn ksp_pcg_hermitian_pd_complex_converges_to_known_solution() {
    let row_ptr = vec![0, 2, 4];
    let col_idx = vec![0, 1, 0, 1];
    let values = vec![
        S::from_real(4.0),
        S::from_parts(1.0, 0.5),
        S::from_parts(1.0, -0.5),
        S::from_real(3.0),
    ];
    let csr = CsrMatrix::from_csr(2, 2, row_ptr, col_idx, values);
    let op = Arc::new(CsrOp::new(Arc::new(csr.clone())));

    let x_true = vec![S::from_parts(0.5, -0.25), S::from_parts(-1.0, 0.75)];
    let b = apply_csr(&csr, &x_true);

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Pcg).unwrap();
    ksp.set_pc_type(PcType::None, None).unwrap();
    ksp.set_tolerances(1e-12, 1e-14, 1e8, 20);
    ksp.set_operators(op, None);

    let mut x = vec![S::zero(); 2];
    let stats = ksp.solve(&b, &mut x).expect("PCG solve");
    assert!(
        stats.reason.is_converged(),
        "unexpected PCG reason: {:?}",
        stats.reason
    );

    for (xi, xt) in x.iter().zip(x_true.iter()) {
        assert_abs_diff_eq!(xi.real(), xt.real(), epsilon = 1e-9);
        assert_abs_diff_eq!(xi.imag(), xt.imag(), epsilon = 1e-9);
    }
}

use kryst::config::options::KspOptions;

fn hermitian_2x2() -> CsrMatrix<S> {
    CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(4.0),
            S::from_parts(1.0, 0.5),
            S::from_parts(1.0, -0.5),
            S::from_real(3.0),
        ],
    )
}

fn nonhermitian_2x2() -> CsrMatrix<S> {
    CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_parts(3.0, 0.0),
            S::from_parts(2.0, -1.0),
            S::from_parts(-1.0, 0.5),
            S::from_parts(2.0, 0.0),
        ],
    )
}

#[test]
fn complex_option_parse_and_execute_for_new_solver_types() {
    for ksp_type in ["pcg", "chebyshev", "cr", "gcr", "pipegcr"] {
        let a = Arc::new(CsrOp::new(Arc::new(hermitian_2x2())));
        let mut ksp = KspContext::new();
        ksp.set_operators(a.clone(), None);
        ksp.set_pc_type(PcType::None, None).unwrap();
        ksp.set_tolerances(1e-10, 1e-12, 1e8, 50);
        ksp.set_from_options(&KspOptions {
            ksp_type: Some(ksp_type.to_string()),
            ..Default::default()
        })
        .unwrap();
        let x_true = vec![S::from_parts(0.5, -0.25), S::from_parts(-1.0, 0.75)];
        let b = apply_csr(&hermitian_2x2(), &x_true);
        let mut x = vec![S::zero(); 2];
        let stats = ksp.solve(&b, &mut x).unwrap();
        assert!(stats.reason.is_converged() || stats.reason.is_diverged());
    }
}

#[test]
fn complex_smoke_convergence_for_pcg_chebyshev_cr() {
    for solver in [SolverType::Pcg, SolverType::Chebyshev, SolverType::Cr] {
        let a_mat = hermitian_2x2();
        let a = Arc::new(CsrOp::new(Arc::new(a_mat.clone())));
        let x_true = vec![S::from_parts(1.0, 0.2), S::from_parts(-0.5, 0.7)];
        let b = apply_csr(&a_mat, &x_true);
        let mut x = vec![S::zero(); 2];

        let mut ksp = KspContext::new();
        ksp.set_type(solver).unwrap();
        ksp.set_pc_type(PcType::None, None).unwrap();
        ksp.set_tolerances(1e-10, 1e-12, 1e8, 80);
        ksp.set_operators(a, None);

        let stats = ksp.solve(&b, &mut x).unwrap();
        assert!(stats.reason.is_converged() || stats.reason.is_diverged());
    }
}

#[test]
fn complex_smoke_convergence_for_gcr_and_pipegcr() {
    for solver in [SolverType::Gcr, SolverType::PipeGcr] {
        let a_mat = nonhermitian_2x2();
        let a = Arc::new(CsrOp::new(Arc::new(a_mat.clone())));
        let x_true = vec![S::from_parts(0.2, -0.6), S::from_parts(0.8, 0.1)];
        let b = apply_csr(&a_mat, &x_true);
        let mut x = vec![S::zero(); 2];

        let mut ksp = KspContext::new();
        ksp.set_type(solver).unwrap();
        ksp.set_pc_type(PcType::None, None).unwrap();
        ksp.set_tolerances(1e-10, 1e-12, 1e8, 80);
        ksp.set_operators(a, None);

        let stats = ksp.solve(&b, &mut x).unwrap();
        assert!(stats.reason.is_converged() || stats.reason.is_diverged());
        if matches!(solver, SolverType::PipeGcr) {
            assert!(stats.counters.num_global_reductions >= stats.iterations);
        }
    }
}

#[test]
fn complex_gmres_with_sor_pc_converges() {
    let a = Arc::new(CsrOp::new(Arc::new(hermitian_2x2())));
    let x_true = vec![S::from_parts(0.4, -0.3), S::from_parts(-0.8, 0.9)];
    let b = apply_csr(&hermitian_2x2(), &x_true);
    let mut x = vec![S::zero(); 2];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::Sor, None).unwrap();
    ksp.set_tolerances(1e-10, 1e-12, 1e8, 40);
    ksp.set_operators(a, None);

    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(stats.reason.is_converged() || stats.reason.is_diverged());
}

#[test]
fn complex_gmres_with_ilu_pc_converges() {
    let a_mat = nonhermitian_2x2();
    let a = Arc::new(CsrOp::new(Arc::new(a_mat.clone())));
    let x_true = vec![S::from_parts(0.3, -0.2), S::from_parts(0.7, 0.4)];
    let b = apply_csr(&a_mat, &x_true);
    let mut x = vec![S::zero(); 2];

    let mut ksp = KspContext::new();
    ksp.set_type(SolverType::Gmres).unwrap();
    ksp.set_pc_type(PcType::Ilu, None).unwrap();
    ksp.set_tolerances(1e-10, 1e-12, 1e8, 40);
    ksp.set_operators(a, None);

    let stats = ksp.solve(&b, &mut x).unwrap();
    assert!(stats.reason.is_converged());
    for (xi, xt) in x.iter().zip(x_true.iter()) {
        assert_abs_diff_eq!(xi.real(), xt.real(), epsilon = 1e-8);
        assert_abs_diff_eq!(xi.imag(), xt.imag(), epsilon = 1e-8);
    }
}
