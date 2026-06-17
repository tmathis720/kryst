#![cfg(feature = "complex")]

use std::sync::Arc;

use kryst::algebra::prelude::*;
use kryst::matrix::op::CsrOp;
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::amg::{AMG, AMGBuilder, AmgTransferOperators, CoarseSolve, RelaxPhase};
use kryst::preconditioner::{PcSide, Preconditioner};

#[test]
fn amg_complex_setup_apply_small() {
    let csr = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(1.0)]);
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "native_diagonal");
    assert_eq!(amg.complex_setup_fallback_reason(), None);
    let stats = amg.stats().expect("native diagonal stats");
    assert_eq!(stats.num_levels, 1);
    assert_eq!(stats.levels[0].nnz_a, 1);
    assert_eq!(stats.complex_setup_mode.as_str(), "native_diagonal");

    let rhs = vec![S::from_parts(1.0, -2.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    assert!(out[0].real().is_finite());
    assert!(out[0].imag().is_finite());
}

#[test]
fn amg_complex_apply_without_setup_rejects_missing_native_state() {
    let amg = AMG::default();
    let rhs = vec![S::from_parts(1.0, -1.0)];
    let mut out = vec![S::zero(); rhs.len()];

    let err = amg
        .apply(PcSide::Left, &rhs, &mut out)
        .expect_err("complex AMG apply should require native setup state");
    assert!(
        err.to_string().contains("native complex setup state"),
        "unexpected error: {err}"
    );
}

#[test]
fn amg_complex_diagonal_uses_complex_inverse() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_parts(2.0, 1.0), S::from_parts(-1.0, 3.0)],
    );
    let op = CsrOp::new(Arc::new(csr.clone()));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "native_diagonal");
    assert_eq!(amg.complex_setup_fallback_reason(), None);
    let stats = amg.stats().expect("native diagonal stats");
    assert_eq!(stats.num_levels, 1);
    assert_eq!(stats.levels[0].eff_nnz_a, Some(csr.nnz()));
    assert!(stats.levels[0].max_row_sum_a > 3.0);

    let rhs = vec![S::from_parts(3.0, -4.0), S::from_parts(2.0, 5.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let expected = vec![S::from_parts(0.4, -2.2), S::from_parts(1.3, -1.1)];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1e-12, "got={got:?}, expected={exp:?}");
    }
}

#[test]
fn amg_complex_rejects_coarse_ilu_until_native_hierarchy_support() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.25),
            S::from_parts(-1.0, -0.25),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .coarse_solve(CoarseSolve::ILU)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    let err = amg
        .setup(&op)
        .expect_err("complex coarse ILU should be rejected");
    assert!(
        err.to_string().contains("coarse_solve=ILU"),
        "unexpected error: {err}"
    );
}

#[test]
fn amg_complex_native_hierarchy_supports_coarse_smoother() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.25),
            S::from_parts(-1.0, -0.25),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let rhs = vec![S::from_parts(1.0, -0.5), S::from_parts(-0.25, 0.75)];
    let mut amg = AMGBuilder::new()
        .coarse_solve(CoarseSolve::Smoother)
        .max_coarse_size(1)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .jacobi_omega(1.0)
        .num_grid_sweeps(RelaxPhase::Coarsest, 1)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    amg.setup(&op).expect("complex coarse smoother setup");
    assert_eq!(amg.complex_setup_mode_label(), "native_hierarchy");
    let stats = amg.stats().expect("coarse smoother stats");
    assert_eq!(
        stats
            .levels
            .last()
            .and_then(|level| level.coarse_solver.as_deref()),
        Some("Smoother")
    );

    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out)
        .expect("complex coarse smoother apply");
    let expected = vec![rhs[0] / S::from_real(2.0), rhs[1] / S::from_real(2.0)];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1e-12, "got={got:?}, expected={exp:?}");
    }
}

#[test]
fn amg_complex_native_hierarchy_required_uses_native_hierarchy_for_imaginary_coupling() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.25),
            S::from_parts(-1.0, -0.25),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .max_coarse_size(1)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    amg.setup(&op)
        .expect("native complex hierarchy should handle imaginary coupling");
    assert_eq!(amg.complex_setup_mode_label(), "native_hierarchy");
    assert_eq!(amg.complex_setup_fallback_reason(), None);
}

#[test]
fn amg_complex_native_coarse_solve_preserves_imaginary_coupling() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(3.0),
            S::from_parts(-1.0, 0.5),
            S::from_parts(-1.0, -0.5),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr.clone()));
    let mut amg = AMGBuilder::new()
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    amg.setup(&op)
        .expect("small complex operator should use native coarse algebra");
    assert_eq!(amg.complex_setup_mode_label(), "native_coarse");
    assert_eq!(amg.complex_setup_fallback_reason(), None);
    let stats = amg.stats().expect("native coarse stats");
    assert_eq!(stats.num_levels, 1);
    assert_eq!(stats.levels[0].nnz_a, csr.nnz());
    assert_eq!(stats.complex_setup_mode.as_str(), "native_coarse");

    let rhs = vec![S::from_parts(1.0, -0.5), S::from_parts(-0.25, 0.75)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let mut ax = vec![S::zero(); rhs.len()];
    kryst::core::traits::MatVec::matvec(&csr, &out, &mut ax);
    for (got, expected) in ax.iter().zip(rhs.iter()) {
        assert!(
            (*got - *expected).abs() < 1e-12,
            "got={got:?}, expected={expected:?}"
        );
    }
}

#[test]
fn amg_complex_native_coarse_respects_spd_side_restriction() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.25),
            S::from_parts(-1.0, -0.25),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();

    let rhs = vec![S::one(); 2];
    let mut out = vec![S::zero(); 2];
    let err = amg
        .apply(PcSide::Right, &rhs, &mut out)
        .expect_err("SPD AMG should reject right preconditioning");
    assert!(err.to_string().contains("only Left"));
}

#[test]
fn amg_complex_native_coarse_numeric_update_refactors_values() {
    let pattern_row_ptr = vec![0, 2, 4];
    let pattern_col_idx = vec![0, 1, 0, 1];
    let csr1 = CsrMatrix::from_csr(
        2,
        2,
        pattern_row_ptr.clone(),
        pattern_col_idx.clone(),
        vec![
            S::from_real(3.0),
            S::from_parts(-1.0, 0.5),
            S::from_parts(-1.0, -0.5),
            S::from_real(2.0),
        ],
    );
    let csr2 = CsrMatrix::from_csr(
        2,
        2,
        pattern_row_ptr,
        pattern_col_idx,
        vec![
            S::from_real(4.0),
            S::from_parts(-0.5, 0.25),
            S::from_parts(-0.5, -0.25),
            S::from_real(3.0),
        ],
    );
    let op1 = CsrOp::new(Arc::new(csr1));
    let op2 = CsrOp::new(Arc::new(csr2.clone()));
    let mut amg = AMG::default();
    amg.setup(&op1).unwrap();
    assert!(amg.supports_numeric_update());

    amg.update_numeric(&op2).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "native_coarse");

    let rhs = vec![S::from_parts(1.0, -0.5), S::from_parts(-0.25, 0.75)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();
    let mut ax = vec![S::zero(); rhs.len()];
    kryst::core::traits::MatVec::matvec(&csr2, &out, &mut ax);
    for (got, expected) in ax.iter().zip(rhs.iter()) {
        assert!((*got - *expected).abs() < 1e-12);
    }
}

#[test]
fn amg_complex_projected_hierarchy_numeric_update_refreshes_values() {
    let row_ptr = vec![0, 3, 6, 9];
    let col_idx = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let csr1 = CsrMatrix::from_csr(
        3,
        3,
        row_ptr.clone(),
        col_idx.clone(),
        vec![
            S::from_real(4.0),
            S::from_parts(-1.0, 0.2),
            S::from_parts(0.0, -0.1),
            S::from_parts(-1.0, -0.2),
            S::from_real(4.2),
            S::from_parts(-1.1, 0.1),
            S::from_parts(0.0, 0.1),
            S::from_parts(-1.1, -0.1),
            S::from_real(3.8),
        ],
    );
    let csr2 = CsrMatrix::from_csr(
        3,
        3,
        row_ptr,
        col_idx,
        vec![
            S::from_real(6.0),
            S::from_parts(-0.5, 0.2),
            S::from_parts(0.0, -0.1),
            S::from_parts(-0.5, -0.2),
            S::from_real(5.5),
            S::from_parts(-0.6, 0.1),
            S::from_parts(0.0, 0.1),
            S::from_parts(-0.6, -0.1),
            S::from_real(5.0),
        ],
    );
    let op1 = CsrOp::new(Arc::new(csr1));
    let op2 = CsrOp::new(Arc::new(csr2));
    let mut amg = AMGBuilder::new()
        .max_coarse_size(1)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .unwrap();
    amg.setup(&op1).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "native_hierarchy");

    let rhs = vec![
        S::from_parts(1.0, -0.3),
        S::from_parts(-0.5, 0.7),
        S::from_parts(0.25, 0.4),
    ];
    let mut before = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut before).unwrap();

    amg.update_numeric(&op2).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "native_hierarchy");
    let mut after = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut after).unwrap();

    assert!(
        before
            .iter()
            .zip(after.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-8),
        "projected hierarchy output did not change after numeric refresh"
    );
}

#[test]
fn amg_complex_native_hierarchy_honors_direct_dense_coarse_solver() {
    let csr = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_parts(-1.0, 0.2),
            S::from_parts(0.0, -0.1),
            S::from_parts(-1.0, -0.2),
            S::from_real(4.2),
            S::from_parts(-1.1, 0.1),
            S::from_parts(0.0, 0.1),
            S::from_parts(-1.1, -0.1),
            S::from_real(3.8),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .max_coarse_size(1)
        .coarse_solve(CoarseSolve::DirectDense)
        .num_grid_sweeps(RelaxPhase::Coarsest, 0)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    amg.setup(&op).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "native_hierarchy");
    let stats = amg.stats().expect("stats");
    assert_eq!(
        stats
            .levels
            .last()
            .and_then(|level| level.coarse_solver.as_deref()),
        Some("DirectDense")
    );

    let rhs = vec![
        S::from_parts(1.0, -0.3),
        S::from_parts(-0.5, 0.7),
        S::from_parts(0.25, 0.4),
    ];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();
    assert!(out.iter().all(|v| v.abs().is_finite()));
}

#[test]
fn amg_complex_numeric_update_rejects_pattern_change() {
    let csr1 = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
        ],
    );
    let csr2 = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_real(3.0), S::from_real(4.0)],
    );
    let op1 = CsrOp::new(Arc::new(csr1));
    let op2 = CsrOp::new(Arc::new(csr2));
    let mut amg = AMG::default();
    amg.setup(&op1).unwrap();

    let err = amg
        .update_numeric(&op2)
        .expect_err("numeric update must reject a changed sparsity pattern");
    assert!(err.to_string().contains("unchanged sparsity"));
}

#[test]
fn amg_complex_native_hierarchy_required_allows_real_valued_complex_matrix() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_real(-1.0),
            S::from_real(-1.0),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    amg.setup(&op)
        .expect("real-valued complex operator can use real hierarchy");
}

#[test]
fn amg_complex_native_hierarchy_required_allows_complex_diagonal_fast_path() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![S::from_parts(2.0, 1.0), S::from_parts(-1.0, 3.0)],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    amg.setup(&op)
        .expect("complex diagonal fast path is native scalar algebra");

    let rhs = vec![S::from_parts(3.0, -4.0), S::from_parts(2.0, 5.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let expected = vec![S::from_parts(0.4, -2.2), S::from_parts(1.3, -1.1)];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((*got - *exp).abs() < 1e-12, "got={got:?}, expected={exp:?}");
    }
}

#[test]
fn amg_complex_transfer_override_plumbing() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.5),
            S::from_parts(-1.0, -0.5),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMG::default();

    let p = CsrMatrix::from_csr(
        2,
        1,
        vec![0, 1, 2],
        vec![0, 0],
        vec![S::from_real(1.0), S::from_real(1.0)],
    );
    amg.set_level_transfer_operators(0, AmgTransferOperators::from_prolongation_adjoint(p));

    amg.setup(&op).unwrap();
    let rhs = vec![S::from_parts(1.0, 0.5), S::from_parts(-0.5, 1.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    assert!(
        out.iter()
            .all(|v| v.real().is_finite() && v.imag().is_finite())
    );
}

#[test]
fn amg_complex_native_hierarchy_accepts_imaginary_transfer_override() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.5),
            S::from_parts(-1.0, -0.5),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMG::default();

    let p = CsrMatrix::from_csr(
        2,
        1,
        vec![0, 1, 2],
        vec![0, 0],
        vec![S::from_parts(1.0, 0.2), S::from_real(1.0)],
    );
    let transfer = AmgTransferOperators::from_prolongation_adjoint(p);
    assert_eq!(transfer.restriction.values()[0], S::from_parts(1.0, -0.2));
    assert_eq!(transfer.restriction.values()[1], S::from_real(1.0));
    amg.set_level_transfer_operators(0, transfer);

    amg.setup(&op)
        .expect("native complex hierarchy should preserve imaginary transfer values");
    assert_eq!(amg.complex_setup_mode_label(), "native_hierarchy");

    let rhs = vec![S::from_parts(1.0, -0.25), S::from_parts(0.5, 0.75)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();
    assert!(
        out.iter()
            .all(|v| v.real().is_finite() && v.imag().is_finite())
    );
}

#[test]
fn amg_complex_native_hierarchy_uses_hermitian_restriction() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_real(0.5),
            S::from_real(0.5),
            S::from_real(3.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let mut amg = AMGBuilder::new()
        .max_coarse_size(1)
        .max_levels(2)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .num_grid_sweeps(RelaxPhase::Coarsest, 0)
        .coarse_solve(CoarseSolve::DirectDense)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");

    let p0 = S::from_parts(1.0, 1.0);
    let p1 = S::from_real(1.0);
    let p = CsrMatrix::from_csr(2, 1, vec![0, 1, 2], vec![0, 0], vec![p0, p1]);
    let deliberately_wrong_r = CsrMatrix::from_csr(
        1,
        2,
        vec![0, 2],
        vec![0, 1],
        vec![S::from_real(9.0), S::from_parts(-4.0, 2.0)],
    );
    amg.set_level_transfer_operators(
        0,
        AmgTransferOperators {
            prolongation: p,
            restriction: deliberately_wrong_r,
        },
    );

    amg.setup(&op)
        .expect("native complex hierarchy should build with complex transfer");
    assert_eq!(amg.complex_setup_mode_label(), "native_hierarchy");

    let rhs = vec![S::from_parts(1.0, 2.0), S::from_parts(3.0, -1.0)];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let omega = 2.0 / 3.0;
    let a0 = S::from_real(2.0);
    let a1 = S::from_real(3.0);
    let offdiag = S::from_real(0.5);
    let mut expected = vec![omega * rhs[0] / a0, omega * rhs[1] / a1];
    let residual = vec![
        rhs[0] - (a0 * expected[0] + offdiag * expected[1]),
        rhs[1] - (offdiag * expected[0] + a1 * expected[1]),
    ];
    let coarse_rhs = p0.conj() * residual[0] + p1.conj() * residual[1];
    let coarse_a = p0.conj() * (a0 * p0 + offdiag * p1) + p1.conj() * (offdiag * p0 + a1 * p1);
    let coarse_sol = coarse_rhs / coarse_a;
    expected[0] = expected[0] + p0 * coarse_sol;
    expected[1] = expected[1] + p1 * coarse_sol;
    let post_residual = vec![
        rhs[0] - (a0 * expected[0] + offdiag * expected[1]),
        rhs[1] - (offdiag * expected[0] + a1 * expected[1]),
    ];
    expected[0] = expected[0] + omega * post_residual[0] / a0;
    expected[1] = expected[1] + omega * post_residual[1] / a1;
    for (got, expected) in out.iter().zip(expected.iter()) {
        assert!(
            (*got - *expected).abs() < 1e-12,
            "got={got:?}, expected={expected:?}"
        );
    }
}

#[test]
fn amg_complex_native_hierarchy_honors_l1_jacobi_smoother() {
    let csr = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_parts(-1.5, 0.25),
            S::from_parts(0.5, -0.5),
            S::from_parts(-0.75, -0.2),
            S::from_real(3.5),
            S::from_parts(-1.25, 0.4),
            S::from_parts(0.25, 0.5),
            S::from_parts(-1.0, -0.3),
            S::from_real(2.75),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let rhs = vec![
        S::from_parts(1.0, -0.3),
        S::from_parts(-0.5, 0.7),
        S::from_parts(0.25, 0.4),
    ];

    let mut jacobi = AMGBuilder::new()
        .max_coarse_size(1)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("jacobi amg build");
    jacobi.setup(&op).expect("jacobi native hierarchy setup");
    let mut jacobi_out = vec![S::zero(); rhs.len()];
    jacobi
        .apply(PcSide::Left, &rhs, &mut jacobi_out)
        .expect("jacobi native hierarchy apply");

    let mut l1 = AMGBuilder::new()
        .max_coarse_size(1)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::L1Jacobi)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("l1 amg build");
    l1.setup(&op).expect("l1 native hierarchy setup");
    assert_eq!(l1.complex_setup_mode_label(), "native_hierarchy");
    let stats = l1.stats().expect("l1 stats");
    assert_eq!(stats.levels[0].selected_relax_pre, "L1Jacobi");
    assert_eq!(stats.levels[0].selected_relax_post, "L1Jacobi");

    let mut l1_out = vec![S::zero(); rhs.len()];
    l1.apply(PcSide::Left, &rhs, &mut l1_out)
        .expect("l1 native hierarchy apply");

    assert!(
        jacobi_out
            .iter()
            .zip(l1_out.iter())
            .any(|(&jacobi, &l1)| (jacobi - l1).abs() > 1e-10),
        "native hierarchy ignored L1-Jacobi smoother"
    );
}

#[test]
fn amg_complex_native_hierarchy_honors_chebyshev_smoother() {
    let csr = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_parts(-1.5, 0.25),
            S::from_parts(0.5, -0.5),
            S::from_parts(-0.75, -0.2),
            S::from_real(3.5),
            S::from_parts(-1.25, 0.4),
            S::from_parts(0.25, 0.5),
            S::from_parts(-1.0, -0.3),
            S::from_real(2.75),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let rhs = vec![
        S::from_parts(0.5, -0.75),
        S::from_parts(-1.25, 0.25),
        S::from_parts(0.75, 0.5),
    ];

    let mut jacobi = AMGBuilder::new()
        .max_coarse_size(1)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("jacobi amg build");
    jacobi.setup(&op).expect("jacobi native hierarchy setup");
    let mut jacobi_out = vec![S::zero(); rhs.len()];
    jacobi
        .apply(PcSide::Left, &rhs, &mut jacobi_out)
        .expect("jacobi native hierarchy apply");

    let mut cheb = AMGBuilder::new()
        .max_coarse_size(1)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Chebyshev)
        .chebyshev_degree(2)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("chebyshev amg build");
    cheb.setup(&op).expect("chebyshev native hierarchy setup");
    assert_eq!(cheb.complex_setup_mode_label(), "native_hierarchy");
    let stats = cheb.stats().expect("chebyshev stats");
    assert_eq!(stats.levels[0].selected_relax_pre, "Chebyshev");
    assert_eq!(stats.levels[0].selected_relax_post, "Chebyshev");

    let mut cheb_out = vec![S::zero(); rhs.len()];
    cheb.apply(PcSide::Left, &rhs, &mut cheb_out)
        .expect("chebyshev native hierarchy apply");

    assert!(
        jacobi_out
            .iter()
            .zip(cheb_out.iter())
            .any(|(&jacobi, &cheb)| (jacobi - cheb).abs() > 1e-10),
        "native hierarchy ignored Chebyshev smoother"
    );
}

#[test]
fn amg_complex_native_hierarchy_honors_gauss_seidel_smoother() {
    let csr = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            S::from_real(4.0),
            S::from_parts(-1.5, 0.25),
            S::from_parts(0.5, -0.5),
            S::from_parts(-0.75, -0.2),
            S::from_real(3.5),
            S::from_parts(-1.25, 0.4),
            S::from_parts(0.25, 0.5),
            S::from_parts(-1.0, -0.3),
            S::from_real(2.75),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let rhs = vec![
        S::from_parts(0.5, -0.75),
        S::from_parts(-1.25, 0.25),
        S::from_parts(0.75, 0.5),
    ];

    let mut jacobi = AMGBuilder::new()
        .max_coarse_size(1)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::Jacobi)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("jacobi amg build");
    jacobi.setup(&op).expect("jacobi native hierarchy setup");
    let mut jacobi_out = vec![S::zero(); rhs.len()];
    jacobi
        .apply(PcSide::Left, &rhs, &mut jacobi_out)
        .expect("jacobi native hierarchy apply");

    let mut gs = AMGBuilder::new()
        .max_coarse_size(1)
        .grid_relax_type_all(kryst::preconditioner::amg::RelaxType::GaussSeidel)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("gs amg build");
    gs.setup(&op).expect("gs native hierarchy setup");
    assert_eq!(gs.complex_setup_mode_label(), "native_hierarchy");
    let stats = gs.stats().expect("gs stats");
    assert_eq!(stats.levels[0].selected_relax_pre, "GaussSeidel");
    assert_eq!(stats.levels[0].selected_relax_post, "GaussSeidel");

    let mut gs_out = vec![S::zero(); rhs.len()];
    gs.apply(PcSide::Left, &rhs, &mut gs_out)
        .expect("gs native hierarchy apply");

    assert!(
        jacobi_out
            .iter()
            .zip(gs_out.iter())
            .any(|(&jacobi, &gs)| (jacobi - gs).abs() > 1e-10),
        "native hierarchy ignored Gauss-Seidel smoother"
    );
}

#[test]
fn amg_complex_transfer_override_rejects_zero_coarse_columns() {
    let csr = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_real(2.0),
            S::from_parts(-1.0, 0.25),
            S::from_parts(-1.0, -0.25),
            S::from_real(2.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr));
    let p = CsrMatrix::from_csr(2, 0, vec![0, 0, 0], vec![], vec![]);
    let r = CsrMatrix::from_csr(0, 2, vec![0], vec![], vec![]);
    let mut amg = AMGBuilder::new()
        .max_coarse_size(1)
        .require_native_complex_hierarchy(true)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");
    amg.set_level_transfer_operators(
        0,
        AmgTransferOperators {
            prolongation: p,
            restriction: r,
        },
    );

    let err = amg
        .setup(&op)
        .expect_err("zero-column transfer override should be rejected");
    assert!(
        err.to_string().contains("zero coarse columns"),
        "unexpected error: {err}"
    );
}

#[test]
fn amg_complex_apply_residual_acceptance() {
    let csr = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            S::from_parts(4.0, 0.0),
            S::from_parts(-1.0, 0.2),
            S::from_parts(0.0, -0.1),
            S::from_parts(-1.0, -0.2),
            S::from_parts(4.2, 0.0),
            S::from_parts(-1.1, 0.1),
            S::from_parts(0.0, 0.1),
            S::from_parts(-1.1, -0.1),
            S::from_parts(3.8, 0.0),
        ],
    );
    let op = CsrOp::new(Arc::new(csr.clone()));
    let mut amg = AMGBuilder::new()
        .max_coarse_size(1)
        .logging_level(1)
        .build(&faer::Mat::<f64>::zeros(0, 0))
        .expect("amg build");
    amg.setup(&op).unwrap();
    assert_eq!(amg.complex_setup_mode_label(), "native_hierarchy");
    assert_eq!(amg.complex_setup_fallback_reason(), None);
    assert_eq!(
        amg.stats().expect("stats").complex_setup_mode.as_str(),
        "native_hierarchy"
    );
    assert_eq!(
        amg.stats()
            .expect("stats")
            .complex_setup_fallback_reason
            .as_deref(),
        None
    );
    let stats = amg.stats().expect("stats");
    assert!(stats.levels[0].max_row_sum_a > 0.0);
    assert_eq!(stats.levels[0].eff_nnz_a, Some(csr.nnz()));
    assert_eq!(
        stats.selected_dist_coarse_route.as_deref(),
        Some("root_gather")
    );
    assert_eq!(
        stats.dist_route_fallback,
        vec![
            "root_gather".to_string(),
            "local_prototype".to_string(),
            "superlu_dist".to_string()
        ]
    );

    let rhs = vec![
        S::from_parts(1.0, -0.3),
        S::from_parts(-0.5, 0.7),
        S::from_parts(0.25, 0.4),
    ];
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let mut a_out = vec![S::zero(); rhs.len()];
    kryst::core::traits::MatVec::matvec(&csr, &out, &mut a_out);
    let r_inf = a_out
        .iter()
        .zip(rhs.iter())
        .map(|(l, r)| (*l - *r).abs())
        .fold(0.0f64, f64::max);
    assert!(
        r_inf.is_finite() && r_inf < 2.5,
        "residual too large: {r_inf}"
    );
}

#[test]
fn amg_complex_residual_reason_code_guard() {
    let csr = CsrMatrix::from_csr(1, 1, vec![0, 1], vec![0], vec![S::from_real(2.0)]);
    let op = CsrOp::new(Arc::new(csr.clone()));
    let mut amg = AMG::default();
    amg.setup(&op).unwrap();

    let rhs = vec![S::from_parts(1.0, 1.0)];
    let mut out = vec![S::zero(); 1];
    amg.apply(PcSide::Left, &rhs, &mut out).unwrap();

    let mut ax = vec![S::zero(); 1];
    kryst::core::traits::MatVec::matvec(&csr, &out, &mut ax);
    let r = (ax[0] - rhs[0]).abs();
    assert!(r < 1e-10, "unexpected residual guard trip: {r}");
}
