#![cfg(feature = "complex")]

use super::*;

fn poisson_1d_complex(n: usize) -> CsrMatrix<S> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            vals.push(S::from_parts(-1.0, 0.0));
        }
        col_idx.push(i);
        vals.push(S::from_parts(2.0, 0.05 * (i as f64 + 1.0) / n as f64));
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(S::from_parts(-1.0, 0.0));
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

#[test]
fn complex_amg_cycle_supports_ilu0_smoother() {
    let a = poisson_1d_complex(48);
    let mut amg = AMGBuilder::new()
        .grid_relax_type_all(RelaxType::Ilu0)
        .coarse_solve(CoarseSolve::DirectDense)
        .build(&Mat::<f64>::zeros(0, 0))
        .expect("amg build");
    amg.setup(&a).expect("complex AMG setup with ILU0");

    let rhs: Vec<S> = (0..a.nrows())
        .map(|i| S::from_parts((i as f64).sin(), (0.5 * i as f64).cos()))
        .collect();
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out)
        .expect("complex AMG apply with ILU0");

    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn complex_amg_cycle_supports_ras_smoother() {
    let a = poisson_1d_complex(48);
    let mut amg = AMGBuilder::new()
        .grid_relax_type_all(RelaxType::Ras)
        .coarse_solve(CoarseSolve::DirectDense)
        .build(&Mat::<f64>::zeros(0, 0))
        .expect("amg build");
    amg.setup(&a).expect("complex AMG setup with RAS");

    let rhs: Vec<S> = (0..a.nrows())
        .map(|i| S::from_parts((0.25 * i as f64).cos(), (i as f64).sin()))
        .collect();
    let mut out = vec![S::zero(); rhs.len()];
    amg.apply(PcSide::Left, &rhs, &mut out)
        .expect("complex AMG apply with RAS");

    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn symmetric_non_galerkin_requires_symmetric_smoother() {
    let mut cfg = AMGConfig::default();
    cfg.non_galerkin.enabled = true;
    cfg.non_galerkin.symmetry = NgSymmetry::Symmetric;
    cfg.grid_relax_type[RelaxPhase::Down.ix()] = RelaxType::GaussSeidel;
    cfg.grid_relax_type[RelaxPhase::Up.ix()] = RelaxType::GaussSeidel;

    let err = validate_relax_policy(&cfg, cfg.coarse_solve).unwrap_err();
    match err {
        KError::InvalidInput(msg) => assert!(msg.contains("symmetric smoother")),
        other => panic!("unexpected error: {other:?}"),
    }
}
