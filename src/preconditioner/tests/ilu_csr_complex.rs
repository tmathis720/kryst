#![cfg(feature = "complex")]

use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::ilu_csr::{
    IluComplexKernelMode, IluCsr, IluCsrConfig, IluKind, IlutParams, ReorderingKind,
    ReorderingOptions,
};
use crate::utils::conditioning::ConditioningOptions;

fn tridiag_csr_complex(n: usize, a: S, b: S, c: S) -> CsrMatrix<S> {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(3 * n);
    let mut vals = Vec::with_capacity(3 * n);
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            vals.push(a);
        }
        col_idx.push(i);
        vals.push(b);
        if i + 1 < n {
            col_idx.push(i + 1);
            vals.push(c);
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

fn residual_norm(a: &CsrMatrix<S>, x: &[S], b: &[S]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.nrows() {
        let mut ax = S::zero();
        let (cols, vals) = a.row(i);
        for (&j, &v) in cols.iter().zip(vals.iter()) {
            ax += v * x[j];
        }
        let r = b[i] - ax;
        sum += r.abs() * r.abs();
    }
    sum.sqrt()
}

#[test]
fn iluk_complex_native_kernel_is_default() {
    let a = tridiag_csr_complex(
        8,
        S::from_parts(-1.0, 0.1),
        S::from_parts(4.0, 0.2),
        S::from_parts(-1.0, -0.1),
    );
    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: 1 },
        reordering: ReorderingOptions::default(),
        conditioning: ConditioningOptions::default(),
        ..IluCsrConfig::default()
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).unwrap();
    assert_eq!(pc.complex_kernel_mode(), IluComplexKernelMode::Native);

    let rhs = vec![S::from_parts(1.0, -0.2); 8];
    let mut y = vec![S::zero(); 8];
    pc.apply(crate::preconditioner::PcSide::Left, &rhs, &mut y)
        .unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn iluk_complex_can_force_degraded_fallback() {
    let a = tridiag_csr_complex(
        8,
        S::from_parts(-1.0, 0.3),
        S::from_parts(4.0, -0.1),
        S::from_parts(-1.0, 0.05),
    );
    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: 1 },
        reordering: ReorderingOptions::default(),
        conditioning: ConditioningOptions::default(),
        ..IluCsrConfig::default()
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.set_complex_force_degraded(true);
    pc.setup(&a).unwrap();
    assert_eq!(
        pc.complex_kernel_mode(),
        IluComplexKernelMode::DegradedRealProjection
    );
}

#[test]
fn ilut_complex_uses_native_kernel_and_respects_numeric_update() {
    let a = tridiag_csr_complex(
        10,
        S::from_parts(-1.0, 0.25),
        S::from_parts(3.5, -0.1),
        S::from_parts(-0.8, -0.2),
    );
    let cfg = IluCsrConfig {
        kind: IluKind::Ilut {
            params: IlutParams {
                droptol_abs: 1e-7,
                droptol_rel: 1e-2,
                p_l: 3,
                p_u: 3,
                ..IlutParams::default()
            },
        },
        reordering: ReorderingOptions::default(),
        conditioning: ConditioningOptions::default(),
        ..IluCsrConfig::default()
    };
    let mut pc = IluCsr::new_with_config(cfg);
    pc.setup(&a).unwrap();
    assert_eq!(pc.complex_kernel_mode(), IluComplexKernelMode::Native);

    let rhs = vec![S::from_parts(1.0, 0.5); 10];
    let mut y = vec![S::zero(); 10];
    pc.apply(crate::preconditioner::PcSide::Left, &rhs, &mut y)
        .unwrap();
    assert!(y.iter().all(|v| v.is_finite()));

    let mut a2 = a.clone();
    let vals = a2.values_mut();
    for (k, v) in vals.iter_mut().enumerate() {
        *v += S::from_parts(0.0, 1e-3 * (k as f64 + 1.0));
    }
    pc.update_numeric(&a2).unwrap();
    let mut y2 = vec![S::zero(); 10];
    pc.apply(crate::preconditioner::PcSide::Left, &rhs, &mut y2)
        .unwrap();
    assert!(y2.iter().all(|v| v.is_finite()));
}

#[test]
fn ilut_complex_native_beats_degraded_residual() {
    let n = 16;
    let a = tridiag_csr_complex(
        n,
        S::from_parts(-1.0, 0.6),
        S::from_parts(4.2, -0.4),
        S::from_parts(-0.7, 0.5),
    );
    let rhs: Vec<S> = (0..n)
        .map(|i| S::from_parts((i + 1) as f64 / n as f64, -0.2))
        .collect();

    let cfg = IluCsrConfig {
        kind: IluKind::Ilut {
            params: IlutParams {
                droptol_abs: 1e-8,
                droptol_rel: 1e-2,
                p_l: 4,
                p_u: 4,
                ..IlutParams::default()
            },
        },
        reordering: ReorderingOptions::default(),
        conditioning: ConditioningOptions::default(),
        ..IluCsrConfig::default()
    };

    let mut native = IluCsr::new_with_config(cfg.clone());
    native.setup(&a).unwrap();
    assert_eq!(native.complex_kernel_mode(), IluComplexKernelMode::Native);

    let mut degraded = IluCsr::new_with_config(cfg);
    degraded.set_complex_force_degraded(true);
    degraded.setup(&a).unwrap();
    assert_eq!(
        degraded.complex_kernel_mode(),
        IluComplexKernelMode::DegradedRealProjection
    );

    let mut y_native = vec![S::zero(); n];
    let mut y_degraded = vec![S::zero(); n];
    native
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_native)
        .unwrap();
    degraded
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_degraded)
        .unwrap();

    let rn = residual_norm(&a, &y_native, &rhs);
    let rd = residual_norm(&a, &y_degraded, &rhs);
    assert!(
        rn <= rd * 1.05,
        "native residual {rn} should be <= degraded {rd}"
    );
}

#[test]
fn iluk_complex_native_beats_degraded_residual() {
    let n = 18;
    let a = tridiag_csr_complex(
        n,
        S::from_parts(-1.1, 0.45),
        S::from_parts(4.0, -0.35),
        S::from_parts(-0.9, 0.4),
    );
    let rhs: Vec<S> = (0..n)
        .map(|i| S::from_parts(0.5 + i as f64 / n as f64, -0.15))
        .collect();

    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: 2 },
        reordering: ReorderingOptions::default(),
        conditioning: ConditioningOptions::default(),
        ..IluCsrConfig::default()
    };

    let mut native = IluCsr::new_with_config(cfg.clone());
    native.setup(&a).unwrap();
    let mut degraded = IluCsr::new_with_config(cfg);
    degraded.set_complex_force_degraded(true);
    degraded.setup(&a).unwrap();

    let mut y_native = vec![S::zero(); n];
    let mut y_degraded = vec![S::zero(); n];
    native
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_native)
        .unwrap();
    degraded
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_degraded)
        .unwrap();

    let rn = residual_norm(&a, &y_native, &rhs);
    let rd = residual_norm(&a, &y_degraded, &rhs);
    assert!(
        rn <= rd * 1.1,
        "ILU(k) native residual {rn} should be <= degraded residual {rd}"
    );
}

#[test]
fn complex_setup_uses_distinct_nonsymmetric_permutation_path() {
    let row_ptr = vec![0, 2, 4, 6, 8];
    let col_idx = vec![0, 1, 0, 2, 1, 3, 2, 3];
    let vals = vec![
        S::from_parts(1.0, 0.0),
        S::from_parts(9.0, 0.5),
        S::from_parts(8.0, -0.3),
        S::from_parts(1.2, 0.0),
        S::from_parts(0.8, 0.1),
        S::from_parts(7.5, -0.2),
        S::from_parts(6.0, 0.4),
        S::from_parts(1.1, 0.0),
    ];
    let a = CsrMatrix::from_csr(4, 4, row_ptr, col_idx, vals);
    let rhs = vec![
        S::from_parts(1.0, -0.2),
        S::from_parts(0.3, 0.6),
        S::from_parts(-0.5, 0.1),
        S::from_parts(0.9, -0.4),
    ];

    let mut cfg = IluCsrConfig {
        kind: IluKind::Ilu0,
        ..IluCsrConfig::default()
    };
    cfg.reordering = ReorderingOptions {
        kind: ReorderingKind::None,
        symmetric: true,
        deterministic: true,
    };

    let mut sym = IluCsr::new_with_config(cfg.clone());
    sym.setup(&a).unwrap();
    let mut y_sym = vec![S::zero(); 4];
    sym.apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_sym)
        .unwrap();

    cfg.reordering.symmetric = false;
    let mut nonsym = IluCsr::new_with_config(cfg);
    nonsym.setup(&a).unwrap();
    let mut y_nonsym = vec![S::zero(); 4];
    nonsym
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_nonsym)
        .unwrap();

    let diff: f64 = y_sym
        .iter()
        .zip(y_nonsym.iter())
        .map(|(a, b)| (*a - *b).abs())
        .sum();
    assert!(
        diff > 1e-8,
        "expected different outputs between symmetric and nonsymmetric permutation paths"
    );
}

#[test]
fn complex_nonsymmetric_numeric_update_matches_fresh_setup() {
    let row_ptr = vec![0, 2, 4, 6, 8];
    let col_idx = vec![0, 1, 0, 2, 1, 3, 2, 3];
    let vals = vec![
        S::from_parts(1.0, 0.0),
        S::from_parts(10.0, 0.0),
        S::from_parts(8.0, 0.0),
        S::from_parts(1.0, 0.0),
        S::from_parts(1.0, 0.0),
        S::from_parts(7.0, 0.0),
        S::from_parts(6.0, 0.0),
        S::from_parts(1.0, 0.0),
    ];
    let a = CsrMatrix::from_csr(4, 4, row_ptr.clone(), col_idx.clone(), vals);
    let rhs = vec![
        S::from_parts(1.0, 0.2),
        S::from_parts(-0.4, 0.1),
        S::from_parts(0.7, -0.3),
        S::from_parts(0.2, 0.5),
    ];

    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: 1 },
        reordering: ReorderingOptions {
            kind: ReorderingKind::None,
            symmetric: false,
            deterministic: true,
        },
        ..IluCsrConfig::default()
    };

    let mut updated = IluCsr::new_with_config(cfg.clone());
    updated.setup(&a).unwrap();

    let mut vals2 = a.values().to_vec();
    for (k, v) in vals2.iter_mut().enumerate() {
        *v += S::from_parts(0.05 * (k as f64 + 1.0), -0.01 * (k as f64));
    }
    let a2 = CsrMatrix::from_csr(4, 4, row_ptr, col_idx, vals2);
    updated.update_numeric(&a2).unwrap();
    let mut y_update = vec![S::zero(); 4];
    updated
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_update)
        .unwrap();

    let mut fresh = IluCsr::new_with_config(cfg);
    fresh.setup(&a2).unwrap();
    let mut y_fresh = vec![S::zero(); 4];
    fresh
        .apply(crate::preconditioner::PcSide::Left, &rhs, &mut y_fresh)
        .unwrap();

    for (i, (yu, yf)) in y_update.iter().zip(y_fresh.iter()).enumerate() {
        assert!(
            (*yu - *yf).abs() < 1e-8,
            "numeric update should match fresh setup at index {i}"
        );
    }
}

fn grid_csr_complex(nx: usize, ny: usize) -> CsrMatrix<S> {
    let n = nx * ny;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptr.push(0);
    for y in 0..ny {
        for x in 0..nx {
            let i = y * nx + x;
            if y > 0 {
                col_idx.push(i - nx);
                vals.push(S::from_parts(-0.7, 0.12));
            }
            if x > 0 {
                col_idx.push(i - 1);
                vals.push(S::from_parts(-0.9, -0.08));
            }
            col_idx.push(i);
            vals.push(S::from_parts(4.4 + 0.03 * i as f64, 0.35));
            if x + 1 < nx {
                col_idx.push(i + 1);
                vals.push(S::from_parts(-0.8, 0.07));
            }
            if y + 1 < ny {
                col_idx.push(i + nx);
                vals.push(S::from_parts(-0.6, -0.11));
            }
            row_ptr.push(col_idx.len());
        }
    }
    CsrMatrix::from_csr(n, n, row_ptr, col_idx, vals)
}

fn norm2(v: &[S]) -> f64 {
    v.iter().map(|z| z.abs() * z.abs()).sum::<f64>().sqrt()
}

fn assert_complex_ilu_reorder_apply_consistency_and_residual_reduction(kind: ReorderingKind) {
    use crate::preconditioner::PcSide;
    use crate::utils::permutation::{Permutation, amd_csr, permute_csr_symmetric, rcm_csr};

    let a = grid_csr_complex(4, 4);
    let n = a.nrows();
    let rhs: Vec<S> = (0..n)
        .map(|i| S::from_parts(1.0 + 0.05 * i as f64, -0.4 + 0.02 * i as f64))
        .collect();

    let cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: 1 },
        reordering: ReorderingOptions {
            kind,
            symmetric: true,
            deterministic: true,
        },
        conditioning: ConditioningOptions::default(),
        ..IluCsrConfig::default()
    };
    let mut reordered = IluCsr::new_with_config(cfg);
    reordered.setup(&a).unwrap();
    assert_eq!(
        reordered.complex_kernel_mode(),
        IluComplexKernelMode::Native
    );

    let mut actual = vec![S::zero(); n];
    reordered.apply(PcSide::Left, &rhs, &mut actual).unwrap();
    assert!(actual.iter().all(|v| v.is_finite()));

    let a_real = CsrMatrix::from_csr(
        a.nrows(),
        a.ncols(),
        a.row_ptr().to_vec(),
        a.col_idx().to_vec(),
        a.values().iter().map(|v| v.real()).collect(),
    );
    let perm = match kind {
        ReorderingKind::None => Permutation::identity(n),
        ReorderingKind::Rcm => rcm_csr(&a_real),
        ReorderingKind::Amd => amd_csr(&a_real),
    };
    let a_perm = permute_csr_symmetric(&a, &perm);
    let manual_cfg = IluCsrConfig {
        kind: IluKind::Iluk { k: 1 },
        reordering: ReorderingOptions::default(),
        conditioning: ConditioningOptions::default(),
        ..IluCsrConfig::default()
    };
    let mut manual = IluCsr::new_with_config(manual_cfg);
    manual.setup(&a_perm).unwrap();
    assert_eq!(manual.complex_kernel_mode(), IluComplexKernelMode::Native);

    let mut rhs_perm = vec![S::zero(); n];
    perm.apply_vec(&rhs, &mut rhs_perm);
    let mut y_perm = vec![S::zero(); n];
    manual.apply(PcSide::Left, &rhs_perm, &mut y_perm).unwrap();
    let mut expected = vec![S::zero(); n];
    perm.apply_vec_t(&y_perm, &mut expected);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*a - *e).abs() < 1e-9,
            "reordered apply should match explicit permutation pipeline for {kind:?} at {i}: actual={a:?}, expected={e:?}"
        );
    }

    let rhs_norm = norm2(&rhs);
    let reduced = residual_norm(&a, &actual, &rhs);
    assert!(
        reduced < 0.35 * rhs_norm,
        "{kind:?} residual {reduced:.6e} should reduce RHS norm {rhs_norm:.6e}"
    );
}

#[test]
fn ilu_csr_complex_apply_pipeline_none_is_consistent_and_reduces_residual() {
    assert_complex_ilu_reorder_apply_consistency_and_residual_reduction(ReorderingKind::None);
}

#[test]
fn ilu_csr_complex_apply_pipeline_rcm_is_consistent_and_reduces_residual() {
    assert_complex_ilu_reorder_apply_consistency_and_residual_reduction(ReorderingKind::Rcm);
}

#[test]
fn ilu_csr_complex_apply_pipeline_amd_is_consistent_and_reduces_residual() {
    assert_complex_ilu_reorder_apply_consistency_and_residual_reduction(ReorderingKind::Amd);
}
