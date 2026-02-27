#![cfg(feature = "complex")]

use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::ilu_csr::{
    IluComplexKernelMode, IluCsr, IluCsrConfig, IluKind, IlutParams, ReorderingOptions,
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
