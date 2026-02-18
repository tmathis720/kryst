#![cfg(feature = "complex")]

use crate::algebra::prelude::*;
use crate::matrix::sparse::CsrMatrix;
use crate::preconditioner::Preconditioner;
use crate::preconditioner::ilu_csr::{
    IluComplexKernelMode, IluCsr, IluCsrConfig, IluKind, ReorderingOptions,
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
    let mut cfg = IluCsrConfig {
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
