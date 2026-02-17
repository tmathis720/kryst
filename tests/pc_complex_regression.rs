#![cfg(all(feature = "complex", feature = "backend-faer"))]

use kryst::algebra::prelude::*;
use kryst::config::options::PcOptions;
use kryst::context::pc_context::PcType;
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::PcSide;
use kryst::preconditioner::legacy::Preconditioner;
use kryst::preconditioner::sor::{MatSorType, SorPc};
use kryst::utils::diagnostics::PcDiagnostics;

#[test]
fn pc_diagnostics_reports_complex_capability_mode() {
    let sor = PcDiagnostics::from_options(Some(PcType::Sor), Some(&PcOptions::default()));
    let ilutp = PcDiagnostics::from_options(Some(PcType::Ilutp), Some(&PcOptions::default()));
    let ilu0 = PcDiagnostics::from_options(Some(PcType::Ilu0), Some(&PcOptions::default()));

    assert_eq!(sor.complex_support, "native_complex");
    assert_eq!(ilu0.complex_support, "native_complex");
    assert_eq!(ilutp.complex_support, "projected_complex");
}

#[test]
fn sor_native_complex_preserves_imaginary_coupling() {
    let a = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 1, 3],
        vec![0, 0, 1],
        vec![
            S::from_parts(2.0, 0.0),
            S::from_parts(1.0, 2.0),
            S::from_parts(3.0, 0.0),
        ],
    );
    let mut pc = SorPc::new(1.0, 1, MatSorType::APPLY_LOWER, 0.0);
    pc.setup(&a).expect("setup");

    let rhs = vec![S::from_real(1.0), S::from_real(1.0)];
    let mut out = vec![S::zero(); 2];
    pc.apply(PcSide::Left, &rhs, &mut out).expect("apply");

    assert!(
        out[1].imag().abs() > 1e-12,
        "expected imaginary coupling, got {out:?}"
    );
}

#[test]
fn ilu_csr_native_complex_beats_split_baseline_on_coupled_system() {
    use kryst::preconditioner::Preconditioner as _;
    use kryst::preconditioner::ilu_csr::{IluCsr, IluCsrConfig, IluKind};

    let a = CsrMatrix::from_csr(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            S::from_parts(4.0, 0.0),
            S::from_parts(0.0, 1.0),
            S::from_parts(0.0, 1.0),
            S::from_parts(3.0, 0.0),
        ],
    );
    let mut pc = IluCsr::new_with_config(IluCsrConfig {
        kind: IluKind::Ilu0,
        ..IluCsrConfig::default()
    });
    pc.setup(&a).expect("setup complex ilu_csr");

    let rhs = vec![S::from_parts(1.0, -1.5), S::from_parts(-0.75, 2.0)];
    let mut native = vec![S::zero(); 2];
    pc.apply(PcSide::Left, &rhs, &mut native)
        .expect("native apply");

    // degraded baseline: split solve on real and imaginary parts with same real factors
    let mut real_part = vec![0.0; 2];
    let mut imag_part = vec![0.0; 2];
    let xr = vec![rhs[0].real(), rhs[1].real()];
    let xi = vec![rhs[0].imag(), rhs[1].imag()];
    pc.apply_op(kryst::preconditioner::Op::NoTrans, &xr, &mut real_part)
        .expect("split real");
    pc.apply_op(kryst::preconditioner::Op::NoTrans, &xi, &mut imag_part)
        .expect("split imag");
    let split = vec![
        S::from_parts(real_part[0], imag_part[0]),
        S::from_parts(real_part[1], imag_part[1]),
    ];

    let residual_norm = |z: &[S]| {
        let mut az = vec![S::zero(); 2];
        kryst::core::traits::MatVec::matvec(&a, z, &mut az);
        az.iter()
            .zip(rhs.iter())
            .map(|(l, r)| (*l - *r).abs())
            .fold(0.0f64, f64::max)
    };

    let rn = residual_norm(&native);
    let rs = residual_norm(&split);
    assert!(
        rn < rs * 0.2,
        "native residual {rn} not better than split {rs}"
    );
}
