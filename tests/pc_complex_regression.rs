#![cfg(all(feature = "complex", feature = "backend-faer"))]

use kryst::algebra::prelude::*;
use kryst::config::options::PcOptions;
use kryst::context::pc_context::PcType;
use kryst::core::traits::MatVec;
use kryst::matrix::sparse::CsrMatrix;
use kryst::preconditioner::sor::{MatSorType, SorPc};
use kryst::preconditioner::{ApproxInvKind, ApproxInvParams, SpaiCsr};
use kryst::preconditioner::{PcSide, Preconditioner};
use kryst::utils::diagnostics::PcDiagnostics;

#[test]
fn pc_diagnostics_reports_complex_capability_mode() {
    let sor = PcDiagnostics::from_options(Some(PcType::Sor), Some(&PcOptions::default()));
    let ilutp = PcDiagnostics::from_options(Some(PcType::Ilutp), Some(&PcOptions::default()));
    let ilu0 = PcDiagnostics::from_options(Some(PcType::Ilu0), Some(&PcOptions::default()));
    let ainv =
        PcDiagnostics::from_options(Some(PcType::ApproxInverse), Some(&PcOptions::default()));

    assert_eq!(sor.complex_support, "native_complex");
    assert_eq!(ilu0.complex_support, "native_complex");
    assert_eq!(ilutp.complex_support, "projected_complex");
    assert_eq!(ainv.complex_support, "native_complex");
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
    let mut real_part = vec![S::zero(); 2];
    let mut imag_part = vec![S::zero(); 2];
    let xr = vec![S::from_real(rhs[0].real()), S::from_real(rhs[1].real())];
    let xi = vec![S::from_real(rhs[0].imag()), S::from_real(rhs[1].imag())];
    pc.apply_op(kryst::preconditioner::Op::NoTrans, &xr, &mut real_part)
        .expect("split real");
    pc.apply_op(kryst::preconditioner::Op::NoTrans, &xi, &mut imag_part)
        .expect("split imag");
    let split = vec![
        S::from_parts(real_part[0].real(), imag_part[0].real()),
        S::from_parts(real_part[1].real(), imag_part[1].real()),
    ];

    let residual_norm = |z: &Vec<S>| {
        let mut az = vec![S::zero(); 2];
        MatVec::matvec(&a, z, &mut az);
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

#[test]
fn approxinv_spai_complex_setup_apply_residual_acceptance() {
    let a = CsrMatrix::from_csr(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            S::from_parts(4.0, 0.2),
            S::from_parts(-1.0, 0.4),
            S::from_parts(0.5, -0.2),
            S::from_parts(-1.2, -0.3),
            S::from_parts(3.8, 0.1),
            S::from_parts(-0.7, 0.25),
            S::from_parts(0.4, 0.15),
            S::from_parts(-0.9, -0.35),
            S::from_parts(3.2, -0.1),
        ],
    );
    let rhs = vec![
        S::from_parts(1.0, -0.5),
        S::from_parts(-0.25, 0.75),
        S::from_parts(0.5, 0.25),
    ];

    let mut pc = SpaiCsr::new_with_params(ApproxInvParams {
        kind: ApproxInvKind::SPAI,
        levels: 1,
        max_per_col: 3,
        drop_tol: 1e-6,
        ..ApproxInvParams::default()
    });
    pc.setup(&a).expect("spai complex setup");
    assert!(pc.complex_setup_used_native());

    let mut out = vec![S::zero(); rhs.len()];
    pc.apply(PcSide::Left, &rhs, &mut out)
        .expect("spai complex apply");

    let mut az = vec![S::zero(); rhs.len()];
    MatVec::matvec(&a, &out, &mut az);
    let rn = az
        .iter()
        .zip(rhs.iter())
        .map(|(l, r)| (*l - *r).abs())
        .fold(0.0f64, f64::max);
    assert!(rn.is_finite() && rn < 3.0, "SPAI residual too large: {rn}");
}
