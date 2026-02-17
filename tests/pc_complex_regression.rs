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

    assert_eq!(sor.complex_support, "native_complex");
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
