#![cfg(all(feature = "complex", feature = "complex_ilu"))]

use faer::Mat;
use kryst::algebra::prelude::*;
use kryst::preconditioner::ilu::{IluBuilder, IluType};
use kryst::preconditioner::legacy::Preconditioner;
use kryst::preconditioner::PcSide;

fn make_diagonal_matrix(diag: &[S]) -> Mat<S> {
    let n = diag.len();
    Mat::from_fn(n, n, |i, j| if i == j { diag[i] } else { S::zero() })
}

fn mat_vec_mul(a: &Mat<S>, x: &[S]) -> Vec<S> {
    let n = a.nrows();
    let mut out = vec![S::zero(); n];
    for i in 0..n {
        let mut sum = S::zero();
        for j in 0..a.ncols() {
            sum = sum + a[(i, j)] * x[j];
        }
        out[i] = sum;
    }
    out
}

#[test]
fn ilu0_complex_diagonal_matches_inverse() {
    let diag = [
        S::from_parts(3.0, -0.5),
        S::from_parts(2.0, 1.25),
        S::from_parts(4.5, -2.0),
        S::from_parts(5.0, 0.75),
    ];
    let matrix = make_diagonal_matrix(&diag);

    let mut ilu = IluBuilder::new()
        .ilu_type(IluType::ILU0)
        .build::<S>()
        .expect("complex ILU0 build");
    ilu.setup(&matrix).expect("complex ILU0 setup");
    assert!(ilu.complex_setup_used_native());
    assert!(ilu.complex_setup_fallback_reason().is_none());

    let rhs = [
        S::from_parts(1.0, -2.0),
        S::from_parts(-3.0, 0.5),
        S::from_parts(0.25, 1.5),
        S::from_parts(-0.75, -1.25),
    ];
    let mut z = vec![S::zero(); rhs.len()];
    ilu.apply(PcSide::Left, &rhs, &mut z)
        .expect("complex ILU0 apply");

    for (i, (&expected_diag, &rhs_i)) in diag.iter().zip(rhs.iter()).enumerate() {
        let expected = rhs_i / expected_diag;
        let diff = z[i] - expected;
        assert!(
            diff.abs() < 1e-12,
            "entry {} mismatch: got {:?}, expected {:?}",
            i,
            z[i],
            expected
        );
    }
}

#[test]
fn ilu0_diagonally_dominant_has_finite_solution() {
    let matrix = Mat::from_fn(3, 3, |i, j| {
        if i == j {
            S::from_parts(6.0 + i as f64, (-1.5 + i as f64) * 0.1)
        } else if (i as isize - j as isize).abs() == 1 {
            S::from_parts(0.25 * (1 + i + j) as f64, -0.3 * (i as f64 - j as f64))
        } else {
            S::zero()
        }
    });

    let mut ilu = IluBuilder::new()
        .ilu_type(IluType::ILU0)
        .build::<S>()
        .expect("complex ILU0 build");
    ilu.setup(&matrix)
        .expect("complex ILU0 setup for diagonally dominant matrix");
    assert!(ilu.complex_setup_used_native());

    assert_eq!(ilu.pivot_stats().num_floors, 0, "unexpected pivot floors");

    let rhs = [
        S::from_parts(1.5, -0.25),
        S::from_parts(-2.25, 0.5),
        S::from_parts(0.75, 1.0),
    ];
    let mut z = vec![S::zero(); rhs.len()];
    ilu.apply(PcSide::Left, &rhs, &mut z)
        .expect("complex ILU0 apply");

    for (idx, value) in z.iter().enumerate() {
        assert!(
            value.real().is_finite() && value.imag().is_finite(),
            "non-finite output at {}: {:?}",
            idx,
            value
        );
    }

    let residual = mat_vec_mul(&matrix, &z)
        .iter()
        .zip(rhs.iter())
        .map(|(&az, &rhs_i)| (az - rhs_i).abs())
        .fold(0.0, f64::max);
    assert!(residual < 1e-10, "residual too large: {residual}");
}

#[test]
fn ilu_diagnostics_reports_native_complex_mode() {
    use kryst::context::pc_context::PcType;
    use kryst::utils::diagnostics::PcDiagnostics;

    let diag = PcDiagnostics::from_options(Some(PcType::Ilu0), None);
    assert_eq!(diag.complex_support, "native_complex");
}
