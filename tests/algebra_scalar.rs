#![cfg(not(feature = "complex"))]
use kryst::algebra::blas::{dot_conj, nrm2};
use kryst::algebra::scalar::{KrystScalar, R, S};

fn assert_close(label: &str, actual: R, expected: R) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= 1e-12 * expected.abs().max(1.0),
        "{label} |diff|={diff:e} expected={expected:e} actual={actual:e}"
    );
}

fn assert_imag_small(label: &str, value: S) {
    let imag = value.imag().abs();
    assert!(
        imag <= 1e-12 * value.real().abs().max(1.0),
        "{label} imaginary part too large: {imag:e} for value={value:?}"
    );
}

#[test]
fn dot_conj_matches_norm_squared_real_inputs() {
    let x = [
        S::from_real(1.0),
        S::from_real(-2.5),
        S::from_real(3.75),
        S::from_real(-0.5),
    ];
    let inner = dot_conj(&x, &x);
    assert_imag_small("real inner product imaginary", inner);

    let norm = nrm2(&x);
    assert_close("‖x‖₂²", inner.real(), norm * norm);
}

#[test]
fn dot_conj_is_hermitian_for_real_vectors() {
    let x = [
        S::from_real(1.5),
        S::from_real(-0.25),
        S::from_real(0.0),
        S::from_real(2.0),
    ];
    let y = [
        S::from_real(-3.0),
        S::from_real(4.25),
        S::from_real(1.0),
        S::from_real(-0.5),
    ];

    let xy = dot_conj(&x, &y);
    let yx = dot_conj(&y, &x);
    assert_imag_small("xy", xy);
    assert_imag_small("yx", yx);
    assert_close("hermitian real", xy.real(), yx.real());
}

#[cfg(feature = "complex")]
#[test]
fn dot_conj_matches_norm_squared_complex_inputs() {
    let x = [
        S::from_parts(1.0, 0.5),
        S::from_parts(-2.5, 1.25),
        S::from_parts(0.0, -3.0),
        S::from_parts(4.0, -0.75),
    ];

    let inner = dot_conj(&x, &x);
    let norm = nrm2(&x);
    let expected = norm * norm;
    assert_close("‖x‖₂² complex", inner.real(), expected);
    assert!(
        inner.imag().abs() <= 32.0 * f64::EPSILON,
        "self inner product must be real; got {inner:?}"
    );
}

#[cfg(feature = "complex")]
#[test]
fn dot_conj_is_hermitian_for_complex_vectors() {
    let x = [
        S::from_parts(1.0, 0.5),
        S::from_parts(-0.25, 2.0),
        S::from_parts(3.0, -1.5),
    ];
    let y = [
        S::from_parts(-2.5, 1.0),
        S::from_parts(0.5, -1.75),
        S::from_parts(4.0, 0.25),
    ];

    let xy = dot_conj(&x, &y);
    let yx = dot_conj(&y, &x);
    let diff = xy - yx.conj();
    let tol = 1e-12 * xy.abs().max(yx.abs()).max(1.0);
    assert!(
        diff.abs() <= tol,
        "dot_conj not Hermitian: xy={xy:?}, yx={yx:?}, diff={diff:?}, tol={tol:e}"
    );
}
