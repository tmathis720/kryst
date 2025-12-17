#![cfg(not(feature = "complex"))]
use kryst::algebra::scalar::{KrystScalar, R, S};
use kryst::solver::common::givens::{apply_complex_givens, build_complex_givens};

fn assert_unitary(c: R, s: S) {
    let norm = (c * c + s.abs() * s.abs()).sqrt();
    let diff = (norm - 1.0).abs();
    assert!(
        diff <= 32.0 * f64::EPSILON,
        "rotation not unitary: |[c,s]|={norm} (c={c}, s={s:?})"
    );
}

#[test]
fn givens_rotation_preserves_norm_for_real_scalars() {
    let a = S::from_real(3.0);
    let b = S::from_real(-4.0);
    let expected = {
        let absa = a.abs();
        let absb = b.abs();
        (absa * absa + absb * absb).sqrt()
    };

    let (c, s) = build_complex_givens(a, b);
    assert!(c >= 0.0);
    assert_unitary(c, s);

    let mut h0 = a;
    let mut h1 = b;
    apply_complex_givens(&mut h0, &mut h1, c, s);

    let scale = expected.max(1.0);
    assert!(
        h1.abs() <= 1e-12 * scale,
        "second entry not eliminated: |h1|={} (scale={scale})",
        h1.abs()
    );
    let err = (h0.abs() - expected).abs();
    assert!(
        err <= 1e-12 * scale,
        "rotation does not preserve norm: expected={expected}, got={}, err={err}",
        h0.abs()
    );
}

#[cfg(feature = "complex")]
#[test]
fn givens_rotation_preserves_norm_for_complex_scalars() {
    let a = S::from_parts(1.5, -0.25);
    let b = S::from_parts(-0.75, 2.0);
    let expected = {
        let absa = a.abs();
        let absb = b.abs();
        (absa * absa + absb * absb).sqrt()
    };

    let (c, s) = build_complex_givens(a, b);
    assert!(c >= 0.0);
    assert_unitary(c, s);

    let mut h0 = a;
    let mut h1 = b;
    apply_complex_givens(&mut h0, &mut h1, c, s);

    let scale = expected.max(1.0);
    assert!(
        h1.abs() <= 1e-12 * scale,
        "second entry not eliminated: |h1|={} (scale={scale})",
        h1.abs()
    );
    let err = (h0.abs() - expected).abs();
    assert!(
        err <= 1e-12 * scale,
        "rotation does not preserve norm: expected={expected}, got={}, err={err}",
        h0.abs()
    );
}
