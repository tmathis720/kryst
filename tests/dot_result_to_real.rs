#![cfg(not(feature = "complex"))]
use kryst::algebra::prelude::*;
use kryst::solver::common::dot_result_to_real;

#[test]
fn dot_to_real_returns_real_part() {
    let x = [1.0, -2.0, 3.5, 0.25];
    let y = [0.5, 2.0, -1.0, -4.0];
    let mut acc = S::zero();
    for (&xi, &yi) in x.iter().zip(&y) {
        acc = acc + S::from_real(xi * yi);
    }

    let real = dot_result_to_real(acc);
    let expected: f64 = x.iter().zip(&y).map(|(&a, &b)| a * b).sum();
    assert!((real - expected).abs() < 1e-12);
}

#[cfg(feature = "complex")]
#[test]
fn dot_to_real_tolerates_tiny_imaginary_drift() {
    let re = 10.0;
    let drift = 1e-14 * (1.0 + re.abs());

    for &sign in &[1.0, -1.0] {
        let value = S::from_parts(re, sign * drift);
        let real = dot_result_to_real(value);
        assert!((real - re).abs() < 1e-12);
    }
}

#[cfg(feature = "complex")]
#[test]
fn dot_to_real_returns_real_part_even_with_large_imaginary() {
    let value = S::from_parts(1.0, 1e-3);
    let real = dot_result_to_real(value);
    assert!((real - 1.0).abs() < 1e-12);
}
