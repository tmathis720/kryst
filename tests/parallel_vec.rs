#![cfg(not(feature = "complex"))]
use kryst::algebra::parallel::{par_axpy, par_dot_conj_local, par_sum_abs2_local};
use kryst::algebra::prelude::*;

fn approx_eq(a: R, b: R) -> bool {
    let eps = 1e-12;
    let scale = 1.0 + a.abs().max(b.abs());
    (a - b).abs() <= eps * scale
}

#[test]
fn axpy_real_equivalence() {
    let x: Vec<S> = (0..10_000).map(|i| S::from_real(i as f64 * 0.25)).collect();
    let mut y_scalar = vec![S::from_real(1.0); x.len()];
    let mut y_parallel = y_scalar.clone();

    super_scalar_axpy(&x, S::from_real(2.0), &mut y_scalar);
    par_axpy(&x, S::from_real(2.0), &mut y_parallel);

    for (expected, got) in y_scalar.iter().zip(&y_parallel) {
        assert!(approx_eq((*expected - *got).abs(), R::default()));
    }
}

#[test]
fn dot_conj_matches_scalar() {
    let n = 50_000;
    let x: Vec<S> = (0..n)
        .map(|i| S::from_real(((i % 7) as f64) - 3.0))
        .collect();
    let y: Vec<S> = (0..n)
        .map(|i| S::from_real(((i % 11) as f64) - 5.0))
        .collect();

    let parallel = par_dot_conj_local(&x, &y);
    let scalar = {
        let mut acc = S::zero();
        for (&xi, &yi) in x.iter().zip(&y) {
            acc = acc + xi.conj() * yi;
        }
        acc
    };

    assert!(approx_eq((parallel - scalar).abs(), R::default()));
}

#[test]
fn sum_abs2_matches_scalar() {
    let n = 80_000;
    let x: Vec<S> = (0..n).map(|i| S::from_real((i as f64).sin())).collect();

    let parallel = par_sum_abs2_local(&x);
    let scalar = {
        let mut acc = R::default();
        for &value in &x {
            let a = value.abs();
            acc += a * a;
        }
        acc
    };

    assert!(approx_eq(parallel, scalar));
}

fn super_scalar_axpy(x: &[S], alpha: S, y: &mut [S]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = *yi + alpha * xi;
    }
}
